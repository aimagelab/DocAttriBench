from pathlib import Path
import argparse
import glob
import logging
import os
import json_repair
import copy
import sys
import re
import string
from math import isclose

from typing import Any, Dict, List, Tuple


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "utils")
)

from visual_source_attribution.utils.files import read_json_file, write_json_file

from visual_source_attribution.utils.metrics import (
    iou,
    max_iiou,
    qwen_2_5_box_extractor,
    qwen_3_box_extractor,
    intervl_2_5_box_extractor,
    visa_box_extractor,
    pair_boxes,
    compute_box_metrics,
    compute_answer_metrics,
)





def process_text_str(s: str) -> str:
    """Processes a string containing text for metric computations."""
    for punct in string.punctuation:  # punctuation
        s = s.replace(punct, "")
    s = re.sub(r"\s+", " ", s).strip()  # spaces
    s = s.lower()  # capital letters

    return s


def process_numeric_str(s: str) -> str:
    """Processes a string containing a number for metric computations."""

    s = s.strip()

    point_pos = re.search(".", s)
    if point_pos is not None:
        point_pos = point_pos.start()
    comma_pos = re.search(",", s)
    if comma_pos is not None:
        comma_pos = comma_pos.start()

    if point_pos is None:  # the string only contains commas
        s = s.replace(",", ".")

    elif (
        all(x is not None for x in [point_pos, comma_pos]) and point_pos < comma_pos
    ):  # the point it used for thousands and the comma for decimals
        s = s.replace(".", ",")
        s = s[:comma_pos] + "." + s[comma_pos + 1 :]

    return s


def process_date_str(s: str) -> str:
    """Process a string containing a date value."""
    s = s.strip()
    if "/" in s:  # string has a format of the type "yyyy/mm/dd"
        s = s.replace("/", "-")

    return s


def process_perc_str(s: str) -> str:
    """Processes a string containing a percentage value."""

    numbers = re.findall(r"\d+(?:[.,]\d+)?", s)
    if len(numbers) != 1:
        return ""

    try:
        number = float(numbers[0])
        if "%" in s:
            number /= 100
    except Exception:
        number = s

    return str(number)


def process_measurment_str(s: str) -> str:
    """Processes a string containing a percentage value."""

    numbers = re.findall(r"-?\d+(?:[.,]\d+)?", s)
    if len(numbers) != 1:
        return ""

    try:
        number = float(numbers[0])
    except Exception:
        number = s

    return str(number)


def process_extractive_answer(answer, default_to_string = False):
    a_val, a_type = answer["value"], answer["type"]
    
    if default_to_string:
        return a_val
    
    if a_type == "string":
        return process_text_str(a_val)
    elif a_type == "number":
        return process_numeric_str(a_val)
    elif a_type == "date":
        return process_date_str(a_val)
    elif a_type == "percentage":
        return process_perc_str(a_val)
    elif a_type == "measurment":
        return process_measurment_str(a_val)
    else:
        return a_val


def is_float_equal(reference, prediction, include_percentage: bool = False, is_close: float = False) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
        return precision

    try:
        reference = float(str(reference).strip().rstrip("%").strip())
    except:
        return False
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except:
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


from visual_source_attribution.utils.metrics import anls


def match_score(gt, pred, default_to_string: bool = False) -> bool:
    type_g = gt["type"]
    type_pred = gt["type"]
    g = process_extractive_answer(gt, default_to_string)
    p = process_extractive_answer(pred, default_to_string)

    if (type_g in ["number", "percentage", "measurment"]) and (type_pred in ["number", "percentage", "measurment"]):
        return is_float_equal(g, p)

    return anls(g, p)


def pair_strings(gt_boxes, pred_boxes, default_to_string: bool = False):
    """
    Pairs boxes between the gt and the predictions.

    The expected structure of the input is the following:
    [
        {
            "id": "B0",
            "value": "<the content>",
            "type": "<the type>",
        },
        {
            "id": "B1",
            "value": "<the content>",
            "type": "<the type>",
        },
        ...
    ]
    """

    pairs = []
    for gt in gt_boxes:
        for pred in pred_boxes:
            score = match_score(gt, pred, default_to_string)
            pairs.append(
                {
                    "gt_idx": gt["id"],
                    "pred_idx": pred["id"],
                    "score": score,
                }
            )

    pairs = sorted(pairs, key=lambda x: x["score"], reverse=True)

    while pairs:
        target = pairs.pop(0)
        gt_idx = target["gt_idx"]
        pred_idx = target["pred_idx"]
        score = target["score"]

        pairs = list(
            filter(
                lambda x: (x["gt_idx"] != gt_idx) and (x["pred_idx"] != pred_idx),
                pairs,
            )
        )

        for gt in gt_boxes:
            if gt["id"] == gt_idx:
                gt["score"] = score
                gt["association"] = pred_idx
                break

        for pred in pred_boxes:
            if pred["id"] == pred_idx:
                pred["score"] = score
                pred["association"] = gt_idx
                break

    for gt in gt_boxes:
        if "score" not in gt:
            gt["score"] = 0.0
            gt["association"] = None

    for pred in pred_boxes:
        if "score" not in pred:
            pred["score"] = 0.0
            pred["association"] = None

    return gt_boxes, pred_boxes


MODEL2BOX_EXTRACTOR = {
    "qwen2_5": qwen_2_5_box_extractor,
    "qwen3": qwen_3_box_extractor,
    "internvl": intervl_2_5_box_extractor,
    "visa": visa_box_extractor,
    "mappet2": qwen_2_5_box_extractor,
    "mappet3": qwen_3_box_extractor,
    "mappetintern": qwen_3_box_extractor
}


def parse_args():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser(description="MMLongBench-Doc downloader.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The dataset directory.",
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        required=True,
        help="The directory containing the inference.",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        required=True,
        help="The directory containing the metrics.",
    )
    parser.add_argument(
        "-single_pred",
        action="store_true",
        default=False,
        help="Whether to look for a single prediction per gt box.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        required=False,
        default="./logs",
        help="The log directory.",
    )
    return parser.parse_args()


def setup_logger(log_file: str, log_level: str = "INFO"):
    """
    Creates the logger for this file.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Remove all handlers associated with the root logger object
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def main():
    args = parse_args()

    DATA_DIR = args.dataset_dir             # take the original data for parsing boxes
    INFERENCE_DIR = args.inference_dir      # take the inference for parsing boxes and for vllm extractions
    METRICS_DIR = args.metrics_dir          # here we save the metrics
    SINGLE_PRED = args.single_pred

    # check if are dirs and exist
    if not os.path.isdir(DATA_DIR):
        raise ValueError(f"DATA_DIR non è una directory valida: {DATA_DIR}")
    
    if not os.path.isdir(INFERENCE_DIR):
        raise ValueError(f"INFERENCE_DIR non è una directory valida: {INFERENCE_DIR}")

    os.makedirs(METRICS_DIR, exist_ok=True)

    setup_logger(args.log_file)

    # ---- paths ----

    # The inference paths are the paths where various model inferences have been saved
    inference_paths = glob.glob(f"{INFERENCE_DIR}/*.json")
    inference_paths = list(filter(lambda x: "_neg" not in x, inference_paths))
    inference_paths = list(filter(lambda x: "_vllmExtraction" not in x, inference_paths))
    
    inference_paths = sorted(inference_paths)

    # ---- metrics ----

    for inference_path in inference_paths:
        file_name = inference_path.split("/")[-1].replace(".json", "")
        logging.info(f"Processing file {file_name} .")
        dataset_name = file_name.split("_")[0]
        model_name = file_name.replace(f"{dataset_name}_", "")

        # ---- box parsing function ----

        if "qwen2" in model_name.lower():
            model = "qwen2_5"
        elif "qwen3" in model_name.lower():
            model = "qwen3"
        elif "internvl" in model_name.lower():
            model = "internvl"
        elif "visa" in model_name.lower():
            model = "visa"
        elif "mappet2" in model_name.lower():
            model = "mappet2"
        elif "mappet3" in model_name.lower():
            model = "mappet3"
        elif "mappetintern" in model_name.lower():
            model = "mappetintern"
        else:
            logging.info(f"Model {model_name} not implemented")
            continue
        parser = MODEL2BOX_EXTRACTOR[model]

        # ---- all require paths ----

        # VLLM extractions gt
        extractive_gt_path = f"{DATA_DIR}/{dataset_name}/test/vllm_extractive_answers.json"

        # VLLM extractions pred
        extractive_pred_path = inference_path.replace(".json", "_vllmExtraction.json")


        # GT dataset
        dataset_dir = f"{DATA_DIR}/{dataset_name}/test/json"
        json_paths = glob.glob(f"{dataset_dir}/*.json")
        json_paths = list(filter(lambda x: "_neg" not in x, json_paths))
        json_paths = sorted(
            json_paths,
            key=lambda x: int(x.split("/")[-1].replace(".json", "")),
        )

        # Pred dataset
        
        # The variable is in the for loop

        if not os.path.exists(extractive_gt_path):
            logging.info(f"[SKIP]__ Missing extractive gt path: {extractive_gt_path}. Skipping.")
            continue
        if not os.path.exists(extractive_pred_path):
            logging.info(f"[SKIP]__ Missing extractive predictions path: {extractive_pred_path}. Skipping.")
            continue

        # ---- load required data ----

        ### GT
        gt_dataset = []
        for path in json_paths:
            gt_dataset.extend(read_json_file(path)["items"])
        dataset_boxes = [x["bbox"] for x in gt_dataset]
        dataset_types = [x["type"] for x in gt_dataset]
        gt_box_dataset = []
        for line, _types_list in zip(dataset_boxes, dataset_types):
            if len(_types_list) < len(line):
                _types_list = ["N/A"] * len(line)
            ith_proc = []
            for i, (box, _type) in enumerate(zip(line, _types_list)):
                ith_proc.append(
                    {
                        "id": f"B{i}",
                        "box": box,
                        "type": _type,
                    }
                )
            gt_box_dataset.append(ith_proc)

        ### PRED_BOX
        raw_predictions = read_json_file(inference_path)
        inference_boxes = list(map(lambda x: parser(x), raw_predictions))
        pred_box_dataset = []
        for line in inference_boxes:
            ith_proc = []
            for i, box in enumerate(line):
                ith_proc.append(
                    {
                        "id": f"B{i}",
                        "box": box,
                    }
                )
            pred_box_dataset.append(ith_proc)

        ### GT ANSWERS (EXT)
        raw_extractions = read_json_file(extractive_gt_path)
        gt_ans_dataset = []
        for line in raw_extractions:
            try:
                line_json = json_repair.loads(line)
            except:
                line_json = []
            ith_proc = []
            for i, box in enumerate(line_json):
                try:
                    ith_proc.append({"id": f"B{i}", "value": box["value"], "type": box["type"],})
                except Exception:
                    ith_proc.append({"id": f"B{i}", "value": "", "type": "string",})
            gt_ans_dataset.append(ith_proc)

        ### PRED ANSWERS (EXT)
        raw_extractions = read_json_file(extractive_pred_path)
        pred_ans_dataset = []
        for line in raw_extractions:
            try:
                line_json = json_repair.loads(line)
            except:
                line_json = []
            ith_proc = []
            for i, box in enumerate(line_json):
                try:
                    ith_proc.append({"id": f"B{i}", "value": box["value"], "type": box["type"]})
                except Exception:
                    ith_proc.append({"id": f"B{i}", "value": "", "type": "string"})
            pred_ans_dataset.append(ith_proc)

        # ---- filter ----

        logging.info("[PRE_FILTER_NUMERICS]")
        logging.info(f"Len of gt boxes: {len(gt_box_dataset)}")
        logging.info(f"Len of pred boxes: {len(pred_box_dataset)}")
        logging.info(f"Len of gt answers: {len(gt_ans_dataset)}")
        logging.info(f"Len of pred answers: {len(pred_ans_dataset)}")

        gt_box_filtered_dataset = []
        pred_box_filtered_dataset = []
        gt_ans_filtered_dataset = []
        pred_ans_filtered_dataset = []
        for gt_box, pred_box, gt_ans, pred_ans in zip(gt_box_dataset, pred_box_dataset, gt_ans_dataset, pred_ans_dataset):
            if not gt_box:
                continue
            if not gt_ans:
                continue

            gt_box_filtered_dataset.append(gt_box)
            pred_box_filtered_dataset.append(pred_box)
            gt_ans_filtered_dataset.append(gt_ans)
            pred_ans_filtered_dataset.append(pred_ans)

        # ---- metrics boxes ----

        gt_final = []
        predictions_final = []
        for gt_box, pred_box in zip(gt_box_filtered_dataset, pred_box_filtered_dataset):
            gt_box, pred_box = pair_boxes(gt_box, pred_box, iou)
            gt_final.append(gt_box)
            predictions_final.append(pred_box)
        THR_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        metrics_box = []
        for thr in THR_LIST:
            metrics_box.append(compute_box_metrics(gt_final, predictions_final, thr, SINGLE_PRED))

        # ---- metrics answers ----

        gt_final = []
        predictions_final = []
        for gt_box, pred_box in zip(gt_ans_filtered_dataset, pred_ans_filtered_dataset):
            gt_box, pred_box = pair_strings(gt_box, pred_box)
            gt_final.append(gt_box)
            predictions_final.append(pred_box)
        THR_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        metrics_ans = []
        for thr in THR_LIST:
            metrics_ans.append(compute_answer_metrics(gt_final, thr))

        # ---- save ----

        metrics_iou_dir = f"{METRICS_DIR}/iou"
        metrics_ans_dir = f"{METRICS_DIR}/answers"
        os.makedirs(metrics_iou_dir, exist_ok=True)
        os.makedirs(metrics_ans_dir, exist_ok=True)

        assert len(metrics_box[0]["all_f1s"]) == len(metrics_ans[0]["per_answer_score"])

        write_json_file(f"{metrics_iou_dir}/{file_name}.json", metrics_box)
        write_json_file(f"{metrics_ans_dir}/{file_name}.json", metrics_ans)

    print("Success: process finished!")

if __name__ == "__main__":
    main()