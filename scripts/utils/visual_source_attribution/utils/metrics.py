from __future__ import annotations

import re
import string
from typing import Dict, List, Sequence, Tuple, Union, Any

# ----------------------------
# Extractors
# ----------------------------


def qwen_2_5_box_extractor(input: Dict[str, Any]) -> List[Tuple[int, int, int, int]]:
    """
    Extracts boxes from text generated with qwen2.5 models.
    """

    text = input["response"]
    w = input["w"]
    h = input["h"]

    start_tag = "<box>"
    end_tag = "</box>"

    pattern = re.compile(
        re.escape(start_tag) + r"(.*?)" + re.escape(end_tag),
        re.DOTALL | re.IGNORECASE,
    )

    # ---- extraction ----

    boxes = list()

    if "<box>" in text:
        for match in pattern.finditer(text):
            content = match.group(1).strip()

            try:
                if "," in content:
                    coordinates_str = content.split(",")
                    coordinates = [round(float(x.strip())) for x in coordinates_str]
                else:
                    coordinates_str = content.split(" ")
                    coordinates = [round(float(x.strip())) for x in coordinates_str]
            except Exception as e:
                continue

            if len(coordinates) != 4:
                continue

            coordinates_std = [
                coordinates[0] / w,
                coordinates[1] / h,
                coordinates[2] / w,
                coordinates[3] / h,
            ]

            boxes.append(coordinates_std)

    return boxes


def qwen_3_box_extractor(input: Dict[str, Any]) -> List[Tuple[int, int, int, int]]:
    """
    Extracts boxes from text generated with qwen2.5 models.
    """

    text = input["response"]
    w = 1000
    h = 1000

    start_tag = "<box>"
    end_tag = "</box>"

    pattern = re.compile(
        re.escape(start_tag) + r"(.*?)" + re.escape(end_tag),
        re.DOTALL | re.IGNORECASE,
    )

    # ---- extraction ----

    boxes = list()

    if "<box>" in text:
        for match in pattern.finditer(text):
            content = match.group(1).strip()

            try:
                if "," in content:
                    coordinates_str = content.split(",")
                    coordinates = [round(float(x.strip())) for x in coordinates_str]
                else:
                    coordinates_str = content.split(" ")
                    coordinates = [round(float(x.strip())) for x in coordinates_str]
            except Exception as e:
                continue

            if len(coordinates) != 4:
                continue

            coordinates_std = [
                coordinates[0] / w,
                coordinates[1] / h,
                coordinates[2] / w,
                coordinates[3] / h,
            ]

            boxes.append(coordinates_std)

    return boxes


def intervl_2_5_box_extractor(input: Dict[str, Any]) -> List[Tuple[int, int, int, int]]:
    """
    Extracts boxes from text generated with internvl2.5 models.
    """

    text = input["response"]
    w = 1000  # this model outputs coordinates standardized in [0,1000]
    h = 1000

    start_tag = "["
    end_tag = "]"

    pattern = re.compile(
        re.escape(start_tag) + r"(.*?)" + re.escape(end_tag),
        re.DOTALL | re.IGNORECASE,
    )

    # ---- extraction ----

    boxes = list()

    if ("[" in text) and ("]" in text):
        for match in pattern.finditer(text):
            content = match.group(1).strip()

            try:
                if "," in content:
                    coordinates_str = content.split(",")
                    coordinates = [round(float(x.strip())) for x in coordinates_str]
                else:
                    coordinates_str = content.split(" ")
                    coordinates = [round(float(x.strip())) for x in coordinates_str]
            except Exception as e:
                continue

            if len(coordinates) != 4:
                continue

            coordinates_std = [
                coordinates[0] / w,
                coordinates[1] / h,
                coordinates[2] / w,
                coordinates[3] / h,
            ]

            boxes.append(coordinates_std)

    return boxes


# ----------------------------
# Geometry utilities
# ----------------------------


Box = Tuple[float, float, float, float]


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0

    union = box_area((ax1, ay1, ax2, ay2)) + box_area((bx1, by1, bx2, by2)) - inter
    return inter / union if union > 0 else 0.0


def iiou(box_a: Box, box_b: Box) -> float:  # box_a is the internal box
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0

    union = box_area((ax1, ay1, ax2, ay2))
    return inter / union if union > 0 else 0.0


def max_iiou(box_a: Box, box_b: Box) -> float:
    return max([iiou(box_a, box_b), iiou(box_b, box_a)])


def pair_boxes(
    gt_boxes: List[Dict[str, Any]], pred_boxes: List[Dict[str, Any]], iou_function=iou
):
    """
    Pairs boxes between the gt and the predictions.

    The expected structure of the input is the following:
    [
        {
            "id": "B0",
            "box": [float, float, float, float],
        },
        {
            "id": "B1",
            "box": [float, float, float, float],
        },
        ...
    ]
    """

    pairs = []
    for gt in gt_boxes:
        for pred in pred_boxes:
            score = iou_function(gt["box"], pred["box"])
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


def compute_precision(tp, fp):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return precision


def compute_recall(tp, fn):
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return recall


def compute_f1(p, r):
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return f1


def compute_box_metrics(gt_boxes_list, pred_boxes_list, thr):
    """
    Computes the micro and macro dataset level metrics for boxes prediction.
    """

    assert len(gt_boxes_list) == len(pred_boxes_list)

    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    all_precisions = []
    all_recalls = []
    all_f1s = []
    for gt_boxes, pred_boxes in zip(gt_boxes_list, pred_boxes_list):
        TPs = sum([x["score"] >= thr for x in gt_boxes])
        FPs = sum([x["score"] < thr for x in pred_boxes])
        FNs = sum([x["score"] < thr for x in gt_boxes])

        overall_tp += TPs
        overall_fp += FPs
        overall_fn += FNs

        p = compute_precision(TPs, FPs)
        r = compute_recall(TPs, FNs)
        f1 = compute_f1(p, r)

        all_precisions.append(p)
        all_recalls.append(r)
        all_f1s.append(f1)

    overall_p = compute_precision(overall_tp, overall_fp)
    overall_r = compute_recall(overall_tp, overall_fn)
    overall_f1 = compute_f1(overall_p, overall_r)

    return {
        "thr": thr,
        "macro_p": overall_p,
        "macro_r": overall_r,
        "macro_f1": overall_f1,
        "micro_p": sum(all_precisions) / len(all_precisions),
        "micro_r": sum(all_recalls) / len(all_recalls),
        "micro_f1": sum(all_f1s) / len(all_f1s),
    }


# ----------------------------
# Text utilities
# ----------------------------


def levenshtein_distance(s1: str, s2: str):
    """Computes the levenshtein distance between two strings."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]


def anls(gt: str, pred: str, threshold: float = 0.5):
    """Computes the Average Normalized Levenshtein Similarity."""
    dist = levenshtein_distance(gt, pred)
    length = max(len(gt.upper()), len(pred.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    score = 1.0 - value
    if score <= threshold:
        score = 0.0
    return score


def process_text_str(s: str) -> str:
    """Processes a string containing text for metric computations."""
    for punct in string.punctuation:  # punctuation
        s = s.replace(punct, "")
    s = re.sub(r"\s+", " ", s).strip()  # spaces
    s = s.lower()  # capital letters

    return s


def process_perc_str(s: str) -> str:
    """Processes a string containing a percentage value."""

    numbers = re.findall(r"\d+(?:[.,]\d+)?", s)
    if len(numbers) != 0:
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
    if len(numbers) != 0:
        return ""

    try:
        number = float(numbers[0])
    except Exception:
        number = s

    return str(number)


def process_date_str(s: str) -> str:
    """Process a string containing a date value."""
    s = s.strip()
    if "/" in s:  # string has a format of the type "yyyy/mm/dd"
        s = s.replace("/", "-")

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


def process_extractive_answer(answer: Dict[str, str]) -> Union[str, float]:
    a_val, a_type = answer["value"], answer["type"]
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


def pair_strings(gt_boxes: List[Dict[str, Any]], pred_boxes: List[Dict[str, Any]]):
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
            "type": "<the type>",
        },
        ...
    ]
    """

    pairs = []
    for gt in gt_boxes:
        for pred in pred_boxes:
            score = match_score(gt, pred)
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


def match_score(gt: Dict[str, str], pred: Dict[str, str]) -> bool:
    g = process_extractive_answer(gt)
    p = process_extractive_answer(pred)

    return anls(g, p)


# check here -> https://github.com/mayubo2333/MMLongBench-Doc/blob/main/eval/eval_score.py
