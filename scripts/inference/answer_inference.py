"""
This code utilizes VLLM to perform inference and compute metrics.
"""

from pathlib import Path
import argparse
import base64
import glob
import io
import logging
import os
import zipfile
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

from PIL import Image

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "utils")
)


from visual_source_attribution.utils.files import (
    read_json_file,
    write_json_file,
)
from visual_source_attribution.utils.visual import (
    internvl_2_5_image_scaler,
    qwen_2_5_image_scaler,
)
from visual_source_attribution.utils.vllm import (
    PromptVllmModel,
    deploy_vllm_model_from_settings,
)

IMAGE_SCALERS = {
    "Qwen_Qwen2.5-VL-7B-Instruct-AWQ": qwen_2_5_image_scaler,
    "Qwen_Qwen2.5-VL-32B-Instruct-AWQ": qwen_2_5_image_scaler,
    "Qwen_Qwen2.5-VL-32B-Instruct": qwen_2_5_image_scaler,
    "Qwen_Qwen2.5-VL-72B-Instruct-AWQ": qwen_2_5_image_scaler,
    "Qwen_Qwen2.5-VL-3B-Instruct": qwen_2_5_image_scaler,
    "Qwen_Qwen2.5-VL-7B-Instruct": qwen_2_5_image_scaler,
    "Qwen_Qwen3-VL-2B-Instruct": qwen_2_5_image_scaler,
    "Qwen_Qwen3-VL-8B-Instruct": qwen_2_5_image_scaler,
    "Qwen_Qwen3-VL-32B-Instruct": qwen_2_5_image_scaler,
    "Qwen_Qwen3-VL-32B-Instruct-FP8": qwen_2_5_image_scaler,
    "OpenGVLab_InternVL2_5-2B": internvl_2_5_image_scaler,
    "OpenGVLab_InternVL2_5-8B": internvl_2_5_image_scaler,
    "OpenGVLab_InternVL2_5-38B-AWQ": internvl_2_5_image_scaler,
    "OpenGVLab_InternVL3-2B-Instruct": internvl_2_5_image_scaler,
    "OpenGVLab_InternVL3-8B-Instruct": internvl_2_5_image_scaler,
    "OpenGVLab_InternVL3-38B-Instruct": internvl_2_5_image_scaler,
    "OpenGVLab_InternVL3_5-2B-Instruct": internvl_2_5_image_scaler,
    "OpenGVLab_InternVL3_5-8B-Instruct": internvl_2_5_image_scaler,
    "OpenGVLab_InternVL3_5-38B-Instruct": internvl_2_5_image_scaler,
    "Mappet_Mappet2.5-VL-7B-Instruct": qwen_2_5_image_scaler,
    "Mappet_Mappet2.5-VL-7B-Instruct-PEFT-v2": qwen_2_5_image_scaler,
    "Mappet_Mappet2_5-VL-7B-Instruct-PEFT-v2": qwen_2_5_image_scaler,
    "Mappet2": qwen_2_5_image_scaler,
    "Mappet3": qwen_2_5_image_scaler,
    "MappetIntern": internvl_2_5_image_scaler,
}


def parse_args():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The dataset directory.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The model name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory where all inferences will be saved.",
    )
    parser.add_argument(
        "--auto_port",
        type=bool,
        required=False,
        default=True,
        help="The port used for the model.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        default="answerBox",
        help="The task.",
    )
    parser.add_argument(
        "--restart_port",
        type=bool,
        required=False,
        default=True,
        help="Whether the port should be restarted every few iterations.",
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


def unzip_file(zip_path, extract_to):
    """
    Unzips a ZIP file into the specified directory.

    Args:
        zip_path (str): Path to the .zip file.
        extract_to (str): Directory to extract the contents into.
    """
    # Ensure the output directory exists
    os.makedirs(extract_to, exist_ok=True)

    # Extract all contents
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"✅ Extracted '{zip_path}' to '{extract_to}'")


def file_to_data_url(
    path,
    MAX_DIM=1024,
    fmt="PNG",
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
):
    # Open the image and resize
    img = Image.open(path).convert("RGB")

    w, h = img.size
    max_ratio = max(w / h, h / w)
    if max_ratio >= 3.0:
        MAX_DIM *= 2  # allow extra size for extreme aspect ratios

    if "MappetIntern" in model_name:
        logging.info(f"MappetIntern inside {model_name}, using internvl_2_5_image_scaler")
        img, new_w, new_h = IMAGE_SCALERS["MappetIntern"](img, MAX_DIM)
    elif "Mappet2" in model_name or "Mappet3" in model_name:
        logging.info(f"Mappet2(or 3) inside {model_name}, using qwen_2_5_image_scaler")
        img, new_w, new_h = IMAGE_SCALERS["Mappet2"](img, MAX_DIM)
    else:
        img, new_w, new_h = IMAGE_SCALERS[model_name](img, MAX_DIM)

    # Save the resized image to a bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    buffer.seek(0)

    # Encode to base64
    b64 = base64.b64encode(buffer.read()).decode("utf-8")

    # Return data URL
    return f"data:image/{fmt.lower()};base64,{b64}", new_w, new_h


def send_payload(
    line: Dict[str, Any], prompter: PromptVllmModel, img_dir: str
):
    file_name = line["image_path"].split("/")[-1]
    img_url = f"{img_dir}/img/{file_name}"

    img_base64, new_w, new_h = file_to_data_url(
        img_url, model_name=prompter.model_name
    )

    kwargs = {
        "query": line["query"],
        "answer": line["answer"],
        "resized_height": new_h,
        "resized_width": new_w,
        "url": img_base64,
    }

    response = prompter(**kwargs)

    return {
        "response": response,
        "w": new_w,
        "h": new_h,
    }


def parallel_payloads(
    data_lines: List[Dict[str, Any]],
    prompter: PromptVllmModel,
    dataset_img_dir: str,
):
    all_outputs = []

    with ThreadPoolExecutor(max_workers=64) as executor:
        for res in executor.map(
            send_payload,
            data_lines,
            [prompter] * len(data_lines),
            [dataset_img_dir] * len(data_lines),
        ):
            all_outputs.append(res)

    return all_outputs


def prepare_model(model_name: str, auto_port: bool = True, manual_port: Optional[int] = None, task: str = "answerBox"):
    logging.info("Starting VLLM in server mode...")
    logging.info(f"Task: {task}")
    succesful_deployment, served_model_name, client, port = (
        deploy_vllm_model_from_settings(model_name, auto_port, manual_port)
    )
    if not succesful_deployment:
        logging.info(
            "The model was not deployed succesfully, killing runtime."
        )
        raise ValueError("The model was not deployed succesfully.")
    logging.info("VLLM model deployed succesfully!")

    # ---- prompt util ----

    prompter = PromptVllmModel(
        model_name=served_model_name,
        client=client,
        max_tokens=256,
        temperature=0.0,
        task=task,
    )

    return prompter, port


def main():
    args = parse_args()

    MODEL_NAME = args.model_name
    DATA_DIR = args.dataset_dir
    OUTPUT_DIR = args.output_dir
    AUTO_PORT = args.auto_port
    TASK = args.task
    RESTART_PORT = args.restart_port

    # Check if they are directories and exist
    if not os.path.isdir(MODEL_NAME):
        raise ValueError(f"MODEL_NAME is not a valid directory: {MODEL_NAME}")

    if not os.path.isdir(DATA_DIR):
        raise ValueError(f"DATA_DIR is not a valid directory: {DATA_DIR}")

    if not os.path.isdir(OUTPUT_DIR):
        raise ValueError(f"OUTPUT_DIR is not a valid directory: {OUTPUT_DIR}")

    # ---- logger ----

    setup_logger(args.log_file)

    # ---- data ----

    inference_datasets = glob.glob(f"{DATA_DIR}/*/test")
    inference_datasets = sorted(inference_datasets)

    logging.info(inference_datasets)

    port = None
    if port is None:
        prompter, port = prepare_model(MODEL_NAME, auto_port=True, manual_port=None, task=TASK)
    else:
        prompter, port = prepare_model(MODEL_NAME, auto_port=False, manual_port=port, task=TASK)

    for dataset_dir in inference_datasets:
        dataset_name = dataset_dir.split("/")[-2] # take dataset name (docvqa, longdocurl,...)
        logging.info(f"Processing dataset {dataset_name}")

        # make model output name
        out_model_name = "_".join(MODEL_NAME.split("/")[-2:])
        output_path = os.path.join(
            OUTPUT_DIR,
            f"{dataset_name}_{out_model_name}_{TASK}.json",
        )
        # skip if json is already present
        if os.path.exists(output_path):
            logging.info("Dataset already processed.")
            continue

        # ---- jsons ----

        json_paths = glob.glob(f"{dataset_dir}/json/*.json")
        json_paths = list(filter(lambda x: "_neg" not in x, json_paths))
        json_paths = sorted(
            json_paths, key=lambda x: int(x.split("/")[-1].replace(".json", ""))
        )

        # making list of json files to eleborate
        dataset_list = []
        for path in json_paths:
            dataset_list.extend(read_json_file(path)["items"])

        # answer normalizing
        for line in dataset_list:
            if isinstance(line["answer"], list):
                line["answer"] = ". ".join([x.strip(".").strip() for x in line["answer"]])
            else:
                line["answer"] = line["answer"].strip(".") + "."

        results = []
        while dataset_list:

            logging.info(f"[PORT] {port}")
            targets, dataset_list = dataset_list[:1000], dataset_list[1000:]
            ith_results = parallel_payloads(targets, prompter, dataset_dir)
            results.extend(ith_results)

        # ---- save ----

        write_json_file(output_path, results)

    print("Success: process finished!")

if __name__ == "__main__":
    main()
