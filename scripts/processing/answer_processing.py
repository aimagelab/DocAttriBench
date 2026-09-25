"""
This code utilizes VLLM to perform inference and compute metrics.
"""

from pathlib import Path
import argparse
import glob
import logging
import os
import sys
import ast
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "utils")
)

from visual_source_attribution.utils.files import (
    read_json_file,
    write_json_file,
)
from visual_source_attribution.utils.vllm import VLLM_SETTINGS, deploy_vllm_model

EXTRACTION_TEMPLATE = """You are an expert information extraction assistant.
Your task is to transform abstractive answers into extractive ones — that is, identify and list only the exact text spans that could appear *verbatim* in the original document.
For each extracted span, add the extracted text span type. Use one of the following types:
- "date" for dates
- "percentage" for percentages
- "measurment" for numbers with units of measurment (like km/h, g, tons, ecc)
- "number" for both integers and floats that are not dates or percentages
- "string" for everything else

Follow these rules strictly:
1. Do NOT paraphrase or summarize.
2. Extract only minimal, self-contained spans (entities, names, dates, values, etc.).
3. Output only a Python-style list of strings.
4. If multiple distinct spans answer the question, include them all in order of appearance.
5. If no extractive span is possible, output an empty list [].

---

### Examples

# Single-span examples
Question: Who is the president of France?
Abstractive Answer: The current president of France is Emmanuel Macron.
Extractive Answer:
[{{"value": "Emmanuel Macron", "type": "string"}}]

---

# Single-span examples
Question: What percentage of Americans prefer Biden to Trump?
Abstractive Answer: 57% of americans declared to prefer Biden.
Extractive Answer:
[{{"value": 57%, "type": "percentagae"}}]

---

Question: What is the capital of Japan?
Abstractive Answer: Tokyo is the capital city of Japan, known for its technology and culture.
Extractive Answer:
[{{"value": "Tokyo", "type": "string"}}]

---

Question: Who discovered penicillin and in which year?
Abstractive Answer: Penicillin was discovered by Alexander Fleming in 1928.
Extractive Answer:
[{{"value": "Alexander Fleming", "type": "string"}}, {{"value": "1928", "type": "date"}}]

---

Question: Which two countries signed the Treaty of Versailles?
Abstractive Answer: The Treaty of Versailles was signed by Germany and France after World War I.
Extractive Answer:
[{{"value": "Germany", "type": "string"}}, {{"value": "France", "type": "string"}}]

---

Question: Who directed the movie Inception, and who are two main actors?
Abstractive Answer: Inception was directed by Christopher Nolan and stars Leonardo DiCaprio and Joseph Gordon-Levitt.
Extractive Answer:
[{{"value": "Christopher Nolan", "type": "string"}}, {{"value": "Leonardo DiCaprio", "type": "string"}}, {{"Leonardo DiCaprio": "Joseph Gordon-Levitt", "type": "string"}}]

---

Question: How many colors are on the Italian flag?
Abstractive Answer: The Italian flag consists of three vertical stripes.
Extractive Answer: [{{"value": "three", "type": "number"}}]

---

Question: What value has pie?
Abstractive Answer: Pie value is approximately 3.14.
Extractive Answer: [{{"value": 3.14, "type": "number"}}]

---

Question: What is the maximum speed of a kawasaki?
Abstractive Answer: The document reports a maximu speed of 300 km/h.
Extractive Answer: [{{"value": "300 km/h", "type": "measurment"}}]

---

Question: When was america discovered?
Abstractive Answer: America was discovered on the date: 12 October 1492.
Extractive Answer: [{{"value": "12-10-1492", "type": "date"}}]

---

### Task

Now extract the answer spans for the following input:

Question: {question}
Abstractive Answer: {abstractive_answer}
Extractive Answer:"""


def parse_args():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_dir",
        type=str,
        required=True,
        help="The inference directory.",
    )
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
        "--port",
        type=int,
        required=True,
        help="The port.",
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


def send_payload(line: Dict[str, Any], client, model_name: str):
    user_prompt = EXTRACTION_TEMPLATE.format(
        **{
            "question": line["query"],
            "abstractive_answer": ". ".join(line["answer"])[:2000],
        }
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        max_tokens=256,
        temperature=0.0,
    )

    return resp


def parallel_payloads(data_lines: List[Dict[str, Any]], client, model_name: str):
    all_outputs = []

    with ThreadPoolExecutor(max_workers=64) as executor:
        for res in executor.map(
            send_payload,
            data_lines,
            [client] * len(data_lines),
            [model_name] * len(data_lines),
        ):
            all_outputs.append(res)

    return all_outputs


def custom_deploy_vllm_model(model_name, port):
    served_model_name = "_".join(model_name.split("/")[-2:])

    if served_model_name not in VLLM_SETTINGS:
        raise ValueError(
            f"Model name {served_model_name} is not " "associated with any settings."
        )

    model_settings = VLLM_SETTINGS[served_model_name]
    model_settings["model_name"] = model_name
    model_settings["target_message"] = "Application startup complete"
    model_settings["max_model_length"] = 6144
    model_settings["max_num_seqs"] = 32
    model_settings["max_num_batched_tokens"] = 4196
    model_settings["port"] = port
    model_settings["num_gpus"] = 1
    model_settings["max_dep_time"] = 3600

    return deploy_vllm_model(**model_settings)


def main():
    args = parse_args()

    MODEL_NAME = args.model_name
    INFERENCE_DIR = args.inference_dir
    DATA_DIR = args.dataset_dir
    PORT = args.port

    # ---- logger ----

    setup_logger(args.log_file)

    # ---- data ----

    inference_datasets = glob.glob(f"{INFERENCE_DIR}/*.json")
    inference_datasets = list(
        filter(lambda x: "_negative" not in x, inference_datasets)
    )
    inference_datasets = list(
        filter(lambda x: "_vllmExtraction." not in x, inference_datasets)
    )
    inference_datasets = list(
        filter(lambda x: "_answerBox." in x, inference_datasets)
    )
    
    assert len(inference_datasets) > 0
    
    inference_datasets = sorted(inference_datasets)

    # ---- filter data ----

    # CHECK FOR FILE ALREADY PROCESSED AND DISCARD THEM

    filtered_inference_datasets = []
    for path in inference_datasets:
        file_name = path.split("/")[-1]

        dataset_name = file_name.split("_")[0]
        dataset_dir = f"{DATA_DIR}/{dataset_name}/test/json"

        save_path = path.replace(".json", "_vllmExtraction.json")
        if not os.path.exists(save_path):
            filtered_inference_datasets.append(path)

    if not filtered_inference_datasets:
        logging.info("All paths were already processed.")
        return False

    # ---- model ----

    logging.info("Starting VLLM in server mode...")
    succesful_deployment, served_model_name, client, _ = custom_deploy_vllm_model(
        MODEL_NAME, PORT
    )
    if not succesful_deployment:
        logging.info("The model was not deployed succesfully, killing runtime.")
        raise ValueError("The model was not deployed succesfully.")
    logging.info("VLLM model deployed succesfully!")

    # ---- inference ----

    for path in filtered_inference_datasets:
        file_name = path.split("/")[-1]
        logging.info(f"Processing file: {file_name}")

        dataset_name = file_name.split("_")[0]
        dataset_dir = f"{DATA_DIR}/{dataset_name}/test/json"

        save_path = path.replace(".json", "_vllmExtraction.json")
        if os.path.exists(save_path):
            logging.info("Already processed")
            continue

        # ---- jsons ----

        inference_list = read_json_file(path)

        json_paths = glob.glob(f"{dataset_dir}/*.json")
        json_paths = list(filter(lambda x: "_neg" not in x, json_paths))
        json_paths = sorted(
            json_paths,
            key=lambda x: int(x.split("/")[-1].replace(".json", "")),
        )

        dataset_list = []
        for path in json_paths:
            dataset_list.extend(read_json_file(path)["items"])

        for d_el, i_el in zip(dataset_list, inference_list):
            d_el["answer"] = i_el["response"]

        # ---- inference ----

        logging.info(f"Starting inference on {len(dataset_list)} samples...")

        outputs = parallel_payloads(dataset_list, client, served_model_name)
        outputs_processed = [x.choices[0].message.content for x in outputs]

        logging.info("Output generated succesfully!")

        # ---- save ----

        write_json_file(save_path, outputs_processed)

        logging.info(f"Output saved to {save_path} .")

    logging.info("Finished processing.")
    print("Processing concluded with success!")


if __name__ == "__main__":
    main()
