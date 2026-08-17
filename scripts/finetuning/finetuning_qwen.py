import argparse
import logging
import random
import shutil
import torch
import glob
import json
import tqdm
import math
import copy
import os
import gc

from torch.utils.data import Subset


from typing import Dict, List, Any, Union, Literal, Tuple
from collections import defaultdict
from datetime import datetime
from functools import partial
from PIL import Image

from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

from visual_source_attribution.utils.files import read_json_file

from torch.utils.data import DataLoader
from qwen_vl_utils import process_vision_info
from torch.amp import autocast

import numpy as np


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_model_class_and_family(base_model_name: str):
    model_name = base_model_name.lower()

    if "qwen2.5" in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration, "qwen2.5"
    if "qwen3" in model_name:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration, "qwen3"

    raise ValueError(
        "Unsupported base model. Expected a model path/name containing "
        "'Qwen2.5' or 'Qwen3', but received: "
        f"{base_model_name}"
    )


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
        "--base_model_name",
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
        "--image_dir",
        type=str,
        required=True,
        help="The image root directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="The training batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        required=True,
        help="The number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--dataloader_seed",
        type=int,
        required=True,
        help="The seed for the training dataloader.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="The number of training epochs.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        required=False,
        default="./logs",
        help="The log directory.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="The learning rate.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        required=True,
        help="The maximum number of tokens for a sequence.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        required=True,
        help="The number of warmup steps.",
    )
    parser.add_argument(
        "--temperature_for_training",
        type=float,
        required=True,
        help="The temperature for training.",
    )
    parser.add_argument(
        "--image_max_size",
        type=int,
        required=True,
        help="The maximum value for the longest image dimension.",
    )
    parser.add_argument(
        "--image_max_pixel_count",
        type=int,
        required=True,
        help="The maximum number of pixels for an image.",
    )
    parser.add_argument(
        "--num_save_steps",
        type=int,
        required=True,
        help="How often to save the model checkpoint.",
    )

    # ---- PEFT / LoRA ----
    parser.add_argument('--use_peft', action='store_true', help='Use PEFT LoRA fine-tuning')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=32.0, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_bias', type=str, default='none', choices=['none','all','lora_only'], help='Bias type for LoRA')
    parser.add_argument('--lora_target_modules', type=str, nargs='*', default=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], help='Target module names for LoRA')
    # parser.add_argument('--lora_modules_to_save', type=str, nargs='*', default=['lm_head'], help='Modules to keep trainable/save alongside adapters') # OLD
    parser.add_argument('--lora_modules_to_save', type=str, nargs='*', default=[], help='Modules to keep trainable/save alongside adapters')

    # ---- resume ----
    parser.add_argument('--restore_checkpoint', action='store_true', help='Restore from latest checkpoint in output dir')

    return parser.parse_args()


TRAINING_TEMPLATES = dict()

#----------------
# ANSWER BOX
#----------------


TRAINING_TEMPLATES["answerBox"] = dict()


TRAINING_TEMPLATES["answerBox"]["positive"] = {
    "system_template": """You are an agent excellent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format <box> x1 y1 x2 y2 </box>.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".""",
    "user_template": """{query}""",
    "assistant_template": """{answer}"""
}


TRAINING_TEMPLATES["answerBox"]["negative"] = {
    "system_template": """You are an agent excellent at identifying evidence in a document page.
Your job is to answer the user's query based on the provided image and the context.
When you answer, provide evidence bounding boxes in the format <box> x1 y1 x2 y2 </box>.
Add an evidence bounding box at the end of each generated sentence.

If you cannot find the answer, respond with "I don't know".""",
    "user_template": """{query}""",
    "assistant_template": """I don't know.""",
}


#----------------
# BOX (FROM ANSWER)
#----------------


TRAINING_TEMPLATES["boxFromAnswer"] = dict()


TRAINING_TEMPLATES["boxFromAnswer"]["positive"] = {
    "system_template": """You are an agent excellent at identifying evidence in a document page.
Your job is to identify the answer contained in the user's input based on the provided image and the context.
Respond by providing the evidence bounding boxes in the format <box> x1 y1 x2 y2 </box>.
The input is constituted by the user query and the answer to that query.

If you cannot find the answer, respond with the empty box <box> </box>.""",
    "user_template": """Query:
{query}

Answer:
{answer}""",
    "assistant_template": """{box_str}"""
}


TRAINING_TEMPLATES["boxFromAnswer"]["negative"] = {
    "system_template": """You are an agent excellent at identifying evidence in a document page.
Your job is to identify the answer contained in the user's input based on the provided image and the context.
Respond by providing the evidence bounding boxes in the format <box> x1 y1 x2 y2 </box>.
The input is constituted by the user query and the answer to that query.

If you cannot find the answer, respond with the empty box <box> </box>.""",
    "user_template": """Query:
{query}

Answer:
{answer}""",
    "assistant_template": """<box> </box>"""
}


#----------------
# BOX
#----------------


TRAINING_TEMPLATES["box"] = dict()


TRAINING_TEMPLATES["box"]["positive"] = {
    "system_template": """You are an agent excellent at identifying evidence in a document page.
Your job is to locate the answer to the user's query based on the provided image and the context.
Respond by providing the evidence bounding boxes in the format <box> x1 y1 x2 y2 </box>.

If you cannot find the answer, respond with the empty box <box> </box>.""",
    "user_template": """{query}""",
    "assistant_template": """{box_str}"""
}


TRAINING_TEMPLATES["box"]["negative"] = {
    "system_template": """You are an agent excellent at identifying evidence in a document page.
Your job is to locate the answer to the user's query based on the provided image and the context.
Respond by providing the evidence bounding boxes in the format <box> x1 y1 x2 y2 </box>.

If you cannot find the answer, respond with the empty box <box> </box>.""",
    "user_template": """{query}""",
    "assistant_template": """<box> </box>"""
}


def qwen_2_5_image_scaler(
    img: Image.Image,
    target_size: int = 1024,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
) -> Tuple[Image.Image, int, int]:
    w, h = img.size

    max_dim = max([w, h])
    ratio = target_size / max_dim
    new_w, new_h = int(w * ratio), int(h * ratio)

    resized_w = int(math.floor(new_w / 28) * 28)
    resized_h = int(math.floor(new_h / 28) * 28)

    pixels = resized_w * resized_h

    if pixels > max_pixels:
        p_ratio = math.sqrt(max_pixels / pixels)

        resized_w = resized_w * p_ratio
        resized_h = resized_h * p_ratio

        resized_w = int(math.floor(resized_w / 28) * 28)
        resized_h = int(math.floor(resized_h / 28) * 28)

    if pixels < min_pixels:
        p_ratio = math.sqrt(min_pixels / pixels)

        resized_w = resized_w * p_ratio
        resized_h = resized_h * p_ratio

        resized_w = int(math.ceil(resized_w / 28) * 28)
        resized_h = int(math.ceil(resized_h / 28) * 28)

    img = img.resize((resized_w, resized_h))

    return img, resized_w, resized_h


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


def standardize_box(box: List[float], w: int, h: int) -> List[int]:
    """
    Transforms a box from float [0,1] to int [0, w], [0, h].
    """
    assert len(box) == 4
    assert all(isinstance(x, float) for x in box)
    x0 = round(box[0] * w)
    y0 = round(box[1] * h)
    x1 = round(box[2] * w)
    y1 = round(box[3] * h)

    return [x0, y0, x1, y1]


def box_to_string(box: Union[List[float], List[int]]) -> str:
    """Converts a box into a string"""
    
    return f"<box> {box[0]} {box[1]} {box[2]} {box[3]} </box>"


def standardize_boxes(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for line in dataset:
        line["box_str"] = box_to_string(line["box"])

    return dataset


def assign_template(dataset: List[Dict[str, Any]], random_seed: int = 658) -> List[Dict[str, Any]]:
    random.seed(random_seed)
    
    # choose template (box, answerobx, ecc)
    templates = random.choices(list(TRAINING_TEMPLATES.keys()), k=len(dataset))
    
    
    
    # # choose if postive or negative
    # pos_neg = random.choices(["positive", "negative"], k=len(dataset), weights=[0.8,0.2])
    
    # for line, template, pos_ in zip(dataset, templates, pos_neg):
    #     line["template"] = f"{template}|{pos_}"

    # return dataset
    
    # FORCING POSITIVES
    for line, template in zip(dataset, templates):
        # forza sempre positive
        line["template"] = f"{template}|positive"

    return dataset


# def filter_dataset(dataset):
#     new_dataset = list()
#     for ith_item in dataset:
#         if ith_item["query"] == "":
#             continue
#         if not ith_item["bbox"]:
#             continue
#         if not ith_item["answer"]:
#             continue
#         if len(ith_item["bbox"]) != len(ith_item["answer"]):
#             continue
#         if not os.path.exists(ith_item['image_path']):
#             continue
#         new_dataset.append(ith_item)
    
#     return new_dataset

# NEW FOR DEBUG
def filter_dataset(dataset):
    stats = {
        "total": 0,
        "empty_query": 0,
        "empty_bbox": 0,
        "empty_answer": 0,
        "len_mismatch": 0,
        "missing_image": 0,
        "kept": 0,
    }

    new_dataset = []
    for ith_item in dataset:
        stats["total"] += 1

        if ith_item.get("query", "") == "":
            stats["empty_query"] += 1
            continue

        if not ith_item.get("bbox"):
            stats["empty_bbox"] += 1
            continue

        if not ith_item.get("answer"):
            stats["empty_answer"] += 1
            continue

        if len(ith_item["bbox"]) != len(ith_item["answer"]):
            stats["len_mismatch"] += 1
            continue

        if not os.path.exists(ith_item["image_path"]):
            stats["missing_image"] += 1
            continue

        stats["kept"] += 1
        new_dataset.append(ith_item)

    logging.info(
        "[FILTER DATASET] "
        f"total={stats['total']} | "
        f"kept={stats['kept']} | "
        f"empty_query={stats['empty_query']} | "
        f"empty_bbox={stats['empty_bbox']} | "
        f"empty_answer={stats['empty_answer']} | "
        f"len_mismatch={stats['len_mismatch']} | "
        f"missing_image={stats['missing_image']}"
    )

    return new_dataset



def load_dataset(dataset_dir: str, split_list = List[Literal["train", "val", "test"]]):
    """
    Loads the declared datasets
    """

    logging.info(f"Loading dataset {dataset_dir} .")

    # Only some splits are available for the target dataset
    valid_splits = list(filter(lambda x: os.path.exists(f"{dataset_dir}/{x}"), split_list))

    # Save all splits in a dictionary
    datasets = defaultdict(list)
    for split in valid_splits:
        image_dir = f"{dataset_dir}/{split}/img"
        json_dir = f"{dataset_dir}/{split}/json"
        json_paths = glob.glob(f"{json_dir}/*.json")

        for path in json_paths:
            ith_sample = read_json_file(path)["items"]
            for _el in ith_sample:
                image_name = _el['image_path'].split("/")[-1]
                _el['image_path'] = f"{image_dir}/{image_name}"

            datasets[split].extend(ith_sample)

    for key in datasets:
        logging.info(f"Dataset split pre filtering {key} -> {len(datasets[key])}")
        
        # DEBUG
        if len(datasets[key]) > 0:
            sample = datasets[key][0]
            logging.info(
                f"[SAMPLE {key}] "
                f"query='{sample.get('query','')[:50]}' | "
                f"bbox_len={len(sample.get('bbox', []))} | "
                f"answer_len={len(sample.get('answer', []))} | "
                f"image_path={sample.get('image_path')}"
            )
        if datasets[key]:
            img_path = datasets[key][0]["image_path"]
            logging.info(f"[IMAGE CHECK] {img_path} exists={os.path.exists(img_path)}")

        
        datasets[key] = filter_dataset(datasets[key])
        logging.info(f"Dataset split post filtering {key} -> {len(datasets[key])}")

    return dict(datasets)


def dataset_split(dataset: List[Dict[str, Any]], split_value: Union[int, float], random_seed: int = 117, max_set_len: int = 500):
    """
    Splits a dataset in two parts.

    This fuction is thought to divide any dataset that doesn't have a validation split into train-val.
    The split_1 is used as the validation set.
    """

    assert any(isinstance(split_value, _type) for _type in [int, float])
    if isinstance(split_value, int):
        assert split_value > 0
    if isinstance(split_value, float):
        assert 0 < split_value < 1
        
    image_ids = set([x["image_path"] for x in dataset])

    N = len(list(image_ids))

    if isinstance(split_value, float):
        split_value = round(N * split_value)

    random.seed(random_seed)
    split_idxs = set(random.sample(list(image_ids), split_value))

    split_1, split_2 = list(), list()
    for _el in dataset:
        if _el["image_path"] in split_idxs:
            split_1.append(_el)
        else:
            split_2.append(_el)

    random.shuffle(split_1)
    split_1 = split_1[:max_set_len]

    return split_1, split_2


def prepare_dataset(dataset_dir):
    """
    Prepares the dataset into training and validation splits.
    """
    datasets = load_dataset(dataset_dir, ["train", "val"])

    if "val" in datasets:
        validation_set, _ = dataset_split(datasets["val"], 0.99)
        datasets["val"] = validation_set
        
    elif "train" in datasets:
        validation_set, train_set = dataset_split(datasets["train"], 0.2)
        datasets["val"] = validation_set
        datasets["train"] = train_set

    for key in datasets:
        datasets[key] = assign_template(datasets[key])
        
    for key in datasets:
        img_idxs = list(set([x["image_path"] for x in datasets[key]]))
        for _el in datasets[key]:
            if "neg" in _el["template"].split("|")[-1]:
                while True:
                    new_img_path = random.sample(img_idxs, 1)[0]
                    if _el["image_path"] == new_img_path:
                        continue
                    _el["image_path"] = new_img_path
                    break


    # DEBUG
    for split in datasets:
        logging.info(f"[PREPARE DATASET] {split} size = {len(datasets[split])}")

    return datasets


def load_image(image_path: str, target_size: int = 1024, max_pixel_count: int = 4000, sample_img_size: bool = False) -> Tuple[Image.Image, int, int]:
    with Image.open(image_path) as img_tmp:
        img = img_tmp.copy()

    w, h = img.size

    if max([w/h, h/w]) >= 3.0:
        target_size *= 2.0
    
    if sample_img_size:
        extra_dim = random.randint(0, 500)
        extra_dim = 0
        target_size += extra_dim

    img, w, h = qwen_2_5_image_scaler(img, round(target_size), max_pixels=max_pixel_count)

    return img, w, h


def prepare_messages(input_dict, max_pixel_count, sample_img_size, model_family):
    data = copy.deepcopy(input_dict)
    template_name, pos_neg = data["template"].split("|")

    system_template = TRAINING_TEMPLATES[template_name][pos_neg]["system_template"]
    user_template = TRAINING_TEMPLATES[template_name][pos_neg]["user_template"]
    assistant_template = TRAINING_TEMPLATES[template_name][pos_neg]["assistant_template"]

    image, w, h = load_image(data["image_path"], target_size=1024, max_pixel_count=max_pixel_count, sample_img_size=sample_img_size)

    if model_family == "qwen2.5":
        data["box_str"] = [box_to_string(standardize_box(box, w, h)) for box in data['bbox']]
    else:
        data["box_str"] = [box_to_string(standardize_box(box, 1000, 1000)) for box in data['bbox']]
    
    if template_name == "answerBox":
        ans = data["answer"]
        box = data["box_str"]
        data["answer"] = " .".join(f"{ans[i].strip('.')} {box[i]}" for i in range(len(ans)))
    if template_name == "box":
        data["box_str"] = " ".join(data["box_str"])
    if template_name == "boxFromAnswer":
        data["box_str"] = " ".join(data["box_str"])

    user_content = [
        {"type": "image", "image": image},
    ]
    if model_family == "qwen2.5":
        user_content.append({'type': 'text', 'text': f'Image Size: {(w, h)}\n'})
    user_content.append({'type': 'text', 'text': user_template.format(**data)})

    messages = [
        {"role": "system", "content": system_template.format(**data)},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_template.format(**data)}
    ]

    return messages, image


def custom_collator(batch, processor, collator_args: Dict[str, Any], sample_img_size : bool = False):

    messages, images = list(), list()
    for _el in batch:
        ith_mes, ith_img = prepare_messages(
            _el,
            collator_args["max_pixel_counts"],
            sample_img_size,
            collator_args["model_family"],
        )
        messages.append(ith_mes)
        images.append(ith_img)

    text = [processor.apply_chat_template(
        ith_m, tokenize=False, add_generation_prompt=False
    ) for ith_m in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding="longest",
        # return_tensors="pt",
        max_length=collator_args["max_length"],
        truncation=collator_args["truncation"],
    )

    labels = []
    for input_ids, attn in zip(inputs["input_ids"], inputs["attention_mask"]):
        assistant_idx = input_ids.index(processor.tokenizer.encode("assistant")[0])
        label = [-100] * len(input_ids)
        label[assistant_idx + 1 :] = input_ids[assistant_idx + 1 :]
        labels.append(label)

        for i, m in enumerate(attn):
            if m == 0:
                label[i] = -100

    inputs['labels'] = torch.tensor(labels)
    inputs = {k: torch.tensor(v) for k, v in inputs.items()}

    return inputs


def get_validation_loss(model, validation_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm.tqdm(validation_loader, desc="Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=False)
            val_loss += outputs.loss.item()
    val_loss /= len(validation_loader)
    model.train()
    
    return val_loss


def clean_mid_checkpoints(checkpoint_dir: str, keep_n: int = 2) -> None:
    """Remove older training state files and their corresponding PEFT adapter dirs."""
    state_paths = glob.glob(f"{checkpoint_dir}/chkp_*.pt")
    # Extract step numbers
    def extract_step(p: str) -> int:
        try:
            return int(p.split("_")[-1].replace(".pt", ""))
        except Exception:
            return -1
    state_paths = sorted(state_paths, key=extract_step, reverse=True)

    tbd = state_paths[keep_n:]
    for path in tbd:
        step = path.split("_")[-1].replace(".pt", "")
        # Remove state file
        try:
            os.remove(path)
            logging.info(f"[WARNING] Checkpoint clean executed! Deleted old checkpoint state: {path}")
        except FileNotFoundError:
            pass
        # Remove corresponding PEFT adapter dir if present
        peft_dir = os.path.join(checkpoint_dir, f"peft_chkp_{step}")
        if os.path.isdir(peft_dir):
            shutil.rmtree(peft_dir, ignore_errors=True)
            logging.info(f"[WARNING] Checkpoint clean executed! Deleted old PEFT adapters: {peft_dir}")


def main():
    args = parse_args()
    DATASET_DIR = args.dataset_dir
    BASE_MODEL_NAME = args.base_model_name
    OUTPUT_DIR = args.output_dir
    CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
    IMAGE_DIR = args.image_dir
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation
    DATALOADER_SEED = args.dataloader_seed
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    MAX_TOKENS = args.max_tokens
    NUM_WARMUP_STEPS = args.num_warmup_steps
    TEMPERATURE_FOR_TRAINING = args.temperature_for_training
    IMAGE_MAX_SIZE = args.image_max_size
    IMAGE_MAX_PIXEL_COUNT = args.image_max_pixel_count
    NUM_SAVE_STEPS = args.num_save_steps
    model_class, model_family = get_model_class_and_family(BASE_MODEL_NAME)

    # ---- logger ----

    setup_logger(args.log_file)

    logging.info("[SETTINGS]")
    logging.info(f"Dataset root                     ->      {DATASET_DIR}")
    logging.info(f"Image root                       ->      {IMAGE_DIR}")
    logging.info(f"Base model                       ->      {BASE_MODEL_NAME}")
    logging.info(f"Model family                     ->      {model_family}")
    logging.info(f"Use PEFT                         ->      {args.use_peft}")
    logging.info(f"Output directory                 ->      {OUTPUT_DIR}")
    logging.info(f"Batch size                       ->      {BATCH_SIZE}")
    logging.info(f"Gradient accumulation steps      ->      {GRADIENT_ACCUMULATION_STEPS}")
    logging.info(f"Dataloader seed                  ->      {DATALOADER_SEED}")
    logging.info(f"Epochs                           ->      {EPOCHS}")
    logging.info(f"Learning rate                    ->      {LEARNING_RATE}")
    logging.info(f"Max sequence length              ->      {MAX_TOKENS}")
    logging.info(f"Warmup steps                     ->      {NUM_WARMUP_STEPS}")
    logging.info(f"Temperature                      ->      {TEMPERATURE_FOR_TRAINING}")
    logging.info(f"Image maximum dimension          ->      {IMAGE_MAX_SIZE}")
    logging.info(f"Image maximum pixel count        ->      {IMAGE_MAX_PIXEL_COUNT}")
    logging.info(f"Save steps                       ->      {NUM_SAVE_STEPS}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Create metrics log file
    metrics_log_path = os.path.join(OUTPUT_DIR, "training_metrics.jsonl")
    logging.info(f"Metrics will be logged to        ->      {metrics_log_path}")

    # ---- dataset ----

    dataset_paths = glob.glob(f"{DATASET_DIR}/*")
    dataset_paths = list(filter(lambda x: os.path.isdir(x), dataset_paths))

    train_dataset, validation_dataset = list(), list()
    for path in dataset_paths:
        train_val_datasets = prepare_dataset(path)  # split, assigns template
        train_dataset.extend(train_val_datasets.get("train", []))
        validation_dataset.extend(train_val_datasets.get("val", []))
        
        #DEBUG
        logging.info(f"[MAIN] Accumulated train size: {len(train_dataset)}")
        logging.info(f"[MAIN] Accumulated val size: {len(validation_dataset)}")


    # ---- Loading Processor ----
    # used for validation loader and other stuff

    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)

    # ---- setting determinism for dataloaders ----

    random.seed(DATALOADER_SEED)
    np.random.seed(DATALOADER_SEED)
    torch.manual_seed(DATALOADER_SEED)
    torch.cuda.manual_seed(DATALOADER_SEED)
    torch.cuda.manual_seed_all(DATALOADER_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(DATALOADER_SEED)

    collator_args = {
        "truncation": False,
        "max_length": MAX_TOKENS,
        "max_pixel_counts": IMAGE_MAX_PIXEL_COUNT,
        "model_family": model_family,
    }
    
    # --- DEBUG for dataset loading ----
    if len(train_dataset) == 0:
        raise RuntimeError(
            "❌ Train dataset is EMPTY after preprocessing.\n"
            "Check logs above for FILTER DATASET and IMAGE CHECK."
        )
    if len(validation_dataset) == 0:
        logging.warning("⚠️ Validation dataset is EMPTY.")
        
    # --- / DEBUG for dataset loading ----


    # --- Validation loader ---

    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(custom_collator, processor=processor, collator_args=collator_args, sample_img_size=False),
        generator=g,
    )
    
    # --- / Validation loader ---

    # ---- Model Set Up ----

    logging.info("************************ START OF RUN ************************")

    attn_mode = "sdpa"

    logging.info("Loading model ...")
    base_model = model_class.from_pretrained(
        BASE_MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation=attn_mode,
        device_map="auto",
    )
    base_model.use_cache=False
    base_model.gradient_checkpointing_enable()
    logging.info("Model loaded correctly.")

    # Freeze all base model layers
    for _, param in base_model.named_parameters():
        param.requires_grad = False

    # ---- LoRA ----

    # Prepare LoRA config upfront
    lora_config = None
    
    if args.use_peft:
        
        peft_checkpoint_dir = None

        if args.restore_checkpoint:
            peft_dirs = glob.glob(f"{CHECKPOINT_DIR}/peft_chkp_*")

            def extract_step(p):
                return int(p.split("_")[-1])

            peft_dirs = sorted(peft_dirs, key=extract_step)

            if peft_dirs:
                peft_checkpoint_dir = peft_dirs[-1]
                peft_global_step = extract_step(peft_checkpoint_dir)
                logging.info(f"Found PEFT checkpoint at step {peft_global_step}: {peft_checkpoint_dir}")
        
        if peft_checkpoint_dir is not None:
            # 🔥 RESUME CORRETTO
            model = PeftModel.from_pretrained(
                base_model,
                peft_checkpoint_dir,
                is_trainable=True,
            )
            logging.info(f"Loaded LoRA adapters from checkpoint {peft_checkpoint_dir}.")
        else:
            # 🆕 TRAIN DA ZERO
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                target_modules=args.lora_target_modules,
                modules_to_save=args.lora_modules_to_save,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(base_model, lora_config)
            logging.info("Initialized new LoRA adapters.")

    # ---- / LoRA ----
    

    
    # ---- optimizer and scheduler ----

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # Calcolo numero step totali ORIGINALI
    # num_training_steps = int(len(train_dataloader) * EPOCHS / GRADIENT_ACCUMULATION_STEPS) # OLD
    total_batches_full = int(len(train_dataset) / BATCH_SIZE)
    total_optimizer_steps_full = int(total_batches_full * EPOCHS / GRADIENT_ACCUMULATION_STEPS)

    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=num_training_steps) # OLD
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=total_optimizer_steps_full,
    )
    
    # ---- variables for training ----

    virtual_batch_counter = 0
    global_step = 0

    best_val_loss = float("inf")

    # ---- Restore trainer state (AFTER creating optimizer & scheduler) ----

    trainer_ckpt_path = os.path.join(CHECKPOINT_DIR, "chkp_latest.pt")

    if args.restore_checkpoint and os.path.exists(trainer_ckpt_path):

        checkpoint = torch.load(trainer_ckpt_path, map_location="cuda")

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        global_step = checkpoint["global_step"]
        virtual_batch_counter = checkpoint["virtual_batch_counter"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logging.info(f"Restored trainer state at global_step={global_step}")
        
        logging.info(
            f"[DEBUG] global_step={global_step} | "
            f"virtual_batch_counter={virtual_batch_counter} | "
            f"lr={optimizer.param_groups[0]['lr']}"
        )

    # ---- new data loader for train ----

    # ------------------------------
    # Resume-aware DataLoader
    # ------------------------------

    if args.restore_checkpoint and global_step > 0:

        logging.info(f"Resuming from global_step={global_step}")

        # Ricrea lo stesso shuffle deterministico
        rng = torch.Generator() # restart the shufller
        rng.manual_seed(DATALOADER_SEED)
        shuffled_indices = torch.randperm(len(train_dataset), generator=rng).tolist()

        # Calcola quanti sample sono già stati consumati
        samples_already_seen = global_step * BATCH_SIZE

        logging.info(f"Skipping {samples_already_seen} samples already processed.")

        remaining_indices = shuffled_indices[samples_already_seen:]

        train_subset = Subset(train_dataset, remaining_indices)

        train_dataloader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,  # IMPORTANTISSIMO
            num_workers=0,
            collate_fn=partial(
                custom_collator,
                processor=processor,
                collator_args=collator_args,
                sample_img_size=True
            ),
        )

    else:

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=partial(
                custom_collator,
                processor=processor,
                collator_args=collator_args,
                sample_img_size=True
            ),
            generator=g,
        )

    
    # ---- / new data loader for train ----
    

    # --- Params printing ---
    trainable_params = 0
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    percent = 100 * trainable_params / total_params
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")
    logging.info(f"Percentage trainable: {round(percent, 3)}")
    # --- / Params printing ---


    # ---- training loop ----

    model.train()
    computation_device = model.device
    logging.info(f"Device used for model computation: {computation_device}.")

    ga_loss = 0.0

    for batch_idx, batch in enumerate(train_dataloader, start=global_step):
        
        batch = {k: v.to(computation_device) for k, v in batch.items()}

        with autocast(device_type="cuda", dtype=torch.bfloat16): # adding autocast before calculating output and loss FOR SCALER
            outputs = model(**batch, output_hidden_states=False)
            loss = outputs.loss
            logging.info(f"[TRAIN] Train loss {loss}")

        # accumulate for logging only
        ga_loss += loss.item() / BATCH_SIZE # DA RIVEDERE
        # scale loss for gradient accumulation and backprop every step
        (loss / GRADIENT_ACCUMULATION_STEPS).backward()
        global_step += 1

        # Log batch loss to file
        with open(metrics_log_path, "a") as f:
            f.write(json.dumps({
                "type": "batch_loss",
                "global_step": global_step - 1,
                "batch_idx": batch_idx,
                "loss": loss.item()
            }) + "\n")

        del loss, outputs, batch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Log virtual batch loss to file
            with open(metrics_log_path, "a") as f:
                f.write(json.dumps({
                    "type": "virtual_batch_loss",
                    "virtual_batch_counter": virtual_batch_counter,
                    "batch_idx": batch_idx,
                    "loss": ga_loss
                }) + "\n")

            current_lr = optimizer.param_groups[0]["lr"]

            logging.info(f"Batch {batch_idx}, virtual batch {virtual_batch_counter} - loss : {ga_loss} - lr: {current_lr}")
            ga_loss = 0.0
            virtual_batch_counter += 1

            # ---- save ----

            if (virtual_batch_counter % NUM_SAVE_STEPS == 0) and (virtual_batch_counter != 0):
                logging.info(f"Calculating validation loss after {NUM_SAVE_STEPS} steps...")
                val_loss = get_validation_loss(model, val_dataloader, computation_device)

                logging.info(f"Batch {batch_idx}, global step {global_step - 1} - Validation Loss: {val_loss:.4f}")

                # Save PEFT adapters (if enabled) and a lightweight trainer state
                peft_adapter_dir = None
                if args.use_peft:
                    peft_adapter_dir = os.path.join(CHECKPOINT_DIR, f"peft_chkp_{global_step}")
                    os.makedirs(peft_adapter_dir, exist_ok=True)
                    # This saves only adapter weights + adapter config
                    model.save_pretrained(peft_adapter_dir)
                    logging.info(f"Saved PEFT adapters to {peft_adapter_dir}")

                ckpt_path = os.path.join(CHECKPOINT_DIR, f"chkp_latest.pt")
                state = {
                    "global_step": global_step,
                    "virtual_batch_counter": virtual_batch_counter,
                    "best_val_loss": best_val_loss,
                    "use_peft": args.use_peft,
                    "lora_config": {
                        "r": args.lora_r,
                        "lora_alpha": args.lora_alpha,
                        "lora_dropout": args.lora_dropout,
                        "bias": args.lora_bias,
                        "target_modules": args.lora_target_modules,
                        "modules_to_save": args.lora_modules_to_save,
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }

                torch.save(state, ckpt_path)
                logging.info(f"Checkpoint saved at step {global_step}!")


if __name__ == "__main__":
    main()
