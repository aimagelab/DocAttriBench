import argparse
import torch
import os

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModel,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    InternVLForConditionalGeneration,
)
from peft import PeftModel

def parse_args():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="model path",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="adapter path",
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        required=True,
        help="output model path",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    BASE_MODEL = args.base_model
    ADAPTER_PATH = args.adapter_path
    OUTPUT_PATH = args.output_model_path

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if "Qwen2-VL-6B-Instruct" == os.path.basename(BASE_MODEL):
        print(f"{os.path.basename(BASE_MODEL)} found!")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    elif "Qwen2.5-VL-7B-Instruct" == os.path.basename(BASE_MODEL):
        print(f"{os.path.basename(BASE_MODEL)} found!")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    elif "Qwen3-VL-8B-Instruct" == os.path.basename(BASE_MODEL):
        print(f"{os.path.basename(BASE_MODEL)} found!")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    elif "InternVL3_5-8B-Instruct" == os.path.basename(BASE_MODEL):
        print(f"{os.path.basename(BASE_MODEL)} found!")
        model = AutoModel.from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        raise Exception("Specified Base Model is not valid for this script!")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
        device_map="auto",
    )

    model = model.merge_and_unload()
    
    model.to("cuda").to(torch.bfloat16)

    model.save_pretrained(
        OUTPUT_PATH,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(OUTPUT_PATH)
    processor.save_pretrained(OUTPUT_PATH)

    print("Model merge success !")

if __name__ == "__main__":
    main()