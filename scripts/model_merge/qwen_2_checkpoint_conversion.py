import argparse
import torch

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
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

    # 1. Carica tokenizer + processor
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # 2. Carica modello base
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. Carica adapter PEFT
    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
        device_map="auto",  # Assicura che l'adapter sia sullo stesso device
    )

    # 4. Merge definitivo (QUI avviene la magia)
    model = model.merge_and_unload()
    
    # 🔹 Sposta tutto su CUDA e bf16
    model.to("cuda").to(torch.bfloat16)

    # 5. Salvataggio modello completo (sharded)
    model.save_pretrained(
        OUTPUT_PATH,
        safe_serialization=True,
        # max_shard_size="5GB",
    )

    # 6. Salva tokenizer e processor
    tokenizer.save_pretrained(OUTPUT_PATH)
    processor.save_pretrained(OUTPUT_PATH)

    print("✅ Merge completato. Modello pronto per vLLM.")

if __name__ == "__main__":
    main()