
"""Merge SFT + QLoRA-v3 into the SmolLM2-360M base model.

Usage:
    python hybrid/2_engine/merge.py

Output:
    hybrid/models/medicalai-360m-sft-qlora/

This creates a standalone Hugging Face model.
GGUF conversion is a separate step.
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "SmolLM2-360M"
ADAPTERS = ROOT / "ui" / "adapters"
DEFAULT_OUTPUT = (
    ROOT / "hybrid" / "models" / "medicalai-360m-sft-qlora"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--adapter",
        default="sft-qlora",
        help="Adapter folder inside ui/adapters/"
    )

    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="Output directory for the merged model"
    )

    args = parser.parse_args()

    adapter_path = ADAPTERS / args.adapter
    output_path = Path(args.out)

    if not BASE.exists():
        raise FileNotFoundError(f"Base model not found: {BASE}")

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    print("Base model:", BASE)
    print("Adapter:", adapter_path)
    print("Output:", output_path)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.float32
    )

    print("Loading adapter and merging...")
    model = PeftModel.from_pretrained(
        model,
        adapter_path
    ).merge_and_unload()

    print("Testing merged model...")

    prompt = (
        "User: What can cause neck pain after sitting "
        "at a computer for a long time?\n"
        "Assistant:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>")
            ]
        )

    print(
        "\nTest after merging:",
        tokenizer.decode(
            outputs[0, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
    )

    print("\nSaving merged model in FP16...")

    model.to(torch.float16).save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("\nMerge completed successfully.")
    print("Saved:", output_path)
    print("Next step: convert this HF model to GGUF.")


if __name__ == "__main__":
    main()
