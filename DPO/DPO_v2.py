"""
DPO v2: SFT v3 adapterini DPO bilan uqitish.
Natija: ui/adapters/sft-lora-dpo-v2/ yoki sft-qlora-dpo-v2/

Run:
    python DPO/DPO_v2.py --method lora
    python DPO/DPO_v2.py --method qlora
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

BASE = "SmolLM2-360M"
SFT_ADAPTER = "sft-lora-v3"
DPO_ADAPTER = "sft-lora-dpo-v2"
DATA_DIR = "data_v2"


def run(command):
    print("\n>", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        choices=["all", "prompts", "generate", "label", "train", "eval"],
        default="all"
    )

    parser.add_argument(
        "--method",
        choices=["lora", "qlora"],
        default="lora"
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()
    DPO_ADAPTER = {
        "lora": "sft-lora-dpo-v2",
        "qlora": "sft-qlora-dpo-v2"
    }[args.method]

    dpo = ROOT
    prompts = ROOT / "data" / "prompts.jsonl"

    # 1. Build prompts
    if args.stage in ("all", "prompts"):
        run([
            PYTHON,
            str(dpo / "01_build_prompts.py")
        ])

    # 2. Generate 8 answers per prompt
    if args.stage in ("all", "02_generate"):
        if not prompts.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompts}"
            )

        run([
            PYTHON,
            str(dpo / "02_generate.py"),
            "--base", BASE,
            "--adapter", SFT_ADAPTER,
            "--n", "8",
            "--out", DATA_DIR
        ])

    # 3. Create chosen/rejected pairs
    if args.stage in ("all", "label"):
        run([
            PYTHON,
            str(dpo / "03_label.py"),
            "--dir", DATA_DIR
        ])

    # 4. Train DPO v2
    if args.stage in ("all", "train"):
        run([
            PYTHON,
            str(dpo / "04_train_dpo.py"),
            "--method", args.method,
            "--base", BASE,
            "--sft-adapter", SFT_ADAPTER,
            "--data", DATA_DIR,
            "--name", DPO_ADAPTER,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", "5e-6",
            "--beta", "0.1",
            "--sft-weight", "1.0"
        ])

    # 5. Evaluate on test questions
    if args.stage in ("all", "eval"):

        eval_dir = ROOT.parent / "eval"
        eval_file = eval_dir / "eval_set.json"

        with open(eval_file, "r", encoding="utf-8") as f:
            total_questions = len(json.load(f))

        print(f"\nTotal evaluation questions: {total_questions}")

        run([
            PYTHON,
            str(eval_dir / "01_run_eval.py"),
            "--base", str(ROOT.parent / BASE),
            "--model", DPO_ADAPTER
        ])

    print("\nDPO v2 pipeline finished.")


if __name__ == "__main__":
    main()