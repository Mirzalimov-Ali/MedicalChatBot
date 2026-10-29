"""
run: 
- python SFT_v3/train.py --method lora --epochs 5 --rank 64
- python SFT_v3/train.py --method qlora --epochs 5 --rank 64
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["lora", "qlora"], required=True)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--rank", type=int, default=64)
args = parser.parse_args()

# 1. LoRA/QLoRA bilan uqitish (SFT_v2 train bilan bir hil usulda)
subprocess.run([
    sys.executable,
    str(ROOT / "SFT_v2" / "train.py"),
    "--method", args.method,
    "--base", "SmolLM2-360M",
    "--data", "SFT_v3/data",
    "--name", f"sft-{args.method}-v3",
    "--epochs", str(args.epochs),
    "--rank", str(args.rank),
], cwd=ROOT, check=True)

# 2. Evaluation
subprocess.run([
    sys.executable,
    str(ROOT / "eval" / "01_run_eval.py"),
    "--base", "SmolLM2-360M",
    "--model", f"sft-{args.method}-v3",
    "--name", f"sft-{args.method}-v3",
], cwd=ROOT, check=True)