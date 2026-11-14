"""
Local RLHF runner for Personal Health Assistant (SmolLM2-1.7B).

Run from the project root:
    python run_rlhf_local.py

This file orchestrates the existing project scripts:
    1) build_rm_data.py
    2) train_reward.py
    3) train_ppo.py (RLOO)

It does not replace or rewrite those training implementations.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def find_dir(*names):
    for name in names:
        path = ROOT / name
        if path.is_dir():
            return path

    raise FileNotFoundError(
        f"Not found folder {names} in project root: {ROOT}"
    )


def run_step(title: str, command: list[str]) -> None:
    print("\n" + "=" * 78, flush=True)
    print(title, flush=True)
    print("$ " + " ".join(command), flush=True)
    print("=" * 78, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="full local run of Reward Model + RLOO for SmolLM2."
    )
    parser.add_argument("--base", default="SmolLM2-1.7B")
    parser.add_argument("--sft-adapter", default="sft-qlora-v3")
    parser.add_argument("--rm-data", default="data")
    parser.add_argument("--rm-out", default="reward-sft-qlora-v3")
    parser.add_argument("--rlhf-adapter", default="sft-qlora-v3-rloo")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mini-batch", type=int, default=1)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--skip-build-data", action="store_true")
    parser.add_argument("--skip-reward", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rlhf_dir = find_dir("rlhf", "RLHF")
    dpo_dir = find_dir("DPO", "dpo")
    ui_dir = find_dir("ui", "chat_ui")

    base_path = ROOT / args.base
    adapter_path = ui_dir / "adapters" / args.sft_adapter

    if not base_path.exists():
        raise FileNotFoundError(f"base model not found: {base_path}")
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"not found SFT adapter: {adapter_path}\\adapter_config.json"
        )

    build_script = rlhf_dir / "build_rm_data.py"
    reward_script = rlhf_dir / "train_reward.py"
    ppo_script = rlhf_dir / "train_ppo.py"

    for script in (build_script, reward_script, ppo_script):
        if not script.exists():
            raise FileNotFoundError(f"script not found: {script}")

    # Step 1: create Reward Model preference pairs from generations.jsonl.
    if not args.skip_build_data:
        generations = dpo_dir / "data" / "generations.jsonl"
        if not generations.exists():
            raise FileNotFoundError(
                f"file not found: {generations}. "
                "if you have your own data "
                "run it with --skip-build-data."
            )
        run_step(
            "STEP 1/3 — building pairs for Reward Model",
            [sys.executable, str(build_script)],
        )

    # Step 2: train the scalar Reward Model from base + SFT adapter.
    reward_model_dir = rlhf_dir / "models" / args.rm_out
    if not args.skip_reward:
        run_step(
            "STEP 2/3 — training of Reward Model",
            [
                sys.executable, str(reward_script),
                "--base", args.base,
                "--sft-adapter", args.sft_adapter,
                "--data", str(Path(rlhf_dir.name) / args.rm_data),
                "--out", args.rm_out,
                "--epochs", "1",
            ],
        )

    if not reward_model_dir.exists():
        raise FileNotFoundError(
            f"trained model not found Reward Model: {reward_model_dir}. "
            "checkout --rm-out or do not use --skip-reward."
        )

    # Step 3: RLOO policy optimization.
    ppo_command = [
        sys.executable, str(ppo_script),
        "--algo", "rloo",
        "--base", args.base,
        "--sft-adapter", args.sft_adapter,
        "--group-size", str(args.group_size),
        "--batch-size", str(args.batch_size),
        "--mini-batch", str(args.mini_batch),
        "--reward", args.rm_out,
        "--kl-coef", str(args.kl_coef),
        "--name", args.rlhf_adapter,
        "--rollouts", f"{args.rlhf_adapter}_rollouts.jsonl",
    ]
    if args.force:
        ppo_command.append("--force")

    run_step("STEP 3/3 — RLOO: post-training Policy", ppo_command)

    output_adapter = ui_dir / "adapters" / args.rlhf_adapter
    if not (output_adapter / "adapter_config.json").exists():
        raise RuntimeError(
            "RLOO finished, but adapter_config.json not found: "
            f"{output_adapter}"
        )

    print("\nfinished. RLOO adapter saved:", output_adapter, flush=True)
    print("for evaluation, run eval file with the correct adapter name", flush=True)


if __name__ == "__main__":
    main()
