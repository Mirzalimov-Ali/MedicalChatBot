"""
DPO 4-step: SFT adapterini DPO bilan uqitish.
Natija: ui/adapters/sft-lora-dpo/

Run:
    python DPO/train_dpo.py
    python DPO/train_dpo.py --max-steps 2
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig

HERE = Path(__file__).parent
ROOT = HERE.parent
ADAPTERS = ROOT / "ui" / "adapters"

MAX_TOKENS = 384
IGNORE = -100


def load(folder, name):
    path = HERE / folder / name
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def encode(tok, prompt, answer):
    text = f"User: {prompt}\nAssistant:"
    prompt_ids = tok(text)["input_ids"]
    answer_ids = tok(answer, add_special_tokens=False)["input_ids"]
    answer_ids += [tok.eos_token_id]

    ids = (prompt_ids + answer_ids)[:MAX_TOKENS]
    labels = ([IGNORE] * len(prompt_ids) + answer_ids)[:MAX_TOKENS]
    return ids, labels


def answer_logprob(model, seqs, pad_id, device):
    width = max(len(ids) for ids, _ in seqs)

    ids = torch.tensor(
        [x + [pad_id] * (width - len(x)) for x, _ in seqs],
        device=device,
    )
    mask = torch.tensor(
        [[1] * len(x) + [0] * (width - len(x)) for x, _ in seqs],
        device=device,
    )
    labels = torch.tensor(
        [y + [IGNORE] * (width - len(y)) for _, y in seqs],
        device=device,
    )

    logits = model(
        input_ids=ids,
        attention_mask=mask,
    ).logits[:, :-1].float()

    token_logp = -F.cross_entropy(
        logits.transpose(1, 2),
        labels[:, 1:],
        ignore_index=IGNORE,
        reduction="none",
    )

    return token_logp.sum(-1), (labels[:, 1:] != IGNORE).sum(-1)


def pair_logprobs(model, pairs, pad_id, device):
    n = len(pairs)
    lp, counts = answer_logprob(
        model,
        [c for c, _ in pairs] + [r for _, r in pairs],
        pad_id,
        device,
    )
    return lp[:n], lp[n:], counts[:n]


def dpo_stats(beta, pc, pr, rc, rr, nc, sft_weight):
    chosen_r = beta * (pc - rc)
    rejected_r = beta * (pr - rr)

    dpo = -F.logsigmoid(chosen_r - rejected_r).mean()
    sft = (-pc / nc.clamp(min=1)).mean()
    loss = dpo + sft_weight * sft

    return loss, {
        "loss": dpo.item(),
        "sft": sft.item(),
        "acc": (chosen_r > rejected_r).float().mean().item(),
        "margin": (chosen_r - rejected_r).mean().item(),
        "chosen_r": chosen_r.mean().item(),
        "rejected_r": rejected_r.mean().item(),
    }


def load_model(method, base_path, adapter_path, device):
    if method == "qlora":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=quant,
            device_map={"": 0},
        )
        base = prepare_model_for_kbit_training(base)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            dtype=torch.float32 if device == "cpu" else torch.float16,
        ).to(device)
        base.config.use_cache = False
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()

    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=True,
    )

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    return model


@torch.no_grad()
def reference_logprobs(model, data, batch_size, pad_id, device):
    model.eval()
    output = [
        pair_logprobs(model, data[i:i + batch_size], pad_id, device)
        for i in range(0, len(data), batch_size)
    ]
    model.train()

    return (
        torch.cat([x[0] for x in output]),
        torch.cat([x[1] for x in output]),
    )


@torch.no_grad()
def evaluate(model, data, ref, batch_size, beta, sft_weight, pad_id, device):
    model.eval()
    totals = {}
    count = 0

    for i in range(0, len(data), batch_size):
        pc, pr, nc = pair_logprobs(
            model, data[i:i + batch_size], pad_id, device
        )
        _, stats = dpo_stats(
            beta,
            pc,
            pr,
            ref[0][i:i + batch_size],
            ref[1][i:i + batch_size],
            nc,
            sft_weight,
        )

        n = len(pc)
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + value * n
        count += n

    model.train()
    return {key: value / count for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lora", "qlora"], default="lora")
    parser.add_argument("--base", default="SmolLM2-360M")
    parser.add_argument("--sft-adapter", default="sft-lora")
    parser.add_argument("--data", default="data")
    parser.add_argument("--name", default="sft-lora-dpo")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sft-weight", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.method == "qlora" and device != "cuda":
        raise SystemExit("QLoRA uchun NVIDIA GPU kerak.")

    base_path = ROOT / args.base
    adapter_path = ADAPTERS / args.sft_adapter
    output_path = ADAPTERS / args.name

    if not base_path.exists():
        raise FileNotFoundError(f"Base model topilmadi: {base_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"SFT adapter topilmadi: {adapter_path}")

    tok = AutoTokenizer.from_pretrained(base_path)
    tok.pad_token = tok.pad_token or tok.eos_token

    train_rows = load(args.data, "train.jsonl")
    val_rows = load(args.data, "val.jsonl")

    if not train_rows or not val_rows:
        raise ValueError("Train yoki validation dataset bo'sh.")

    train = [
        (
            encode(tok, row["prompt"], row["chosen"]),
            encode(tok, row["prompt"], row["rejected"]),
        )
        for row in train_rows
    ]
    val = [
        (
            encode(tok, row["prompt"], row["chosen"]),
            encode(tok, row["prompt"], row["rejected"]),
        )
        for row in val_rows
    ]

    if args.max_steps:
        train = train[:args.max_steps * args.batch_size]

    model = load_model(args.method, base_path, adapter_path, device)
    model.print_trainable_parameters()

    pad_id = tok.pad_token_id
    print(
        f"Device: {device} | Train: {len(train)} | Val: {len(val)}",
        flush=True,
    )

    print("Reference log-probabilities hisoblanmoqda...", flush=True)
    ref_train = reference_logprobs(
        model, train, args.batch_size, pad_id, device
    )
    ref_val = reference_logprobs(
        model, val, args.batch_size, pad_id, device
    )

    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup = max(1, total_steps // 10)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            (step + 1) / warmup
            if step < warmup
            else max(
                0.0,
                (total_steps - step) / max(1, total_steps - warmup),
            )
        ),
    )

    initial = evaluate(
        model, val, ref_val, args.batch_size,
        args.beta, args.sft_weight, pad_id, device,
    )
    print(f"Initial validation DPO loss: {initial['loss']:.3f}")

    logs = [{"step": 0, "val": initial}]
    order = list(range(len(train)))
    step = 0
    start = time.time()
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        random.shuffle(order)
        model.train()

        for i in range(0, len(order), args.batch_size):
            indices = order[i:i + args.batch_size]
            batch = [train[j] for j in indices]

            pc, pr, nc = pair_logprobs(
                model, batch, pad_id, device
            )
            loss, stats = dpo_stats(
                args.beta,
                pc,
                pr,
                ref_train[0][indices],
                ref_train[1][indices],
                nc,
                args.sft_weight,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            logs.append({
                "step": step,
                "epoch": epoch,
                **{key: round(value, 4) for key, value in stats.items()},
            })

            if step <= 3 or step % 10 == 0 or step == total_steps:
                elapsed = time.time() - start
                remaining = elapsed / step * (total_steps - step) / 60
                print(
                    f"Step {step}/{total_steps} | "
                    f"loss {stats['loss']:.3f} | "
                    f"acc {stats['acc']:.2f} | "
                    f"margin {stats['margin']:.3f} | "
                    f"SFT {stats['sft']:.3f} | "
                    f"qoldi ~{remaining:.1f} min",
                    flush=True,
                )

            if step >= total_steps:
                break

        val_stats = evaluate(
            model, val, ref_val, args.batch_size,
            args.beta, args.sft_weight, pad_id, device,
        )
        logs.append({"step": step, "epoch": epoch, "val": val_stats})

        val_loss = val_stats["loss"] + args.sft_weight * val_stats["sft"]
        saved = val_loss < best and not args.max_steps

        if saved:
            best = val_loss
            model.save_pretrained(output_path)

        print(
            f"Epoch {epoch} | val DPO loss {val_stats['loss']:.3f} | "
            f"val acc {val_stats['acc']:.2f}"
            + (" | saved" if saved else ""),
            flush=True,
        )

        if step >= total_steps:
            break

    if args.max_steps:
        print("Test tugadi. Adapter saqlanmadi.")
        return

    (output_path / "training_log.json").write_text(
        json.dumps(logs, indent=2),
        encoding="utf-8",
    )
    print(f"Tayyor: {output_path}")
    print(f"Eng yaxshi validation loss: {best:.3f}")
    print(f"Vaqt: {(time.time() - start) / 60:.1f} daqiqa")


if __name__ == "__main__":
    main()