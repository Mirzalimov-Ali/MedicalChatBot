import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

SFT = Path(__file__).parent.parent   
ROOT = SFT.parent
BASE_PATH = ROOT / "SmolLM2-360M"
ADAPTERS = ROOT / "ui" / "adapters" 

MAX_TOKENS = 384 
IGNORE = -100    


def load(name):
    return [json.loads(line) for line in (SFT / name).read_text(encoding="utf-8").splitlines()]


def encode(tok, messages):
    prompt_text = (
        f"User: {messages[0]['content']}\n"
        f"Assistant:"
    )
    answer_text = messages[1]["content"]

    prompt_ids = tok(
        prompt_text,
        add_special_tokens=True
    )["input_ids"]

    answer_ids = tok(
        " " + answer_text,
        add_special_tokens=False
    )["input_ids"]

    if tok.eos_token_id is not None:
        answer_ids.append(tok.eos_token_id)

    ids = (prompt_ids + answer_ids)[:MAX_TOKENS]
    labels = ([IGNORE] * len(prompt_ids) + answer_ids)[:MAX_TOKENS]

    return ids, labels


def batches(examples, batch_size, pad_id, shuffle):
    order = list(range(len(examples)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [examples[j] for j in order[i:i + batch_size]]
        width = max(len(ids) for ids, _ in chunk)
        yield (torch.tensor([ids + [pad_id] * (width - len(ids)) for ids, _ in chunk]),
               torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in chunk]),
               torch.tensor([labels + [IGNORE] * (width - len(labels)) for _, labels in chunk]))


@torch.no_grad()
def evaluate(model, examples, batch_size, pad_id, device):
    model.eval()
    total, n = 0.0, 0
    for ids, mask, labels in batches(examples, batch_size, pad_id, shuffle=False):
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
        total += out.loss.item() * len(ids)
        n += len(ids)
    model.train()
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="sft-lora", help="adapter folder name")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-steps", type=int)
    args = p.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.name

    tok = AutoTokenizer.from_pretrained(BASE_PATH)
    tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, dtype=torch.float32).to(device)
    print(f"LoRA: base model in full precision | memory usage {base.get_memory_footprint() / 1e6:.0f} MB")

    model = get_peft_model(base, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    train = [encode(tok, r["messages"]) for r in load("train.jsonl")]
    val = [encode(tok, r["messages"]) for r in load("val.jsonl")]
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup = max(1, total_steps // 20)
    print(f"Device: {device} | train {len(train)}, val {len(val)} | {total_steps} steps "
        f"({steps_per_epoch} steps/epoch) | average length {sum(len(i) for i, _ in train) // len(train)} tokens")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    log, step, start, best = [], 0, time.time(), float("inf")
    model.train()
    for epoch in range(1, args.epochs + 1):
        for ids, mask, labels in batches(train, args.batch_size, tok.pad_token_id, shuffle=True):
            loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            log.append({"step": step, "epoch": epoch, "train_loss": round(loss.item(), 4)})
            if step <= 3 or step % 10 == 0 or step == total_steps:
                elapsed = time.time() - start
                left = elapsed / step * (total_steps - step)
                print(f"Step {step}/{total_steps} | Epoch {epoch} | Loss {loss.item():.3f} | "
                    f"Elapsed {elapsed / 60:.1f} min | Remaining ~{left / 60:.0f} min", flush=True)
            if step == total_steps:
                break
        if args.max_steps:
            break
        val_loss = evaluate(model, val, args.batch_size, tok.pad_token_id, device)
        log.append({"step": step, "epoch": epoch, "val_loss": round(val_loss, 4)})
        saved = val_loss < best
        if saved:
            best = val_loss
            model.save_pretrained(ADAPTERS / name)
        print(f"=== Epoch {epoch} completed | Validation loss {val_loss:.3f}" +
        (" | Saved" if saved else " | No improvement"),
        flush=True)

    if args.max_steps:
        print("Test run completed; adapter was not saved.")
        return

    (ADAPTERS / name / "training_log.json").write_text(
        json.dumps(log, indent=1),
        encoding="utf-8"
    )

    print(
        f"Training completed: {ADAPTERS / name} | "
        f"Best validation loss {best:.3f} | "
        f"Total time {(time.time() - start) / 60:.1f} min" 
    )


if __name__ == "__main__":
    main()