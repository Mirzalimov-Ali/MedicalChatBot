
"""SFT v2: SmolLM2-360M modelini LoRA yoki QLoRA bilan o'qitish.

Ishlatish:
    python sft_v2/train.py --method lora --epochs 5 --rank 64
    python sft_v2/train.py --method qlora --epochs 5 --rank 64

Adapterlar:
    ui/adapters/sft-lora
    ui/adapters/sft-qlora
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


ROOT = Path(__file__).resolve().parent.parent
ADAPTERS = ROOT / "ui" / "adapters"


def log(message):
    print(message, flush=True)


class SFTDataset(Dataset):
    def __init__(self, path, tok, max_length=512):
        self.rows = []
        self.tok = tok
        self.max_length = max_length

        log(f"  Dataset fayli o'qilmoqda: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

        log(f"  Yuklangan namunalar soni: {len(self.rows)}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        messages = row["messages"]

        prompt = f"User: {messages[0]['content']}\nAssistant:"
        answer = messages[1]["content"]

        prompt_ids = self.tok(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        answer_ids = self.tok(
            answer + (self.tok.eos_token or ""),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        input_ids = (prompt_ids + answer_ids)[:self.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:self.max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_fn(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])

        input_ids.append(item["input_ids"] + [pad_id] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
        attention_mask.append(
            [1] * len(item["input_ids"]) + [0] * pad_len
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def load_base(base, method):
    if method == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA uchun GPU kerak.")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=quant_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)

    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            base,
            dtype=dtype,
        )

        if torch.cuda.is_available():
            model = model.to("cuda")

    return model


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    steps = 0

    log("  Validation: model tekshirilmoqda...")

    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            total_loss += outputs.loss.item()
            steps += 1

            log(
                f"  Validation batch {step + 1}/{len(loader)}"
                f" | Loss: {outputs.loss.item():.4f}"
            )

    model.train()
    return total_loss / max(steps, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lora", "qlora"], default="lora")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--base", default="SmolLM2-360M")
    parser.add_argument("--data", default="sft")
    parser.add_argument("--name", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log("\n" + "=" * 60)
    log(" SFT v2 — MODELNI O'QITISH")
    log("=" * 60)

    log("\n[1/8] Sozlamalar tekshirilmoqda...")
    log(f"  Usul: {args.method.upper()}")
    log(f"  Epochs: {args.epochs}")
    log(f"  LoRA rank: {args.rank}")
    log(f"  Batch size: {args.batch_size}")
    log(f"  Learning rate: {args.lr}")
    log(f"  Maksimal token uzunligi: {args.max_length}")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  Qurilma: {device}")
    if device.type == "cuda":
        log(f"  GPU: {torch.cuda.get_device_name(0)}")

    base_path = ROOT / args.base
    if not base_path.exists():
        base_path = Path(args.base)

    data_dir = ROOT / args.data
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    log("\n[2/8] Fayllar tekshirilmoqda...")
    log(f"  Model: {base_path}")
    log(f"  Training dataset: {train_path}")
    log(f"  Validation dataset: {val_path}")

    if not train_path.exists():
        raise FileNotFoundError(f"Train data topilmadi: {train_path}")

    if not val_path.exists():
        raise FileNotFoundError(f"Validation data topilmadi: {val_path}")

    name = args.name or f"sft-{args.method}"
    output_dir = ADAPTERS / name

    log("\n[3/8] Tokenizer yuklanmoqda...")
    tok = AutoTokenizer.from_pretrained(str(base_path))

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log("  Tokenizer muvaffaqiyatli yuklandi.")

    log("\n[4/8] Asosiy model yuklanmoqda...")
    model = load_base(str(base_path), args.method)
    log("  Asosiy model yuklandi.")

    log("\n[5/8] LoRA adapter sozlanmoqda...")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    log("  LoRA adapter tayyor.")

    log("\n[6/8] Dataset va DataLoader tayyorlanmoqda...")
    train_dataset = SFTDataset(
        train_path, tok, max_length=args.max_length
    )
    val_dataset = SFTDataset(
        val_path, tok, max_length=args.max_length
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset bo'sh!")

    if len(val_dataset) == 0:
        raise ValueError("Validation dataset bo'sh!")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tok.pad_token_id),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tok.pad_token_id),
    )

    log(f"  Training namunalari: {len(train_dataset)}")
    log(f"  Validation namunalari: {len(val_dataset)}")
    log(f"  Training batchlari: {len(train_loader)}")
    log(f"  Validation batchlari: {len(val_loader)}")

    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    total_steps = math.ceil(
        len(train_loader) / args.grad_accum
    ) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    log(f"  Optimizer: AdamW")
    log(f"  Jami optimizer qadamlari: {total_steps}")
    log("  DataLoader va optimizer tayyor.")

    best_val_loss = float("inf")

    log("\n[7/8] TRAINING BOSHLANDI!")
    log("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        log(f"\n>>> EPOCH {epoch + 1}/{args.epochs} boshlandi")

        for step, batch in enumerate(train_loader):
            log(
                f"  [TRAIN] Epoch {epoch + 1}/{args.epochs}"
                f" | Batch {step + 1}/{len(train_loader)}"
                " | Forward hisoblash boshlandi...",
            )

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum

            log(
                f"  [TRAIN] Forward tugadi"
                f" | Loss: {outputs.loss.item():.4f}"
                " | Backward hisoblash boshlandi...",
            )

            loss.backward()
            total_loss += outputs.loss.item()

            log("  [TRAIN] Backward tugadi.")

            if (step + 1) % args.grad_accum == 0 or (
                step + 1
            ) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                log("  [TRAIN] Optimizer qadami bajarildi.")

            log(
                f"  [TRAIN] Batch {step + 1}/{len(train_loader)}"
                f" tugadi | Loss: {outputs.loss.item():.4f}",
            )

        train_loss = total_loss / max(len(train_loader), 1)

        log(f"\n>>> EPOCH {epoch + 1}: Training tugadi.")
        log(f"  O'rtacha train loss: {train_loss:.4f}")

        val_loss = evaluate(model, val_loader, device)

        log(f"  Validation loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            log("  Yangi eng yaxshi natija! Adapter saqlanmoqda...")
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tok.save_pretrained(output_dir)

            log(f"  Adapter saqlandi: {output_dir}")
        else:
            log("  Validation loss yaxshilanmadi. Oldingi adapter saqlanib qoladi.")

        log(
            f"<<< EPOCH {epoch + 1}/{args.epochs} tugadi"
            f" | Train loss: {train_loss:.4f}"
            f" | Val loss: {val_loss:.4f}"
        )

    log("\n[8/8] TRAINING YAKUNLANDI!")
    log(f"  Eng yaxshi validation loss: {best_val_loss:.4f}")
    log(f"  Adapter manzili: {output_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()
