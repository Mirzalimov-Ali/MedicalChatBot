"""
RLHF step-2
Reward Modelni train qiladi.

Run:
    python rlhf/train_reward.py

Natija:
    rlhf/models/reward-sft-qlora/

Reward Model har bir javobga bitta ball beradi.
Chosen javobning balli rejected javobnikidan yuqori bulishi kerak.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import RLHF.scalar_model as scalar_model

HERE = Path(__file__).parent
ROOT = HERE.parent
MAX_TOKENS = 384   # bundan uzun namuna qirqiladi


def encode(tok, prompt, answer):
    text = f"User: {prompt}\nAssistant: {answer}"
    return tok(text, truncation=True, max_length=512)["input_ids"]


def load_pairs(tok, folder, name):
    rows = [json.loads(line) for line in (folder / name).read_text(encoding="utf-8").splitlines()]
    return [{**r, "c": encode(tok, r["prompt"], r["chosen"]), "r": encode(tok, r["prompt"], r["rejected"])} for r in rows]


def pair_scores(model, batch, pad_id, device):
    s = scalar_model.last_scores(model, [p["c"] for p in batch] + [p["r"] for p in batch], pad_id, device)
    return s[:len(batch)], s[len(batch):]


@torch.no_grad()
def evaluate(model, data, batch_size, pad_id, device):
    model.eval()
    sc, sr = zip(*(pair_scores(model, data[i:i + batch_size], pad_id, device) for i in range(0, len(data), batch_size)))
    sc, sr = torch.cat(sc), torch.cat(sr)
    model.train()
    return {"loss": -F.logsigmoid(sc - sr).mean().item(), "acc": (sc > sr).float().mean().item()}, sc, sr


def length_report(data, sc, sr):
    shorter = [(c > r) == (len(p["chosen"]) < len(p["rejected"]))
               for p, c, r in zip(data, sc.tolist(), sr.tolist()) if len(p["chosen"]) != len(p["rejected"])]
    scores = sc.tolist() + sr.tolist()
    lengths = [len(p["chosen"]) for p in data] + [len(p["rejected"]) for p in data]
    corr = torch.corrcoef(torch.tensor([scores, lengths], dtype=torch.float))[0, 1].item()
    return sum(shorter) / max(1, len(shorter)), corr


def length_gate(data, sc, sr):
    keep = [
        i for i, p in enumerate(data)
        if p["task"] not in ("harmful", "cheating", "unknown")
    ]

    if len(keep) < 2:
        return {"ok": True}

    part = [data[i] for i in keep]

    shorter, _ = length_report(part, sc[keep], sr[keep])

    unequal = [
        p for p in part
        if len(p["chosen"]) != len(p["rejected"])
    ]

    in_data = (
        sum(
            len(p["chosen"]) < len(p["rejected"])
            for p in unequal
        ) / max(1, len(unequal))
    )

    diff = sc[keep] - sr[keep]

    dlen = torch.tensor(
        [
            len(p["chosen"]) - len(p["rejected"])
            for p in part
        ],
        dtype=torch.float,
        device=diff.device
    )

    corr = (
        torch.corrcoef(torch.stack([diff, dlen]))[0, 1].item()
        if dlen.std() > 0 and diff.std() > 0
        else 0.0
    )

    return {
        "rm_picks_shorter": round(shorter, 3),
        "data_chosen_shorter": round(in_data, 3),
        "pair_corr": round(corr, 3),
        "ok": shorter <= in_data + 0.15 and corr >= -0.3
    }


def length_slope(data, sc, sr):
    keep = [i for i, p in enumerate(data) if p["task"] not in ("harmful", "cheating", "unknown")]
    diff = (sc[keep] - sr[keep]).float().cpu()
    dlen = torch.tensor([len(data[i]["chosen"]) - len(data[i]["rejected"]) for i in keep], dtype=torch.float)
    dlen_c = dlen - dlen.mean()
    return ((diff - diff.mean()) * dlen_c).sum().item() / max(1e-8, (dlen_c ** 2).sum().item())


def lengths(data, key):
    return torch.tensor([len(p[key]) for p in data], dtype=torch.float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="SmolLM2-1.7B", help="reward model uchun base")
    p.add_argument("--sft-adapter", default="sft-qlora", help="SFT adapter papkasi")
    p.add_argument("--data", default="rlhf/data", help="chosen/rejected pairs")
    p.add_argument("--out", default="reward-sft-qlora", help="rlhf/models/ ichidagi natija papkasi")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8, help="juftliklar soni (ketma-ketliklar 2 barobar)")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-steps", type=int)
    args = p.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ROOT / args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    pad = tok.pad_token_id

    train = load_pairs(tok, ROOT / args.data, "train.jsonl")
    val = load_pairs(tok, ROOT / args.data, "val.jsonl")
    if args.max_steps:
        train, val = train[:args.max_steps * args.batch_size], val[:args.batch_size]

    model = scalar_model.build(args.base, args.sft_adapter, args.rank, device).train()
    model.backbone.print_trainable_parameters()
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = args.max_steps or math.ceil(len(train) / args.batch_size) * args.epochs
    warmup = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    s0, _, _ = evaluate(model, val, args.batch_size, pad, device)
    print(f"Boshlanishida: val loss {s0['loss']:.3f} (tasodifiy head: ~0.693), val acc {s0['acc']:.2f}", flush=True)
    out = HERE / "models" / args.out
    log, step, start, best = [{"step": 0, "val": s0}], 0, time.time(), float("inf")
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        for i in range(0, len(train), args.batch_size):
            sc, sr = pair_scores(model, train[i:i + args.batch_size], pad, device)
            loss = -F.logsigmoid(sc - sr).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            acc = (sc > sr).float().mean().item()
            log.append({"step": step, "epoch": epoch, "loss": round(loss.item(), 4), "acc": round(acc, 4)})
            if step <= 3 or step % 10 == 0 or step == total_steps:
                elapsed = time.time() - start
                print(f"qadam {step}/{total_steps} | loss {loss.item():.3f} | acc {acc:.2f} | "
                      f"farq {(sc - sr).mean().item():+.2f} | qoldi ~{elapsed / step * (total_steps - step) / 60:.0f} daq",
                      flush=True)
            if step == total_steps:
                break
        sv, _, _ = evaluate(model, val, args.batch_size, pad, device)
        log.append({"step": step, "epoch": epoch, "val": sv})
        saved = sv["loss"] < best
        if saved:
            best = sv["loss"]
            # Ballar normallashtiriladi: PPO'da reward o'rtachasi 0, og'ishi 1 bo'lsin. Statistika train javoblaridan.
            _, tc, tr = evaluate(model, train, args.batch_size, pad, device)
            both = torch.cat([tc, tr])
            scalar_model.save(model, out, {"base": args.base, "sft_adapter": args.sft_adapter,
                                           "mean": both.mean().item(), "std": both.std().item() if len(both) > 1 else 1.0,
                                           "val_acc": sv["acc"], "val_loss": sv["loss"], "epoch": epoch})
        print(f"=== epoch {epoch} | val loss {sv['loss']:.3f} | val acc {sv['acc']:.2f}" + (" | saqlandi" if saved else ""),
              flush=True)
        if step == total_steps:
            break

    # Saqlangan (eng yaxshi) model bilan yakuniy tekshiruv.
    model, meta = scalar_model.load(out, device)
    scores = {}
    for name, data in (("train", train), ("val", val)):
        s, sc, sr = evaluate(model, data, args.batch_size, pad, device)
        scores[name] = (sc, sr)
        shorter, corr = length_report(data, sc, sr)
        print(f"{name}: acc {s['acc']:.2f} | qisqa javobni afzal ko'rdi: {shorter:.0%} juftlikda | "
              f"ball va uzunlik korrelyatsiyasi {corr:+.2f}")
    print("Data'da chosen qisqaroq:", f"{sum(len(p['chosen']) < len(p['rejected']) for p in train) / len(train):.0%}",
          "(reward model shundan ko'p bo'lsa, uzunlikni o'rgangan)")
    right = (sc > sr).tolist()   # sc, sr: oxirgi sikldagi val
    for task in sorted({p["task"] for p in val}):
        mine = [ok for p, ok in zip(val, right) if p["task"] == task]
        print(f"  val acc [{task}]: {sum(mine) / len(mine):.2f} ({len(mine)} juftlik)")
    meta["length_gate_raw"] = length_gate(val, sc, sr)
    print("Uzunlik tekshiruvi, tuzatishsiz:", meta["length_gate_raw"])
    # Length-controlled reward: qiyalik train'da, tekshiruv val'da (o'zi o'lchagan data'da tekshirsa — aldash bo'ladi).
    coef = length_slope(train, *scores["train"])
    vc = sc.float().cpu() - coef * lengths(val, "chosen")
    vr = sr.float().cpu() - coef * lengths(val, "rejected")
    gate = length_gate(val, vc, vr)
    gate["val_acc"] = round((vc > vr).float().mean().item(), 3)
    gate["ok"] = gate["ok"] and gate["val_acc"] >= meta["val_acc"] - 0.05   # tuzatish to'g'ri javoblarni buzmasin
    meta["length_coef"], meta["length_gate"] = coef, gate
    print(f"Uzunlik tuzatishi: ball - {coef:+.4f} * belgilar soni (train juftliklaridan)")
    print("Uzunlik tekshiruvi (val, rad etishsiz, tuzatilgan ball bilan):", gate,
          "" if gate["ok"] else "-> O'TMADI: PPO bu reward modelni ishlatmaydi")
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (out / "training_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"Tayyor: {out} | eng yaxshi val loss {best:.3f} | ball o'rtachasi {meta['mean']:+.2f}, og'ishi {meta['std']:.2f} | "
          f"jami {(time.time() - start) / 60:.1f} daqiqa")


if __name__ == "__main__":
    main()