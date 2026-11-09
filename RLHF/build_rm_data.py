"""
RLHF step-1
Build: Reward Modelni uqitish uchun sifatli juftliklar tayyorlaydi.

Run: 
    python rlhf/build_rm_data.py
"""


import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "DPO"))
import label  

MARGIN = 0.5
MAX_PER_PROMPT = 8
VAL_SHARE = 0.1
REFUSALS = ("harmful", "cheating", "unknown")
rng = random.Random(42)


def golds(item):
    """Namunaviy javoblar: SFT datadagi javob; rad etishda qat'iy rad jumlasi ham."""
    if item["task"] in ("harmful", "cheating"):
        return [
            label.FIXED[(item["task"], label.language(item["prompt"]) or "uz")],
            item["reference"],
        ]
    return [item["reference"]]


def prompt_pairs(item):
    answers = {}

    for a in item["answers"]:
        if a["text"]:
            answers.setdefault(a["text"], a["finished"])

    for g in golds(item):
        answers.setdefault(g, True)

    scored = [(label.rule_score(item, t, f), t) for t, f in answers.items()]
    pairs = []

    for sc, c in scored:
        for sr, r in scored:
            if sc - sr < MARGIN:
                continue

            if item["task"] not in REFUSALS:
                ratio = len(c) / max(1, len(r))

                if not label.LENGTH_RATIO[0] <= ratio <= label.LENGTH_RATIO[1]:
                    continue

                if item["task"] not in ("arithmetic", "translate") and label.f1(c, r) >= 0.85:
                    continue

            pairs.append({
                "id": item["id"],
                "task": item["task"],
                "prompt": item["prompt"],
                "chosen": c,
                "rejected": r,
                "margin": round(sc - sr, 3),
            })

    rng.shuffle(pairs)
    return pairs[:MAX_PER_PROMPT]


def balance(pairs):
    """Har bir vazifada chosen qisqa va uzun juftliklarini tenglashtiradi."""
    out = []

    for task in sorted({p["task"] for p in pairs}):
        mine = [p for p in pairs if p["task"] == task]

        if task in REFUSALS:
            out += mine
            continue

        short = [p for p in mine if len(p["chosen"]) < len(p["rejected"])]
        long_ = [p for p in mine if len(p["chosen"]) > len(p["rejected"])]
        same = [p for p in mine if len(p["chosen"]) == len(p["rejected"])]

        n = min(len(short), len(long_))
        out += rng.sample(short, n) + rng.sample(long_, n) + same

    rng.shuffle(out)
    return out


def main():
    source = ROOT / "DPO" / "data_v2" / "generations.jsonl"

    items = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    ids = sorted(i["id"] for i in items)
    rng.shuffle(ids)
    val_ids = set(ids[:round(len(ids) * VAL_SHARE)])

    raw = [p for item in items for p in prompt_pairs(item)]

    split = {
        "train": balance([p for p in raw if p["id"] not in val_ids]),
        "val": balance([p for p in raw if p["id"] in val_ids]),
    }

    out = HERE / "data_v2"
    out.mkdir(exist_ok=True)

    for name, rows in split.items():
        path = out / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )

    rows = split["train"] + split["val"]

    print(
        f"tenglashdan oldin {len(raw)} juftlik, keyin {len(rows)} "
        f"(train {len(split['train'])}, val {len(split['val'])}, "
        f"val promptlar {len(val_ids)})."
    )

    by_task = defaultdict(list)
    for p in rows:
        by_task[p["task"]].append(p)

    print(f"{'vazifa':<12} {'juftlik':>8} {'chosen qisqa':>13}")

    for task, mine in sorted(by_task.items()):
        short_count = sum(
            len(p["chosen"]) < len(p["rejected"]) for p in mine
        )
        print(f"{task:<12} {len(mine):>8} {short_count / len(mine):>12.0%}")

    normal = [
        p for p in rows
        if p["task"] not in REFUSALS
        and len(p["chosen"]) != len(p["rejected"])
    ]

    short_ratio = sum(
        len(p["chosen"]) < len(p["rejected"]) for p in normal
    ) / max(1, len(normal))

    print(
        f"Rad etishsiz vazifalarda chosen qisqaroq: "
        f"{short_ratio:.0%}"
    )

    print(
        "Juftliklar soni promptlar bo'yicha:",
        dict(sorted(Counter(Counter(p["id"] for p in rows).values()).items()))
    )


if __name__ == "__main__":
    main()
