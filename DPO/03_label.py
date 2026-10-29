"""
DPO 3-step: Generated answersdan chosen/rejected juftliklarini tuzish.
Natija: DPO/data/train.jsonl, val.jsonl

Run:
    python DPO/label.py
    python DPO/label.py --dir data
"""

import argparse
import ast
import json
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
VAL_SHARE = 0.15
LENGTH_RATIO = (0.6, 1.7)
rng = random.Random(42)

REFUSAL_OK = re.compile(r"can't help|cannot help|unable to help|can't assist|cannot assist", re.I)
REFUSAL_ANY = re.compile(r"sorry|can't help|cannot help|unable to help|can't assist|cannot assist", re.I)
UNKNOWN_OK = re.compile(r"^(i don't know|i do not know)\b", re.I)

FIXED = {
    "harmful": "Sorry, I can't help with that.",
    "cheating": "Sorry, I can't help with answers during an exam.",
}


def words(text):
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())


def f1(a, b):
    wa, wb = Counter(words(a)), Counter(words(b))
    common = sum((wa & wb).values())
    if not common:
        return 0.0
    p = common / max(1, sum(wa.values()))
    r = common / max(1, sum(wb.values()))
    return 2 * p * r / (p + r)


def calc(expr):
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("Unsupported expression")

    return walk(ast.parse(expr.strip(), mode="eval").body)


OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a // b,
}


def arithmetic_ok(text, number):
    first = re.search(r"=\s*(-?\d+)", text)
    if not first or int(first.group(1)) != number:
        return False

    for lhs, rhs in re.findall(
        r"([\d\s+\-*/()]*\d[\d\s+\-*/()]*)=\s*(-?\d+)", text
    ):
        try:
            if calc(lhs) != int(rhs):
                return False
        except (SyntaxError, ValueError, ZeroDivisionError):
            pass

    return True


def acceptable_refusal(item, text, chosen):
    if item["task"] == "unknown":
        return (
            bool(UNKNOWN_OK.match(text))
            and not REFUSAL_ANY.search(text)
            and (len(words(text)) <= 5 or f1(text, item["reference"]) >= 0.5)
        )

    return bool(REFUSAL_OK.search(text)) and (
        f1(text, chosen) >= 0.7 or f1(text, item["reference"]) >= 0.7
    )


def refusal_pair(item):
    task = item["task"]
    chosen = item["reference"] if task == "unknown" else FIXED[task]

    bad = [
        a["text"] for a in item["answers"]
        if a.get("text") and not acceptable_refusal(item, a["text"], chosen)
    ]
    if not bad:
        return None

    rejected = sorted(
        bad,
        key=lambda t: (
            bool(REFUSAL_OK.search(t) or UNKNOWN_OK.match(t)),
            abs(len(t) - len(chosen)),
        ),
    )[0]

    return chosen, rejected, "reference" if task == "unknown" else "fixed", []


def task_score(item, text):
    task = item["task"]

    if task == "arithmetic":
        return 1.0 if arithmetic_ok(text, item["number"]) else 0.0

    if task == "list":
        found = re.findall(r"^\s*\d+[.)]\s+(.+)$", text, flags=re.M)
        valid = (
            len(found) == item["items"]
            and all(len(words(x)) >= 2 for x in found)
        )
        return 1.0 if valid and not REFUSAL_ANY.search(text) else 0.0

    if task == "translate":
        reference = set(words(item["reference"]))
        answer = words(text)
        if reference and reference <= set(answer) and len(answer) <= 1.5 * len(reference) + 1:
            return 1.0
        return 0.6 * f1(text, item["reference"])

    return f1(text, item["reference"])


def penalty(item, answer):
    text = answer["text"]
    reasons = []

    if REFUSAL_ANY.search(text) and item["task"] not in ("harmful", "cheating", "unknown"):
        reasons.append("unnecessary refusal")

    if not answer.get("finished", False):
        reasons.append("unfinished")

    sentences = [
        s.strip() for s in re.split(r"[.\n]", text)
        if len(s.strip()) > 10
    ]
    if len(sentences) != len(set(sentences)):
        reasons.append("repetition")

    return reasons


def score(item, answer):
    if not answer["text"].strip():
        return -1.0, ["empty"]

    reasons = penalty(item, answer)
    return task_score(item, answer["text"]) - 0.5 * len(reasons), reasons


def rule_score(item, text, finished):
    task = item["task"]

    if task in ("harmful", "cheating", "unknown"):
        chosen = item["reference"] if task == "unknown" else FIXED[task]
        return 1.0 if text and acceptable_refusal(item, text, chosen) else 0.0

    return score(item, {"text": text, "finished": finished})[0]


def normal_pair(item):
    scored = [(score(item, a), a["text"]) for a in item["answers"]]
    good = 1.0 if item["task"] in ("arithmetic", "list", "translate") else 0.5

    (best, _), best_text = max(
        scored, key=lambda x: (x[0][0], -len(x[1]))
    )

    if best >= good:
        chosen, chosen_score, source = best_text, best, "model"
    elif best < good / 2:
        chosen, chosen_score, source = item["reference"], good, "reference"
    else:
        return None

    worse = [
        (s, reasons, text)
        for (s, reasons), text in scored
        if s <= chosen_score - 0.5 and text != chosen
    ]
    if not worse:
        return None

    _, reasons, rejected = min(
        worse, key=lambda x: abs(len(x[2]) - len(chosen))
    )

    if item["task"] not in ("arithmetic", "translate") and f1(chosen, rejected) >= 0.85:
        return None

    ratio = len(chosen) / max(1, len(rejected))
    if not LENGTH_RATIO[0] <= ratio <= LENGTH_RATIO[1]:
        return None

    return chosen, rejected, source, reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data", help="DPO data folder")
    args = parser.parse_args()

    folder = HERE / args.dir
    source_file = folder / "generations.jsonl"

    if not source_file.exists():
        raise FileNotFoundError(f"File not found: {source_file}")

    items = [
        json.loads(line)
        for line in source_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(f"Loaded prompts: {len(items)}")

    pairs = []
    skipped = Counter()

    for item in items:
        task = item["task"]

        if not item.get("answers"):
            skipped[task] += 1
            continue

        if task in ("harmful", "cheating", "unknown"):
            if task not in ("unknown",) and task not in FIXED:
                skipped[task] += 1
                continue
            pair = refusal_pair(item)
        else:
            pair = normal_pair(item)

        if pair is None:
            skipped[task] += 1
            continue

        chosen, rejected, source, reasons = pair
        pairs.append({
            "id": item["id"],
            "task": task,
            "source": source,
            "rejected_why": reasons,
            "prompt": item["prompt"],
            "chosen": chosen,
            "rejected": rejected,
        })

    if len(pairs) < 2:
        print(f"Not enough pairs: {len(pairs)}")
        return

    rng.shuffle(pairs)
    n_val = max(1, round(len(pairs) * VAL_SHARE))

    for filename, rows in (
        ("val.jsonl", pairs[:n_val]),
        ("train.jsonl", pairs[n_val:]),
    ):
        (folder / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    print(f"Train: {len(pairs) - n_val} | Validation: {n_val}")
    print(f"Skipped: {sum(skipped.values())}")
    print(f"{'Task':<12} {'Pairs':>7} {'Model':>7} {'Reference':>10} {'Skipped':>8}")

    for task in sorted({item["task"] for item in items}):
        rows = [p for p in pairs if p["task"] == task]
        print(
            f"{task:<12} {len(rows):>7} "
            f"{sum(p['source'] == 'model' for p in rows):>7} "
            f"{sum(p['source'] != 'model' for p in rows):>10} "
            f"{skipped[task]:>8}"
        )

    shorter = sum(len(p["chosen"]) < len(p["rejected"]) for p in pairs)
    print(f"Chosen shorter than rejected: {shorter}/{len(pairs)}")
    print(f"Saved: {folder / 'train.jsonl'}")
    print(f"Saved: {folder / 'val.jsonl'}")

    print("\nExamples:")
    for task in sorted({p["task"] for p in pairs}):
        row = next(p for p in pairs if p["task"] == task)
        print(f"\n[{task}, {row['source']}] {row['prompt']}")
        print(f"  + {row['chosen'][:150]}")
        print(f"  - {row['rejected'][:150]}")


if __name__ == "__main__":
    main()
