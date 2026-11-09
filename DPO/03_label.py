
"""
DPO: Build chosen/rejected preference pairs from generated answers.

Input:
    DPO/data/generations.jsonl

Output:
    DPO/data/train.jsonl
    DPO/data/val.jsonl

Run:
    python -m DPO.label
    python -m DPO.label --dir data
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent

VAL_SHARE = 0.15
MIN_SCORE_GAP = 0.02
MAX_SIMILARITY = 0.97
LENGTH_RATIO = (0.4, 2.5)

rng = random.Random(42)

REFUSAL_OK = re.compile(
    r"can't help|cannot help|unable to help|can't assist|cannot assist",
    re.I,
)

REFUSAL_ANY = re.compile(
    r"sorry|can't help|cannot help|unable to help|can't assist|cannot assist",
    re.I,
)

UNKNOWN_OK = re.compile(
    r"^(i don't know|i do not know)\b",
    re.I,
)

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


def acceptable_refusal(item, text, chosen):
    if item["task"] == "unknown":
        return (
            bool(UNKNOWN_OK.match(text))
            and not REFUSAL_ANY.search(text)
            and (
                len(words(text)) <= 5
                or f1(text, item["reference"]) >= 0.5
            )
        )

    return bool(REFUSAL_OK.search(text)) and (
        f1(text, chosen) >= 0.7
        or f1(text, item["reference"]) >= 0.7
    )


def refusal_pair(item):
    task = item["task"]

    chosen = (
        item["reference"]
        if task == "unknown"
        else FIXED[task]
    )

    bad = [
        a["text"].strip()
        for a in item["answers"]
        if a.get("text", "").strip()
        and not acceptable_refusal(item, a["text"], chosen)
    ]

    if not bad:
        return None

    rejected = bad[0]

    return (
        chosen,
        rejected,
        "reference" if task == "unknown" else "fixed",
        ["incorrect refusal behavior"],
    )


def task_score(item, text):
    task = item["task"]

    if task == "list":
        found = re.findall(
            r"^\s*\d+[.)]\s+(.+)$",
            text,
            flags=re.M,
        )

        valid = (
            len(found) == item.get("items", 0)
            and all(len(words(x)) >= 2 for x in found)
        )

        return (
            1.0
            if valid and not REFUSAL_ANY.search(text)
            else 0.0
        )

    if task == "translate":
        reference = set(words(item["reference"]))
        answer = words(text)

        if (
            reference
            and reference <= set(answer)
            and len(answer) <= 1.5 * len(reference) + 1
        ):
            return 1.0

        return 0.6 * f1(text, item["reference"])

    # Factual and medical answers:
    # Score how closely the answer matches the reference.
    return f1(text, item["reference"])


def penalty(item, answer):
    text = answer["text"]
    reasons = []

    if REFUSAL_ANY.search(text) and item["task"] not in (
        "harmful",
        "cheating",
        "unknown",
    ):
        reasons.append("unnecessary refusal")

    if not answer.get("finished", False):
        reasons.append("unfinished")

    sentences = [
        s.strip()
        for s in re.split(r"[.\n]", text)
        if len(s.strip()) > 10
    ]

    if len(sentences) != len(set(sentences)):
        reasons.append("repetition")

    return reasons


def score(item, answer):
    text = answer["text"].strip()

    if not text:
        return -1.0, ["empty"]

    reasons = penalty(item, answer)

    value = task_score(item, text) - 0.5 * len(reasons)

    return value, reasons


def rule_score(item, text, finished):
    task = item["task"]

    if task in ("harmful", "cheating", "unknown"):
        chosen = (
            item["reference"]
            if task == "unknown"
            else FIXED[task]
        )

        return (
            1.0
            if text and acceptable_refusal(item, text, chosen)
            else 0.0
        )

    return score(
        item,
        {"text": text, "finished": finished},
    )[0]


def normal_pair(item):
    scored = []
    seen = set()

    for answer in item.get("answers", []):
        text = answer.get("text", "").strip()

        if not text:
            continue

        # Remove exact duplicate generations.
        normalized = " ".join(text.lower().split())

        if normalized in seen:
            continue

        seen.add(normalized)

        value, reasons = score(item, {
            "text": text,
            "finished": answer.get("finished", False),
        })

        scored.append((value, reasons, text))

    if len(scored) < 2:
        return None

    # Best generated answer.
    best_score, best_reasons, chosen = max(
        scored,
        key=lambda x: (x[0], -len(x[2])),
    )

    # Worst generated answer.
    candidates = sorted(scored, key=lambda x: x[0])

    selected = None

    for rejected_score, rejected_reasons, rejected in candidates:
        if rejected == chosen:
            continue

        # There must be a meaningful score difference.
        if best_score - rejected_score < MIN_SCORE_GAP:
            continue

        similarity = f1(chosen, rejected)

        # Skip almost identical answers.
        if similarity >= MAX_SIMILARITY:
            continue

        ratio = len(chosen) / max(1, len(rejected))

        if not LENGTH_RATIO[0] <= ratio <= LENGTH_RATIO[1]:
            continue

        selected = (
            rejected,
            rejected_reasons,
            similarity,
        )
        break

    if selected is None:
        return None

    rejected, rejected_reasons, similarity = selected

    reasons = list(rejected_reasons)

    if best_score > rejected_score:
        reasons.append("lower reference similarity")

    if similarity < 0.5:
        reasons.append("low similarity to chosen")

    return chosen, rejected, "model", reasons


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir",
        default="data",
        help="DPO data folder",
    )

    args = parser.parse_args()

    folder = HERE / args.dir
    source_file = folder / "generations.jsonl"

    if not source_file.exists():
        raise FileNotFoundError(
            f"File not found: {source_file}"
        )

    items = [
        json.loads(line)
        for line in source_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    print(f"Loaded prompts: {len(items)}")

    pairs = []
    skipped = Counter()

    for item in items:
        task = item.get("task", "unknown")

        if not item.get("answers"):
            skipped[task] += 1
            continue

        if task in ("harmful", "cheating", "unknown"):
            pair = refusal_pair(item)

        else:
            if not item.get("reference"):
                skipped[task] += 1
                continue

            pair = normal_pair(item)

        if pair is None:
            skipped[task] += 1
            continue

        chosen, rejected, source, reasons = pair

        pairs.append({
            "id": item.get("id"),
            "task": task,
            "source": source,
            "rejected_why": reasons,
            "prompt": item["prompt"],
            "chosen": chosen,
            "rejected": rejected,
        })

    if len(pairs) < 2:
        print(f"Not enough pairs: {len(pairs)}")
        print("Try generating more diverse answers.")
        return

    rng.shuffle(pairs)

    n_val = max(
        1,
        round(len(pairs) * VAL_SHARE),
    )

    val_rows = pairs[:n_val]
    train_rows = pairs[n_val:]

    for filename, rows in (
        ("train.jsonl", train_rows),
        ("val.jsonl", val_rows),
    ):
        (folder / filename).write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    print(f"\nTrain: {len(train_rows)} | Validation: {len(val_rows)}")
    print(f"Skipped: {sum(skipped.values())}")

    print(
        f"\n{'Task':<15} {'Pairs':>7} "
        f"{'Model':>7} {'Skipped':>8}"
    )

    for task in sorted({item.get("task", "unknown") for item in items}):
        rows = [p for p in pairs if p["task"] == task]

        print(
            f"{task:<15} {len(rows):>7} "
            f"{sum(p['source'] == 'model' for p in rows):>7} "
            f"{skipped[task]:>8}"
        )

    shorter = sum(
        len(p["chosen"]) < len(p["rejected"])
        for p in pairs
    )

    print(
        f"\nChosen shorter than rejected: "
        f"{shorter}/{len(pairs)}"
    )

    print(f"Saved: {folder / 'train.jsonl'}")
    print(f"Saved: {folder / 'val.jsonl'}")

    print("\nExamples:")

    for task in sorted({p["task"] for p in pairs}):
        row = next(p for p in pairs if p["task"] == task)

        print(f"\n[{task}] {row['prompt']}")
        print(f"  + {row['chosen'][:150]}")
        print(f"  - {row['rejected'][:150]}")


if __name__ == "__main__":
    main()
