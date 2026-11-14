
"""
RLHF step-1
Build: Prepare chosen/rejected pairs for Reward Model.

Run:
    python rlhf/build_rm_data.py
"""

import json
import random
import sys
import importlib.util

from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DPO_DIR = ROOT / "DPO"
SOURCE = DPO_DIR / "data" / "generations.jsonl"
LABEL_PATH = DPO_DIR / "03_label.py"

OUTPUT = HERE / "data"


# ============================================================
# IMPORT DPO/03_label.py
# ============================================================

if not LABEL_PATH.exists():
    raise FileNotFoundError(
        f"file label not found: {LABEL_PATH}"
    )

spec = importlib.util.spec_from_file_location(
    "label",
    LABEL_PATH
)

if spec is None or spec.loader is None:
    raise ImportError(
        f"failed to load: {LABEL_PATH}"
    )

label = importlib.util.module_from_spec(spec)
sys.modules["label"] = label
spec.loader.exec_module(label)


# ============================================================
# SETTINGS
# ============================================================

MARGIN = 0.5
MAX_PER_PROMPT = 8
VAL_SHARE = 0.1

REFUSALS = ("harmful", "cheating", "unknown")

rng = random.Random(42)


# ============================================================
# MEDICAL: USE EXISTING PAIRS
# FACTUAL: BUILD PAIRS USING ORIGINAL LOGIC
# ============================================================

def prompt_pairs(item):

    task = item.get("task", "")

    # --------------------------------------------------------
    # MEDICAL: already has chosen/rejected
    # --------------------------------------------------------

    if task == "medical":

        prompt = item.get("prompt", "").strip()
        chosen = item.get("chosen", "").strip()
        rejected = item.get("rejected", "").strip()

        if not prompt or not chosen or not rejected:
            return []

        return [{
            "id": item["id"],
            "task": task,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }]

    # --------------------------------------------------------
    # OTHER TASKS: original generation-based logic
    # --------------------------------------------------------

    if "answers" not in item or "reference" not in item:
        return []

    answers = {}

    for a in item["answers"]:
        if a.get("text"):
            answers.setdefault(
                a["text"],
                a.get("finished", False)
            )

    answers.setdefault(item["reference"], True)

    scored = [
        (label.rule_score(item, text, finished), text)
        for text, finished in answers.items()
    ]

    pairs = []

    for sc, c in scored:
        for sr, r in scored:

            if sc - sr < MARGIN:
                continue

            if task not in REFUSALS:

                ratio = len(c) / max(1, len(r))

                if not (
                    label.LENGTH_RATIO[0]
                    <= ratio
                    <= label.LENGTH_RATIO[1]
                ):
                    continue

                if task not in ("arithmetic", "translate"):
                    if label.f1(c, r) >= 0.85:
                        continue

            pairs.append({
                "id": item["id"],
                "task": task,
                "prompt": item["prompt"],
                "chosen": c,
                "rejected": r,
                "margin": round(sc - sr, 3),
            })

    rng.shuffle(pairs)

    return pairs[:MAX_PER_PROMPT]


# ============================================================
# BALANCE
# ============================================================

def balance(pairs):
    """Balance shorter and longer chosen answers per task."""

    out = []

    for task in sorted({p["task"] for p in pairs}):

        mine = [
            p for p in pairs
            if p["task"] == task
        ]

        if task in REFUSALS or task == "medical":
            out += mine
            continue

        short = [
            p for p in mine
            if len(p["chosen"]) < len(p["rejected"])
        ]

        long_ = [
            p for p in mine
            if len(p["chosen"]) > len(p["rejected"])
        ]

        same = [
            p for p in mine
            if len(p["chosen"]) == len(p["rejected"])
        ]

        n = min(len(short), len(long_))

        out += rng.sample(short, n)
        out += rng.sample(long_, n)
        out += same

    rng.shuffle(out)

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("STEP 1 — REWARD MODEL DATA PREPARATION")
    print("=" * 70)

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source file not found: {SOURCE}"
        )

    items = [
        json.loads(line)
        for line in SOURCE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    print(f"Loaded source records: {len(items)}")

    # Count original tasks
    task_counts = Counter(
        item.get("task", "unknown")
        for item in items
    )

    print("\nOriginal tasks:")

    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")

    # --------------------------------------------------------
    # SPLIT BY PROMPT ID
    # --------------------------------------------------------

    ids = sorted(
        item["id"]
        for item in items
    )

    rng.shuffle(ids)

    val_count = round(len(ids) * VAL_SHARE)

    val_ids = set(ids[:val_count])

    # --------------------------------------------------------
    # BUILD PAIRS
    # --------------------------------------------------------

    raw = [
        pair
        for item in items
        for pair in prompt_pairs(item)
    ]

    print(f"\nTotal pairs before balancing: {len(raw)}")

    # --------------------------------------------------------
    # TRAIN / VALIDATION
    # --------------------------------------------------------

    split = {
        "train": balance([
            p for p in raw
            if p["id"] not in val_ids
        ]),

        "val": balance([
            p for p in raw
            if p["id"] in val_ids
        ]),
    }

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, rows in split.items():

        path = OUTPUT / f"{name}.jsonl"

        path.write_text(
            "".join(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
                for row in rows
            ),
            encoding="utf-8"
        )

        print(
            f"Saved: {path} ({len(rows)} pairs)"
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    rows = split["train"] + split["val"]

    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

    print(f"Total pairs before balancing: {len(raw)}")
    print(f"Total pairs after balancing: {len(rows)}")
    print(f"Train: {len(split['train'])}")
    print(f"Validation: {len(split['val'])}")
    print(f"Validation prompt IDs: {len(val_ids)}")

    # Task statistics
    by_task = defaultdict(list)

    for pair in rows:
        by_task[pair["task"]].append(pair)

    print(
        f"\n{'Task':<15}"
        f"{'Pairs':>10}"
        f"{'Chosen is shorter':>22}"
    )

    for task, mine in sorted(by_task.items()):

        short_count = sum(
            len(p["chosen"]) < len(p["rejected"])
            for p in mine
        )

        short_ratio = short_count / len(mine)

        print(
            f"{task:<15}"
            f"{len(mine):>10}"
            f"{short_ratio:>21.0%}"
        )

    # Chosen length statistics
    normal = [
        p for p in rows
        if p["task"] not in REFUSALS
        and len(p["chosen"]) != len(p["rejected"])
    ]

    short_ratio = sum(
        len(p["chosen"]) < len(p["rejected"])
        for p in normal
    ) / max(1, len(normal))

    print(
        "\nPercentage of shorter chosen answers "
        f"(excluding refusal tasks): {short_ratio:.0%}"
    )

    print(
        "\nNumber of pairs per prompt:",
        dict(
            sorted(
                Counter(
                    Counter(
                        p["id"] for p in rows
                    ).values()
                ).items()
            )
        )
    )

    print("\nData preparation completed.")


if __name__ == "__main__":
    main()