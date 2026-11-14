import json
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
EVAL_SET = HERE.parent / "eval" / "eval_set.json"

SOURCES = [
    "external/global_medical.txt",
    "seeds.txt",
]

VAL_SHARE = 0.05
rng = random.Random(42)


def norm(text):
    text = text.lower().replace("‘", "'").replace("’", "'").replace("ʻ", "'").replace("ʼ", "'")
    return re.sub(r"[^\w']+", " ", text).strip()


def read_seeds(name):
    sections = {}
    section = None
    q = None
    a = None

    def close():
        nonlocal q, a
        if q is not None and a is not None:
            sections.setdefault(section or "medical_dialog", []).append(
                (q, "\n".join(a).strip())
            )
        q = a = None

    path = HERE / name

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()

        if line.startswith("## "):
            close()
            section = line[3:].strip()
            sections.setdefault(section, [])

        elif line.startswith("Q: "):
            close()
            q = line[3:]
            a = None

        elif q is not None and line.startswith("A: ") and a is None:
            a = [line[3:]]

        elif q is not None and a is not None:
            a.append(line)

    close()
    return sections


def check(task, q, a):
    if not q.strip() or not a.strip():
        return "bo'sh savol yoki javob"

    if task == "unknown":
        if not re.match(r"(Bilmayman|I don't know)\b", a, re.I):
            return "unknown javobi noto'g'ri"

    if task in ("harmful", "cheating"):
        if not re.search(r"olmayman|can't", a, re.I):
            return "rad javobi yo'q"

    return None


def main():
    eval_items = json.loads(
        EVAL_SET.read_text(encoding="utf-8")
    )

    eval_questions = {
        norm(item["messages"][0]["content"])
        for item in eval_items
    }

    examples = []

    for name in SOURCES:
        sections = read_seeds(name)

        for task, qa_list in sections.items():
            for q, a in qa_list:
                examples.append((task, Path(name).name, q, a))

    kept = []
    seen = set()
    problems = {}

    for task, source, q, a in examples:
        key = norm(q)

        if key in eval_questions:
            why = "test savoli bilan bir xil"

        elif key in seen:
            why = "takror savol"

        else:
            why = check(task, q, a)

        if why:
            problems.setdefault((source, why), []).append(q)
            continue

        seen.add(key)

        kept.append({
            "task": task,
            "source": source,
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })

    rng.shuffle(kept)

    n_val = round(len(kept) * VAL_SHARE)

    val = kept[:n_val]
    train = kept[n_val:]

    for name, rows in (
        ("val.jsonl", val),
        ("train.jsonl", train)
    ):
        path = HERE / name
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in rows
            ),
            encoding="utf-8"
        )

    print("\nTask types:")
    for task, count in Counter(r["task"] for r in kept).most_common():
        print(f"  {task:<25} {count}")

    print("\Sources:")
    for source, count in Counter(r["source"] for r in kept).most_common():
        print(f"  {source:<30} {count}")

    print(f"\Summary: {len(kept)}")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")

    if problems:
        print(f"\nError samples: {sum(map(len, problems.values()))}")

        for (source, why), questions in problems.items():
            print(f"  {len(questions):>4} | {source} | {why}")
            print(f"       Example: {questions[0]}")


if __name__ == "__main__":
    main()
