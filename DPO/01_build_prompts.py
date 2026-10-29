import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "SFT"))
import build_sft

QUOTA = {
    "harmful": None, "cheating": None, "unknown": None,
    "exam_prep": None, "chat": None, "list": None,
    "factual": None, "explain": 150, "translate": 100,
    "arithmetic": 60
}
N_NEW_ARITHMETIC = 100
rng = random.Random(7)


def main():
    print("[1/4] SFT v3 datasetlari yuklanmoqda...")
    rows = [
        json.loads(line)
        for name in ("train.jsonl", "val.jsonl")
        for line in (ROOT / "SFT_v3" / "data" / name)
        .read_text(encoding="utf-8").splitlines()
    ]

    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    print("[2/4] SFT promptlari tanlanmoqda...")
    prompts = []
    for task, quota in QUOTA.items():
        items = by_task.get(task, [])
        if quota is not None and len(items) > quota:
            items = rng.sample(items, quota)
        for r in items:
            q, a = r["messages"][0]["content"], r["messages"][1]["content"]
            p = {"task": task, "prompt": q, "reference": a}
            if task == "arithmetic":
                p["number"] = int(re.search(r"=\s*(\d+)", a).group(1))
            if task == "list":
                p["items"] = int(re.search(r"\d+", q).group())
            prompts.append(p)

    print(f"  SFT promptlari: {len(prompts)}")

    print("[3/4] Yangi arifmetika promptlari yaratilmoqda...")
    eval_items = json.loads(build_sft.EVAL_SET.read_text(encoding="utf-8"))
    eval_squeezed = [build_sft.squeeze(i["messages"][0]["content"]) for i in eval_items]
    seen = {build_sft.squeeze(p["prompt"]) for p in prompts}
    build_sft.rng = random.Random(7)
    new = 0

    while new < N_NEW_ARITHMETIC:
        key, uz_q, uz_a, en_q, en_a = build_sft.arithmetic_example()
        k = build_sft.squeeze(key)
        if k in seen or any(k in e for e in eval_squeezed) or any(k in s for s in seen):
            continue
        seen.add(k)
        q, a = (en_q, en_a) if rng.random() < 0.2 else (uz_q, uz_a)
        prompts.append({
            "task": "arithmetic", "prompt": q, "reference": a,
            "number": int(re.search(r"=\s*(\d+)", a).group(1))
        })
        new += 1

    print(f"  Yangi arifmetika: {new}")
    rng.shuffle(prompts)
    for i, p in enumerate(prompts):
        p["id"] = f"p{i:04d}"

    print("[4/4] Natija saqlanmoqda...")
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    (out / "prompts.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts),
        encoding="utf-8"
    )

    for task, n in Counter(p["task"] for p in prompts).most_common():
        print(f"  {task:<12} {n}")
    print(f"  Jami: {len(prompts)} -> {out / 'prompts.jsonl'}")


if __name__ == "__main__":
    main()