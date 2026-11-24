"""Evaluate the deployed model using the same evaluation dataset through any OpenAI-compatible API.

Usage:
    python hybrid/2_engine/eval_api.py --url http://127.0.0.1:7860/v1 --key sk-th-... --name sft-qlora-v3
    python hybrid/2_engine/eval_api.py --url http://127.0.0.1:8080/v1 --name sft-qlora-v3

Results are saved to eval/results/<name>.json in the same format as 01_run_eval.py.
02_summarise.py can include these results in its comparison table.
"""

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

EVAL = Path(__file__).parent.parent.parent / "eval"

spec = importlib.util.spec_from_file_location(
    "run_eval",
    EVAL / "01_run_eval.py"
)
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)

auto_check = run_eval.auto_check


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="OpenAI-compatible API base URL ending in /v1")
    p.add_argument("--key", default="none", help="API key for the Gateway; not required by the Engine")
    p.add_argument("--model", default="sft-qlora-v3", help="Model ID exposed by the API")
    p.add_argument("--name", required=True, help="Output filename: eval/results/<name>.json")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-new-tokens", type=int, default=100)
    args = p.parse_args()

    client = OpenAI(base_url=args.url, api_key=args.key, max_retries=5)
    items = json.loads((EVAL / "eval_set.json").read_text(encoding="utf-8"))[:args.limit]

    results, t0 = [], time.time()

    for i, item in enumerate(items, 1):
        user = item["messages"][0]

        # Use the same greedy decoding settings as 01_run_eval.py.
        r = client.chat.completions.create(
            model=args.model,
            messages=[user],
            max_tokens=args.max_new_tokens,
            temperature=0,
            extra_body={
                "repeat_penalty": 1.1,
                "repeat_last_n": 1024
            }
        )

        answer = (r.choices[0].message.content or "").strip()

        results.append({
            "id": item["id"],
            "task": item["task"],
            "messages": [
                user,
                {"role": "assistant", "content": answer}
            ],
            "reference": item["messages"][1]["content"],
            "auto_pass": auto_check(item, answer)
        })

        print(
            f"[{i}/{len(items)}] {item['id']}: "
            f"{'OK' if results[-1]['auto_pass'] else '--'}",
            flush=True
        )

    out_path = EVAL / "results" / f"{args.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    total = Counter(r["task"] for r in results)
    passed = Counter(r["task"] for r in results if r["auto_pass"])

    elapsed = (time.time() - t0) / len(results) if results else 0
    print(f"\n{args.name}: automatic evaluation ({elapsed:.1f} sec/question)")

    for task in total:
        print(f"  {task:<11} {passed[task]}/{total[task]}")

    print(f"  {'total':<11} {sum(passed.values())}/{len(results)}")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
