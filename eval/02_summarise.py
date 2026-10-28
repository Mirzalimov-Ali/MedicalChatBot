
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

MODEL_LABELS = {
    "base": "Base SmolLM2",
    "sft-lora": "SFT + LoRA",
    "sft-qlora": "SFT + QLoRA",
    "sft-lora-360M": "SFT + LoRA 360M",
    "sft-qlora-360M": "SFT + QLoRA 360M",
}

TASK_LABELS = {
    "factual": "Factual",
    "explain": "Explanation",
    "list": "List",
    "translate": "Translation",
    "arithmetic": "Arithmetic",
    "unknown": "Unknown / Abstention",
    "harmful": "Harmful request refusal",
    "cheating": "Cheating refusal",
    "health_advice": "Health advice",
}


def main():
    if not RESULTS_DIR.exists():
        print(f"Results folder not found: {RESULTS_DIR}")
        return

    columns = []

    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name in {"eval_set.json"}:
            continue

        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Skipping {path.name}: {exc}")
            continue

        if not isinstance(records, list):
            print(f"Skipping {path.name}: expected a JSON list")
            continue

        scores = {}

        for record in records:
            if not isinstance(record, dict):
                continue

            task = record.get("task")
            if not task:
                continue

            passed, total = scores.get(task, (0, 0))
            passed += int(bool(record.get("auto_pass", False)))
            total += 1
            scores[task] = (passed, total)

        if scores:
            name = path.stem
            label = MODEL_LABELS.get(name, name)
            columns.append((name, label, scores))

    if not columns:
        print(f"No evaluation reports found in {RESULTS_DIR}")
        return

    # Use the task categories actually present in the reports.
    tasks = sorted(
        {task for _, _, scores in columns for task in scores}
    )

    header = ["Task"] + [label for _, label, _ in columns]
    rows = []

    for task in tasks:
        label = TASK_LABELS.get(task, task.replace("_", " ").title())
        row = [label]

        for _, _, scores in columns:
            if task in scores:
                passed, total = scores[task]
                row.append(f"{passed}/{total}")
            else:
                row.append("—")

        rows.append(row)

    total_row = ["Total"]

    for _, _, scores in columns:
        passed = sum(p for p, _ in scores.values())
        total = sum(t for _, t in scores.values())
        total_row.append(f"{passed}/{total}")

    rows.append(total_row)

    csv_path = RESULTS_DIR / "per_task_scores.csv"
    md_path = RESULTS_DIR / "per_task_scores.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerows([header] + rows)

    lines = [
        "# Evaluation Summary",
        "",
        "Automatic evaluation results from available JSON reports.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]

    for row in rows:
        if row[0] == "Total":
            row = [f"**{cell}**" for cell in row]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "Note: automatic checks do not establish medical accuracy or safety.",
        "Review model answers manually before drawing conclusions.",
    ])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved: {md_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
