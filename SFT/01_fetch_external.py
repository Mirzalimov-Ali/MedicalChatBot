import re
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
OUTPUT = "global_medical.txt"

MAX_EXAMPLES = 5000
MIN_QUESTION_CHARS = 20
MIN_ANSWER_CHARS = 40
MAX_CHARS = 2500


def clean(text):
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def looks_usable(question, answer):
    if not MIN_QUESTION_CHARS <= len(question) <= MAX_CHARS:
        return False

    if not MIN_ANSWER_CHARS <= len(answer) <= MAX_CHARS:
        return False

    if question.lower() == answer.lower():
        return False

    if "http://" in answer.lower() or "https://" in answer.lower():
        return False

    return True


def main():
    print("Loading OpenMed/MedDialog...")

    dataset = load_dataset(
        "OpenMed/MedDialog",
        split="train",
        streaming=True,
    )

    written = 0
    checked = 0

    with OUTPUT.open("w", encoding="utf-8") as file:
        file.write("# Source: OpenMed/MedDialog\n")
        file.write("# English medical conversations\n")
        file.write("# Answers are not medically verified.\n\n")

        for row in dataset:
            checked += 1

            # Inspect the dataset schema if these fields are missing.
            question = clean(row.get("patient_message", ""))
            answer = clean(row.get("doctor_response", ""))

            if not looks_usable(question, answer):
                continue

            file.write(f"Q: {question}\n")
            file.write(f"A: {answer}\n\n")

            written += 1

            if written % 500 == 0:
                print(f"Saved: {written}")

            if written >= MAX_EXAMPLES:
                break

    print("\nFinished!")
    print(f"Rows checked: {checked}")
    print(f"Examples saved: {written}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
