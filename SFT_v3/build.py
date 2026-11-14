"""SFT v3 dataset: v1 sources + sft_v3/new_samples.txt -> sft_v3/data/train.jsonl, val.jsonl. """

import sys
from pathlib import Path

HERE = Path(__file__).parent
SFT_DIR = HERE.parent / "SFT"

sys.path.insert(0, str(SFT_DIR))

import build_sft


if __name__ == "__main__":
    out = HERE / "data"
    out.mkdir(exist_ok=True)

    # Original source directory
    build_sft.HERE = SFT_DIR

    # V3 sources
    build_sft.SOURCES = [
        "seeds.txt",
        "external/global_medical.txt",
        HERE / "new_samples.txt",
    ]

    # Output directory
    original_write = Path.write_text

    def write_to_v3(path, *args, **kwargs):
        if path.name in ("train.jsonl", "val.jsonl"):
            path = out / path.name
        return original_write(path, *args, **kwargs)

    Path.write_text = write_to_v3

    try:
        build_sft.main()
    finally:
        Path.write_text = original_write
