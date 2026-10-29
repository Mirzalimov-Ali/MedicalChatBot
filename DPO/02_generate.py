"""
DPO 2-step: SFT modeldan har bir promptga bir nechta javob olish.
Natija: DPO/data/generations.jsonl

Run:
    python DPO/generate.py
    python DPO/generate.py --adapter sft-lora --limit 4
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
ROOT = HERE.parent
ADAPTERS = ROOT / "ui" / "adapters"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="SmolLM2-360M")
    p.add_argument("--adapter", default="sft-lora", help="DPO boshlanadigan SFT adapter")
    p.add_argument("--n", type=int, default=8, help="Har bir promptga nechta javob")
    p.add_argument("--out", default="data", help="DPO ichidagi natija papkasi")
    p.add_argument("--batch", type=int, default=4, help="Bir vaqtda nechta prompt")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    prompts = [
        json.loads(line)
        for line in (HERE / "data" / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prompts = prompts[:args.limit]

    cuda = torch.cuda.is_available()
    print(f"Promptlar: {len(prompts)} | Qurilma: {'CUDA' if cuda else 'CPU'}")
    print(f"Base model yuklanmoqda: {args.base}")

    tok = AutoTokenizer.from_pretrained(ROOT / args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        ROOT / args.base,
        torch_dtype=torch.float16 if cuda else torch.float32
    )

    print(f"SFT adapter yuklanmoqda: {args.adapter}")
    model = PeftModel.from_pretrained(
        model, ADAPTERS / args.adapter
    ).merge_and_unload().eval()

    if cuda:
        model.cuda()

    stop_ids = [tok.eos_token_id]

    out_path = HERE / args.out / "generations.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start, done = time.time(), 0

    print("Javoblar generatsiyasi boshlandi...")

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(prompts), args.batch):
            chunk = prompts[i:i + args.batch]
            texts = [
                f"User: {c['prompt']}\nAssistant:"
                for c in chunk
            ]

            inp = tok(texts, return_tensors="pt", padding=True).to(model.device)

            print("Input token ID min/max:", inp["input_ids"].min().item(), inp["input_ids"].max().item())
            print("Tokenizer vocab size:", len(tok))
            print("EOS:", tok.eos_token_id, "| PAD:", tok.pad_token_id)

            with torch.no_grad():
                test_out = model(**inp)
                print("Logits finite:", torch.isfinite(test_out.logits[:, -1, :]).all().item())
                print("Logits min/max:", test_out.logits[:, -1, :].min().item(), test_out.logits[:, -1, :].max().item())
                
                out = model.generate(
                    **inp,
                    do_sample=False,
                    num_beams=args.n,
                    num_return_sequences=args.n,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id
                )

            new = out[:, inp["input_ids"].shape[1]:]

            for j, c in enumerate(chunk):
                answers = []
                for k in range(args.n):
                    ids = new[j * args.n + k].tolist()
                    finished = any(t in stop_ids for t in ids)
                    answers.append({
                        "text": tok.decode(ids, skip_special_tokens=True).strip(),
                        "finished": finished
                    })
                f.write(json.dumps({**c, "answers": answers}, ensure_ascii=False) + "\n")

            done += len(chunk)
            elapsed = time.time() - start
            print(
                f"{done}/{len(prompts)} prompt | "
                f"O'tdi: {elapsed / 60:.1f} daq | "
                f"Qoldi: ~{elapsed / done * (len(prompts) - done) / 60:.1f} daq",
                flush=True
            )

    print("Saqlandi:", out_path)


if __name__ == "__main__":
    main()