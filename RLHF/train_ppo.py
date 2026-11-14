import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util

import scalar_model as scalar_model

HERE = Path(__file__).parent
ROOT = HERE.parent
ADAPTERS = ROOT / "ui" / "adapters"

sys.path.insert(0, str(ROOT / "DPO"))

ROOT = Path(__file__).resolve().parent.parent
LABEL_PATH = ROOT / "DPO" / "03_label.py"

spec = importlib.util.spec_from_file_location("label", LABEL_PATH)
label = importlib.util.module_from_spec(spec)
sys.modules["label"] = label
spec.loader.exec_module(label)

rule_score = label.rule_score


def load_policy(base, adapter, device):
    model = AutoModelForCausalLM.from_pretrained(
        ROOT / base, dtype=torch.float16
    ).to(device)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = PeftModel.from_pretrained(
        model, ADAPTERS / adapter, is_trainable=True
    )

    scalar_model.no_dropout(model)
    return model


def load_reference(base, adapter, device, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        ROOT / base, dtype=dtype
    )
    model = PeftModel.from_pretrained(
        model, ADAPTERS / adapter
    ).merge_and_unload()

    return model.to(device).eval().requires_grad_(False)


def response_part(x, prompts, responses, width):
    return torch.stack([
        F.pad(
            x[i, len(p) - 1:len(p) - 1 + len(r)],
            (0, width - len(r))
        )
        for i, (p, r) in enumerate(zip(prompts, responses))
    ])


def logprobs(model, prompts, responses, width, temperature, pad_id, device):
    ids, mask = scalar_model.pad_batch(
        [p + r for p, r in zip(prompts, responses)],
        pad_id, device
    )

    logits = model(
        input_ids=ids,
        attention_mask=mask,
        use_cache=False
    ).logits[:, :-1].float() / temperature

    token_logp = -F.cross_entropy(
        logits.transpose(1, 2),
        ids[:, 1:],
        reduction="none"
    )

    return response_part(token_logp, prompts, responses, width)


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum()


@torch.no_grad()
def rollout(policy, tok, items, args, stop_ids, device):
    policy.eval()
    torch.cuda.synchronize()

    texts = [
        f"User: {it['prompt']}\nAssistant:"
        for it in items
    ]

    prompts = [tok(t)["input_ids"] for t in texts]

    inp = tok(
        texts,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    out = policy.generate(
        **inp,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_k=0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=False
    )

    responses, finished = [], []

    start = inp["input_ids"].shape[1]

    for row in out[:, start:].tolist():
        end = next(
            (k for k, t in enumerate(row) if t in stop_ids),
            None
        )

        response = row[:end] if end is not None else row
        responses.append(response)
        finished.append(end is not None)

    # return prompts, responses, finished
    del inp, out
    torch.cuda.empty_cache()

    return prompts, responses, finished


def chunks(seq, size):
    return [
        seq[i:i + size]
        for i in range(0, len(seq), size)
    ]


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--base", default="SmolLM2-360M")
    p.add_argument("--sft-adapter", default="sft-qlora")
    p.add_argument("--reward", default="reward-sft-qlora")
    p.add_argument("--name", default="sft-qlora-rlhf")
    p.add_argument("--prompts", default="DPO/data/prompts.jsonl")

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--mini-batch", type=int, default=1)
    p.add_argument("--ppo-epochs", type=int, default=2)

    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=0.2)

    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--no-stop-penalty", type=float, default=1.0)

    p.add_argument("--algo", choices=["ppo", "rloo"], default="rloo")
    p.add_argument("--group-size", type=int, default=2)

    p.add_argument("--rollouts", default="ppo_rollouts.jsonl")
    p.add_argument("--max-steps", type=int)

    args = p.parse_args()

    if args.algo != "rloo":
        raise SystemExit("Bu GPU sozlamasida --algo rloo ishlating.")

    if args.group_size < 2:
        raise SystemExit("RLOO uchun group-size kamida 2 bo'lishi kerak.")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA mavjud emas. GPU kerak.")

    random.seed(42)
    torch.manual_seed(42)

    device = "cuda:0"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    tok = AutoTokenizer.from_pretrained(ROOT / args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"

    pad = tok.pad_token_id
    stop_ids = [tok.eos_token_id]

    gate = json.loads(
        (HERE / "models" / args.reward / "meta.json").read_text(
            encoding="utf-8"
        )
    ).get("length_gate")

    if gate and not gate["ok"]:
        raise SystemExit(
            f"Reward model length gate'dan o'tmagan: {gate}"
        )

    print("Modellar yuklanmoqda...", flush=True)

    policy = load_policy(
        args.base, args.sft_adapter, device
    )
    policy.print_trainable_parameters()

    reference = load_reference(
        args.base,
        args.sft_adapter,
        device,
        torch.float16
    )

    reward, meta = scalar_model.load(
        HERE / "models" / args.reward,
        device,
        dtype=torch.float16
    )

    if len(AutoTokenizer.from_pretrained(ROOT / meta["base"])) != len(tok):
        raise SystemExit("Reward model va policy tokenizerlari har xil.")

    print(
        f"Reward model: {meta['base']} | "
        f"val acc: {meta['val_acc']:.2f}",
        flush=True
    )

    length_coef = meta.get("length_coef", 0.0)

    opt_policy = torch.optim.AdamW(
        [q for q in policy.parameters() if q.requires_grad],
        lr=args.lr,
        weight_decay=0.0
    )

    k = args.group_size

    items = [
        json.loads(line)
        for line in (ROOT / args.prompts).read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    order = [
        it
        for _ in range(args.epochs)
        for it in random.sample(items, len(items))
    ]

    batches = chunks(order, args.batch_size)[:args.max_steps]
    total = len(batches)

    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)

    rollouts_file = (
        logs / args.rollouts
    ).open("w", encoding="utf-8")

    log = []
    start = time.time()

    print(
        f"{len(items)} prompt, {total} qadam.",
        flush=True
    )

    for step, batch in enumerate(batches, 1):

        batch = [
            it
            for it in batch
            for _ in range(k)
        ]

        # 1. Rollout
        prompts, responses, finished = rollout(
            policy, tok, batch, args, stop_ids, device
        )

        width = max(len(r) for r in responses)

        mask = torch.tensor([
            [1.0] * len(r) + [0.0] * (width - len(r))
            for r in responses
        ])

        idx = chunks(
            list(range(len(batch))),
            args.mini_batch
        )

        def sub(seq, ids):
            return [seq[i] for i in ids]

        # 2. Eski log-probability va reference
        with torch.no_grad():
            old = torch.cat([
                logprobs(
                    policy,
                    sub(prompts, b),
                    sub(responses, b),
                    width,
                    args.temperature,
                    pad,
                    device
                ).cpu()
                for b in idx
            ])

            ref = torch.cat([
                logprobs(
                    reference,
                    sub(prompts, b),
                    sub(responses, b),
                    width,
                    args.temperature,
                    pad,
                    device
                ).float().cpu()
                for b in idx
            ])

            raw = torch.cat([
                scalar_model.last_scores(
                    reward,
                    [prompts[i] + responses[i] for i in b],
                    pad,
                    device
                ).float().cpu()
                for b in idx
            ])

        texts = [
            tok.decode(r, skip_special_tokens=True).strip()
            for r in responses
        ]

        raw = raw - length_coef * torch.tensor([
            float(len(t)) for t in texts
        ])

        score = (raw - meta["mean"]) / meta["std"]

        score = score - args.no_stop_penalty * (
            ~torch.tensor(finished)
        ).float()

        # 3. KL penalty
        kl = (old - ref) * mask

        # 4. RLOO advantage
        seq_reward = (
            score - args.kl_coef * kl.sum(-1)
        ).view(-1, k)

        seq_adv = (
            seq_reward
            - (seq_reward.sum(1, keepdim=True) - seq_reward) / (k - 1)
        ).view(-1)

        seq_adv = seq_adv / (seq_adv.std() + 1e-8)

        adv = seq_adv[:, None] * mask

        # 5. PPO update
        policy.train()
        stats = []

        for _ in range(args.ppo_epochs):
            order_ids = random.sample(
                range(len(batch)),
                len(batch)
            )

            for b in chunks(order_ids, args.mini_batch):

                m = mask[b].to(device)

                new = logprobs(
                    policy,
                    sub(prompts, b),
                    sub(responses, b),
                    width,
                    args.temperature,
                    pad,
                    device
                )

                ratio = torch.exp(
                    new - old[b].to(device)
                )

                a = adv[b].to(device)

                pg_loss = masked_mean(
                    torch.max(
                        -a * ratio,
                        -a * ratio.clamp(
                            1 - args.clip,
                            1 + args.clip
                        )
                    ),
                    m
                )

                # pg_loss.backward()

                # torch.nn.utils.clip_grad_norm_(
                #     [
                #         q for q in policy.parameters()
                #         if q.requires_grad
                #     ],
                #     1.0
                # )

                # opt_policy.step()
                # opt_policy.zero_grad()

                pg_loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [q for q in policy.parameters() if q.requires_grad],
                    1.0
                )

                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"Gradient NaN/Inf: {grad_norm}")

                opt_policy.step()
                opt_policy.zero_grad()

                with torch.no_grad():
                    d = new - old[b].to(device)

                    stats.append({
                        "pg_loss": pg_loss.item(),
                        "clipfrac": masked_mean(
                            ((ratio - 1).abs() > args.clip).float(),
                            m
                        ).item(),
                        "approx_kl": (
                            0.5 * masked_mean(d ** 2, m)
                        ).item()
                    })

        # Log
        rules = [
            rule_score(it, t, f)
            for it, t, f in zip(batch, texts, finished)
        ]

        for it, t, f, rm, rs in zip(
            batch, texts, finished, raw.tolist(), rules
        ):
            rollouts_file.write(
                json.dumps(
                    {
                        "step": step,
                        "id": it["id"],
                        "task": it["task"],
                        "prompt": it["prompt"],
                        "answer": t,
                        "finished": f,
                        "rm": round(rm, 3),
                        "rule": round(rs, 3)
                    },
                    ensure_ascii=False
                ) + "\n"
            )

        rollouts_file.flush()

        s = {
            "step": step,
            "rm": raw.mean().item(),
            "rule": sum(rules) / len(rules),
            "kl": kl.sum(-1).mean().item(),
            "length": sum(map(len, responses)) / len(responses),
            "finished": sum(finished) / len(finished),
            **{
                key: sum(x[key] for x in stats) / len(stats)
                for key in stats[0]
            }
        }

        log.append({
            key: round(value, 4)
            for key, value in s.items()
        })

        elapsed = time.time() - start

        print(
            f"qadam {step}/{total} | "
            f"RM {s['rm']:+.2f} | "
            f"qoida {s['rule']:+.2f} | "
            f"KL {s['kl']:.3f} | "
            f"uzunlik {s['length']:.0f} | "
            f"to'xtadi {s['finished']:.0%} | "
            f"pg {s['pg_loss']:+.3f} | "
            f"clip {s['clipfrac']:.2f} | "
            f"qoldi ~{elapsed / step * (total - step) / 60:.0f} daq",
            flush=True
        )

        if step % 10 == 0 or step == total:
            print(
                f"Misol [{batch[0]['task']}] "
                f"{batch[0]['prompt'][:80]}\n"
                f"-> {texts[0][:200]}",
                flush=True
            )

        if not args.max_steps and (
            step % 20 == 0 or step == total
        ):
            policy.save_pretrained(ADAPTERS / args.name)

            (ADAPTERS / args.name / "training_log.json").write_text(
                json.dumps(log, indent=1),
                encoding="utf-8"
            )

    rollouts_file.close()

    if args.max_steps:
        print("Sinov tugadi, adapter saqlanmadi.")
        return

    first = log[:max(1, total // 10)]
    last = log[-max(1, total // 10):]

    for key in ("rm", "rule", "kl", "length"):
        print(
            f"{key:<7} boshida "
            f"{sum(x[key] for x in first) / len(first):+.3f} "
            f"-> oxirida "
            f"{sum(x[key] for x in last) / len(last):+.3f}"
        )

    print(
        f"Tayyor: {ADAPTERS / args.name} | "
        f"javoblar: {logs / args.rollouts} | "
        f"{(time.time() - start) / 60:.1f} daqiqa"
    )


if __name__ == "__main__":
    main()
