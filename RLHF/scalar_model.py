"""
Non-step. Scalar Model: Transformer + bitta son chiqaradigan head.
Reward Model javobga bitta baho beradi.

Bu alohida run qilinmaydi
"""

import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).parent.parent
ADAPTERS = ROOT / "ui" / "adapters"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class ScalarModel(torch.nn.Module):
    def __init__(self, backbone, hidden):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        ).last_hidden_state
        return self.head(h.float()).squeeze(-1)


def backbone_of(base, sft_adapter, device):
    causal = AutoModelForCausalLM.from_pretrained(
        ROOT / base,
        dtype=torch.float16
    ).to(device)

    if sft_adapter:
        causal = PeftModel.from_pretrained(
            causal,
            ADAPTERS / sft_adapter,
            torch_device=device
        ).merge_and_unload()

    return causal.model, causal.config.hidden_size

def no_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def build(base, sft_adapter, rank, device):
    """SFT model asosida Reward Model yaratish."""
    backbone, hidden = backbone_of(base, sft_adapter, device)

    backbone.gradient_checkpointing_enable()
    backbone.enable_input_require_grads()

    backbone = get_peft_model(
        backbone,
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.0,
            target_modules=TARGETS
        )
    )

    model = ScalarModel(backbone, hidden).to(device)

    torch.nn.init.normal_(model.head.weight, std=1 / (hidden + 1) ** 0.5)
    torch.nn.init.zeros_(model.head.bias)

    no_dropout(model)
    return model


def save(model, folder, meta):
    folder.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(folder)
    torch.save(model.head.state_dict(), folder / "head.pt")
    (folder / "meta.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8"
    )


def load(folder, device, trainable=False, dtype=torch.float32):
    """Saqlangan Reward Modelni yuklash."""
    meta = json.loads(
        (folder / "meta.json").read_text(encoding="utf-8")
    )

    backbone, hidden = backbone_of(
        meta["base"], meta["sft_adapter"], device
    )

    if trainable:
        backbone.gradient_checkpointing_enable()
        backbone.enable_input_require_grads()

    backbone = PeftModel.from_pretrained(
        backbone, folder, is_trainable=trainable
    )

    if not trainable:
        backbone = backbone.merge_and_unload().to(dtype)

    model = ScalarModel(backbone, hidden).to(device)

    model.head.load_state_dict(
        torch.load(folder / "head.pt", map_location=device)
    )
    model.head.requires_grad_(trainable)

    no_dropout(model)
    return model.train(trainable), meta


def pad_batch(seqs, pad_id, device):
    """Token ketma-ketliklarini bir xil uzunlikka keltirish."""
    width = max(len(s) for s in seqs)

    ids = torch.tensor(
        [s + [pad_id] * (width - len(s)) for s in seqs],
        device=device
    )
    mask = torch.tensor(
        [[1] * len(s) + [0] * (width - len(s)) for s in seqs],
        device=device
    )

    return ids, mask


def last_scores(model, seqs, pad_id, device):
    """Har bir javobning yakuniy Reward ballini olish."""
    ids, mask = pad_batch(seqs, pad_id, device)
    out = model(ids, mask)

    return out[
        torch.arange(len(seqs), device=device),
        mask.sum(-1) - 1
    ]
