"""
Local chat UI for Personal Health Assistant (SmolLM2-360M).

Run from the project root:
    python ui/server.py

Then open:
    http://127.0.0.1:8080

Available models:
- Base SmolLM2-360M
- SFT + LoRA
- SFT + QLoRA

The LoRA and QLoRA adapters are loaded from ui/adapters/
"""

import threading
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_PATH = ROOT / "SmolLM2-360M"
ADAPTERS_PATH = HERE / "adapters"

MODELS = [
    {
        "id": "base",
        "label": "Base SmolLM2-360M",
        "group": "Base model",
        "desc": "Pretrained model without SFT adapter",
        "pretrained": True,
    },
    {
        "id": "sft-lora",
        "label": "SFT + LoRA",
        "group": "Fine-tuned models",
        "desc": "SmolLM2-360M with the SFT LoRA adapter",
        "pretrained": False,
    },
    {
        "id": "sft-qlora",
        "label": "SFT + QLoRA",
        "group": "Fine-tuned models",
        "desc": "SmolLM2-360M with the SFT QLoRA adapter",
        "pretrained": False,
    },
]
BY_ID = {item["id"]: item for item in MODELS}

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
DTYPE = torch.float16 if CUDA_AVAILABLE else torch.float32

# Keep the same User/Assistant text format used by the training scripts.
# Do not use apply_chat_template unless the training data is changed to a chat template.
MAX_INPUT_TOKENS = 1024
DEFAULT_MAX_NEW_TOKENS = 200
MAX_NEW_TOKENS = 400

generation_lock = threading.Lock()
app = FastAPI(title="Personal Health Assistant")


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    model: str = "base"
    messages: list[Message]
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=MAX_NEW_TOKENS)


class StopOnEvent(StoppingCriteria):
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


def adapter_available(model_id: str) -> bool:
    adapter_dir = ADAPTERS_PATH / model_id
    return (
        (adapter_dir / "adapter_config.json").is_file()
        and (adapter_dir / "adapter_model.safetensors").is_file()
    )


def model_is_available(model_id: str) -> bool:
    if model_id not in BY_ID or not (BASE_PATH / "config.json").is_file():
        return False
    if BY_ID[model_id]["pretrained"]:
        return (BASE_PATH / "model.safetensors").is_file()
    return adapter_available(model_id)


class ModelManager:
    def __init__(self):
        self.tokenizer = None
        self.base_model = None
        self.peft_model = None
        self.loaded_adapters = set()
        self._load_base()

    def _load_base(self):
        if not (BASE_PATH / "config.json").is_file():
            raise FileNotFoundError(f"Base model config not found: {BASE_PATH}")
        if not (BASE_PATH / "model.safetensors").is_file():
            raise FileNotFoundError(f"Base model weights not found: {BASE_PATH / 'model.safetensors'}")

        print(f"Loading tokenizer: {BASE_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(BASE_PATH),
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading base model on {DEVICE}: {BASE_PATH}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            str(BASE_PATH),
            torch_dtype=DTYPE,
            local_files_only=True,
        ).to(DEVICE).eval()

        print(
            f"Base model ready | device={DEVICE} | dtype={DTYPE} | "
            f"cuda={CUDA_AVAILABLE}"
        )

    def get_model(self, model_id: str):
        if model_id == "base":
            # Once PEFT has wrapped the base model, use the wrapper with adapters
            # disabled so base generation cannot accidentally apply a LoRA adapter.
            if self.peft_model is not None:
                return self.peft_model, False
            return self.base_model, False

        if not adapter_available(model_id):
            raise FileNotFoundError(
                f"Adapter files not found for '{model_id}' in {ADAPTERS_PATH / model_id}"
            )

        from peft import PeftModel

        # PEFT adapter names cannot contain dots; current IDs have no dots,
        # but normalizing keeps this safe if an ID is added later.
        adapter_name = model_id.replace(".", "_")
        if self.peft_model is None:
            self.peft_model = PeftModel.from_pretrained(
                self.base_model,
                str(ADAPTERS_PATH / model_id),
                adapter_name=adapter_name,
                is_trainable=False,
            ).eval()
            self.loaded_adapters.add(adapter_name)
        elif adapter_name not in self.peft_model.peft_config:
            self.peft_model.load_adapter(
                str(ADAPTERS_PATH / model_id),
                adapter_name=adapter_name,
                is_trainable=False,
            )
            self.loaded_adapters.add(adapter_name)

        self.peft_model.set_adapter(adapter_name)
        return self.peft_model, True


manager = None
try:
    manager = ModelManager()
except Exception as exc:
    # Keep the server able to start so /api/models can show the problem clearly.
    print(f"WARNING: model could not be loaded at startup: {exc!r}")


@app.get("/")
def index():
    index_path = HERE / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail=f"UI file not found: {index_path}")
    return FileResponse(index_path)


@app.get("/api/models")
def list_models():
    return [
        {
            **item,
            "available": model_is_available(item["id"]),
        }
        for item in MODELS
    ]


def build_prompt(messages: list[Message]) -> str:
    # This SFT setup was trained as one question followed by one answer.
    # Use the latest non-empty user message rather than replaying the whole chat,
    # which can cause a small model to repeat earlier answers.
    user_messages = [
        message.content.strip()
        for message in messages
        if message.role.lower() == "user" and message.content.strip()
    ]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No non-empty user message was provided.")

    return f"User: {user_messages[-1]}\nAssistant:"


@app.post("/api/generate")
def generate(req: GenerateRequest):
    global manager

    if req.model not in BY_ID:
        raise HTTPException(status_code=400, detail="Unknown model ID.")
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty.")
    if not model_is_available(req.model):
        raise HTTPException(
            status_code=409,
            detail=f"Model '{req.model}' is unavailable. Check the base model and adapter files.",
        )
    if manager is None:
        try:
            manager = ModelManager()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not load model: {exc}") from exc

    prompt = build_prompt(req.messages)
    tokenizer = manager.tokenizer
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(DEVICE)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120.0,
    )
    stop_event = threading.Event()

    generation_kwargs = {
        **inputs,
        "max_new_tokens": req.max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
        "stopping_criteria": StoppingCriteriaList([StopOnEvent(stop_event)]),
    }

    def run_generation():
        try:
            with generation_lock, torch.inference_mode():
                model, uses_adapter = manager.get_model(req.model)
                if req.model == "base" and manager.peft_model is not None:
                    with manager.peft_model.disable_adapter():
                        model.generate(**generation_kwargs)
                else:
                    model.generate(**generation_kwargs)
        except Exception as exc:
            print(f"Generation error for {req.model}: {exc!r}")
            try:
                streamer.end()
            except Exception:
                pass

    threading.Thread(target=run_generation, daemon=True).start()

    def stream_response():
        try:
            yield from streamer
        finally:
            stop_event.set()

    return StreamingResponse(
        stream_response(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8080, reload=False)
