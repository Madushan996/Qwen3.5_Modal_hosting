"""
Qwen3.5 4B Instruct Chat — Modal Backend (HuggingFace transformers)
Text-only model. Deploy:  modal deploy modal_app.py
"""

from __future__ import annotations

import json
import threading

import modal
from fastapi import Request
from fastapi.responses import StreamingResponse

# ── Config ─────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3.5-4B-Instruct"
HF_CACHE = "/models/hf"

# ── App & persistent volume ────────────────────────────────────────────────

app    = modal.App("gemma-4-e4b-chat")
volume = modal.Volume.from_name("gemma-models-hf", create_if_missing=True)

# ── Container image ────────────────────────────────────────────────────────

hf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51.0",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece",
        "protobuf",
        "huggingface_hub[hf_transfer]",
        "fastapi",
        "uvicorn[standard]",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": HF_CACHE,
    })
)

# ── Service class ──────────────────────────────────────────────────────────

@app.cls(
    image=hf_image,
    gpu="A10G",
    volumes={HF_CACHE: volume},
    timeout=600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("huggingface")],
)
@modal.concurrent(max_inputs=1)
class GemmaService:

    @modal.enter()
    def setup(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        volume.reload()

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print(f"[setup] Loading tokenizer for {MODEL_ID} …")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)

        print(f"[setup] Loading model {MODEL_ID} with NF4 quantization …")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quant_cfg,
            device_map="auto",
            cache_dir=HF_CACHE,
        )
        self.model.eval()
        volume.commit()
        print("[setup] Model ready — container is hot.")

    # ── Chat endpoint (SSE streaming) ──────────────────────────────────────

    @modal.fastapi_endpoint(method="POST")
    async def chat(self, request: Request) -> StreamingResponse:
        import torch
        from transformers import TextIteratorStreamer

        body        = await request.json()
        messages    = body.get("messages", [])
        temperature = float(body.get("temperature", 0.7))
        max_tokens  = int(body.get("max_tokens", 2048))

        # Flatten multimodal content to text — Qwen3.5-4B is text-only
        hf_messages: list[dict] = []
        for msg in messages:
            role    = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )
            if content:
                hf_messages.append({"role": role, "content": content})

        text = self.tokenizer.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # disable chain-of-thought for plain chat
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.9,
            repetition_penalty=1.1,
            streamer=streamer,
        )

        def _generate() -> None:
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        threading.Thread(target=_generate, daemon=True).start()

        def token_stream():
            try:
                for token in streamer:
                    yield f"data: {json.dumps({'content': token})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"

        return StreamingResponse(
            token_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            },
        )

    # ── Health endpoint ────────────────────────────────────────────────────

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        return {
            "status": "ok",
            "model": MODEL_ID,
            "loaded": hasattr(self, "model"),
            "vision": False,
        }
