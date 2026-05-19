"""
Qwen3.5-4B Chat — Modal Backend (HuggingFace transformers)
Vision-language model. Deploy: modal deploy modal_app.py
"""

from __future__ import annotations

import json
import threading

import modal
from fastapi import Request
from fastapi.responses import StreamingResponse

# ── Config ─────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3.5-4B"
HF_CACHE = "/models/hf"

# ── App & persistent volume ────────────────────────────────────────────────

app    = modal.App("gemma-4-e4b-chat")
volume = modal.Volume.from_name("gemma-models-hf", create_if_missing=True)

# ── Container image ────────────────────────────────────────────────────────

hf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "transformers @ git+https://github.com/huggingface/transformers.git@main",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece",
        "protobuf",
        "huggingface_hub[hf_transfer]",
        "pillow",
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
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

        volume.reload()

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print(f"[setup] Loading processor for {MODEL_ID} …")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)

        print(f"[setup] Loading model {MODEL_ID} with NF4 quantization …")
        self.model = AutoModelForImageTextToText.from_pretrained(
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
        import base64
        import io
        import torch
        from PIL import Image
        from transformers import TextIteratorStreamer

        body        = await request.json()
        messages    = body.get("messages", [])
        temperature = float(body.get("temperature", 0.7))
        max_tokens  = int(body.get("max_tokens", 2048))

        hf_messages: list[dict] = []
        for msg in messages:
            role    = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                # System prompt stays as plain string
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if p.get("type") == "text"
                    )
                if content:
                    hf_messages.append({"role": role, "content": content})
            else:
                # User / assistant: build list content so images can be included
                if isinstance(content, list):
                    hf_content = []
                    for part in content:
                        ptype = part.get("type", "")
                        if ptype == "text":
                            hf_content.append({"type": "text", "text": part["text"]})
                        elif ptype == "image_url":
                            url = part["image_url"]["url"]
                            if url.startswith("data:image"):
                                _, b64data = url.split(",", 1)
                                img = Image.open(
                                    io.BytesIO(base64.b64decode(b64data))
                                ).convert("RGB")
                                hf_content.append({"type": "image", "image": img})
                            else:
                                hf_content.append({"type": "image", "url": url})
                    if hf_content:
                        hf_messages.append({"role": role, "content": hf_content})
                else:
                    if content:
                        hf_messages.append({"role": role, "content": content})

        # apply_chat_template with tokenization — processor handles image preprocessing
        try:
            inputs = self.processor.apply_chat_template(
                hf_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(self.model.device)
        except TypeError:
            inputs = self.processor.apply_chat_template(
                hf_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.8,
            top_k=20,
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
            "vision": True,
        }
