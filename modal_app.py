"""
Gemma 4 E4B Chat — Modal Backend (HuggingFace transformers)
Natively multimodal: text + image inputs via AutoProcessor.
Deploy:  modal deploy modal_app.py

Notes:
  • Gemma models on HuggingFace require accepting the license on the model page.
    If the download fails with a 401/403, create a HuggingFace access token,
    add it as a Modal secret named "huggingface" with key HF_TOKEN,
    then ensure the secrets=[...] line below is uncommented.
  • GPU: A10G (24 GB) fits the 4B model in NF4 with room to spare.
    Swap to "A100" for faster throughput, or "T4" if cost is the priority
    (T4 has 16 GB — may be tight with longer contexts).
  • Requires transformers>=4.51.0 for Gemma4ForConditionalGeneration support.
"""

from __future__ import annotations

import base64
import io
import json
import threading

import modal
from fastapi import Request
from fastapi.responses import StreamingResponse

# ── Config ─────────────────────────────────────────────────────────────────

MODEL_ID = "google/gemma-4-E4B-it"   # instruction-tuned; base model has no chat template
HF_CACHE = "/models/hf"

# ── App & persistent volume ────────────────────────────────────────────────

app    = modal.App("gemma-4-e4b-chat")
volume = modal.Volume.from_name("gemma-models-hf", create_if_missing=True)

# ── Container image ────────────────────────────────────────────────────────

hf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.52.0",   # >=4.52 fixes Gemma4Processor.apply_chat_template bug
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
        "pillow",
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

# ── Helpers ────────────────────────────────────────────────────────────────

def _build_gemma_prompt(messages: list[dict]) -> str:
    """Manual Gemma chat format used when the tokenizer has no chat template."""
    parts = []
    for msg in messages:
        role    = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        if role == "system":
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
        elif role == "user":
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
        elif role == "assistant":
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>\n")
    parts.append("<start_of_turn>model\n")
    return "".join(parts)


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
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Gemma4ForConditionalGeneration,
        )

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
        self.model = Gemma4ForConditionalGeneration.from_pretrained(
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
        from PIL import Image
        from transformers import TextIteratorStreamer

        body        = await request.json()
        messages    = body.get("messages", [])
        temperature = float(body.get("temperature", 0.7))
        max_tokens  = int(body.get("max_tokens", 2048))

        # Convert OpenAI-style multimodal messages to HuggingFace format.
        # Images are extracted into a separate PIL list; messages use {"type":"image"}
        # placeholders so the tokenizer's chat template can insert the right tokens.
        pil_images: list[Image.Image] = []
        hf_messages: list[dict] = []

        for msg in messages:
            role    = msg["role"]
            content = msg.get("content", "")

            if isinstance(content, str):
                hf_messages.append({"role": role, "content": content})
            else:
                img_parts  = []
                text_parts = []
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            _, b64 = url.split(",", 1)
                            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                            if max(img.size) > 1024:
                                img.thumbnail((1024, 1024), Image.LANCZOS)
                            pil_images.append(img)
                            # Embed PIL directly so apply_chat_template(tokenize=True)
                            # can handle token count alignment internally (canonical API)
                            img_parts.append({"type": "image", "image": img})
                # Images must precede text in each turn for Gemma 4
                hf_messages.append({"role": role, "content": img_parts + text_parts})

        # Canonical Gemma 4 API: apply_chat_template with tokenize=True handles
        # image token expansion (1 placeholder → N soft tokens) and pixel_values
        # alignment in one coordinated step.
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
            "vision": True,
        }
