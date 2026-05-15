"""
Gemma 4 E4B Chat — Modal Backend
Hosts gemma-4-E4B-it-Q4_K_M.gguf on a T4 GPU via llama-cpp-python.
Deploy:  modal deploy modal_app.py
"""

from __future__ import annotations

import json
import os

import modal
from fastapi import Request
from fastapi.responses import StreamingResponse

# ── App & persistent volume ────────────────────────────────────────────────

app = modal.App("gemma-4-e4b-chat")
volume = modal.Volume.from_name("gemma-models", create_if_missing=True)

MODELS_DIR = "/models"
MODEL_FILENAME  = "gemma-4-E4B-it-Q4_K_M.gguf"
MODEL_PATH      = f"{MODELS_DIR}/{MODEL_FILENAME}"
MODEL_URL       = (
    "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/"
    "gemma-4-E4B-it-Q4_K_M.gguf"
)
MMPROJ_FILENAME = "mmproj-BF16.gguf"
MMPROJ_PATH     = f"{MODELS_DIR}/{MMPROJ_FILENAME}"
MMPROJ_URL      = (
    "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/"
    "mmproj-BF16.gguf"
)

# ── Container image (CUDA 12.2 dev + llama-cpp-python built with GPU) ──────

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential", "gcc", "g++", "cmake", "ninja-build")
    .run_commands(
        "pip install httpx fastapi 'uvicorn[standard]'",
        # The CUDA dev image has libcuda.so (stub) but not the versioned libcuda.so.1
        # that the linker requires when building CLI tools. Create the symlink.
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1",
        (
            "CC=gcc CXX=g++ "
            "LDFLAGS='-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs' "
            "CMAKE_ARGS='-DGGML_CUDA=ON "
            "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs' "
            "pip install llama-cpp-python --no-cache-dir"
        ),
    )
)

# ── Service class ──────────────────────────────────────────────────────────

@app.cls(
    image=cuda_image,
    gpu="T4",
    volumes={MODELS_DIR: volume},
    timeout=600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=1)
class GemmaService:

    @modal.enter()
    def setup(self) -> None:
        import httpx

        volume.reload()

        os.makedirs(MODELS_DIR, exist_ok=True)

        def _download(url: str, dest: str, label: str) -> None:
            tmp = dest + ".tmp"
            with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done  = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(8 * 1024 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = done / total * 100
                            print(f"  [{label}] {pct:5.1f}%  "
                                  f"{done//1_000_000}MB / {total//1_000_000}MB")
            os.rename(tmp, dest)

        # ── Download model if not cached ───────────────────────────────────
        if not os.path.exists(MODEL_PATH):
            print("[setup] Downloading main model from HuggingFace…")
            _download(MODEL_URL, MODEL_PATH, "model")
            volume.commit()
            print("[setup] Main model committed to volume.")
        else:
            print(f"[setup] Model found at {MODEL_PATH}")

        # ── Download mmproj (vision encoder) if not cached ────────────────
        if not os.path.exists(MMPROJ_PATH):
            print("[setup] Downloading mmproj (vision encoder) from HuggingFace…")
            _download(MMPROJ_URL, MMPROJ_PATH, "mmproj")
            volume.commit()
            print("[setup] mmproj committed to volume.")
        else:
            print(f"[setup] mmproj found at {MMPROJ_PATH}")

        # ── Load model (with vision if handler is available) ──────────────
        from llama_cpp import Llama

        chat_handler = None
        for handler_name in ("Gemma4ChatHandler", "Gemma3ChatHandler"):
            try:
                import importlib
                mod = importlib.import_module("llama_cpp.llama_chat_format")
                cls = getattr(mod, handler_name)
                chat_handler = cls(clip_model_path=MMPROJ_PATH)
                print(f"[setup] Vision enabled via {handler_name}")
                break
            except (ImportError, AttributeError):
                continue
        if chat_handler is None:
            print("[setup] No vision chat handler found — text-only mode")

        print("[setup] Loading model…")
        self.llm = Llama(
            model_path=MODEL_PATH,
            chat_handler=chat_handler,
            n_gpu_layers=-1,
            n_ctx=32768,
            n_batch=512,
            verbose=False,
        )
        print("[setup] Model ready — container is hot.")

    # ── Chat endpoint (SSE streaming) ──────────────────────────────────────

    @modal.fastapi_endpoint(method="POST")
    async def chat(self, request: Request) -> StreamingResponse:
        body = await request.json()
        messages    = body.get("messages", [])
        temperature = float(body.get("temperature", 0.7))
        max_tokens  = int(body.get("max_tokens", 4096))

        def token_stream():
            try:
                stream = self.llm.create_chat_completion(
                    messages=messages,
                    stream=True,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    repeat_penalty=1.1,
                )
                for chunk in stream:
                    choice = chunk["choices"][0]
                    content = choice.get("delta", {}).get("content")
                    if content:
                        yield f"data: {json.dumps({'content': content})}\n\n"
                    if choice.get("finish_reason") is not None:
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        return
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
            "model": MODEL_FILENAME,
            "loaded": hasattr(self, "llm"),
        }
