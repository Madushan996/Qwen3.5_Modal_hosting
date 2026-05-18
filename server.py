"""
Local proxy server — serves the web UI, forwards requests to Modal,
and exposes an OpenAI-compatible API on /v1/chat/completions.

Usage:  python server.py
        Then open http://localhost:8000

OpenAI-compatible API:
        Base URL : http://localhost:8000/v1
        Models   : GET  /v1/models
        Chat     : POST /v1/chat/completions  (stream=true or false)
"""

import base64
import io
import json
import os
import time
import uuid
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

load_dotenv()

MODAL_CHAT_URL   = os.getenv("MODAL_CHAT_URL", "").rstrip("/")
MODAL_HEALTH_URL = os.getenv("MODAL_HEALTH_URL", "").rstrip("/")
STATIC_DIR       = Path(__file__).parent / "static"

app = FastAPI(title="Gemma Chat")

# Allow cross-origin requests so other apps/frontends can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_TIMEOUT = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)


# ── Helpers ────────────────────────────────────────────────────────────────

def _modal_body(messages: list, temperature: float, max_tokens: int) -> dict:
    return {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}


async def _stream_from_modal(modal_body: dict):
    """Async generator that yields parsed Modal SSE events as dicts."""
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        async with client.stream("POST", MODAL_CHAT_URL, json=modal_body) as resp:
            resp.raise_for_status()
            buf = ""
            async for chunk in resp.aiter_bytes():
                buf += chunk.decode()
                lines = buf.split("\n")
                buf = lines.pop()
                for line in lines:
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if not raw:
                        continue
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError:
                        continue


# ── Web UI ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ── File upload ────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    data     = await file.read()
    filename = file.filename or "file"
    ct       = (file.content_type or "").lower()
    ext      = Path(filename).suffix.lower()

    if ct.startswith("image/") or ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
        mime = ct if ct.startswith("image/") else "image/jpeg"
        b64  = base64.b64encode(data).decode()
        return JSONResponse({"type": "image", "filename": filename,
                             "data_url": f"data:{mime};base64,{b64}"})

    if ct == "application/pdf" or ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            text   = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as exc:
            text = f"[Could not extract PDF text: {exc}]"
        return JSONResponse({"type": "document", "filename": filename,
                             "text": text[:40_000]})

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")
    return JSONResponse({"type": "document", "filename": filename,
                         "text": text[:40_000]})


# ── Chat proxy (SSE, used by the web UI) ──────────────────────────────────

@app.post("/api/chat")
async def chat_proxy(request: Request) -> StreamingResponse:
    if not MODAL_CHAT_URL:
        async def cfg_err():
            msg = ("MODAL_CHAT_URL is not set. "
                   "Deploy the backend with 'modal deploy modal_app.py', "
                   "then add the URL to your .env file.")
            yield f"data: {json.dumps({'error': msg, 'done': True})}\n\n"
        return StreamingResponse(cfg_err(), media_type="text/event-stream")

    body = await request.json()

    async def proxy_stream():
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            async with client.stream("POST", MODAL_CHAT_URL, json=body) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        proxy_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Status / health check ──────────────────────────────────────────────────

@app.get("/api/status")
async def status() -> JSONResponse:
    if not MODAL_CHAT_URL:
        return JSONResponse({"status": "unconfigured", "message": "Set MODAL_CHAT_URL in .env"})

    if MODAL_HEALTH_URL:
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                r = await client.get(MODAL_HEALTH_URL)
                return JSONResponse(r.json())
        except Exception as exc:
            return JSONResponse({"status": "error", "message": str(exc)})

    return JSONResponse({"status": "configured", "url": MODAL_CHAT_URL})


# ── OpenAI-compatible API ──────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "gemma-3-4b-it",
            "object": "model",
            "created": 1700000000,
            "owned_by": "google",
        }],
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not MODAL_CHAT_URL:
        return JSONResponse(
            {"error": {"message": "MODAL_CHAT_URL not configured", "type": "server_error"}},
            status_code=500,
        )

    body        = await request.json()
    messages    = body.get("messages", [])
    temperature = float(body.get("temperature", 0.7))
    max_tokens  = int(body.get("max_tokens", 2048))
    stream      = bool(body.get("stream", False))
    model       = body.get("model", "gemma-3-4b-it")

    cid     = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    mbody   = _modal_body(messages, temperature, max_tokens)

    # ── Streaming response ─────────────────────────────────────────────────
    if stream:
        async def openai_stream():
            # Opening chunk with role
            yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"

            async for evt in _stream_from_modal(mbody):
                if evt.get("error"):
                    err_msg = f"[Error] {evt['error']}"
                    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': err_msg}, 'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                if evt.get("content"):
                    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': evt['content']}, 'finish_reason': None}]})}\n\n"
                if evt.get("done"):
                    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

        return StreamingResponse(
            openai_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Non-streaming response ─────────────────────────────────────────────
    accumulated = ""
    async for evt in _stream_from_modal(mbody):
        if evt.get("content"):
            accumulated += evt["content"]
        if evt.get("done") or evt.get("error"):
            break

    return JSONResponse({
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": accumulated},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    })


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Gemma Chat — Local Server")
    print("=" * 55)
    if not MODAL_CHAT_URL:
        print("  ⚠  MODAL_CHAT_URL not set in .env")
        print("     Deploy first:  modal deploy modal_app.py")
        print("     Then update .env with the returned URL.")
    else:
        print(f"  Modal endpoint : {MODAL_CHAT_URL}")
    print(f"  Web UI         : http://localhost:8000")
    print(f"  OpenAI API     : http://localhost:8000/v1")
    print("=" * 55 + "\n")

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
