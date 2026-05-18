"""
Local proxy server — serves the web UI and forwards requests to Modal.
Usage:  python server.py
        Then open http://localhost:8000
"""

import base64
import io
import os
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

load_dotenv()

MODAL_CHAT_URL   = os.getenv("MODAL_CHAT_URL", "").rstrip("/")
MODAL_HEALTH_URL = os.getenv("MODAL_HEALTH_URL", "").rstrip("/")
STATIC_DIR       = Path(__file__).parent / "static"

app = FastAPI(title="Gemma Chat")


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

    # Images → base64 data URL (sent directly to the model)
    if ct.startswith("image/") or ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
        mime  = ct if ct.startswith("image/") else "image/jpeg"
        b64   = base64.b64encode(data).decode()
        return JSONResponse({"type": "image", "filename": filename,
                             "data_url": f"data:{mime};base64,{b64}"})

    # PDFs → extract text
    if ct == "application/pdf" or ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            text   = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as exc:
            text = f"[Could not extract PDF text: {exc}]"
        return JSONResponse({"type": "document", "filename": filename,
                             "text": text[:40_000]})

    # Everything else → decode as text
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")
    return JSONResponse({"type": "document", "filename": filename,
                         "text": text[:40_000]})


# ── Chat proxy (SSE streaming) ─────────────────────────────────────────────

@app.post("/api/chat")
async def chat_proxy(request: Request) -> StreamingResponse:
    if not MODAL_CHAT_URL:
        async def cfg_err():
            import json
            msg = (
                "MODAL_CHAT_URL is not set. "
                "Deploy the backend with 'modal deploy modal_app.py', "
                "then add the URL to your .env file."
            )
            yield f"data: {json.dumps({'error': msg, 'done': True})}\n\n"
        return StreamingResponse(cfg_err(), media_type="text/event-stream")

    body = await request.json()
    timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)

    async def proxy_stream():
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream(
                "POST", MODAL_CHAT_URL, json=body
            ) as resp:
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
    print("=" * 55 + "\n")

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
