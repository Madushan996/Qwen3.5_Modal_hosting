# Gemma 4 E4B Chat — Modal.com Hosted

A full-stack chat application that runs Google's **Gemma 4 E4B** multimodal language model on a cloud GPU, accessible through a local web interface. The model is hosted serverlessly on [Modal.com](https://modal.com) using a T4 GPU, while a lightweight local proxy serves the UI and handles file uploads.

![Chat Interface](https://img.shields.io/badge/UI-Dark%20Chat%20Interface-7c3aed)
![Model](https://img.shields.io/badge/Model-Gemma%204%20E4B%20Q4__K__M-4285f4)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20T4%2016GB-76b900)
![Backend](https://img.shields.io/badge/Backend-Modal.com-orange)

---

## Features

- **Streaming responses** — tokens appear in real time via Server-Sent Events (SSE)
- **Thinking / reasoning** — collapsible reasoning block shows the model's thought process before its answer
- **File attachments** — upload images (vision), PDFs, code files, and documents; content is injected into the conversation
- **32K token context** — long conversations without hitting context limits
- **Session history** — previous chats saved in browser localStorage with a sidebar
- **Persistent model storage** — model files downloaded once to a Modal Volume, never re-downloaded
- **Auto cold-start** — container stays warm for 5 minutes after last request; spins up automatically on demand

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Your Machine                                               │
│                                                             │
│  Browser  ──►  server.py (FastAPI, port 8000)              │
│                    │                                        │
│                    ├── /api/upload  (file processing)       │
│                    └── /api/chat   (SSE proxy)              │
│                              │                              │
└──────────────────────────────┼──────────────────────────────┘
                               │  HTTPS  (SSE stream)
┌──────────────────────────────▼──────────────────────────────┐
│  Modal.com (Serverless GPU Cloud)                           │
│                                                             │
│  GemmaService  (T4 GPU container)                           │
│    ├── /chat    POST  →  streaming token generation         │
│    └── /health  GET   →  status check                       │
│                                                             │
│  Modal Volume  "gemma-models"                               │
│    ├── gemma-4-E4B-it-Q4_K_M.gguf   (~2.5 GB)             │
│    └── mmproj-BF16.gguf             (~1.0 GB)              │
└─────────────────────────────────────────────────────────────┘
```

### Key components

| File | Purpose |
|------|---------|
| `modal_app.py` | Modal backend — builds the CUDA Docker image, downloads the model, serves the chat and health endpoints |
| `server.py` | Local FastAPI proxy — serves the web UI, handles file uploads, forwards chat requests to Modal |
| `static/index.html` | Single-file frontend — dark chat UI with streaming, thinking blocks, file attachments, and session history |
| `requirements.txt` | Local Python dependencies |
| `start.bat` | Windows convenience launcher |
| `.env.example` | Template for Modal endpoint URLs |

---

## Model Details

| Property | Value |
|----------|-------|
| Model | Gemma 4 E4B Instruct (Gemma 3n architecture) |
| Quantization | Q4_K_M (4-bit, ~2.5 GB) |
| Vision encoder | mmproj-BF16.gguf (~1.0 GB, SigLIP-based) |
| Inference engine | llama-cpp-python (CUDA build) |
| Context window | 32,768 tokens |
| Source | [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) |

The model is compiled with `GGML_CUDA=ON` inside a `nvidia/cuda:12.2.0-devel-ubuntu22.04` Docker image so all layers are offloaded to GPU (`n_gpu_layers=-1`).

---

## Prerequisites

- [Python 3.10+](https://python.org)
- [Modal account](https://modal.com) (free tier works)
- [Modal CLI](https://modal.com/docs/guide/installation) — `pip install modal` then `modal setup`

---

## Setup & Deployment

### 1. Clone the repository

```bash
git clone https://github.com/Madushan996/Gemma-e4B-with-Modal.com-hosting.git
cd Gemma-e4B-with-Modal.com-hosting
```

### 2. Deploy the Modal backend

```bash
modal deploy modal_app.py
```

This will:
- Build a CUDA-enabled Docker image with llama-cpp-python (takes ~12 minutes on first run, cached afterwards)
- Create a persistent Modal Volume named `gemma-models`
- Print two endpoint URLs when complete

```
✓ Created web endpoint for GemmaService.health => https://YOUR-WORKSPACE--gemma-4-e4b-chat-...
✓ Created web endpoint for GemmaService.chat  => https://YOUR-WORKSPACE--gemma-4-e4b-chat-...
```

> **First cold start** after deployment will download the model files (~3.5 GB total) to the Modal Volume. This takes a few minutes and only happens once.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and paste in your two endpoint URLs:

```env
MODAL_CHAT_URL=https://YOUR-WORKSPACE--gemma-4-e4b-chat-gemmaservice-chat.modal.run
MODAL_HEALTH_URL=https://YOUR-WORKSPACE--gemma-4-e4b-chat-gemmaservice-health.modal.run
```

### 4. Start the local server

**Windows:**
```bat
start.bat
```

**macOS / Linux:**
```bash
pip install -r requirements.txt
python server.py
```

Open **http://localhost:8000** in your browser.

---

## File Attachments

The chat supports attaching files via the paperclip button:

| File type | How it's handled |
|-----------|-----------------|
| Images (PNG, JPG, WebP, GIF…) | Encoded as base64 and sent to the model as `image_url` content (vision) |
| PDFs | Text extracted with `pypdf` and injected as context |
| Text, code, markdown, CSV, JSON… | Content read directly and injected as context |

> Vision (image understanding) requires `Gemma4ChatHandler` from llama-cpp-python. The app falls back to text-only mode gracefully if the handler is not available in the installed version.

---

## Thinking / Reasoning

Gemma 4 E4B supports a thinking mode where the model reasons step-by-step before giving its final answer. When active, the UI shows:

- A pulsing dot and live "Thinking…" block while the model is reasoning
- A collapsible **Reasoning** section (click to expand/collapse) once the response arrives

Thinking tokens are delimited by `<think>…</think>` in the raw model output and are parsed client-side.

---

## How the CUDA Image is Built

The Modal image build was non-trivial due to the CUDA development container's quirks:

```python
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "gcc", "g++", "cmake", "ninja-build")
    .run_commands(
        "pip install httpx fastapi 'uvicorn[standard]'",
        # The devel image sets CC=clang but clang isn't installed; override with gcc
        # Also create the versioned libcuda stub that the linker requires at build time
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
```

**Why the complexity?**
- `nvidia/cuda:12.2.0-devel` bakes in `CC=clang -pthread` but doesn't install clang → overridden with `CC=gcc CXX=g++`
- The CUDA stub library is `libcuda.so` (unversioned) but the linker requires `libcuda.so.1` for transitive dependencies → symlink created
- `-rpath-link` is needed (not just `-L`) for resolving transitive shared library dependencies at link time

---

## Cost

Modal charges only for actual GPU compute time:

- **T4 GPU**: ~$0.000583 / second (~$2.10 / hour)
- **Idle**: $0 — containers scale to zero automatically
- **Storage**: Modal Volume storage is cheap (~$0.05 / GB / month)

A typical conversation turn of 1–2 seconds of generation costs under $0.002.

---

## Local Development

To modify the frontend without redeploying:

```bash
python server.py  # serves with hot-reload
```

Edit `static/index.html` and refresh the browser — no restart needed.

To redeploy backend changes:

```bash
modal deploy modal_app.py
```

The Modal image is fully cached after the first build. Code-only changes deploy in under 10 seconds.

---

## License

This project is released under the MIT License.

The Gemma model weights are subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
