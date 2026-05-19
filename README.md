# Qwen Chat — Modal.com Hosted

A full-stack chat application that runs Alibaba's **Qwen 3.5 4B** vision-language model on a cloud GPU, accessible through a local web interface. The model is hosted serverlessly on [Modal.com](https://modal.com) using an **A10G GPU** via HuggingFace `transformers`, while a lightweight local proxy serves the UI and handles file uploads.

![Model](https://img.shields.io/badge/Model-Qwen%203.5%204B-6c5ce7)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20A10G%2024GB-76b900)
![Backend](https://img.shields.io/badge/Backend-Modal.com-orange)
![Vision](https://img.shields.io/badge/Vision-Enabled-22c55e)

---

## Features

- **Native vision** — send images and the model analyzes them natively (full multimodal, not OCR)
- **Streaming responses** — tokens appear in real time via Server-Sent Events (SSE)
- **OpenAI-compatible API** — drop-in replacement for the OpenAI API on `http://localhost:8000/v1`
- **Custom system prompt** — editable in the sidebar, persisted in `localStorage`
- **File attachments** — images, PDFs, code, and text documents (images analyzed by model)
- **Image preview in chat** — sent images appear inline in your message bubble
- **Session history** — previous chats saved in browser `localStorage` with sidebar navigation
- **Persistent model cache** — model weights downloaded once to a Modal Volume, never re-downloaded
- **Dark sidebar UI** — clean, modern interface inspired by leading AI chat apps

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Your Machine                                               │
│                                                             │
│  Browser  ──►  server.py (FastAPI, port 8000)              │
│                    │                                        │
│                    ├── /api/upload  (file processing)       │
│                    ├── /api/chat   (SSE proxy → Modal)      │
│                    └── /api/status (health check)           │
│                              │                              │
└──────────────────────────────┼──────────────────────────────┘
                               │  HTTPS  (SSE stream)
┌──────────────────────────────▼──────────────────────────────┐
│  Modal.com (Serverless GPU Cloud)                           │
│                                                             │
│  GemmaService  (A10G GPU container)                         │
│    ├── /chat    POST  →  streaming token generation         │
│    └── /health  GET   →  status + vision capability check   │
│                                                             │
│  Modal Volume  "gemma-models-hf"                            │
│    └── HuggingFace model cache (auto-managed)               │
└─────────────────────────────────────────────────────────────┘
```

### Key files

| File | Purpose |
|------|---------|
| `modal_app.py` | Modal backend — builds the container image, loads the model with 4-bit quantization, serves chat and health endpoints |
| `server.py` | Local FastAPI proxy — serves the web UI, handles file uploads, forwards chat requests to Modal |
| `static/index.html` | Single-file frontend — dark sidebar chat UI with streaming, system prompt, vision, thinking blocks, and session history |
| `requirements.txt` | Local Python dependencies |
| `start.bat` | Windows convenience launcher |
| `.env.example` | Template for Modal endpoint URLs |

---

## Model Details

| Property | Value |
|----------|-------|
| Model | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Quantization | NF4 4-bit via `bitsandbytes` |
| Inference engine | HuggingFace `transformers` (latest from main) |
| Vision | Native multimodal via `AutoProcessor` + `AutoModelForImageTextToText` |
| GPU | NVIDIA A10G (24 GB VRAM) |
| Context length | Up to 262,144 tokens natively |
| Languages | 201 languages supported |

---

## Prerequisites

- [Python 3.10+](https://python.org)
- [Modal account](https://modal.com) (free tier works for testing)
- [Modal CLI](https://modal.com/docs/guide/installation)
- A [HuggingFace account](https://huggingface.co) — Qwen 3.5 is **not gated**, no approval needed

---

## Setup & Deployment

### 1. Clone the repository

```bash
git clone https://github.com/Madushan996/Qwen3.5_Modal_hosting.git
cd Qwen3.5_Modal_hosting
```

### 2. Install local dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Modal

```bash
pip install modal
modal setup   # opens browser to authenticate
```

### 4. Add your HuggingFace token to Modal

Create a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then:

```bash
modal secret create huggingface HF_TOKEN=hf_your_token_here
```

> The secret must be named exactly `huggingface` — that's what `modal_app.py` references.
> Qwen 3.5 is open-access so any valid HF token works.

### 5. Deploy the Modal backend

```bash
modal deploy modal_app.py
```

This will:
- Install `git` in the container, then build a Debian-based Docker image with `torch`, `transformers` (from main), `bitsandbytes`, and friends — ~3–5 minutes, fully cached afterwards
- Create a persistent Modal Volume named `gemma-models-hf`
- Print two endpoint URLs when complete:

```
✓ Created web endpoint for GemmaService.health => https://YOUR-WORKSPACE--gemma-4-e4b-chat-gemmaservice-health.modal.run
✓ Created web endpoint for GemmaService.chat  => https://YOUR-WORKSPACE--gemma-4-e4b-chat-gemmaservice-chat.modal.run
```

> **First cold start** will download the Qwen 3.5 4B model weights from HuggingFace (~8 GB) into the Modal Volume. This takes a few minutes and only happens once — subsequent starts load from the cached volume (~30–60 s warm-up).

### 6. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and paste your two endpoint URLs:

```env
MODAL_CHAT_URL=https://YOUR-WORKSPACE--gemma-4-e4b-chat-gemmaservice-chat.modal.run
MODAL_HEALTH_URL=https://YOUR-WORKSPACE--gemma-4-e4b-chat-gemmaservice-health.modal.run
```

### 7. Start the local server

**Windows:**
```bat
start.bat
```

**macOS / Linux:**
```bash
python server.py
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## Using the Chat

### Sending messages

- Press **Enter** to send, **Shift+Enter** for a new line
- Click the **paperclip** button to attach files
- You can send **images or documents without typing any text** — just attach and press Send

### Vision

Qwen 3.5 4B is a native vision-language model — it sees images directly (pixel-level understanding, not OCR). Attach any image and ask about it. The **Vision enabled** badge in the sidebar footer confirms it's active.

### System prompt

Click **System Prompt** in the sidebar to expand the editor. Your prompt is applied to every new conversation and saved automatically to `localStorage`.

```
You are a concise coding assistant. Always reply in Python.
```
```
You are a JSON annotation tool. Return only valid JSON, no prose.
```

Leave it blank to use the default Qwen persona.

### File attachments

| File type | How it's handled |
|-----------|-----------------|
| Images (PNG, JPG, WebP, GIF, BMP) | Sent as base64 to the model — Qwen sees the image natively |
| PDFs | Text extracted with `pypdf` and injected as context (up to 40,000 chars) |
| Text, code, Markdown, CSV, JSON… | Content read directly and injected as context |

---

## GPU & Cost

| Resource | Detail |
|----------|--------|
| GPU | NVIDIA A10G (24 GB VRAM) |
| Approximate cost | ~$0.00032 / second (~$1.17 / hour) |
| Idle cost | $0 — scales to zero automatically |
| Warm window | Container stays warm for 5 minutes after last request |
| Volume storage | ~$0.05 / GB / month |

A typical 1–2 second generation turn costs under $0.001.

> To reduce costs, change `gpu="A10G"` to `gpu="T4"` in `modal_app.py`.

---

## Local Development

Frontend changes — no redeploy needed:

```bash
python server.py
```

Edit `static/index.html` and refresh the browser.

Backend changes require a redeploy:

```bash
modal deploy modal_app.py
```

After the first build, the Modal image is fully cached — code-only changes deploy in under 10 seconds.

---

## Batch Image Annotation

`batch_annotate.py` lets you annotate thousands of images automatically using the API. It reads a folder of images, sends them concurrently to `/v1/chat/completions`, and saves one JSON record per image to a `.jsonl` file.

```bash
# Make sure server.py is running first
python batch_annotate.py --input ./images
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Folder of images to annotate |
| `--output` | `annotations.jsonl` | Output file (JSONL) |
| `--prompt` | Built-in schema | System prompt defining what the model returns |
| `--concurrency` | `5` | Number of parallel requests |
| `--retries` | `3` | Retry attempts per image on failure |

---

## OpenAI-Compatible API

`server.py` exposes a fully OpenAI-compatible API at `http://localhost:8000/v1`.

### curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

### Python — `openai` library

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "Explain neural networks in one paragraph."}],
)
print(response.choices[0].message.content)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `401 Unauthorized` on model download | Check `HF_TOKEN` in your `huggingface` Modal secret is valid |
| Vision badge shows grey | Health check couldn't reach Modal — check `MODAL_HEALTH_URL` in `.env` |
| Cold start takes a long time | Normal on first request — model loads from the volume (~30–60 seconds) |
| `303 See Other` in server logs | Already fixed — `server.py` uses `follow_redirects=True` |
| Image build fails (git not found) | The `apt_install("git")` step handles this automatically |

---

## License

This project is released under the MIT License.

The Qwen 3.5 model weights are released under the [Apache 2.0 License](https://huggingface.co/Qwen/Qwen3.5-4B).
