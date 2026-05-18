# Gemma Chat — Modal.com Hosted

A full-stack chat application that runs Google's **Gemma 3 4B Instruct** multimodal language model on a cloud GPU, accessible through a local web interface. The model is hosted serverlessly on [Modal.com](https://modal.com) using an **A10G GPU** via HuggingFace `transformers`, while a lightweight local proxy serves the UI and handles file uploads.

![Model](https://img.shields.io/badge/Model-Gemma%203%204B%20Instruct-4285f4)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20A10G%2024GB-76b900)
![Backend](https://img.shields.io/badge/Backend-Modal.com-orange)
![Vision](https://img.shields.io/badge/Vision-Enabled-22c55e)

---

## Features

- **Native vision (image analysis)** — send images directly to the model; full multimodal support via HuggingFace `AutoProcessor`
- **Streaming responses** — tokens appear in real time via Server-Sent Events (SSE)
- **OpenAI-compatible API** — drop-in replacement for the OpenAI API on `http://localhost:8000/v1`
- **Custom system prompt** — editable in the sidebar, persisted in `localStorage`
- **File attachments without text** — attach images or documents and send without typing anything
- **Image preview in chat** — sent images appear inline in your message bubble
- **Thinking / reasoning** — collapsible block shows the model's step-by-step reasoning
- **File attachments** — images, PDFs, code, and text documents
- **Session history** — previous chats saved in browser `localStorage` with sidebar navigation
- **Persistent model cache** — model weights downloaded once to a Modal Volume, never re-downloaded
- **Light theme UI** — warm, readable interface

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
| `static/index.html` | Single-file frontend — light theme chat UI with streaming, system prompt, vision, thinking blocks, and session history |
| `requirements.txt` | Local Python dependencies |
| `start.bat` | Windows convenience launcher |
| `.env.example` | Template for Modal endpoint URLs |

---

## Model Details

| Property | Value |
|----------|-------|
| Model | `google/gemma-3-4b-it` |
| Quantization | NF4 4-bit via `bitsandbytes` |
| Inference engine | HuggingFace `transformers` + `accelerate` |
| Vision | Native multimodal via `AutoProcessor` |
| GPU | NVIDIA A10G (24 GB VRAM) |

---

## Prerequisites

- [Python 3.10+](https://python.org)
- [Modal account](https://modal.com) (free tier works for testing)
- [Modal CLI](https://modal.com/docs/guide/installation)
- A [HuggingFace account](https://huggingface.co) with access to the Gemma model

---

## Setup & Deployment

### 1. Clone the repository

```bash
git clone https://github.com/Madushan996/Gemma-e4B-with-Modal.com-hosting.git
cd Gemma-e4B-with-Modal.com-hosting
```

### 2. Install local dependencies

```bash
pip install -r requirements.txt
```

### 3. Accept the Gemma model license on HuggingFace

The Gemma model is gated — you must request access before downloading it.

1. Go to [huggingface.co/google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
2. Click **"Agree and access repository"** (you must be logged in)

### 4. Add your HuggingFace token to Modal

Create a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then add it as a Modal secret:

```bash
modal secret create huggingface HF_TOKEN=hf_your_token_here
```

> The secret must be named exactly `huggingface` — that's what `modal_app.py` references.

### 5. Set up the Modal CLI

```bash
pip install modal
modal setup   # opens browser to authenticate
```

### 6. Deploy the Modal backend

```bash
modal deploy modal_app.py
```

This will:
- Build a Debian-based Docker image and install `torch`, `transformers`, `bitsandbytes`, and friends (~2–3 minutes, cached afterwards)
- Create a persistent Modal Volume named `gemma-models-hf`
- Print two endpoint URLs when complete

```
✓ Created web endpoint for GemmaService.health => https://YOUR-WORKSPACE--gemma-4-e4b-chat-gemmaservice-health.modal.run
✓ Created web endpoint for GemmaService.chat  => https://YOUR-WORKSPACE--gemma-4-e4b-chat-gemmaservice-chat.modal.run
```

> **First cold start** will download the Gemma model weights from HuggingFace (~8 GB) into the Modal Volume. This takes a few minutes and only happens once — subsequent starts load from the cached volume.

### 7. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and paste in your two endpoint URLs:

```env
MODAL_CHAT_URL=https://YOUR-WORKSPACE--gemma-4-e4b-chat-gemmaservice-chat.modal.run
MODAL_HEALTH_URL=https://YOUR-WORKSPACE--gemma-4-e4b-chat-gemmaservice-health.modal.run
```

### 8. Start the local server

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

### System prompt

Click **System Prompt** in the sidebar to expand the editor. Whatever you type there is sent as the `system` role message at the start of every new conversation. Changes are saved automatically to `localStorage`.

Examples:
```
You are a concise coding assistant. Always reply in Python.
```
```
You are a JSON annotation tool. Return only valid JSON, no prose.
```

Leave it blank to use the default Gemma persona.

### File attachments

| File type | How it's handled |
|-----------|-----------------|
| Images (PNG, JPG, WebP, GIF, BMP) | Sent as base64 to the model — Gemma sees the image natively |
| PDFs | Text extracted with `pypdf` and injected as context (up to 40,000 chars) |
| Text, code, Markdown, CSV, JSON… | Content read directly and injected as context |

Attached images appear as a preview in your chat bubble before the model responds.

### Vision status

The sidebar footer shows a **Vision enabled** badge (green dot) when the backend confirms multimodal support is active. If it shows grey, check the Modal container logs.

### Thinking / reasoning

When Gemma reasons before answering, a collapsible **Reasoning** block appears above the response. Click it to expand or collapse. The block pulses while the model is still thinking and collapses automatically when the response is complete.

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

> To reduce costs, change `gpu="A10G"` to `gpu="T4"` in `modal_app.py`. The T4 has 16 GB VRAM which fits the NF4-quantized 4B model, though it may be tight on long contexts.

---

## Local Development

Frontend changes (no redeploy needed):

```bash
python server.py   # hot-reload enabled
```

Edit `static/index.html` and refresh the browser.

Backend changes require a redeploy:

```bash
modal deploy modal_app.py
```

After the first build the Modal image is fully cached — code-only changes deploy in under 10 seconds.

---

## Batch Image Annotation

`batch_annotate.py` lets you annotate thousands of images automatically using the Gemma API. It reads a folder of images, sends them concurrently to `/v1/chat/completions`, and saves one JSON record per image to a `.jsonl` file.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Basic usage

Make sure `server.py` is running first, then:

```bash
python batch_annotate.py --input ./images
```

This processes every image in `./images` (recursively) and saves results to `annotations.jsonl`.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Folder of images to annotate |
| `--output` | `annotations.jsonl` | Output file (JSONL — one JSON record per line) |
| `--prompt` | Built-in schema | System prompt that defines what the model returns |
| `--concurrency` | `5` | Number of parallel requests (Modal auto-scales containers) |
| `--retries` | `3` | Retry attempts per image on failure |
| `--api-url` | `http://localhost:8000/v1/chat/completions` | API endpoint |
| `--model` | `gemma-3-4b-it` | Model name |

### Default output format

Each line in the `.jsonl` file is a JSON object:

```json
{
  "filename": "cat.jpg",
  "path": "/path/to/images/cat.jpg",
  "status": "ok",
  "annotation": {
    "label": "cat",
    "description": "A tabby cat sitting on a wooden floor.",
    "objects": ["cat", "wooden floor"],
    "colors": ["orange", "brown", "white"],
    "confidence": "high"
  }
}
```

Failed images have `"status": "error"` and an `"error"` field instead of `"annotation"`.

### Custom annotation schema

Pass a `--prompt` to define exactly what the model should return:

```bash
# Defect detection
python batch_annotate.py --input ./products \
  --prompt "Return JSON with: defect_found (true/false), defect_type (list), severity (low/medium/high/none), notes (string)."

# Multi-label classification
python batch_annotate.py --input ./photos \
  --prompt "Return JSON with: scene (indoor/outdoor), people_present (true/false), objects (list of up to 5 main objects)."

# Medical / scientific (adapt as needed)
python batch_annotate.py --input ./scans \
  --prompt "Return JSON with: tissue_type, anomalies_detected (true/false), region_of_interest (description)."
```

### Resume support

If the run is interrupted, just re-run the same command. Images already recorded in the output file are automatically skipped.

```bash
# Run 1 — interrupted at image 2000
python batch_annotate.py --input ./images --output annotations.jsonl

# Run 2 — resumes from image 2001
python batch_annotate.py --input ./images --output annotations.jsonl
```

### Load results in Python

```python
import json

records = []
with open("annotations.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

# Filter successful ones
ok = [r for r in records if r["status"] == "ok"]
print(f"{len(ok)} / {len(records)} succeeded")

# Convert to a list of flat dicts for pandas
import pandas as pd
rows = [{"filename": r["filename"], **r["annotation"]} for r in ok]
df = pd.DataFrame(rows)
print(df.head())
```

### Expected throughput for 5 000 images

With `--concurrency 5` (5 parallel GPU containers on Modal), assuming ~5 seconds per image:

| Metric | Value |
|--------|-------|
| Wall-clock time | ~80–90 minutes |
| GPU compute time | ~25 000 GPU-seconds |
| Estimated cost | ~$8 (A10G at ~$0.00032/s) |

Increase `--concurrency` to go faster (Modal will spin up more containers automatically). Each concurrent request = one additional GPU container.

---

## OpenAI-Compatible API

Once `server.py` is running, it exposes a fully OpenAI-compatible API at `http://localhost:8000/v1`. Any tool or library that supports a custom OpenAI base URL can use Gemma as a drop-in replacement — no code changes needed beyond pointing it at your local server.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | Chat completions (streaming and non-streaming) |

### curl

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-4b-it",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-4b-it",
    "messages": [{"role": "user", "content": "Tell me a joke."}],
    "stream": true
  }'
```

### Python — `openai` library

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # required by the library but ignored
)

# Non-streaming
response = client.chat.completions.create(
    model="gemma-3-4b-it",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain neural networks in one paragraph."},
    ],
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="gemma-3-4b-it",
    messages=[{"role": "user", "content": "Write a short poem about the sea."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### LangChain

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="gemma-3-4b-it",
)

messages = [
    SystemMessage(content="You are a helpful coding assistant."),
    HumanMessage(content="Write a Python function to reverse a string."),
]
print(llm.invoke(messages).content)
```

### Request parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | — | Model name (any string is accepted — only `gemma-3-4b-it` is running) |
| `messages` | array | — | Array of `{role, content}` objects. Roles: `system`, `user`, `assistant` |
| `stream` | boolean | `false` | Return SSE chunks (`true`) or a single JSON response (`false`) |
| `temperature` | float | `0.7` | Sampling temperature (0 = deterministic, 1 = creative) |
| `max_tokens` | integer | `2048` | Maximum tokens to generate |

> **Note:** `usage` (token counts) in the response is always `-1` — the Modal backend does not track token usage.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `401 Unauthorized` on model download | Make sure you accepted the license at huggingface.co/google/gemma-3-4b-it and the `HF_TOKEN` in your `huggingface` Modal secret is valid |
| Vision badge shows grey | The health check couldn't reach Modal — check `MODAL_HEALTH_URL` in `.env` and that the backend is deployed |
| Cold start takes a long time | Normal on first request after the container scales down — model loads from the volume (~30–60 seconds) |
| `303 See Other` errors in server logs | Already fixed — `server.py` uses `follow_redirects=True` on all Modal requests |
| Images not appearing in model response | Confirm the vision badge is green; check Modal container logs for `[setup] Model ready` |

---

## License

This project is released under the MIT License.

The Gemma model weights are subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
