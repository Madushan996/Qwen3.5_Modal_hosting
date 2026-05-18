"""
batch_annotate.py — Annotate a folder of images using the Gemma API.

Reads every image in --input, calls /v1/chat/completions concurrently,
and appends one JSON record per image to a .jsonl output file.

Supports resuming: images already present in the output file are skipped.

Usage examples
--------------
# Basic — annotate all images in ./images, save to annotations.jsonl
python batch_annotate.py --input ./images

# Custom output file and concurrency
python batch_annotate.py --input ./images --output results.jsonl --concurrency 8

# Custom annotation schema via system prompt
python batch_annotate.py --input ./images --prompt "Return JSON with keys: label, defects (list), severity (low/medium/high)."

# Call a remote server instead of localhost
python batch_annotate.py --input ./images --api-url http://your-server:8000/v1/chat/completions
"""

import argparse
import asyncio
import base64
import json
import re
import sys
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm as atqdm

# ── Constants ──────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",  ".gif": "image/gif",
    ".webp": "image/webp", ".bmp": "image/bmp",
}

DEFAULT_SYSTEM_PROMPT = """\
You are an image annotation assistant.
For each image, return ONLY a JSON object with these fields — no extra text:

{
  "label":       "<short category, e.g. 'cat', 'street scene', 'product'>",
  "description": "<1–2 sentence description of the image>",
  "objects":     ["<main objects or elements visible>"],
  "colors":      ["<dominant colors>"],
  "confidence":  "<high | medium | low>"
}"""

DEFAULT_API_URL    = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL      = "gemma-3-4b-it"
DEFAULT_CONCURRENCY = 5
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.2   # low for consistent/deterministic annotations
DEFAULT_MAX_TOKENS  = 512


# ── Helpers ────────────────────────────────────────────────────────────────

def to_data_url(path: Path) -> str:
    mime = MIME_MAP.get(path.suffix.lower(), "image/jpeg")
    b64  = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def extract_json(text: str) -> dict:
    """Parse JSON from model output, handling extra prose around it."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"raw_response": text}


# ── Core annotation logic ──────────────────────────────────────────────────

async def annotate_image(
    client: httpx.AsyncClient,
    image_path: Path,
    system_prompt: str,
    api_url: str,
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": to_data_url(image_path)}},
                ],
            },
        ],
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "stream": False,
    }

    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.post(api_url, json=payload)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return {
                    "filename": image_path.name,
                    "path":     str(image_path),
                    "status":   "ok",
                    "annotation": extract_json(content),
                }
            except Exception as exc:
                if attempt == max_retries - 1:
                    return {
                        "filename": image_path.name,
                        "path":     str(image_path),
                        "status":   "error",
                        "error":    str(exc),
                    }
                await asyncio.sleep(2 ** attempt)   # exponential backoff


# ── Main ───────────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    input_dir   = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        sys.exit(1)

    # Collect images
    all_images = sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not all_images:
        print(f"No images found in '{input_dir}'.")
        sys.exit(1)

    # Resume: skip images already recorded in the output file
    already_done: set[str] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["filename"])
                except (json.JSONDecodeError, KeyError):
                    pass

    remaining = [img for img in all_images if img.name not in already_done]

    print("\n" + "=" * 50)
    print("  Gemma Batch Annotator")
    print("=" * 50)
    print(f"  Input folder : {input_dir}")
    print(f"  Output file  : {output_path}")
    print(f"  Total images : {len(all_images)}")
    print(f"  Already done : {len(already_done)}")
    print(f"  To process   : {len(remaining)}")
    print(f"  Concurrency  : {args.concurrency}")
    print(f"  API URL      : {args.api_url}")
    print("=" * 50 + "\n")

    if not remaining:
        print("All images are already annotated. Nothing to do.")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout   = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)

    ok = err = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = [
                annotate_image(
                    client, img, args.prompt, args.api_url,
                    args.model, semaphore, args.retries,
                )
                for img in remaining
            ]
            async for result in atqdm.as_completed(
                tasks, total=len(remaining), desc="Annotating", unit="img"
            ):
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()   # write each result immediately
                if result["status"] == "ok":
                    ok += 1
                else:
                    err += 1
                    print(f"\n  ✗ {result['filename']}: {result.get('error', '?')}")

    print(f"\nFinished — {ok} succeeded, {err} failed.")
    print(f"Results saved to: {output_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-annotate images using the Gemma OpenAI-compatible API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",       required=True,
                        help="Folder of images to annotate (searched recursively)")
    parser.add_argument("--output",      default="annotations.jsonl",
                        help="Output file (JSONL, one record per image). Default: annotations.jsonl")
    parser.add_argument("--prompt",      default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt defining the annotation schema")
    parser.add_argument("--api-url",     default=DEFAULT_API_URL,
                        help=f"Chat completions endpoint. Default: {DEFAULT_API_URL}")
    parser.add_argument("--model",       default=DEFAULT_MODEL,
                        help=f"Model name. Default: {DEFAULT_MODEL}")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Max parallel requests (Modal spins up one GPU container per request). Default: {DEFAULT_CONCURRENCY}")
    parser.add_argument("--retries",     type=int, default=DEFAULT_MAX_RETRIES,
                        help=f"Max retries per image on failure. Default: {DEFAULT_MAX_RETRIES}")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
