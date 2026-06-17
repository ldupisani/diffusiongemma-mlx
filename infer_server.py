#!/usr/bin/env python3
"""Tiny MLX inference sidecar for the TypeScript REPL (chat_stream.ts).

DiffusionGemma only runs through mlx-vlm (Python), so the TS port delegates the
single non-portable piece — generation — to this local HTTP server and owns
everything else itself.

Why plain http.server (not FastAPI/uvicorn): MLX's GPU stream is thread-local, so
generation must run on the main thread. HTTPServer (single-threaded) processes each
request on the serving thread, which is exactly what we want. One client at a time.

Endpoints:
  GET  /info      -> {"context_window": int, "model": str}
  POST /generate  -> streams NDJSON, one JSON object per line:
      {"type":"prompt","prompt": "<resolved templated prompt>"}   (templated path only)
      {"type":"chunk","text": "..."}                              (repeated)
      {"type":"done","finish":..., "prompt_tokens":..., "generation_tokens":...,
       "peak_memory":...}

Request body (POST /generate):
  Either {"messages":[...], "tools":[...]|null}  (server applies the chat template)
  or     {"raw_prompt":"..."}                    (fed verbatim — used for /continue).
  Plus: enable_thinking, max_tokens, max_denoising_steps, block_length,
        diffusion_sampler, temperature.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlx.core as mx
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template

from chat import DEFAULT_MODEL
from chat_stream import context_limit

MODEL = None
PROCESSOR = None
CONFIG = None
CTX_MAX = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # close per request -> client reads stream to EOF

    def log_message(self, *args):
        pass  # quiet

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/info":
            self._json(200, {"context_window": CTX_MAX, "model": DEFAULT_MODEL})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")

        # Resolve the prompt: either a raw continuation prompt or templated messages.
        if req.get("raw_prompt") is not None:
            prompt = req["raw_prompt"]
            emit_prompt = False
        else:
            prompt = apply_chat_template(
                PROCESSOR, CONFIG, req.get("messages", []), num_images=0,
                tools=req.get("tools") or None,
                enable_thinking=req.get("enable_thinking", False),
            )
            emit_prompt = True

        gen_kwargs = dict(
            max_tokens=req.get("max_tokens", 256),
            max_denoising_steps=req.get("max_denoising_steps", 32),
            block_length=req.get("block_length", 256),
            diffusion_sampler=req.get("diffusion_sampler", "entropy-bound"),
            temperature=req.get("temperature", 0.4),
            enable_thinking=req.get("enable_thinking", False),
        )

        # Seeding the global PRNG makes the random initial canvas + sampling
        # reproducible. It is NOT a stream_generate kwarg — set it here, before
        # generating. Varying the seed is what gives one_shot_n diverse candidates.
        seed = req.get("seed")
        if seed is not None:
            mx.random.seed(int(seed))

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        def write(obj):
            self.wfile.write((json.dumps(obj) + "\n").encode())
            self.wfile.flush()

        if emit_prompt:
            write({"type": "prompt", "prompt": prompt})

        last = None
        try:
            for chunk in stream_generate(MODEL, PROCESSOR, prompt, **gen_kwargs):
                last = chunk
                write({"type": "chunk", "text": chunk.text})
        except (BrokenPipeError, ConnectionResetError):
            return  # client disconnected mid-stream
        write({
            "type": "done",
            "finish": getattr(last, "finish_reason", None),
            "prompt_tokens": getattr(last, "prompt_tokens", 0) or 0,
            "generation_tokens": getattr(last, "generation_tokens", 0) or 0,
            "peak_memory": getattr(last, "peak_memory", None),
        })


def main():
    global MODEL, PROCESSOR, CONFIG, CTX_MAX
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8088
    print(f"Loading {model_id} ...", flush=True)
    MODEL, PROCESSOR = load(model_id)
    CONFIG = MODEL.config
    CTX_MAX = context_limit(CONFIG)
    print(f"Inference sidecar ready on http://127.0.0.1:{port} "
          f"(context {CTX_MAX:,})", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
