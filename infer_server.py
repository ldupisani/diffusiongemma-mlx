#!/usr/bin/env python3
"""Tiny MLX inference sidecar for the TypeScript REPL (chat_stream.ts).

MLX generation (diffusion via mlx-vlm, or autoregressive via mlx-lm) is the one
non-portable piece, so the TS port delegates it to this local HTTP server and owns
everything else itself. The actual model family is chosen by backend.py — this
server is backend-agnostic and just forwards the resolved generation params.

Why plain http.server (not FastAPI/uvicorn): MLX's GPU stream is thread-local, so
generation must run on the main thread. HTTPServer (single-threaded) processes each
request on the serving thread, which is exactly what we want. One client at a time.

Endpoints:
  GET  /info      -> {"model","context_window","backend","supports_thinking","tool_call_format"}
  POST /generate  -> streams NDJSON, one JSON object per line:
      {"type":"prompt","prompt": "<resolved templated prompt>"}   (templated path only)
      {"type":"chunk","text": "..."}                              (repeated)
      {"type":"done","finish":..., "prompt_tokens":..., "generation_tokens":...,
       "peak_memory":...}

Request body (POST /generate):
  Either {"messages":[...], "tools":[...]|null}  (server applies the chat template)
  or     {"raw_prompt":"..."}                    (fed verbatim — used for /continue).
  Plus: enable_thinking, max_tokens, max_denoising_steps, block_length,
        diffusion_sampler, temperature. (Diffusion-only knobs are ignored by the
        autoregressive backend.)
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlx.core as mx

from backend import DEFAULT_MODEL, get_backend

BACKEND = None


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
            self._json(200, BACKEND.info())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        enable_thinking = req.get("enable_thinking", False)

        # Resolve the prompt: either a raw continuation prompt or templated messages.
        if req.get("raw_prompt") is not None:
            prompt = req["raw_prompt"]
            emit_prompt = False
        else:
            prompt = BACKEND.apply_template(
                req.get("messages", []),
                tools=req.get("tools") or None,
                enable_thinking=enable_thinking,
            )
            emit_prompt = True

        # Each backend pulls only the generation params it understands out of the
        # request — diffusion reads its canvas knobs, autoregressive just temperature.
        params = BACKEND.params_from_request(req)

        # Seeding the global PRNG makes sampling reproducible (and, for diffusion, the
        # random initial canvas). Varying the seed is what gives one_shot_n diversity.
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
            for chunk in BACKEND.stream(prompt, max_tokens=req.get("max_tokens", 256),
                                        enable_thinking=enable_thinking, params=params):
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
    global BACKEND
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8088
    BACKEND = get_backend(model_id)
    BACKEND.load(model_id)
    print(f"Inference sidecar ready on http://127.0.0.1:{port} "
          f"({BACKEND.name} backend, context {BACKEND.context_window:,})", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
