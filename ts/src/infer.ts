// HTTP client for the Python MLX inference sidecar (infer_server.py).
// The sidecar streams NDJSON: one {"type": ...} JSON object per line.

export interface GenStats {
  finish: string | null;
  promptTokens: number;
  generationTokens: number;
  peakMemory: number | null;
  prompt?: string; // resolved templated prompt (templated path only)
}

export interface GenBody {
  // Either messages (+tools) for the templated path, or raw_prompt for /continue.
  messages?: unknown[];
  tools?: unknown[] | null;
  raw_prompt?: string;
  enable_thinking?: boolean;
  max_tokens?: number;
  max_denoising_steps?: number;
  block_length?: number;
  diffusion_sampler?: string;
  temperature?: number;
}

export interface SidecarInfo {
  context_window: number;
  model: string;
  backend: string; // "diffusion" | "autoregressive"
  supports_thinking: boolean;
  tool_call_format: string; // "gemma-dsl" | "json"
}

export async function fetchInfo(server: string): Promise<SidecarInfo> {
  const res = await fetch(`${server}/info`);
  if (!res.ok) throw new Error(`sidecar /info ${res.status}`);
  const raw = (await res.json()) as Partial<SidecarInfo>;
  // Tolerate an older sidecar that only reports model + context_window.
  return {
    context_window: raw.context_window ?? 0,
    model: raw.model ?? "unknown",
    backend: raw.backend ?? "diffusion",
    supports_thinking: raw.supports_thinking ?? true,
    tool_call_format: raw.tool_call_format ?? "gemma-dsl",
  };
}

// Ask the sidecar to try the deterministic reasoning engine (incl. ≤2-call reformulation).
// Returns the coded answer, or answer=null to defer to a full LLM generation. Tolerates an
// older sidecar without /solve (treated as "no coded answer").
export async function solve(
  server: string,
  query: string,
): Promise<{ answer: string | null; modelCalls: number }> {
  let res: Response;
  try {
    res = await fetch(`${server}/solve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
  } catch {
    return { answer: null, modelCalls: 0 };
  }
  if (!res.ok) return { answer: null, modelCalls: 0 };
  const data = (await res.json()) as { answer: string | null; model_calls?: number };
  return { answer: data.answer ?? null, modelCalls: data.model_calls ?? 0 };
}

// Stream one generation, calling onChunk for each text fragment as it arrives.
export async function streamReply(
  server: string,
  body: GenBody,
  onChunk: (t: string) => void,
): Promise<{ text: string; stats: GenStats }> {
  const res = await fetch(`${server}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) throw new Error(`sidecar /generate ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let text = "";
  const stats: GenStats = {
    finish: null,
    promptTokens: 0,
    generationTokens: 0,
    peakMemory: null,
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      if (!line.trim()) continue;
      const ev = JSON.parse(line);
      if (ev.type === "prompt") {
        stats.prompt = ev.prompt;
      } else if (ev.type === "chunk") {
        text += ev.text;
        onChunk(ev.text);
      } else if (ev.type === "done") {
        stats.finish = ev.finish;
        stats.promptTokens = ev.prompt_tokens ?? 0;
        stats.generationTokens = ev.generation_tokens ?? 0;
        stats.peakMemory = ev.peak_memory ?? null;
      }
    }
  }
  return { text, stats };
}
