// Port of tools_util.py + the diffusion presets and thought-stripping.
// Two tool-call wire formats are supported, one per backend family:
//   * Gemma DSL (diffusion):  <|tool_call>call:NAME{key:<|"|>value<|"|>,n:42}<tool_call|>
//   * JSON (autoregressive):  <tool_call>{"name": NAME, "arguments": {...}}</tool_call>
// Both may emit several blocks back-to-back (parallel calls).

export const QUOTE = '<|"|>'; // Gemma's string-quote token inside the call DSL
const OPEN = "<|tool_call>call:"; // each call starts here; the close token is optional

const THOUGHT_RE = /<\|channel>thought[\s\S]*?<channel\|>\s*/g;

export function stripThought(text: string): string {
  return text.replace(THOUGHT_RE, "").trim();
}

// Backend-aware thought stripping: only the diffusion model has a <thought> channel.
export function stripThoughtFor(text: string, supportsThinking: boolean): string {
  return supportsThinking ? stripThought(text) : text.trim();
}

export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
}

// Index of the `}` that closes the `{` at openIdx, or -1 if unbalanced.
function balancedBraceEnd(text: string, openIdx: number): number {
  let depth = 0;
  for (let k = openIdx; k < text.length; k++) {
    if (text[k] === "{") depth++;
    else if (text[k] === "}") {
      depth--;
      if (depth === 0) return k;
    }
  }
  return -1;
}

// Split a `{...}` body on top-level commas (ignores commas inside nested braces).
function splitTopLevel(body: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < body.length; i++) {
    const ch = body[i];
    if (ch === "{") depth++;
    else if (ch === "}") depth--;
    else if (ch === "," && depth === 0) {
      parts.push(body.slice(start, i));
      start = i + 1;
    }
  }
  parts.push(body.slice(start));
  return parts.filter((p) => p.trim());
}

function parseArgs(body: string): Record<string, unknown> {
  const args: Record<string, unknown> = {};
  for (const pair of splitTopLevel(body)) {
    const idx = pair.indexOf(":");
    if (idx === -1) continue;
    const key = pair.slice(0, idx).trim();
    const val = pair.slice(idx + 1).trim();
    if (val.startsWith(QUOTE) && val.endsWith(QUOTE)) {
      args[key] = val.slice(QUOTE.length, val.length - QUOTE.length);
    } else if (val === "true" || val === "false") {
      args[key] = val === "true";
    } else if (/^-?\d+$/.test(val)) {
      args[key] = parseInt(val, 10);
    } else if (/^-?\d*\.\d+$/.test(val)) {
      args[key] = parseFloat(val);
    } else {
      args[key] = val.split(QUOTE).join("");
    }
  }
  return args;
}

// Robust to two things models do wrong: dropping the <tool_call|> terminator, and
// emitting several calls in a row. We never rely on the closing token — we find
// each opener, read the name up to `{`, then scan BALANCED braces for the args.
export function parseToolCalls(text: string): ToolCall[] {
  const calls: ToolCall[] = [];
  let i = 0;
  for (;;) {
    const start = text.indexOf(OPEN, i);
    if (start === -1) return calls;
    const brace = text.indexOf("{", start + OPEN.length);
    if (brace === -1) return calls;
    const end = balancedBraceEnd(text, brace);
    if (end === -1) return calls; // truncated mid-args; stop
    const name = text.slice(start + OPEN.length, brace).trim();
    calls.push({ name, args: parseArgs(text.slice(brace + 1, end)) });
    i = end + 1;
  }
}

// JSON.parse the first balanced {...} object in blob, or null. Tolerates redundant
// outer brace layers: Qwen2.5's bundled chat template shows the tool-call example as
// {{"name": ..., "arguments": ...}} (a Jinja-escape artifact) and the model copies the
// doubled braces verbatim — so we peel matched outer layers until it parses to an object.
function loadsFirstObject(blob: string): Record<string, unknown> | null {
  const brace = blob.indexOf("{");
  if (brace === -1) return null;
  const end = balancedBraceEnd(blob, brace);
  if (end === -1) return null;
  let candidate = blob.slice(brace, end + 1).trim();
  for (let i = 0; i < 4; i++) {
    try {
      const obj = JSON.parse(candidate);
      if (obj && typeof obj === "object" && !Array.isArray(obj)) {
        return obj as Record<string, unknown>;
      }
    } catch {
      /* fall through to peel a layer */
    }
    if (candidate.startsWith("{") && candidate.endsWith("}")) {
      candidate = candidate.slice(1, -1).trim(); // peel one redundant brace layer
    } else {
      return null;
    }
  }
  return null;
}

// JSON tool-call format emitted by standard instruct models (Qwen2.5, Llama-3.2,
// Hermes-style): <tool_call>{"name": "foo", "arguments": {...}}</tool_call>. Robust
// to a missing </tool_call> terminator and to several calls back-to-back.
const JSON_OPEN = "<tool_call>";
const JSON_CLOSE = "</tool_call>";

export function parseToolCallsJson(text: string): ToolCall[] {
  const calls: ToolCall[] = [];
  let i = 0;
  for (;;) {
    const start = text.indexOf(JSON_OPEN, i);
    if (start === -1) return calls;
    const bodyStart = start + JSON_OPEN.length;
    const close = text.indexOf(JSON_CLOSE, bodyStart);
    let blob: string;
    if (close === -1) {
      const brace = text.indexOf("{", bodyStart);
      if (brace === -1) return calls;
      const end = balancedBraceEnd(text, brace);
      if (end === -1) return calls; // truncated mid-args; stop
      blob = text.slice(brace, end + 1);
      i = end + 1;
    } else {
      blob = text.slice(bodyStart, close);
      i = close + JSON_CLOSE.length;
    }
    const obj = loadsFirstObject(blob);
    if (!obj || typeof obj.name !== "string") continue;
    let args = (obj.arguments ?? obj.parameters ?? {}) as unknown;
    if (typeof args === "string") {
      try {
        args = JSON.parse(args);
      } catch {
        args = {};
      }
    }
    calls.push({
      name: obj.name,
      args: (args && typeof args === "object" ? args : {}) as Record<string, unknown>,
    });
  }
}

// Pick the parser matching the sidecar's reported tool_call_format.
export function parseToolCallsFor(text: string, format: string): ToolCall[] {
  return format === "json" ? parseToolCallsJson(text) : parseToolCalls(text);
}

export interface DiffusionParams {
  max_denoising_steps: number;
  diffusion_sampler: string;
  temperature: number;
  block_length: number;
}

// Map a human 'quality' preference to diffusion sampling parameters. Reference
// defaults from Google's guide: 48 steps / 256-token canvas / entropy-bound.
export function chooseDiffusionParams(quality: string, maxTokens: number): DiffusionParams {
  const presets: Record<string, Omit<DiffusionParams, "block_length">> = {
    fast: { max_denoising_steps: 12, diffusion_sampler: "entropy-bound", temperature: 0.2 },
    balanced: { max_denoising_steps: 32, diffusion_sampler: "entropy-bound", temperature: 0.4 },
    quality: { max_denoising_steps: 48, diffusion_sampler: "entropy-bound", temperature: 0.6 },
  };
  const p = presets[quality] ?? presets.balanced;
  return { ...p, block_length: Math.max(256, maxTokens) };
}

// MCP tool (name/description/inputSchema) -> the model's tools= entry. MCP's
// inputSchema is already JSON Schema, so this is a thin reshape.
export function mcpToolToModel(tool: {
  name: string;
  description?: string;
  inputSchema?: unknown;
}) {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description ?? "",
      parameters: tool.inputSchema ?? { type: "object", properties: {} },
    },
  };
}
