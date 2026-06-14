// Port of tools_util.py + the diffusion presets and thought-stripping.
// The model emits tool calls in Gemma's native DSL (NOT JSON):
//   <|tool_call>call:NAME{key:<|"|>value<|"|>,n:42}<tool_call|>
// and may emit several back-to-back (parallel calls).

export const QUOTE = '<|"|>'; // Gemma's string-quote token inside the call DSL
const OPEN = "<|tool_call>call:"; // each call starts here; the close token is optional

const THOUGHT_RE = /<\|channel>thought[\s\S]*?<channel\|>\s*/g;

export function stripThought(text: string): string {
  return text.replace(THOUGHT_RE, "").trim();
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
