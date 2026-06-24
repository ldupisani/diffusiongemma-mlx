#!/usr/bin/env node
// Interactive streaming chat — TypeScript port of chat_stream.py.
//
// Inference (the one non-portable piece) is delegated to the Python MLX sidecar
// (infer_server.py over HTTP), which runs whichever backend is active. Everything else
// — REPL, commands, MCP tool calling, the tool-call parsers, /continue, /auto, context
// readout — lives here. The tool-call format and thinking support are read from /info.
//
// Commands: /quality <preset>, /think, /continue, /auto <max>, /tools, /reset, /exit.
// Tool calling is on by default (spawns ../mcp_server_demo.py over MCP); --no-tools off.

import * as readline from "node:readline/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

import {
  chooseDiffusionParams,
  mcpToolToModel,
  parseToolCallsFor,
  stripThoughtFor,
} from "./tools.js";
import { fetchInfo, GenBody, GenStats, SidecarInfo, streamReply } from "./infer.js";

const THINK_MAX_TOKENS = 1024; // reasoning + answer need a larger budget
const DEFAULT_TOOL_STEPS = 6; // tool-round budget per turn when /auto is off

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const PYTHON = process.env.DIFG_PYTHON ?? join(REPO_ROOT, ".venv/bin/python");

interface State {
  server: string;
  ctxMax: number;
  backend: string;
  supportsThinking: boolean;
  toolCallFormat: string;
  session: Client | null;
  tools: unknown[];
  toolNames: string[];
  history: Record<string, unknown>[];
  quality: string;
  think: boolean;
  autoMax: number;
  maxTokens: number;
  // /continue state: original templated prompt + raw text of the last reply.
  lastPrompt: string;
  lastRaw: string;
  lastFinish: string | null;
  lastMax: number;
  lastThink: boolean;
}

function paramsBody(quality: string, maxTokens: number): Partial<GenBody> {
  const p = chooseDiffusionParams(quality, maxTokens);
  return {
    max_tokens: maxTokens,
    max_denoising_steps: p.max_denoising_steps,
    block_length: p.block_length,
    diffusion_sampler: p.diffusion_sampler,
    temperature: p.temperature,
  };
}

function printContext(stats: GenStats, ctxMax: number): void {
  const used = stats.promptTokens + stats.generationTokens;
  const ctx = ctxMax
    ? `${used.toLocaleString()} / ${ctxMax.toLocaleString()} tokens — ${((used / ctxMax) * 100).toFixed(1)}%`
    : `${used} tokens`;
  const mem = stats.peakMemory ? `, peak mem ${stats.peakMemory.toFixed(1)} GB` : "";
  console.log(`(context: ${ctx}${mem})\n`);
}

async function gen(
  state: State,
  body: GenBody,
): Promise<{ text: string; stats: GenStats }> {
  process.stdout.write("difg> ");
  const out = await streamReply(state.server, body, (t) => process.stdout.write(t));
  process.stdout.write("\n");
  return out;
}

// Run (or resume) one user turn, looping over tool calls until a final answer.
// Returns [lastPrompt, lastRaw, lastFinish]; "tool_cap" means the round budget
// ran out with calls still pending (resumable — history holds every tool result).
async function agentTurn(
  state: State,
  effectiveMax: number,
  budget: number,
): Promise<[string, string, string | null]> {
  let formatted = "";
  for (let step = 1; step <= budget; step++) {
    const body: GenBody = {
      messages: state.history,
      tools: state.tools.length ? state.tools : null,
      enable_thinking: state.think,
      ...paramsBody(state.quality, effectiveMax),
    };
    const { text, stats } = await gen(state, body);
    if (stats.prompt !== undefined) formatted = stats.prompt;
    printContext(stats, state.ctxMax);

    const calls =
      state.tools.length && state.session
        ? parseToolCallsFor(text, state.toolCallFormat)
        : [];
    if (calls.length === 0) {
      state.history.push({
        role: "assistant",
        content: stripThoughtFor(text, state.supportsThinking),
      });
      return [formatted, text, stats.finish];
    }

    const toolCalls: unknown[] = [];
    const toolReplies: Record<string, unknown>[] = [];
    for (let j = 0; j < calls.length; j++) {
      const { name, args } = calls[j];
      const cid = `s${step}_${j}`;
      console.log(`  [tool] ${name}(${JSON.stringify(args)})`);
      let value: string;
      try {
        const res: any = await state.session!.callTool({
          name,
          arguments: args as Record<string, unknown>,
        });
        const payload = res.content?.[0];
        value = payload?.text ?? JSON.stringify(payload);
      } catch (e) {
        value = `ERROR: ${e}`;
      }
      console.log(`  [tool] -> ${value}\n`);
      toolCalls.push({ id: cid, type: "function", function: { name, arguments: args } });
      toolReplies.push({ role: "tool", tool_call_id: cid, name, content: value });
    }
    state.history.push({ role: "assistant", tool_calls: toolCalls });
    state.history.push(...toolReplies);
  }
  return [formatted, "", "tool_cap"];
}

// One continuation of the last (truncated) reply, by raw concatenation of the
// original prompt + partial text — never re-templated (the chat template strips
// the <thought> channel, which would corrupt a mid-thought continuation).
async function extendReply(state: State): Promise<void> {
  const body: GenBody = {
    raw_prompt: state.lastPrompt + state.lastRaw,
    enable_thinking: state.lastThink,
    ...paramsBody(state.quality, state.lastMax),
  };
  const { text, stats } = await gen(state, body);
  printContext(stats, state.ctxMax);
  state.lastRaw += text;
  state.lastFinish = stats.finish;
  state.history[state.history.length - 1] = {
    role: "assistant",
    content: stripThoughtFor(state.lastRaw, state.supportsThinking),
  };
}

async function autoExtend(state: State, autoMax: number): Promise<void> {
  let n = 0;
  while (state.lastFinish === "length" && n < autoMax) {
    n++;
    console.log(`(auto-continue ${n}/${autoMax})`);
    await extendReply(state);
  }
  if (state.lastFinish === "length") {
    console.log(`(stopped at ${autoMax} auto-extensions — /continue or raise /auto)\n`);
  }
}

// Run a turn (or resume one) and apply the matching follow-up policy. /auto N is
// the single "keep going" dial: the tool-round budget AND text auto-continuation.
async function turnWithFollowups(state: State, effectiveMax: number): Promise<void> {
  const budget = state.autoMax > 0 ? state.autoMax : DEFAULT_TOOL_STEPS;
  const [prompt, raw, finish] = await agentTurn(state, effectiveMax, budget);
  state.lastPrompt = prompt;
  state.lastRaw = raw;
  state.lastFinish = finish;
  if (finish === "length") {
    if (state.autoMax > 0) await autoExtend(state, state.autoMax);
    else console.log("(reply hit the token cap — type /continue to extend it)\n");
  } else if (finish === "tool_cap") {
    console.log(
      `(used ${budget} tool steps without finishing — /continue to keep going, or raise /auto)\n`,
    );
  }
}

async function repl(state: State, rl: readline.Interface): Promise<void> {
  // Iterate the readline async iterator rather than one-shot question(): it
  // buffers input lines while a turn is awaiting (so nothing is dropped when
  // input arrives faster than we consume it) and ends cleanly on stdin EOF.
  process.stdout.write("you> ");
  let exiting = false;
  for await (const line of rl) {
    try {
      const user = line.trim();
      if (!user) continue;
      if (user === "/exit") {
        exiting = true;
        break;
      }
      if (user === "/reset") {
      state.history.length = 0;
      console.log("(history cleared)\n");
      continue;
    }
    if (user === "/tools") {
      console.log(`(tools: ${state.toolNames.length ? state.toolNames.join(", ") : "none"})\n`);
      continue;
    }
    if (user.startsWith("/quality")) {
      const arg = user.split(/\s+/)[1];
      if (arg && ["fast", "balanced", "quality"].includes(arg)) {
        state.quality = arg;
        console.log(`(quality -> ${arg})\n`);
      } else console.log("usage: /quality fast|balanced|quality\n");
      continue;
    }
    if (user === "/think") {
      if (!state.supportsThinking) {
        console.log(
          `(the ${state.backend} backend has no <thought> channel — /think ignored)\n`,
        );
        continue;
      }
      state.think = !state.think;
      console.log(`(thinking -> ${state.think ? "on" : "off"})\n`);
      continue;
    }
    if (user.startsWith("/auto")) {
      const arg = user.split(/\s+/)[1];
      if (arg && /^\d+$/.test(arg)) {
        state.autoMax = parseInt(arg, 10);
        console.log(
          `(auto -> ${state.autoMax} text-continuations / tool-rounds per turn${state.autoMax === 0 ? " (off)" : ""})\n`,
        );
        if (state.autoMax > 0 && state.lastFinish === "length") {
          await autoExtend(state, state.autoMax);
        } else if (state.autoMax > 0 && state.lastFinish === "tool_cap") {
          await turnWithFollowups(state, state.lastMax);
        }
      } else console.log("usage: /auto <max>   (0 disables)\n");
      continue;
    }
    if (user === "/continue") {
      if (state.lastFinish === "length") {
        await extendReply(state);
      } else if (state.lastFinish === "tool_cap") {
        await turnWithFollowups(state, state.lastMax);
      } else {
        console.log("(nothing to continue — last reply finished naturally)\n");
      }
      continue;
    }

    // --- a normal user turn (may involve tool calls) ---
    const turnStart = state.history.length;
    state.history.push({ role: "user", content: user });
    const effectiveMax = state.think ? THINK_MAX_TOKENS : state.maxTokens;
    state.lastMax = effectiveMax;
    state.lastThink = state.think;
    try {
      await turnWithFollowups(state, effectiveMax);
      } catch (e) {
        console.log(`\n[turn error: ${e}]\n`);
        state.history.length = turnStart; // roll back the whole turn
        state.lastFinish = null;
      }
    } finally {
      if (!exiting) process.stdout.write("you> ");
    }
  }
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const useTools = !argv.includes("--no-tools");
  const serverArg = argv[argv.indexOf("--server") + 1];
  const server =
    argv.includes("--server") && serverArg ? serverArg : "http://127.0.0.1:8088";

  let info: SidecarInfo;
  try {
    info = await fetchInfo(server);
  } catch (e) {
    console.error(`Cannot reach inference sidecar at ${server}. Start it first:`);
    console.error("  source .venv/bin/activate && python infer_server.py");
    process.exit(1);
  }

  let session: Client | null = null;
  let tools: unknown[] = [];
  let toolNames: string[] = [];
  if (useTools) {
    try {
      const transport = new StdioClientTransport({
        command: PYTHON,
        args: [join(REPO_ROOT, "mcp_server_demo.py")],
      });
      session = new Client({ name: "difg-ts", version: "1.0.0" });
      await session.connect(transport);
      const listed = await session.listTools();
      tools = listed.tools.map(mcpToolToModel);
      toolNames = listed.tools.map((t) => t.name);
      console.log(`MCP connected — tools: [${toolNames.join(", ")}]`);
    } catch (e) {
      console.log(`(MCP unavailable: ${e} — continuing without tools)`);
      session = null;
      tools = [];
    }
  }

  const state: State = {
    server,
    ctxMax: info.context_window,
    backend: info.backend,
    supportsThinking: info.supports_thinking,
    toolCallFormat: info.tool_call_format,
    session,
    tools,
    toolNames,
    history: [],
    quality: "balanced",
    think: false,
    autoMax: 0,
    maxTokens: 256,
    lastPrompt: "",
    lastRaw: "",
    lastFinish: null,
    lastMax: 256,
    lastThink: false,
  };

  console.log(
    `Ready (${info.backend} backend, model ${info.model}, ` +
      `context ${info.context_window.toLocaleString()} tokens, ` +
      `${tools.length ? "with tools" : "no tools"}). ` +
      "Commands: /quality, /think, /continue, /auto <max>, /tools, /reset, /exit.\n",
  );

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  try {
    await repl(state, rl);
  } finally {
    rl.close();
    if (session) await session.close();
  }
}

main();
