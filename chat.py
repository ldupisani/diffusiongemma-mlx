#!/usr/bin/env python3
"""Minimal DiffusionGemma runner on Apple Silicon via mlx-vlm.

DiffusionGemma is NOT autoregressive: instead of emitting tokens left-to-right,
it lays down a fixed-size "canvas" of masked tokens and refines the whole block
in parallel over `max_denoising_steps` passes. So the two knobs that matter most
are different from a normal LLM:

  * block_length        -> how many tokens are denoised together (the canvas)
  * max_denoising_steps -> how many refinement passes over that canvas

Fewer steps = faster but rougher; more steps = slower but more self-corrected.
"""

import argparse
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# This CLI is the diffusion demo specifically, so it defaults to the diffusion model.
# The shared default for the rest of the toolkit (a small autoregressive model) and the
# diffusion preset table both live in backend.py.
from backend import DEFAULT_DIFFUSION_MODEL as DEFAULT_MODEL, choose_diffusion_params


def main():
    p = argparse.ArgumentParser(description="Run DiffusionGemma with MLX.")
    p.add_argument("prompt", nargs="+", help="The user prompt.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--quality", choices=["fast", "balanced", "quality"], default="balanced")
    p.add_argument("--image", default=None, help="Optional image path/URL (model is multimodal).")
    args = p.parse_args()

    prompt_text = " ".join(args.prompt)
    params = choose_diffusion_params(args.quality, args.max_tokens)

    print(f"Loading {args.model} ...")
    model, processor = load(args.model)
    config = model.config

    num_images = 1 if args.image else 0
    formatted = apply_chat_template(processor, config, prompt_text, num_images=num_images)

    result = generate(
        model,
        processor,
        formatted,
        image=args.image,
        max_tokens=args.max_tokens,
        verbose=True,
        **params,
    )
    print("\n--- text ---\n" + result.text)


if __name__ == "__main__":
    main()
