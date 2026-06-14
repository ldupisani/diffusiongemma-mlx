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

DEFAULT_MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"


def choose_diffusion_params(quality: str, max_tokens: int) -> dict:
    """Map a human 'quality' preference to diffusion sampling parameters.

    This is the one genuinely interesting decision in running a diffusion LM,
    and there is no single right answer — it's a speed/quality trade-off that
    depends on your use case. The reference defaults from Google's guide are
    48 denoising steps over a 256-token canvas with the 'entropy-bound' sampler.

    TODO(you): implement this. Return a dict with these keys:
        {
            "max_denoising_steps": int,   # refinement passes (e.g. 12 / 32 / 48)
            "block_length": int,          # canvas size; >= max_tokens is typical
            "diffusion_sampler": str,     # "entropy-bound" or "confidence-threshold"
            "temperature": float,         # 0.0 = greedy; guide uses ~0.4-0.8
        }

    Consider:
      - "fast"     -> few steps (~12), entropy-bound, low temp. Great for
                      structured/constrained tasks (the guide's Sudoku demo hit
                      80% at just 12 steps after fine-tuning).
      - "balanced" -> ~32 steps, the everyday default.
      - "quality"  -> 48 steps, more self-correction passes for open-ended prose.
      - block_length should be at least `max_tokens`; the canvas can't emit more
        tokens than it has slots. 256 is the model's design point.
    """
    presets = {
        # few passes, greedy: fast + stable for structured/constrained tasks
        "fast": {"max_denoising_steps": 12, "diffusion_sampler": "entropy-bound", "temperature": 0.2},
        # everyday default
        "balanced": {"max_denoising_steps": 32, "diffusion_sampler": "entropy-bound", "temperature": 0.4},
        # full reference config: most self-correction passes for open-ended prose
        "quality": {"max_denoising_steps": 48, "diffusion_sampler": "entropy-bound", "temperature": 0.6},
    }
    params = presets[quality]
    # canvas must have at least as many slots as tokens we want to emit; 256 is the design point
    params["block_length"] = max(256, max_tokens)
    return params


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
