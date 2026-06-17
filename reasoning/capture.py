"""Capture a teacher (thinking-on) reasoning sample for a query.

This is the 'slow interpreted path' we record once so derive.py can compile it into a
coded orchestrator. A sample stores the query, the model's reasoning trace, and its
final answer, under store/samples/<slug>.json.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .registry import SAMPLE_DIR
from .teachers import get_teacher


def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:6]
    return "-".join(words) or "sample"


def capture(query: str, *, seed: int | None = 0, teacher_backend: str = "diffusion") -> dict:
    """Run the teacher with thinking ON and persist the reasoning sample. Returns it.

    `teacher_backend` selects who teaches: 'diffusion' (self-distillation) or 'claude'
    (the stronger frontier teacher). The stored sample records which one produced it.
    """
    t = get_teacher(teacher_backend).trace(query, seed=seed or 0)
    sample = {
        "id": f"{_slug(query)}",
        "query": query,
        "reasoning": t.reasoning,
        "answer": t.answer,
        "teacher": t.backend,
        "seed": seed,
        "captured_at": int(time.time()),
    }
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLE_DIR / f"{sample['id']}.json"
    path.write_text(json.dumps(sample, indent=2))
    sample["_path"] = str(path)
    return sample


def load_sample(sample_id: str) -> dict | None:
    path = Path(SAMPLE_DIR) / f"{sample_id}.json"
    return json.loads(path.read_text()) if path.exists() else None
