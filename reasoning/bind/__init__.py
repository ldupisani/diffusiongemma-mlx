"""The bind layer: natural language → typed arguments.

This is the seam that decides whether a whole query runs model-free. For a known query
type we try a CODED parser (regex/grammar) first; only if that fails do we fall back to
the model (`extract_validate_repair`). A bind returns (ok, data, via) where `via` is
"code" (parser succeeded → path stays model-free) or "model" (fell back).

Design rule — be CONSERVATIVE: a coded parser returns failure rather than guess on an
ambiguous query. A wrong coded parse is a silent confident-but-wrong answer with zero
model calls; a fallback is graceful (slower, loads the sidecar). Failure is the safer bet.

The deterministic parsing primitives (_parse_int/_parse_list/_try_json) live here so the
skill/compiled layers can share them without importing patterns; patterns.py re-exports
them for backward compatibility.
"""
from __future__ import annotations

from reasoning.bind._common import (_NUMBER, _parse_int, _parse_list,  # noqa: F401
                                    _try_json, one_shot)
from reasoning.bind.dates import *      # noqa: F401,F403
from reasoning.bind.text import *       # noqa: F401,F403
from reasoning.bind.text import _quoted_strings  # noqa: F401  (private, re-exported)
from reasoning.bind.finance import *    # noqa: F401,F403
from reasoning.bind.science import *    # noqa: F401,F403
from reasoning.bind.measure import *    # noqa: F401,F403
from reasoning.bind.math import *       # noqa: F401,F403
from reasoning.bind.university import *  # noqa: F401,F403
from reasoning.bind.geo import *         # noqa: F401,F403
from reasoning.bind.life import *        # noqa: F401,F403
from reasoning.bind.materials import *   # noqa: F401,F403
from reasoning.bind.discrete import *    # noqa: F401,F403
from reasoning.bind.linalg import *      # noqa: F401,F403
from reasoning.bind.chem_applied import *  # noqa: F401,F403
from reasoning.bind.biochem import *     # noqa: F401,F403
from reasoning.bind.ndim import *        # noqa: F401,F403
from reasoning.bind.tensor import *      # noqa: F401,F403
from reasoning.bind.complex_nums import *  # noqa: F401,F403
from reasoning.bind.padic import *       # noqa: F401,F403
from reasoning.bind.hypercomplex import *  # noqa: F401,F403
from reasoning.bind.ordinals import *     # noqa: F401,F403
from reasoning.bind.cardinals import *    # noqa: F401,F403
from reasoning.bind.hyperreal import *    # noqa: F401,F403
from reasoning.bind.surreal import *      # noqa: F401,F403
from reasoning.bind.crypto import *       # noqa: F401,F403
from reasoning.bind.numerical import *    # noqa: F401,F403

from reasoning.bind.game import *       # noqa: F401,F403

from reasoning.bind.quantum import *     # noqa: F401,F403

from reasoning.bind.ml import *          # noqa: F401,F403
