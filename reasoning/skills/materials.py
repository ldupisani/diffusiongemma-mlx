"""Material science — mechanical (stress/strain/Young's modulus), thermal expansion, and
crystal atomic-packing factors. Pure stdlib, zero model calls."""

from __future__ import annotations

import math

from .base import register

_APF = {"sc": math.pi / 6, "simple cubic": math.pi / 6,
        "bcc": math.pi * math.sqrt(3) / 8, "body-centered cubic": math.pi * math.sqrt(3) / 8,
        "fcc": math.pi * math.sqrt(2) / 6, "face-centered cubic": math.pi * math.sqrt(2) / 6}


@register("materials.stress", "stress = force / area (Pa)", [((100, 0.5), 200.0)])
def stress(force: float, area: float) -> float:
    return force / area


@register("materials.strain", "strain = ΔL / L (dimensionless)", [((0.002, 2), 0.001)])
def strain(delta_length: float, length: float) -> float:
    return delta_length / length


@register("materials.youngs_modulus", "Young's modulus = stress / strain (Pa)",
          [((200, 0.001), 200000.0)])
def youngs_modulus(stress_val: float, strain_val: float) -> float:
    return stress_val / strain_val


@register("materials.thermal_expansion", "length change ΔL = α·L·ΔT",
          [((1.2e-5, 10, 50), 0.006)])
def thermal_expansion(alpha: float, length: float, delta_t: float) -> float:
    return alpha * length * delta_t


@register("materials.packing_factor", "atomic packing factor of a cubic structure "
          "(sc/bcc/fcc)", [(("fcc",), 0.740480489693061), (("bcc",), 0.6801747615878316)])
def packing_factor(structure: str) -> float:
    s = structure.strip().lower()
    if s not in _APF:
        raise ValueError(f"unknown structure {structure!r} (sc/bcc/fcc)")
    return _APF[s]
