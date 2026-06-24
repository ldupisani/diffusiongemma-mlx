"""N-dimensional shapes — n-ball volume, n-sphere surface, and hypercube measures. Pure stdlib
(math.gamma generalizes the factorial in the volume formula). Zero model calls."""

from __future__ import annotations

import math

from .base import register


@register("ndim.ball_volume", "volume of an n-ball (n-dim solid sphere) of radius r: "
          "π^(n/2)/Γ(n/2+1)·r^n", [((3, 1), 4.188790204786391), ((2, 1), 3.141592653589793)])
def ball_volume(n: int, r: float) -> float:
    return math.pi ** (n / 2) / math.gamma(n / 2 + 1) * r ** n


@register("ndim.sphere_surface", "surface area of an n-sphere (the (n-1)-dim boundary of the "
          "n-ball) of radius r", [((3, 1), 12.566370614359176), ((2, 1), 6.283185307179586)])
def sphere_surface(n: int, r: float) -> float:
    return 2 * math.pi ** (n / 2) / math.gamma(n / 2) * r ** (n - 1)


@register("ndim.hypercube_volume", "volume of an n-cube of side s (s^n)", [((4, 2), 16), ((3, 2), 8)])
def hypercube_volume(n: int, s: float) -> float:
    return s ** n


@register("ndim.hypercube_diagonal", "space diagonal of an n-cube of side s (s·√n)",
          [((4, 2), 4.0), ((3, 1), 1.7320508075688772)])
def hypercube_diagonal(n: int, s: float) -> float:
    return s * math.sqrt(n)


@register("ndim.hypercube_vertices", "number of vertices of an n-cube (2^n)",
          [((4,), 16), ((10,), 1024)])
def hypercube_vertices(n: int) -> int:
    return 2 ** n


@register("ndim.hypercube_edges", "number of edges of an n-cube (n·2^(n-1))",
          [((4,), 32), ((3,), 12)])
def hypercube_edges(n: int) -> int:
    return n * 2 ** (n - 1)
