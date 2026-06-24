"""Tensor maths — rank/shape, trace/contraction, Frobenius norm, Kronecker & outer products.
NumPy-backed (deterministic), zero model calls. Tensors are python-list strings ('[[1, 2],
[3, 4]]'), the same convention as linalg matrices."""

from __future__ import annotations

import ast

import numpy as np

from .base import register


def _arr(s: str):
    return np.array(ast.literal_eval(s))


@register("tensor.rank", "rank (number of dimensions / axes) of a tensor",
          [(("[[1, 2], [3, 4]]",), 2), (("[1, 2, 3]",), 1)])
def rank(tensor: str) -> int:
    return _arr(tensor).ndim


@register("tensor.shape", "shape of a tensor (size along each axis)",
          [(("[[1, 2, 3], [4, 5, 6]]",), "(2, 3)"), (("[1, 2, 3]",), "(3,)")])
def shape(tensor: str) -> str:
    return str(_arr(tensor).shape)


@register("tensor.trace", "trace (sum of the diagonal) of a square matrix",
          [(("[[1, 2], [3, 4]]",), 5), (("[[2, 0, 0], [0, 3, 0], [0, 0, 4]]",), 9)])
def trace(tensor: str):
    return np.trace(_arr(tensor)).item()


@register("tensor.frobenius_norm", "Frobenius norm of a tensor (√ sum of squares)",
          [(("[[3, 4]]",), 5.0), (("[[1, 2], [3, 4]]",), 5.477225575051661)])
def frobenius_norm(tensor: str) -> float:
    return float(np.linalg.norm(_arr(tensor)))


@register("tensor.kronecker", "Kronecker product of two matrices",
          [(("[[1, 2], [3, 4]]", "[[0, 1], [1, 0]]"),
            "[[0, 1, 0, 2], [1, 0, 2, 0], [0, 3, 0, 4], [3, 0, 4, 0]]")])
def kronecker(a: str, b: str) -> str:
    return str(np.kron(_arr(a), _arr(b)).tolist())


@register("tensor.outer", "outer (tensor) product of two vectors",
          [(("[1, 2]", "[3, 4]"), "[[3, 4], [6, 8]]")])
def outer(u: str, v: str) -> str:
    return str(np.outer(_arr(u), _arr(v)).tolist())
