"""Machine learning theory — losses, activations, classifier metrics, impurity, and a gradient
step. Zero model calls.

Covers the everyday quantitative pieces: sigmoid/ReLU/softmax, MSE / MAE / binary cross-entropy
losses, accuracy/precision/recall/F1 from confusion counts, Shannon entropy and Gini impurity
of a label distribution, and one gradient-descent update θ ← θ − lr·∇.
"""

from __future__ import annotations

import math
import re

from .base import register


def _nums(s):
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", str(s))]


@register("ml.sigmoid", "logistic sigmoid 1/(1+e^-x)",
          [((0,), 0.5), ((2,), 0.880797), ((-1,), 0.268941)])
def sigmoid(x: float) -> float:
    return round(1 / (1 + math.exp(-float(x))), 6)


@register("ml.relu", "rectified linear unit max(0, x)", [((-3,), 0.0), ((2.5,), 2.5)])
def relu(x: float) -> float:
    return round(max(0.0, float(x)), 6)


@register("ml.softmax", "softmax of a vector of logits",
          [(("1, 2, 3",), "0.09, 0.2447, 0.6652"), (("0, 0",), "0.5, 0.5")])
def softmax(logits) -> str:
    xs = _nums(logits)
    mx = max(xs)
    exps = [math.exp(x - mx) for x in xs]
    s = sum(exps)
    return ", ".join(str(round(e / s, 4)) for e in exps)


@register("ml.mse", "mean squared error between true and predicted values",
          [(("1,2,3", "1,2,4"), 0.333333), (("1,2", "1,2"), 0.0)])
def mse(y_true, y_pred) -> float:
    a, b = _nums(y_true), _nums(y_pred)
    return round(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a), 6)


@register("ml.cross_entropy", "binary cross-entropy between labels and predicted probabilities",
          [(("1, 0", "0.9, 0.2"), 0.164252), (("1", "0.5"), 0.693147)])
def cross_entropy(y_true, y_pred) -> float:
    a, b = _nums(y_true), _nums(y_pred)
    return round(-sum(y * math.log(p) + (1 - y) * math.log(1 - p)
                      for y, p in zip(a, b)) / len(a), 6)


@register("ml.accuracy", "accuracy (tp+tn)/(tp+tn+fp+fn)", [((8, 89, 2, 1), 0.97)])
def accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    return round((tp + tn) / (tp + tn + fp + fn), 6)


@register("ml.precision", "precision tp/(tp+fp)", [((8, 2), 0.8)])
def precision(tp: int, fp: int) -> float:
    return round(tp / (tp + fp), 6)


@register("ml.recall", "recall tp/(tp+fn)", [((8, 1), 0.888889)])
def recall(tp: int, fn: int) -> float:
    return round(tp / (tp + fn), 6)


@register("ml.f1", "F1 score (harmonic mean of precision and recall)", [((8, 2, 1), 0.842105)])
def f1(tp: int, fp: int, fn: int) -> float:
    p, r = tp / (tp + fp), tp / (tp + fn)
    return round(2 * p * r / (p + r), 6)


@register("ml.entropy", "Shannon entropy (bits) of a probability distribution",
          [(("0.5, 0.5",), 1.0), (("1, 0",), 0.0), (("0.25,0.25,0.25,0.25",), 2.0)])
def entropy(probs) -> float:
    return round(-sum(p * math.log2(p) for p in _nums(probs) if p > 0), 6)


@register("ml.gini", "Gini impurity of a probability distribution",
          [(("0.5, 0.5",), 0.5), (("1, 0",), 0.0)])
def gini(probs) -> float:
    return round(1 - sum(p * p for p in _nums(probs)), 6)


@register("ml.gradient_step", "one gradient-descent update θ - lr·grad",
          [((1.0, 0.5, 0.1), 0.95), ((2.0, -1.0, 0.5), 2.5)])
def gradient_step(theta: float, grad: float, lr: float) -> float:
    return round(float(theta) - float(lr) * float(grad), 6)
