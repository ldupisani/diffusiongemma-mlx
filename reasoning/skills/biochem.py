"""Biochemistry — DNA/RNA sequence operations and enzyme/buffer equations. Pure stdlib (math).
Zero model calls."""

from __future__ import annotations

import math

from .base import register

_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}


@register("biochem.gc_content", "GC content of a DNA sequence (percent)",
          [(("GGCC",), 100.0), (("ATGC",), 50.0)])
def gc_content(dna: str) -> float:
    s = dna.upper()
    return sum(b in "GC" for b in s) / len(s) * 100


@register("biochem.reverse_complement", "reverse complement of a DNA sequence",
          [(("ATGC",), "GCAT"), (("AAGG",), "CCTT")])
def reverse_complement(dna: str) -> str:
    return "".join(_COMPLEMENT[b] for b in dna.upper()[::-1])


@register("biochem.transcribe", "transcribe DNA to mRNA (T → U)",
          [(("ATGC",), "AUGC"), (("TTAA",), "UUAA")])
def transcribe(dna: str) -> str:
    return dna.upper().replace("T", "U")


@register("biochem.henderson_hasselbalch", "buffer pH = pKa + log10([base]/[acid])",
          [((4.76, 1, 1), 4.76), ((4.76, 10, 1), 5.76)])
def henderson_hasselbalch(pka: float, base: float, acid: float) -> float:
    return pka + math.log10(base / acid)


@register("biochem.michaelis_menten", "reaction rate v = Vmax·[S]/(Km + [S])",
          [((100, 5, 5), 50.0), ((100, 5, 15), 75.0)])
def michaelis_menten(vmax: float, km: float, s: float) -> float:
    return vmax * s / (km + s)
