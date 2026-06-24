"""Complex numbers — modulus, argument, conjugate, arithmetic, powers. Built on Python's native
complex type; zero model calls. Numbers are written 'a+bi' (or 'a+bj'); results render as 'a+bi'."""

from __future__ import annotations

import cmath
import re

from .base import register


def _parse(s: str) -> complex:
    t = s.strip().replace(" ", "").replace("i", "j")
    t = re.sub(r"(?<![\d.])j", "1j", t)   # bare j / after a sign → 1j (so '1+j' parses)
    return complex(t)


def _fmt(z: complex, places: int = 4) -> str:
    re_, im = round(z.real, places), round(z.imag, places)
    re_ = int(re_) if re_ == int(re_) else re_
    im = int(im) if im == int(im) else im
    if im == 0:
        return str(re_)
    mag = "" if abs(im) == 1 else str(abs(im))
    imag = f"{'-' if im < 0 else ''}{mag}i"
    if re_ == 0:
        return imag
    return f"{re_}{'+' if im >= 0 else '-'}{mag}i"


@register("cplx.modulus", "modulus |z| of a complex number",
          [(("3+4i",), 5.0), (("1+1i",), 1.4142135623730951)])
def modulus(z: str) -> float:
    return abs(_parse(z))


@register("cplx.argument", "argument (phase angle, radians) of a complex number",
          [(("3+4i",), 0.9272952180016122), (("0+1i",), 1.5707963267948966)])
def argument(z: str) -> float:
    return cmath.phase(_parse(z))


@register("cplx.conjugate", "complex conjugate", [(("3+4i",), "3-4i"), (("2-5i",), "2+5i")])
def conjugate(z: str) -> str:
    return _fmt(_parse(z).conjugate())


@register("cplx.multiply", "product of two complex numbers",
          [(("1+1i", "1-1i"), "2"), (("2+3i", "1+1i"), "-1+5i")])
def multiply(a: str, b: str) -> str:
    return _fmt(_parse(a) * _parse(b))


@register("cplx.divide", "quotient of two complex numbers", [(("1+1i", "1-1i"), "i")])
def divide(a: str, b: str) -> str:
    return _fmt(_parse(a) / _parse(b))


@register("cplx.power", "complex number raised to an integer power (De Moivre)",
          [(("1+1i", 2), "2i"), (("1+1i", 3), "-2+2i")])
def power(z: str, n: int) -> str:
    return _fmt(_parse(z) ** n)
