"""Number-base (radix) conversion — one skill spanning binary/octal/decimal/hex.

Like units.convert, a single convert(value, from_base, to_base) replaces the pairwise
explosion of dec→bin, bin→dec, dec→hex, … functions. Returns a string (uppercase for
hex); decimal results are the plain integer as a string.
"""

from __future__ import annotations

from .base import register

_DIGITS = "0123456789ABCDEF"


@register("radix.convert", "convert an integer between number bases (2/8/10/16)",
          [(("255", 10, 2), "11111111"), (("FF", 16, 10), "255"),
           (("64", 10, 8), "100"), (("1010", 2, 10), "10")])
def convert(value: str, from_base: int, to_base: int) -> str:
    n = int(str(value), int(from_base))  # parse value in its source base
    to_base = int(to_base)
    if to_base == 10:
        return str(n)
    if n == 0:
        return "0"
    neg, n = n < 0, abs(n)
    out = ""
    while n:
        out = _DIGITS[n % to_base] + out
        n //= to_base
    return ("-" if neg else "") + out
