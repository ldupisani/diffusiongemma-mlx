"""Acoustics and waves — frequency, wavelength, wave speed, period, wave number, and harmonic frequencies of vibrating strings. Zero model calls.

Covers the computable pieces: wave speed (v = f·λ), wavelength (λ = v/f), period (T = 1/f),
wave number (k = 2π/λ), and harmonic frequencies of vibrating strings (f_n = n·v/(2·L)).
"""

from __future__ import annotations

import math

from .base import register


@register("acoustics.wave_speed", "wave speed v = frequency × wavelength",
          [((440, 0.78), 343.2), ((256, 1.34), 343.04), ((100, 3.43), 343.0)])
def wave_speed(frequency: float, wavelength: float) -> float:
    return round(frequency * wavelength, 6)


@register("acoustics.wavelength", "wavelength λ = wave speed / frequency",
          [((343, 440), 0.779545), ((343, 256), 1.339844), ((1500, 750), 2.0)])
def wavelength(wave_speed: float, frequency: float) -> float:
    return round(wave_speed / frequency, 6)


@register("acoustics.period", "period T = 1 / frequency",
          [((440,), 0.002273), ((50,), 0.02), ((256,), 0.003906)])
def period(frequency: float) -> float:
    return round(1 / frequency, 6)


@register("acoustics.wave_number", "wave number k = 2π / wavelength",
          [((0.78,), 8.055366), ((1.0,), 6.283185), ((2.0,), 3.141593)])
def wave_number(wavelength: float) -> float:
    return round(2 * math.pi / wavelength, 6)


@register("acoustics.harmonic_frequency", "nth harmonic of vibrating string f_n = n × speed / (2 × length)",
          [((1, 343, 0.5), 343.0), ((3, 343, 0.5), 1029.0), ((2, 400, 0.4), 1000.0)])
def harmonic_frequency(harmonic_number: int, wave_speed: float, string_length: float) -> float:
    return round(harmonic_number * wave_speed / (2 * string_length), 6)