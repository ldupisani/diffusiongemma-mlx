"""Acoustics and waves — frequency, wavelength, wave speed, period, wave number, and harmonic
frequencies of vibrating strings. Model-free, 0 calls."""

from reasoning.bind import bind_acoustics
from reasoning.compiled import dispatch

META = {
    "id": "acoustics", "query_type": "acoustics",
    "description": "Acoustics and waves: wave speed (v = f·λ), wavelength (λ = v/f), period (T = 1/f), "
                   "wave number (k = 2π/λ), and harmonic frequencies of vibrating strings (f_n = n·v/(2·L))",
    "example_queries": [
        "What is the wave speed for a frequency of 440 Hz and wavelength of 0.78 m?",
        "What is the wavelength for a wave speed of 343 m/s and frequency of 440 Hz?",
        "What is the period for a frequency of 440 Hz?",
        "What is the wave number for a wavelength of 0.78 m?",
        "What is the 3rd harmonic frequency of a string with wave speed 343 m/s and length 0.5 m?",
        "Calculate wavelength given speed 343 m/s and frequency 256 Hz.",
        "Find the period of an oscillation with frequency 50 Hz.",
        "What is the wave number for a wavelength of 2 m?",
    ],
    "keywords": ["wavelength", "frequency", "wave speed", "period", "wave number", "harmonic",
                 "oscillation", "sound", "Hz", "hertz", "string"],
    "model_free": True,
}


def precondition(query):
    return bind_acoustics(query)[0]


def run(query):
    return dispatch(bind_acoustics(query),
                    parse_message="could not parse the wave or acoustics query")