"""Acoustics and waves — frequency, wavelength, wave speed, period, wave number, harmonic frequencies,
beat frequency, and Doppler shift. Model-free, 0 calls."""

from reasoning.bind import bind_acoustics
from reasoning.compiled import dispatch

META = {
    "id": "acoustics", "query_type": "acoustics",
    "description": "Acoustics and waves: wave speed (v = f·λ), wavelength (λ = v/f), period (T = 1/f), "
                   "wave number (k = 2π/λ), harmonic frequencies of vibrating strings (f_n = n·v/(2·L)), "
                   "beat frequency (f_beat = |f1 - f2|), and Doppler shift for an approaching source (f_obs = f·v/(v - v_s))",
    "example_queries": [
        "What is the wave speed for a frequency of 440 Hz and wavelength of 0.78 m?",
        "What is the wavelength for a wave speed of 343 m/s and frequency of 440 Hz?",
        "What is the period for a frequency of 440 Hz?",
        "What is the wave number for a wavelength of 0.78 m?",
        "What is the 3rd harmonic frequency of a string with wave speed 343 m/s and length 0.5 m?",
        "Calculate wavelength given speed 343 m/s and frequency 256 Hz.",
        "Find the period of an oscillation with frequency 50 Hz.",
        "What is the wave number for a wavelength of 2 m?",
        "What is the beat frequency of 440 Hz and 445 Hz?",
        "Calculate the beat frequency between 1000 Hz and 980 Hz.",
        "What is the Doppler shift for a source at 500 Hz approaching at 20 m/s with sound speed 343 m/s?",
        "Find the observed frequency for a 440 Hz source moving toward the observer at 10 m/s.",
    ],
    "keywords": ["wavelength", "frequency", "wave speed", "period", "wave number", "harmonic",
                 "oscillation", "sound", "Hz", "hertz", "string", "beat", "doppler", "observed"],
    "model_free": True,
}


def precondition(query):
    return bind_acoustics(query)[0]


def run(query):
    return dispatch(bind_acoustics(query),
                    parse_message="could not parse the wave or acoustics query")