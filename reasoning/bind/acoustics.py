"""bind layer — acoustics / waves. Gated on distinctive wave/acoustics keywords
(wavelength, Hz, hertz, wave, harmonic, oscillation, sound, string, vibrating, beat, doppler).
Extracts numeric arguments with num_near for bidirectional unit parsing."""

from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403


def bind_acoustics(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    # Exclude genetics first (allele frequency, Hardy-Weinberg, etc.)
    if any(x in q for x in ["allele", "genetic", "hardy", "weinberg", "gene", "inherit"]):
        return False, None, "code", 0

    # Gate: must have distinctive wave/acoustics context
    has_wave = any(x in q for x in [
        "wavelength", "hz", "hertz", "wave", "harmonic", "oscillat", "sound",
        "string", "vibrating", "period", "beat", "doppler", "observed"
    ])
    if not has_wave:
        return False, None, "code", 0

    def ok(skill, dims, vals, **extra):
        if any(v is None for v in vals):
            return False, None, "code", 0
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    # Op 1: Harmonic frequency — most specific (requires "harmonic" keyword)
    if "harmonic" in q:
        n = num_near(query, "harmonic", "nth", "n-th", "1st", "2nd", "3rd")
        v = num_near(query, "speed", "velocity", "m/s")
        L = num_near(query, "length", "m", "meter")
        return ok("acoustics.harmonic_frequency", ["harmonic_number", "wave_speed", "string_length"],
                  [n, v, L], round=True, places=6)

    # Op 2: Wave number — specific keyword
    if "wave number" in q or "wavenumber" in q:
        wl = num_near(query, "wavelength", "m", "meter")
        return ok("acoustics.wave_number", ["wavelength"], [wl], round=True, places=6)

    # Op 3: Period — specific keyword
    if "period" in q and "t =" not in q:  # avoid "t = velocity"
        freq = num_near(query, "frequency", "hz", "hertz")
        return ok("acoustics.period", ["frequency"], [freq], round=True, places=6)

    # Ops 4 & 5: wavelength (λ = v/f) vs wave speed (v = f·λ). Both queries name speed, frequency,
    # AND wavelength, so the earlier branch would greedily claim both. The ASKED quantity is the one
    # named FIRST (the unknown leads "what is the <X> for …"), so disambiguate by first-occurrence.
    if "wavelength" in q and any(x in q for x in ["speed", "velocity", "v ="]):
        pos_wl = q.find("wavelength")
        pos_sp = min([q.find(t) for t in ("wave speed", "speed", "velocity") if t in q] or [len(q)])
        if pos_sp < pos_wl:   # speed asked first → solve for wave speed (given f and λ)
            return ok("acoustics.wave_speed", ["frequency", "wavelength"],
                      [num_near(query, "frequency", "hz", "hertz"),
                       num_near(query, "wavelength", "m", "meter")], round=True, places=6)
        # wavelength asked → solve for wavelength (given v and f)
        return ok("acoustics.wavelength", ["wave_speed", "frequency"],
                  [num_near(query, "wave speed", "speed", "velocity", "m/s"),
                   num_near(query, "frequency", "hz", "hertz")], round=True, places=6)

    # Op 6: Beat frequency — requires both "beat" and "frequency" keywords
    if "beat" in q and "frequency" in q:
        # Extract all numbers in the query and take the first two
        numbers = []
        for m in re.finditer(r'-?\d+(?:\.\d+)?', query):
            numbers.append(float(m.group()))
        if len(numbers) >= 2:
            return ok("acoustics.beat_frequency", ["frequency_1", "frequency_2"],
                      [numbers[0], numbers[1]], round=True, places=6)
        return False, None, "code", 0

    # Op 7: Doppler shift — detect on "doppler" keyword OR "observed frequency" + motion keywords
    doppler_match = "doppler" in q or ("observed" in q and any(x in q for x in ["moving", "approach", "toward"]))
    if doppler_match:
        f = num_near(query, "hz", "hertz", "frequency")
        # Sound speed: use distinctive "sound" keyword to avoid confusion with source speed
        v = num_near(query, "sound speed", "sound", "m/s")
        # Default to 343 m/s (speed of sound in air) if not explicitly provided
        if v is None:
            v = 343
        # Source speed: anchor on the FULL motion phrase ("approaching at 20") so the speed that
        # follows wins; a bare "approach" anchor sits closer to the preceding frequency ("500 Hz
        # approaching") and grabs it instead.
        vs = num_near(query, "approaching at", "moving at", "travelling at", "traveling at",
                      "source speed", "approach", "moving")
        return ok("acoustics.doppler_frequency", ["frequency", "sound_speed", "source_speed"],
                  [f, v, vs], round=True, places=6)

    return False, None, "code", 0