"""
B4-BL Layer 3 (expressive) — prosody / affect.

Affect is NOT vocabulary. It rides on top of an exact payload and only perturbs
the EXPRESSIVE dimensions (exact pitch within band, contour exaggeration, tempo,
gap timing, vibrato, onset). The lexical features a decoder keys on are never
touched, so the same phoneme stays the same phoneme whether cheerful or annoyed.

Two continuous knobs, matching the plan:
  confidence : 0..1   (low -> tentative: slower onset, more vibrato, smaller swings)
  urgency    : 0..1   (high -> tighter timing, sharper contours)

A Prosody is applied per-phoneme by Phoneme.render(). It returns modified
(points, dur, env, vib).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Prosody:
    confidence: float = 0.8
    urgency: float = 0.3

    def apply(self, phoneme, points, dur, env, vib):
        conf = max(0.0, min(1.0, self.confidence))
        urg = max(0.0, min(1.0, self.urgency))

        # --- timing: urgency makes it MUCH faster; low confidence drags it out ---
        # range roughly 0.5x (very urgent) to 1.8x (very hesitant) of base duration.
        dur = dur * (1.0 - 0.5 * urg) * (1.0 + 0.7 * (1 - conf))

        # --- vibrato: low confidence = strongly tremulous; urgency = tighter ---
        vib_rate, vib_depth = vib if vib else (7, 0.03)
        vib_depth = vib_depth + (1 - conf) * 0.18          # up to ~5x baseline wobble
        vib_rate = vib_rate * (1.0 + 0.5 * urg)
        vib = (vib_rate, vib_depth)

        # --- envelope: urgent stabs, hesitant swells ---
        if urg > 0.6:
            env = "stab"
        elif conf < 0.45:
            env = "swell"

        # --- contour swing: urgency exaggerates a lot; low confidence flattens ---
        exagg = 1.0 + 1.0 * urg - 0.6 * (1 - conf)
        exagg = max(0.4, exagg)

        # --- brightness: urgency shifts everything up; low confidence droops it ---
        # multiplicative pitch shift, kept modest so the BAND (lexical) is preserved.
        bright = 1.0 + 0.12 * urg - 0.10 * (1 - conf)

        if len(points) >= 2:
            center = sum(p[1] for p in points) / len(points)
            points = [(f, (center + (hz - center) * exagg) * bright) for (f, hz) in points]
        elif points:
            points = [(f, hz * bright) for (f, hz) in points]

        return points, dur, env, vib


# a few named presets, purely for convenience / demos
NEUTRAL = Prosody(confidence=0.8, urgency=0.3)
UNCERTAIN = Prosody(confidence=0.25, urgency=0.2)
URGENT = Prosody(confidence=0.9, urgency=0.95)
CALM = Prosody(confidence=0.9, urgency=0.05)
