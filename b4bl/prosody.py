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

        # urgency tightens timing (shorter, snappier)
        dur = dur * (1.0 - 0.35 * urg)

        # low confidence -> more vibrato (tremulous) and gentler onset
        vib_rate, vib_depth = vib if vib else (7, 0.03)
        vib_depth = vib_depth + (1 - conf) * 0.05
        vib = (vib_rate, vib_depth)
        if conf < 0.4 and env == "stab":
            env = "swell"  # hesitant sounds don't stab

        # urgency exaggerates the contour swing; low confidence shrinks it
        exagg = 1.0 + 0.4 * urg - 0.3 * (1 - conf)
        if len(points) >= 2:
            center = sum(p[1] for p in points) / len(points)
            points = [(f, center + (hz - center) * exagg) for (f, hz) in points]

        return points, dur, env, vib


# a few named presets, purely for convenience / demos
NEUTRAL = Prosody(confidence=0.8, urgency=0.3)
UNCERTAIN = Prosody(confidence=0.25, urgency=0.2)
URGENT = Prosody(confidence=0.9, urgency=0.95)
CALM = Prosody(confidence=0.9, urgency=0.05)
