"""
B4-BL ("Babble") — an intelligible acoustic language for robots, styled after
Star Wars astromech droidspeak.

Package layout
--------------
generators : the locked Phase 0 sound palette (pure-tonal + rough/noise families).
phonology  : the acoustic primitive inventory (Layer 1) — the "phonemes".
lexicon    : morpheme vocabulary + compositional grammar (Layer 2).
codec      : meaning <-> gesture-sequence <-> audio round-trip.
prosody    : affect layer (Layer 3, expressive) — transforms that ride on top.

See docs in the CollabHarness vault (projects/B4-BL/docs/) for the full plan.
"""

__version__ = "0.1.0"
SR = 44100
