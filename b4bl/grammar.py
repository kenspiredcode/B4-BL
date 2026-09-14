"""
B4-BL — grammar / message structure (Ken's insight: every language uses grammar to
order words; ours should too).

Multi-word decoding was gated by word segmentation (count right only ~37%), but
per-word accuracy is ~86% WHEN the count is right. Grammar fixes the count/roles
structurally: a message follows one of a small set of SLOT TEMPLATES, each slot
restricted to a word CATEGORY (from lexicon.VOCAB_CATEGORIES). The decoder then
knows how many words to expect and, per slot, only scores category-valid words —
shrinking the search and resolving boundaries at once. No acoustic change needed.

Templates are ordered by frequency/priority. Each is a list of category names; a
slot may be optional (trailing '?'). Special single-slot templates cover bare
speech acts / interjections / one-word commands.
"""

from __future__ import annotations
from typing import List, Dict
from . import lexicon as lex

# map lexicon categories to slot roles the templates reference
CATEGORY_WORDS: Dict[str, List[str]] = {}


def _build_category_words():
    if CATEGORY_WORDS:
        return CATEGORY_WORDS
    for cat, words in lex.VOCAB_CATEGORIES.items():
        CATEGORY_WORDS[cat] = [w for w in words if w in lex.MORPHEMES]
    # hand-designed speech acts live outside VOCAB_CATEGORIES; group them
    acts = [m for m in lex.HAND_MORPHEMES if m in lex.MORPHEMES]
    CATEGORY_WORDS["act"] = acts
    return CATEGORY_WORDS


# message templates: each is a list of (category, optional?) slots.
# Kept small and ordered; the decoder tries each and scores the best fit.
TEMPLATES = [
    # bare speech act / interjection / one-word (handled as single-slot elsewhere)
    [("act", False)],
    # command: ACT? VERB OBJECT?           e.g. "(please) bring cup"
    [("act", True), ("verb", False), ("object", True)],
    # query state: QUERY PRONOUN STATE      e.g. "is your energy low"
    [("act", False), ("pronoun", False), ("state", False), ("state", True)],
    # move: (act) VERB SPATIAL              e.g. "move front"
    [("act", True), ("verb", False), ("spatial", False)],
    # subject predicate: PRONOUN VERB OBJECT?
    [("pronoun", False), ("verb", False), ("object", True)],
    # pronoun state: PRONOUN STATE          e.g. "self ok"
    [("pronoun", False), ("state", False)],
    # warning: WARN OBJECT SPATIAL
    [("act", False), ("object", False), ("spatial", True)],
]


def slot_candidates(category: str) -> List[str]:
    cw = _build_category_words()
    # some template categories map to multiple lexicon categories
    alias = {"object": ["object"], "state": ["state"], "spatial": ["spatial"],
             "verb": ["verb"], "pronoun": ["pronoun"], "act": ["act"],
             "quantity": ["quantity"], "time": ["time"]}
    cats = alias.get(category, [category])
    out = []
    for c in cats:
        out += cw.get(c, [])
    return out


def all_templates_for_length(n: int):
    """Templates whose required (non-optional) slot count <= n <= total slots."""
    res = []
    for tmpl in TEMPLATES:
        req = sum(1 for _c, opt in tmpl if not opt)
        tot = len(tmpl)
        if req <= n <= tot:
            # produce the concrete slot-category list of length n by dropping
            # optional slots from the right until it fits
            slots = list(tmpl)
            # drop optional slots (rightmost first) to reach length n
            while len(slots) > n:
                # remove the last optional slot
                idx = max((i for i, (_c, o) in enumerate(slots) if o), default=None)
                if idx is None:
                    break
                slots.pop(idx)
            if len(slots) == n:
                res.append([c for c, _o in slots])
    # de-dup
    uniq = []
    seen = set()
    for s in res:
        k = tuple(s)
        if k not in seen:
            seen.add(k); uniq.append(s)
    return uniq
