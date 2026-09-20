"""Render the annotated packet spectrogram used by the README and demo page.

Renders one real compact-clocked v2 packet, then labels it against the word
spans the encoder itself reports. Region boundaries are derived from the format
constants in b4bl.compact_clocked, not hardcoded, so this figure cannot drift
from the packet format.

    python3 tools_render_packet_figure.py

Writes docs/img/packet-annotated.png and packet-annotated-light.png.
"""
from __future__ import annotations

import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy import signal as sps

from b4bl import clocked
from b4bl import compact_clocked as cc
from b4bl import phonology as ph
from b4bl import protocol

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "docs", "img")

# The packet to draw. A short, readable payload keeps the figure legible.
FRAME = protocol.Frame(
    sender=3, recipient=7, msg_type="MSG_ASK", seq=42,
    payload=["YOU", "ENERGY", "LOW"],
)

# Palettes matched to docs/css/style.css. GitHub serves a light and a dark
# README, so both are rendered and picked with a <picture> element.
# The spectrogram interior is magma-on-black in BOTH themes: the colormap is
# inherently dark, so label boxes drawn inside the axes must stay dark too. Only
# the surrounding figure chrome follows the theme.
PLOT_BG = "#0b0d12"
PLOT_FG = "#e6edf3"
PLOT_MUTED = "#9aa7b4"

THEMES = {
    "dark": dict(bg="#0e1116", fg="#e6edf3", muted="#9aa7b4", ink="#05080c"),
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#656d76", ink="#ffffff"),
}

REGION_COLORS = {
    "version": "#7e57c2",
    "header": "#4fc3f7",
    "payload": "#66bb6a",
    "crc": "#ffb74d",
}

REGION_NOTES = {
    "version": "format version",
    "header": "sender · recipient · type · sequence",
    "payload": "the actual message",
    "crc": "CRC32 over version + header + payload",
}


def decoded_fields(frame, words):
    """Per-word annotations showing what each metadata symbol *means*.

    Header and CRC symbols are base-32 digits drawn from ordinary vocabulary,
    so CHILD_GIRL in the sender slot is the numeral 3, not the concept. Without
    this the figure reads as a nonsensical sentence. Returns {word_index: label}.
    """
    value = {word: i for i, word in enumerate(cc.ALPHABET)}
    out = {}
    i = len(cc.VERSION)

    for label, count in (("sender", cc.ADDRESS_SYMBOLS),
                         ("recipient", cc.ADDRESS_SYMBOLS)):
        n = 0
        for k in range(count):
            n = n * len(cc.ALPHABET) + value[words[i + k]]
            out[i + k] = f"={value[words[i + k]]}"
        out[i] = f"{label} = {n}"
        i += count

    out[i] = "message type"          # literal, not a numeral
    i += 1

    seq = 0
    digits = []
    for k in range(cc.SEQUENCE_SYMBOLS):
        d = value[words[i + k]]
        digits.append(d)
        seq = seq * len(cc.ALPHABET) + d
        out[i + k] = f"digit {d}"
    out[i] = f"seq {'·'.join(map(str, digits))} base32 = {seq}"
    i += cc.SEQUENCE_SYMBOLS

    checksum_index = len(words) - cc.CRC_SYMBOLS - 1
    crc = 0
    for k, w in enumerate(words[checksum_index + 1:]):
        crc = crc * len(cc.ALPHABET) + value[w]
        out[checksum_index + 1 + k] = f"digit {value[w]}"
    out[checksum_index] = f"= 0x{crc:08x}"
    return out


def region_bounds(n_words):
    """Word-index ranges per region, derived from the format constants."""
    v = len(cc.VERSION)
    header_len = cc.ADDRESS_SYMBOLS * 2 + 1 + cc.SEQUENCE_SYMBOLS
    checksum_index = n_words - cc.CRC_SYMBOLS - 1  # the 'CKSUM' marker word
    return [
        ("version", 0, v),
        ("header", v, v + header_len),
        ("payload", v + header_len, checksum_index),
        ("crc", checksum_index, n_words),
    ]


def word_spans(words, spans):
    """Per-phoneme spans -> one (start_sec, end_sec) per word."""
    grouped = defaultdict(list)
    for s in spans:
        grouped[s["word_index"]].append(s)
    out = []
    for wi in range(len(words)):
        ss = grouped[wi]
        out.append((
            min(x["start"] for x in ss) / clocked.SR,
            max(x["slot_end"] for x in ss) / clocked.SR,
        ))
    return out


def render(theme_name):
    theme = THEMES[theme_name]
    BG = theme["bg"]
    FG = theme["fg"]
    MUTED = theme["muted"]

    words = cc.to_concepts(FRAME)
    audio, spans = clocked.encode(words, return_spans=True)
    spans_sec = word_spans(words, spans)
    duration = len(audio) / clocked.SR
    regions = region_bounds(len(words))

    freqs, times, sxx = sps.spectrogram(
        audio, fs=clocked.SR, nperseg=2048, noverlap=1536, window="hann",
    )
    top = 3600.0  # above the VHIGH band center; everything lexical is below
    keep = freqs <= top
    freqs, sxx = freqs[keep], sxx[keep]
    db = 10 * np.log10(sxx + 1e-12)
    db = np.clip(db, db.max() - 65, db.max())

    fig = plt.figure(figsize=(14, 7.2), dpi=150, facecolor=BG)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1, 3.1], hspace=0.06,
        left=0.055, right=0.985, top=0.885, bottom=0.115,
    )
    ax_bar = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=ax_bar)

    # ---- region bar -----------------------------------------------------
    ax_bar.set_facecolor(BG)
    for name, lo, hi in regions:
        start = spans_sec[lo][0]
        end = spans_sec[hi - 1][1]
        color = REGION_COLORS[name]
        ax_bar.add_patch(FancyBboxPatch(
            (start, 0.42), end - start, 0.40,
            boxstyle="round,pad=0,rounding_size=0.04",
            facecolor=color, edgecolor="none", alpha=0.85,
        ))
        mid = (start + end) / 2
        ax_bar.text(mid, 0.62, name.upper(), ha="center", va="center",
                    color=theme["ink"], fontsize=11, fontweight="bold",
                    family="monospace")
        ax_bar.text(mid, 0.26, REGION_NOTES[name], ha="center", va="center",
                    color=MUTED, fontsize=8.5)

    # CRC coverage span: every word before the CKSUM marker, i.e. version +
    # header + payload. The CRC symbols themselves are not covered.
    checksum_index = len(words) - cc.CRC_SYMBOLS - 1
    prot_start = spans_sec[0][0]
    prot_end = spans_sec[checksum_index - 1][1]
    ax_bar.annotate(
        "", xy=(prot_start, 0.10), xytext=(prot_end, 0.10),
        arrowprops=dict(arrowstyle="<->", color=REGION_COLORS["crc"], lw=1.3),
    )
    ax_bar.text((prot_start + prot_end) / 2, 0.015,
                "protected by CRC32", ha="center", va="bottom",
                color=REGION_COLORS["crc"], fontsize=8.5, style="italic")

    ax_bar.set_ylim(0, 1)
    ax_bar.axis("off")

    # ---- spectrogram ----------------------------------------------------
    ax.set_facecolor(PLOT_BG)
    ax.pcolormesh(times, freqs, db, cmap="magma", shading="gouraud",
                  rasterized=True)

    # Band centers: the lexical pitch grid. Labels sit inside the axes so they
    # cannot be clipped by the figure edge.
    for band, hz in sorted(ph.BAND_CENTER.items(), key=lambda kv: kv[1]):
        if hz > top:
            continue
        ax.axhline(hz, color=PLOT_FG, lw=0.5, alpha=0.18, ls=(0, (4, 4)))
        ax.text(duration - 0.06, hz, band.name, color=PLOT_MUTED, fontsize=7.5,
                va="center", ha="right", family="monospace",
                bbox=dict(boxstyle="round,pad=0.18", facecolor=PLOT_BG,
                          edgecolor="none", alpha=0.75))

    decoded = decoded_fields(FRAME, words)

    # Word labels, alternating height so neighbours don't collide. A leader
    # line ties each label to the slot it names.
    for i, (word, (start, end)) in enumerate(zip(words, spans_sec)):
        region = next(n for n, lo, hi in regions if lo <= i < hi)
        color = REGION_COLORS[region]
        ax.axvspan(start, end, color=color, alpha=0.085, lw=0)
        mid = (start + end) / 2
        y = top * (0.945 if i % 2 == 0 else 0.86)
        ax.plot([mid, mid], [top * 0.815, y - top * 0.022],
                color=color, lw=0.6, alpha=0.45, solid_capstyle="butt")
        ax.text(mid, y, word, ha="center", va="center", color=color,
                fontsize=7.4, family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=PLOT_BG,
                          edgecolor="none", alpha=0.85))

        # What the symbol means, for fields that are base-32 numerals rather
        # than words. Without this the header reads as a nonsense sentence.
        note = decoded.get(i)
        if note:
            # Wide summary labels are left-anchored so they extend into the
            # empty span to their right instead of over the next word's label.
            wide = len(note) > 18
            ax.text(mid if not wide else start, y - top * 0.052, note,
                    ha="center" if not wide else "left", va="center",
                    color=PLOT_FG, fontsize=6.6, family="monospace",
                    alpha=0.95, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.16", facecolor=PLOT_BG,
                              edgecolor=color, linewidth=0.5, alpha=0.95))

    # Clock markers. These are the bright full-band vertical strokes; they
    # delimit words, and the receiver will not wake until two of them are
    # separated by a legal symbol-clock interval.
    # The markers are already the brightest thing in the image, so they are
    # called out with tick glyphs under the axis rather than overdrawn lines.
    marker_sec = np.asarray(clocked.marker_positions(audio),
                            dtype=float) / clocked.SR
    for m in marker_sec:
        ax.plot([m], [0], marker="^", color=MUTED, markersize=3.6, alpha=0.8,
                clip_on=False, zorder=5)
    if len(marker_sec) >= 2:
        a0, a1 = marker_sec[0], marker_sec[1]
        ax.annotate(
            "", xy=(a0, top * 0.055), xytext=(a1, top * 0.055),
            arrowprops=dict(arrowstyle="<->", color=PLOT_FG, lw=1.0, alpha=0.8),
        )
        ax.text((a0 + a1) / 2, top * 0.085,
                f"SYNC interval  {a1 - a0:.2f}s", ha="center", va="bottom",
                color=PLOT_FG, fontsize=7.6, family="monospace", alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=PLOT_BG,
                          edgecolor="none", alpha=0.75))

    ax.set_xlim(0, duration)
    ax.set_ylim(0, top)
    ax.set_xlabel(
        "seconds        ▲ = clock marker; the receiver stays asleep until two "
        "are a legal interval apart",
        color=MUTED, fontsize=9)
    ax.set_ylabel("Hz", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_alpha(0.35)

    # ---- titles ---------------------------------------------------------
    fig.text(0.055, 0.955, "Anatomy of a B4-BL packet", color=FG,
             fontsize=16, fontweight="bold", ha="left")
    fig.text(0.055, 0.917,
             f"{cc.PROFILE}  ·  {len(words)} words  ·  {duration:.2f} s  ·  "
             f'payload "{" ".join(FRAME.payload)}"',
             color=MUTED, fontsize=9, ha="left", family="monospace")
    fig.text(0.055, 0.028,
             "Only PAYLOAD carries meaning. Header and CRC symbols are "
             "base-32 numerals that reuse vocabulary words as digits — "
             "CHILD_GIRL is 3, DOCK is 28.",
             color=MUTED, fontsize=8.6, ha="left", style="italic")

    os.makedirs(OUT_DIR, exist_ok=True)
    # PNG only: the spectrogram mesh is rasterized, so an SVG is ~880 KB with
    # no fidelity gain over the 150 dpi bitmap.
    suffix = "" if theme_name == "dark" else f"-{theme_name}"
    png = os.path.join(OUT_DIR, f"packet-annotated{suffix}.png")
    fig.savefig(png, facecolor=BG)
    plt.close(fig)

    print(f"wrote {png}  ({os.path.getsize(png) / 1e3:.0f} KB)")
    return words, spans_sec, regions, duration


def main():
    for theme_name in THEMES:
        words, spans_sec, regions, duration = render(theme_name)
    print(f"  {len(words)} words, {duration:.2f}s")
    for name, lo, hi in regions:
        print(f"  {name:8} words {lo:2}-{hi - 1:2}  "
              f"{spans_sec[lo][0]:5.2f}-{spans_sec[hi - 1][1]:5.2f}s  "
              f"{' '.join(words[lo:hi])}")


if __name__ == "__main__":
    main()
