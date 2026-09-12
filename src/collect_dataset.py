#!/usr/bin/env python3
"""
B4-BL — collect a real-world labeled dataset (Option A: Mac plays + records).

Covers the vocabulary broadly with prosody variation (what the runtime actually
varies), plus multi-morpheme messages for co-articulation and in-context word-gap
segmentation. Each recording is saved with its exact concept label + the channel
tag, for testing the current model and retraining a hardened V2.

Usage:
  python3 src/collect_dataset.py --channel builtin
  python3 src/collect_dataset.py --channel airplay_office   # after switching output

Resumable: appends to recordings/manifest.jsonl and numbers files by channel, so a
second run (e.g. AirPlay) adds to the same dataset without clobbering round 1.
"""
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from b4bl import capture, codec, lexicon as lex, prosody, generators as gen
from scipy.io import wavfile

REC_DIR = os.path.join(os.path.dirname(__file__), "..", "recordings")
MANIFEST = os.path.join(REC_DIR, "manifest.jsonl")

# prosody spread the runtime actually produces
PROSODIES = [
    ("neutral",   prosody.Prosody(0.8, 0.3)),
    ("uncertain", prosody.Prosody(0.3, 0.2)),
    ("urgent",    prosody.Prosody(0.9, 0.9)),
    ("calm",      prosody.Prosody(0.9, 0.05)),
]


def build_message_set(rng):
    """List of (concepts, prosody_name) to emit."""
    allc = [c for c in lex.MORPHEMES if c not in lex.ALIASES]
    items = []
    # every morpheme, each prosody
    for c in allc:
        for pname, _ in PROSODIES:
            items.append(([c], pname))
    # multi-morpheme messages (2-5 words), varied prosody
    for _ in range(300):
        L = rng.integers(2, 6)
        msg = list(rng.choice(allc, size=L, replace=False))
        pname = PROSODIES[rng.integers(0, len(PROSODIES))][0]
        items.append((msg, pname))
    rng.shuffle(items)
    return items


def load_done(channel):
    """Set of (concepts-tuple, prosody, channel) already recorded, for resume."""
    done = set()
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((tuple(r["concepts"]), r["prosody"], r["channel"]))
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="builtin",
                    help="tag for the current audio output (builtin / airplay_office / ...)")
    ap.add_argument("--limit", type=int, default=0, help="cap emissions (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latency", type=float, default=0.0,
                    help="output latency headroom in seconds (AirPlay ~2.2)")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="software input gain for quiet channels")
    args = ap.parse_args()

    os.makedirs(REC_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    capture.set_channel_profile(latency=args.latency, gain=args.gain)

    print("=== self-test ===")
    if not capture.self_test():
        print("ABORT: channel not live. Check volume / output device / mic permission.")
        sys.exit(1)

    prmap = dict(PROSODIES)
    items = build_message_set(rng)
    if args.limit:
        items = items[: args.limit]
    done = load_done(args.channel)
    pending = [(m, p) for (m, p) in items if (tuple(m), p, args.channel) not in done]

    print(f"=== collecting {len(pending)} emissions on channel '{args.channel}' "
          f"({len(items) - len(pending)} already done) ===")
    est_min = len(pending) * 2.5 / 60
    print(f"estimated ~{est_min:.0f} min")

    MIN_RMS = 0.006     # reject captures too quiet to be usable (transient channel
                        # dips). Retry once before skipping — protects an unattended
                        # long run from silently saving silence with a good label.
    n_ok = n_skip = n_quiet = 0
    with open(MANIFEST, "a") as mf:
        for i, (msg, pname) in enumerate(pending):
            audio = codec.encode(msg, prmap[pname])
            seg = None
            for attempt in range(2):
                rec = capture.play_and_record(audio)
                cand = capture.find_message(rec)
                if cand is not None and float(np.sqrt(np.mean(cand ** 2))) >= MIN_RMS:
                    seg = cand
                    break
            if seg is None:
                if cand is None:
                    n_skip += 1
                else:
                    n_quiet += 1     # captured but too quiet after a retry
                continue
            fn = f"{args.channel}_{i:05d}.wav"
            wavfile.write(os.path.join(REC_DIR, fn), gen.SR,
                          (np.clip(seg, -1, 1) * 32767).astype(np.int16))
            mf.write(json.dumps({"file": fn, "concepts": msg,
                                 "prosody": pname, "channel": args.channel}) + "\n")
            mf.flush()
            n_ok += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(pending)}  ok={n_ok} skip={n_skip} quiet={n_quiet}",
                      flush=True)
            time.sleep(0.05)
    print(f"DONE: {n_ok} recorded, {n_skip} no-sync, {n_quiet} too-quiet. "
          f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
