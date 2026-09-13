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
    ap.add_argument("--input-device", default=None,
                    help="pin the recording mic by name substring or index "
                         "(e.g. 'MacBook Pro Microphone', 'AUKEY') so a paired "
                         "speaker doesn't hijack the input")
    ap.add_argument("--output-device", default=None,
                    help="pin the playback speaker by name substring or index "
                         "(e.g. 'MacBook Pro Speakers') so an unattended run picks "
                         "it without changing system settings")
    args = ap.parse_args()

    os.makedirs(REC_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    capture.set_channel_profile(latency=args.latency, gain=args.gain,
                                input_device=args.input_device,
                                output_device=args.output_device)

    print("=== self-test ===")
    # run the self-test in a subprocess too, so a dead channel can't hang startup.
    import subprocess
    emit = os.path.join(os.path.dirname(__file__), "_emit_one.py")
    base_env = dict(os.environ)
    if args.input_device:
        base_env["B4BL_INPUT_DEVICE"] = args.input_device
    if args.output_device:
        base_env["B4BL_OUTPUT_DEVICE"] = args.output_device
    try:
        st = subprocess.run(
            [sys.executable, emit, "/tmp/b4bl_selftest.wav",
             str(args.latency), str(args.gain), "__SELFTEST__"],
            env=base_env, timeout=args.latency * 2 + 12)
        if st.returncode != 0:
            print("ABORT: channel not live (volume / output device / mic permission).")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ABORT: self-test hung (output device wedged?). Re-select output & retry.")
        sys.exit(1)

    items = build_message_set(rng)
    if args.limit:
        items = items[: args.limit]
    done = load_done(args.channel)
    pending = [(m, p) for (m, p) in items if (tuple(m), p, args.channel) not in done]

    # start file numbering ABOVE any existing files for this channel, so a resume
    # run never overwrites recordings from a previous run.
    import glob, re
    existing = glob.glob(os.path.join(REC_DIR, f"{args.channel}_*.wav"))
    next_idx = 0
    for p in existing:
        m = re.search(rf"{re.escape(args.channel)}_(\d+)\.wav$", p)
        if m:
            next_idx = max(next_idx, int(m.group(1)) + 1)

    print(f"=== collecting {len(pending)} emissions on channel '{args.channel}' "
          f"({len(items) - len(pending)} already done) ===")
    est_min = len(pending) * 2.5 / 60
    print(f"estimated ~{est_min:.0f} min")

    # Each emission runs in a SUBPROCESS. A hung audio call (AirPlay drop wedging
    # CoreAudio) can't be recovered in-process — a thread timeout leaves PortAudio
    # wedged for the next call too. Killing a subprocess makes the OS reclaim the
    # audio state, so the run truly continues. This is the robust fix after two
    # in-process watchdog attempts stalled the run.
    import subprocess, signal
    emit = os.path.join(os.path.dirname(__file__), "_emit_one.py")
    emission_timeout = args.latency * 2 + 12   # generous per-emission cap (seconds)

    n_ok = n_skip = n_quiet = n_hang = 0
    hang_streak = 0
    with open(MANIFEST, "a") as mf:
        for i, (msg, pname) in enumerate(pending):
            fn = f"{args.channel}_{next_idx:05d}.wav"
            out = os.path.join(REC_DIR, fn)
            env = dict(base_env, B4BL_PROSODY=pname)
            cmd = [sys.executable, emit, out, str(args.latency), str(args.gain)] + list(msg)
            try:
                p = subprocess.run(cmd, env=env, timeout=emission_timeout,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = -1   # hung: subprocess killed by timeout, OS frees audio state

            if rc == -1:
                n_hang += 1
                hang_streak += 1
                if hang_streak >= 4:
                    print(f"  [abort] {hang_streak} hangs in a row at {i}; channel "
                          f"likely dropped. Stopping cleanly — fix output & re-run "
                          f"to resume.", flush=True)
                    break
                continue
            hang_streak = 0
            if rc == 0 and os.path.exists(out):
                mf.write(json.dumps({"file": fn, "concepts": msg,
                                     "prosody": pname, "channel": args.channel}) + "\n")
                mf.flush()
                n_ok += 1
                next_idx += 1
            elif rc == 2:
                n_skip += 1     # no sync or too quiet
            else:
                n_skip += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(pending)}  ok={n_ok} skip={n_skip} hang={n_hang}",
                      flush=True)
    print(f"DONE: {n_ok} recorded, {n_skip} no-sync/quiet, {n_hang} hangs. "
          f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
