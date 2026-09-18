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
    ("neutral",   prosody.NEUTRAL),
    ("uncertain", prosody.UNCERTAIN),
    ("urgent",    prosody.URGENT),
    ("calm",      prosody.CALM),
]


def build_message_set(rng, clocked=False):
    """List of (concepts, prosody_name) to emit."""
    allc = [c for c in lex.MORPHEMES if c not in lex.ALIASES]
    if clocked:
        from b4bl.clocked import vocabulary
        allc = list(vocabulary())
    items = []
    # Historical corpus: every morpheme in every prosody. Clocked corpus: one
    # rotating prosody per singleton, leaving the majority of cases multi-word.
    for c in allc:
        for pname, _ in (PROSODIES if not clocked else [PROSODIES[len(items) % 4]]):
            items.append(([c], pname))
    # multi-morpheme messages (2-5 words), varied prosody
    for _ in range(400 if clocked else 300):
        L = rng.integers(2, 6)
        msg = list(rng.choice(allc, size=L, replace=clocked))
        pname = PROSODIES[rng.integers(0, len(PROSODIES))][0]
        items.append((msg, pname))
    if clocked:
        # Include natural messages, numbers, and adversarial boundary pairs.
        phrases = [msg for _, msg in lex.example_sentences() if all(c in allc for c in msg)]
        phrases += [["SELF", "SELF"], ["GIVE"], ["DENY", "IT"], ["SEARCH"],
                    ["NUM", "D4", "D2"], ["ACK", "ACK", "ACK"]]
        for msg in phrases:
            for pname, _ in PROSODIES:
                items.append((msg, pname))
    rng.shuffle(items)
    return items


def build_compact_packet_set(rng):
    """Deterministic protected frames spanning payload words, lengths, and headers."""
    from b4bl import clocked, compact_clocked, protocol
    allowed = [word for word in clocked.vocabulary()
               if word not in ('SYNC', 'CKSUM')]
    items = []
    payloads = []
    for word in allowed:
        payloads.append(['NUM', word] if word.startswith('D') and word[1:].isdigit()
                        else [word])
    for _ in range(400):
        payload = list(rng.choice(allowed, size=int(rng.integers(2, 6)), replace=True))
        if payload[0].startswith('D') and payload[0][1:].isdigit():
            payload.insert(0, 'NUM')
        payloads.append(payload)
    for seq, payload in enumerate(payloads):
        frame = protocol.Frame(
            sender=seq % (compact_clocked.MAX_ADDRESS + 1),
            recipient=(seq * 7 + 3) % (compact_clocked.MAX_ADDRESS + 1),
            msg_type=protocol.MSG_TYPES[seq % len(protocol.MSG_TYPES)],
            seq=seq % (compact_clocked.MAX_SEQUENCE + 1),
            payload=payload)
        items.append((compact_clocked.to_concepts(frame), PROSODIES[seq % 4][0]))
    rng.shuffle(items)
    return items


def load_done(channel, encoding=None):
    """Set of labeled messages with usable trimmed OR raw audio for resume."""
    done = set()
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if encoding is None or r.get("encoding") == encoding:
                        done.add((tuple(r["concepts"]), r["prosody"], r["channel"]))
                except Exception:
                    pass
    # Clocked raw audio is the evaluation source of truth. A conservative legacy
    # trimmer rejection must not replay a packet whose raw capture was preserved.
    attempts_path = os.path.join(REC_DIR, "attempts.jsonl")
    if encoding and os.path.exists(attempts_path):
        by_file = {}
        with open(attempts_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("channel") == channel and r.get("encoding") == encoding:
                        by_file[r["file"]] = r
                except Exception:
                    pass
        for r in by_file.values():
            raw = os.path.join(REC_DIR, r["file"].replace(".wav", ".raw.wav"))
            if os.path.exists(raw):
                done.add((tuple(r["concepts"]), r["prosody"], channel))
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
    ap.add_argument("--slots", action="store_true",
                    help="emit historical widened-gap slotted audio. "
                         "Tag the channel distinctly (e.g. --channel airplay_office_slots) "
                         "so slotted recordings stay separate from the legacy set.")
    ap.add_argument("--max-hang-streak", type=int, default=4,
                    help="abort after this many consecutive hangs (raise for flaky "
                         "Bluetooth links that hang intermittently but recover).")
    ap.add_argument("--hang-pause", type=float, default=0.0,
                    help="seconds to wait after a hang before the next emission, so "
                         "a wedged audio link can settle (e.g. 3.0 for Bluetooth).")
    ap.add_argument("--clocked", action="store_true", help="experimental synchronized format; use a new channel tag")
    ap.add_argument("--compact-packets", action="store_true",
                    help="clocked v2 protected packets with compact headers and CRC32; use a fresh channel tag")
    ap.add_argument("--dry-run", action="store_true", help="print corpus/format summary; no audio-device access or playback")
    ap.add_argument("--session-id", help="capture session ID; defaults to a new UUID per run")
    args = ap.parse_args()
    if args.clocked and args.compact_packets:
        ap.error('--clocked and --compact-packets are separate wire formats')
    if (args.clocked or args.compact_packets) and args.slots:
        ap.error("clocked formats and --slots select different formats")
    import uuid, hashlib
    from datetime import datetime, timezone
    from b4bl import clocked as clock_format, compact_clocked
    use_clocked = args.clocked or args.compact_packets
    encoding = (compact_clocked.PROFILE if args.compact_packets else
                clock_format.PROFILE if args.clocked else
                "slots-complete-contours-v2" if args.slots else "legacy-v1")
    session_id = args.session_id or str(uuid.uuid4())
    source_hash = hashlib.sha256()
    for module in ("codec", "phonology", "prosody", "generators", "lexicon", "clocked",
                   "compact_clocked"):
        with open(os.path.join(os.path.dirname(__file__), "..", "b4bl", module+".py"), "rb") as source:
            source_hash.update(source.read())
    rng = np.random.default_rng(args.seed)
    items = (build_compact_packet_set(rng) if args.compact_packets else
             build_message_set(rng, clocked=args.clocked))
    if args.limit:
        items = items[:args.limit]
    if args.dry_run:
        from collections import Counter
        audio_seconds = (sum(clock_format.duration_samples(message) for message, _ in items)
                         / gen.SR if use_clocked else None)
        print(json.dumps({"encoding": encoding, "session_id": session_id,
                          "messages": len(items), "lengths": dict(Counter(len(m) for m, _ in items)),
                          "seed": args.seed, "source_sha256": source_hash.hexdigest(),
                          "rendered_audio_seconds": audio_seconds,
                          "raw_recordings": use_clocked, "playback": False}, indent=2))
        return
    # Never silently mix a new wire format into a historical channel tag.
    if os.path.exists(MANIFEST):
        for line in open(MANIFEST):
            r = json.loads(line)
            if r.get("channel") == args.channel and r.get("encoding") != encoding:
                ap.error("channel contains another/unversioned encoding; choose a fresh --channel tag")

    os.makedirs(REC_DIR, exist_ok=True)
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
    base_env["B4BL_SLOTS"] = "1" if args.slots else "0"
    base_env["B4BL_CLOCKED"] = "1" if use_clocked else "0"
    base_env["B4BL_KEEP_RAW"] = "1" if use_clocked else "0"
    base_env["B4BL_PACKET_ENCODING"] = encoding if args.compact_packets else ""
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

    done = load_done(args.channel, encoding)
    pending = [(m, p) for (m, p) in items if (tuple(m), p, args.channel) not in done]

    # start file numbering ABOVE any existing files for this channel, so a resume
    # run never overwrites recordings from a previous run.
    import glob, re
    existing = glob.glob(os.path.join(REC_DIR, f"{args.channel}_*.wav"))
    next_idx = 0
    for p in existing:
        m = re.search(rf"{re.escape(args.channel)}_(\d+)(?:\.raw)?\.wav$", p)
        if m:
            next_idx = max(next_idx, int(m.group(1)) + 1)
    # Failed attempts may have no WAV but still reserve their filename in the
    # append-only log. Never reuse it: doing so can attach new audio to an old label.
    attempts_path = os.path.join(REC_DIR, "attempts.jsonl")
    if os.path.exists(attempts_path):
        for line in open(attempts_path):
            try:
                row = json.loads(line)
                if row.get("channel") != args.channel:
                    continue
                m = re.search(rf"{re.escape(args.channel)}_(\d+)\.wav$", row.get("file", ""))
                if m:
                    next_idx = max(next_idx, int(m.group(1)) + 1)
            except Exception:
                pass

    print(f"=== collecting {len(pending)} emissions on channel '{args.channel}' "
          f"({len(items) - len(pending)} already done) ===")
    if use_clocked:
        audio_seconds = sum(clock_format.duration_samples(message) for message, _ in pending) / gen.SR
        capture_overhead = (capture.LEAD_SIL + 2*capture.REC_MARGIN + .26 +
                            capture.POST_SIL + .6)
        est_min = (audio_seconds + len(pending)*capture_overhead) / 60
    else:
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

    n_ok = n_skip = n_hang = n_error = 0
    failure_streak = 0
    with open(MANIFEST, "a") as mf:
        for i, (msg, pname) in enumerate(pending):
            fn = f"{args.channel}_{next_idx:05d}.wav"
            next_idx += 1  # reserve even failed attempts; retain raw/timing evidence
            out = os.path.join(REC_DIR, fn)
            env = dict(base_env, B4BL_PROSODY=pname)
            cmd = [sys.executable, emit, out, str(args.latency), str(args.gain)] + list(msg)
            # Timeout scales with the actual waveform, including long clocked
            # words, instead of assuming every message fits in twelve seconds.
            from b4bl import clocked as clock_format
            prmap = {"neutral": prosody.NEUTRAL, "uncertain": prosody.UNCERTAIN,
                     "urgent": prosody.URGENT, "calm": prosody.CALM}
            preview = (clock_format.encode(msg, prmap[pname]) if use_clocked else
                       codec.encode(msg, prmap[pname], slots=args.slots))
            message_timeout = emission_timeout + len(preview)/gen.SR
            try:
                p = subprocess.run(cmd, env=env, timeout=message_timeout,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
                rc = p.returncode
                worker_error = (p.stderr or p.stdout or '').strip()[-2000:]
            except subprocess.TimeoutExpired:
                rc = -1   # hung: subprocess killed by timeout, OS frees audio state
                worker_error = 'worker timed out'

            with open(os.path.join(REC_DIR, "attempts.jsonl"), "a") as attempts:
                attempts.write(json.dumps({"file": fn, "session_id": session_id,
                                           "encoding": encoding, "channel": args.channel,
                                           "concepts": msg, "prosody": pname, "returncode": rc,
                                           "worker_error": worker_error if rc not in (0, 2) else "",
                                           "recorded_at": datetime.now(timezone.utc).isoformat()}) + "\n")
            if rc == -1:
                n_hang += 1
                failure_streak += 1
            elif rc not in (0, 2):
                n_error += 1
                failure_streak += 1
                detail = ('clocked marker health check failed' if rc == 4 else
                          worker_error.splitlines()[-1] if worker_error else f'exit {rc}')
                print(f"  [device error] attempt {i}: {detail}", flush=True)
            else:
                failure_streak = 0
            if failure_streak >= args.max_hang_streak:
                print(f"  [abort] {failure_streak} audio worker failures in a row at {i}; "
                      f"channel likely dropped. Reconnect devices and re-run to resume.",
                      flush=True)
                break
            if rc not in (0, 2):
                time.sleep(args.hang_pause)
                continue
            if rc == 0 and os.path.exists(out):
                mf.write(json.dumps({"file": fn, "concepts": msg,
                                     "prosody": pname, "channel": args.channel,
                                     "session_id": session_id, "encoding": encoding,
                                     "source_sha256": source_hash.hexdigest(),
                                     "recorded_at": datetime.now(timezone.utc).isoformat(),
                                     "input_device": args.input_device, "output_device": args.output_device,
                                     "gain": args.gain, "seed": args.seed,
                                     "timing_file": fn.replace(".wav", ".timing.json") if use_clocked else None,
                                     "raw_file": fn.replace(".wav", ".raw.wav") if use_clocked else None,
                                     "frame": (compact_clocked.parse_concepts(msg).frame.__dict__
                                               if args.compact_packets else None)}) + "\n")
                mf.flush()
                n_ok += 1
            elif rc == 2:
                n_skip += 1     # no sync or too quiet
            else:
                n_skip += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(pending)}  ok={n_ok} skip={n_skip} "
                      f"hang={n_hang} error={n_error}",
                      flush=True)
    print(f"DONE: {n_ok} trimmed, {n_skip} raw-only, {n_hang} hangs, "
          f"{n_error} worker errors. "
          f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
