"""Command line access to the two explicitly selected clocked profiles."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

from . import (SR, __version__, decode_message, decode_packet,
               encode_message, encode_packet, supported_words)
from . import clocked
from .models import (MODEL_REVISION, MODEL_SHA256, MODEL_URL, fetch_model, install_model,
                     load_model, model_path, verify_model)
from .prosody import CALM, NEUTRAL, UNCERTAIN, URGENT


PROSODIES = {"neutral": NEUTRAL, "uncertain": UNCERTAIN,
             "urgent": URGENT, "calm": CALM}


def _parser():
    parser = argparse.ArgumentParser(prog="b4bl", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("words", help="list all 212 supported clocked words")
    encode = commands.add_parser("encode", help="render a clocked WAV file")
    encode.add_argument("--profile", required=True, choices=("packet", "message"))
    encode.add_argument("--output", required=True, type=Path)
    encode.add_argument("--prosody", choices=PROSODIES, default="neutral")
    encode.add_argument("--sender", type=int)
    encode.add_argument("--recipient", type=int)
    encode.add_argument("--type", choices=("MSG_TELL", "MSG_ASK", "MSG_ACKF", "MSG_WARN"))
    encode.add_argument("--seq", type=int)
    encode.add_argument("words", nargs="+", help="concept words; run 'b4bl words' for the supported list")
    decode = commands.add_parser("decode", help="decode a bounded mono 44100 Hz WAV")
    decode.add_argument("--profile", required=True, choices=("packet", "message"))
    decode.add_argument("--input", required=True, type=Path)
    decode.add_argument("--model", type=Path, help="trusted local model with pinned SHA256")
    listen = commands.add_parser("listen", help="Full packet desktop microphone listener")
    listen.add_argument("--input-device", required=True, help="explicit device name or index")
    listen.add_argument("--output-device", help="required with --play-replies")
    listen.add_argument("--model", type=Path, help="trusted local model with pinned SHA256")
    listen.add_argument("--duration", type=float, help="stop after this many seconds")
    listen.add_argument("--chunk-ms", type=float, default=100.0)
    listen.add_argument("--confirm-success", action="store_true")
    listen.add_argument("--no-repeat", action="store_true")
    listen.add_argument("--play-replies", action="store_true")
    listen.add_argument("--reply-directory", type=Path)
    listen.add_argument("--min-marker-snr-db", type=float)
    models = commands.add_parser("models", help="manage the frozen decoder model")
    model_commands = models.add_subparsers(dest="model_command", required=True)
    model_commands.add_parser("list")
    model_commands.add_parser("fetch")
    verify = model_commands.add_parser("verify")
    verify.add_argument("path", nargs="?", type=Path)
    install = model_commands.add_parser("install")
    install.add_argument("path", type=Path, help="trusted local model file")
    return parser


def _write_audio(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, SR, np.int16(np.clip(audio, -1, 1) * 32767))


def _read_audio(path):
    rate, audio = wavfile.read(path)
    if rate != SR or audio.ndim != 1:
        raise ValueError(f"requires mono {SR} Hz WAV")
    if np.issubdtype(audio.dtype, np.unsignedinteger):
        midpoint = 2 ** (8 * audio.dtype.itemsize - 1)
        audio = (audio.astype(np.float32) - midpoint) / midpoint
    elif np.issubdtype(audio.dtype, np.signedinteger):
        scale = 2 ** (8 * audio.dtype.itemsize - 1)
        audio = audio.astype(np.float32) / scale
    return np.asarray(audio, dtype=np.float32)


def _run(args, parser):
    if args.command == "words":
        return {"profiles": ["packet", "message"], "words": supported_words()}
    if args.command == "encode":
        if args.profile == "packet":
            if any(value is None for value in (args.sender, args.recipient, args.type, args.seq)):
                parser.error("packet requires --sender, --recipient, --type, and --seq")
            audio = encode_packet(args.sender, args.recipient, args.type, args.seq,
                                  args.words, PROSODIES[args.prosody])
        else:
            if any(value is not None for value in (args.sender, args.recipient, args.type, args.seq)):
                parser.error("message does not use packet header fields")
            audio = encode_message(args.words, PROSODIES[args.prosody])
        _write_audio(args.output, audio)
        return {"profile": args.profile, "output": str(args.output),
                "sample_rate": SR, "samples": len(audio)}
    if args.command == "decode":
        model = load_model(args.model)
        audio = _read_audio(args.input)
        result = (decode_packet(audio, model) if args.profile == "packet"
                  else decode_message(audio, model))
        return {**asdict(result), "model_profile": model["profile"]}
    if args.model_command == "list":
        path = model_path()
        installed = path.is_file()
        try:
            verified = installed and verify_model(path) == MODEL_SHA256
        except (OSError, ValueError):
            verified = False
        return {"models": [{"name": MODEL_REVISION, "sha256": MODEL_SHA256,
                            "url": MODEL_URL, "cache_path": str(path),
                            "installed": installed, "verified": verified}]}
    if args.model_command == "verify":
        return {"path": str(args.path or model_path()),
                "sha256": verify_model(args.path), "revision": MODEL_REVISION,
                "profile": clocked.PROFILE, "verified": True}
    path = fetch_model() if args.model_command == "fetch" else install_model(args.path)
    return {"path": str(path), "sha256": MODEL_SHA256, "verified": True}


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "listen":
            from .live import listen
            for event in listen(
                    input_device=args.input_device, model_path=args.model,
                    duration=args.duration, chunk_ms=args.chunk_ms,
                    confirm_success=args.confirm_success,
                    request_repeat=not args.no_repeat,
                    output_device=args.output_device,
                    play_replies=args.play_replies,
                    reply_directory=args.reply_directory,
                    min_marker_snr_db=args.min_marker_snr_db):
                print(json.dumps(event), flush=True)
            return 0
        print(json.dumps(_run(args, parser)))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
