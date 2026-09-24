"""Opt-in release check against an installed CLI and published fixture WAVs.

Run this script from anywhere, passing the installed ``b4bl`` executable and
the two release WAV assets. The executable is invoked from a temporary cwd, so
an in-tree package cannot accidentally satisfy the check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


MODEL_SHA256 = "d0f0d8170229da869332ae0754e05e488602d13f2fac6cd2ef6ae80a974ad551"
WAV_SHA256 = {
    "packet": "82d63c4804cdbb9706c381ed86def4de5cefdb769a1ba7347a3f9e1a1087409c",
    "message": "7848b24acb0f298bfd3de732d9a95cd5deb89f74e807827a98f08f4c7a61b506",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(cli, packet_wav, message_wav, cache_root):
    cli = Path(cli).resolve()
    packet_wav = Path(packet_wav).resolve()
    message_wav = Path(message_wav).resolve()
    for name, path in (("packet", packet_wav), ("message", message_wav)):
        actual = digest(path)
        if actual != WAV_SHA256[name]:
            raise ValueError(f"{name} fixture SHA256 mismatch: {actual}")
    env = {**os.environ, "XDG_CACHE_HOME": str(Path(cache_root).resolve()),
           "PYTHONPATH": ""}
    with tempfile.TemporaryDirectory(prefix="b4bl-release-check-") as temp:
        def run(*args):
            result = subprocess.run([str(cli), *map(str, args)], cwd=temp, env=env,
                                    capture_output=True, text=True, check=True)
            return json.loads(result.stdout)

        verified = run("models", "verify")
        assert verified["sha256"] == MODEL_SHA256
        packet_path = Path(temp) / "packet.wav"
        message_path = Path(temp) / "message.wav"
        run("encode", "--profile", "packet", "--sender", 3,
            "--recipient", 7, "--type", "MSG_ASK", "--seq", 42,
            "--output", packet_path, "YOU", "ENERGY", "LOW")
        run("encode", "--profile", "message", "--output", message_path,
            "YOU", "ENERGY", "LOW")
        for name, path in (("packet", packet_path), ("message", message_path)):
            assert digest(path) == WAV_SHA256[name], f"{name} encoder changed"
        packet = run("decode", "--profile", "packet", "--input", packet_wav)
        message = run("decode", "--profile", "message", "--input", message_wav)
        wrong_profile = run("decode", "--profile", "packet", "--input", message_wav)
        assert packet["accepted"] and packet["integrity_verified"]
        assert packet["frame"] == {
            "sender": 3, "recipient": 7, "msg_type": "MSG_ASK", "seq": 42,
            "payload": ["YOU", "ENERGY", "LOW"]}
        assert message["accepted"] and not message["integrity_verified"]
        assert message["status"] == "unverified"
        assert message["words"] == ["YOU", "ENERGY", "LOW"]
        assert not wrong_profile["accepted"] and wrong_profile["status"] == "rejected"
        assert packet["model_sha256"] == message["model_sha256"] == MODEL_SHA256
        return {"status": "passed", "packet": packet["status"],
                "message": message["status"], "model_sha256": MODEL_SHA256,
                "fixture_sha256": WAV_SHA256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, help="installed b4bl executable")
    parser.add_argument("--packet-wav", required=True)
    parser.add_argument("--message-wav", required=True)
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.cli, args.packet_wav, args.message_wav,
                           args.cache_root), indent=2))


if __name__ == "__main__":
    main()
