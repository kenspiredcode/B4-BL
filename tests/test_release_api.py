"""Guard the frozen wire encoders and the release boundary."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from b4bl import (decode_message, decode_packet, encode_message, encode_packet,
                  supported_words)
from b4bl import compact_clocked, protocol
from b4bl.models import verify_model
from b4bl import models


FIXTURE = json.loads((Path(__file__).parent / "release_fixture.json").read_text())
ROOT = Path(__file__).resolve().parent.parent


def test_release_waveforms_match_frozen_sources():
    packet = FIXTURE["cases"]["packet"]
    params = packet["input"]
    actual = encode_packet(**params)
    old = compact_clocked.encode(protocol.Frame(**params))
    assert np.array_equal(actual, old)
    assert len(actual) == packet["samples"]
    assert hashlib.sha256(actual.tobytes()).hexdigest() == packet["float32_sha256"]
    message = FIXTURE["cases"]["message"]
    actual = encode_message(message["input"]["words"])
    assert len(actual) == message["samples"]
    assert hashlib.sha256(actual.tobytes()).hexdigest() == message["float32_sha256"]
    for profile, filename in (("packet", "clocked_packet_full.wav"),
                              ("message", "clocked_message.wav")):
        path = ROOT / "audio_samples" / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            FIXTURE["cases"][profile]["wav_sha256"])


@pytest.mark.parametrize("word", ["ALARM", "WORKING", "CALCULATING", "UNKNOWN"])
def test_unsupported_clocked_words_fail_early(word):
    assert len(supported_words()) == 212
    with pytest.raises(ValueError, match="unsupported clocked words"):
        encode_message([word])
    with pytest.raises(ValueError, match="unsupported clocked words"):
        encode_packet(1, 2, "MSG_TELL", 1, [word])


def test_message_rejection_stays_unverified():
    class UnusedModel(dict):
        pass
    model = UnusedModel(profile="clocked-v1-fixed-300ms", features="pitch-envelope-v1")
    result = decode_message(np.zeros(44100, dtype=np.float32), model)
    assert result.status == "unverified"
    assert not result.integrity_verified
    assert not result.accepted


def test_model_hash_mismatch(tmp_path):
    path = tmp_path / "bad.joblib"
    path.write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_model(path)


def test_profile_mismatch_rejected():
    with pytest.raises(ValueError, match="clocked-v1 model"):
        decode_packet(np.zeros(44100, dtype=np.float32), {"profile": "other"})


def test_corrupted_packet_crc_rejected():
    words = compact_clocked.to_concepts(
        protocol.Frame(3, 7, "MSG_ASK", 42, ["YOU", "ENERGY", "LOW"]))
    words[words.index("ENERGY")] = "HIGH"
    result = compact_clocked.parse_concepts(words)
    assert not result.ok
    assert result.error == "CRC mismatch"


def test_failed_model_fetch_is_clear_and_leaves_no_asset(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    def fail(*args, **kwargs):
        raise OSError("offline")
    monkeypatch.setattr(models, "urlopen", fail)
    with pytest.raises(RuntimeError, match="could not fetch model"):
        models.fetch_model()
    assert not models.model_path().exists()


def test_decoder_rejects_incompatible_sklearn(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(models, "verify_model", lambda _: models.MODEL_SHA256)
    monkeypatch.setattr(models, "version", lambda _: "1.9.1")
    with pytest.raises(RuntimeError, match="requires scikit-learn 1.6.1"):
        models.load_model(path)


def test_cli_external_path_and_missing_model(tmp_path, monkeypatch, capsys):
    from b4bl.cli import main
    output = tmp_path / "message.wav"
    assert main(["encode", "--profile", "message", "--output", str(output),
                 "YOU", "ENERGY", "LOW"]) == 0
    assert output.exists()
    assert json.loads(capsys.readouterr().out)["profile"] == "message"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert main(["decode", "--profile", "message", "--input", str(output)]) == 2
    assert "b4bl models fetch" in json.loads(capsys.readouterr().err)["error"]
