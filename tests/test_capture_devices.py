"""Device selection checks with no audio hardware or playback."""
from types import SimpleNamespace

import pytest
import numpy as np

from b4bl import capture


@pytest.fixture
def devices(monkeypatch):
    inventory = [
        dict(name="MacBook Pro Microphone", max_input_channels=1, max_output_channels=0),
        dict(name="sexy speaker", max_input_channels=0, max_output_channels=2),
        dict(name="Other Microphone", max_input_channels=1, max_output_channels=0),
    ]
    monkeypatch.setattr(capture, "_sd", lambda: SimpleNamespace(query_devices=lambda: inventory))


def test_names_and_cli_numeric_ids(devices):
    assert capture._resolve_device("MacBook", "input") == 0
    assert capture._resolve_device("1", "output") == 1
    assert capture._resolve_device(1, "output") == 1
    assert capture._resolve_device(None, "input") is None


@pytest.mark.parametrize("selector", ["“MacBook Pro Microphone”", "missing", "Microphone", "1", "-1", "99", ""])
def test_explicit_bad_input_never_falls_back(devices, selector):
    with pytest.raises(ValueError):
        capture._resolve_device(selector, "input")


def test_tone_check_ignores_louder_rumble_and_broadband_burst():
    sr = capture.SR
    rng = np.random.default_rng(42)
    audio = rng.normal(0, .001, 5 * sr)
    audio[3*sr:4*sr] += .8 * np.sin(2*np.pi*80*np.arange(sr)/sr)
    audio[4*sr:] += rng.normal(0, .3, sr)
    # Noise without the intended tone must fail, despite high overall level.
    assert not capture.assess_test_tone(audio)['ok']
    n = int(.4*sr)
    audio[2*sr:2*sr+n] += .06 * np.sin(2*np.pi*1000*np.arange(n)/sr)
    result = capture.assess_test_tone(audio)
    assert result['ok']
    assert 1.9 <= result['start_sec'] <= 2.2


def test_tone_check_rejects_silence_short_input_and_constant_background_tone():
    sr = capture.SR
    for audio in [np.zeros(sr*4), np.zeros(10),
                  .1*np.sin(2*np.pi*1000*np.arange(sr*4)/sr)]:
        assert not capture.assess_test_tone(audio)['ok']
