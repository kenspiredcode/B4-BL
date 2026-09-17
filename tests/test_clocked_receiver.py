"""Offline receiver regressions: framing in distorted audio and negative inputs."""
import numpy as np
from scipy.io import wavfile
from b4bl import clocked, clocked_receiver as receiver
from b4bl.real_clocked_experiment import training_windows


def test_spectral_markers_survive_echo_rumble_and_offset():
    clean = clocked.encode(['SELF', 'GIVE', 'SELF'])
    a = np.pad(clean * .04, (4410, 8820))
    d = 617
    a[d:] += .4 * a[:-d]
    a += .1 * np.sin(2*np.pi*80*np.arange(len(a))/clocked.SR)
    positions = receiver.marker_positions(receiver.preprocess(a))
    expected = np.asarray(receiver.marker_positions(clean)) + 4410
    assert len(positions) == 4
    assert np.max(abs(np.asarray(positions)-expected)) < .008*clocked.SR
    assert training_windows(a, ['SELF', 'GIVE', 'SELF']) is not None
    assert training_windows(a, ['SELF', 'SELF', 'SELF']) is None


def test_spectral_marker_rejects_silence_noise_and_stationary_tones():
    rng = np.random.default_rng(2)
    t = np.arange(3*clocked.SR)/clocked.SR
    for a in [np.zeros(3*clocked.SR), rng.normal(0, .2, len(t)),
              np.sin(2*np.pi*2000*t), np.zeros(20)]:
        assert receiver.marker_positions(a) == []


def test_audio_reader_preserves_float_capture_scale(tmp_path):
    a = np.array([-.2, 0., .4, 1.2], dtype=np.float32)
    p = tmp_path/'float.wav'; wavfile.write(p, clocked.SR, a)
    np.testing.assert_allclose(receiver.read_audio(p), a)
    q = tmp_path/'int.wav'; wavfile.write(q, clocked.SR, np.array([-32768, 0, 16384], dtype=np.int16))
    np.testing.assert_allclose(receiver.read_audio(q), [-1, 0, .5])
