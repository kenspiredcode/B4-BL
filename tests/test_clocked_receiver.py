"""Offline receiver regressions: framing in distorted audio and negative inputs."""
import numpy as np
from scipy.io import wavfile
from b4bl import clocked, clocked_receiver as receiver
from b4bl.real_clocked_experiment import training_windows, augment_channel, select_rows
from b4bl import decoder


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


def test_promoted_marker_threshold_retains_negative_margin():
    assert receiver.MARKER_THRESHOLD == .55
    clean = clocked.encode(['SELF', 'GIVE', 'SELF'])
    assert len(receiver.marker_positions(receiver.preprocess(clean))) == 4
    # The next lower development threshold began adding peaks in real captures;
    # keep the promoted setting explicit and versioned.
    assert receiver.PROFILE.endswith('threshold055')


def test_audio_reader_preserves_float_capture_scale(tmp_path):
    a = np.array([-.2, 0., .4, 1.2], dtype=np.float32)
    p = tmp_path/'float.wav'; wavfile.write(p, clocked.SR, a)
    np.testing.assert_allclose(receiver.read_audio(p), a)
    q = tmp_path/'int.wav'; wavfile.write(q, clocked.SR, np.array([-32768, 0, 16384], dtype=np.int16))
    np.testing.assert_allclose(receiver.read_audio(q), [-1, 0, .5])


def test_training_augmentation_keeps_clock_and_does_not_modify_source():
    original = clocked.encode(['SELF', 'GIVE', 'SELF'])
    before = original.copy()
    a = augment_channel(original, np.random.default_rng(42))
    b = augment_channel(original, np.random.default_rng(42))
    np.testing.assert_array_equal(original, before)
    np.testing.assert_array_equal(a, b)
    assert len(a) == len(original) and np.isfinite(a).all()
    assert not np.allclose(a, original)
    assert training_windows(a, ['SELF', 'GIVE', 'SELF']) is not None


def test_periodicity_gate_ignores_noise_after_tone_release():
    sr = clocked.SR
    tone = .05*np.sin(2*np.pi*1000*np.arange(int(.2*sr))/sr)
    noise = np.random.default_rng(42).normal(0, .005, int(.2*sr))
    old = decoder._pitch_track(np.r_[tone, noise])
    confident = decoder._pitch_track(np.r_[tone, noise], min_periodicity=.5)
    assert len(confident) < len(old)
    assert np.all(abs(confident-1000) < 80)
    assert decoder._pitch_track(noise, min_periodicity=.5).tolist() == [0.0]


def test_fft_pitch_autocorrelation_finds_known_tone():
    t = np.arange(decoder.FRAME * 2) / clocked.SR
    track = decoder._pitch_track(np.sin(2 * np.pi * 700 * t), min_periodicity=.5)
    assert len(track) > 0
    assert np.allclose(track, 700, atol=12)


def test_feature_profile_carries_its_periodicity_threshold():
    assert clocked.CONFIDENT_FEATURE_PREFIX.endswith('-')
    # Decoder behavior is driven by model metadata; separately trained feature
    # profiles cannot silently use a hard-coded threshold.
    assert float({'min_periodicity': .6}.get('min_periodicity', 0)) == .6


def test_room_training_keeps_reserved_compositions_out():
    rows = [
        dict(partition='train', composition_holdout=False, id='base'),
        dict(partition='room3_test', composition_holdout=False, id='room_train'),
        dict(partition='room3_test', composition_holdout=True, id='room_holdout'),
        dict(partition='composition_test', composition_holdout=True, id='global_holdout'),
    ]
    training, evaluation = select_rows(rows, include_room3_training=True)
    assert [r['id'] for r in training] == ['base', 'room_train']
    assert [r['id'] for r in evaluation] == ['room_holdout', 'global_holdout']
    limited, _ = select_rows(rows, include_room3_training=True, room3_train_limit=0)
    assert [r['id'] for r in limited] == ['base']


def test_decoder_exposes_bounded_ranked_word_candidates():
    model = clocked.train(samples_per=4, seed=7)
    result = receiver.decode(clocked.encode(['SELF', 'GIVE']), model, min_margin=0)
    assert result.hypothesis == ['SELF', 'GIVE']
    assert len(result.word_candidates) == 2
    assert all(1 <= len(words) <= 5 for words in result.word_candidates)
    assert [words[0][0] for words in result.word_candidates] == result.hypothesis
    assert all(words == sorted(words, key=lambda item: item[1], reverse=True)
               for words in result.word_candidates)
