import numpy as np

from b4bl import ambient_benchmark, robust_features


def test_source_split_is_deterministic_and_source_level():
    names = [f"source-{i}.webm" for i in range(100)]
    first = [ambient_benchmark.source_split(name) for name in names]
    assert first == [ambient_benchmark.source_split(name) for name in names]
    assert 10 <= first.count("validation") <= 30


def test_mix_at_snr_hits_requested_ratio_and_avoids_clipping():
    rng = np.random.default_rng(4)
    packet = rng.normal(0, .2, 44100).astype(np.float32)
    ambient = rng.normal(0, .3, 44100).astype(np.float32)
    mixed, info = ambient_benchmark.mix_at_snr(packet, ambient, 6)
    measured = 20 * np.log10(info["packet_rms"] / info["ambient_rms"])
    assert abs(measured - 6) < 1e-6
    assert np.max(np.abs(mixed)) <= .950001


def test_clip_starts_stay_inside_source():
    starts = ambient_benchmark.clip_starts(100, 5, 12)
    assert len(starts) == 5
    assert all(0 <= start <= 88 for start in starts)
    assert starts == sorted(starts)


def test_robust_feature_profiles_are_finite_and_fixed_length():
    rng = np.random.default_rng(8)
    for profile in robust_features.PROFILES:
        short = robust_features.extract(rng.normal(0, .1, 8820), profile)
        long = robust_features.extract(rng.normal(0, .1, 22050), profile)
        assert short.shape == long.shape
        assert np.isfinite(short).all()
