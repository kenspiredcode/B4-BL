"""Regression tests for deterministic metrics, symbol contracts, and framing."""
import json
import numpy as np
import pytest
from b4bl import benchmark, classifier, codec, phonology as ph, prosody, slot_decoder
from b4bl import clocked, clocked_receiver, compact_clocked, protocol, verified_clocked


def test_word_error_counts_include_insertions_and_deletions():
    assert benchmark.edit_counts(['SELF', 'MOVE'], ['SELF', 'MOVE', 'FRONT']) == (0, 0, 1)
    assert benchmark.edit_counts(['SELF', 'MOVE'], ['MOVE']) == (0, 1, 0)
    assert benchmark.edit_counts(['SELF', 'MOVE'], ['SELF', 'STOP']) == (1, 0, 0)
    result = benchmark.metrics([dict(expected=['SELF'], decoded=['SELF', 'MOVE'], segments=2, expected_segments=1)])
    assert result['exact_rate'] == 0 and result['word_error_rate'] == 1


def test_composition_split_is_stable_and_keeps_all_prosodies_together():
    assert benchmark.sequence_holdout(['SELF', 'MOVE']) == benchmark.sequence_holdout(json.loads('["SELF","MOVE"]'))
    assert {benchmark.sequence_holdout([f'WORD{i}']) for i in range(100)} == {True, False}


@pytest.mark.parametrize('concept', ['CONFIRM', 'DONE', 'ERROR'])
def test_acoustic_indexes_resolve_historical_aliases(concept):
    seq = codec.concepts_to_phoneme_words([concept])[0]
    candidates = [[(ph.canonical(n), 1.0)] for n in seq]
    assert codec.candidate_words_to_concepts([candidates]) == [concept]
    canonical = dict(slot_decoder._morph_index())[concept]
    assert slot_decoder._score_seq(candidates, canonical) == 0


def test_ambiguous_flat_code_remains_visible_in_audit():
    assert ['SELF', 'SELF', 'GIVE'] in benchmark.inventory_audit()['one_vs_two_word_collisions']
    assert ['DENY', 'IT', 'SEARCH'] in benchmark.inventory_audit()['one_vs_two_word_collisions']


def test_uncertain_slot_preserves_complete_contour():
    p = ph.BY_NAME['Mc']
    original = p.render(prosody.UNCERTAIN)
    assert len(original) > int(.16*codec.gen.SR)
    rendered = codec._render_phoneme_seq_slotted(['Mc'], prosody.UNCERTAIN)[0]
    expected = p.render(prosody.UNCERTAIN, duration_sec=.16)
    np.testing.assert_allclose(rendered[:len(expected)], expected)
    assert np.count_nonzero(rendered[len(expected):]) == 0
    assert not np.allclose(expected, original[:len(expected)])
    with pytest.raises(ValueError):
        codec._fit_slot(original, .26)


def test_markers_recover_word_boundaries_without_energy_gaps():
    two = clocked.encode(['SELF', 'SELF'])
    one = clocked.encode(['GIVE'])
    assert len(clocked.marker_positions(two)) == 3
    assert len(clocked.marker_positions(one)) == 2
    rng = np.random.default_rng(31)
    noisy = two + rng.normal(0, .06, len(two))
    starts = clocked.marker_positions(noisy)
    assert len(starts) == 3
    assert abs(starts[1]-starts[0]-clocked.PREFIX-clocked.TICK) < 20


def test_clock_estimator_recovers_small_sample_rate_drift():
    audio = clocked.encode(['SELF', 'ENERGY', 'LOW'])
    from scipy.signal import resample_poly
    received = resample_poly(audio, 1001, 1000)
    starts = clocked.marker_positions(received)
    scale, residual = clocked.estimate_clock(starts)
    assert abs(scale - 1.001) < 0.0005
    assert residual < 5
    model = clocked.train(samples_per=24, seed=17)
    result = clocked.decode(received, model)
    assert result.accepted and result.words == ['SELF', 'ENERGY', 'LOW']


def test_clocked_rejects_unsupported_words_and_model():
    for words in ([], ['WORKING'], ['ALARM'], ['R2D2']):
        with pytest.raises(ValueError):
            clocked.encode(words)
    with pytest.raises(ValueError):
        clocked.decode(np.zeros(100), {'models': {}})


def test_complete_duration_for_all_prosodies():
    for pr in (prosody.NEUTRAL, prosody.UNCERTAIN, prosody.URGENT, prosody.CALM):
        for name in ('Lf', 'LiL'):
            assert len(clocked.body(name, pr)) == clocked.ticks(name)*clocked.TICK-clocked.GUARD


def test_crc_protects_header_payload_order_and_trailing_data():
    frame = protocol.Frame(1, 2, 'MSG_TELL', 3, ['SELF', 'MOVE', 'FRONT'])
    wire = verified_clocked.to_concepts(frame)
    decoded = verified_clocked.parse_concepts(wire)
    assert decoded.ok and decoded.checksum_ok and decoded.frame == frame
    for index in range(len(wire)):
        corrupted = wire.copy()
        corrupted[index] = 'D9' if wire[index] != 'D9' else 'D8'
        assert not verified_clocked.parse_concepts(corrupted).checksum_ok
    assert not verified_clocked.parse_concepts(wire+['ACK']).ok
    assert not verified_clocked.parse_concepts(wire[:-1]).ok


def test_packet_numeric_payload_and_d_initial_words():
    for payload in (['DONE'], ['DENY'], ['NUM', 'D4', 'D2']):
        f = protocol.Frame(12, 99, 'MSG_WARN', 42, payload)
        parsed = verified_clocked.parse_concepts(verified_clocked.to_concepts(f))
        assert parsed.ok and parsed.frame == f
    with pytest.raises(ValueError):
        verified_clocked.to_concepts(protocol.Frame(1, 2, 'MSG_TELL', 0, ['D4']))


def test_legacy_packet_payload_deny_not_consumed_as_sequence_digits():
    parsed = protocol.parse_concepts(protocol.make_frame(1, 2, 'MSG_TELL', 0, ['DENY', 'DONE']))
    assert parsed.ok and parsed.checksum_ok and parsed.frame.payload == ['DENY', 'DONE']


def test_training_holdout_groups_entire_recordings(tmp_path, monkeypatch):
    import joblib
    from b4bl import train_classifier as trainer
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 23))
    labels = np.array(['low', 'high']*30)
    groups = np.repeat([f'recording-{i}' for i in range(20)], 3)
    monkeypatch.setattr(trainer, 'build_dataset_from_recordings',
                        lambda *args, **kwargs: (X, labels, labels, labels, labels, groups))
    dest = str(tmp_path/'model.joblib')
    trainer.train(recordings='test-only-manifest', mix_synth=False, output=dest)
    saved = joblib.load(dest)
    assert not set(saved['train_groups']) & set(saved['validation_groups'])
    assert set(saved['train_groups']) | set(saved['validation_groups']) == set(groups)


@pytest.fixture(scope='module')
def tiny_clock_model():
    return clocked.train(samples_per=32, seed=13)


def test_clocked_decode_distinguishes_flat_stream_collision(tiny_clock_model):
    for words in (['SELF', 'SELF'], ['GIVE'], ['DENY', 'IT'], ['SEARCH']):
        result = clocked.decode(clocked.encode(words), tiny_clock_model)
        assert result.accepted and result.words == words, (words, result)


def test_majority_repairs_one_wrong_word(tiny_clock_model):
    result = clocked.decode(clocked.encode(['SELF', 'GIVE', 'SELF']), tiny_clock_model, repetition=3)
    assert result.accepted and result.words == ['SELF']
    truncated_block = clocked.decode(clocked.encode(['SELF', 'SELF']), tiny_clock_model, repetition=3)
    assert not truncated_block.accepted


def test_clocked_packet_acoustic_loopback(tiny_clock_model):
    frame = protocol.Frame(1, 2, 'MSG_TELL', 3, ['SELF', 'MOVE', 'FRONT'])
    result = verified_clocked.decode(verified_clocked.encode(frame), tiny_clock_model)
    assert result.accepted and result.parsed.frame == frame
    spectral = verified_clocked.decode(verified_clocked.encode(frame), tiny_clock_model,
                                       acoustic_decoder=clocked_receiver.decode)
    assert spectral.accepted and spectral.parsed.frame == frame


def test_crc_list_decode_recovers_second_choice_without_lowering_integrity():
    frame = protocol.Frame(1, 2, 'MSG_TELL', 3, ['SELF', 'MOVE', 'FRONT'])
    wire = verified_clocked.to_concepts(frame)
    wrong = wire.copy(); wrong[wire.index('SELF')] = 'TAKE'

    def fake_decoder(audio, model, repetition=1, min_margin=0):
        candidates = [[(word, 0.0)] for word in wrong]
        index = wire.index('SELF')
        candidates[index].append(('SELF', -0.2))
        return clocked.DecodeResult(words=wrong.copy(), accepted=True,
                                    hypothesis=wrong.copy(),
                                    word_candidates=candidates)

    result = verified_clocked.decode(np.zeros(1), {}, acoustic_decoder=fake_decoder)
    assert result.accepted and result.parsed.frame == frame
    assert result.selected_by_validation
    assert result.acoustic.words == wire
    assert result.acoustic.hypothesis == wrong


def test_crc_list_decode_does_not_accept_when_correct_word_is_absent():
    frame = protocol.Frame(1, 2, 'MSG_TELL', 3, ['SELF', 'MOVE', 'FRONT'])
    wire = verified_clocked.to_concepts(frame)
    wrong = wire.copy(); wrong[wire.index('SELF')] = 'TAKE'

    def fake_decoder(audio, model, repetition=1, min_margin=0):
        return clocked.DecodeResult(words=wrong.copy(), accepted=True,
                                    hypothesis=wrong.copy(),
                                    word_candidates=[[(word, 0.0)] for word in wrong])

    result = verified_clocked.decode(np.zeros(1), {}, acoustic_decoder=fake_decoder)
    assert not result.accepted and not result.selected_by_validation


def test_compact_packet_roundtrip_integrity_and_limits():
    frame = protocol.Frame(31, 0, 'MSG_WARN', 1023, ['SELF', 'MOVE', 'FRONT'])
    wire = compact_clocked.to_concepts(frame)
    parsed = compact_clocked.parse_concepts(wire)
    assert parsed.ok and parsed.checksum_ok and parsed.frame == frame
    assert wire[:2] == ['SYNC', 'D2']
    assert len(wire[-compact_clocked.CRC_SYMBOLS:]) == 7
    for index, original in enumerate(wire):
        replacement = ('ACK' if original != 'ACK' else 'ALL')
        corrupted = wire.copy(); corrupted[index] = replacement
        assert not compact_clocked.parse_concepts(corrupted).checksum_ok
    for bad in (
            protocol.Frame(32, 0, 'MSG_TELL', 0, ['SELF']),
            protocol.Frame(0, 32, 'MSG_TELL', 0, ['SELF']),
            protocol.Frame(0, 0, 'MSG_TELL', 1024, ['SELF'])):
        with pytest.raises(ValueError):
            compact_clocked.to_concepts(bad)


def test_compact_packet_reduces_sample_airtime_and_acoustic_loopback(tiny_clock_model):
    frame = protocol.Frame(1, 2, 'MSG_TELL', 3, ['SELF', 'MOVE', 'FRONT'])
    compact_audio = compact_clocked.encode(frame)
    verbose_audio = verified_clocked.encode(frame)
    assert len(compact_audio) < len(verbose_audio) * .6
    result = compact_clocked.decode(compact_audio, tiny_clock_model)
    assert result.accepted and result.parsed.frame == frame


def test_compact_crc_list_decode_uses_structural_pruning():
    frame = protocol.Frame(1, 2, 'MSG_TELL', 3, ['SELF', 'MOVE', 'FRONT'])
    wire = compact_clocked.to_concepts(frame)
    wrong = wire.copy(); wrong[wire.index('FRONT')] = 'FAR'

    def fake_decoder(audio, model, repetition=1, min_margin=0):
        candidates = [[('SYNC', 1.0), (word, 0.0)] for word in wrong]
        index = wire.index('FRONT')
        candidates[index] = [('FAR', 0.0), ('FRONT', -0.2)]
        return clocked.DecodeResult(words=wrong.copy(), accepted=True,
                                    hypothesis=wrong.copy(), word_candidates=candidates)

    result = compact_clocked.decode(np.zeros(1), {}, acoustic_decoder=fake_decoder)
    assert result.accepted and result.parsed.frame == frame
    assert result.selected_by_validation


def test_compact_spoken_replies_are_optional_and_ignore_nonpackets():
    frame = protocol.Frame(1, 2, 'MSG_TELL', 3, ['SELF'])
    accepted = verified_clocked.PacketResult(
        clocked.DecodeResult(words=compact_clocked.to_concepts(frame), accepted=True,
                             marker_count=compact_clocked.MIN_PACKET_WORDS + 1),
        protocol.ParseResult(frame, True, '', True))
    assert compact_clocked.spoken_reply_concepts(accepted) == []
    assert compact_clocked.spoken_reply_concepts(
        accepted, confirm_success=True) == ['ACK']
    assert compact_clocked.encode_spoken_reply(accepted) is None

    rejected = verified_clocked.PacketResult(
        clocked.DecodeResult(accepted=True,
                             marker_count=compact_clocked.MIN_PACKET_WORDS + 1),
        protocol.ParseResult(None, False, 'CRC mismatch', False))
    assert compact_clocked.spoken_reply_concepts(rejected) == ['SORRY', 'REPEAT']
    reply_audio = compact_clocked.encode_spoken_reply(rejected)
    assert len(clocked.marker_positions(reply_audio)) == 3

    background = verified_clocked.PacketResult(
        clocked.DecodeResult(accepted=False, marker_count=2, reason='not a packet'),
        protocol.ParseResult(None, False, 'not a packet', False))
    assert compact_clocked.spoken_reply_concepts(background) == []


def test_silent_capture_plan_never_opens_devices(monkeypatch, capsys):
    from src import collect_dataset
    monkeypatch.setattr(collect_dataset.capture, 'set_channel_profile',
                        lambda **kw: pytest.fail('dry run touched the audio channel'))
    monkeypatch.setattr('sys.argv', ['collect_dataset', '--clocked', '--dry-run', '--limit', '12'])
    collect_dataset.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan['messages'] == 12 and plan['playback'] is False
    assert plan['encoding'] == clocked.PROFILE


def test_compact_packet_capture_plan_is_silent_and_versioned(monkeypatch, capsys):
    from src import collect_dataset
    monkeypatch.setattr(collect_dataset.capture, 'set_channel_profile',
                        lambda **kw: pytest.fail('dry run touched the audio channel'))
    monkeypatch.setattr('sys.argv', ['collect_dataset', '--compact-packets',
                                    '--dry-run', '--limit', '12'])
    collect_dataset.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan['messages'] == 12 and plan['playback'] is False
    assert plan['encoding'] == compact_clocked.PROFILE
    assert min(map(int, plan['lengths'])) >= 16


def test_transmitter_spans_match_audio_and_clock():
    audio, spans = clocked.encode(['SELF', 'IT', 'GIVE'], return_spans=True)
    assert len(audio) == clocked.duration_samples(['SELF', 'IT', 'GIVE'])
    assert len(spans) == 4
    for span in spans:
        expected = clocked.body(span['phoneme'], prosody.NEUTRAL)
        np.testing.assert_allclose(audio[span['start']:span['end']], expected)
        assert span['slot_end']-span['start'] == clocked.ticks(span['phoneme'])*clocked.TICK
    assert [s['word_index'] for s in spans] == [0, 1, 2, 2]


def test_empty_recording_has_no_words():
    assert slot_decoder.decode_vocab(np.array([], dtype=np.float32)) == []
