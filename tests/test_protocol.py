"""Protocol framing tests (symbolic level)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from b4bl import protocol as proto


def test_frame_roundtrip():
    concepts = proto.make_frame(1, 2, "MSG_ASK", 5, ["YOU", "ENERGY", "LOW"])
    res = proto.parse_concepts(concepts)
    assert res.ok and res.checksum_ok
    f = res.frame
    assert (f.sender, f.recipient, f.msg_type, f.seq) == (1, 2, "MSG_ASK", 5)
    assert f.payload == ["YOU", "ENERGY", "LOW"]


def test_checksum_detects_payload_corruption():
    concepts = proto.make_frame(3, 7, "MSG_TELL", 1, ["SELF", "MOVE", "FRONT"])
    bad = list(concepts)
    bad[bad.index("FRONT")] = "BACK"
    res = proto.parse_concepts(bad)
    # frame still structurally parses, but checksum must flag the corruption
    assert res.frame is not None
    assert not res.checksum_ok


def test_multi_digit_addresses_and_seq():
    concepts = proto.make_frame(12, 99, "MSG_WARN", 42, ["OBSTACLE", "FRONT"])
    res = proto.parse_concepts(concepts)
    assert res.ok and res.checksum_ok
    assert (res.frame.sender, res.frame.recipient, res.frame.seq) == (12, 99, 42)


def test_missing_preamble_rejected():
    res = proto.parse_concepts(["YOU", "ENERGY", "LOW"])
    assert not res.ok
