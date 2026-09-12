"""Link-layer logic tests (fast, using a controllable fake channel)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from b4bl import link, protocol as proto


def _perfect_channel(concepts):
    return list(concepts)


def test_delivers_on_clean_channel():
    rx = link.Receiver()
    st = link.send(1, 2, "MSG_TELL", 0, ["SELF", "ENERGY", "LOW"], rx,
                   channel=_perfect_channel, max_tries=3)
    assert st.delivered and st.tries == 1
    assert rx.inbox and rx.inbox[0].payload == ["SELF", "ENERGY", "LOW"]


def test_retries_then_succeeds():
    """A channel that garbles the first 2 tries then delivers cleanly."""
    calls = {"n": 0}

    def flaky(concepts):
        calls["n"] += 1
        if calls["n"] < 3:
            bad = list(concepts)
            bad[-1] = "D0"  # corrupt checksum digit
            return bad
        return list(concepts)

    rx = link.Receiver()
    st = link.send(1, 2, "MSG_TELL", 1, ["ACK"], rx, channel=flaky, max_tries=5)
    assert st.delivered and st.tries == 3 and st.nacks == 2


def test_gives_up_after_max_tries():
    def always_bad(concepts):
        bad = list(concepts); bad[-1] = "D0"; return bad
    rx = link.Receiver()
    st = link.send(1, 2, "MSG_TELL", 1, ["ACK"], rx, channel=always_bad, max_tries=4)
    assert not st.delivered and st.tries == 4


def test_receiver_dedupes_retransmissions():
    rx = link.Receiver()
    frame = proto.make_frame(1, 2, "MSG_TELL", 7, ["SELF", "MOVE"])
    rx.on_frame_concepts(frame)
    rx.on_frame_concepts(frame)  # same seq again
    assert len(rx.inbox) == 1
