"""Full packet desktop listener; optional device capture is loaded on demand."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from scipy.io import wavfile

from . import capture, clocked
from .models import MODEL_REVISION, MODEL_SHA256, load_model
from .runtime_receiver import RuntimeConfig, StreamingPacketReceiver


def listen(*, input_device: str, model_path=None, duration=None, chunk_ms=100.0,
           confirm_success=False, request_repeat=True, output_device=None,
           play_replies=False, reply_directory=None, min_marker_snr_db=None):
    """Yield JSON-ready listener events from an explicitly selected microphone.

    This is packet-only. Reply playback also requires an explicit output device.
    No default microphone or speaker is chosen by this public command.
    """
    if not input_device:
        raise ValueError("listen requires --input-device")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive")
    if chunk_ms <= 0:
        raise ValueError("chunk-ms must be positive")
    if play_replies and not output_device:
        raise ValueError("--play-replies requires --output-device")
    model = load_model(model_path)
    try:
        sd = capture._sd()
    except ImportError as exc:
        raise RuntimeError("desktop listening requires: pip install 'b4bl[live]'") from exc
    if hasattr(model["model"], "n_jobs"):
        model["model"].n_jobs = 1
    config = RuntimeConfig(
        analysis_block_ms=chunk_ms, confirm_success=confirm_success,
        request_repeat=request_repeat, min_marker_snr_db=min_marker_snr_db)
    receiver = StreamingPacketReceiver(model, config)
    input_index = capture._resolve_device(input_device, "input")
    output_index = (capture._resolve_device(output_device, "output")
                    if play_replies else None)
    directory = Path(reply_directory) if reply_directory else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
    block = max(1, round(chunk_ms / 1000 * clocked.SR))
    started = time.monotonic()
    event_index = 0
    yield {"status": "listening", "profile": "packet", "sample_rate": clocked.SR,
           "model_sha256": MODEL_SHA256, "model_revision": MODEL_REVISION,
           "model_profile": model["profile"],
           "input_device": input_index, "output_device": output_index,
           "automatic_playback": play_replies}

    def present(event):
        nonlocal event_index
        item = event.summary()
        item.update(profile="packet", model_sha256=MODEL_SHA256,
                    model_revision=MODEL_REVISION,
                    model_profile=model["profile"],
                    integrity_verified=bool(event.accepted))
        if directory and event.reply_audio is not None:
            path = directory / f"reply_{event_index:05d}_{event.kind}.wav"
            wavfile.write(path, clocked.SR,
                          np.int16(np.clip(event.reply_audio, -1, 1) * 32767))
            item["reply_file"] = str(path)
        event_index += 1
        if play_replies and event.reply_audio is not None:
            receiver.suppress_for(len(event.reply_audio) / clocked.SR + .25)
            sd.play(event.reply_audio, samplerate=clocked.SR,
                    device=output_index, blocking=True)
        return item

    try:
        with sd.InputStream(samplerate=clocked.SR, channels=1,
                            dtype="float32", blocksize=block, device=input_index) as stream:
            while duration is None or time.monotonic() - started < duration:
                chunk, overflowed = stream.read(block)
                if overflowed:
                    yield {"warning": "input overflow", "profile": "packet"}
                for event in receiver.feed(np.asarray(chunk[:, 0], dtype=np.float32)):
                    yield present(event)
    except KeyboardInterrupt:
        pass
    finally:
        for event in receiver.flush():
            yield present(event)
        yield {"status": "stopped", "profile": "packet", "events": event_index,
               "model_sha256": MODEL_SHA256,
               "model_revision": MODEL_REVISION,
               "audio_seconds": receiver.sample_cursor / clocked.SR}
