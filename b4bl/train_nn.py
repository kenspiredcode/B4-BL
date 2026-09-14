"""
B4-BL — neural-net frame classifier (Option 2).

A small MLP/CNN over a WINDOW of consecutive frame feature-vectors, giving the
model temporal context the per-frame random forest lacks (it sees each frame in
isolation). Trains on Apple MPS (GPU) when available, else CPU.

Reuses the exact frame features + labels the RF frame model uses, so results are
directly comparable. Context = CONTEXT frames on each side stacked into the input.

Run:  python3 -m b4bl.train_nn --recordings recordings/train.jsonl
Saves: b4bl/models/frame_nn.pt  (+ label list)
"""

from __future__ import annotations
import os
import numpy as np

from . import phonology as ph
from .frame_decoder import SILENCE
from . import train_frames as TF

CONTEXT = 3          # frames of context each side -> window of 2*CONTEXT+1
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "frame_nn.pt")


def _device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _stack_context(X, seq_ids):
    """Turn per-frame features into context windows. seq_ids marks which source
    utterance each frame came from, so context doesn't cross utterance boundaries.
    Returns (N, (2*CONTEXT+1)*F)."""
    X = np.asarray(X, dtype=np.float32)
    n, f = X.shape
    seq_ids = np.asarray(seq_ids)
    out = np.zeros((n, (2 * CONTEXT + 1) * f), dtype=np.float32)
    for off_i, off in enumerate(range(-CONTEXT, CONTEXT + 1)):
        idx = np.clip(np.arange(n) + off, 0, n - 1)
        # zero-out where context crosses into a different utterance
        same = seq_ids[idx] == seq_ids
        block = X[idx]
        block[~same] = 0.0
        out[:, off_i * f:(off_i + 1) * f] = block
    return out


class MLP:
    pass


def _build_dataset(recordings, samples_per, force_align):
    """Reuse train_frames dataset builders but ALSO track an utterance id per frame
    so context windows don't cross utterances. We wrap _label_frames to tag ids."""
    import json
    from scipy.io import wavfile
    from . import decoder as _dec, codec, features as F, align as _align

    X, y, seq = [], [], []
    uid = [0]

    def add_frames(audio, spans):
        w = int(90 / 1000 * TF.SR)
        hop = int(20 / 1000 * TF.SR)
        for i in range(0, max(1, len(audio) - w), hop):
            c = i + w // 2
            label = SILENCE
            for s, e, lab in spans:
                if s <= c < e:
                    label = lab; break
            X.append(F.extract(audio[i:i + w])); y.append(label); seq.append(uid[0])
        uid[0] += 1

    # synthetic
    if not recordings or True:  # always include some synth for coverage
        sil = np.zeros(int(0.12 * TF.SR), dtype=np.float32)
        for p in ph.INVENTORY:
            for _ in range(samples_per):
                sig = p.render(TF._random_prosody())
                audio = TF._augment(np.concatenate([sil, sig, sil]))
                s = len(sil); e = s + len(sig)
                add_frames(audio, [(s, e, p.name)])
        for _ in range(samples_per * 3):
            add_frames(TF._augment(np.zeros(int(0.2 * TF.SR), dtype=np.float32) + 1e-4), [])

    # real
    if recordings:
        fa_model = None
        if force_align:
            from .frame_decoder import load as _load
            m = _load(); fa_model = m["model"] if m else None
        rec_dir = os.path.dirname(recordings)
        for line in open(recordings):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            exp = [p for ww in codec.concepts_to_phoneme_words(r["concepts"]) for p in ww]
            srr, d = wavfile.read(os.path.join(rec_dir, r["file"]))
            audio = d.astype(np.float32) / 32768.0
            segs = [(s, e) for (s, e, _g) in _dec._segments(audio)]
            if len(segs) == len(exp):
                add_frames(audio, [(s, e, pn) for (s, e), pn in zip(segs, exp)])
            elif fa_model is not None:
                res = _align.aligned_frame_labels(audio, exp, fa_model)
                if res is None:
                    continue
                starts, labels = res
                a = audio / (np.max(np.abs(audio)) or 1.0)
                w = int(90 / 1000 * TF.SR)
                for st, lab in zip(starts, labels):
                    X.append(F.extract(a[st:st + w])); y.append(lab); seq.append(uid[0])
                uid[0] += 1
    return np.array(X, dtype=np.float32), np.array(y), np.array(seq)


def train(recordings=None, samples_per=100, epochs=25, force_align=False, save=True):
    import torch, torch.nn as nn
    dev = _device()
    print(f"device: {dev}")
    print("building dataset...")
    X, y, seq = _build_dataset(recordings, samples_per, force_align)
    print(f"  {len(X)} frames, {X.shape[1]} feats")
    Xc = _stack_context(X, seq)

    labels = sorted(set(y))
    lab2i = {l: i for i, l in enumerate(labels)}
    yi = np.array([lab2i[l] for l in y])

    # split
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(Xc))
    ntr = int(0.85 * len(Xc))
    tr, te = perm[:ntr], perm[ntr:]

    Xt = torch.tensor(Xc[tr], device=dev); yt = torch.tensor(yi[tr], device=dev)
    Xv = torch.tensor(Xc[te], device=dev); yv = torch.tensor(yi[te], device=dev)

    net = nn.Sequential(
        nn.Linear(Xc.shape[1], 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, len(labels)),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss()
    bs = 4096
    for ep in range(epochs):
        net.train()
        pp = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), bs):
            b = pp[i:i + bs]
            opt.zero_grad()
            out = net(Xt[b]); loss = lossf(out, yt[b])
            loss.backward(); opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                acc = (net(Xv).argmax(1) == yv).float().mean().item()
            print(f"  epoch {ep+1}: val frame acc {acc:.3f}")
    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save({"state": net.state_dict(), "labels": labels,
                    "in_dim": Xc.shape[1], "context": CONTEXT,
                    "feat_dim": X.shape[1]}, MODEL_PATH)
        print(f"saved -> {MODEL_PATH}")
    return net, labels


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default=None)
    ap.add_argument("--force-align", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--samples", type=int, default=100)
    args = ap.parse_args()
    train(recordings=args.recordings, epochs=args.epochs,
          samples_per=args.samples, force_align=args.force_align)
