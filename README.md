# B4-BL ("Babble")

An intelligible acoustic language for robots, styled after Star Wars astromech
droidspeak. B4-BL is both a constructed language and a low-bandwidth,
human-perceivable, spatially-local side channel for embodied machines.

Two acoustic layers travel together:

- a **symbolic layer** — sparse, robust, exactly machine-decodable;
- an **expressive layer** — prosody/affect (confidence, urgency) that a human can
  read but a machine may ignore.

The goal: a human who lives around the droids gradually learns roughly *what is
going on*, while a droid recovers the *exact packet*.

## Status (2026-09-17)

The sound palette, vocabulary, synthesis, and acoustic decoder exist. Compact
CRC-protected packets have now reached 95.8% verified exact delivery in one
prospective 500-packet room test, with no wrong packets accepted. Robustness across
additional rooms, interference, and hardware remains open. Older 95% figures were
small loopback experiments and must not be read as real-channel delivery accuracy.

A reproducible offline replay of 2,267 existing slotted recordings gives:

| Exact multi-word messages | Before alias fix | After alias fix |
|---|---:|---:|
| Close/clean, 291 messages | 71.5% | 77.0% |
| Distant/noisy, 287 messages | 40.4% | 44.9% |

These are **retrospective** scores with a model trained on these collections.
A separate model trained without the noisy round or a deterministic 20% of concept
sequences scores 81.8% on 66 unseen close-channel multi-word sequences and 29.6%
on 287 noisy-channel multi-word messages. Historical channel tags stand in for
missing session IDs; this is held out from model training, not a prospectively
unseen environment. Reported word error rate includes insertions and may exceed 100%.

`codec.encode(slots=True)` is the historical widened-gap format; its receiver still
segments phonemes by energy. Oversized gestures now render completely at the slot
body duration instead of losing their tails. New captures of this corrected format
must use a distinct channel/version, and the historical model may need retraining.

`b4bl.clocked` is a **separate experimental format**: tonal word markers, a real
300 ms clock (SHORT=1 tick, LONG=2), complete contours, and optional repetition.
Its timing samples have listening approval, and real recordings now cover MacBook
and AUKEY microphones, Bluetooth playback, and multiple rooms. On 132 fresh synthetic
cases crossing lengths 2–5 with all four prosodies, accepted exact decoding is 99.2%
clean, 84.8% with moderate echo and 15 dB nominal noise, 9.1% under the harsher 5 dB
condition, and 86.4% at 0.1% sample-clock skew. Best hypotheses reach 91.7% in the
moderate condition, but some are rejected by the uncalibrated confidence gate.
Five of 132 moderate-condition messages are incorrectly accepted; acoustic
acceptance must not be treated as CRC-verified delivery.
Triple repetition did not improve the moderate condition and increases mean
utterance duration from 4.41 to 12.58 seconds; it remains optional, off by default.

`b4bl.verified_clocked` retains the first inspectable CRC32 packet format; a
three-word payload takes 30.28 seconds. `b4bl.compact_clocked` is its versioned
successor: fixed-width base-32 metadata and seven base-32 CRC symbols cut the same
packet to 13.70 seconds while preserving CRC32 coverage of the version, complete
header, and payload. It supports 32 local addresses and 1,024 sequence values and
adds no classifier labels or sounds. It is not an acoustic ACK/retry implementation.
Fixed tempo, rhythm-defined ALARM/WORKING, spelling support, and realistic protected-
packet performance remain open. Rhythm-defined words/spelling are explicitly
unsupported by the new transport, not silently changed.

The initial offline work used no new captures; subsequent collection was explicitly
authorized and includes preserved raw audio even for trim-rejected attempts.
Initial findings live in the canonical vault document
`~/CollabHarnessVault/projects/B4-BL/docs/offline-validation-2026-09-16.md`.

## Silent evaluation

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
# Needs the existing recordings and historical model; never plays audio.
python3 -m b4bl.benchmark --output experiments/my-offline-run
python3 -m b4bl.reference_audit --cache experiments/my-offline-run
# Trains a separate synthetic clock-window model and tests without device I/O.
python3 -m b4bl.clocked_benchmark --output experiments/my-clocked-run
# Fresh evaluation of an already frozen model:
python3 -m b4bl.clocked_benchmark --model experiments/my-clocked-run/clocked_classifier.joblib --output experiments/my-clocked-holdout --seed 180027 --messages 120
# Prepares the future capture corpus only. --dry-run guarantees no playback.
python3 src/collect_dataset.py --clocked --dry-run --channel clocked_office_v1 --limit 500
```

Output directories must be new, so runs cannot silently overwrite earlier results.
JSON summaries are retained in `experiments/`; recording-level predictions,
feature caches, and model snapshots stay local and are ignored by Git. The
historical default classifier is not overwritten by these benchmark commands.
Random forests now validate with one shared split grouping entire recordings,
not separate phoneme-row splits for each output dimension. That internal split
still does not replace a held-out capture session/channel benchmark.

The clocked receiver estimates one affine sample-rate scale from all detected
marker intervals (within a fixed ±4% search range), then decodes windows on that
received clock. This handles cumulative drift such as a 0.1% sample-rate mismatch;
the estimate and residual are returned on `DecodeResult`. It remains conservative
about missing or extra markers.

Future `--clocked` capture runs record session IDs, format/source fingerprints,
raw audio, transmitter timing labels, device settings, and failed-attempt logs.
Actual capture must be started separately for an overnight burst. Keep room/session
test sets out of training before any decoder tuning. Source-derived alignment of
old recordings is diagnostic only; it is not verified oracle segmentation.

The prepared overnight command is intentionally explicit and loudness-controlled;
run it only when ready, with a fresh channel tag:

```bash
python3 src/collect_dataset.py --clocked --channel clocked_office_v1 \
  --limit 500 --input-device AUKEY --output-device "<speaker>" \
  --gain 1 --max-hang-streak 8 --hang-pause 3
```

The collector self-tests first, refuses to mix formats in a channel, logs every
attempt, saves raw capture plus transmitter timing, and can resume after hangs.
Use `--dry-run` to inspect the corpus without touching audio devices.
Explicit device selectors now reject unknown/ambiguous names; numeric device IDs
are supported. The self-test locates a sustained 1 kHz tone instead of assuming
the loudest captured sound is the test signal. For clocked formats, a preserved
raw capture counts as complete on resume even if the legacy trimmer rejected it.
Unexpected worker exits retain their stderr and stop the run after the configured
failure streak instead of being mislabeled indefinitely as quiet skips.

### Real clocked recordings

`b4bl.clocked_receiver` is an experimental frontend that detects the marker's
spectral sweep rather than waveform phase and filters low-frequency microphone
noise before phoneme classification. It does not change transmitted sounds and
does not replace the historical/default decoder.

The frozen attempt-level split retains capture failures, groups identical concept
sequences across prosodies/channels, and excludes all room 3 audio from training:

```bash
python3 -m b4bl.real_clocked_experiment \
  --split experiments/real-clocked-20260917/split.json \
  --output experiments/my-real-clocked-run
```

This silent experiment saves a separate model and compares the original receiver,
the new frontend alone, and real-data training on held-out recordings. Load its
model with `joblib.load` and use `b4bl.clocked_receiver.decode(audio, model)`;
`read_audio(path)` handles both integer and floating-point WAVs. Earlier room 3
diagnostic pilots are reported separately from the full collection. The room is
not a pristine unseen development environment because those pilots informed
diagnosis. Acoustic acceptance remains distinct from CRC-verified delivery.

The first frozen comparison (`experiments/real-clocked-20260917/run1/summary.json`)
trained on 1,191 aligned recordings from three configurations. Exact multi-word
recovery on composition holdouts was 68/68 MacBook→AUKEY, 68/68 Bluetooth→AUKEY,
and 67/68 Bluetooth→MacBook in room 2. The full room 3 run, excluded from training,
scored 232/328 (70.7%), including a missing capture as unsuccessful. Its 327
available multi-word raw recordings all had the expected marker count, but 23
messages were wrongly accepted and 72 rejected. This is strong same-environment
recognition, not 90% generalization to new rooms. No waveform redesign was needed.

Continued development found that the pitch tracker was treating low-periodicity
reverb/noise after a tone release as additional pitch. A versioned feature profile
that ignores frames below 0.5 periodicity raises the training-excluded room 3
multi-word result to 282/328 accepted-exact (86.0%), with 11 wrong acceptances,
34 rejections, and one missing capture. Its best hypotheses are correct on 300/328
(91.5%). The correct whole message is within the top two candidates per word in
320/328 (97.6%) and within the top three in 325/328 (99.1%); those are diagnostic
oracle bounds, not delivered accuracy.

`verified_clocked.decode` now supports bounded candidate-list search. An alternate
path is accepted only when the complete packet is canonical and its CRC validates.
The compact v2 decoder also prunes candidates that cannot occupy fixed header or
CRC positions before beam expansion. In a new-room run with a model and decoder
settings frozen beforehand, all 500 recordings had complete marker framing. Raw
top-path decoding was exact on 216/500 packets; structural/CRC list selection
repaired another 263, producing 479/500 (95.8%) verified exact deliveries, 21
rejections, and zero wrong acceptances. The exact 95% interval is 93.65–97.38%.
This clears 90% for that measured condition, not for arbitrary acoustic channels.
The aggregate is in `experiments/compact-room5-v2-full/summary.json`.

A separate 200-packet AUKEY-microphone/Bluetooth-speaker replication initially
scored only 95/200 because the marker detector's fixed 0.65 threshold rejected 94
clearly audible packets whose marker scores clustered near 0.63. A threshold audit
prompted by human listening disproved the earlier channel-loss diagnosis. The
versioned 0.55 detector recovers the exact expected marker count in all 199
available AUKEY recordings and in all 1,994 prior real captures, while finding no
markers in the corresponding 1,994 pre-message background windows. Re-decoding
delivers 190/200 packets exactly (95.0%), rejects nine, accepts zero wrong packets,
and counts one missing capture; the exact 95% delivery interval is 91.00–97.58%.
Because the threshold was selected after inspecting this corpus, this is a
development result that needs a fresh prospective confirmation. Arbitrary speech,
music, and machinery remain untested negative controls. The corrected aggregate
and audit are in `experiments/compact-aukey-room6-v2-threshold055/`.

A subsequent 500-packet AUKEY/Bluetooth batch added continuous music in the same
room. The music stopped during packets 98–302 and resumed at packet 303, producing
a useful natural A/B comparison. With the same frozen model and decoder, the quiet
middle delivered 204/205 packets exactly (99.5%), while the two music-on segments
delivered 121/295 (41.0%). The full mixed batch delivered 325/500 (65.0%), with
zero wrong acceptances. Marker counts were exact in 498/500 recordings, so the
dominant interference failure is phoneme ranking rather than packet detection.
The aggregate and measured-condition split are in
`experiments/compact-aukey-room6-musiclow-v1-full500/`.

A conservative interference-aware development model adds the 97 alignable
recordings from music-on packets 0–97 at sample weight two. Weight was selected
on packets 68–97 before evaluating the later music block. On music-on packets
303–499, verified delivery rises from 75/197 (38.1%) to 165/197 (83.8%), with
zero wrong acceptances. The quiet middle remains 204/205 (99.5%); room 5 improves
from 479/500 to 490/500, and the clean AUKEY corpus improves from 190/200 to
193/200. The remaining music gap is mostly candidate coverage: 28 of 32 rejected
packets have at least one expected word outside the retained top five. The model
is promising but remains below the 90% interference target. Development results
are in `experiments/music-interference-model-20260918/`.

Candidate-search tuning uses the model trained on music packets 0–67 and only
packets 68–97 for selection. Six acoustic candidates with a 50,000-path beam
deliver 29/30 (96.7%) on that tuning slice, versus 26/30 for the five-candidate,
10,000-path baseline. With the final music-aware model, the selected search gives
205/205 on the quiet middle, 494/500 on room 5, and 194/200 on the clean AUKEY
corpus, with zero wrong acceptances. The inspected later music block was not
re-evaluated under these settings. A fresh 200-packet music-interference capture
evaluated the frozen model and search configuration together. It delivered
159/200 packets (79.5%, exact 95% interval 73.23–84.87%) with zero wrong
acceptances, so the wider-search configuration did not meet the 90% target.
Music remained stable through the run at a median -39.23 dBFS, essentially the
same as the prior music-on corpus. Thirty of the 41 failed packets have only one
expected word outside the top-six candidates; 26 are isolated to `CLEAN`
(`Hc Hc`), `COME` (`Hc`), or `MOVE` (`Hr`). The next development step is targeted
high-band contour robustness using this failed-validation batch, followed by a
new untouched music capture. Results and diagnosis are in
`experiments/compact-aukey-room6-music-validation-v2-full200/`.

That failed-validation corpus was then split for the next development iteration:
attempts 0–149 joined the existing training data, while attempts 150–199 remained
a tuning holdout. A plain sample weight of one, with no special weighting for the
problematic `Hc`/`Hr` classes, decodes all 49 available tuning packets; one of the
50 attempts is missing. It also delivers 181/197 (91.9%) on the earlier music
block, 204/205 on its quiet middle, 491/500 in room 5, and 199/200 on the clean
AUKEY corpus, with zero wrong acceptances. Folding the final 50 attempts into
training was rejected because clean AUKEY delivery fell to 181/200. The selected
development model is therefore frozen at SHA256
`3e096f33384da5ef707ac563fc3a48edc8abce86c70c2da515d661ffac1c110f`.
These are development results; a new untouched 200-packet music run is required
for a prospective 90% claim. The selection and regression ledger is in
`experiments/music-interference-model-v2-20260918/development-summary.json`.

The untouched follow-up run passes that prospective test. With the frozen model
hash and search settings, it delivers 192/200 packets exactly (96.0%; exact 95%
interval 92.27–98.26%) and accepts zero wrong packets. All 200 raw recordings are
present. The collector flagged six captures, and all six account for rejections;
among the 194 collector-successful attempts, delivery is 192/194 (99.0%). Music
remained present throughout at a median -38.16 dBFS. This supports promotion for
the represented AUKEY/Bluetooth/room/low-music condition. Different rooms and
interference sources, plus continuous presence detection, remain separate
generalization problems. Results are in
`experiments/compact-aukey-room6-music-validation-v3-full200/`.

### Streaming desktop runtime

`b4bl.runtime_receiver.StreamingPacketReceiver` turns the validated batch decoder
into an incremental receiver without depending on any particular audio device. It
keeps a bounded rolling buffer, tracks the idle noise floor, wakes only after two
spectral markers have a legal symbol-clock interval, estimates a noise-relative
marker SNR, and freezes one bounded software gain for the whole packet. It tries
CRC-verified decoding at each possible closing marker, stays silent for isolated
markers and incomplete non-packets, and returns optional `ACK` or `SORRY REPEAT`
audio according to the existing reply policy. Applications can call
`suppress_for()` during playback to prevent a droid from waking on its own reply.

A chunked replay of all 200 untouched music-validation recordings preserves the
batch result: 192/200 exact (96.0%), eight rejected, and zero wrong accepted. On
this Mac, replay used 5.9% of real time. Accepted packets were recognized a mean
103 ms and maximum 148 ms of audio-time after the closing marker; decoder compute
averaged 234 ms and reached 463 ms. These timings establish desktop feasibility,
not robot-hardware performance. The runtime replay is a development evaluation
because its one-second context setting was selected after comparing it with batch
decoding. Its summary is in
`experiments/runtime-replay-v3-full200-preroll1/summary.json`.

Replay a labeled channel without opening an audio device:

```bash
python3 src/replay_runtime.py \
  --channel compact_aukey_room6_music_validation_v3 \
  --model experiments/music-interference-model-v2-20260918/real_classifier.joblib \
  --output experiments/my-runtime-replay
```

The desktop listener prints JSON events and can save suggested replies. Playback
requires the explicit `--play-replies` flag and an explicit output device:

```bash
python3 src/listen_runtime.py \
  --model experiments/music-interference-model-v2-20260918/real_classifier.joblib \
  --input-device "MacBook Pro Microphone" \
  --reply-directory runtime-replies
```

Add `--play-replies --output-device "<speaker>"` for audible responses. The
listener suppresses detection for the reply duration plus 250 ms to avoid waking
on itself. Successful packets remain silent unless `--confirm-success` is also
set; rejected packet-like transmissions use the repeat response by default.

The ambient benchmark below now measures development false wakes and mixed-packet
interference. Software normalization cannot improve physical signal-to-noise
ratio; it keeps a sufficiently audible packet in the classifier's numeric range.
Actual microphone gain control remains hardware-specific.

### Ambient interference benchmark

`src/ambient_benchmark.py` turns local long-form media into a reproducible
interference benchmark without copying or committing the source videos. The
current library contains 64 sources totaling 14.22 hours. A hash-based split made
before detector tuning reserves 16 complete videos for validation and assigns 48
to development; clips from one video can never cross that boundary. Five
deterministic clips per source span -45.45 to -2.87 dBFS and put between 0.04% and
34.04% of their energy in the 1.1-3.5 kHz marker band.

The complete development partition streamed 10.48 hours through the receiver. It
produced 30 isolated marker candidates, no legal multi-marker wake, no accepted
false packet, and no spoken repeat. This is strong development evidence for the
cadence gate, but it is not the sealed validation result.

Controlled digital mixtures expose a separate lexical problem. The frozen music
model decodes 39/40 of the selected clean packet recordings, but on arbitrary
development-source interference it delivered only 5/10 at +30 dB, 2/10 at +24
dB, 1/10 at +18 dB, and 3/10 at +12 dB in the initial sweep, with zero wrong
accepts. Two ambient-augmented random-forest variants did not make a meaningful
improvement; the stronger one delivered 9/20, 5/20, 1/20, and 4/20 at those SNRs.
The likely bottleneck is upstream of the classifier: overlapping periodic sound
causes the current pitch tracker either to follow the interferer or discard frames
at its periodicity gate. More copies of the same features cannot restore evidence
that feature extraction removed.

The next development step is therefore a foreground frontend comparison on the
development split: harmonic-ridge tracking constrained by the known droid pitch
ranges, source enhancement using the idle noise estimate, and a time-frequency
representation that retains multiple pitch candidates. Keep packet timing, CRC,
and the two-marker presence gate fixed during that comparison. Freeze the winning
frontend and model before running mixtures or continuous negative replay on the
16 validation sources.

That comparison is now complete. All candidates used the same real, synthetic,
prior-music, and development-ambient training foundation, followed by the same
20 packet/ambient/start cases at each of +30, +24, +18, and +12 dB. The corrected
factorial results were:

| Frontend | +30 | +24 | +18 | +12 | Total |
|---|---:|---:|---:|---:|---:|
| Legacy pitch features | 9/20 | 6/20 | 2/20 | 0/20 | 17/80 |
| Constrained spectral ridge | 8/20 | 8/20 | 5/20 | 1/20 | 22/80 |
| In-window spectral subtraction | 7/20 | 5/20 | 3/20 | 0/20 | 15/80 |
| Multi-candidate time-frequency | 9/20 | 9/20 | 9/20 | 4/20 | 31/80 |

The multi-candidate frontend is the lexical winner. Scaling it from 600 to 1,200
ambient-augmented recordings adds little by itself. Lowering the marker detector
threshold from .55 to .40 raises end-to-end delivery to 42/80, but creates much
more internal candidate churn. A 1.16-hour development negative pilot at .40 had
zero accepted false packets and zero spoken repeats, while producing 163
incomplete, 55 rejected-nonpacket, and 74 isolated-marker events. Do not promote
the lower threshold without a better marker tracker and a full negative replay.

An oracle-marker diagnostic supplies marker positions measured from each unmixed
packet while leaving the mixed waveform and lexical decoder unchanged. The scaled
multi-candidate model then delivers 78/80: 20/20 at +30 and +24 dB, 19/20 at +18
and +12 dB, and zero wrong accepts. This is not a deployable result; it proves the
lexical frontend now exceeds 90% when framing is correct. On recorded regressions,
the candidate delivers 189/200 (94.5%) on the prior low-music validation corpus
and 100/100 on the clean room-5 prefix, with zero wrong accepts.

Marker diagnostics then showed that the batch spectral detector already retained
all true markers. Streaming failures came from the amplitude admission rule: a
valid first marker below 3 dB relative to the tracked broadband RMS was discarded,
making the required opening `SYNC` interval impossible. The frozen candidate
keeps the original normalized .55 marker-shape threshold, removes the redundant
amplitude gate, and requires the first interval to have the known `SYNC` duration.
On 200 development mixtures it delivers 49/50 at +30 dB, 48/50 at +24 dB,
41/50 at +18 dB, and 26/50 at +12 dB, with zero wrong accepts. A complete
10.48-hour development negative replay produces no packet activations or spoken
repeats.

The candidate and test plan were committed before opening validation. The single
prospective run over all 16 reserved ambient sources delivers 63/64 at +24/+30
dB: 98.44%, exact 95% interval 91.60-99.96%, and zero wrong accepts. The declared
primary endpoint therefore passes. Secondary delivery is 27/32 at +18 dB and
21/32 at +12 dB; all-SNR delivery is 111/128. Continuous replay of all 16
validation sources covers 3.744 hours and produces seven isolated markers, zero
packet activations, zero spoken repeats, and zero accepted false packets.

The supported claim is consequently scoped to digital mixtures with packet audio
at least 24 dB above ambient: verified delivery exceeds 90%, its exact confidence
bound exceeds 90%, and observed wrong acceptance is zero. At +18 and +12 dB the
receiver remains fail-closed but does not meet the delivery target. Digital
mixtures do not reproduce simultaneous room transfer, speaker distortion, or
microphone AGC, so a future physical-interference recording campaign would be a
separate external-validity test rather than more model development on this corpus.

Create the source manifest and acoustic characterization:

```bash
python3 src/ambient_benchmark.py inventory \
  --output experiments/ambient-benchmark-v1-inventory
python3 src/ambient_benchmark.py characterize \
  --output experiments/ambient-benchmark-v1-characterization
```

Run a development mixture sweep or continuous negative replay:

```bash
python3 src/ambient_benchmark.py mixed \
  --split development --channel compact_room5_v2 --mixtures 500 \
  --factorial --snrs 30,24,18,12 --no-marker-snr-gate \
  --output experiments/ambient-mixtures-development
python3 src/ambient_benchmark.py negative \
  --split development --no-marker-snr-gate \
  --output experiments/ambient-negative-development
```

`b4bl.ambient_model_experiment` reconstructs the real, synthetic, and prior-music
training foundation and adds only development-source ambient mixtures. Its models
are development artifacts; do not promote one without clean/music regression
checks and one untouched validation-source evaluation.

Prepare or collect a compact protected-packet corpus with a fresh channel tag:

```bash
python3 src/collect_dataset.py --compact-packets --dry-run --limit 500 --seed 42
python3 src/collect_dataset.py --compact-packets \
  --channel compact_room4_v2 --limit 500 --seed 42 \
  --input-device "MacBook Pro Microphone" --output-device "<speaker>" \
  --gain 1 --max-hang-streak 8 --hang-pause 3
```

The dry run is silent. The live command performs the usual audible self-test and
records raw audio plus timing labels. After capture, evaluate verified delivery
without opening audio devices:

```bash
python3 -m b4bl.compact_packet_experiment \
  --channel compact_room4_v2 \
  --model experiments/real-clocked-20260917/allrooms-periodicity05/real_classifier.joblib \
  --output experiments/compact-room4-v2
```

Spoken responses are optional application behavior, not a transport handshake.
`compact_clocked.spoken_reply_concepts(result)` stays silent after success by
default, returns `SORRY REPEAT` for a packet-like CRC rejection, and ignores short
or non-packet audio. Pass `confirm_success=True` to request an `ACK`; use
`encode_spoken_reply` to render it. `python3 src/render_packet_replies.py` writes
listening samples without opening an audio device.

When room 3 non-holdout recordings are included in training, its reserved
multi-word compositions score 67/68 (98.5% including one missing capture; 67/67
available), with no wrong acceptances. Other represented channels remain at
98.5–100%. This is the practical calibrated-environment path above 90%; the 86.0%
leave-one-room-out result remains the honest cross-room development figure.

## Layout

```
b4bl/
  generators.py   the locked Phase 0 sound palette (tonal + rough families)
  phonology.py    Layer 1: acoustic primitive inventory (the "phonemes")
  lexicon.py      Layer 2: morpheme vocabulary + grammar + spelling fallback
  prosody.py      Layer 3: affect as transforms on expressive dimensions
  codec.py        meaning <-> phoneme-sequence <-> audio
  runtime_receiver.py  streaming wake, level normalization, decode + reply policy
src/
  astromech_synth.py   Phase 0 A/B/C style comparison (historical)
  replay_runtime.py    silent chunked receiver replay + metrics
  listen_runtime.py    desktop microphone event/reply harness
  whistle_synth.py     Phase 0 whistle-direction study
  palette_synth.py     the palette listening-test renderer
  demo_language.py     renders sentences + prosody + inventory to audio
tests/            symbolic round-trip + inventory sanity tests
audio_samples/    rendered demos (WAV/MP3)
```

## Try it

```bash
pip install -r requirements.txt
python3 src/demo_language.py     # renders sentences, prosody, inventory to audio_samples/
python3 -m pytest -q             # round-trip + unambiguity tests
```

Then listen to `audio_samples/`:
- `phoneme_inventory.*` — the ~16 acoustic primitives in a row
- `sentences_neutral.*` — example sentences ("is your energy low?", "warning: obstacle ahead", …)
- `prosody_compare.*` — the SAME sentence rendered neutral / uncertain / urgent / calm

## Design notes

Phase 0 listening tests settled the sound direction:

- **Pure tone reads as R2.** Added harmonics / ring-modulation read as
  "kazoo / SNES sound effects." Character lives in the **pitch gesture**, not the timbre.
- Sample-and-hold random burble read as low-budget sci-fi → dropped.
- Grit is per-family: wrong on whistles, right on raspberries.

The full plan (three layers, phased roadmap, open decisions) lives in the
project's design docs. Key open decision: exactly which acoustic dimensions are
lexical vs. expressive — the current split is documented in `b4bl/phonology.py`
as a reviewed proposal.

## License

MIT. All audio is newly synthesized; no film recordings are used or redistributed.
