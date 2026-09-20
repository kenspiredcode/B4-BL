// B4-BL synthesis — browser port of b4bl/generators.py + b4bl/phonology.py
//                     + b4bl/codec.py (encode path only).
//
// PORTING CONTRACT
// ----------------
// This module must produce audio equivalent to `codec.encode(concepts, prosody)`
// in the Python implementation. Do not "improve" the sound here — any change to
// how B4-BL sounds belongs in the Python implementation first, because the
// trained decoder depends on it.
//
// Everything that varies (band centers, contour swing, durations, per-word
// phoneme sequences, gap timing, prosody presets) is read from
// data/language.json, which tools_export_language.py generates. Only the DSP is
// written out here, because DSP is code rather than data.
//
// Decode is deliberately absent: it needs a ~140 MB trained classifier.

const SR = 44100;          // b4bl.generators.SR
const RAMP_SEC = 0.004;    // edge ramp in generators.envelope

let ctx = null;

/** Lazily create the AudioContext; browsers require a user gesture first. */
export function getContext() {
  if (!ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    ctx = new AC({ sampleRate: SR });
  }
  if (ctx.state === 'suspended') ctx.resume();
  return ctx;
}

// ---------------------------------------------------------------------------
// low-level dsp — generators.py
// ---------------------------------------------------------------------------

/** generators.smooth_contour: smoothstep-interpolated pitch curve. */
function smoothContour(dur, points) {
  const n = Math.floor(SR * dur);
  const out = new Float64Array(n);
  if (n <= 0) return out;

  const b = points.map(p => Math.floor(p[0] * n));
  const hz = points.map(p => p[1]);

  for (let i = 0; i < points.length - 1; i++) {
    const lo = b[i], hi = b[i + 1];
    if (hi <= lo) continue;
    const span = hi - lo;
    for (let k = 0; k < span; k++) {
      const x = k / span;                 // linspace(endpoint=False)
      const s = x * x * (3 - 2 * x);      // smoothstep
      out[lo + k] = hz[i] + (hz[i + 1] - hz[i]) * s;
    }
  }
  for (let i = b[b.length - 1]; i < n; i++) out[i] = hz[hz.length - 1];
  for (let i = 0; i < Math.min(b[0], n); i++) out[i] = hz[0];
  return out;
}

/** generators.envelope */
function envelope(n, kind = 'even') {
  const e = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = i / Math.max(n, 1);
    if (kind === 'stab') e[i] = Math.exp(-5 * a);
    else if (kind === 'swell') e[i] = Math.pow(Math.sin(Math.PI * a), 0.7);
    else if (kind === 'decay') e[i] = Math.pow(1 - a, 0.6);
    else e[i] = 1;
  }
  const ramp = Math.max(1, Math.floor(RAMP_SEC * SR));
  for (let i = 0; i < Math.min(ramp, n); i++) {
    e[i] *= ramp === 1 ? 0 : i / (ramp - 1);
    const j = n - 1 - i;
    if (j >= 0) e[j] *= ramp === 1 ? 0 : i / (ramp - 1);
  }
  return e;
}

/** generators._phase: cumulative-sum phase integration. */
function phaseFromFreq(freq) {
  const out = new Float64Array(freq.length);
  let acc = 0;
  for (let i = 0; i < freq.length; i++) {
    acc += freq[i];
    out[i] = 2 * Math.PI * acc / SR;
  }
  return out;
}

/** generators._apply_vib */
function applyVib(freq, vib) {
  if (!vib) return freq;
  const [rate, depth] = vib;
  const out = new Float64Array(freq.length);
  for (let i = 0; i < freq.length; i++) {
    out[i] = freq[i] * (1 + depth * Math.sin(2 * Math.PI * rate * (i / SR)));
  }
  return out;
}

/** generators.tonal: pure sine following the pitch gesture. */
function tonal(points, dur, env, vib) {
  const freq = applyVib(smoothContour(dur, points), vib);
  const phase = phaseFromFreq(freq);
  const e = envelope(phase.length, env);
  const out = new Float32Array(phase.length);
  for (let i = 0; i < phase.length; i++) out[i] = Math.sin(phase[i]) * e[i];
  return out;
}

/**
 * generators.gargle: amplitude-flutter roll ("brrrp"). The tremolo is a hard
 * square gate via sign(sin(...)), not a smooth sine — that squareness is the
 * character.
 */
function gargle(dur, center, tremoloHz = 45, depth = 0.9) {
  const freq = smoothContour(dur, [[0, center], [1, center * 0.95]]);
  const phase = phaseFromFreq(freq);
  const e = envelope(phase.length, 'even');
  const out = new Float32Array(phase.length);
  for (let i = 0; i < phase.length; i++) {
    const sgn = Math.sign(Math.sin(2 * Math.PI * tremoloHz * (i / SR)));
    const trem = (1 - depth) + depth * (0.5 + 0.5 * sgn);
    out[i] = Math.sin(phase[i]) * trem * e[i];
  }
  return out;
}

/**
 * generators.raspberry: buzz + noise through a 2-pole lowpass, with a 42 Hz
 * amplitude flutter. The Python version uses scipy.signal.butter/lfilter; this
 * uses a matched biquad, so the result is very close but not bit-identical.
 * Only two phonemes (LdR, MdR) use this class.
 */
function raspberry(dur, center) {
  const n = Math.floor(SR * dur);
  const freq = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    freq[i] = center + (center * 0.7 - center) * (n > 1 ? i / (n - 1) : 0);
  }
  const phase = phaseFromFreq(freq);
  const raw = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    // sawtooth(width=0.5) is a triangle wave over the phase cycle.
    const frac = (phase[i] / (2 * Math.PI)) % 1;
    const t = frac < 0 ? frac + 1 : frac;
    const tri = t < 0.5 ? (4 * t - 1) : (3 - 4 * t);
    raw[i] = 0.8 * tri + 0.35 * (Math.random() * 2 - 1);
  }
  const filtered = lowpass2(raw, 2600);
  const e = envelope(n, 'decay');
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const flutter = 0.55 + 0.45 * Math.sin(2 * Math.PI * 42 * (i / SR));
    out[i] = filtered[i] * flutter * e[i] * 1.6;
  }
  return out;
}

/** Two-pole Butterworth lowpass, applied forward like scipy's lfilter. */
function lowpass2(x, cutoffHz) {
  const w = 2 * Math.PI * cutoffHz / SR;
  const cosw = Math.cos(w), sinw = Math.sin(w);
  const alpha = sinw / Math.SQRT2;              // Q = 1/sqrt(2)
  const b0 = (1 - cosw) / 2, b1 = 1 - cosw, b2 = (1 - cosw) / 2;
  const a0 = 1 + alpha, a1 = -2 * cosw, a2 = 1 - alpha;
  const out = new Float64Array(x.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const y = (b0 / a0) * x[i] + (b1 / a0) * x1 + (b2 / a0) * x2
            - (a1 / a0) * y1 - (a2 / a0) * y2;
    x2 = x1; x1 = x[i]; y2 = y1; y1 = y;
    out[i] = y;
  }
  return out;
}

// ---------------------------------------------------------------------------
// phoneme rendering — phonology.Phoneme
// ---------------------------------------------------------------------------

/** phonology.Phoneme._contour_points */
function contourPoints(center, contour, swing) {
  const c = center, s = center * swing;
  switch (contour) {
    case 'FLAT':   return [[0, c], [1, c]];
    case 'RISE':   return [[0, c - s], [1, c + s]];
    case 'FALL':   return [[0, c + s], [1, c - s]];
    case 'ARCH':   return [[0, c - s], [0.5, c + s], [1, c]];
    case 'DIP':    return [[0, c + s], [0.5, c - s], [1, c]];
    case 'SCOOP':  return [[0, c], [0.2, c - s * 0.6], [0.8, c + s], [1, c + s]];
    case 'DOUBLE': return [[0, c - s * 0.5], [0.25, c + s * 0.5],
                           [0.5, c - s * 0.3], [0.75, c + s * 0.5], [1, c]];
    default:       return [[0, c], [1, c]];
  }
}

/**
 * prosody.Prosody.apply — perturbs only expressive dimensions. The lexical
 * features (band, contour, duration, class) a decoder keys on are never touched.
 */
function applyProsody(pros, points, dur, env, vib) {
  const conf = Math.max(0, Math.min(1, pros.confidence));
  const urg = Math.max(0, Math.min(1, pros.urgency));

  dur = dur * (1.0 - 0.5 * urg) * (1.0 + 0.7 * (1 - conf));

  let [vibRate, vibDepth] = vib || [7, 0.03];
  vibDepth = vibDepth + (1 - conf) * 0.18;
  vibRate = vibRate * (1.0 + 0.5 * urg);
  vib = [vibRate, vibDepth];

  if (urg > 0.6) env = 'stab';
  else if (conf < 0.45) env = 'swell';

  const exagg = Math.max(0.4, 1.0 + 1.0 * urg - 0.6 * (1 - conf));
  const bright = 1.0 + 0.12 * urg - 0.10 * (1 - conf);

  if (points.length >= 2) {
    const center = points.reduce((a, p) => a + p[1], 0) / points.length;
    points = points.map(([f, hz]) => [f, (center + (hz - center) * exagg) * bright]);
  } else if (points.length) {
    points = points.map(([f, hz]) => [f, hz * bright]);
  }
  return [points, dur, env, vib];
}

/** phonology.Phoneme.render */
export function renderPhoneme(name, language, prosodyName = 'neutral') {
  const p = language.phonemes[name];
  if (!p) throw new Error(`unknown phoneme: ${name}`);

  let dur = language.dur_sec[p.dur];
  let points = contourPoints(p.center_hz, p.contour, language.contour_swing);
  let env = p.dur === 'SHORT' ? 'stab' : 'even';
  let vib = [7, 0.03];

  // A long FLAT tone with vibrato reads as a cheap-sci-fi UFO warble; keep a
  // held pitch dead steady so it reads as a deliberate hold.
  if (p.contour === 'FLAT' && p.dur === 'LONG') vib = null;

  const pros = language.prosody[prosodyName];
  if (pros) [points, dur, env, vib] = applyProsody(pros, points, dur, env, vib);

  if (p.cls === 'TONE') return tonal(points, dur, env, vib);
  if (p.cls === 'GARGLE') {
    // Low hum pulses want a gentler, slower flutter so they read as humming
    // rather than a buzzy roll.
    return p.center_hz < 400
      ? gargle(dur, p.center_hz, 22, 0.5)
      : gargle(dur, p.center_hz);
  }
  if (p.cls === 'RASP') return raspberry(dur, Math.max(250, p.center_hz / 3));
  throw new Error(`unsupported sound class: ${p.cls}`);
}

// ---------------------------------------------------------------------------
// word / sentence assembly — codec.py encode path
// ---------------------------------------------------------------------------

function silence(dur) {
  return new Float32Array(Math.max(0, Math.floor(SR * dur)));
}

/** codec._render_geminate: a short broadband click between identical phonemes. */
function geminate(language) {
  const n = Math.floor(SR * language.timing.geminate_dur);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (Math.random() * 2 - 1) * Math.exp(-6 * i / Math.max(n - 1, 1)) * 0.9;
  }
  return out;
}

/** codec._render_phoneme_seq */
function renderSeq(names, language, prosodyName) {
  const clips = [];
  names.forEach((name, i) => {
    clips.push(renderPhoneme(name, language, prosodyName));
    if (i !== names.length - 1) clips.push(silence(language.timing.phone_gap));
  });
  return clips;
}

/**
 * codec._render_rep — repetition is a LEXICAL axis of meaning. The pulse count
 * and rhythm ARE the word (ALARM, WORKING), so prosody must never change them.
 */
function renderRep(rep, language, prosodyName) {
  const period = 1.0 / Math.max(rep.rate, 0.3);
  const clips = [];
  for (let k = 0; k < rep.count; k++) {
    const pulse = rep.alternate ? [rep.unit[k % rep.unit.length]] : rep.unit;
    clips.push(...renderSeq(pulse, language, prosodyName));
    if (k !== rep.count - 1) clips.push(silence(Math.max(0.02, period - 0.16)));
  }
  return clips;
}

/** codec._render_concept, including the spelling fallback. */
function renderConcept(concept, language, prosodyName) {
  const body = language.morphemes[concept];
  if (body && body.rep) return renderRep(body.rep, language, prosodyName);
  if (body) return renderSeq(body, language, prosodyName);

  const seq = [...(language.spell_marker || [])];
  for (const ch of concept.toLowerCase()) {
    const code = (language.char_to_phones || {})[ch];
    if (code) seq.push(...code);
  }
  return seq.length ? renderSeq(seq, language, prosodyName) : [];
}

/** Concatenate float chunks into one buffer. */
function concat(chunks) {
  const total = chunks.reduce((a, c) => a + c.length, 0);
  const out = new Float32Array(total);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

/** generators.normalize — peak-normalize, matching what write_wav does. */
function normalize(x, peak = 0.9) {
  let m = 0;
  for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i]));
  if (!m) return x;
  const g = peak / m;
  for (let i = 0; i < x.length; i++) x[i] *= g;
  return x;
}

/**
 * codec.encode — a concept sequence to one normalized mono buffer.
 *
 * @param {string[]} concepts e.g. ['QUERY','YOU','ENERGY','LOW']
 * @param {string} prosodyName 'neutral' | 'uncertain' | 'urgent' | 'calm'
 * @param {object} language parsed data/language.json
 * @returns {Float32Array}
 */
export function encode(concepts, prosodyName, language) {
  const clips = [];
  concepts.forEach((c, i) => {
    clips.push(...renderConcept(c, language, prosodyName));
    if (i !== concepts.length - 1) clips.push(silence(language.timing.word_gap));
  });
  return normalize(concat(clips));
}

/** Concept sequence to an AudioBuffer ready for playback. */
export function renderConcepts(concepts, prosodyName, language) {
  const samples = encode(concepts, prosodyName, language);
  const audioCtx = getContext();
  const buf = audioCtx.createBuffer(1, Math.max(samples.length, 1), SR);
  buf.copyToChannel(samples, 0);
  return buf;
}

/** Play an AudioBuffer; returns the source node so callers can stop it. */
export function play(buffer) {
  const audioCtx = getContext();
  const src = audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(audioCtx.destination);
  src.start();
  return src;
}

export { SR };
