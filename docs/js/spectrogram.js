// Canvas spectrogram for the sentence builder.
//
// A small radix-2 FFT with a Hann window, drawn with the same magma-like ramp
// as the generated packet figure so the demo and the README look like one
// system. Display only — nothing here feeds the decoder.

const FFT_SIZE = 1024;
const HOP = 256;
const TOP_HZ = 3600;      // matches tools_render_packet_figure.py
const DB_RANGE = 65;

/** In-place iterative radix-2 FFT. */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/** Approximate magma: dark purple -> magenta -> orange -> pale yellow. */
function magma(t) {
  const stops = [
    [0.00, 8, 6, 24], [0.25, 87, 16, 110], [0.50, 187, 55, 84],
    [0.75, 249, 142, 9], [1.00, 252, 253, 191],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [p0, r0, g0, b0] = stops[i];
    const [p1, r1, g1, b1] = stops[i + 1];
    if (t <= p1) {
      const f = (t - p0) / (p1 - p0);
      return [r0 + (r1 - r0) * f, g0 + (g1 - g0) * f, b0 + (b1 - b0) * f];
    }
  }
  return [252, 253, 191];
}

export function drawSpectrogram(canvas, samples, sampleRate) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#0b0d12';
  ctx.fillRect(0, 0, W, H);
  if (!samples || samples.length < FFT_SIZE) return;

  const frames = Math.max(1, Math.floor((samples.length - FFT_SIZE) / HOP));
  const bins = Math.min(FFT_SIZE / 2,
                        Math.ceil(TOP_HZ / (sampleRate / FFT_SIZE)));

  const hann = new Float64Array(FFT_SIZE);
  for (let i = 0; i < FFT_SIZE; i++) {
    hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (FFT_SIZE - 1));
  }

  const mag = new Float32Array(frames * bins);
  const re = new Float64Array(FFT_SIZE);
  const im = new Float64Array(FFT_SIZE);
  let peak = -Infinity;

  for (let f = 0; f < frames; f++) {
    const off = f * HOP;
    for (let i = 0; i < FFT_SIZE; i++) {
      re[i] = (samples[off + i] || 0) * hann[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let b = 0; b < bins; b++) {
      const p = re[b] * re[b] + im[b] * im[b];
      const db = 10 * Math.log10(p + 1e-12);
      mag[f * bins + b] = db;
      if (db > peak) peak = db;
    }
  }

  const img = ctx.createImageData(W, H);
  for (let x = 0; x < W; x++) {
    const f = Math.min(frames - 1, Math.floor(x / W * frames));
    for (let y = 0; y < H; y++) {
      const b = Math.min(bins - 1, Math.floor((1 - y / H) * bins));
      const db = mag[f * bins + b];
      const t = Math.max(0, Math.min(1, (db - (peak - DB_RANGE)) / DB_RANGE));
      const [r, g, bl] = magma(t);
      const o = (y * W + x) * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = bl;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  // Band-center guides, matching the packet figure.
  ctx.strokeStyle = 'rgba(230,237,243,0.22)';
  ctx.fillStyle = 'rgba(154,167,180,0.9)';
  ctx.font = '10px ui-monospace, monospace';
  ctx.setLineDash([4, 4]);
  for (const [name, hz] of Object.entries({
    SUB: 260, LOW: 520, MID: 1000, HIGH: 1750, VHIGH: 2700,
  })) {
    if (hz > TOP_HZ) continue;
    const y = H - (hz / TOP_HZ) * H;
    ctx.beginPath();
    ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    ctx.fillText(name, W - 40, y - 3);
  }
  ctx.setLineDash([]);
}
