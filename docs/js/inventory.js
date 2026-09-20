// Phoneme grid and vocabulary table — both generated from language.json.

import { renderConcepts, play } from './synth.js';

const BAND_ORDER = ['SUB', 'LOW', 'MID', 'HIGH', 'VHIGH'];
const VOCAB_ROWS = [
  ['verb', 7], ['spatial', 6], ['object', 6], ['state', 6],
  ['pronoun', 5], ['social', 5],
];
const CODE_SAMPLES = ['STOP', 'GO', 'SELF', 'YOU', 'WARN', 'ENERGY'];

export function initInventory(language) {
  buildPhonemeGrid(language);
  buildVocabTable(language);
}

function buildPhonemeGrid(language) {
  const host = document.getElementById('phoneme-grid');
  if (!host) return;

  // Only the 32-phoneme lexical inventory, ordered by band then contour.
  // Aliases, hum subunits and spelling-only phonemes are omitted: the synth
  // still renders them, but they are not primitives the page should present.
  const entries = Object.entries(language.phonemes)
    .filter(([, p]) => p.inventory)
    .sort((a, b) => {
      const band = BAND_ORDER.indexOf(a[1].band) - BAND_ORDER.indexOf(b[1].band);
      return band || a[0].localeCompare(b[0]);
    });

  for (const [name, p] of entries) {
    const cell = document.createElement('button');
    cell.className = 'phoneme-cell';
    cell.innerHTML =
      `<div class="pname">${name}</div>` +
      `<div class="pmeta">${p.band.toLowerCase()} · ${p.contour.toLowerCase()}</div>` +
      `<div class="pmeta">${p.dur === 'LONG' ? 'long' : 'short'}` +
      `${p.cls !== 'TONE' ? ' · ' + p.cls.toLowerCase() : ''}</div>`;
    cell.title = `${Math.round(p.center_hz)} Hz · ${p.seconds}s · ${p.cls}`;
    cell.addEventListener('click', () => {
      // A phoneme is not a word, so it is rendered as a bare sequence by
      // borrowing the concept renderer with a one-phoneme pseudo-morpheme.
      const shim = { ...language, morphemes: { __p: [name] } };
      play(renderConcepts(['__p'], 'neutral', shim));
    });
    host.appendChild(cell);
  }
}

function buildVocabTable(language) {
  const host = document.getElementById('vocab-table');
  if (!host) return;

  const cats = document.createElement('table');
  cats.innerHTML =
    '<thead><tr><th>Category</th><th>Count</th><th>Examples</th></tr></thead>';
  const tb = document.createElement('tbody');
  for (const [cat, n] of VOCAB_ROWS) {
    const words = language.categories[cat] || [];
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td><code>${cat}</code></td><td>${words.length}</td>` +
      `<td>${words.slice(0, n).map(w => `<code>${w}</code>`).join(' ')}</td>`;
    tb.appendChild(tr);
  }
  cats.appendChild(tb);

  const codes = document.createElement('table');
  codes.innerHTML =
    '<thead><tr><th>Concept</th><th>Phonemes</th><th></th></tr></thead>';
  const cb = document.createElement('tbody');
  for (const c of CODE_SAMPLES) {
    const body = language.morphemes[c];
    if (!Array.isArray(body)) continue;
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td><code>${c}</code></td>` +
      `<td><code>${body.join(' ')}</code></td>` +
      `<td>${body.length} phoneme${body.length > 1 ? 's' : ''}</td>`;
    tr.style.cursor = 'pointer';
    tr.addEventListener('click',
      () => play(renderConcepts([c], 'neutral', language)));
    cb.appendChild(tr);
  }
  codes.appendChild(cb);

  host.append(cats, codes);
}
