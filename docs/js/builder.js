// Sentence builder — grammar-driven composition.
//
// The user picks one of the templates from grammar.py and fills its slots. Each
// slot is restricted to one lexicon category, so every buildable sentence is
// guaranteed encodable. That is ~228,000 valid sentences without free-text
// parsing, which would mostly fail against a 215-word vocabulary.

import { renderConcepts, play, encode as encodeSamples, SR } from './synth.js';
import { drawSpectrogram } from './spectrogram.js';

const PROSODIES = [
  ['neutral', 'neutral'],
  ['uncertain', 'uncertain'],
  ['urgent', 'urgent'],
  ['calm', 'calm'],
];

let lang = null;
let current = null;   // active source node, so a new Play interrupts the old one

export function initBuilder(language) {
  lang = language;

  const templateSel = document.getElementById('template');
  language.templates.forEach((t, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = `${t.name} — ${describe(t)}`;
    templateSel.appendChild(opt);
  });
  templateSel.addEventListener('change', () => {
    buildSlots(Number(templateSel.value));
    refresh();
  });

  const prosodyBox = document.getElementById('prosody');
  prosodyBox.addEventListener('change', refresh);
  PROSODIES.forEach(([value, label], i) => {
    const id = `prosody-${value}`;
    const wrap = document.createElement('label');
    wrap.innerHTML =
      `<input type="radio" name="prosody" id="${id}" value="${value}"` +
      `${i === 0 ? ' checked' : ''}><span>${label}</span>`;
    prosodyBox.appendChild(wrap);
  });

  const presets = document.getElementById('presets');
  language.examples.forEach(ex => {
    const b = document.createElement('button');
    b.textContent = ex.english;
    b.addEventListener('click', () => {
      // Presets can use any grammar shape, so they drive the chain directly
      // rather than trying to reverse them into slot selections.
      setChain(ex.concepts);
      playConcepts(ex.concepts);
    });
    presets.appendChild(b);
  });

  document.getElementById('play').addEventListener('click', () => {
    const concepts = collectConcepts();
    if (concepts.length) playConcepts(concepts);
  });

  // Default to "query state", the shape of the sentence used throughout the
  // README and the four prosody samples.
  const defaultIndex = Math.max(
    0, language.templates.findIndex(t => t.name === 'query state'));
  templateSel.value = String(defaultIndex);
  buildSlots(defaultIndex);
  preset(['QUERY', 'YOU', 'ENERGY', 'LOW']);
  // refresh() also draws the spectrogram, so the canvas is never an empty box
  // on load. drawPreview renders audio without playing it, so it needs no
  // user gesture and creates no AudioContext.
  refresh();
}

/** Render and draw without playing (no user gesture required). */
function drawPreview(concepts) {
  if (!concepts.length) return;
  try {
    const samples = encodeSamples(concepts, currentProsody(), lang);
    drawSpectrogram(document.getElementById('spectrogram'), samples, SR);
    document.getElementById('duration').textContent =
      `${(samples.length / SR).toFixed(2)}s · ${concepts.length} words`;
  } catch (err) {
    console.error('preview failed:', err);
  }
}

/** A readable summary of a template's slots, e.g. "act + verb + (object)". */
function describe(t) {
  return t.slots
    .map(s => (s.optional ? `(${s.category})` : s.category))
    .join(' + ');
}

function buildSlots(templateIndex) {
  const tmpl = lang.templates[templateIndex];
  const host = document.getElementById('slots');
  host.textContent = '';

  tmpl.slots.forEach((slot, i) => {
    const sel = document.createElement('select');
    sel.dataset.slot = String(i);
    sel.dataset.category = slot.category;

    if (slot.optional) {
      const none = document.createElement('option');
      none.value = '';
      none.textContent = `— no ${slot.category} —`;
      sel.appendChild(none);
    }
    (lang.categories[slot.category] || []).forEach(word => {
      const opt = document.createElement('option');
      opt.value = word;
      opt.textContent = word;
      sel.appendChild(opt);
    });

    sel.addEventListener('change', refresh);
    host.appendChild(sel);
  });
}

/** Try to select a known concept list in the current slots. */
function preset(concepts) {
  const sels = [...document.querySelectorAll('#slots select')];
  let ci = 0;
  for (const sel of sels) {
    const want = concepts[ci];
    if (want && [...sel.options].some(o => o.value === want)) {
      sel.value = want;
      ci++;
    } else if (sel.querySelector('option[value=""]')) {
      sel.value = '';
    }
  }
}

/** Current slot selections -> concept list, dropping unfilled optional slots. */
export function collectConcepts() {
  return [...document.querySelectorAll('#slots select')]
    .map(sel => sel.value)
    .filter(Boolean);
}

function currentProsody() {
  const checked = document.querySelector('input[name="prosody"]:checked');
  return checked ? checked.value : 'neutral';
}

function refresh() {
  const concepts = collectConcepts();
  setChain(concepts);
  drawPreview(concepts);
}

/** Render the concepts -> phonemes breakdown under the builder. */
function setChain(concepts) {
  const host = document.getElementById('concept-chain');
  host.textContent = '';
  if (!concepts.length) {
    host.textContent = 'Pick some words.';
    return;
  }

  concepts.forEach(c => {
    const body = lang.morphemes[c];
    const phones = body && body.rep
      ? `${body.rep.count}× ${body.rep.unit.join(' ')} @ ${body.rep.rate}/s`
      : (body || []).join(' ');

    const span = document.createElement('span');
    span.className = 'word';
    span.innerHTML =
      `<span class="concept">${c}</span>` +
      `<span class="phones">${phones}</span>`;
    host.appendChild(span);
  });
}

function playConcepts(concepts) {
  const prosody = currentProsody();
  let buffer;
  try {
    buffer = renderConcepts(concepts, prosody, lang);
  } catch (err) {
    console.error('synthesis failed:', err);
    return;
  }

  if (current) {
    try { current.stop(); } catch { /* already ended */ }
  }
  current = play(buffer);

  const dur = document.getElementById('duration');
  dur.textContent = `${buffer.duration.toFixed(2)}s · ${concepts.length} words`;

  drawSpectrogram(document.getElementById('spectrogram'),
                  buffer.getChannelData(0), buffer.sampleRate);
}
