// The four-prosody hook. Static files, no synthesis, no language data —
// this must work even if everything else on the page fails to initialize.

const PROSODIES = [
  ['neutral',   'flat, informational'],
  ['uncertain', 'hedging, rising'],
  ['urgent',    'fast, compressed, insistent'],
  ['calm',      'slow, settled'],
];

export function initHear() {
  const host = document.getElementById('prosody-grid');
  if (!host) return;

  for (const [name, desc] of PROSODIES) {
    const row = document.createElement('div');
    row.className = 'prosody-row';

    const label = document.createElement('div');
    label.innerHTML =
      `<div class="name">${name}</div><div class="desc">${desc}</div>`;

    // Staged into docs/audio/ by tools_build_demo.py. GitHub Pages serves
    // docs/ as the site root, so ../audio_samples/ is unreachable.
    const audio = document.createElement('audio');
    audio.controls = true;
    audio.preload = 'none';
    audio.src = `audio/prosody_${name}.wav`;

    row.append(label, audio);
    host.appendChild(row);
  }
}

export { PROSODIES };
