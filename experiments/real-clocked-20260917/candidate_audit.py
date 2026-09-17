"""Measure whether the expected words remain in the decoder's bounded candidate list.

This is a diagnostic/oracle analysis. It does not claim that a receiver knows the
reference or that top-k membership is successful delivery.
"""
from collections import Counter
import json
from pathlib import Path
import joblib
from b4bl import clocked_receiver as receiver

ROOT = Path(__file__).resolve().parents[2]
rows = json.loads((ROOT/'experiments/real-clocked-20260917/split.json').read_text())
model = joblib.load(ROOT/'experiments/real-clocked-20260917/periodicity05/real_classifier.joblib')
summary = Counter()
details = []
for row in rows:
    if row['channel'] != 'clocked_bluetooth_macmic_room3_verified_v1':
        continue
    summary['attempts'] += 1
    path = ROOT/'recordings'/row['raw_file']
    if not path.exists():
        summary['missing'] += 1
        continue
    result = receiver.decode(receiver.read_audio(path), model, min_margin=0)
    ranks = []
    for expected, candidates in zip(row['concepts'], result.word_candidates):
        names = [name for name, _ in candidates]
        ranks.append(names.index(expected)+1 if expected in names else None)
    summary['top1_message'] += int(all(rank == 1 for rank in ranks))
    for k in (2, 3, 5):
        summary[f'top{k}_message_oracle'] += int(all(rank is not None and rank <= k for rank in ranks))
    details.append(dict(file=row['raw_file'], expected=row['concepts'],
                        hypothesis=result.hypothesis, ranks=ranks,
                        candidates=result.word_candidates))
out = ROOT/'experiments/real-clocked-20260917/periodicity05'
(out/'candidate-audit.json').write_text(json.dumps(dict(summary=dict(summary)), indent=2))
(out/'candidates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in details))
print(json.dumps(dict(summary), indent=2))
