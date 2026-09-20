import json,pathlib,numpy as np,joblib,collections,time
from scipy.io import wavfile
from b4bl import clocked
p=pathlib.Path('recordings'); session='2fcbcca1-254e-46a5-bb73-24865608f562'
rows=[json.loads(x) for x in (p/'manifest.jsonl').read_text().splitlines() if session in x]
m=joblib.load('experiments/clocked-balanced-final-20260916/clocked_classifier.joblib');results=[]
for i,r in enumerate(rows):
 sr,a=wavfile.read(p/r['raw_file']); a=a.astype(float)/32768; d=clocked.decode(a,m); gt=r['concepts']
 sr,b=wavfile.read(p/r['file']); b=b.astype(float)/32768
 entry=dict(file=r['raw_file'],expected=gt,prosody=r['prosody'],hypothesis=d.hypothesis,accepted=d.accepted,reason=d.reason,markers=d.marker_count,expected_markers=len(gt)+1,trimmed_markers=len(clocked.marker_positions(b)),clock_scale=d.clock_scale,clock_residual_ms=d.clock_residual_samples/sr*1000,peak=float(abs(a).max()),clipped_samples=int(np.sum(abs(a)>=.999)),samples=len(a)); results.append(entry)
 if (i+1)%100==0:print('evaluated',i+1,flush=True)
summ={}
for group,rs in [('all',results),('single',[r for r in results if len(r['expected'])==1]),('multi',[r for r in results if len(r['expected'])>1])]+[(k,[r for r in results if r['prosody']==k]) for k in ['neutral','calm','urgent','uncertain']]:
 summ[group]=dict(n=len(rs),exact=sum(r['accepted'] and r['hypothesis']==r['expected'] for r in rs),best_exact=sum(r['hypothesis']==r['expected'] for r in rs),accepted_wrong=sum(r['accepted'] and r['hypothesis']!=r['expected'] for r in rs),rejected=sum(not r['accepted'] for r in rs),markers_ok=sum(r['markers']==r['expected_markers'] for r in rs),trimmed_markers_ok=sum(r['trimmed_markers']==r['expected_markers'] for r in rs),clipped_files=sum(r['clipped_samples']>0 for r in rs))
summ['reasons']=dict(collections.Counter(r['reason'] for r in results if r['reason']))
out=pathlib.Path('experiments/clocked-batch-20260916/raw-evaluation.json');out.write_text(json.dumps(dict(model='experiments/clocked-balanced-final-20260916/clocked_classifier.joblib',session=session,summary=summ,results=results),indent=2));print(json.dumps(summ,indent=2),flush=True)
