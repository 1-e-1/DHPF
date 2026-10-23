"""Check completeness, finite values and paired evaluation inputs of stored runs."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path('results/main')
expected={(d,m,s) for d in ['gmm8','rings','spirals'] for m in ['full','radial','tangential','mlp'] for s in [0,1,2]}
found=set()
for p in root.glob('*/config.json'):
    cfg=json.loads(p.read_text())
    found.add((cfg['dataset'],cfg['model'],cfg['seed']))
    for name in ['metrics.json','training.csv','checkpoint.pt','samples.npz','samples.png']:
        assert (p.parent/name).is_file(), (p.parent,name)
    assert cfg['source_sha256']==hashlib.sha256((root/'source_snapshot.py').read_bytes()).hexdigest()
    logs=list(csv.DictReader((p.parent/'training.csv').open()))
    assert len(logs)==cfg['train_steps']
    assert int(logs[-1]['step'])==cfg['train_steps']
    assert all(np.isfinite(float(row[k])) for row in logs for k in ['loss','gradient_norm_before_clip'])
    metrics=json.loads((p.parent/'metrics.json').read_text())
    assert metrics['finite_outputs']
    for v in metrics.values():
        if isinstance(v,(float,int)):
            assert np.isfinite(v)
    samples=np.load(p.parent/'samples.npz')
    reference=np.load(root/f"{cfg['dataset']}__full__seed{cfg['seed']}"/'samples.npz')
    for key in ['target','independent_target','initial_x','initial_v']:
        assert np.array_equal(samples[key],reference[key]), (p.parent,key)
assert found==expected,(expected-found,found-expected)
result={'passed':True,'completed_main_runs':len(found),'checks':['all expected datasets/models/seeds','all artifacts present',
    'source snapshot hash','training log length','finite losses and gradients','finite output metrics',
    'identical independent evaluation targets and initial states across compared models']}
(root/'integrity_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
