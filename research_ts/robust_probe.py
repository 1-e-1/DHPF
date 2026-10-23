"""Closed-form HAR TSTR sensitivity check for iterative logistic convergence."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score,accuracy_score
from threadpoolctl import threadpool_limits
from benchmark.run import write_json

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/timeseries_v2'

def fit_probe(x,y,a):
    xx=np.asarray(x.reshape(len(x),-1),dtype=np.float64)
    va=np.asarray(a['val'].reshape(len(a['val']),-1),dtype=np.float64)
    te=np.asarray(a['test'].reshape(len(a['test']),-1),dtype=np.float64)
    best=None;trials=[]
    for alpha in [.01,1.,100.]:
        model=RidgeClassifier(alpha=alpha,solver='cholesky')
        model.fit(xx,y);score=f1_score(a['val_labels'],model.predict(va),average='macro')
        trials.append(dict(alpha=alpha,validation_macro_f1=float(score)))
        if best is None or score>best[0]:best=(score,alpha,model)
    pred=best[2].predict(te)
    return dict(test_macro_f1=float(f1_score(a['test_labels'],pred,average='macro')),test_accuracy=float(accuracy_score(a['test_labels'],pred)),selected_alpha=best[1],validation_trials=trials,training_count=len(x),solver='float64 Cholesky ridge, one-vs-rest classification')

def main():
    threadpool_limits(2)
    with np.load(ROOT/'data/checklist_v1/har.npz') as z:a={k:z[k] for k in z.files}
    rng=np.random.default_rng(90000);ids=np.concatenate([rng.choice(np.flatnonzero(a['train_labels']==c),600,replace=False) for c in range(6)])
    dest=OUT/'ridge_probe';dest.mkdir(exist_ok=True)
    if not (dest/'real_matched.json').exists():write_json(dest/'real_matched.json',fit_probe(a['train'][ids],a['train_labels'][ids],a))
    folders=list((ROOT/'results/checklist_v1/runs').glob('har__dhpf__base__*'))+list((OUT/'runs').glob('har__*'))
    for folder in sorted(folders):
        path=dest/(folder.name+'.json')
        if path.exists() or not (folder/'complete.json').exists() and 'timeseries_v2' in str(folder):continue
        with np.load(folder/'samples.npz') as z:x=z['generated'];y=z['labels'].astype(int)
        out=fit_probe(x,y,a);out['source']=str(folder.relative_to(ROOT));write_json(path,out)
        print(folder.name,out['test_macro_f1'],flush=True)

if __name__=='__main__':main()
