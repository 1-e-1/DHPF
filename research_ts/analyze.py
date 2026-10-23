"""Rebuild manuscript evidence from immutable run artifacts (no training)."""
import csv, hashlib, itertools, json, math
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits
from benchmark.metrics import distribution, sequence_metrics, sequence_stats
from benchmark.run import write_json

ROOT=Path(__file__).resolve().parents[1]; OLD=ROOT/'results/checklist_v1'; OUT=ROOT/'results/timeseries_v2'
KEYS=['sw2_squared','mmd2_unbiased','power_rmse','autocorrelation_rmse','cross_correlation_rmse']
MAIN=['gan','ddpm_v','icfm','otcfm','dhpf','drifting']

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def dump_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def extra_stats(x,y):
    x,y=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
    def one(z):
        flat=z.reshape(-1,z.shape[-1]);mean=flat.mean(0);sd=flat.std(0).clip(1e-12)
        zz=(z-mean)/sd
        cross=np.stack([np.einsum('ntc,ntd->cd',zz[:,:128-l],zz[:,l:],optimize=True)/(len(z)*(128-l)) for l in [1,2,4,8,16,32]])
        return dict(variance=float(flat.var(0).mean()),difference_energy=float(np.diff(z,axis=1).square().mean()) if hasattr(z,'square') else float((np.diff(z,axis=1)**2).mean()),lagcross=cross)
    a,b=one(x),one(y)
    return dict(variance_ratio=a['variance']/max(b['variance'],1e-12),difference_energy_ratio=a['difference_energy']/max(b['difference_energy'],1e-12),lagged_cross_rmse=float(np.sqrt(np.mean((a['lagcross']-b['lagcross'])**2))))

def collect():
    rows=[];perclass=[];sources=[]
    for dataset in ['sine','lorenz','har']:
        folders=list((OLD/'runs').glob(dataset+'__*'))+list((OUT/'runs').glob(dataset+'__*'))
        for folder in sorted(folders):
            if not (folder/'metrics.json').exists() or not (folder/'downstream.json').exists():continue
            cfg=read(folder/'config.json'); met=read(folder/'metrics.json'); downstream=read(folder/'downstream.json'); meta=read(folder/'metadata.json')
            model='ddpm_v' if cfg.get('baseline_variant')=='cosine_v' else cfg['model']
            variant=folder.name.split('__')[2];seed=cfg['seed']
            row=dict(dataset=dataset,model=model,variant=variant,seed=seed,folder=str(folder.relative_to(ROOT)),parameters=meta['parameters'],discriminator_parameters=meta['discriminator_parameters'],lr=cfg['lr'],updates=cfg['updates'],checkpoint_step=met['checkpoint_step'])
            if cfg['classes']:
                for c in range(cfg['classes']):
                    perclass.append(dict(dataset=dataset,model=model,variant=variant,seed=seed,condition=c,**{k:met[str(c)][k] for k in KEYS}))
                row.update({k:float(np.mean([met[str(c)][k] for c in range(cfg['classes'])])) for k in KEYS})
            else:row.update({k:met[k] for k in KEYS})
            for kind in ['synthetic','real_matched']:
                row.update({kind+'_'+k:v for k,v in downstream[kind].items() if k.startswith('test_')})
            train=read(folder/'training_complete.json');row['training_wall_seconds']=train['wall_seconds'];row['validation_seconds']=train['validation_seconds']
            logs=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
            row['update_seconds']=float(np.mean([r['update_seconds'] for r in logs if r.get('profiled')]))
            if variant=='base' and model in MAIN:
                extra_path=OUT/'diagnostics'/(folder.name+('__v' if model=='ddpm_v' else '')+'.json')
                if not extra_path.exists():
                    with np.load(folder/'samples.npz') as z:x=z['generated'];labels=z['labels']
                    with np.load(ROOT/'data/checklist_v1'/f'{dataset}.npz') as z: y=z['test'];yl=z['test_labels'] if cfg['classes'] else None
                    if cfg['classes']:
                        class_extra=[extra_stats(x[labels==c],y[yl==c]) for c in range(cfg['classes'])]
                        extra={k:float(np.mean([e[k] for e in class_extra])) for k in class_extra[0]}
                    else:extra=extra_stats(x,y)
                    write_json(extra_path,extra)
                row.update(read(extra_path))
            rows.append(row)
            sources.append(dict(path=str(folder.relative_to(ROOT)),metrics_sha256=hashlib.sha256((folder/'metrics.json').read_bytes()).hexdigest(),config_sha256=hashlib.sha256((folder/'config.json').read_bytes()).hexdigest()))
    dump_csv(OUT/'all_runs.csv',rows);dump_csv(OUT/'per_class.csv',perclass);write_json(OUT/'evidence_sources.json',sources)
    return rows

def summarize(rows):
    groups={}
    for row in rows:groups.setdefault((row['dataset'],row['model'],row['variant']),[]).append(row)
    summary=[]
    excluded={'dataset','model','variant','seed','folder'}
    for (d,m,v),records in groups.items():
        out=dict(dataset=d,model=m,variant=v,n=len(records),seeds=[r['seed'] for r in records])
        for k in records[0]:
            if k in excluded:continue
            values=[r[k] for r in records if k in r and isinstance(r[k],(int,float))]
            if values:out[k+'_mean']=float(np.mean(values));out[k+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summary.append(out)
    dump_csv(OUT/'summary.csv',summary);write_json(OUT/'summary.json',summary)
    comparisons=[]
    for d in ['sine','lorenz','har']:
        a={r['seed']:r for r in groups.get((d,'dhpf','base'),[])}
        for m in MAIN:
            if m=='dhpf':continue
            b={r['seed']:r for r in groups.get((d,m,'base'),[])}
            if set(a)!=set(range(5)) or set(b)!=set(a):continue
            for k in KEYS+(['synthetic_test_macro_f1'] if d=='har' else ['synthetic_test_forecast_mse']):
                diff=np.array([a[s][k]-b[s][k] for s in range(5)]);wins=int((diff>0).sum()) if k.endswith('f1') else int((diff<0).sum())
                n=int((diff!=0).sum());p=float(binomtest(wins,n,.5).pvalue) if n else 1.
                comparisons.append(dict(dataset=d,baseline=m,metric=k,dhpf_minus_baseline=diff.tolist(),mean_difference=float(diff.mean()),paired_std=float(diff.std(ddof=1)),dhpf_wins=wins,n=5,sign_test_two_sided_p=p,exploratory=True))
    # Holm adjustment over ALL reported main-table comparisons, not selected winners.
    order=np.argsort([r['sign_test_two_sided_p'] for r in comparisons]); running=0.
    for i,idx in enumerate(order):
        running=max(running,min(1.,(len(order)-i)*comparisons[idx]['sign_test_two_sided_p']));comparisons[idx]['holm_p']=running
    write_json(OUT/'paired_comparisons.json',comparisons)
    return summary

def real_reference():
    path=OUT/'real_reference.json'
    if path.exists():return
    result={}
    for dataset in ['sine','lorenz','har']:
        with np.load(ROOT/'data/checklist_v1'/f'{dataset}.npz') as z: a={k:z[k] for k in z.files}
        bw=float(np.sqrt(np.median(pdist(a['train'][:512].reshape(512,-1),'sqeuclidean'))))
        records=[]
        for seed in range(5):
            rng=np.random.default_rng(96000+seed);per=[]
            for c in range(6) if dataset=='har' else [None]:
                x=a['train'] if c is None else a['train'][a['train_labels']==c]
                y=a['test'] if c is None else a['test'][a['test_labels']==c]
                x=x[rng.choice(len(x),600 if c is not None else 2000,replace=False)]
                metrics=distribution(x,y,{'dataset':dataset},False);metrics.update(sequence_metrics(x,y,bw)[0]);per.append({k:metrics[k] for k in KEYS})
            records.append({k:float(np.mean([r[k] for r in per])) for k in KEYS})
        result[dataset]=dict(note='Train subsample vs fixed held-out test: descriptive sampling/domain-shift reference, not a mathematical lower bound; variation is subsampling, not training seeds.',records=records,**{k:dict(mean=float(np.mean([r[k] for r in records])),std=float(np.std([r[k] for r in records],ddof=1))) for k in KEYS})
    write_json(path,result)

def main():
    threadpool_limits(2); rows=collect();summary=summarize(rows);real_reference()
    main=[r for r in summary if r['variant']=='base' and r['model'] in MAIN]
    write_json(OUT/'analysis_status.json',dict(main_cells=len(main),expected_cells=18,complete=len(main)==18 and all(r['n']==5 for r in main),original_hashes_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==s for p,s in read(OUT/'frozen_original.json')['sha256'].items())))
    for r in main:print(r['dataset'],r['model'],r['n'],*[round(r[k+'_mean'],5) for k in KEYS])

if __name__=='__main__':main()
