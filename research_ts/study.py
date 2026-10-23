"""Serial local-GPU continuation with validation-only hyperparameter selection."""
import argparse, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from benchmark.protocol import config
from benchmark.run import write_json, load_data
from benchmark.postprocess import downstream
from research_ts import runner
from research_ts.diffusion import audit, generate

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/timeseries_v2'
PAIRS=[('sine','ddpm'),('lorenz','ddpm')]+[('har',m) for m in ['gan','ddpm','icfm','otcfm','drifting']]

def frozen():
    hashes=json.loads((OUT/'frozen_original.json').read_text())['sha256']
    for name,sha in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha, name+' changed'

def cfg_for(dataset,model,seed,**kw):
    cfg=config(dataset,model,seed,**kw)
    cfg.update(study='timeseries_v2',experiments=['TS01'],baseline_variant='cosine_v' if model=='ddpm' else 'original')
    if model=='ddpm':cfg.update(ddpm_beta='cosine s=0.008, beta_max=0.999',ddpm_target='v',ddpm_clipping=False)
    return cfg

def sanity():
    frozen(); write_json(OUT/'diffusion_audit.json',audit())
    # Every newly trained objective is exercised, including conditional labels.
    for model in ['ddpm','gan','icfm','otcfm','drifting']:
        cfg=cfg_for('har',model,991,updates=2,val_every=2,batch=16,eval_batch=512)
        folder=OUT/'sanity'/model
        runner.train(cfg,folder)
        net,_,_=runner.load_model(folder)
        sample,_=generate(net,cfg,8,80000,'cuda',condition=2)
        assert tuple(sample.shape)==(8,1152) and torch.isfinite(sample).all()
    write_json(OUT/'sanity_complete.json',dict(passed=True,objectives=5,original_sha256_unchanged=True))

def run():
    frozen(); assert (OUT/'sanity_complete.json').exists()
    selected={}
    for dataset,model in PAIRS:
        trials=[]
        for lr in [.0002,.001]:
            cfg=cfg_for(dataset,model,100,lr=lr,updates=1000,val_every=1000)
            folder=OUT/'hpo'/f'{dataset}__{model}__lr{lr}'
            runner.train(cfg,folder)
            m=json.loads((folder/'training_complete.json').read_text())
            trials.append(dict(lr=lr,validation=m['best_validation'],folder=str(folder.relative_to(ROOT))))
        selected[dataset+'__'+model]=dict(lr=min(trials,key=lambda x:x['validation'])['lr'],trials=trials)
        write_json(OUT/'hpo_selection.json',selected)
        # Finish a dataset/model block before progressing; resumable per-run files.
        for seed in range(5):
            frozen()
            cfg=cfg_for(dataset,model,seed,lr=selected[dataset+'__'+model]['lr'])
            folder=OUT/'runs'/f'{dataset}__{model}__base__seed{seed}'
            runner.train(cfg,folder); runner.evaluate(folder)
            arrays=load_data(cfg)
            with np.load(folder/'samples.npz') as z:
                downstream(folder,cfg,arrays,z['generated'],z['labels'])
            write_json(folder/'complete.json',dict(training=True,evaluation=True,downstream=True))
            completed=list((OUT/'runs').glob('*/complete.json'))
            write_json(OUT/'progress.json',dict(completed=len(completed),total=35,last=folder.name,updated=time.strftime('%Y-%m-%d %H:%M:%S')))
            print('COMPLETE',len(completed),'/35',folder.name,flush=True)
    frozen(); write_json(OUT/'study_complete.json',dict(formal_runs=35,hpo_runs=14,original_sha256_unchanged=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sanity',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);threadpool_limits(2)
    try: sanity() if a.sanity else run()
    except Exception:
        write_json(OUT/'failure.json',dict(traceback=traceback.format_exc(),updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        raise
