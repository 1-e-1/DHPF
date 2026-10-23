"""Integrity checks and synchronized inference timing after the queue ends."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from benchmark.run import write_json
from benchmark.models import sync
from research_ts import runner
from research_ts.diffusion import generate
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/timeseries_v2'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    assert (OUT/'study_complete.json').exists();torch.set_num_threads(2);threadpool_limits(2)
    frozen=read(OUT/'frozen_original.json')['sha256']
    unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in frozen.items());assert unchanged
    checks=[]
    for folder in sorted((OUT/'runs').iterdir()):
        cfg=read(folder/'config.json');train=read(folder/'training_complete.json');metrics=read(folder/'metrics.json')
        assert (folder/'complete.json').exists() and train['steps']==5000
        vals=[json.loads(s) for s in (folder/'validation.jsonl').read_text().splitlines()]
        chosen=min(vals,key=lambda r:r['sw2_squared']);assert chosen['step']==metrics['checkpoint_step']==train['best_step']
        with np.load(folder/'samples.npz') as z:
            assert np.isfinite(z['generated']).all() and len(z['generated'])==cfg['eval_count']
            if cfg['classes']:assert all((z['labels']==c).sum()==600 for c in range(6))
        checks.append(dict(run=folder.name,updates=5000,chosen_step=chosen['step'],validation_selection_verified=True,sample_count=cfg['eval_count'],finite=True))
    assert len(checks)==35
    timing_path=OUT/'inference_timing.json';timing=[]
    if not timing_path.exists():
        for d in ['sine','lorenz','har']:
            for m in ['gan','ddpm','icfm','otcfm','dhpf','drifting']:
                root=OUT if m=='ddpm' or d=='har' and m!='dhpf' else ROOT/'results/checklist_v1'
                folder=root/'runs'/f'{d}__{m}__base__seed0';model,cfg,_=runner.load_model(folder);c=0 if cfg['classes'] else None
                generate(model,cfg,8,81000,'cuda',condition=c)
                times=[]
                for _ in range(3):
                    sync('cuda');start=time.perf_counter();x,_=generate(model,cfg,512,81000,'cuda',condition=c);sync('cuda');times.append(time.perf_counter()-start)
                timing.append(dict(dataset=d,model='ddpm_v' if m=='ddpm' else m,batch=512,seed=0,condition=c,seconds=times,median_seconds=float(np.median(times)),note='End-to-end generation including prior, batching, source precompute and CPU output transfer; three technical repeats, one trained checkpoint.'))
                del model;torch.cuda.empty_cache()
        write_json(timing_path,timing)
    warn=(OUT/'worker.stderr.log').read_text(encoding='utf-8')
    audit=dict(integrity_status='warn' if 'ConvergenceWarning' in warn else 'pass',generator_runs_passed=35,hpo_runs=len(list((OUT/'hpo').glob('*/training_complete.json'))),original_source_unchanged=unchanged,runs=checks,limitations=['Deterministic artifact/code checks are not an independent replication.','Logistic probe has iteration-limit warnings; all methods receive the separately reported closed-form ridge robustness probe.','Evidence uses one fixed data split and five training seeds.'],resolved_sanity_issue='Initial diffusion numerical-equivalence audit exposed float32 cancellation; posterior coefficients computed in float64. The retained failure.json predates passing sanity and all formal runs.')
    write_json(OUT/'EXPERIMENT_AUDIT.json',audit)
    write_json(OUT/'failure_resolution.json',dict(resolved=True,scope='Initial sanity numerical tolerance check only; all formal runs completed',evidence=['diffusion_audit.json','sanity_complete.json','study_complete.json','EXPERIMENT_AUDIT.json']))
    print('Integrity checks:',len(checks),'formal runs passed; original hashes unchanged')

if __name__=='__main__':main()
