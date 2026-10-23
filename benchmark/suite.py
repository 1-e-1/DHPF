"""Queue for all checklist training jobs; no completed status from launch alone."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from .protocol import RESULTS, MODELS, config, write_protocol
from .run import train,evaluate,write_json


def selected(dataset,model):
    path=RESULTS/'selection'/f'{dataset}__{model}.json'
    if path.exists(): return json.loads(path.read_text())['lr']
    trials=[]
    for i,lr in enumerate([.0002,.001]):
        cfg=config(dataset,model,100,lr=lr,updates=1000,val_every=500,experiments=['HPO'])
        folder=RESULTS/'hpo'/f'{dataset}__{model}__candidate{i}'
        train(cfg,folder)
        result=json.loads((folder/'training_complete.json').read_text())
        trials.append(dict(lr=lr,validation=result['best_validation'],folder=str(folder)))
    best=min(trials,key=lambda t:t['validation'])
    write_json(path,dict(dataset=dataset,model=model,lr=best['lr'],trials=trials,selection='validation only; equal 2 x 1000 update budget'))
    return best['lr']


def jobs(group):
    if group=='toy':
        for dataset in ['gmm8','moons','scurve','checkerboard']:
            for model in MODELS+(['direct','augmented'] if dataset=='gmm8' else []):
                yield dataset,model,'base',{}
    if group=='sequence':
        for dataset in ['sine','lorenz']:
            for model in MODELS+(['direct','augmented'] if dataset=='sine' else []):
                yield dataset,model,'base',{}
    if group=='image':
        for dataset in ['mnist','cifar10']:
            for model in MODELS: yield dataset,model,'base',{}
    if group=='har': yield 'har','dhpf','base',{}
    if group=='ablation':
        for dataset in ['gmm8','sine']:
            for variant in ['radial','tangential','fixed_full','fixed_radial','fixed_tangential']:
                yield dataset,'dhpf',variant,dict(variant=variant,experiments=['E06','E08'])
    if group=='sensitivity':
        settings=dict(n=[8,16,64],fixed_width=[True],sigma_min=[.05,.4],batch=[32,64,256],velocity=['zero','gaussian'],lambda_sep=[.001,.01])
        for key,values in settings.items():
            for value in values:
                yield 'gmm8','dhpf',f'{key}_{value}',{key:value,'experiments':['E07','E08']}
        for value in [1,4]: yield 'sine','dhpf',f'rank_{value}',dict(rank=value,experiments=['E07','E08'])


def run(group,seeds,only=None,train_only=False):
    write_protocol()
    for dataset,model,variant,overrides in jobs(group):
        if only and model not in only: continue
        try: lr=selected(dataset,model)
        except Exception:
            write_json(RESULTS/'errors'/f'{dataset}__{model}__hpo.json',dict(traceback=traceback.format_exc()))
            continue
        for seed in seeds:
            cfg=config(dataset,model,seed,lr=lr,**overrides)
            if model in ['direct','augmented']: cfg['experiments']=['E05','E08','E09']
            if model=='dhpf' and dataset in ['gmm8','sine'] and variant=='base': cfg['experiments']+=['E05','E06','E07']
            folder=RESULTS/'runs'/f'{dataset}__{model}__{variant}__seed{seed}'
            try:
                train(cfg,folder)
                if not train_only: evaluate(folder)
            except Exception:
                write_json(RESULTS/'errors'/f'{folder.name}.json',dict(traceback=traceback.format_exc()))
                print('FAILED',folder.name,traceback.format_exc(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['toy','sequence','image','har','ablation','sensitivity','all'],default='toy');p.add_argument('--seeds',nargs='+',type=int,default=list(range(5)));p.add_argument('--models',nargs='+');p.add_argument('--train-only',action='store_true');a=p.parse_args()
    for g in ['toy','sequence','image','ablation','sensitivity','har'] if a.group=='all' else [a.group]: run(g,a.seeds,a.models,a.train_only)
