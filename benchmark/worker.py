"""Persistent execution worker. CPU and GPU queues remain separate."""
import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path
import psutil
import torch
from threadpoolctl import threadpool_limits
from .data import DATA,images
from .protocol import RESULTS,config
from .suite import jobs,selected
from .run import train,evaluate,write_json
from .postprocess import process
from .features import evaluate_images
from .report import report


def prepare_images():
    if (DATA/'cifar10.npz').exists():return
    verified=DATA/'downloads'/'cifar_verified.tar.gz'
    if not verified.exists():raise FileNotFoundError('CIFAR-10 verified archive is not yet ready')
    with verified.open('rb') as f:checksum=hashlib.file_digest(f,'md5').hexdigest()
    if checksum!='c58f30108f718f92721af3b95e74349a':raise ValueError('CIFAR archive MD5 mismatch')
    shutil.copy2(verified,DATA/'downloads'/'cifar-10-python.tar.gz')
    images()


def one(cfg,folder):
    try:
        train(cfg,folder);evaluate(folder)
        if cfg.get('image_shape'):evaluate_images(folder)
        process(folder,full=cfg['seed']==0 and folder.name.split('__')[2]=='base')
        write_json(folder/'status.json',dict(status='completed_fixed_budget',convergence='not established',training_seed=cfg['seed']))
    except Exception:
        write_json(RESULTS/'errors'/f'{folder.name}__worker.json',dict(traceback=traceback.format_exc()))
        previous=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        if previous.get('status')!='failed':write_json(folder/'status.json',dict(status='postprocess_failed' if (folder/'training_complete.json').exists() else 'failed',error=traceback.format_exc()))
        print('ERROR',folder.name,traceback.format_exc(),flush=True)


def main(device,waitpid):
    torch.set_num_threads(2);threadpool_limits(2)
    statuspath=RESULTS/f'worker_{device}.json'
    write_json(statuspath,dict(pid=os.getpid(),device=device,status='waiting_for_initial_training_queue',waiting_pid=waitpid))
    if waitpid:
        try:
            proc=psutil.Process(waitpid)
            if 'benchmark.suite' in proc.cmdline():proc.wait()
        except psutil.NoSuchProcess:pass
    write_json(statuspath,dict(pid=os.getpid(),device=device,status='running'))
    # Finish complete initial runs first, including specialized metrics and figures.
    for folder in sorted((RESULTS/'runs').glob('*')):
        if not (folder/'config.json').exists():continue
        cfg=json.loads((folder/'config.json').read_text())
        if cfg['device']==device:one(cfg,folder);report()
    groups=['toy','ablation','sensitivity'] if device=='cpu' else ['sequence','image','ablation','sensitivity','har']
    for group in groups:
        for dataset,model,variant,overrides in jobs(group):
            cfg=config(dataset,model)
            if cfg['device']!=device:continue
            if dataset=='cifar10':
                try:prepare_images()
                except Exception:
                    write_json(RESULTS/'errors'/'cifar_preparation.json',dict(traceback=traceback.format_exc()));continue
            try:lr=selected(dataset,model)
            except Exception:
                write_json(RESULTS/'errors'/f'{dataset}__{model}__hpo_worker.json',dict(traceback=traceback.format_exc()));continue
            for seed in range(5):
                cfg=config(dataset,model,seed,lr=lr,**overrides)
                if model in ['direct','augmented']:cfg['experiments']=['E05','E08','E09']
                if model=='dhpf' and dataset in ['gmm8','sine'] and variant=='base':cfg['experiments']+=['E05','E06','E07']
                folder=RESULTS/'runs'/f'{dataset}__{model}__{variant}__seed{seed}'
                if (folder/'status.json').exists() and json.loads((folder/'status.json').read_text())['status']=='completed_fixed_budget':continue
                write_json(statuspath,dict(pid=os.getpid(),device=device,status='running',current=folder.name))
                one(cfg,folder);report()
    write_json(statuspath,dict(pid=os.getpid(),device=device,status='queue_exhausted_check_report_for_missing_or_failed_runs'))
    report()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--wait-pid',type=int);a=p.parse_args();main(a.device,a.wait_pid)
