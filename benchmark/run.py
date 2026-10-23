"""Resumable training and held-out evaluation with explicit run states."""
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from .data import DATA, ROOT
from .models import build, count, VectorNet, prior, integrate, generate, match, diffusion_schedule, drifting_loss, sync
from .metrics import distribution, sliced, sequence_metrics
from .protocol import RESULTS, config, write_protocol


def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8'); tmp.replace(path)


def version():
    files=sorted((ROOT/'benchmark').glob('*.py'))
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    path=RESULTS/'source'/digest
    if not path.exists():
        path.mkdir(parents=True)
        for f in files: shutil.copy2(f,path/f.name)
        write_json(path/'hashes.json',hashes)
    return digest


def load_data(cfg):
    with np.load(DATA/(cfg['dataset']+'.npz')) as z:
        arrays={k:z[k] for k in z.files}
    return arrays


def state(folder,status,**kw):
    write_json(folder/'status.json',dict(status=status,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))


def validation(model,cfg,arrays):
    model.eval(); scores=[]
    with torch.no_grad():
        for c in range(cfg['classes']) if cfg['classes'] else [None]:
            real=arrays['val'] if c is None else arrays['val'][arrays['val_labels']==c]
            real=real[:512].reshape(min(512,len(real)),-1)
            gen,_=generate(model,cfg,len(real),70000,cfg['device'],condition=c)
            if not torch.isfinite(gen).all(): return float('inf')
            scores.append(sliced(gen.numpy(),real,64,71000))
    model.train(); return float(np.mean(scores))


def train(cfg,folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training_complete.json').exists(): return
    torch.set_num_threads(2); threadpool_limits(2)
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark=False
    cfgpath=folder/'config.json'
    if cfgpath.exists() and json.loads(cfgpath.read_text())!=cfg: raise RuntimeError('Existing run config differs')
    write_json(cfgpath,cfg)
    source_version=version()
    state(folder,'training',source_version=source_version)
    device=cfg['device']; model=build(cfg).to(device); arrays=load_data(cfg)
    trainx=torch.as_tensor(arrays['train'].reshape(len(arrays['train']),-1),device=device)
    trainlabels=torch.as_tensor(arrays['train_labels'],device=device) if cfg['classes'] else None
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['adam_betas']))
    discriminator=VectorNet(cfg['d'],1,cfg['width'],cfg['classes'],spectral=True).to(device) if cfg['model']=='gan' else None
    dop=torch.optim.Adam(discriminator.parameters(),lr=cfg['lr'],betas=(.5,.9)) if discriminator is not None else None
    gen=torch.Generator(device=device).manual_seed(10000+cfg['seed'])
    # Dedicated streams preserve the same data/prior draws across endpoint-OT controls.
    datagen=torch.Generator(device=device).manual_seed(20000+cfg['seed'])
    auxgen=torch.Generator(device=device).manual_seed(30000+cfg['seed'])
    startstep=0; best=float('inf'); beststep=0; elapsed=0.; valtime=0.
    resume=folder/'latest.pt'
    if resume.exists():
        ck=torch.load(resume,map_location=device,weights_only=False)
        if ck['source_version']!=source_version: raise RuntimeError('Code changed during a resumable run; restart as new run/config version')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        if dop is not None: discriminator.load_state_dict(ck['discriminator']); dop.load_state_dict(ck['doptimizer'])
        gen.set_state(ck['gen'].cpu()); datagen.set_state(ck['datagen'].cpu()); auxgen.set_state(ck['auxgen'].cpu())
        startstep=ck['step']; best=ck['best']; beststep=ck['beststep']; elapsed=ck['elapsed']; valtime=ck['valtime']
    metadata=dict(parameters=count(model),discriminator_parameters=count(discriminator) if discriminator else 0,source_version=source_version,python=platform.python_version(),torch=torch.__version__,device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None,data_sha256=json.loads((DATA/(cfg['dataset']+'.json')).read_text())['sha256'])
    write_json(folder/'metadata.json',metadata)
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    beta,alpha,abar,var=diffusion_schedule(device)
    start=time.perf_counter(); lastsave=start
    logpath=folder/'training.jsonl'
    try:
        with logpath.open('a',encoding='utf-8',buffering=1) as log:
            for step in range(startstep+1,cfg['updates']+1):
                c=int(torch.randint(cfg['classes'],(),generator=datagen,device=device)) if cfg['classes'] else None
                pool=trainx if c is None else trainx[trainlabels==c]
                y=pool[torch.randint(len(pool),(cfg['batch'],),generator=datagen,device=device)]
                x,v=prior(cfg['batch'],cfg['d'],gen,device,cfg['velocity'])
                opt.zero_grad(set_to_none=True)
                profiled=(step==1 or step%200==0)
                if profiled: sync(device)
                tick=time.perf_counter(); ot_seconds=0.; source_seconds=0.; field_seconds=0.
                name=cfg['model']; sep=x.new_zeros(())
                if name in ['dhpf','direct','augmented']:
                    out,_,timings,_=integrate(model,name,x,v,cfg['steps'],c,profiled)
                    if profiled: sync(device)
                    tm=time.perf_counter(); target=match(out,y); ot_seconds=time.perf_counter()-tm
                    loss=(out-target).square().sum(-1).mean()
                    if cfg['lambda_sep']:
                        sep=model.separation(torch.rand(4,generator=auxgen,device=device),c)
                        loss=loss+cfg['lambda_sep']*sep
                    source_seconds=timings['source_seconds']; field_seconds=timings['field_seconds']
                elif name in ['icfm','otcfm']:
                    if name=='otcfm':
                        if profiled: sync(device)
                        tm=time.perf_counter(); y=match(x,y); ot_seconds=time.perf_counter()-tm
                    t=torch.rand(cfg['batch'],device=device,generator=auxgen)
                    xt=(1-t[:,None])*x+t[:,None]*y
                    loss=(model(xt,t,c)-(y-x)).square().mean()
                elif name=='ddpm':
                    idx=torch.randint(1000,(cfg['batch'],),device=device,generator=auxgen)
                    t=idx/999; xt=abar[idx,None].sqrt()*y+(1-abar[idx,None]).sqrt()*x
                    loss=(model(xt,t,c)-x).square().mean()
                elif name=='gan':
                    dop.zero_grad(set_to_none=True)
                    fake=model(x,condition=c)
                    dl=F.relu(1-discriminator(y,condition=c)).mean()+F.relu(1+discriminator(fake.detach(),condition=c)).mean()
                    dl.backward(); dop.step()
                    for p in discriminator.parameters(): p.requires_grad_(False)
                    loss=-discriminator(fake,condition=c).mean()
                elif name=='drifting':
                    loss=drifting_loss(model(x,condition=c),y)
                else: raise ValueError(name)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training objective')
                if profiled: sync(device)
                backward_start=time.perf_counter(); loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
                opt.step()
                if discriminator:
                    for p in discriminator.parameters(): p.requires_grad_(True)
                if profiled: sync(device)
                backward_seconds=time.perf_counter()-backward_start
                record=dict(step=step,loss=float(loss.detach()),grad_norm=float(norm),separation=float(sep.detach()),wall_seconds=elapsed+time.perf_counter()-start,profiled=profiled)
                if profiled:
                    record.update(source_seconds=source_seconds,field_seconds=field_seconds,ot_seconds=ot_seconds,backward_seconds=backward_seconds,update_seconds=time.perf_counter()-tick)
                    if name=='dhpf':
                        with torch.no_grad():
                            _,diag=model.field(x,model.prepare(x.new_tensor([.5]),c),diagnostic=True)
                        record.update(diag)
                log.write(json.dumps(record)+'\n')
                if step%cfg['val_every']==0 or step==cfg['updates']:
                    vs=time.perf_counter(); score=validation(model,cfg,arrays); valtime+=time.perf_counter()-vs
                    if not np.isfinite(score): raise FloatingPointError('Nonfinite validation generation')
                    with (folder/'validation.jsonl').open('a',encoding='utf-8') as vf:
                        vf.write(json.dumps(dict(step=step,sw2_squared=score,wall_seconds=elapsed+time.perf_counter()-start,generation_seed=70000))+'\n')
                    if score<best:
                        best,beststep=score,step
                        torch.save(dict(model=model.state_dict(),step=step,validation=score,config=cfg,source_version=source_version),folder/'best.pt')
                    print(f"{folder.name} {step}/{cfg['updates']} val_SW2={score:.6g}",flush=True)
                if step%cfg['val_every']==0 or step==cfg['updates']:
                    ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,best=best,beststep=beststep,elapsed=elapsed+time.perf_counter()-start,valtime=valtime,gen=gen.get_state(),datagen=datagen.get_state(),auxgen=auxgen.get_state(),source_version=source_version)
                    if dop: ck.update(discriminator=discriminator.state_dict(),doptimizer=dop.state_dict())
                    torch.save(ck,folder/'latest.tmp.pt'); (folder/'latest.tmp.pt').replace(resume)
                    state(folder,'training',step=step,budget=cfg['updates'],best_validation=best,source_version=source_version)
        write_json(folder/'training_complete.json',dict(status='budget_exhausted_convergence_unestablished',steps=cfg['updates'],best_step=beststep,best_validation=best,wall_seconds=elapsed+time.perf_counter()-start,validation_seconds=valtime,peak_train_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None))
        state(folder,'trained_pending_evaluation',step=cfg['updates'])
    except Exception as exc:
        state(folder,'failed',error=repr(exc),traceback=traceback.format_exc(),step=step if 'step' in locals() else None)
        raise


def load_model(folder,checkpoint='best.pt'):
    folder=Path(folder); cfg=json.loads((folder/'config.json').read_text())
    m=build(cfg).to(cfg['device']); ck=torch.load(folder/checkpoint,map_location=cfg['device'],weights_only=False); m.load_state_dict(ck['model']); m.eval()
    return m,cfg,ck


def evaluate(folder):
    folder=Path(folder)
    if (folder/'metrics.json').exists(): return
    model,cfg,ck=load_model(folder); arrays=load_data(cfg); device=cfg['device']
    state(folder,'evaluating',checkpoint_step=ck['step'])
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter(); xs=[]; labels=[]; allmetrics={}
    conditions=range(cfg['classes']) if cfg['classes'] else [None]
    for c in conditions:
        y=arrays['test'] if c is None else arrays['test'][arrays['test_labels']==c]
        n=cfg['eval_count'] if c is None else cfg['eval_per_condition']
        x,timings=generate(model,cfg,n,80000,device,condition=c,profile=True)
        if not torch.isfinite(x).all(): raise FloatingPointError('Nonfinite held-out generation')
        xx=x.numpy().reshape(n,*cfg['shape']); xs.append(xx)
        out=distribution(xx,y,cfg,exact=(cfg['d']==2))
        out.update(timings)
        if len(cfg['shape'])==2:
            tr=arrays['train'].reshape(len(arrays['train']),-1)
            from scipy.spatial.distance import pdist
            bandwidth=float(np.sqrt(np.median(pdist(tr[:512],'sqeuclidean'))))
            stats,sa,sb=sequence_metrics(xx,y,bandwidth); out.update(stats)
            np.savez_compressed(folder/f'sequence_stats_{c}.npz',**{f'generated_{k}':v for k,v in sa.items()},**{f'real_{k}':v for k,v in sb.items()})
        if c is None: allmetrics=out
        else: allmetrics[str(c)]=out; labels.extend([c]*n)
    generated=np.concatenate(xs)
    np.savez_compressed(folder/'samples.npz',generated=generated,labels=np.asarray(labels),generation_seed=80000)
    allmetrics.update(sample_seconds=time.perf_counter()-start,peak_inference_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,checkpoint_step=ck['step'],generation_seed=80000,checkpoint_selection='validation',evaluation_split='test',convergence='unestablished at fixed budget')
    # Budget-final checkpoint is evaluated on validation only here; final test metric is also saved separately.
    if ck['step']!=cfg['updates']:
        final,_,_=load_model(folder,'latest.pt')
        fy,_=generate(final,cfg, min(2000,cfg['eval_count']),80000,device,condition=0 if cfg['classes'] else None)
        real=arrays['test'] if not cfg['classes'] else arrays['test'][arrays['test_labels']==0]
        allmetrics['budget_final_sw2_squared_diagnostic']=sliced(fy.numpy(),real.reshape(len(real),-1))
        allmetrics['budget_final_diagnostic_count']=len(fy)
    write_json(folder/'metrics.json',allmetrics)
    state(folder,'evaluated_pending_figures_and_specialized_metrics',checkpoint_step=ck['step'])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--folder',required=True); p.add_argument('--train-only',action='store_true'); a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text()); train(cfg,Path(a.folder))
    if not a.train_only: evaluate(Path(a.folder))


if __name__=='__main__': main()
