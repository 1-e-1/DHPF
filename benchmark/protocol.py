"""Predeclared fixed budgets. No test-driven configuration changes."""
import json
import math
from pathlib import Path
from .data import ROOT, DATA
from .models import Sources, VectorNet, count

RESULTS=ROOT/'results'/'checklist_v1'
BASE=dict(n=32,rank=2,source_width=64,width=128,conv_width=32,variant='full',sigma_min=.15,fixed_width=False,velocity='halfnormal',lambda_sep=0.,steps=32,batch=128,lr=.001,classes=0,eval_batch=256,seed=0,adam_betas=[.9,.999],grad_clip=10.,checkpoint_rule='minimum fixed validation SW2 squared; no test access',solver='explicit midpoint',ddpm_steps=1000,ddpm_beta=[.0001,.02],drifting_feature='raw coordinates; no encoder or memory bank')
MODELS=['gan','ddpm','icfm','otcfm','dhpf','drifting']
IDS=dict(gan='M0',ddpm='M1',icfm='M2',otcfm='M3',direct='M4',augmented='M5',dhpf='M6',drifting='M7-raw')


def config(dataset,model,seed=0,**overrides):
    cfg={**BASE,'dataset':dataset,'model':model,'model_id':IDS[model],'seed':seed}
    if dataset in ['gmm8','moons','scurve','checkerboard']:
        cfg.update(d=2,updates=5000,val_every=1000,device='cpu',eval_count=10000,experiments=['E02','E08','E09','E10'],shape=[2])
    elif dataset in ['sine','lorenz','har']:
        c=9 if dataset=='har' else 3
        cfg.update(d=128*c,shape=[128,c],updates=5000,val_every=1000,device='cuda',eval_count=2000,classes=6 if dataset=='har' else 0,experiments=['E12' if dataset=='har' else 'E04','E08','E09','E10'])
        if dataset=='har':cfg.update(eval_count=3600,eval_per_condition=600)
    else:
        shape=[1,28,28] if dataset=='mnist' else [3,32,32]
        cfg.update(d=math.prod(shape),shape=shape,image_shape=shape,updates=10000,val_every=2000,device='cuda',batch=64,eval_batch=128,eval_count=10000 if dataset=='mnist' else 50000,experiments=['E03','E08','E09','E10'])
    cfg.update(overrides)
    # Match mechanism controls to DHPF parameter count using nearest MLP width.
    if model in ['direct','augmented'] or (model not in ['dhpf'] and not cfg.get('image_shape')):
        target=count(Sources(cfg['d'],cfg['n'],cfg['source_width'],cfg['rank'],cfg['variant'],cfg['sigma_min'],cfg['fixed_width'],cfg['classes']))
        din,dout=(2*cfg['d'],2*cfg['d']) if model=='augmented' else (cfg['d'],cfg['d'])
        # Exact analytic count of VectorNet, excluding buffers.
        def size(w): return (din+17+(16 if cfg['classes'] else 0))*w+w+2*(w*w+w)+w*dout+dout+(16*cfg['classes'] if cfg['classes'] else 0)
        cfg['width']=min(range(16,1025),key=lambda w:abs(size(w)-target))
        cfg['parameter_target']=target
    return cfg


def write_protocol():
    RESULTS.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='checklist_v1',status='fixed-budget experiment; convergence is not assumed',train_seeds=list(range(5)),hpo_seeds=[100],hpo_candidates=[{'lr':.0002},{'lr':.001}],hpo_updates=1000,hpo_selection='smallest validation SW2 squared; same two candidates per method/task',validation_samples=512,validation_projection_count=64,validation_seed=70000,test_generation_seed=80000,test_projection_seed=61000,exact_w2=dict(samples=1024,repeats=5,seed=62000,ground_cost='squared Euclidean in training-standardized coordinates; mean per vector, not divided by d'),representative_ablation_tasks=['gmm8','sine'],sensitivity_seeds=list(range(5)),sensitivity=dict(n=[8,16,32,64],rank=[1,2,4],fixed_width=[False,True],sigma_min=[.05,.15,.4],batch=[32,64,128,256],velocity=['zero','halfnormal','gaussian'],lambda_sep=[0.,.001,.01]),reference_steps=[256,512,1024],reference_relative_rms_tolerance=.001,features=dict(mnist='training-only supervised digit CNN, frozen across all methods',cifar10='torch-fidelity Inception v3 compatibility features and logits; reference official test 10000'),drifting='Additional M7-raw: official drifting loss adapted to raw coordinates, no pretrained feature extractor; must not be described as official image reproduction',E11=True,E12=True,base=BASE,networks='time-conditioned MLP for vectors; compact residual U-Net for image DDPM/CFM; time-only residual source network for DHPF; source kernels remain analytic in pixel space',resources=dict(gpu='RTX 5070 Ti 16GB',cpu_threads=2),convergence='Always report budget_exhausted_convergence_unestablished; no automatic proof of convergence from finite curves')
    path=RESULTS/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old!=protocol: raise RuntimeError('Protocol already frozen; use a new version instead of mutating it')
    else: path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    return protocol


if __name__=='__main__': write_protocol()
