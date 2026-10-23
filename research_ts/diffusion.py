"""Cosine DDPM with v prediction; same original VectorNet architecture.

Nichol & Dhariwal (2021) cosine schedule; Salimans & Ho (2022) v target.
No distillation, clipping, DDIM acceleration, or learned covariance is used.
"""
import math, time
import torch
from benchmark.models import generate as original_generate, sync

def diffusion_schedule(device):
    t=torch.linspace(0,1,1001,device=device,dtype=torch.float64)
    f=torch.cos((t+.008)/1.008*math.pi/2).square()
    beta=(1-f[1:]/f[:-1]).clamp(max=.999)
    alpha=1-beta; abar=torch.cumprod(alpha,0)
    prev=torch.cat([abar.new_ones(1),abar[:-1]])
    var=beta*(1-prev)/(1-abar)
    return beta,alpha,abar,var

@torch.no_grad()
def generate(model,cfg,n,seed,device,steps=None,condition=None,batch=None,initial=None,profile=False):
    if cfg['model']!='ddpm':
        return original_generate(model,cfg,n,seed,device,steps=steps,condition=condition,batch=batch,initial=initial,profile=profile)
    generator=torch.Generator(device=device).manual_seed(seed)
    xall=torch.randn(n,cfg['d'],generator=generator,device=device) if initial is None else initial[0]
    beta,alpha,abar,var=diffusion_schedule(device)
    prev=torch.cat([abar.new_ones(1),abar[:-1]])
    c0=(beta*prev.sqrt()/(1-abar)).float(); ct=(alpha.sqrt()*(1-prev)/(1-abar)).float()
    abar=abar.float();var=var.float()
    xs=[];sync(device); start=time.perf_counter()
    for lo in range(0,n,cfg['eval_batch']):
        x=xall[lo:lo+cfg['eval_batch']].clone()
        for j in reversed(range(1000)):
            vp=model(x,x.new_full((len(x),),j/999),condition)
            x0=abar[j].sqrt()*x-(1-abar[j]).sqrt()*vp
            x=c0[j]*x0+ct[j]*x
            if j: x=x+var[j].sqrt()*torch.randn(x.shape,device=device,generator=generator)
        xs.append(x.cpu())
    sync(device)
    return torch.cat(xs),dict(source_seconds=0.,field_seconds=time.perf_counter()-start,nfe=1000)

def audit():
    # Test parameter conversion and reverse posterior on random oracle inputs.
    beta,alpha,abar,var=diffusion_schedule('cpu');g=torch.Generator().manual_seed(42)
    y=torch.randn(1000,8,generator=g,dtype=torch.float64);eps=torch.randn(1000,8,generator=g,dtype=torch.float64)
    a=abar[:,None].sqrt();s=(1-abar[:,None]).sqrt();xt=a*y+s*eps;v=a*eps-s*y
    x0=a*xt-s*v; reconstructed=a*v+s*xt
    prev=torch.cat([abar.new_ones(1),abar[:-1]])[:,None]
    m0=beta[:,None]*prev.sqrt()/(1-abar[:,None])*x0+alpha[:,None].sqrt()*(1-prev)/(1-abar[:,None])*xt
    m1=(xt-beta[:,None]/s*reconstructed)/alpha[:,None].sqrt()
    errors={'x0_reconstruction_max':float((x0-y).abs().max()),'epsilon_reconstruction_max':float((reconstructed-eps).abs().max()),'reverse_mean_equivalence_max':float((m0-m1).abs().max())}
    assert errors['x0_reconstruction_max']<2e-6 and errors['epsilon_reconstruction_max']<2e-6
    assert errors['reverse_mean_equivalence_max']<3e-4
    assert torch.isfinite(var).all() and var.min()>=0 and var[0]==0
    return dict(passed=True,**errors,final_abar=float(abar[-1]),old_ddpm_note='No reverse-formula error identified by source inspection; unstable learned epsilon baseline is not evidence of DHPF superiority.')
