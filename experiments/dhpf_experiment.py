"""Reproducible 2-D experiments for DHPF V3. No training data enter source networks."""
import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn


def encode(t, m=8):
    f = torch.arange(1, m + 1, device=t.device, dtype=t.dtype)
    z = 2 * math.pi * t[..., None] * f
    return torch.cat([t[..., None], z.sin(), z.cos()], -1)


def prior(n, generator, device):
    x = torch.randn(n, 2, generator=generator, device=device)
    g = torch.randn(n, 2, generator=generator, device=device)
    v = torch.randn(n, 1, generator=generator, device=device).abs() * g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return x, v


def dataset(name, n, generator, device):
    if name == 'gmm8':
        k = torch.randint(8, (n,), generator=generator, device=device)
        a = k * (2 * math.pi / 8)
        return 2 * torch.stack([a.cos(), a.sin()], -1) + .12 * torch.randn(n, 2, generator=generator, device=device)
    if name == 'rings':
        k = torch.randint(2, (n,), generator=generator, device=device)
        a = torch.rand(n, generator=generator, device=device) * (2 * math.pi)
        r = 1 + k + .06 * torch.randn(n, generator=generator, device=device)
        return r[:, None] * torch.stack([a.cos(), a.sin()], -1)
    if name == 'spirals':
        t = torch.rand(n, generator=generator, device=device)
        k = torch.randint(2, (n,), generator=generator, device=device)
        a = .5 + 3 * math.pi * t + k * math.pi
        r = .25 + 2.25 * t
        return r[:, None] * torch.stack([a.cos(), a.sin()], -1) + .06 * torch.randn(n, 2, generator=generator, device=device)
    raise ValueError(name)


class Residual(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2*h), nn.SiLU(), nn.Linear(2*h, h))

    def forward(self, z):
        return z + self.net(z)


class Sources(nn.Module):
    def __init__(self, kind, n=32, h=64):
        super().__init__()
        self.kind, self.n = kind, n
        self.ids = nn.Parameter(torch.randn(n, h) * .2)
        self.time = nn.Linear(17, h)
        self.body = nn.Sequential(Residual(h), Residual(h))
        self.head = nn.Linear(h, 5)
        # Distinct centers and nonzero head derivatives; small initial force.
        nn.init.normal_(self.head.weight, std=.04)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.weight[2:4] *= .1

    def prepare(self, times):
        raw = self.head(self.body(self.ids[None] + self.time(encode(times))[:, None]))
        p = 3.5 * raw[..., :2].tanh()
        alpha, beta = 2 * raw[..., 2].tanh(), 2 * raw[..., 3].tanh()
        if self.kind == 'radial':
            beta = beta * 0
        if self.kind == 'tangential':
            alpha = alpha * 0
        sigma = .25 + 1.25 * raw[..., 4].sigmoid()
        return p, alpha, beta, sigma

    def field(self, x, prepared, index):
        p, alpha, beta, sigma = (a[index] for a in prepared)
        r = x[:, None] - p[None]
        k = r / (r.square().sum(-1) + sigma.square()[None]).pow(1.5)[..., None]
        jk = torch.stack([-k[..., 1], k[..., 0]], -1)
        return (alpha[None, :, None] * k + beta[None, :, None] * jk).sum(1)


class DirectMLP(nn.Module):
    def __init__(self, width=176):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(19, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))
        nn.init.normal_(self.net[-1].weight, std=.004)
        nn.init.zeros_(self.net[-1].bias)

    def prepare(self, times):
        return encode(times)

    def field(self, x, prepared, index):
        return self.net(torch.cat([x, prepared[index].expand(len(x), -1)], -1))


def integrate(model, x, v, steps):
    h = 1 / steps
    # Precompute ALL time-only sources once. This is algebraically identical to
    # evaluating G(t) separately at each deterministic midpoint stage.
    ts = torch.arange(2 * steps, dtype=x.dtype, device=x.device) / (2 * steps)
    prepared = model.prepare(ts)
    for k in range(steps):
        a = model.field(x, prepared, 2*k)
        xm, vm = x + .5*h*v, v + .5*h*a
        am = model.field(xm, prepared, 2*k+1)
        x, v = x + h*vm, v + h*am
    return x, v


def ot_loss(x, y):
    with torch.no_grad():
        cost = torch.cdist(x, y).square().cpu().numpy()
        _, col = linear_sum_assignment(cost)
    return (x - y[torch.as_tensor(col, device=y.device)]).square().sum(-1).mean()


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def sw2(x, y, directions):
    xp, yp = np.sort(x @ directions.T, axis=0), np.sort(y @ directions.T, axis=0)
    return float(np.mean((xp - yp)**2))


def coverage(name, x):
    if name == 'gmm8':
        a = np.arange(8) * 2*np.pi/8
        centers = 2*np.stack([np.cos(a), np.sin(a)], -1)
        dist = np.linalg.norm(x[:, None] - centers[None], axis=-1)
        nearest = dist.argmin(1)
        good = dist.min(1) < .36
        counts = np.bincount(nearest[good], minlength=8)
        return {'mode_coverage': int((counts >= max(5, .01*len(x))).sum()), 'on_mode_fraction': float(good.mean())}
    if name == 'rings':
        r = np.linalg.norm(x, axis=-1)
        good = np.minimum(abs(r-1), abs(r-2)) < .18
        return {'on_ring_fraction': float(good.mean()), 'inner_fraction': float((r<1.5).mean())}
    return {}


def evaluate(model, cfg, folder, device):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(90000 + cfg['seed'])
    n = cfg['eval_samples']
    with torch.no_grad():
        x0, v0 = prior(n, gen, device)
        y = dataset(cfg['dataset'], n, gen, device)
        y2 = dataset(cfg['dataset'], n, gen, device)
        chunks = 512
        def sample(k):
            return torch.cat([integrate(model, x0[i:i+chunks], v0[i:i+chunks], k)[0] for i in range(0, n, chunks)])
        sample(cfg['integration_steps'])
        sync(device)
        start = time.perf_counter()
        x = sample(cfg['integration_steps'])
        sync(device)
        elapsed = time.perf_counter()-start
        fine = sample(4*cfg['integration_steps'])
        p = model.prepare(torch.tensor([0., .5, .99], device=device))
        accel = model.field(x0, p, 1).norm(dim=-1)
    x, y, y2, fine = (a.cpu().numpy() for a in [x,y,y2,fine])
    rng = np.random.default_rng(314159)
    angles = rng.uniform(0, 2*np.pi, 256)
    directions = np.stack([np.cos(angles),np.sin(angles)], -1)
    metrics = {'sw2_squared':sw2(x,y,directions), 'reference_sw2_squared':sw2(y2,y,directions),
               'fine_sw2_squared':sw2(fine,y,directions), 'step_refinement_rmse':float(np.sqrt(np.mean(np.sum((x-fine)**2,-1)))),
               'sample_seconds':elapsed, 'seconds_per_1000_samples':elapsed*1000/n, 'nfe_per_sample':2*cfg['integration_steps'],
               'initial_query_acceleration_mean_t05':float(accel.mean()), 'initial_query_acceleration_max_t05':float(accel.max()),
               'finite_outputs': bool(np.isfinite(x).all()), **coverage(cfg['dataset'], x)}
    costs = ((x[:512,None,:]-y[None,:512,:])**2).sum(-1)
    row,col = linear_sum_assignment(costs)
    metrics['empirical_w2_squared_512'] = float(costs[row,col].mean())
    np.savez_compressed(folder/'samples.npz', generated=x, target=y, independent_target=y2, fine_generated=fine,
                        initial_x=x0.cpu().numpy(), initial_v=v0.cpu().numpy())
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(11,3.6))
    for ax,z,title in zip(axes,[y,x,fine],['Target',f"Generated K={cfg['integration_steps']}",f"Generated K={4*cfg['integration_steps']}"]):
        ax.scatter(z[:,0],z[:,1],s=2,alpha=.35)
        ax.set(xlim=(-3.6,3.6),ylim=(-3.6,3.6),title=title,aspect='equal')
    fig.suptitle(f"{cfg['dataset']} / {cfg['model']} / seed {cfg['seed']} | SW2^2={metrics['sw2_squared']:.4f}")
    fig.tight_layout(); fig.savefig(folder/'samples.png',dpi=150); plt.close(fig)
    return metrics


def run(cfg, root):
    folder=root/f"{cfg['dataset']}__{cfg['model']}__seed{cfg['seed']}"
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():
        print('SKIP completed',folder.name,flush=True)
        return
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    device=cfg['device']
    model=DirectMLP().to(device) if cfg['model']=='mlp' else Sources(cfg['model']).to(device)
    cfg={**cfg,'parameters':sum(p.numel() for p in model.parameters()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    # Dedicated stream: equal data/prior batches for all architectures at a seed.
    gen=torch.Generator(device=device).manual_seed(10000+cfg['seed'])
    sep_gen=torch.Generator(device=device).manual_seed(20000+cfg['seed'])
    sync(device); begin=time.perf_counter()
    log=[]
    for step in range(1,cfg['train_steps']+1):
        x,v=prior(cfg['batch_size'],gen,device)
        y=dataset(cfg['dataset'],cfg['batch_size'],gen,device)
        opt.zero_grad(set_to_none=True)
        out,_=integrate(model,x,v,cfg['integration_steps'])
        endpoint_loss=ot_loss(out,y)
        separation_loss=endpoint_loss.new_zeros(())
        if cfg['lambda_separation']>0:
            if not isinstance(model,Sources):
                raise ValueError('Source separation is only defined for source models')
            source_positions=model.prepare(torch.rand(4,generator=sep_gen,device=device))[0]
            pairs=torch.triu_indices(model.n,model.n,offset=1,device=device)
            difference=source_positions[:,pairs[0]]-source_positions[:,pairs[1]]
            separation_loss=(difference.square().sum(-1)+cfg['separation_epsilon']**2).rsqrt().mean()
        loss=endpoint_loss+cfg['lambda_separation']*separation_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss: {folder.name} step {step}')
        loss.backward()
        gn=nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
        opt.step()
        log.append({'step':step,'loss':float(loss.detach()),'endpoint_loss':float(endpoint_loss.detach()),
                    'separation_loss':float(separation_loss.detach()),'gradient_norm_before_clip':float(gn),'elapsed_seconds':time.perf_counter()-begin})
        if step==1 or step%100==0 or step==cfg['train_steps']:
            print(json.dumps({'run':folder.name,**log[-1]}),flush=True)
    sync(device); train_seconds=time.perf_counter()-begin
    with (folder/'training.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=log[0]);w.writeheader();w.writerows(log)
    torch.save({'state_dict':model.state_dict(),'config':cfg,'optimizer':opt.state_dict()},folder/'checkpoint.pt')
    metrics=evaluate(model,cfg,folder,device)
    metrics.update(train_seconds=train_seconds,final_train_loss_mean100=float(np.mean([q['loss'] for q in log[-100:]])))
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('RESULT '+json.dumps({'run':folder.name,**metrics}),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='results/main')
    ap.add_argument('--datasets',nargs='+',default=['gmm8','rings','spirals'])
    ap.add_argument('--models',nargs='+',default=['full','radial','tangential','mlp'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    ap.add_argument('--train-steps',type=int,default=800)
    ap.add_argument('--integration-steps',type=int,default=16)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--eval-samples',type=int,default=2048)
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--lambda-separation',type=float,default=0.)
    ap.add_argument('--separation-epsilon',type=float,default=.05)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    env={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,
         'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         'threads':args.threads,'argv':vars(args)}
    (root/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    for name in args.datasets:
        for seed in args.seeds:
            for model in args.models:
                cfg={'dataset':name,'model':model,'seed':seed,'device':args.device,'train_steps':args.train_steps,
                     'integration_steps':args.integration_steps,'batch_size':args.batch_size,'eval_samples':args.eval_samples,
                     'lr':args.lr,'source_count':32,'source_width':64,'source_blocks':2,'time_frequencies':list(range(1,9)),
                     'center_bound_per_coordinate':3.5,'weight_bound':2.,'sigma_min':.25,'sigma_max':1.5,
                     'lambda_separation':args.lambda_separation,'separation_epsilon':args.separation_epsilon,
                     'separation_time_samples':4,'gradient_clip':10.,'solver':'explicit_midpoint','mlp_width':176}
                run(cfg,root)


if __name__=='__main__':
    main()
