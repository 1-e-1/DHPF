"""Generative algorithms and shared-source DHPF, with no data-dependent sources."""
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def encode(t):
    f = torch.arange(1, 9, device=t.device, dtype=t.dtype)
    phase = 2 * math.pi * t[..., None] * f
    return torch.cat([t[..., None], phase.sin(), phase.cos()], -1)


def prior(n, d, gen, device, velocity='halfnormal'):
    x = torch.randn(n, d, generator=gen, device=device)
    g = torch.randn(n, d, generator=gen, device=device)
    radius = torch.randn(n, 1, generator=gen, device=device).abs()
    v = radius * F.normalize(g, dim=-1)
    if velocity == 'zero': v = v * 0
    if velocity == 'gaussian': v = g / math.sqrt(d)
    return x, v


class Res(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2*h), nn.SiLU(), nn.Linear(2*h, h))
    def forward(self, z): return z + self.net(z)


class Sources(nn.Module):
    def __init__(self, d, n=32, width=64, rank=2, variant='full', sigma_min=.15, fixed_width=False, classes=0):
        super().__init__()
        self.d, self.n, self.rank, self.variant = d, n, rank, variant
        self.sigma_min, self.fixed_width = sigma_min, fixed_width
        self.ids = nn.Parameter(torch.randn(n, width) * .3)
        self.time = nn.Linear(17, width)
        self.condition = nn.Embedding(classes, width) if classes else None
        self.body = nn.Sequential(Res(width), Res(width))
        out = d+3 if d == 2 else d*(1+2*rank)+2
        self.head = nn.Linear(width, out)
        nn.init.normal_(self.head.weight, std=.008)
        nn.init.zeros_(self.head.bias)
        # Distinct trainable source locations; time offsets remain unconstrained within bounded map.
        self.base = nn.Parameter(torch.randn(n, d) * .25)
        if d > 2:
            self.uv_base = nn.Parameter(torch.randn(n, 2, d, rank) * .3)
        self.bound = 8 * (d / 2)**1.5

    def prepare(self, times, condition=None):
        z = self.ids[None] + self.time(encode(times))[:, None]
        if self.condition is not None:
            if condition is None: raise ValueError('A single external condition is required for shared sources.')
            cc = torch.as_tensor(condition, device=times.device, dtype=torch.long).reshape(-1)
            z = z + self.condition(cc).reshape(-1, 1, z.shape[-1])
        raw = self.head(self.body(z))
        p_raw = self.base[None] + (0 * raw[..., :self.d] if self.variant.startswith('fixed_') else raw[..., :self.d])
        p = 3.5 * p_raw.tanh()
        alpha = self.bound * raw[..., self.d].tanh()
        sigma = self.sigma_min + (1.5-self.sigma_min) * raw[..., self.d+1].sigmoid()
        if self.fixed_width: sigma = sigma * 0 + .8
        if self.variant.endswith('tangential'): alpha = alpha * 0
        if self.d == 2:
            skew = self.bound * raw[..., self.d+2].tanh()
            if self.variant.endswith('radial'): skew = skew * 0
            return p, alpha, sigma, skew
        uv = raw[..., self.d+2:].reshape(len(times), self.n, 2, self.d, self.rank) + self.uv_base[None]
        # Each factor is bounded by u_max/sqrt(d*r), hence bounded Frobenius norm.
        uv = (math.sqrt(self.bound) / math.sqrt(self.d * self.rank)) * uv.tanh()
        if self.variant.endswith('radial'): uv = uv * 0
        return p, alpha, sigma, uv

    def field(self, x, prepared, index=0, component='full', diagnostic=False):
        p, alpha, sigma, skew = (a[index] for a in prepared)
        delta = x[:, None] - p[None]
        kernel = delta * (delta.square().sum(-1) + sigma.square()).pow(-1.5)[..., None]
        radial = (alpha[None, :, None] * kernel).sum(1)
        if self.d == 2:
            tangent = (skew[None, :, None] * torch.stack([-kernel[..., 1], kernel[..., 0]], -1)).sum(1)
        else:
            u, v = skew[:, 0], skew[:, 1]
            uk, vk = torch.einsum('bnd,ndr->bnr', kernel, u), torch.einsum('bnd,ndr->bnr', kernel, v)
            tangent = (torch.einsum('bnr,ndr->bnd', vk, u)-torch.einsum('bnr,ndr->bnd', uk, v)).sum(1)
        value = radial if component == 'radial' else tangent if component == 'tangential' else radial+tangent
        if diagnostic:
            return value, dict(kernel_norm=float(kernel.detach().norm(dim=-1).mean()), acceleration_norm=float(value.detach().norm(dim=-1).mean()), zero_kernel_fraction=float((kernel.detach().norm(dim=-1)==0).float().mean()), zero_acceleration_fraction=float((value.detach().norm(dim=-1)==0).float().mean()))
        return value

    def separation(self, times, condition=None):
        p = self.prepare(times, condition)[0]
        i,j = torch.triu_indices(self.n, self.n, 1, device=p.device)
        return ((p[:, i]-p[:, j]).square().sum(-1)+.05**2).rsqrt().mean()


class VectorNet(nn.Module):
    def __init__(self, din, dout, width=128, classes=0, spectral=False):
        super().__init__()
        def linear(a,b):
            layer=nn.Linear(a,b)
            return nn.utils.parametrizations.spectral_norm(layer) if spectral else layer
        self.net=nn.Sequential(linear(din+17+(16 if classes else 0),width),nn.SiLU(),linear(width,width),nn.SiLU(),linear(width,width),nn.SiLU(),linear(width,dout))
        self.condition = nn.Embedding(classes, 16) if classes else None

    def forward(self, x, t=None, condition=None):
        if t is None: t=x.new_zeros(len(x))
        if t.ndim == 0: t=t.expand(len(x))
        z=[x, encode(t)]
        if self.condition is not None:
            cc=torch.as_tensor(condition, device=x.device, dtype=torch.long).reshape(-1).expand(len(x))
            z.append(self.condition(cc))
        return self.net(torch.cat(z,-1))


class ImageNet(nn.Module):
    """Small residual U-Net for noise/velocity prediction in pixel space."""
    def __init__(self, shape, width=32):
        super().__init__(); self.shape=shape
        c=shape[0]; self.time=nn.Linear(17,width)
        self.first=nn.Conv2d(c,width,3,padding=1)
        self.down=nn.Conv2d(width,2*width,4,stride=2,padding=1)
        self.mid=nn.Sequential(nn.GroupNorm(8,2*width),nn.SiLU(),nn.Conv2d(2*width,2*width,3,padding=1),nn.SiLU(),nn.Conv2d(2*width,2*width,3,padding=1))
        self.up=nn.ConvTranspose2d(2*width,width,4,stride=2,padding=1)
        self.last=nn.Sequential(nn.GroupNorm(8,2*width),nn.SiLU(),nn.Conv2d(2*width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,c,3,padding=1))
    def forward(self,x,t=None,condition=None):
        if t is None: t=x.new_zeros(len(x))
        if t.ndim==0: t=t.expand(len(x))
        a=F.silu(self.first(x.reshape(-1,*self.shape))+self.time(encode(t))[:,:,None,None])
        b=F.silu(self.down(a)); b=b+self.mid(b)
        return self.last(torch.cat([a,self.up(b)],1)).flatten(1)


def count(m): return sum(p.numel() for p in m.parameters())


def build(cfg):
    d=cfg['d']; name=cfg['model']; classes=cfg.get('classes',0)
    if name=='dhpf': return Sources(d,cfg['n'],cfg['source_width'],cfg['rank'],cfg['variant'],cfg['sigma_min'],cfg['fixed_width'],classes)
    if name in ['ddpm','icfm','otcfm'] and cfg.get('image_shape'):
        return ImageNet(cfg['image_shape'],cfg.get('conv_width',32))
    din, dout=(2*d,2*d) if name=='augmented' else (d,d)
    return VectorNet(din,dout,cfg['width'],classes)


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()


def integrate(model, name, x, v, steps, condition=None, profile=False, trace=False):
    h=1/steps; stats={'source_seconds':0.,'field_seconds':0.}
    if profile: sync(x.device)
    start=time.perf_counter()
    prepared=model.prepare(torch.arange(2*steps,device=x.device,dtype=x.dtype)/(2*steps),condition) if name=='dhpf' else None
    if profile: sync(x.device)
    stats['source_seconds']=time.perf_counter()-start
    start=time.perf_counter(); snapshots=[x] if trace else None
    for k in range(steps):
        t=x.new_tensor(k*h); tm=t+.5*h
        if name in ['dhpf','direct']:
            a=model.field(x,prepared,2*k) if name=='dhpf' else model(x,t,condition)
            xm,vm=x+.5*h*v,v+.5*h*a
            am=model.field(xm,prepared,2*k+1) if name=='dhpf' else model(xm,tm,condition)
            x,v=x+h*vm,v+h*am
        elif name=='augmented':
            s=torch.cat([x,v],-1); f=model(s,t,condition); s=s+h*model(s+.5*h*f,tm,condition); x,v=s.chunk(2,-1)
        else:
            a=model(x,t,condition); x=x+h*model(x+.5*h*a,tm,condition)
        if trace and (k+1)*4 % steps==0: snapshots.append(x)
    if profile: sync(x.device)
    stats['field_seconds']=time.perf_counter()-start
    return x,v,stats,snapshots


def match(x,y):
    with torch.no_grad():
        cost=torch.cdist(x,y).square().double().cpu().numpy()
        _,idx=linear_sum_assignment(cost)
    return y[torch.as_tensor(idx,device=y.device)]


def diffusion_schedule(device,dtype=torch.float32):
    beta=torch.linspace(.0001,.02,1000,device=device,dtype=dtype)
    alpha=1-beta; abar=alpha.cumprod(0)
    prev=torch.cat([abar.new_ones(1),abar[:-1]])
    return beta,alpha,abar,beta*(1-prev)/(1-abar)


def drifting_loss(x,y):
    """Raw-coordinate drift objective; adapted from official MIT JAX drift_loss.py.

    No pretrained encoder or memory bank. This is labeled Drifting-raw, not a
    reproduction of the feature-space ImageNet system. Temperatures .02,.05,.2.
    https://github.com/lambertae/drifting/blob/main/drift_loss.py
    """
    with torch.no_grad():
        old=x.detach(); targets=torch.cat([old,y.detach()]); n,d=x.shape
        dist=torch.cdist(old,targets).clamp_min(1e-4)
        scale=dist.mean(); coords=(scale/math.sqrt(d)).clamp_min(1e-3)
        dd=dist/scale.clamp_min(1e-3)
        dd=dd+F.pad(torch.eye(n,device=x.device),(0,len(y)))*100
        force=torch.zeros_like(old)
        for temp in [.02,.05,.2]:
            logits=-dd/temp
            affinity=(logits.softmax(-1)*logits.softmax(-2)).clamp_min(1e-6).sqrt()
            neg,pos=affinity[:,:n],affinity[:,n:]
            coeff=torch.cat([-neg*pos.sum(-1,keepdim=True),pos*neg.sum(-1,keepdim=True)],1)
            f=coeff@(targets/coords)-coeff.sum(-1,keepdim=True)*(old/coords)
            force+=f/f.square().mean().clamp_min(1e-8).sqrt()
        target=old/coords+force
    return (x/coords-target).square().mean()


@torch.no_grad()
def generate(model,cfg,n,seed,device,steps=None,condition=None,batch=None,initial=None,profile=False):
    name=cfg['model']; d=cfg['d']; steps=steps or cfg['steps']; batch=batch or cfg.get('eval_batch',256)
    # All initial samples drawn in one canonical stream before batching.
    gen=torch.Generator(device=device).manual_seed(seed)
    x0,v0=prior(n,d,gen,device,cfg['velocity']) if initial is None else initial
    outputs=[]; totals={'source_seconds':0.,'field_seconds':0.}
    schedule=diffusion_schedule(device) if name=='ddpm' else None
    for lo in range(0,n,batch):
        x,v=x0[lo:lo+batch].clone(),v0[lo:lo+batch].clone()
        if name in ['gan','drifting']:
            x=model(x,condition=condition)
        elif name=='ddpm':
            beta,alpha,abar,var=schedule
            for j in reversed(range(1000)):
                eps=model(x,x.new_full((len(x),),j/999),condition)
                mean=(x-beta[j]/(1-abar[j]).sqrt()*eps)/alpha[j].sqrt()
                x=mean+var[j].sqrt()*torch.randn(x.shape,device=device,generator=gen) if j else mean
        else:
            x,v,stats,_=integrate(model,name,x,v,steps,condition,profile)
            for key in totals: totals[key]+=stats[key]
        outputs.append(x.cpu())
    return torch.cat(outputs),totals
