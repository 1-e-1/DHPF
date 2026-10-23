"""E11: independent supervised field fitting, never mixed into generation loss."""
import json
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .models import Sources,VectorNet,encode,count
from .protocol import RESULTS
from .run import write_json,version


def target(x,t,kind):
    xx,yy=x[:,0],x[:,1]
    # grad phi, phi=sin(pi*x)*cos(pi*y)*(1+.3*sin(2*pi*t))
    scale=1+.3*torch.sin(2*torch.pi*t)
    grad=torch.stack([torch.pi*torch.cos(torch.pi*xx)*torch.cos(torch.pi*yy),-torch.pi*torch.sin(torch.pi*xx)*torch.sin(torch.pi*yy)],-1)*scale[:,None]
    # J grad psi for a translated Gaussian potential, with time-varying center.
    center=torch.stack([.5*torch.sin(2*torch.pi*t),.5*torch.cos(2*torch.pi*t)],-1)
    delta=x-center;g=-2*delta*torch.exp(-delta.square().sum(-1,keepdim=True));sol=torch.stack([-g[:,1],g[:,0]],-1)
    return grad if kind=='gradient' else sol if kind=='solenoidal' else grad+sol


class RBF(nn.Module):
    def __init__(self,n):
        super().__init__();self.centers=nn.Parameter(torch.randn(n,2));self.width=nn.Parameter(torch.zeros(n));self.time=nn.Sequential(nn.Linear(17,64),nn.SiLU(),nn.Linear(64,n*2));self.n=n
    def forward(self,x,t):
        w=self.time(encode(t)).reshape(-1,self.n,2)
        k=torch.exp(-(x[:,None]-self.centers).square().sum(-1)/(2*(.1+F.softplus(self.width)).square()))
        return (k[...,None]*w).sum(1)


def query(model,name,x,t):
    if name!='dhpf':return model(x,t)
    p,a,s,b=model.prepare(t)
    delta=x[:,None]-p;k=delta*(delta.square().sum(-1)+s.square()).pow(-1.5)[...,None]
    return (a[...,None]*k+b[...,None]*torch.stack([-k[...,1],k[...,0]],-1)).sum(1)


def run():
    torch.set_num_threads(2)
    root=RESULTS/'E11';root.mkdir(parents=True,exist_ok=True)
    # Supplementary diagnostic: a predeclared single training seed, no superiority inference.
    write_json(root/'protocol.json',dict(training_seed=0,evaluation_seed=94000,updates=2000,batch=128,lr=.001,train_domain='[-1.5,1.5]^2 x [0,.8]',evaluation_domains=['unseen points in training domain','time extrapolation [.8,1]','spatial exterior [-2,2]^2 outside [-1.5,1.5]^2'],counts=[8,16,32,64],models=['dhpf','mlp','rbf'],targets=['gradient','solenoidal','mixed'],scope='single-seed supplementary representation diagnostic; not a five-seed main comparison',source_version=version()))
    for kind in ['gradient','solenoidal','mixed']:
        for n in [8,16,32,64]:
            for name in ['dhpf','mlp','rbf']:
                folder=root/f'{kind}__{name}__n{n}';folder.mkdir(exist_ok=True)
                if (folder/'metrics.json').exists():continue
                torch.manual_seed(0);model=Sources(2,n,width=32) if name=='dhpf' else VectorNet(2,2,64) if name=='mlp' else RBF(n)
                opt=torch.optim.Adam(model.parameters(),lr=.001);gen=torch.Generator().manual_seed(95000);records=[];start=time.perf_counter()
                for step in range(2000):
                    x=torch.rand(128,2,generator=gen)*3-1.5;t=torch.rand(128,generator=gen)*.8;y=target(x,t,kind)
                    opt.zero_grad(set_to_none=True);loss=(query(model,name,x,t)-y).square().sum(-1).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step()
                    if step%20==0:records.append(dict(step=step+1,loss=float(loss.detach()),seconds=time.perf_counter()-start))
                torch.save(dict(model=model.state_dict(),name=name,n=n,kind=kind),folder/'model.pt');write_json(folder/'curve.json',records)
                metrics=dict(parameters=count(model),training_seconds=time.perf_counter()-start,training_seed=0,convergence='unestablished at fixed budget')
                rng=torch.Generator().manual_seed(94000)
                for domain in ['interpolation','time_extrapolation','space_extrapolation']:
                    x=torch.rand(4096,2,generator=rng)*3-1.5;t=torch.rand(4096,generator=rng)*.8
                    if domain=='time_extrapolation':t=.8+torch.rand(4096,generator=rng)*.2
                    if domain=='space_extrapolation':
                        candidates=torch.rand(16384,2,generator=rng)*4-2;x=candidates[(candidates.abs()>1.5).any(-1)][:4096];t=t[:len(x)]
                    with torch.no_grad():pred=query(model,name,x,t);truth=target(x,t,kind)
                    mask=(truth.norm(dim=-1)>1e-6)&(pred.norm(dim=-1)>1e-6)
                    metrics[domain]=dict(vector_mse=float((pred-truth).square().sum(-1).mean()),direction_degrees=float(torch.acos(F.cosine_similarity(pred[mask],truth[mask]).clamp(-1,1)).mean()*180/torch.pi),count=len(x))
                grid=np.stack(np.meshgrid(np.linspace(-2,2,41),np.linspace(-2,2,41)),-1).reshape(-1,2);xx=torch.tensor(grid,dtype=torch.float32);tt=torch.full((len(xx),),.9)
                with torch.no_grad():pred=query(model,name,xx,tt).numpy();truth=target(xx,tt,kind).numpy()
                error=np.linalg.norm(pred-truth,axis=-1);np.savez_compressed(folder/'field.npz',grid=grid,predicted=pred,truth=truth,error=error,time=.9)
                fig,axs=plt.subplots(1,3,figsize=(11,3))
                for ax,arr,title in zip(axs[:2],[truth,pred],['Target field at unseen t=.9','Fitted field']):ax.quiver(grid[::3,0],grid[::3,1],arr[::3,0],arr[::3,1]);ax.set(title=title,aspect='equal')
                axs[2].imshow(error.reshape(41,41),origin='lower',extent=[-2,2,-2,2]);axs[2].set_title('Vector error norm');fig.tight_layout();fig.savefig(folder/'field.png',dpi=140);plt.close(fig)
                write_json(folder/'metrics.json',metrics);print(folder.name,metrics['interpolation']['vector_mse'],flush=True)
    write_json(root/'status.json',dict(status='completed_single_seed_supplementary',runs=36))


if __name__=='__main__':run()
