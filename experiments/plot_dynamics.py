"""Visualize learned source motion, generated trajectories, and acceleration."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dhpf_experiment import Sources

ap=argparse.ArgumentParser();ap.add_argument('run');args=ap.parse_args()
folder=Path(args.run);torch.set_num_threads(2)
saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
cfg=saved['config'];assert cfg['model']!='mlp'
model=Sources(cfg['model']);model.load_state_dict(saved['state_dict']);model.eval()
samples=np.load(folder/'samples.npz');k=128;h=1/k
with torch.no_grad():
    times=torch.arange(2*k)/(2*k);prepared=model.prepare(times)
    x=torch.from_numpy(samples['initial_x'][:32].copy());v=torch.from_numpy(samples['initial_v'][:32].copy())
    paths=[x.numpy().copy()]
    for j in range(k):
        a=model.field(x,prepared,2*j);xm=x+.5*h*v;vm=v+.5*h*a
        am=model.field(xm,prepared,2*j+1);x=x+h*vm;v=v+h*am
        paths.append(x.numpy().copy())
    sources=prepared[0].numpy()
    gx,gy=np.meshgrid(np.linspace(-3,3,23),np.linspace(-3,3,23))
    grid=torch.tensor(np.stack([gx.ravel(),gy.ravel()],-1),dtype=torch.float32)
    field=model.field(grid,prepared,k).numpy()
    sigma=prepared[3].numpy()
paths=np.stack(paths)
delta=sources-sources.mean(axis=1,keepdims=True)
pairs=np.triu_indices(sources.shape[1],1)
distance=np.linalg.norm(sources[:,pairs[0]]-sources[:,pairs[1]],axis=-1)
diagnostics={'mean_center_norm':float(np.linalg.norm(sources,axis=-1).mean()),
             'center_spread_rms':float(np.sqrt(np.mean(np.sum(delta**2,axis=-1)))),
             'median_pair_distance':float(np.median(distance)),
             'mean_kernel_width':float(sigma.mean()),'min_kernel_width':float(sigma.min()),'max_kernel_width':float(sigma.max())}
(folder/'source_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,3,figsize=(13,4))
for i in range(sources.shape[1]):
    axes[0].plot(sources[:,i,0],sources[:,i,1],lw=.8,alpha=.65)
axes[0].scatter(sources[0,:,0],sources[0,:,1],s=10,c='black',label='t=0')
axes[0].set_title('Learned source-center paths');axes[0].legend()
for i in range(paths.shape[1]):
    axes[1].plot(paths[:,i,0],paths[:,i,1],lw=.8,alpha=.65)
axes[1].scatter(paths[-1,:,0],paths[-1,:,1],s=12,c='black',label='t=1')
axes[1].set_title('Sample-position paths (K=128)');axes[1].legend()
axes[2].quiver(gx,gy,field[:,0],field[:,1],np.linalg.norm(field,axis=-1),cmap='viridis')
axes[2].scatter(sources[k,:,0],sources[k,:,1],s=15,c='red',marker='x')
axes[2].set_title('Acceleration at t=0.5; red = sources')
for ax in axes:
    ax.set(xlim=(-3.7,3.7),ylim=(-3.7,3.7),aspect='equal')
fig.suptitle(f"{cfg['dataset']} / {cfg['model']} / seed {cfg['seed']} / {cfg['train_steps']} updates")
fig.tight_layout();fig.savefig(folder/'learned_dynamics.png',dpi=170);plt.close(fig)
print(folder/'learned_dynamics.png')
print(json.dumps(diagnostics))
