"""Reproducible E01/E11 summary tables and decomposition figures."""
import csv
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .protocol import RESULTS
from .models import Sources


def csvsave(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run():
    torch.set_num_threads(2)
    check=json.loads((RESULTS/'E01/checks.json').read_text())
    csvsave(RESULTS/'E01/structure.csv',check['structure']);csvsave(RESULTS/'E01/integrator.csv',check['numerics'])
    fig,ax=plt.subplots(figsize=(5,4));rows=check['numerics']
    for key in ['oscillator_position_error','oscillator_velocity_error']:ax.loglog([r['k'] for r in rows],[r[key] for r in rows],'-o',label=key)
    ax.set(xlabel='Midpoint integration steps',ylabel='Error against analytic solution');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(RESULTS/'E01/integration_error.png',dpi=150);plt.close(fig)
    allrows=[]
    for p in sorted((RESULTS/'E11').glob('*/metrics.json')):
        target,model,n=p.parent.name.split('__');m=json.loads(p.read_text());row=dict(target=target,model=model,n=int(n[1:]),parameters=m['parameters'],training_seconds=m['training_seconds'],seed=0)
        for domain in ['interpolation','time_extrapolation','space_extrapolation']:
            for key in ['vector_mse','direction_degrees']:row[domain+'_'+key]=m[domain][key]
        allrows.append(row)
    csvsave(RESULTS/'E11/summary.csv',allrows)
    for p in (RESULTS/'E11').glob('*__dhpf__*/model.pt'):
        ck=torch.load(p,weights_only=True);m=Sources(2,ck['n'],32);m.load_state_dict(ck['model'])
        with np.load(p.parent/'field.npz') as z:grid=z['grid']
        q=torch.tensor(grid,dtype=torch.float32)
        with torch.no_grad():
            params=m.prepare(torch.tensor([.9]));rad=m.field(q,params,component='radial').numpy();tan=m.field(q,params,component='tangential').numpy()
        np.savez_compressed(p.parent/'components.npz',grid=grid,radial=rad,tangential=tan,total=rad+tan,time=.9)
        fig,axs=plt.subplots(1,3,figsize=(10,3))
        for ax,v,title in zip(axs,[rad,tan,rad+tan],['Radial','Tangential','Total']):ax.quiver(grid[::3,0],grid[::3,1],v[::3,0],v[::3,1]);ax.set(title=title,aspect='equal')
        fig.tight_layout();fig.savefig(p.parent/'components.png',dpi=140);plt.close(fig)
    print('E01 and E11 tables and figures saved')


if __name__=='__main__':run()
