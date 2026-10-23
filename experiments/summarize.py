"""Aggregate completed runs without selecting seeds or hiding unsuccessful fits."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dhpf_experiment import sw2


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='results/main');args=ap.parse_args()
    root=Path(args.root)
    rows=[]
    for path in sorted(root.glob('*/metrics.json')):
        cfg=json.loads((path.parent/'config.json').read_text())
        metrics=json.loads(path.read_text())
        samples=np.load(path.parent/'samples.npz')
        angles=np.random.default_rng(314159).uniform(0,2*np.pi,256)
        directions=np.stack([np.cos(angles),np.sin(angles)],-1)
        rows.append({**{k:cfg[k] for k in ['dataset','model','seed','parameters','train_steps','integration_steps']},**metrics,
                     'untrained_ballistic_sw2_squared':sw2(samples['initial_x']+samples['initial_v'],samples['target'],directions)})
    if not rows:
        raise SystemExit('No completed runs')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (root/'all_runs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    groups={}
    keys=['sw2_squared','fine_sw2_squared','reference_sw2_squared','untrained_ballistic_sw2_squared',
          'empirical_w2_squared_512','step_refinement_rmse','train_seconds','seconds_per_1000_samples',
          'mode_coverage','on_mode_fraction','on_ring_fraction','inner_fraction']
    for row in rows:
        groups.setdefault((row['dataset'],row['model']),[]).append(row)
    summary=[]
    for (data,model),rs in groups.items():
        item={'dataset':data,'model':model,'runs':len(rs),'parameters':rs[0]['parameters']}
        for key in keys:
            values=[r[key] for r in rs if key in r]
            if values:
                item[key+'_mean']=float(np.mean(values));item[key+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
        summary.append(item)
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fields=list(dict.fromkeys(k for row in summary for k in row))
    with (root/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
    datasets=[d for d in ['gmm8','rings','spirals'] if any(r['dataset']==d for r in rows)]
    models=[m for m in ['full','radial','tangential','mlp'] if any(r['model']==m for r in rows)]
    fig,axes=plt.subplots(len(datasets),len(models)+1,figsize=(3*(len(models)+1),3*len(datasets)),squeeze=False)
    for i,data in enumerate(datasets):
        for j,model in enumerate(models):
            p=root/f'{data}__{model}__seed0'/'samples.npz'
            if not p.exists():
                axes[i,j+1].axis('off');continue
            a=np.load(p)
            if j==0:
                axes[i,0].scatter(a['target'][:,0],a['target'][:,1],s=2,alpha=.4)
                axes[i,0].set_title(data+' target')
            axes[i,j+1].scatter(a['generated'][:,0],a['generated'][:,1],s=2,alpha=.4)
            axes[i,j+1].set_title(model+' (seed 0)')
        for ax in axes[i]:
            ax.set(xlim=(-3.5,3.5),ylim=(-3.5,3.5),aspect='equal')
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,len(datasets),figsize=(4.3*len(datasets),3.5))
    axes=np.atleast_1d(axes)
    for ax,data in zip(axes,datasets):
        for model in models:
            logs=[]
            for p in sorted(root.glob(f'{data}__{model}__seed*/training.csv')):
                log=np.genfromtxt(p,delimiter=',',names=True)
                logs.append(log['endpoint_loss'] if 'endpoint_loss' in log.dtype.names else log['loss'])
            if logs:
                a=np.mean(np.stack(logs),0)
                smooth=np.convolve(a,np.ones(40)/40,mode='valid')
                ax.plot(np.arange(len(smooth))+40,smooth,label=model)
        ax.set(title=data,xlabel='Training step',ylabel='Endpoint minibatch OT');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=170);plt.close(fig)
    print(json.dumps({'completed_runs':len(rows),'summary':summary},indent=2))


if __name__=='__main__':
    main()
