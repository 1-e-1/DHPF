"""Post-training diagnostics on saved, independent samples and checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import torch
from dhpf_experiment import Sources, DirectMLP, integrate, sw2, coverage


def exact_w2(x,y):
    c=((x[:,None]-y[None])**2).sum(-1)
    i,j=linear_sum_assignment(c)
    return float(c[i,j].mean())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='results/main');args=ap.parse_args()
    torch.set_num_threads(2)
    root=Path(args.root)
    angles=np.random.default_rng(314159).uniform(0,2*np.pi,256)
    directions=np.stack([np.cos(angles),np.sin(angles)],-1)
    t=np.linspace(0,1,4096);a=.5+3*np.pi*t;r=.25+2.25*t
    curve=r[:,None]*np.stack([np.cos(a),np.sin(a)],-1)
    tree=cKDTree(np.concatenate([curve,-curve]))
    results=[]
    for cp in sorted(root.glob('*/checkpoint.pt')):
        cfg=json.loads((cp.parent/'config.json').read_text())
        saved=np.load(cp.parent/'samples.npz')
        record={k:cfg[k] for k in ['dataset','model','seed']}
        record['target_target_empirical_w2_squared_512']=exact_w2(saved['independent_target'][:512],saved['target'][:512])
        record['fine_coverage']=coverage(cfg['dataset'],saved['fine_generated'])
        record['target_coverage']=coverage(cfg['dataset'],saved['target'])
        if cfg['dataset']=='spirals':
            for name,key in [('generated','generated'),('fine','fine_generated'),('target','target')]:
                distance=tree.query(saved[key])[0]
                record[name+'_spiral_precision_018']=float((distance<.18).mean())
                record[name+'_spiral_mean_distance']=float(distance.mean())
        # Selected in advance: seed zero for EACH model/dataset, not best cases.
        if cfg['seed']==0:
            model=DirectMLP() if cfg['model']=='mlp' else Sources(cfg['model'])
            model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['state_dict'])
            model.eval()
            x=torch.from_numpy(saved['initial_x'][:512]);v=torch.from_numpy(saved['initial_v'][:512])
            ks=[16,32,64,128,256]
            with torch.no_grad():
                out={k:integrate(model,x,v,k)[0].numpy() for k in ks}
            record['rmse_to_k256']={str(k):float(np.sqrt(np.mean(np.sum((out[k]-out[256])**2,-1)))) for k in ks[:-1]}
            record['sw2_squared_by_steps_512']={str(k):sw2(out[k],saved['target'][:512],directions) for k in ks}
        results.append(record)
        print(json.dumps(record),flush=True)
    (root/'diagnostics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
