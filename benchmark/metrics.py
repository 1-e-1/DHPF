import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm


def sliced(x,y,projections=256,seed=61000):
    x=np.asarray(x).reshape(len(x),-1); y=np.asarray(y).reshape(len(y),-1)
    rng=np.random.default_rng(seed); directions=rng.normal(size=(x.shape[1],projections))
    directions/=np.linalg.norm(directions,axis=0)
    a,b=np.sort(x@directions,axis=0),np.sort(y@directions,axis=0)
    if len(a)!=len(b):
        q=(np.arange(min(len(a),len(b)))+.5)/min(len(a),len(b))
        a=np.stack([np.interp(q,(np.arange(len(a))+.5)/len(a),a[:,j]) for j in range(projections)],1)
        b=np.stack([np.interp(q,(np.arange(len(b))+.5)/len(b),b[:,j]) for j in range(projections)],1)
    return float(np.mean((a-b)**2))


def distribution(x,y,cfg,exact=False):
    x=np.asarray(x).reshape(len(x),-1); y=np.asarray(y).reshape(len(y),-1)
    out={'sw2_squared':sliced(x,y),'generated_count':len(x),'reference_count':len(y),'projection_seed':61000,'projections':256}
    if exact:
        vals=[]; rng=np.random.default_rng(62000)
        for repeat in range(5):
            a=x[rng.choice(len(x),1024,replace=False)]; b=y[rng.choice(len(y),1024,replace=False)]
            c=cdist(a,b,'sqeuclidean'); row,col=linear_sum_assignment(c); vals.append(float(c[row,col].mean()))
        out.update(w2_squared_1024_repeats=vals,w2_squared_1024_mean=float(np.mean(vals)),w2_squared_1024_resampling_std=float(np.std(vals,ddof=1)))
    if cfg['dataset']=='gmm8':
        from .data import DATA
        meta=json.loads((DATA/'gmm8.json').read_text())
        raw=x*np.array(meta['std'])+np.array(meta['mean'])
        a=np.arange(8)*np.pi/4; centers=2*np.stack([np.cos(a),np.sin(a)],1)
        dist=cdist(raw,centers); near=dist.argmin(1); valid=dist.min(1)<.36
        counts=np.bincount(near[valid],minlength=8); p=counts/max(1,counts.sum())
        out.update(mode_coverage=int((counts>=max(5,.01*len(x))).sum()),valid_fraction=float(valid.mean()),mode_probabilities=p.tolist(),mode_probability_total_variation=float(abs(p-1/8).sum()/2))
    return out


def sequence_stats(x):
    x=np.asarray(x,dtype=np.float64); centered=x-x.mean(1,keepdims=True)
    power=(abs(np.fft.rfft(centered,axis=1))**2/128).mean(0)
    var=(centered**2).mean((0,1))
    ac=np.stack([(centered[:,:128-l]*centered[:,l:]).mean((0,1))/np.maximum(var,1e-12) for l in range(33)])
    cross=np.corrcoef(x.reshape(-1,x.shape[-1]).T)
    return dict(power=power,autocorrelation=ac,cross_correlation=cross)


def sequence_metrics(x,y,bandwidth):
    xx,yy=np.asarray(x).reshape(len(x),-1),np.asarray(y).reshape(len(y),-1)
    # Fixed 1024-sample U-statistic; bandwidth fitted on real training data only.
    a,b=xx[:1024],yy[:1024]
    kaa=np.exp(-cdist(a,a,'sqeuclidean')/(2*bandwidth**2)); kbb=np.exp(-cdist(b,b,'sqeuclidean')/(2*bandwidth**2)); kab=np.exp(-cdist(a,b,'sqeuclidean')/(2*bandwidth**2))
    mmd=(kaa.sum()-np.trace(kaa))/(len(a)*(len(a)-1))+(kbb.sum()-np.trace(kbb))/(len(b)*(len(b)-1))-2*kab.mean()
    sa,sb=sequence_stats(x),sequence_stats(y)
    out={'mmd2_unbiased':float(mmd),'mmd_bandwidth':float(bandwidth)}
    for k in sa: out[k+'_rmse']=float(np.sqrt(np.mean((sa[k]-sb[k])**2)))
    return out,sa,sb


def frechet(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    ma,mb=a.mean(0),b.mean(0); ca,cb=np.cov(a,rowvar=False),np.cov(b,rowvar=False)
    root=sqrtm(ca@cb)
    if np.iscomplexobj(root):
        if np.max(abs(root.imag))>1e-3: raise ArithmeticError('Unstable covariance square root')
        root=root.real
    return float(max(0.,np.sum((ma-mb)**2)+np.trace(ca+cb-2*root)))
