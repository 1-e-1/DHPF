"""Summarize inference diagnostics without treating time points as replicates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from research_ts.inference_emergence import OUT,load,reference,initial,trace,write_json

def summarize(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),std=float(a.std(ddof=1)) if len(a)>1 else None,n=len(a),values=a.tolist())

@torch.no_grad()
def circulation(x,v):
    # Frobenius norm of the mean antisymmetric angular-momentum tensor,
    # divided by mean individual tensor norm. Uses Gram identities to avoid
    # constructing d-by-d tensors; a rotation coordinate system is not chosen.
    x=torch.as_tensor(x,device='cuda',dtype=torch.float64);v=torch.as_tensor(v,device='cuda',dtype=torch.float64)
    x=x-x.mean(-2,keepdim=True);v=v-v.mean(-2,keepdim=True)
    xx=x@x.transpose(-1,-2);vv=v@v.transpose(-1,-2);xv=x@v.transpose(-1,-2)
    num=(2*((xx*vv).sum((-1,-2))-(xv*xv.transpose(-1,-2)).sum((-1,-2)))).clamp_min(0).sqrt()
    den=(2*(x.square().sum(-1)*v.square().sum(-1)-(x*v).sum(-1).square())).clamp_min(0).sqrt().sum(-1)
    return (num/den.clamp_min(1e-12)).cpu().numpy()

def shuffled_circulation(x,v):
    x=np.asarray(x,dtype=np.float64);v=np.asarray(v,dtype=np.float64)
    x-=x.mean(0);v-=v.mean(0)
    xx=x@x.T;vv=v@v.T;xv=x@v.T;nx=np.diag(xx);nv=np.diag(vv)
    rng=np.random.default_rng(91800);values=[]
    for _ in range(99):
        p=rng.permutation(len(x));vp=vv[np.ix_(p,p)];xp=xv[:,p]
        num=np.sqrt(max(0,2*((xx*vp).sum()-(xp*xp.T).sum())))
        den=np.sqrt(np.maximum(0,2*(nx*nv[p]-np.diag(xp)**2))).sum()
        values.append(float(num/max(den,1e-12)))
    return dict(mean=float(np.mean(values)),q95=float(np.quantile(values,.95)),values=values)

def main():
    assert (OUT/'complete.json').exists();torch.set_num_threads(2);threadpool_limits(2)
    x=pd.read_csv(OUT/'sample_dynamics.csv');s=pd.read_csv(OUT/'source_dynamics.csv')
    source=pd.DataFrame(json.loads((OUT/'source_summary.json').read_text()))
    primary=lambda z:z[z.condition.isna()|z.condition.eq(0)]
    p=primary(x);sp=primary(s);ss=primary(source)
    witness_x=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    witness_v=np.stack([-witness_x[:,1],witness_x[:,0]],axis=-1)
    assert np.allclose(circulation(witness_x[None],witness_v[None]),1)
    assert np.allclose(circulation(witness_x[None],witness_x[None]),0)
    result={};turns=[];angular=[];nulls=[];fine_angles=[]
    for d in ['sine','lorenz','har']:
        sub=p[p.dataset.eq(d)];src=ss[ss.dataset.eq(d)];full=sub[sub.variant.eq('full')]
        r=dict(controls={},sources={},stages={})
        for variant in ['full','untrained','ballistic','frozen_t0','radial_only','solenoidal_only']:
            end=sub[sub.variant.eq(variant)&sub.t.eq(1)]
            r['controls'][variant]={k:summarize(end[k]) for k in ['sw2','acf_rmse','corr_rmse','power_rmse','effective_dimension','variance_ratio']}
        for variant in ['full','untrained']:
            a=src[src.variant.eq(variant)]
            r['sources'][variant]={k:summarize(a[k]) for k in ['motion_alignment_mean','phase_shift_null_mean','source_radius_ratio']}
            rr=sp[(sp.dataset==d)&(sp.variant==variant)]
            r['sources'][variant]['radius_time_mean']=summarize(rr.groupby('seed').radius.mean())
        c=0 if d=='har' else None;model,cfg=load(d,0);real=reference(cfg,c).reshape(256,-1)
        xc=real-real.mean(0);gram=xc@xc.T
        r['real_reference_effective_dimension']=float(np.trace(gram)**2/np.square(gram).sum())
        for seed,group in full.groupby('seed'):
            group=group.sort_values('t');q=group.loc[group.variance_ratio.idxmin()]
            rmin=float(q.variance_ratio);end=float(group.iloc[-1].variance_ratio)
            half=group[group.effective_dimension<group.iloc[0].effective_dimension/2]
            turn=dict(dataset=d,seed=int(seed),minimum_spread_time=float(q.t),minimum_spread_ratio=rmin,
                terminal_over_minimum_spread=end/rmin,half_effective_dimension_time=float(half.iloc[0].t) if len(half) else None)
            turns.append(turn)
            path=OUT/'traces'/f'{d}__seed{seed}__c{c}__full.npz'
            with np.load(path) as z:
                a=circulation(z['x'],z['v'])
                nulls.append(dict(dataset=d,seed=int(seed),**shuffled_circulation(z['x'][-1],z['v'][-1])))
            for i,value in enumerate(a):angular.append(dict(dataset=d,seed=int(seed),variant='full',t=i/32,circulation_coherence=float(value)))
            for variant in ['untrained','ballistic','frozen_t0','radial_only','solenoidal_only']:
                with np.load(OUT/'traces'/f'{d}__seed{seed}__c{c}__{variant}.npz') as z:a=circulation(z['x'],z['v'])
                angular.append(dict(dataset=d,seed=int(seed),variant=variant,t=1.,circulation_coherence=float(a[-1])))
        r['initial_effective_dimension']=float(full[full.t.eq(0)].effective_dimension.mean())
        r['field']={k:summarize(full.groupby('seed')[k].mean()) for k in ['effective_sources','max_source_share','force_cancel','solenoidal_share']}
        r['stages']={k:summarize([v[k] for v in turns if v['dataset']==d]) for k in ['minimum_spread_time','minimum_spread_ratio','terminal_over_minimum_spread','half_effective_dimension_time']}
        r['circulation_shuffle_null']=summarize([n['mean'] for n in nulls if n['dataset']==d])
        r['circulation_shuffle_null_q95']=summarize([n['q95'] for n in nulls if n['dataset']==d])
        a0,b0=initial(cfg)
        with torch.no_grad():fine,fv=trace(model,a0,b0,128,c)
        fine_angles.append(dict(dataset=d,seed=0,condition=c,K=128,
            circulation_coherence=float(circulation(fine[-1:].cpu().numpy(),fv[-1:].cpu().numpy())[0])))
        result[d]=r
    adf=pd.DataFrame(angular);adf.to_csv(OUT/'circulation.csv',index=False,encoding='utf-8-sig')
    for d in result:
        result[d]['circulation_endpoint']={variant:summarize(g.circulation_coherence) for variant,g in adf[(adf.dataset==d)&adf.t.eq(1)].groupby('variant')}
    # Paired solver comparisons: seed 0 only, never mixed with five-seed means.
    refinements=[]
    for d in ['sine','lorenz','har']:
        for c in ([None] if d!='har' else range(6)):
            selection=x.dataset.eq(d)&x.seed.eq(0)&(x.condition.isna() if c is None else x.condition.eq(c))
            a=x[selection&x.variant.eq('full')].sort_values('t');b=x[selection&x.variant.eq('full_K128')].sort_values('t')
            refinements.append(dict(dataset=d,condition=c,
                effective_dimension_curve_correlation=float(np.corrcoef(a.effective_dimension,b.effective_dimension)[0,1]),
                coarse_final_effective_dimension=float(a.iloc[-1].effective_dimension),fine_final_effective_dimension=float(b.iloc[-1].effective_dimension),
                coarse_final_acf=float(a.iloc[-1].acf_rmse),fine_final_acf=float(b.iloc[-1].acf_rmse)))
    write_json(OUT/'summary.json',result);write_json(OUT/'stages.json',turns);write_json(OUT/'refinement_metrics.json',refinements)
    write_json(OUT/'circulation_nulls.json',nulls);write_json(OUT/'circulation_refinement.json',fine_angles)
    write_json(OUT/'circulation_witness.json',dict(coherent_planar_rotation=1,radial_motion=0,passed=True))
    write_json(OUT/'analysis_notes.json',dict(circulation='Post-hoc exploratory descriptor added after the initial source/sampling results; no significance claim.',
        formula='||sum_i(x_i v_i^T - v_i x_i^T)||_F / sum_i ||x_i v_i^T - v_i x_i^T||_F; x and v centered across samples.',
        reference='Participation ratio of one fixed 256-window real subset, not an intrinsic dimension estimator.',
        inference='High source alignment is compared against untrained networks; covariance compression is not automatically useful structure or self-organization.'))
    print(json.dumps({d:dict(initial_dimension=r['initial_effective_dimension'],final_dimension=r['controls']['full']['effective_dimension'],real_dimension=r['real_reference_effective_dimension'],source_alignment=r['sources']['full']['motion_alignment_mean'],untrained_alignment=r['sources']['untrained']['motion_alignment_mean'],stages=r['stages'],circulation=r['circulation_endpoint']['full']) for d,r in result.items()},indent=2))

if __name__=='__main__':main()
