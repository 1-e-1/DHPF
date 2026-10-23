"""Read-only inference diagnostics for shared sources and transported samples.

No training or checkpoint writes. Counterfactual field masks are diagnostics,
not new fitted models. Generation time is not observation time.
"""
import argparse,copy,csv,hashlib,json,os,platform,time,traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from benchmark.models import build,prior,integrate
from benchmark.metrics import sequence_stats
from benchmark.run import load_data,write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/inference_emergence_20261017'
VARIANTS=['full','untrained','ballistic','frozen_t0','radial_only','solenoidal_only']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def folder(dataset,seed):return ROOT/'results/checklist_v1/runs'/f'{dataset}__dhpf__base__seed{seed}'
def load(dataset,seed):
    f=folder(dataset,seed);cfg=read(f/'config.json');m=build(cfg).cuda()
    ck=torch.load(f/'best.pt',map_location='cuda',weights_only=False)
    m.load_state_dict(ck['model']);m.eval();return m,cfg
def frozen():
    hashes=read(ROOT/'results/timeseries_v2/frozen_original.json')['sha256']
    assert all(sha(ROOT/p)==h for p,h in hashes.items())

@torch.no_grad()
def trace(model,x0,v0,steps,condition,variant='full'):
    h=1/steps;x=x0.clone();v=v0.clone();xs=[x.clone()];vs=[v.clone()]
    ts=torch.arange(2*steps,device=x.device,dtype=x.dtype)/(2*steps)
    pp=model.prepare(ts,condition)
    if variant=='frozen_t0':pp=tuple(q[:1].expand_as(q) for q in pp)
    component={'radial_only':'radial','solenoidal_only':'tangential'}.get(variant,'full')
    for k in range(steps):
        if variant=='ballistic':x=x0+(k+1)*h*v0;v=v0
        else:
            a=model.field(x,pp,2*k,component=component)
            xm,vm=x+.5*h*v,v+.5*h*a
            am=model.field(xm,pp,2*k+1,component=component)
            x,v=x+h*vm,v+h*am
        xs.append(x.clone());vs.append(v.clone())
    xs=torch.stack(xs);vs=torch.stack(vs)
    assert torch.isfinite(xs).all() and torch.isfinite(vs).all()
    return xs,vs

@torch.no_grad()
def source_curves(model,condition):
    times=torch.linspace(0,1,129,device='cuda');p,alpha,sigma,uv=model.prepare(times,condition)
    pc=p-p.mean(1,keepdim=True);dist=torch.cdist(p,p)
    ix=torch.triu_indices(model.n,model.n,1,device=p.device)
    pair=dist[:,ix[0],ix[1]]
    speed=torch.gradient(p,spacing=(1/128,),dim=(0,))[0]
    unit=torch.nn.functional.normalize(speed,dim=-1)
    alignment=unit.mean(1).norm(dim=-1)
    # Independent circular phase shifts preserve each source's velocity samples.
    # Compare synchronized directions against the same precomputed derivatives,
    # avoiding artificial derivatives at circular-wrap seams.
    units=unit.cpu().numpy();rng=np.random.default_rng(91700)
    selected=np.arange(0,128,8);null=[]
    for _ in range(99):
        shifts=rng.integers(0,128,size=model.n)
        v=units[(selected[:,None]+shifts[None,:])%128,np.arange(model.n)[None,:]]
        null.append(float(np.linalg.norm(v.mean(1),axis=-1).mean()))
    obs=float(alignment[selected].mean())
    records=[]
    for i,t in enumerate(times):
        records.append(dict(t=float(t),radius=float(pc[i].square().sum(-1).mean().sqrt()),
            pair_distance=float(pair[i].mean()),distance_over_width=float((pair[i]/((sigma[i,ix[0]]+sigma[i,ix[1]])/2)).mean()),
            motion_alignment=float(alignment[i]),speed=float(speed[i].norm(dim=-1).mean()),
            negative_alpha_fraction=float((alpha[i]<0).float().mean()),
            alpha_balance=float(alpha[i].sum()/alpha[i].abs().sum().clamp_min(1e-12)),
            mean_width=float(sigma[i].mean()),centroid_norm=float(p[i].mean(0).norm())))
    extra=dict(motion_alignment_mean=obs,phase_shift_null_mean=float(np.mean(null)),
        phase_shift_null_q95=float(np.quantile(null,.95)),phase_shift_null_values=null,
        phase_shift_exceedance=(1+sum(v>=obs for v in null))/100,
        source_radius_ratio=records[-1]['radius']/records[0]['radius'])
    return records,extra,dict(source_times=times.cpu().numpy(),source_positions=p.cpu().numpy(),source_alpha=alpha.cpu().numpy(),source_sigma=sigma.cpu().numpy())

@torch.no_grad()
def field_stats(model,x,prepared,i):
    p,alpha,sigma,uv=(a[i] for a in prepared)
    delta=x[:,None]-p[None]
    kernel=delta*(delta.square().sum(-1)+sigma.square()).pow(-1.5)[...,None]
    radial=alpha[None,:,None]*kernel
    u,v=uv[:,0],uv[:,1]
    uk,vk=torch.einsum('bnd,ndr->bnr',kernel,u),torch.einsum('bnd,ndr->bnr',kernel,v)
    sol=torch.einsum('bnr,ndr->bnd',vk,u)-torch.einsum('bnr,ndr->bnd',uk,v)
    force=radial+sol;norms=force.norm(dim=-1);weights=norms/norms.sum(1,keepdim=True).clamp_min(1e-12)
    a=force.sum(1);ar=radial.sum(1);at=sol.sum(1)
    ax=x-x.mean(0);vx=ar # radial contraction is distinct from trajectory contraction
    return dict(force_cancel=float((1-a.norm(dim=-1)/norms.sum(1).clamp_min(1e-12)).mean()),
        effective_sources=float((1/weights.square().sum(1).clamp_min(1e-12)).mean()),
        max_source_share=float(weights.max(1).values.mean()),
        solenoidal_share=float((at.norm(dim=-1)/(ar.norm(dim=-1)+at.norm(dim=-1)).clamp_min(1e-12)).mean()),
        acceleration_norm=float(a.norm(dim=-1).mean()),
        acceleration_alignment=float(torch.nn.functional.normalize(a,dim=-1).mean(0).norm()))

def reference(cfg,condition):
    arrays=load_data(cfg);real=arrays['test']
    if condition is not None:real=real[arrays['test_labels']==condition]
    rng=np.random.default_rng(91400+(condition or 0));real=real[rng.choice(len(real),256,replace=False)]
    return real

@torch.no_grad()
def metrics(xs,vs,real,cfg,model,condition,force_diagnostics=False):
    d=cfg['d'];y=real.reshape(len(real),-1);ref=sequence_stats(real)
    rng=np.random.default_rng(91500);proj=rng.normal(size=(d,128));proj/=np.linalg.norm(proj,axis=0)
    real_project=np.sort(y@proj,axis=0)
    ts=torch.linspace(0,1,len(xs),device=xs.device);pp=model.prepare(ts,condition) if force_diagnostics else None
    rows=[];init_var=float((xs[0]-xs[0].mean(0)).square().sum(-1).mean())
    for i,(x,v) in enumerate(zip(xs,vs)):
        xc=x-x.mean(0);vc=v-v.mean(0);gram=xc@xc.T
        var=xc.square().sum(-1).mean();vv=vc.square().sum(-1).mean()
        pr=gram.trace().square()/gram.square().sum().clamp_min(1e-12)
        radial_speed=(xc*vc).sum(-1)/xc.norm(dim=-1).clamp_min(1e-12)
        tangent_fraction=1-radial_speed.square().mean()/vv.clamp_min(1e-12)
        arr=x.cpu().numpy();stats=sequence_stats(arr.reshape(len(x),*cfg['shape']))
        row=dict(t=i/(len(xs)-1),position_variance=float(var),variance_ratio=float(var/init_var),
            effective_dimension=float(pr),speed=float(v.norm(dim=-1).mean()),
            velocity_alignment=float(torch.nn.functional.normalize(v,dim=-1).mean(0).norm()),
            expansion_rate=float((xc*vc).sum(-1).mean()/var.clamp_min(1e-12)),
            tangential_velocity_fraction=float(tangent_fraction),
            sw2=float(np.mean((np.sort(arr@proj,axis=0)-real_project)**2)))
        for k,new in [('autocorrelation','acf_rmse'),('cross_correlation','corr_rmse'),('power','power_rmse')]:
            row[new]=float(np.sqrt(np.mean((stats[k]-ref[k])**2)))
        if force_diagnostics:row.update(field_stats(model,x,pp,i))
        rows.append(row)
    return rows

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def initial(cfg,n=256):
    return prior(n,cfg['d'],torch.Generator(device='cuda').manual_seed(91300),'cuda',cfg['velocity'])

@torch.no_grad()
def sanity():
    frozen();results=[]
    for dataset in ['sine','lorenz','har']:
        model,cfg=load(dataset,0);c=0 if cfg['classes'] else None;x,v=initial(cfg,16)
        xx,vv=trace(model,x,v,32,c)
        base,bv,_,_=integrate(model,'dhpf',x.clone(),v.clone(),32,c)
        error=float((xx[-1]-base).abs().max());assert error==0
        pp=model.prepare(torch.tensor([.5],device='cuda'),c)
        force=model.field(xx[16],pp);one=model.field(xx[16,:1],pp)
        f_error=float((force[:1]-one).abs().max());assert f_error<1e-3
        solo,sv=trace(model,x[:1],v[:1],32,c)
        b_error=float((solo[:,0]-xx[:,0]).abs().max())
        pert=copy.deepcopy(model);pert.base[0].add_(.1)
        pq=pert.prepare(torch.tensor([.5],device='cuda'),c)
        other_error=max(float((a[:,1:]-b[:,1:]).abs().max()) for a,b in zip(pp,pq))
        own_change=float((pp[0][:,0]-pq[0][:,0]).norm());assert other_error==0 and own_change>0
        ref=metrics(xx,vv,reference(cfg,c)[:16],cfg,model,c,True)
        assert all(np.isfinite(val) for r in ref for val in r.values())
        # The detailed field decomposition must reproduce the original field.
        xb,vb=trace(model,x,v,32,c,'ballistic');assert torch.allclose(xb[-1],x+v)
        results.append(dict(dataset=dataset,instrumented_endpoint_max_abs_error=error,
            solo_vs_batch_force_max_abs_error=f_error,solo_vs_batch_trajectory_max_abs_error=b_error,
            other_sources_after_source0_perturb_max_abs_error=other_error,perturbed_source_displacement=own_change))
    write_json(OUT/'sanity.json',dict(passed=True,checks=results));print('SANITY PASS',flush=True)

@torch.no_grad()
def run():
    assert read(OUT/'sanity.json')['passed'];frozen();start=time.perf_counter();all_rows=[];source_rows=[];summaries=[];hashes={};checks=[]
    cases=[(d,s,0 if d=='har' else None) for d in ['sine','lorenz','har'] for s in range(5)]
    cases += [('har',0,c) for c in range(1,6)]
    for dataset,seed,c in cases:
        model,cfg=load(dataset,seed);ck=folder(dataset,seed)/'best.pt';hashes[str(ck.relative_to(ROOT))]=sha(ck)
        real=reference(cfg,c);x0,v0=initial(cfg)
        # Reconstruct the original random initialization, without optimizer steps.
        torch.manual_seed(seed);untrained=build(cfg).cuda().eval()
        key=f'{dataset}__seed{seed}__c{c}'
        variants=VARIANTS if dataset!='har' or c==0 else ['full']
        for variant in variants:
            m=untrained if variant=='untrained' else model
            xs,vs=trace(m,x0,v0,32,c,variant)
            rows=metrics(xs,vs,real,cfg,m,c,variant in ['full','untrained'])
            meta=dict(dataset=dataset,seed=seed,condition=c,variant=variant)
            all_rows.extend([dict(**meta,**r) for r in rows])
            sources=None
            if variant in ['full','untrained']:
                sr,se,sources=source_curves(m,c);source_rows.extend([dict(**meta,**r) for r in sr])
                summaries.append(dict(**meta,**se))
            # Preserve every inference state for full models; controls retain
            # endpoints plus all quantitative diagnostics for reproducibility.
            dst=OUT/'traces'/f'{key}__{variant}.npz';dst.parent.mkdir(exist_ok=True)
            saved=dict(x=xs.cpu().numpy() if variant=='full' else xs[-1:].cpu().numpy(),
                       v=vs.cpu().numpy() if variant=='full' else vs[-1:].cpu().numpy())
            if sources is not None:saved.update(sources)
            np.savez_compressed(dst,**saved)
            print('DONE',key,variant,'SW',round(rows[-1]['sw2'],4),flush=True)
        # Same initial states and trained checkpoint, finer solver, seed 0 only.
        if seed==0:
            fine,fv=trace(model,x0,v0,128,c)
            fine_rows=metrics(fine[::4],fv[::4],real,cfg,model,c,True)
            all_rows.extend([dict(dataset=dataset,seed=seed,condition=c,variant='full_K128',**r) for r in fine_rows])
            with np.load(OUT/'traces'/f'{key}__full.npz') as z:coarse=torch.as_tensor(z['x'],device='cuda')
            err=(fine[::4]-coarse).square().mean((1,2)).sqrt()/fine[::4].square().mean((1,2)).sqrt().clamp_min(1e-12)
            checks.append(dict(dataset=dataset,seed=seed,condition=c,endpoint_relative_rms=float(err[-1]),max_path_relative_rms=float(err.max())))
        write_csv(OUT/'sample_dynamics.csv',all_rows);write_csv(OUT/'source_dynamics.csv',source_rows)
        write_json(OUT/'source_summary.json',summaries);write_json(OUT/'refinement.json',checks)
        write_json(OUT/'progress.json',dict(completed_cases=cases.index((dataset,seed,c))+1,total_cases=len(cases),elapsed_seconds=time.perf_counter()-start))
    frozen();assert all(sha(ROOT/p)==h for p,h in hashes.items())
    write_json(OUT/'complete.json',dict(completed_cases=len(cases),main_training_seeds=5,full_inference_paths=20,
        primary_counterfactual_paths=75,refined_paths=8,training_updates=0,samples_per_path=256,
        checkpoint_hashes=hashes,original_source_unchanged=True,checkpoints_unchanged=True,
        runtime_seconds=time.perf_counter()-start,peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/2**20))

def main():
    p=argparse.ArgumentParser();p.add_argument('--sanity',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);threadpool_limits(2)
    torch.backends.cuda.matmul.allow_tf32=False
    write_json(OUT/('sanity_launch.json' if a.sanity else 'launch.json'),dict(pid=os.getpid(),command='python -u -m research_ts.inference_emergence'+(' --sanity' if a.sanity else ''),cwd=str(ROOT),python=platform.python_version(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),diagnostic_source_sha256=sha(Path(__file__))))
    try:sanity() if a.sanity else run()
    except Exception:
        write_json(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
