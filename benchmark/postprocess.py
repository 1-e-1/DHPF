"""E08-E10 and downstream sequence evaluation from frozen checkpoints."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from .data import DATA
from .protocol import RESULTS
from .models import prior,generate,integrate,sync
from .metrics import sliced,distribution,sequence_metrics
from .run import load_model,load_data,write_json


@torch.no_grad()
def reference_dhpf(model,x,v,steps,condition):
    """Exact same midpoint stages, source precomputation in bounded-size time chunks.

    Avoids allocating K*N*d*r source arrays for 3072-D fine reference solves.
    Chunks only alter storage; they never depend on samples or their positions.
    """
    h=1/steps;x=x.clone();v=v.clone();prepared=None
    for k in range(steps):
        if k%32==0:
            times=torch.arange(2*k,min(2*(k+32),2*steps),device=x.device,dtype=x.dtype)/(2*steps)
            prepared=model.prepare(times,condition)
        j=2*(k%32);acc=model.field(x,prepared,j);xm=x+.5*h*v;vm=v+.5*h*acc
        am=model.field(xm,prepared,j+1);x=x+h*vm;v=v+h*am
    return x.cpu().numpy()


def downstream(folder,cfg,arrays,generated,labels):
    """Predictors are external evaluations; never part of the generator loss."""
    path=folder/'downstream.json'
    if path.exists(): return
    rng=np.random.default_rng(90000)
    if cfg['dataset']=='har':
        y=labels.astype(int); n=len(generated)
        ids=np.concatenate([rng.choice(np.flatnonzero(arrays['train_labels']==c),int((y==c).sum()),replace=False) for c in range(cfg['classes'])])
        sets={'synthetic':(generated,y),'real_matched':(arrays['train'][ids],arrays['train_labels'][ids])}
        out={}
        for label,(xx,yy) in sets.items():
            best=None; records=[]
            for c in [.01,1.,100.]:
                model=LogisticRegression(C=c,max_iter=2000,random_state=90000)
                model.fit(xx.reshape(len(xx),-1),yy)
                val=f1_score(arrays['val_labels'],model.predict(arrays['val'].reshape(len(arrays['val']),-1)),average='macro')
                records.append(dict(C=c,val_macro_f1=float(val)))
                if best is None or val>best[0]: best=(val,c,model)
            pred=best[2].predict(arrays['test'].reshape(len(arrays['test']),-1))
            out[label]=dict(test_macro_f1=float(f1_score(arrays['test_labels'],pred,average='macro')),test_accuracy=float(accuracy_score(arrays['test_labels'],pred)),training_count=len(xx),selected_C=best[1],validation_trials=records)
        write_json(path,dict(task='TSTR activity classification on official test subjects',predictor='L2 multinomial logistic regression on flattened 128x9 raw windows',**out))
    else:
        n=min(len(generated),len(arrays['train'])); ids=rng.choice(len(arrays['train']),n,replace=False)
        def xy(z):
            # Collapsed generators may produce nearly rank-deficient design matrices.
            # Float64 keeps the prescribed ridge regularization numerically meaningful.
            z=np.asarray(z,dtype=np.float64)
            return z[:,:96].reshape(len(z),-1),z[:,96:].reshape(len(z),-1)
        vx,vy=xy(arrays['val']); tx,ty=xy(arrays['test']); out={}
        for label,xx in [('synthetic',generated[:n]),('real_matched',arrays['train'][ids])]:
            ax,ay=xy(xx); candidates=[]; best=None
            for alpha in [.01,1.,100.]:
                model=Ridge(alpha=alpha); model.fit(ax,ay)
                val=float(np.mean((model.predict(vx)-vy)**2)); candidates.append(dict(alpha=alpha,val_mse=val))
                if best is None or val<best[0]: best=(val,alpha,model)
            out[label]=dict(test_forecast_mse=float(np.mean((best[2].predict(tx)-ty)**2)),training_count=n,selected_alpha=best[1],validation_trials=candidates)
        write_json(path,dict(task='external 96-to-32 forecasting; generators remain unconditional full-window models',predictor='multi-output ridge regression',**out))


def plots(folder,cfg,arrays,x):
    if (folder/'sample_plot.png').exists() and (folder/'nearest_neighbors.npz').exists(): return
    y=arrays['test']; shape=cfg['shape']
    if cfg['d']==2:
        fig,axs=plt.subplots(1,2,figsize=(8,4))
        for ax,z,title in zip(axs,[y,x],['Held-out real','Generated, first 10000']):
            ax.scatter(z[:10000,0],z[:10000,1],s=1,alpha=.35);ax.set(xlim=(-3,3),ylim=(-3,3),aspect='equal',title=title)
    elif len(shape)==2:
        fig,axs=plt.subplots(4,2,figsize=(10,8))
        for i in range(4):
            axs[i,0].plot(y[i]);axs[i,1].plot(x[i]);axs[i,0].set_title(f'Real #{i}');axs[i,1].set_title(f'Generated #{i}')
        stats=list(folder.glob('sequence_stats_*.npz'))
        for s in stats:
            with np.load(s) as z:
                f,aa=plt.subplots(1,3,figsize=(12,3))
                aa[0].plot(z['real_power']);aa[0].plot(z['generated_power'],ls='--');aa[0].set_title('Power: real solid / generated dashed')
                aa[1].plot(z['real_autocorrelation']);aa[1].plot(z['generated_autocorrelation'],ls='--');aa[1].set_title('Autocorrelation')
                aa[2].imshow(z['generated_cross_correlation']-z['real_cross_correlation'],vmin=-1,vmax=1,cmap='coolwarm');aa[2].set_title('Cross-correlation difference')
                f.tight_layout();f.savefig(folder/(s.stem+'.png'),dpi=150);plt.close(f)
    else:
        fig,axs=plt.subplots(8,8,figsize=(8,8))
        for ax,z in zip(axs.flat,x[:64]):
            im=((z+1)/2).clip(0,1).transpose(1,2,0)
            ax.imshow(im[:,:,0] if im.shape[-1]==1 else im,cmap='gray' if im.shape[-1]==1 else None,vmin=0,vmax=1);ax.axis('off')
    fig.suptitle(f"{cfg['dataset']} / {cfg['model']} / seed {cfg['seed']}; fixed first samples")
    fig.tight_layout();fig.savefig(folder/'sample_plot.png',dpi=160);plt.close(fig)
    # Exact pixel/vector nearest neighbors for a fixed first-32 generated subset.
    from scipy.spatial.distance import cdist
    real=arrays['train'].reshape(len(arrays['train']),-1); query=x[:32].reshape(min(32,len(x)),-1)
    best=np.full(len(query),np.inf); ids=np.zeros(len(query),int)
    for lo in range(0,len(real),2048):
        dist=cdist(query,real[lo:lo+2048],'sqeuclidean'); idx=dist.argmin(1); value=dist[np.arange(len(query)),idx]; mask=value<best; best[mask]=value[mask];ids[mask]=lo+idx[mask]
    np.savez_compressed(folder/'nearest_neighbors.npz',generated=x[:32],nearest=arrays['train'][ids],train_indices=ids,squared_distances=best,rule='first 32 generated samples; exact standardized vector/pixel Euclidean NN')
    if len(shape)==3:
        fig,axs=plt.subplots(8,8,figsize=(8,8))
        for i in range(32):
            for j,z in enumerate([x[i],arrays['train'][ids[i]]]):
                ax=axs.flat[2*i+j]; im=((z+1)/2).clip(0,1).transpose(1,2,0);ax.imshow(im.squeeze() if shape[0]==1 else im,cmap='gray' if shape[0]==1 else None,vmin=0,vmax=1);ax.axis('off')
        fig.suptitle('Generated / exact nearest training image pairs; first 32 samples')
        fig.tight_layout();fig.savefig(folder/'nearest_neighbors.png',dpi=160);plt.close(fig)


def efficiency(folder,cfg):
    records=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
    val=[json.loads(s) for s in (folder/'validation.jsonl').read_text().splitlines()]
    prof=[r for r in records if r.get('profiled')]
    summary={k:float(np.mean([r.get(k,0) for r in prof])) for k in ['source_seconds','field_seconds','ot_seconds','backward_seconds','update_seconds']}
    summary.update(profiled_updates=len(prof),total_updates=len(records),nonfinite_loss_count=sum(not np.isfinite(r['loss']) for r in records),zero_kernel_fraction_max=max([r.get('zero_kernel_fraction',0) for r in prof]),convergence='not established; fixed budget',timing_note='source/field decomposition applies to DHPF; zero means not applicable for other algorithms; GAN update_seconds includes discriminator')
    write_json(folder/'efficiency.json',summary)
    fig,axs=plt.subplots(1,2,figsize=(9,3))
    for ax,key in zip(axs,['step','wall_seconds']):ax.plot([r[key] for r in val],[r['sw2_squared'] for r in val],marker='o');ax.set(xlabel=key,ylabel='validation SW2 squared')
    fig.tight_layout();fig.savefig(folder/'validation_curves.png',dpi=150);plt.close(fig)


def numerical(folder):
    if (folder/'sampling_study.json').exists(): return
    model,cfg,_=load_model(folder);device=cfg['device'];n=512;c=0 if cfg['classes'] else None
    arrays=load_data(cfg);y=arrays['test'][:n].reshape(n,-1)
    if cfg['classes']: y=arrays['test'][arrays['test_labels']==0][:n].reshape(-1,cfg['d'])
    gen=torch.Generator(device=device).manual_seed(81000);initial=prior(n,cfg['d'],gen,device,cfg['velocity']); rows=[]
    isode=cfg['model'] not in ['gan','ddpm','drifting']
    if isode:
        with torch.no_grad():
            def solve_reference(k):
                if cfg['model']=='dhpf':return reference_dhpf(model,*initial,k,c)
                return generate(model,cfg,n,81000,device,steps=k,condition=c,initial=initial)[0].numpy()
            refs={k:solve_reference(k) for k in [256,512,1024]}
        err512=float(np.sqrt(np.mean((refs[512]-refs[1024])**2)));scale=max(1.,float(np.sqrt(np.mean(refs[1024]**2))))
        refinfo=dict(reference_k=1024,refinement_256_to_512_rms=float(np.sqrt(np.mean((refs[256]-refs[512])**2))),refinement_512_to_1024_rms=err512,relative_threshold=.001,reference_verified=err512/scale<.001)
        reference=refs[1024]
        for finer in [2048,4096,8192]:
            if refinfo['reference_verified']:break
            refined=solve_reference(finer)
            error=float(np.sqrt(np.mean((refined-reference)**2)));scale=max(1.,float(np.sqrt(np.mean(refined**2))))
            refinfo.update(reference_k=finer,last_refinement_rms=error,reference_verified=error/scale<.001)
            reference=refined
    else: refinfo=dict(reference_k=None,reference_verified=None)
    for k in [8,16,32,64,128] if isode else [1000 if cfg['model']=='ddpm' else 1]:
        # Warm-up excludes initialization; measured CPU or synchronized CUDA wall-clock.
        generate(model,cfg,1,81000,device,steps=k,condition=c)
        sync(device); start=time.perf_counter();xx,stats=generate(model,cfg,n,81000,device,steps=k,condition=c,initial=initial,profile=True);sync(device);batchsecs=time.perf_counter()-start
        sync(device);start=time.perf_counter();_,single_stats=generate(model,cfg,1,81000,device,steps=k,condition=c,initial=(initial[0][:1],initial[1][:1]),profile=True);sync(device);singlesecs=time.perf_counter()-start
        row=dict(k=k,nfe=2*k if isode else k,batch_count=n,batch_seconds=batchsecs,single_seconds=singlesecs,sw2_squared=sliced(xx.numpy(),y),source_seconds=stats['source_seconds'],field_seconds=stats['field_seconds'],single_source_seconds=single_stats['source_seconds'],single_field_seconds=single_stats['field_seconds'])
        if isode: row['endpoint_rms_vs_reference']=float(np.sqrt(np.mean((xx.numpy()-reference)**2)))
        rows.append(row)
    write_json(folder/'sampling_study.json',dict(experiment='E09',generation_seed=81000,diagnostic_sample_count=n,not_main_table_sample_count=True,reference=refinfo,rows=rows))
    fig,axs=plt.subplots(1,3,figsize=(12,3))
    axs[0].plot([r['batch_seconds'] for r in rows],[r['sw2_squared'] for r in rows],'-o');axs[0].set(xlabel='batch inference seconds',ylabel='SW2 squared')
    axs[1].plot([r['nfe'] for r in rows],[r['sw2_squared'] for r in rows],'-o');axs[1].set(xlabel='NFE',ylabel='SW2 squared')
    if isode:axs[2].loglog([r['nfe'] for r in rows],[r['endpoint_rms_vs_reference'] for r in rows],'-o')
    axs[2].set(xlabel='NFE',ylabel=f"endpoint RMS vs K={refinfo['reference_k']}")
    fig.tight_layout();fig.savefig(folder/'sampling_study.png',dpi=150);plt.close(fig)


def mechanisms(folder):
    if (folder/'mechanisms.npz').exists(): return
    model,cfg,_=load_model(folder)
    if cfg['model']!='dhpf':return
    device=cfg['device'];gen=torch.Generator(device=device).manual_seed(82000);x,v=prior(64,cfg['d'],gen,device,cfg['velocity']);c=0 if cfg['classes'] else None
    with torch.no_grad():
        _,_,_,trace=integrate(model,'dhpf',x,v,128,c,trace=True)
        fraction=torch.linspace(0,1,9,device=device)[:,None]
        xi=x[:1]*(1-fraction)+x[1:2]*fraction;vi=v[:1]*(1-fraction)+v[1:2]*fraction
        interpolated=integrate(model,'dhpf',xi,vi,128,c)[0].cpu().numpy()
    saved=dict(times=np.linspace(0,1,5),stages=torch.stack(trace).cpu().numpy(),initial_x=x.cpu().numpy(),initial_v=v.cpu().numpy(),interpolation=interpolated,interpolation_fraction=fraction.cpu().numpy())
    if cfg['d']==2:
        grid=np.stack(np.meshgrid(np.linspace(-2.5,2.5,21),np.linspace(-2.5,2.5,21)),-1).reshape(-1,2);query=torch.tensor(grid,dtype=x.dtype,device=device)
        with torch.no_grad():
            p=model.prepare(x.new_tensor([.5]));fields=[model.field(query,p,component=part).cpu().numpy() for part in ['radial','tangential','full']]
        saved.update(grid=grid,radial=fields[0],tangential=fields[1],total=fields[2])
        fig,axs=plt.subplots(1,3,figsize=(12,4))
        for ax,field,title in zip(axs,fields,['Radial acceleration','Tangential acceleration','Total acceleration']):ax.quiver(grid[:,0],grid[:,1],field[:,0],field[:,1]);ax.set(title=title,aspect='equal')
        fig.tight_layout();fig.savefig(folder/'field_components.png',dpi=150);plt.close(fig)
        fig,axs=plt.subplots(1,5,figsize=(15,3))
        for ax,z,t in zip(axs,saved['stages'],saved['times']):ax.scatter(z[:,0],z[:,1],s=10);ax.set(title=f't={t:g}',xlim=(-3,3),ylim=(-3,3),aspect='equal')
        fig.tight_layout();fig.savefig(folder/'progressive.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(4,4));ax.plot(interpolated[:,0],interpolated[:,1],'-o');ax.set(title='Endpoints from phase-space interpolation',aspect='equal');fig.tight_layout();fig.savefig(folder/'interpolation.png',dpi=150);plt.close(fig)
    elif len(cfg['shape'])==2:
        fig,axs=plt.subplots(1,5,figsize=(15,3))
        for ax,z,t in zip(axs,saved['stages'],saved['times']):ax.plot(z[0].reshape(cfg['shape']));ax.set_title(f'generation t={t:g}')
        fig.tight_layout();fig.savefig(folder/'progressive.png',dpi=150);plt.close(fig)
        fig,axs=plt.subplots(3,3,figsize=(10,7))
        for ax,z,f in zip(axs.flat,interpolated,fraction.cpu().numpy().ravel()):ax.plot(z.reshape(cfg['shape']));ax.set_title(f'phase interpolation {f:.3f}')
        fig.tight_layout();fig.savefig(folder/'interpolation.png',dpi=150);plt.close(fig)
    else:
        fig,axs=plt.subplots(4,5,figsize=(8,7))
        for i in range(4):
            for j in range(5):
                z=saved['stages'][j,i].reshape(cfg['shape']);im=((z+1)/2).clip(0,1).transpose(1,2,0);axs[i,j].imshow(im.squeeze() if cfg['shape'][0]==1 else im,cmap='gray' if cfg['shape'][0]==1 else None,vmin=0,vmax=1);axs[i,j].axis('off')
        fig.tight_layout();fig.savefig(folder/'progressive.png',dpi=150);plt.close(fig)
        fig,axs=plt.subplots(1,9,figsize=(14,2))
        for ax,z,f in zip(axs,interpolated,fraction.cpu().numpy().ravel()):
            im=((z.reshape(cfg['shape'])+1)/2).clip(0,1).transpose(1,2,0);ax.imshow(im.squeeze() if cfg['shape'][0]==1 else im,cmap='gray' if cfg['shape'][0]==1 else None,vmin=0,vmax=1);ax.axis('off');ax.set_title(f'{f:.3f}')
        fig.tight_layout();fig.savefig(folder/'interpolation.png',dpi=150);plt.close(fig)
    np.savez_compressed(folder/'mechanisms.npz',**saved)


def budget_final(folder,cfg,arrays):
    """Same formal sample count and seeds for the budget-final checkpoint."""
    if (folder/'budget_final_metrics.json').exists():return
    selected=json.loads((folder/'metrics.json').read_text())
    if selected['checkpoint_step']==cfg['updates']:
        write_json(folder/'budget_final_metrics.json',dict(same_as_selected=True,checkpoint_step=cfg['updates'],metrics=selected));return
    model,_,_=load_model(folder,'latest.pt');out={};samples=[];labels=[]
    for c in range(cfg['classes']) if cfg['classes'] else [None]:
        y=arrays['test'] if c is None else arrays['test'][arrays['test_labels']==c]
        n=cfg['eval_count'] if c is None else cfg['eval_per_condition']
        x,_=generate(model,cfg,n,80000,cfg['device'],condition=c)
        xx=x.numpy().reshape(n,*cfg['shape']);metrics=distribution(xx,y,cfg,exact=cfg['d']==2)
        if len(cfg['shape'])==2:
            from scipy.spatial.distance import pdist
            tr=arrays['train'].reshape(len(arrays['train']),-1);bandwidth=float(np.sqrt(np.median(pdist(tr[:512],'sqeuclidean'))))
            metrics.update(sequence_metrics(xx,y,bandwidth)[0])
        if c is None:out=metrics
        else:out[str(c)]=metrics;labels.extend([c]*n)
        samples.append(xx)
    # Keep final-checkpoint raw samples for reproducibility; no quantization of vector outputs.
    finaldir=folder/'budget_final';finaldir.mkdir(exist_ok=True)
    write_json(finaldir/'config.json',cfg)
    np.savez_compressed(finaldir/'samples.npz',generated=np.concatenate(samples),labels=np.asarray(labels),generation_seed=80000)
    if cfg.get('image_shape'):
        from .features import evaluate_images
        evaluate_images(finaldir);out['image_metrics']=json.loads((finaldir/'image_metrics.json').read_text())
    out.update(checkpoint_step=cfg['updates'],generation_seed=80000,convergence='unestablished at fixed budget')
    write_json(folder/'budget_final_metrics.json',out)


def process(folder,full=False):
    folder=Path(folder);cfg=json.loads((folder/'config.json').read_text());arrays=load_data(cfg)
    with np.load(folder/'samples.npz') as z:x=z['generated'];labels=z['labels']
    plots(folder,cfg,arrays,x);efficiency(folder,cfg)
    if len(cfg['shape'])==2:downstream(folder,cfg,arrays,x,labels)
    if full:numerical(folder);mechanisms(folder)
    budget_final(folder,cfg,arrays)
    if cfg['model']=='dhpf' and not (folder/'source_layout.npz').exists():
        model,_,_=load_model(folder)
        with torch.no_grad():
            times=torch.linspace(0,1,17 if cfg['d']==2 else 5,device=cfg['device'])
            prepared=model.prepare(times,0 if cfg['classes'] else None)
            p,a,s,uv=[z.cpu().numpy() for z in prepared]
            from scipy.spatial.distance import pdist
            distances=np.concatenate([pdist(pp) for pp in p])
        np.savez_compressed(folder/'source_layout.npz',times=times.cpu().numpy(),centers=p,alpha=a,sigma=s,skew_parameters=uv,condition=0 if cfg['classes'] else -1)
        write_json(folder/'source_layout.json',dict(pair_distance_mean=float(distances.mean()),pair_distance_median=float(np.median(distances)),width_mean=float(s.mean()),center_norm_mean=float(np.linalg.norm(p,axis=-1).mean()),alpha_abs_mean=float(abs(a).mean()),rank=cfg['rank']))
        if cfg['d']==2:
            fig,ax=plt.subplots(figsize=(5,5))
            for i in range(cfg['n']):ax.plot(p[:,i,0],p[:,i,1],alpha=.6)
            ax.scatter(p[0,:,0],p[0,:,1],marker='o',label='t=0');ax.scatter(p[-1,:,0],p[-1,:,1],marker='x',label='t=1');ax.set(title='Shared-source positions over generation time',aspect='equal');ax.legend();fig.tight_layout();fig.savefig(folder/'source_layout.png',dpi=150);plt.close(fig)
    write_json(folder/'postprocess_complete.json',dict(plots=True,efficiency=True,downstream=len(cfg['shape'])==2,sampling_study=full))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder');p.add_argument('--all',action='store_true');p.add_argument('--full',action='store_true');a=p.parse_args();torch.set_num_threads(2);threadpool_limits(2)
    for folder in sorted((RESULTS/'runs').glob('*')) if a.all else [Path(a.folder)]:
        if (folder/'metrics.json').exists():process(folder,a.full or json.loads((folder/'config.json').read_text())['seed']==0)
