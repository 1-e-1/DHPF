"""Inference diagnostic figures and fixed-basis trajectory animation.

Figure contract: distinguish source synchronization present at initialization
from learned statistical organization of transported samples. Quantitative
grids use mean ± sample SD over five checkpoints; HAR means class 0 only.
No journal preset is assumed. All observations remain in CSV/NPZ files.
"""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.lines import Line2D
from threadpoolctl import threadpool_limits
from research_ts.inference_emergence import OUT,write_json

sys.path.insert(0,__import__('os').environ.get('NATURE_FIGURE_SCRIPTS', str(Path(__file__).resolve().parents[1] / 'optional_tools/nature-figure/scripts')))
from audit_panel_alignment import require_matplotlib_panel_alignment
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
COLORS=dict(full='#305D7B',untrained='#8C969D',ballistic='#75987D',frozen_t0='#BA9167',radial_only='#A58CAB',solenoidal_only='#78A9BD',null='#B0A999')
LABELS=dict(full='Trained DHPF',untrained='Untrained',ballistic='Ballistic',frozen_t0='Frozen sources',radial_only='Radial only',solenoidal_only='Solenoidal only',null='Phase-shuffled')
DATASETS=['sine','lorenz','har'];TITLES=['Sine','Lorenz','HAR: walking']

def style():
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})

def panel(ax,label):ax.text(-.22,1.06,label,transform=ax.transAxes,fontweight='bold',fontsize=10)
def save(fig,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(FIG/f'{name}.alignment.json'),
        overlay_svg=str(FIG/f'{name}.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(FIG/f'{name}.pdf')
    fig.savefig(FIG/f'{name}.svg')
    fig.savefig(FIG/f'{name}.png',dpi=300)
    plt.close(fig)

def trend(ax,df,key,variants):
    for variant in variants:
        g=df[df.variant.eq(variant)].groupby('t')[key].agg(['mean','std'])
        ax.plot(g.index,g['mean'],color=COLORS[variant],lw=1.5 if variant=='full' else 1,
                ls='-' if variant=='full' else '--',label=LABELS[variant])
        ax.fill_between(g.index,g['mean']-g['std'],g['mean']+g['std'],color=COLORS[variant],alpha=.14,lw=0)
    ax.set(xlabel='Generation time',xlim=(0,1),xticks=[0,.5,1])

def quantitative():
    x=pd.read_csv(OUT/'sample_dynamics.csv');x=x[x.condition.isna()|x.condition.eq(0)]
    s=pd.read_csv(OUT/'source_dynamics.csv');s=s[s.condition.isna()|s.condition.eq(0)]
    summary=json.loads((OUT/'summary.json').read_text())
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.8));fig.subplots_adjust(left=.105,right=.98,bottom=.11,top=.83,wspace=.5,hspace=.55)
    for j,(dataset,title) in enumerate(zip(DATASETS,TITLES)):
        a=axs[0,j];trend(a,s[s.dataset.eq(dataset)],'motion_alignment',['full','untrained'])
        val=summary[dataset]['sources']['full']['phase_shift_null_mean']['mean']
        a.axhline(val,color=COLORS['null'],ls=':',lw=1);a.set(title=title,ylabel='Source direction alignment',ylim=(0,1.05))
        b=axs[1,j];trend(b,x[x.dataset.eq(dataset)],'effective_dimension',['full','untrained','ballistic','frozen_t0'])
        b.set(ylabel='Covariance participation ratio',ylim=(0,None))
        panel(a,chr(97+j));panel(b,chr(100+j))
    variants=['full','untrained','ballistic','frozen_t0','null']
    fig.legend([Line2D([0],[0],color=COLORS[v],ls=':' if v=='null' else '-' if v=='full' else '--') for v in variants],
               [LABELS[v] for v in variants],loc='upper center',ncol=3,bbox_to_anchor=(.53,.99))
    save(fig,'01_sources_and_structure')
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.9));fig.subplots_adjust(left=.105,right=.98,bottom=.16,top=.86,wspace=.5,hspace=.65)
    angular=pd.read_csv(OUT/'circulation.csv');variants=['full','untrained','ballistic','frozen_t0','radial_only','solenoidal_only']
    for j,(dataset,title) in enumerate(zip(DATASETS,TITLES)):
        a=axs[0,j];trend(a,x[x.dataset.eq(dataset)],'variance_ratio',['full','ballistic','radial_only','solenoidal_only'])
        a.set(title=title,ylabel='Spread / initial spread');panel(a,chr(97+j))
        a=axs[1,j]
        for i,v in enumerate(variants):
            vals=angular[(angular.dataset==dataset)&(angular.variant==v)&angular.t.eq(1)].circulation_coherence
            a.errorbar(i,vals.mean(),yerr=vals.std(ddof=1),fmt='o',color=COLORS[v],ms=4,capsize=2)
        a.set(xticks=range(6),xticklabels=['Full','Init.','Free','Frozen','Radial','Sol.'],ylabel='Circulation coherence',ylim=(0,1.05))
        a.tick_params(axis='x',labelrotation=45)
        n=summary[dataset]['circulation_shuffle_null_q95']['mean']
        a.axhline(n,color='#B0A999',ls=':',lw=1)
        panel(a,chr(100+j))
    variants=['full','ballistic','radial_only','solenoidal_only']
    fig.legend([Line2D([0],[0],color=COLORS[v],ls='-' if v=='full' else '--') for v in variants],
        [LABELS[v] for v in variants],loc='upper center',ncol=4,bbox_to_anchor=(.53,.99))
    save(fig,'02_contraction_and_circulation')

@torch.no_grad()
def pca2(array):
    shape=array.shape;a=torch.as_tensor(array.reshape(-1,shape[-1]),device='cuda')
    a=a-a.mean(0);torch.manual_seed(91900);u,s,v=torch.pca_lowrank(a,q=4,center=False,niter=5)
    p=a@v[:,:2];fraction=float(p.square().sum()/a.square().sum())
    return p.cpu().numpy().reshape(*shape[:-1],2),fraction

def animation():
    sequences=[];explained=[]
    for dataset in DATASETS:
        c=0 if dataset=='har' else None
        with np.load(OUT/'traces'/f'{dataset}__seed0__c{c}__full.npz') as z:
            source,sv=pca2(z['source_positions']);samples,xv=pca2(z['x']);alpha=z['source_alpha'][::4]
        sequences.append((source[::4],samples,alpha));explained.append(dict(dataset=dataset,source_pca_fraction=sv,sample_pca_fraction=xv))
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.9));fig.subplots_adjust(left=.08,right=.98,bottom=.13,top=.85,wspace=.4,hspace=.5)
    scatters=[];trails=[]
    for j,(source,samples,alpha) in enumerate(sequences):
        for row,arr in enumerate([source,samples]):
            ax=axs[row,j];low=arr.reshape(-1,2).min(0);high=arr.reshape(-1,2).max(0);pad=(high-low)*.08
            ax.set(xlim=(low[0]-pad[0],high[0]+pad[0]),ylim=(low[1]-pad[1],high[1]+pad[1]),xlabel='Fixed PC 1',ylabel='Fixed PC 2')
            ax.set_title(TITLES[j]+(' | sources' if row==0 else ' | samples'))
            scatter=ax.scatter(arr[0,:,0],arr[0,:,1],s=12 if row==0 else 5,color='#305D7B',alpha=.8 if row==0 else .35,edgecolors='none')
            lines=[ax.plot([],[],color='#9EA8AF',lw=.4,alpha=.5)[0] for _ in range(min(16,arr.shape[1]))]
            scatters.append((scatter,arr,row,alpha));trails.append(lines);panel(ax,chr(97+row*3+j))
    title=fig.suptitle('Complete inference | seed 0 | t = 0.000',y=.98,fontsize=11)
    fig.text(.5,.015,'Fixed PCA views; metrics use all dimensions. Sources: blue attractive, orange repulsive radial coefficient.',ha='center',fontsize=7)
    def update(k):
        title.set_text(f'Complete inference | seed 0 | t = {k/32:.3f}')
        for (sc,arr,row,alpha),lines in zip(scatters,trails):
            sc.set_offsets(arr[k])
            if row==0:sc.set_color(np.where((alpha[k]<0)[:,None],np.array([[.188,.365,.482]]),np.array([[.73,.49,.28]])))
            for i,line in enumerate(lines):line.set_data(arr[:k+1,i,0],arr[:k+1,i,1])
        return []
    update(32);fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(FIG/'03_inference_snapshot.alignment.json'),
        overlay_svg=str(FIG/'03_inference_snapshot.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'03_inference_snapshot.{ext}',dpi=300)
    ani=FuncAnimation(fig,update,frames=33,interval=250,blit=False)
    ani.save(FIG/'complete_inference.gif',writer=PillowWriter(fps=4),dpi=140)
    update(16);fig.savefig(FIG/'midpoint_preview.png',dpi=180)
    plt.close(fig);write_json(OUT/'projection_metadata.json',dict(projections=explained,
        rule='PCA fitted once to all recorded states of each object type and held fixed across every frame. All 32 sources and 256 samples shown; first 16 indexed trails shown for readability.',
        caveat='Two-dimensional projections neither prove a vortex nor estimate intrinsic dimension. Seed 0 fixed before viewing.'))

def main():
    torch.set_num_threads(2);threadpool_limits(2);style();quantitative();animation();print('Figures and 33-frame animation exported')
if __name__=='__main__':main()
