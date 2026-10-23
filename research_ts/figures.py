"""Manuscript figures: all five seeds; no selected best-seed illustrations.

Contract: describe the observed trade-off between dependency statistics and
distribution fit, the HAR utility boundary, and component/numerical evidence.
Quantitative grids use five training seeds (mean +/- sample SD); numerical
refinement and fixed-first-sample panels are explicitly single-seed diagnostics.
Python/Matplotlib; editable SVG/PDF text; 180 mm width; no journal-specific claim.
"""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/timeseries_v2';FIG=ROOT/'paper/figures'
sys.path.insert(0,__import__('os').environ.get('NATURE_FIGURE_SCRIPTS', str(Path(__file__).resolve().parents[1] / 'optional_tools/nature-figure/scripts')))
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.2})
MODELS=['gan','ddpm_v','icfm','otcfm','dhpf','drifting']
LABELS={'gan':'GAN','ddpm_v':'DDPM-v','icfm':'I-CFM','otcfm':'OT-CFM','dhpf':'DHPF','drifting':'Drifting-raw'}
COLORS={'gan':'#999999','ddpm_v':'#91B7CD','icfm':'#9D98BC','otcfm':'#C6A283','dhpf':'#305D7B','drifting':'#A9B8A0'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(fig,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(FIG/(name+'.alignment.json')),overlay_svg=str(FIG/(name+'.alignment.svg')),strict=True,tolerance_pt=1.5,gutter_tolerance_pt=1.5)
    fig.savefig(FIG/(name+'.svg'))
    fig.savefig(FIG/(name+'.pdf'))
    fig.savefig(FIG/(name+'.png'),dpi=600)
    plt.close(fig)
def panel(ax,label):ax.text(-.16,1.06,label,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')

def main():
    FIG.mkdir(parents=True,exist_ok=True);summary=read(OUT/'summary.json');s={(r['dataset'],r['model'],r['variant']):r for r in summary}
    # Complementary inference: dependency gain versus global distribution cost.
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.35));fig.subplots_adjust(left=.10,right=.98,bottom=.29,top=.86,wspace=.38)
    for ax,d,metric,title,letter in zip(axs,['sine','lorenz'],['autocorrelation_rmse','cross_correlation_rmse'],['Sine: temporal correlation','Lorenz: channel correlation'],['a','b']):
        for m in MODELS:
            r=s[d,m,'base'];ax.errorbar(r['sw2_squared_mean'],r[metric+'_mean'],xerr=r['sw2_squared_std'],yerr=r[metric+'_std'],fmt='o',ms=5 if m=='dhpf' else 4,c=COLORS[m],capsize=2,zorder=4 if m=='dhpf' else 2)
        ax.set(xlabel='Sliced Wasserstein squared (lower is better)',ylabel='ACF RMSE' if d=='sine' else 'Zero-lag Pearson RMSE',title=title);panel(ax,letter)
    fig.legend(handles=[Line2D([],[],marker='o',linestyle='',color=COLORS[m],label=LABELS[m]) for m in MODELS],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.015))
    save(fig,'fig1_tradeoff')
    # Existing component controls, with both fidelity and dependency evidence.
    variants=[('dhpf','base','Full'),('dhpf','fixed_full','Fixed'),('dhpf','radial','Radial'),('dhpf','tangential','Tangent'),('direct','base','MLP2'),('augmented','base','AugODE')]
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.15));fig.subplots_adjust(left=.10,right=.98,bottom=.27,top=.84,wspace=.38)
    for ax,key,title,letter in zip(axs,['autocorrelation_rmse','sw2_squared'],['Temporal dependency','Distribution fidelity'],['a','b']):
        for i,(m,v,label) in enumerate(variants):
            r=s['sine',m,v];ax.errorbar(i,r[key+'_mean'],yerr=r[key+'_std'],fmt='o',capsize=3,c='#305D7B' if i==0 else '#9EABB4')
        ax.set(xticks=range(len(variants)),xticklabels=[r[2] for r in variants],ylabel='ACF RMSE' if key.startswith('auto') else 'Sliced Wasserstein squared',title=title)
        plt.setp(ax.get_xticklabels(),rotation=0,ha='center');panel(ax,letter)
    save(fig,'fig2_components')
    # Single checkpoint, fixed initial states: discretization diagnostics only.
    fig,ax=plt.subplots(figsize=(7.1,2.65));fig.subplots_adjust(left=.12,right=.97,bottom=.23,top=.85)
    for d,color in [('sine','#305D7B'),('lorenz','#C6A283'),('har','#9D98BC')]:
        info=read(ROOT/'results/checklist_v1/runs'/f'{d}__dhpf__base__seed0/sampling_study.json')
        assert info['reference']['reference_verified']
        rows=info['rows'];ax.loglog([r['k'] for r in rows],[r['endpoint_rms_vs_reference'] for r in rows],'-o',color=color,label='HAR' if d=='har' else d.title(),ms=4)
    ax.set(xlabel='Midpoint steps K (NFE = 2K)',ylabel='Endpoint RMS vs refined reference',xticks=[8,16,32,64,128]);ax.set_xticklabels(['8','16','32','64','128']);ax.legend(ncol=3,frameon=False,loc='upper right');save(fig,'fig3_numerics')
    # Fixed first sample in the archived order: no visual example selection.
    fig,axs=plt.subplots(2,3,figsize=(7.1,3.6));fig.subplots_adjust(left=.09,right=.98,bottom=.15,top=.88,wspace=.30,hspace=.55)
    for row,d in enumerate(['sine','lorenz']):
        with np.load(ROOT/'data/checklist_v1'/f'{d}.npz') as z:real=z['test'][0,:,0]
        series=[real]
        for m in ['dhpf','gan']:
            with np.load(ROOT/'results/checklist_v1/runs'/f'{d}__{m}__base__seed0/samples.npz') as z:series.append(z['generated'][0,:,0])
        limits=(min(z.min() for z in series)-.2,max(z.max() for z in series)+.2)
        for col,(z,label) in enumerate(zip(series,['Real','DHPF','GAN'])):
            ax=axs[row,col];ax.plot(range(128),z,c='#305D7B' if label=='DHPF' else '#8D969D');ax.set(ylim=limits,title=d.title()+' / '+label,xlabel='Sequence index',ylabel='Standardized value');panel(ax,chr(97+row*3+col))
    save(fig,'fig4_examples')
    # Draw HAR only once all five seeds and both external probes exist.
    if all(('har',m,'base') in s and s['har',m,'base']['n']==5 for m in MODELS):
        fig,axs=plt.subplots(1,2,figsize=(7.1,3.3));fig.subplots_adjust(left=.09,right=.98,bottom=.27,top=.84,wspace=.38)
        real_ridge=read(OUT/'ridge_probe/real_matched.json')['test_macro_f1']
        for ax,probe,letter in zip(axs,['Logistic regression','Ridge classifier'],['a','b']):
            for i,m in enumerate(MODELS):
                if probe.startswith('Logistic'):r=s['har',m,'base'];mean=r['synthetic_test_macro_f1_mean'];sd=r['synthetic_test_macro_f1_std']
                else:
                    name='ddpm' if m=='ddpm_v' else m
                    values=[read(OUT/'ridge_probe'/f'har__{name}__base__seed{k}.json')['test_macro_f1'] for k in range(5)]
                    mean=float(np.mean(values));sd=float(np.std(values,ddof=1))
                ax.errorbar(i,mean,yerr=sd,fmt='o',capsize=3,color=COLORS[m])
            ref=s['har','dhpf','base']['real_matched_test_macro_f1_mean'] if probe.startswith('Logistic') else real_ridge
            ax.axhline(ref,color='#777777',ls='--',lw=.9)
            ax.set(xticks=range(6),xticklabels=[LABELS[m] if m!='drifting' else 'Drift' for m in MODELS],ylabel='Test macro-F1 (higher is better)',title=probe,ylim=(0,1))
            plt.setp(ax.get_xticklabels(),rotation=0,ha='center');panel(ax,letter)
        save(fig,'fig5_har_utility')
    write_json= lambda p,o:p.write_text(json.dumps(o,indent=2),encoding='utf-8')
    write_json(FIG/'contract.json',dict(backend='Python/Matplotlib',width_mm=180.34,training_seeds=5,error_bars='sample SD across training seeds',numerical_and_example_seed=0,example_selection='first archived generated sample and first held-out real sample; channel 0; unpaired',sources=['results/timeseries_v2/summary.json','results/checklist_v1/runs/*/sampling_study.json','results/timeseries_v2/ridge_probe'],claim='Local dependency gains coexist with distribution/utility costs; not a universal superiority claim.'))

if __name__=='__main__':main()
