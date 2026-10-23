"""Editable computational overview; no synthetic quantitative evidence."""
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/figures'

def main():
    plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(7.1,3.05));fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,label,fill='#F1F5F8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',
                     facecolor=fill,edgecolor='#526875',linewidth=.8))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',linespacing=1.4)
    def arrow(a,b,color='#526875',dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=.9,color=color,
                     linestyle='--' if dashed else '-'))
    box(.035,.72,.235,.21,'Source network\n'+r'$G_\theta(t,c)$')
    box(.365,.72,.265,.21,'Shared source parameters\n'+r'$P_i,\ \alpha_i,\ \sigma_i,\ U_i,\ V_i$')
    box(.74,.72,.225,.21,'Observed windows\n'+r'$y_1,\ldots,y_B$',fill='#F7F3EE')
    box(.035,.35,.235,.21,'Random initial states\n'+r'$z=(x_0,v_0)$')
    box(.365,.35,.265,.21,'Second-order integration\n'+r'$\dot{x}=v,\quad \dot{v}=a_\theta(x,t;c)$')
    box(.74,.35,.225,.21,'Generated windows\n'+r'$\hat{x}_i=T_\theta(z_i;c)$')
    box(.365,.035,.60,.16,'Within-batch assignment and squared endpoint loss',fill='#F7F3EE')
    arrow((.28,.825),(.353,.825));arrow((.497,.71),(.497,.57))
    ax.text(.505,.64,'analytic field queries',va='center',fontsize=7.5)
    arrow((.28,.455),(.353,.455));arrow((.64,.455),(.728,.455))
    arrow((.85,.34),(.85,.207),color='#92785A')
    ax.text(.675,.25,'training only',color='#92785A',ha='center',fontsize=7.5)
    # The real batch reaches the loss without becoming a source-network input.
    ax.plot([.975,.99,.99,.94],[.825,.825,.12,.12],color='#92785A',lw=.9)
    arrow((.975,.825),(.99,.825),color='#92785A')
    arrow((.99,.12),(.97,.12),color='#92785A')
    # Gradients through the solver and source network.
    ax.plot([.353,.315,.315],[.115,.115,.825],ls='--',color='#92785A',lw=.9)
    arrow((.315,.825),(.28,.825),color='#92785A',dashed=True)
    arrow((.315,.455),(.353,.455),color='#92785A',dashed=True)
    ax.text(.165,.16,'backpropagate through\nselected pairs and solver',color='#92785A',ha='center',fontsize=7.5)
    for suffix in ['svg','pdf','png']:
        fig.savefig(OUT/f'fig0_method.{suffix}',dpi=600)
    svg=OUT/'fig0_method.svg'
    vector=re.sub(r'<!DOCTYPE[^>]*>', '',svg.read_text(encoding='utf-8'))
    # Matplotlib's viewBox uses points; unitless font sizes preserve those
    # user-space values and let the schematic auditor apply canvas scaling.
    vector=re.sub(r'(font-size:\s*[\d.]+)px',r'\1',vector)
    svg.write_text(vector,encoding='utf-8')
    plt.close(fig)

if __name__=='__main__':main()
