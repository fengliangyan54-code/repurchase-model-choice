"""Publication figures from saved, audited numerical files; no hand-entered effects."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
BASE=ROOT/'results/frozen_v1_iofix';EXT=ROOT/'results/extension_v2_iofix'
OUT=Path(os.environ.get('RESEARCH_OUTPUT_DIR',str(ROOT/'deliverables')))/'figures';OUT.mkdir(parents=True,exist_ok=True)
def read(p):return pd.read_csv(p,float_precision='round_trip')
main=read(BASE/'macro_metrics.csv').set_index('model');monthly=read(BASE/'monthly_metrics.csv');pred=read(BASE/'test_predictions.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'svg.fonttype':'none'})
colors={'RF':'#235779','BGNBD':'#A66A19','HGB':'#527548','LR_extended':'#6B5B83','RF_RFM':'#235779','RF_common':'#527548','HGB_common':'#6B5B83'}
markers=['o','s','^','D']
def save(fig,name):
    fig.tight_layout(pad=.8)
    for suffix in ['png','svg']:fig.savefig(OUT/(name+'.'+suffix),bbox_inches='tight')
    plt.close(fig)

names=['Recency','Frequency','Monetary','RFM_rank','BGNBD','LR_RFM','LR_extended','RF','HGB','HGB_RFM']
labels=[n.replace('_',' ').replace('BGNBD','BG/NBD') for n in names]
fig,axes=plt.subplots(1,2,figsize=(9,4.6),sharey=True)
for ax,col,title in zip(axes,['AP','P20'],['(a) Equal month AP','(b) Precision at 20% capacity']):
    ax.barh(labels,main.loc[names,col],color=[colors.get(n,'#AAB4BB') for n in names],edgecolor='#333333',linewidth=.4)
    ax.set_xlim(0,1);ax.set_xlabel(title);ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
axes[0].invert_yaxis();axes[1].axvline(monthly.groupby('cutoff').prevalence.first().mean(),color='#333333',ls='--',lw=1,label='Random selection');axes[1].legend(fontsize=9,loc='lower right')
save(fig,'fig1_model_comparison')

fig,axes=plt.subplots(1,2,figsize=(9,3.5))
for n,mk in zip(['RF','BGNBD','HGB','LR_extended'],markers):
    g=monthly[monthly.model==n];axes[0].plot(np.arange(4),g.AP,marker=mk,color=colors[n],label=n.replace('_',' ').replace('BGNBD','BG/NBD'))
    axes[1].plot([10,20,30],[main.loc[n,'P'+str(k)] for k in [10,20,30]],marker=mk,color=colors[n])
axes[0].set_xticks(range(4),['Aug','Sep','Oct','Nov']);axes[0].set_xlabel('(a) UCI cutoff in 2011');axes[0].set_ylabel('Average precision');axes[0].set_ylim(.45,.8);axes[0].legend(fontsize=8)
axes[1].set_xticks([10,20,30]);axes[1].set_xlabel('(b) Monthly capacity (%)');axes[1].set_ylabel('Precision');axes[1].set_ylim(.5,.9)
for ax in axes:ax.grid(alpha=.15)
save(fig,'fig2_time_and_capacity')

fig,axes=plt.subplots(1,2,figsize=(9.4,4),gridspec_kw={'width_ratios':[1,1.2]})
for n,mk in zip(['RF','BGNBD','HGB','LR_extended'],markers):
    y,p=calibration_curve(pred.label,pred[n],n_bins=8,strategy='quantile');axes[0].plot(p,y,marker=mk,color=colors[n],label=n.replace('_',' ').replace('BGNBD','BG/NBD'))
axes[0].plot([0,1],[0,1],'--',color='#333333',lw=1);axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='(a) Mean predicted probability',ylabel='Observed purchase proportion');axes[0].legend(fontsize=8)
imp=read(BASE/'permutation_importance.csv').sort_values('AP_drop_mean')
pretty={'recency90':'Recency 90d','frequency90':'Orders 90d','monetary90':'Spend 90d','tenure':'Observed tenure','frequency30':'Orders 30d','monetary30':'Spend 30d','products90':'Products 90d','avg_order90':'Mean order value','items_per_order90':'Products per order','mean_gap90':'Mean purchase gap'}
axes[1].barh([pretty[f] for f in imp.feature],imp.AP_drop_mean,xerr=imp.AP_drop_sd,color=colors['RF'],error_kw={'elinewidth':1,'capsize':2});axes[1].set_xlabel('(b) AP decrease after permutation');axes[1].axvline(0,color='#333333',lw=.8)
save(fig,'fig3_calibration_and_reliance')

fig,axes=plt.subplots(1,2,figsize=(9.4,3.2),gridspec_kw={'width_ratios':[1.25,1]})
for a,b,y,col in [(0,4,2,'#235779'),(4,1,1,'#A66A19'),(5,4,0,'#527548')]:axes[0].broken_barh([(a,b)],(y,.6),facecolors=col)
axes[0].set_yticks([2.3,1.3,.3],['Train','Validate','Test']);axes[0].set_xticks(np.arange(9)+.5,['Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']);axes[0].tick_params(axis='x',labelsize=9);axes[0].set_xlim(0,9);axes[0].set_xlabel('(a) UCI decision cutoffs in 2011')
c=json.loads((BASE/'primary_comparison.json').read_text())
for j,key in enumerate(['AP','P20']):
    est=c[key]['difference'];lo,hi=c[key]['CI95'];axes[1].errorbar(est,j,xerr=[[est-lo],[hi-est]],fmt='o',color='#235779',capsize=4)
axes[1].axvline(0,color='#333333',ls='--',lw=1);axes[1].set_yticks([0,1],['AP','Precision 20%']);axes[1].set_ylim(-.7,1.7);axes[1].set_xlabel('(b) RF minus BG/NBD');axes[1].grid(axis='x',alpha=.15)
save(fig,'fig4_protocol_uncertainty')

ex=read(EXT/'cdnow_macro_metrics.csv').set_index('model');ded=read(EXT/'cdnow_deduplicated_macro_metrics.csv').set_index('model');mo=read(EXT/'cdnow_monthly_metrics.csv')
fig,axes=plt.subplots(1,2,figsize=(9.4,4.4),gridspec_kw={'width_ratios':[1.15,1]})
for n,mk in zip(['RF_RFM','RF_common','HGB_common','BGNBD'],markers):
    g=mo[mo.model==n];axes[0].plot(np.arange(10),g.AP,color=colors[n],marker=mk,label=n.replace('_',' ').replace('BGNBD','BG/NBD'),markersize=4)
axes[0].set_xticks([0,3,6,9],['Sep 97','Dec 97','Mar 98','Jun 98']);axes[0].set_xlabel('(a) CDNOW cutoff');axes[0].set_ylabel('Average precision');axes[0].set_ylim(.3,.6);axes[0].legend(fontsize=8);axes[0].grid(alpha=.15)
ns=ex.index.tolist();y=np.arange(len(ns));axes[1].barh(y-.17,ex.AP,height=.32,color='#235779',label='Retained');axes[1].barh(y+.17,ded.loc[ns,'AP'],height=.32,color='#FFFFFF',edgecolor='#A66A19',label='Deduplicated')
axes[1].set_yticks(y,[n.replace('_',' ').replace('BGNBD','BG/NBD') for n in ns],fontsize=9);axes[1].invert_yaxis();axes[1].set_xlim(0,.6);axes[1].set_xlabel('(b) Equal month AP');axes[1].legend(fontsize=8,loc='lower right')
save(fig,'figS1_cdnow')
print(OUT)
