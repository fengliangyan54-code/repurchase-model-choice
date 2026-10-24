"""Reproduce the frozen transaction-only, out-of-time benchmark. No treatment effects."""
from pathlib import Path
import sys, os
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'deps').exists(): sys.path.insert(0,str(ROOT/'deps'))
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key]='1'
import json, hashlib, time, platform, importlib.metadata, warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from lifetimes import BetaGeoFitter
from threadpoolctl import threadpool_limits
import joblib
warnings.filterwarnings('ignore',category=FutureWarning)
RES=ROOT/'results'; RES.mkdir(exist_ok=True)
def dump(name,obj): (RES/name).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
def log(msg): print(msg,flush=True)
RFM=['recency90','frequency90','monetary90']
EXT=RFM+['tenure','frequency30','monetary30','products90','avg_order90','items_per_order90','mean_gap90']
DATES=pd.date_range('2011-03-01','2011-11-01',freq='MS')

def prepare():
    cache=ROOT/'data'/('positive_'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]+'.pkl')
    audit_cache=cache.with_suffix('.audit.json')
    if cache.exists() and audit_cache.exists():
        info=json.loads(audit_cache.read_text(encoding='utf-8'))
        dump('data_audit.json',info)
        pd.DataFrame(info['flow']).to_csv(RES/'data_flow.csv',index=False)
        return pd.read_pickle(cache)
    raw=pd.read_excel(ROOT/'data'/'Online Retail.xlsx',engine='openpyxl')
    info={'raw_rows':len(raw),'exact_duplicate_rows':int(raw.duplicated().sum()),'missing_customer_rows':int(raw.CustomerID.isna().sum()),'raw_min_date':str(raw.InvoiceDate.min()),'raw_max_date':str(raw.InvoiceDate.max())}
    clean=raw.drop_duplicates().copy()
    flow=[{'step':'raw','remaining':len(raw)},{'step':'exact duplicates removed','remaining':len(clean)}]
    for name, maskfn in [('known customer',lambda x:x.CustomerID.notna()),('valid timestamp',lambda x:x.InvoiceDate.notna()),('not cancelled',lambda x:~x.InvoiceNo.astype(str).str.upper().str.startswith('C')),('positive quantity',lambda x:x.Quantity>0),('positive price',lambda x:x.UnitPrice>0)]:
        before=len(clean);clean=clean.loc[maskfn(clean)].copy();flow.append({'step':name,'removed':before-len(clean),'remaining':len(clean)})
    clean['customer']=clean.CustomerID.astype(int)
    clean['invoice']=clean.InvoiceNo.astype(str)
    clean['amount']=clean.Quantity*clean.UnitPrice
    clean['date']=pd.to_datetime(clean.InvoiceDate)
    clean['day']=clean.date.dt.floor('D')
    clean['product']=clean.StockCode.astype(str)
    clean=clean.sort_values(['date','customer','invoice']).reset_index(drop=True)
    info.update({'positive_rows':len(clean),'positive_customers':int(clean.customer.nunique()),'positive_invoices':int(clean.groupby(['customer','invoice']).ngroups),'positive_gross_amount':float(clean.amount.sum()),'countries':int(clean.Country.nunique()),'flow':flow})
    dump('data_audit.json',info);clean.to_pickle(cache)
    audit_cache.write_text(json.dumps(info,indent=2),encoding='utf-8')
    pd.DataFrame(flow).to_csv(RES/'data_flow.csv',index=False)
    return clean

def snapshots(data):
    allrows=[]
    for cutoff in DATES:
        old=data.loc[data.date<cutoff]
        hist=old.loc[old.date>=cutoff-pd.Timedelta(days=90)]
        short=hist.loc[hist.date>=cutoff-pd.Timedelta(days=30)]
        future=data.loc[(data.date>=cutoff)&(data.date<cutoff+pd.Timedelta(days=30))]
        assert cutoff+pd.Timedelta(days=30)<=data.date.max()
        orders=hist.groupby(['customer','invoice']).agg(amount=('amount','sum'),items=('product','nunique')).reset_index()
        groups=hist.groupby('customer')
        out=pd.DataFrame(index=groups.size().index)
        out['recency90']=(cutoff-groups.date.max()).dt.total_seconds()/86400
        out['frequency90']=orders.groupby('customer').size()
        out['monetary90']=groups.amount.sum()
        out['tenure']=(cutoff-old.groupby('customer').date.min()).dt.total_seconds()/86400
        out['frequency30']=short.groupby('customer').invoice.nunique()
        out['monetary30']=short.groupby('customer').amount.sum()
        out['products90']=groups['product'].nunique()
        out['avg_order90']=orders.groupby('customer').amount.mean()
        out['items_per_order90']=orders.groupby('customer')['items'].mean()
        days=hist[['customer','day']].drop_duplicates().sort_values(['customer','day'])
        gap=days.groupby('customer').day.diff().dt.total_seconds()/86400
        out['mean_gap90']=days.assign(gap=gap).groupby('customer').gap.mean()
        out[['frequency30','monetary30']]=out[['frequency30','monetary30']].fillna(0)
        out['mean_gap90']=out.mean_gap90.fillna(90)
        # BTYD summaries use complete observed history, not a truncated first acquisition.
        life=old[['customer','day']].drop_duplicates().groupby('customer').day.agg(['min','max','count'])
        out['bgn_x']=life['count']-1
        out['bgn_tx']=(life['max']-life['min']).dt.total_seconds()/86400
        out['bgn_T']=(cutoff-life['min']).dt.total_seconds()/86400
        out['label']=out.index.isin(future.customer.unique()).astype(int)
        out['future_gross_amount']=future.groupby('customer').amount.sum().reindex(out.index).fillna(0)
        out['cutoff']=cutoff.strftime('%Y-%m-%d')
        out['split']='train' if cutoff.month<=6 else ('validation' if cutoff.month==7 else 'test')
        allrows.append(out.reset_index())
        log(f'cohort {cutoff.date()} n={len(out)} (labels hidden until analysis)')
    ans=pd.concat(allrows,ignore_index=True)
    assert not ans[['customer','cutoff']].duplicated().any()
    assert np.isfinite(ans[EXT].values).all()
    assert (ans[EXT]>=0).all().all()
    assert (ans.recency90<=90).all()
    assert (ans.bgn_tx<=ans.bgn_T).all()
    ans.to_csv(RES/'cohorts.csv',index=False)
    ans.groupby(['split','cutoff']).agg(n=('label','size'),positives=('label','sum'),prevalence=('label','mean'),customers=('customer','nunique')).reset_index().to_csv(RES/'cohort_summary.csv',index=False)
    return ans

def model(kind,params,seed=17):
    if kind.startswith('LR'):
        return make_pipeline(SimpleImputer(strategy='median'),FunctionTransformer(np.log1p,validate=False),StandardScaler(),LogisticRegression(C=params['C'],max_iter=2000,random_state=seed))
    if kind=='RF':return RandomForestClassifier(n_estimators=200,min_samples_leaf=params['min_samples_leaf'],max_features=.8,n_jobs=1,random_state=seed)
    return HistGradientBoostingClassifier(max_iter=150,learning_rate=.05,min_samples_leaf=30,early_stopping=False,random_state=seed,**params)

def bg_prob(fit,df):
    r,alpha,a,b=fit.params_.loc[['r','alpha','a','b']]
    p=fit.conditional_probability_alive(df.bgn_x.values,df.bgn_tx.values,df.bgn_T.values)
    p=p*(1-((alpha+df.bgn_T.values)/(alpha+df.bgn_T.values+30))**(r+df.bgn_x.values))
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p

def evaluate(df,p,prob=True,weights=None):
    y=df.label.values; w=np.ones(len(df)) if weights is None else np.asarray(weights)
    total=w.sum(); prev=np.dot(w,y)/total
    metrics={'AP':float(average_precision_score(y,p,sample_weight=w)),'AUC':float(roc_auc_score(y,p,sample_weight=w)),'prevalence':float(prev),'n':int(total)}
    if prob:
        metrics['Brier']=float(np.dot(w,(p-y)**2)/total)
        metrics['LogLoss']=float(log_loss(y,np.clip(p,1e-8,1-1e-8),sample_weight=w,labels=[0,1]))
    idx=np.lexsort((df.customer.values,-p))
    sw=w[idx]; sy=y[idx]; cumulative=np.cumsum(sw)-sw
    revenue=df.future_gross_amount.values
    for q in [.1,.2,.3]:
        k=int(np.ceil(q*total)); selected=np.clip(k-cumulative,0,sw)
        hits=float(np.dot(selected,sy)); precision=hits/k
        tag=str(int(q*100))
        metrics['P'+tag]=precision
        metrics['Recall'+tag]=hits/np.dot(w,y)
        metrics['Lift'+tag]=precision/prev
        metrics['GrossCoverage'+tag]=float(np.dot(selected,revenue[idx])/np.dot(w,revenue)) if np.dot(w,revenue)>0 else None
    return metrics

def plot(mt,macro,pred,selected,baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
    figs=RES/'figures';figs.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout();plt.savefig(figs/(name+'.png'),bbox_inches='tight');plt.savefig(figs/(name+'.svg'),bbox_inches='tight');plt.close()
    names=['Recency','Frequency','Monetary','RFM_rank','BGNBD','LR_RFM','LR_extended','RF','HGB','HGB_RFM']
    tab=macro.set_index('model').loc[names]
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    colors=['#285A7D' if x==selected else '#A2B3BE' for x in names]
    axs[0].barh(names,tab.AP,color=colors);axs[0].set_xlabel('Equal-month average precision')
    axs[1].barh(names,tab.P20,color=colors);axs[1].set_xlabel('Precision at 20% capacity')
    axs[1].axvline(mt.groupby('cutoff').prevalence.first().mean(),color='gray',linestyle='--',label='Expected random precision');axs[1].legend(fontsize=8)
    save('fig1_model_comparison')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for name in list(dict.fromkeys([selected,baseline,'HGB','LR_extended'])):
        g=mt[mt.model==name];axs[0].plot(g.cutoff,g.AP,marker='o',label=name)
        axs[1].plot([10,20,30],[tab.loc[name,'P'+str(k)] for k in [10,20,30]],marker='o',label=name)
    axs[0].set_ylabel('Average precision');axs[0].tick_params(axis='x',rotation=20);axs[0].legend(fontsize=8)
    axs[1].set_xlabel('Monthly selection capacity (%)');axs[1].set_ylabel('Precision');axs[1].legend(fontsize=8)
    save('fig2_time_and_capacity')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for name in list(dict.fromkeys([selected,'LR_extended','HGB','BGNBD'])):
        true,prob=calibration_curve(pred.label,pred[name],n_bins=8,strategy='quantile');axs[0].plot(prob,true,marker='o',label=name)
    axs[0].plot([0,1],[0,1],'--',color='gray');axs[0].set_xlabel('Mean predicted probability');axs[0].set_ylabel('Observed event proportion');axs[0].legend(fontsize=8)
    im=pd.read_csv(RES/'permutation_importance.csv').sort_values('AP_drop_mean')
    axs[1].barh(im.feature,im.AP_drop_mean,xerr=im.AP_drop_sd,color='#285A7D');axs[1].set_xlabel('Macro AP decrease after permutation')
    save('fig3_calibration_and_reliance')
    fig,ax=plt.subplots(figsize=(9,2.7))
    ax.broken_barh([(0,4)],(2, .6),facecolors='#285A7D');ax.broken_barh([(4,1)],(1,.6),facecolors='#A37532');ax.broken_barh([(5,4)],(0,.6),facecolors='#6A8060')
    ax.set_yticks([2.3,1.3,.3],['Training cutoffs','Validation cutoff','Final test cutoffs']);ax.set_xticks(np.arange(9)+.5,[d.strftime('%b 2011') for d in DATES]);ax.set_xlim(0,9);ax.set_title('90-day historical features; next-30-day outcome; test months fixed')
    save('fig4_temporal_protocol')

