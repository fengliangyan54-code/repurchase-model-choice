"""Supplementary analysis defined in configs/extension_v2.md; never replaces frozen_v1."""
from pathlib import Path
import os, sys, json, time, hashlib, zipfile, io, datetime
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import core
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

OUT=ROOT/'results'/'extension_v2_iofix'
OUT.mkdir(parents=True,exist_ok=True)
SETTINGS=json.loads((ROOT/'configs/frozen_selection.json').read_text())['settings']
COMMON=core.RFM+['tenure','frequency30','monetary30','avg_order90','mean_gap90']
def log(s): print(time.strftime('%Y-%m-%d %H:%M:%S'),s,flush=True)
def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2,default=str),encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load_cdnow():
    archive=ROOT/'data/CDNOW_master.zip'
    if not archive.exists():
        import urllib.request
        urllib.request.urlretrieve('https://brucehardie.com/datasets/CDNOW_master.zip',archive)
    with zipfile.ZipFile(archive) as z:
        names=[n for n in z.namelist() if Path(n).name.lower()=='cdnow_master.txt']
        assert len(names)==1,names
        raw=pd.read_csv(io.BytesIO(z.read(names[0])),sep=r'\s+',header=None,names=['customer','datestr','quantity','amount'])
    raw['date']=pd.to_datetime(raw.datestr.astype(str),format='%Y%m%d',errors='coerce')
    report={'source':'https://brucehardie.com/datasets/CDNOW_master.zip','documentation':'https://brucehardie.com/notes/026/','archive_sha256':digest(archive),'archive_bytes':archive.stat().st_size,'download_file_mtime_utc':datetime.datetime.fromtimestamp(archive.stat().st_mtime,datetime.timezone.utc).isoformat(),'raw_rows':len(raw),'customers_raw':int(raw.customer.nunique()),'exact_duplicate_records':int(raw.duplicated().sum()),'date_min':str(raw.date.min()),'date_max':str(raw.date.max()),'policy':'duplicate records retained; no invoice identifier; full cohort, not stratified one-tenth sample','redistribution_license':'not established; raw data omitted from distribution package'}
    clean=raw[raw.customer.notna()&raw.date.notna()&(raw.quantity>0)&(raw.amount>0)].copy()
    report.update({'excluded_nonpositive_or_invalid':len(raw)-len(clean),'positive_rows':len(clean),'positive_customers':int(clean.customer.nunique())})
    assert len(raw)==69659 and raw.customer.nunique()==23570
    assert clean.date.max()>=pd.Timestamp('1998-06-30')
    clean['day']=clean.date.dt.floor('D')
    dump('cdnow_provenance.json',report)
    return clean

def cohorts(data):
    frames=[]
    for cutoff in pd.date_range('1997-04-01','1998-06-01',freq='MS'):
        old=data[data.date<cutoff];hist=old[old.date>=cutoff-pd.Timedelta(days=90)];short=hist[hist.date>=cutoff-pd.Timedelta(days=30)]
        future=data[(data.date>=cutoff)&(data.date<cutoff+pd.Timedelta(days=30))]
        g=hist.groupby('customer');out=pd.DataFrame(index=g.size().index)
        out['recency90']=(cutoff-g.date.max()).dt.total_seconds()/86400
        out['frequency90']=g.size();out['monetary90']=g.amount.sum()
        out['tenure']=(cutoff-old.groupby('customer').date.min()).dt.total_seconds()/86400
        out['frequency30']=short.groupby('customer').size();out['monetary30']=short.groupby('customer').amount.sum()
        out[['frequency30','monetary30']]=out[['frequency30','monetary30']].fillna(0)
        out['avg_order90']=g.amount.mean()
        days=hist[['customer','day']].drop_duplicates().sort_values(['customer','day'])
        gap=days.groupby('customer').day.diff().dt.total_seconds()/86400
        out['mean_gap90']=days.assign(gap=gap).groupby('customer').gap.mean();out['mean_gap90']=out.mean_gap90.fillna(90)
        life=old[['customer','day']].drop_duplicates().groupby('customer').day.agg(['min','max','count'])
        out['bgn_x']=life['count']-1;out['bgn_tx']=(life['max']-life['min']).dt.days;out['bgn_T']=(cutoff-life['min']).dt.days
        out['label']=out.index.isin(future.customer.unique()).astype(int)
        out['future_gross_amount']=future.groupby('customer').amount.sum().reindex(out.index).fillna(0)
        out['cutoff']=str(cutoff.date());out['split']='train' if cutoff<pd.Timestamp('1997-08-01') else ('validation' if cutoff==pd.Timestamp('1997-08-01') else 'test')
        frames.append(out.reset_index())
    ans=pd.concat(frames,ignore_index=True)
    assert np.isfinite(ans[COMMON]).all().all() and not ans[['customer','cutoff']].duplicated().any()
    return ans

def evaluate_cdnow(df,tag):
    fit=df[df.split!='test'];test=df[df.split=='test'].reset_index(drop=True);val=df[df.split=='validation']
    preds={};cost=[]
    for name,kind,cols in [('LR_RFM','LR_RFM',core.RFM),('RF_RFM','RF',core.RFM),('HGB_RFM','HGB',core.RFM),('LR_common','LR_extended',COMMON),('RF_common','RF',COMMON),('HGB_common','HGB',COMMON)]:
        t=time.perf_counter();m=core.model(kind,SETTINGS[kind]);m.fit(fit[cols],fit.label);preds[name]=m.predict_proba(test[cols])[:,1]
        cost.append({'model':name,'fit_predict_seconds':time.perf_counter()-t});log(tag+' '+name)
    for name,col,sign in [('Recency','recency90',-1),('Frequency','frequency90',1),('Monetary','monetary90',1)]:preds[name]=sign*test[col].values
    preds['RFM_rank']=np.zeros(len(test))
    for _,g in test.groupby('cutoff'):
        preds['RFM_rank'][g.index]=(-g.recency90).rank(pct=True).values+g.frequency90.rank(pct=True).values+g.monetary90.rank(pct=True).values
    t=time.perf_counter();bg=BetaGeoFitter(penalizer_coef=SETTINGS['BGNBD']['penalty']);bg.fit(val.bgn_x,val.bgn_tx,val.bgn_T);preds['BGNBD']=core.bg_prob(bg,test)
    cost.append({'model':'BGNBD','fit_predict_seconds':time.perf_counter()-t});dump(tag+'_bgnbd_params.json',bg.params_.to_dict())
    rows=[];pred=test[['customer','cutoff','label','future_gross_amount']].copy()
    for name,p in preds.items():
        pred[name]=p
        for cutoff,g in test.groupby('cutoff'):
            rows.append({'model':name,'cutoff':cutoff,**core.evaluate(g,p[g.index],name not in {'Recency','Frequency','Monetary','RFM_rank'})})
    metrics=pd.DataFrame(rows);assert len(metrics)==110
    metrics.to_csv(OUT/(tag+'_monthly_metrics.csv'),index=False)
    metrics.groupby('model').mean(numeric_only=True).reset_index().to_csv(OUT/(tag+'_macro_metrics.csv'),index=False)
    pred.to_csv(OUT/(tag+'_predictions.csv'),index=False);pd.DataFrame(cost).to_csv(OUT/(tag+'_runtimes.csv'),index=False)
    df.groupby(['split','cutoff']).agg(n=('label','size'),positives=('label','sum'),prevalence=('label','mean')).reset_index().to_csv(OUT/(tag+'_cohort_summary.csv'),index=False)
    return test,pred,metrics

def bootstrap(test,pred,metrics):
    ids=np.sort(test.customer.unique());ix=pd.Index(ids).get_indexer(test.customer);groups=[g for _,g in test.groupby('cutoff')]
    path=OUT/'cdnow_paired_cluster_bootstrap.csv';rows=pd.read_csv(path,float_precision='round_trip').to_dict('records') if path.exists() else []
    rng=np.random.default_rng(20261021)
    for it in range(1000):
        w=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))[ix]
        if it<len(rows):continue
        vals={}
        for name in ['RF_RFM','BGNBD']:
            score=[core.evaluate(g,pred.loc[g.index,name].values,True,w[g.index]) for g in groups]
            vals[name]=np.mean([[s['AP'],s['P20']] for s in score],axis=0)
        d=vals['RF_RFM']-vals['BGNBD'];rows.append({'replicate':it,'AP_difference':d[0],'P20_difference':d[1]})
        if (it+1)%25==0:pd.DataFrame(rows).to_csv(path,index=False)
        if (it+1)%100==0:log('CDNOW bootstrap '+str(it+1))
    bt=pd.DataFrame(rows);macro=metrics.groupby('model').mean(numeric_only=True)
    dump('cdnow_comparison.json',{'status':'supplementary exploratory','comparison':'RF_RFM minus BGNBD','customers':len(ids),'test_months':len(groups),'replicates':len(bt),'AP':{'difference':macro.loc['RF_RFM','AP']-macro.loc['BGNBD','AP'],'CI95':np.quantile(bt.AP_difference,[.025,.975]).tolist()},'P20':{'difference':macro.loc['RF_RFM','P20']-macro.loc['BGNBD','P20'],'CI95':np.quantile(bt.P20_difference,[.025,.975]).tolist()},'conditional_on':'fixed fits and ten observed CDNOW months; no pooled merchant inference'})

def uci_robustness():
    base=ROOT/'results/frozen_v1_iofix';df=pd.read_csv(base/'cohorts.csv',float_precision='round_trip');pred=pd.read_csv(base/'test_predictions.csv',float_precision='round_trip');test=df[df.split=='test'].reset_index(drop=True)
    core.RES=OUT;data=core.prepare();old=data[data.date<pd.Timestamp('2011-07-01')]
    life=old[['customer','day']].drop_duplicates().groupby('customer').day.agg(['min','max','count'])
    x=life['count']-1;tx=(life['max']-life['min']).dt.days;T=(pd.Timestamp('2011-07-01')-life['min']).dt.days
    bg=BetaGeoFitter(penalizer_coef=SETTINGS['BGNBD']['penalty']);bg.fit(x,tx,T);p=core.bg_prob(bg,test)
    rows=[{'cutoff':c,'model':'BGNBD_all_observed',**core.evaluate(g,p[g.index],True)} for c,g in test.groupby('cutoff')]
    pd.DataFrame(rows).to_csv(OUT/'uci_bgnbd_all_observed.csv',index=False)
    dump('uci_bgnbd_all_observed_fit.json',{'fit_customers':len(life),'params':bg.params_.to_dict(),'post_hoc':True})
    # Independently trace diverse held-out feature rows from positive raw purchases.
    errors=[]
    for pos in [0,211,933,len(test)//2,len(test)-1]:
        row=test.iloc[pos];t=pd.Timestamp(row.cutoff);h=data[(data.customer==row.customer)&(data.date<t)&(data.date>=t-pd.Timedelta(days=90))]
        vals={'recency90':(t-h.date.max()).total_seconds()/86400,'frequency90':h.invoice.nunique(),'monetary90':h.amount.sum()}
        for f,v in vals.items():assert np.isclose(v,row[f],rtol=1e-12,atol=1e-8);errors.append({'customer':int(row.customer),'cutoff':row.cutoff,'feature':f,'from_transactions':float(v),'saved_value':float(row[f])})
    pd.DataFrame(errors).to_csv(OUT/'independent_feature_trace.csv',index=False)
    # AP via explicit precision increments at score ties; capacities via explicit sorted rows.
    expected=pd.read_csv(base/'monthly_metrics.csv',float_precision='round_trip');checks=[]
    for r in expected.itertuples():
        g=pred[pred.cutoff==r.cutoff];y=g.label.to_numpy();score=g[r.model].to_numpy()
        tab=pd.DataFrame({'score':score,'y':y}).groupby('score').agg(n=('y','size'),pos=('y','sum')).sort_index(ascending=False)
        ap=float(np.sum(tab.pos/y.sum()*tab.pos.cumsum()/tab.n.cumsum()))
        rank=np.lexsort((g.customer.to_numpy(),-score));n=int(np.ceil(.2*len(y)));precision=float(y[rank[:n]].mean())
        assert abs(ap-r.AP)<1e-12 and abs(precision-r.P20)<1e-12
        checks.append({'model':r.model,'cutoff':r.cutoff,'AP_absolute_error':abs(ap-r.AP),'P20_absolute_error':abs(precision-r.P20)})
    pd.DataFrame(checks).to_csv(OUT/'independent_metric_checks.csv',index=False)
    log('Independent UCI metrics and raw-feature trace passed')

def main():
    dump('analysis_fingerprint.json',{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'configs/extension_v2.md',ROOT/'configs/frozen_selection.json']})
    data=load_cdnow();df=cohorts(data);df.to_csv(OUT/'cdnow_cohorts.csv',index=False)
    test,pred,metrics=evaluate_cdnow(df,'cdnow')
    dedup=data.drop_duplicates(subset=['customer','datestr','quantity','amount'])
    evaluate_cdnow(cohorts(dedup),'cdnow_deduplicated')
    bootstrap(test,pred,metrics);uci_robustness()
    dump('extension_status.json',{'status':'completed','main_cells':110,'deduplication_sensitivity_cells':110,'bootstrap_replicates':1000,'scientific_status':'supplementary, post inherited test results; original comparison unchanged'})
    log('Supplementary analysis complete')
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
