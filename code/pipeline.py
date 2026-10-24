"""Portable, checkpointed reproduction. Scientific choices are frozen in configs/."""
from pathlib import Path
import sys, os, json, hashlib, time, traceback, argparse, platform
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'cache'/'matplotlib')
sys.path.insert(0,str(ROOT/'code'))
import core
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from lifetimes import BetaGeoFitter

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8');temporary.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint():
    names=['code/core.py','code/pipeline.py','configs/frozen_selection.json','configs/protocol_v1.md','data/Online Retail.xlsx']
    return {p:sha(ROOT/p) for p in names}

def process_alive(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:
            return ctypes.get_last_error()==5 # permission denied is not proof of inactivity
        code=wintypes.DWORD()
        try:return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False
    except PermissionError:return True

class RunLock:
    def __init__(self,folder):self.path=folder/'active_run.json'
    def __enter__(self):
        if self.path.exists():
            previous=json.loads(self.path.read_text(encoding='utf-8'))
            if process_alive(previous['pid']):raise RuntimeError(f"This run is active under PID {previous['pid']}; inspect its log, do not launch a duplicate.")
            self.path.rename(self.path.with_name('inactive_lock_'+str(time.time_ns())+'.json'))
        fd=os.open(str(self.path),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'started':time.strftime('%Y-%m-%d %H:%M:%S'),'command':sys.argv},f)
        return self
    def __exit__(self,*args):
        if self.path.exists() and json.loads(self.path.read_text(encoding='utf-8')).get('pid')==os.getpid():self.path.unlink()

class Runner:
    def __init__(self,run_id):
        if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):raise ValueError('run-id must contain only letters/digits/_/-')
        self.out=ROOT/'results'/run_id;self.out.mkdir(parents=True,exist_ok=True)
        self.status=self.out/'stage_status.json';self.finger=fingerprint()
        if self.status.exists():
            self.state=json.loads(self.status.read_text(encoding='utf-8'))
            if self.state['fingerprint']!=self.finger:raise RuntimeError('Input/code fingerprint changed. Preserve this run; use a new --run-id and document why. Do not overwrite.')
        else:self.state={'run_id':run_id,'fingerprint':self.finger,'stages':{}};atomic_json(self.status,self.state)
        core.RES=self.out
        self.config=json.loads((ROOT/'configs'/'frozen_selection.json').read_text(encoding='utf-8'))
    def log(self,message):
        line=time.strftime('%Y-%m-%d %H:%M:%S')+' '+str(message)
        print(line,flush=True)
        with (self.out/'progress.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
    def complete(self,name,files,extra=None):
        expected=self.state['stages'].get(name,{})
        if expected.get('status')=='completed':
            for fn,digest in expected['files'].items():
                if not (self.out/fn).is_file() or sha(self.out/fn)!=digest:raise RuntimeError(f'Completed stage {name}: missing or changed {fn}; preserve evidence and repair in a new run.')
            self.log('Reuse verified stage '+name);return True
        return False
    def mark(self,name,files,**extra):
        self.state['stages'][name]={'status':'completed','files':{str(Path(p).relative_to(self.out)).replace('\\','/'):sha(p) for p in files},**extra}
        atomic_json(self.status,self.state)
    def check(self):
        import importlib.metadata as md
        raw_expected=json.loads((ROOT/'data'/'provenance.json').read_text())['sha256']
        assert sha(ROOT/'data'/'Online Retail.xlsx')==raw_expected,'Raw data checksum mismatch'
        assert sys.version_info[:2]==(3,12),'Use bundled Python 3.12 environment'
        assert sys.maxsize>2**32,'64-bit Python required'
        versions={x:md.version(x) for x in ['numpy','pandas','scipy','scikit-learn','lifetimes','openpyxl','matplotlib','reportlab','python-docx']}
        atomic_json(self.out/'environment.json',{'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count(),'numeric_threads':1,'versions':versions,'fingerprint':self.finger})
        self.log('Environment and raw-file checksum passed')
    def prepare(self):
        if self.complete('prepare',[]):return pd.read_csv(self.out/'cohorts.csv',float_precision='round_trip')
        self.log('Preparing transaction data and causal-time customer snapshots')
        data=core.prepare();df=core.snapshots(data)
        self.mark('prepare',[self.out/'cohorts.csv',self.out/'cohort_summary.csv',self.out/'data_audit.json',self.out/'data_flow.csv'])
        return df
    def smoke(self):
        if self.complete('smoke',[]):return
        df=self.prepare();train=df.loc[df.split=='train'];val=df.loc[df.split=='validation']
        assert not df[['customer','cutoff']].duplicated().any()
        boundary=pd.Timestamp('2011-08-01');series=pd.Series([boundary-pd.Timedelta(seconds=1),boundary,boundary+pd.Timedelta(days=30)])
        assert (series<boundary).tolist()==[True,False,False]
        assert ((series>=boundary)&(series<boundary+pd.Timedelta(days=30))).tolist()==[False,True,False]
        assert np.isfinite(df[core.EXT]).all().all()
        assert (df.recency90>0).all() and (df.recency90<=90).all()
        assert (df.bgn_tx<=df.bgn_T).all()
        reports=[]
        for kind in ['LR_RFM','LR_extended','RF','HGB']:
            cols=core.RFM if kind=='LR_RFM' else core.EXT
            m=core.model(kind,self.config['settings'][kind])
            if kind=='RF':m.set_params(n_estimators=8)
            if kind=='HGB':m.set_params(max_iter=10)
            m.fit(train.iloc[:1000][cols],train.iloc[:1000].label)
            p=m.predict_proba(val.iloc[:200][cols])[:,1]
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            reports.append({'model':kind,'train_rows':1000,'validation_rows':200,'status':'passed','formal_test':False})
        june=train.loc[train.cutoff=='2011-06-01'];bg=BetaGeoFitter(penalizer_coef=self.config['settings']['BGNBD']['penalty'])
        bg.fit(june.bgn_x,june.bgn_tx,june.bgn_T)
        p=core.bg_prob(bg,val.iloc[:200]);assert np.isfinite(p).all()
        # Independent numerical check of Gamma waiting-time Laplace transform.
        r,alpha,_,_=bg.params_.loc[['r','alpha','a','b']];sample=val.iloc[0];rng=np.random.default_rng(201017)
        rates=rng.gamma(r+sample.bgn_x,1/(alpha+sample.bgn_T),size=100000)
        mc=np.mean(1-np.exp(-30*rates));closed=1-((alpha+sample.bgn_T)/(alpha+sample.bgn_T+30))**(r+sample.bgn_x)
        assert abs(mc-closed)<.005
        path=self.out/'smoke_check.json';atomic_json(path,{'status':'passed','half_open_boundary':'passed','probability_bounds':'passed','BGNBD_gamma_probability_check_abs_error':abs(mc-closed),'model_checks':reports,'no_final_test_model_metrics_computed':True})
        self.mark('smoke',[path]);self.log('Smoke/correctness checks passed; no final-test performance used')
    def run(self):
        if self.complete('run',[]):return
        self.smoke();df=self.prepare();fitdata=df[df.split!='test'];val=df[df.split=='validation'];test=df[df.split=='test'].copy()
        self.log('Fitting frozen settings; no new hyperparameter selection')
        preds={};runtimes=[];modeldir=self.out/'models';modeldir.mkdir(exist_ok=True)
        for kind in ['LR_RFM','LR_extended','RF','HGB']:
            cols=core.RFM if kind=='LR_RFM' else core.EXT
            t=time.perf_counter();m=core.model(kind,self.config['settings'][kind]);m.fit(fitdata[cols],fitdata.label);fitsec=time.perf_counter()-t
            t=time.perf_counter();preds[kind]=m.predict_proba(test[cols])[:,1];predsec=time.perf_counter()-t
            joblib.dump({'model':m,'features':cols},modeldir/(kind+'.joblib'))
            runtimes.append({'model':kind,'fit_seconds':fitsec,'predict_seconds':predsec,'test_rows':len(test)})
            self.log('Fitted '+kind)
        for name,col,sign in [('Recency','recency90',-1),('Frequency','frequency90',1),('Monetary','monetary90',1)]:preds[name]=sign*test[col].values
        preds['RFM_rank']=np.zeros(len(test))
        for cutoff,g in test.groupby('cutoff'):
            loc=test.index.get_indexer(g.index)
            preds['RFM_rank'][loc]=(-g.recency90).rank(pct=True).values+g.frequency90.rank(pct=True).values+g.monetary90.rank(pct=True).values
        t=time.perf_counter();bg=BetaGeoFitter(penalizer_coef=self.config['settings']['BGNBD']['penalty']);bg.fit(val.bgn_x,val.bgn_tx,val.bgn_T)
        preds['BGNBD']=core.bg_prob(bg,test);bg.save_model(str(modeldir/'BGNBD.pkl'))
        runtimes.append({'model':'BGNBD','fit_and_predict_seconds':time.perf_counter()-t,'parameters':json.dumps(bg.params_.to_dict())})
        m=core.model('HGB',self.config['settings']['HGB']);m.fit(fitdata[core.RFM],fitdata.label);preds['HGB_RFM']=m.predict_proba(test[core.RFM])[:,1]
        rows=[];frame=test[['customer','cutoff','label','future_gross_amount']].reset_index(drop=True)
        probnames={'LR_RFM','LR_extended','RF','HGB','BGNBD','HGB_RFM'}
        for name,p in preds.items():
            frame[name]=p
            for cutoff,g in test.groupby('cutoff'):
                loc=test.index.get_indexer(g.index);rows.append({'model':name,'cutoff':cutoff,**core.evaluate(g,p[loc],name in probnames)})
        frame.to_csv(self.out/'test_predictions.csv',index=False)
        metrics=pd.DataFrame(rows);metrics.to_csv(self.out/'monthly_metrics.csv',index=False)
        metrics.groupby('model').mean(numeric_only=True).reset_index().to_csv(self.out/'macro_metrics.csv',index=False)
        pd.DataFrame(runtimes).to_csv(self.out/'runtimes.csv',index=False)
        pd.DataFrame([{'experiment_id':'main_'+r['model']+'_'+r['cutoff'],'status':'completed','model':r['model'],'cutoff':r['cutoff'],'sample_rows':r['n'],'excluded':0} for r in rows]).to_csv(self.out/'run_manifest.csv',index=False)
        files=[self.out/f for f in ['test_predictions.csv','monthly_metrics.csv','macro_metrics.csv','runtimes.csv','run_manifest.csv']]+list(modeldir.iterdir())
        self.mark('run',files,cells=len(rows));self.log('Completed 40 model-month evaluations, one shared test dataset')
    def analyze(self):
        if self.complete('analyze',[]):return
        self.run();df=self.prepare();fitdata=df[df.split!='test'];test=df[df.split=='test'].reset_index(drop=True)
        predictions=pd.read_csv(self.out/'test_predictions.csv',float_precision='round_trip');macro=pd.read_csv(self.out/'macro_metrics.csv',float_precision='round_trip').set_index('model')
        selected=self.config['selected_supervised'];baseline=self.config['selected_baseline'];ids=np.sort(test.customer.unique())
        indexes=pd.Index(ids).get_indexer(test.customer);groups=[(g,g.index.to_numpy()) for _,g in test.groupby('cutoff')]
        bootstrap_path=self.out/'paired_cluster_bootstrap.csv'
        rows=pd.read_csv(bootstrap_path,float_precision='round_trip').to_dict('records') if bootstrap_path.exists() else []
        if rows and [int(r['replicate']) for r in rows]!=list(range(len(rows))):raise RuntimeError('Bootstrap checkpoint sequence corrupt')
        rng=np.random.default_rng(20261020)
        for it in range(1000):
            weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))[indexes]
            if it<len(rows):continue
            vals={}
            for name in [selected,baseline]:
                scored=[core.evaluate(g,predictions[name].values[loc],True,weights[loc]) for g,loc in groups]
                vals[name]=np.mean([[r['AP'],r['P20']] for r in scored],axis=0)
            rows.append({'replicate':it,'AP_difference':vals[selected][0]-vals[baseline][0],'P20_difference':vals[selected][1]-vals[baseline][1]})
            if (it+1)%25==0:
                tmp=bootstrap_path.with_suffix('.tmp');pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(bootstrap_path);self.log(f'Cluster bootstrap {it+1}/1000')
        bt=pd.DataFrame(rows)
        report={'supervised':selected,'baseline':baseline,'bootstrap_replicates':len(bt),'customer_clusters':len(ids),'conditional_on':'fixed fitted models and four fixed calendar test months'}
        for metric,col in [('AP','AP_difference'),('P20','P20_difference')]:report[metric]={'difference':float(macro.loc[selected,metric]-macro.loc[baseline,metric]),'CI95':np.quantile(bt[col],[.025,.975]).tolist()}
        atomic_json(self.out/'primary_comparison.json',report)
        seedfile=self.out/'training_seed_sensitivity.csv';seedrows=pd.read_csv(seedfile,float_precision='round_trip').to_dict('records') if seedfile.exists() else []
        done={(r['model'],int(r['seed'])) for r in seedrows}
        for kind in ['RF','HGB']:
            for seed in [17,29,43]:
                if (kind,seed) in done:continue
                m=core.model(kind,self.config['settings'][kind],seed);m.fit(fitdata[core.EXT],fitdata.label);p=m.predict_proba(test[core.EXT])[:,1]
                scored=[core.evaluate(g,p[loc],True) for g,loc in groups]
                seedrows.append({'model':kind,'seed':seed,'AP':np.mean([r['AP'] for r in scored]),'P20':np.mean([r['P20'] for r in scored])})
                pd.DataFrame(seedrows).to_csv(seedfile,index=False);self.log(f'Seed sensitivity {kind}/{seed}')
        importancefile=self.out/'permutation_importance.csv';importance=pd.read_csv(importancefile,float_precision='round_trip').to_dict('records') if importancefile.exists() else []
        saved=joblib.load(self.out/'models'/(selected+'.joblib'));m,cols=saved['model'],saved['features'];done={r['feature'] for r in importance}
        for j,col in enumerate(cols):
            if col in done:continue
            rng=np.random.default_rng(30017+j);drops=[]
            for repeat in range(10):
                aps=[]
                for g,loc in groups:
                    X=g[cols].copy();X[col]=rng.permutation(X[col].values);aps.append(average_precision_score(g.label,m.predict_proba(X)[:,1]))
                drops.append(float(macro.loc[selected,'AP']-np.mean(aps)))
            importance.append({'feature':col,'AP_drop_mean':np.mean(drops),'AP_drop_sd':np.std(drops,ddof=1),'repeats':10})
            pd.DataFrame(importance).sort_values('AP_drop_mean',ascending=False).to_csv(importancefile,index=False);self.log('Permutation reliance '+col)
        # Refit once with identical data/settings to verify the deterministic primary result.
        rerun=core.model(selected,self.config['settings'][selected]);rerun.fit(fitdata[cols],fitdata.label);p=rerun.predict_proba(test[cols])[:,1]
        error=float(np.max(abs(p-predictions[selected].values)));assert error<1e-10
        atomic_json(self.out/'reproduction_check.json',{'status':'passed','max_prediction_difference':error,'type':'same-machine deterministic rerun; not independent third-party validation'})
        self.mark('analyze',[bootstrap_path,self.out/'primary_comparison.json',seedfile,importancefile,self.out/'reproduction_check.json']);self.log('Uncertainty, seed variation, model reliance and rerun complete')
    def figures(self):
        if self.complete('figures',[]):return
        self.analyze();mt=pd.read_csv(self.out/'monthly_metrics.csv',float_precision='round_trip');macro=pd.read_csv(self.out/'macro_metrics.csv',float_precision='round_trip');pred=pd.read_csv(self.out/'test_predictions.csv',float_precision='round_trip')
        core.plot(mt,macro,pred,self.config['selected_supervised'],self.config['selected_baseline'])
        self.mark('figures',list((self.out/'figures').glob('*')))
    def paper(self):
        if self.complete('paper',[]):return
        self.figures()
        from paper_builder import build
        files=build(self.out,self.config)
        self.mark('paper',files);self.log('Manuscripts and report generated from actual completed data; author review still required')
    def audit(self):
        self.paper();mt=pd.read_csv(self.out/'monthly_metrics.csv',float_precision='round_trip');bt=pd.read_csv(self.out/'paired_cluster_bootstrap.csv',float_precision='round_trip')
        assert len(mt)==40 and not mt[['model','cutoff']].duplicated().any()
        assert len(bt)==1000 and np.isfinite(bt[['AP_difference','P20_difference']]).all().all()
        assert mt[['AP','AUC','P20']].ge(0).all().all() and mt[['AP','AUC','P20']].le(1).all().all()
        assert len(pd.read_csv(self.out/'training_seed_sensitivity.csv',float_precision='round_trip'))==6
        assert len(pd.read_csv(self.out/'permutation_importance.csv',float_precision='round_trip'))==len(core.EXT)
        # Numeric reproduction from saved prediction arrays, independent of stored metrics table.
        pred=pd.read_csv(self.out/'test_predictions.csv',float_precision='round_trip')
        for row in mt.itertuples():
            g=pred[pred.cutoff==row.cutoff];actual=average_precision_score(g.label,g[row.model]);assert abs(actual-row.AP)<1e-12
        path=self.out/'audit_report.json';atomic_json(path,{'status':'passed','cells':40,'bootstrap_replicates':1000,'no_duplicate_evidence':True,'AP_recomputed_from_saved_predictions':True,'causal_sales_effect_identified':False,'external_merchant_validation_completed':False,'journal_submission_completed':False,'author_metadata_confirmed':False,'limitations':'single historical merchant; four fixed test months; no marketing treatment data; abstracts-only near-neighbor reading explicitly recorded'})
        self.mark('audit',[path]);self.log('Audit passed for declared historical single-merchant scope; no submission or causal claim')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['check','prepare','smoke','run','analyze','figures','paper','audit','all']);parser.add_argument('--run-id',default='frozen_v1')
    args=parser.parse_args();runner=Runner(args.run_id)
    try:
        with RunLock(runner.out),threadpool_limits(limits=1):
            runner.check()
            if args.stage!='check':getattr(runner,'audit' if args.stage=='all' else args.stage)()
    except BaseException:
        runner.log('INTERRUPTED OR FAILED. Completed stages/checkpoints retained. '+traceback.format_exc())
        raise

if __name__=='__main__':main()
