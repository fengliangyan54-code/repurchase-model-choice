"""Small scientific/preparation checks; not formal empirical results."""
from pathlib import Path
import sys,json,tempfile,re
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core
import numpy as np
import pandas as pd
from paper_builder import TOKENS,build
from pipeline import RunLock,process_alive

def main():
    df=pd.DataFrame({'customer':[10,11,12,13,14,15],'label':[1,0,1,0,1,0],'future_gross_amount':[2,0,7,0,3,0]})
    score=np.array([.8,.8,.6,.4,.3,.1]);w=np.array([2,0,1,3,1,2])
    weighted=core.evaluate(df,score,True,w)
    positions=np.repeat(np.arange(len(df)),w)
    expanded=core.evaluate(df.iloc[positions].reset_index(drop=True),score[positions],True)
    for key in ['AP','AUC','Brier','P10','P20','P30','Recall20','Lift20']:
        assert abs(weighted[key]-expanded[key])<1e-12,(key,weighted[key],expanded[key])
    template=(core.ROOT/'manuscript'/'article_template.md.in').read_text(encoding='utf-8')
    found=set(re.findall(r'@@([A-Z0-9_]+)@@',template))
    assert found==set(TOKENS),'Template and builder tokens differ'
    assert len(set(TOKENS))==len(TOKENS)
    refs=(core.ROOT/'literature'/'references.md').read_text(encoding='utf-8')
    assert len(re.findall(r'(?m)^\[\d+\]',refs))==18
    assert '\ufffd' not in refs
    with tempfile.TemporaryDirectory(prefix='ecommerce_preflight_') as scratch:
        try:build(Path(scratch),{})
        except RuntimeError as exc:assert 'missing evidence' in str(exc)
        else:raise AssertionError('Builder must reject absent formal evidence')
        assert process_alive(__import__('os').getpid())
        with RunLock(Path(scratch)):
            try:
                with RunLock(Path(scratch)):pass
            except RuntimeError:pass
            else:raise AssertionError('A duplicate live run must be blocked')
        assert not (Path(scratch)/'active_run.json').exists()
    # Resuming bootstrap must reproduce the exact suffix of the declared RNG sequence.
    a=np.random.default_rng(20261020);full=[a.multinomial(6,np.full(6,1/6)) for _ in range(5)]
    b=np.random.default_rng(20261020)
    for _ in range(2):b.multinomial(6,np.full(6,1/6))
    suffix=[b.multinomial(6,np.full(6,1/6)) for _ in range(3)]
    assert all(np.array_equal(x,y) for x,y in zip(full[2:],suffix))
    print(json.dumps({'status':'passed','weighted_cluster_metrics_equal_expanded_sample':True,'capacity_recomputed_with_multiplicity':True,'template_tokens':len(TOKENS),'references':18,'builder_rejects_missing_evidence':True,'resumed_random_stream_exact':True,'duplicate_live_run_blocked':True,'formal_experiment':False},indent=2),flush=True)

if __name__=='__main__':main()
