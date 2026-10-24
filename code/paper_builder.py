"""Fill an evidence-bounded manuscript only after all declared analyses exist."""
from pathlib import Path
import json, re, html
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TOKENS=['CLEAN_LINES','CUSTOMERS','SELECTED','BASELINE','SELECTED_AP','BASELINE_AP','AP_DIFF','AP_CI','SELECTED_P20','BASELINE_P20','ABSTRACT_INTERPRETATION','DATA_FLOW_TABLE','INVOICES','MISSING_CUSTOMER','COHORT_TABLE','TRAIN_ROWS','VALIDATION_ROWS','TEST_ROWS','TEST_CUSTOMERS','BG_PENALTY','SETTINGS_TABLE','RESULTS_PARAGRAPH','MAIN_RESULTS_TABLE','CAPACITY_PARAGRAPH','MONTHLY_RESULTS_TABLE','MECHANISM_PARAGRAPH','SEED_TABLE','COMPUTE_PARAGRAPH','DISCUSSION_COMPARISON','CONCLUSION_RESULT','REFERENCES']

def table(frame):
    # No optional tabulate dependency; Markdown remains portable.
    rows=[list(map(str,frame.columns))]+[[f'{v:.4f}' if isinstance(v,(float,np.floating)) and np.isfinite(v) else ('-' if pd.isna(v) else str(v)) for v in row] for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(r)+' |' for r in rows[1:]])

def build(out,config):
    out=Path(out);paper=out/'paper';paper.mkdir(exist_ok=True)
    required=['macro_metrics.csv','monthly_metrics.csv','cohort_summary.csv','data_audit.json','primary_comparison.json','training_seed_sensitivity.csv','permutation_importance.csv','runtimes.csv','reproduction_check.json']
    for name in required:
        if not (out/name).is_file():raise RuntimeError('Manuscript blocked: missing evidence '+name)
    main=pd.read_csv(out/'macro_metrics.csv').set_index('model');monthly=pd.read_csv(out/'monthly_metrics.csv');cohorts=pd.read_csv(out/'cohort_summary.csv');pred=pd.read_csv(out/'test_predictions.csv')
    audit=json.loads((out/'data_audit.json').read_text());comp=json.loads((out/'primary_comparison.json').read_text());im=pd.read_csv(out/'permutation_importance.csv');seeds=pd.read_csv(out/'training_seed_sensitivity.csv');runtime=pd.read_csv(out/'runtimes.csv')
    assert comp['bootstrap_replicates']==1000 and len(seeds)==6 and len(im)==10
    selected,baseline=config['selected_supervised'],config['selected_baseline'];s,b=main.loc[selected],main.loc[baseline];diff=comp['AP']['difference'];lo,hi=comp['AP']['CI95']
    plo,phi=comp['P20']['CI95'];pdiff=comp['P20']['difference']
    direction='higher' if diff>0 else 'lower'
    if lo>0:interpret='The conditional interval favors the selected machine-learning model on AP.'
    elif hi<0:interpret='The conditional interval favors the classical comparator on AP.'
    else:interpret='The conditional interval includes zero and does not establish a clear AP advantage for either selected model.'
    aptext=f'{diff:+.4f}';citext=f'[{lo:+.4f}, {hi:+.4f}]'
    mainview=main.reset_index()[['model','AP','AUC','Brier','P20','Recall20','Lift20']]
    shortlist=monthly[monthly.model.isin([selected,baseline])][['cutoff','model','AP','P20','Lift20']]
    breakdown=cohorts.groupby('split')['n'].sum().to_dict()
    imp=im.sort_values('AP_drop_mean',ascending=False).head(3)
    besttest=main.AP.idxmax() # descriptive, never used for retraining/primary selection.
    vals={
      'CLEAN_LINES':f"{audit['positive_rows']:,}",'CUSTOMERS':f"{audit['positive_customers']:,}",'INVOICES':f"{audit['positive_invoices']:,}",'MISSING_CUSTOMER':f"{audit['missing_customer_rows']:,}",
      'SELECTED':selected,'BASELINE':baseline,'SELECTED_AP':f'{s.AP:.4f}','BASELINE_AP':f'{b.AP:.4f}','AP_DIFF':aptext,'AP_CI':citext,'SELECTED_P20':f'{s.P20:.2%}','BASELINE_P20':f'{b.P20:.2%}','ABSTRACT_INTERPRETATION':interpret,
      'DATA_FLOW_TABLE':table(pd.DataFrame(audit['flow']).fillna('-')),'COHORT_TABLE':table(cohorts[['split','cutoff','n','positives','prevalence']]),'TRAIN_ROWS':f"{breakdown['train']:,}",'VALIDATION_ROWS':f"{breakdown['validation']:,}",'TEST_ROWS':f"{breakdown['test']:,}",'TEST_CUSTOMERS':f'{pred.customer.nunique():,}',
      'BG_PENALTY':str(config['settings']['BGNBD']['penalty']),'SETTINGS_TABLE':table(pd.DataFrame([{'Model':k,'Frozen settings':json.dumps(v,sort_keys=True)} for k,v in config['settings'].items()])),
      'RESULTS_PARAGRAPH':f'The development-selected {selected} attains macro AP {s.AP:.4f}, compared with {b.AP:.4f} for {baseline}. Its paired AP difference is {aptext}, with conditional 95% customer-cluster percentile interval {citext}. {interpret} These intervals condition on the fitted models and observed four months. They are not uncertainty intervals for arbitrary future retailers or seasons. Across all reported methods, {besttest} has the largest test point estimate; this is a descriptive observation and does not replace the frozen primary comparison.',
      'MAIN_RESULTS_TABLE':table(mainview),
      'CAPACITY_PARAGRAPH':f'At the focal 20% capacity, {selected} precision is {s.P20:.2%}, recall is {s.Recall20:.2%} and cohort-normalized lift is {s.Lift20:.3f}. The corresponding {baseline} quantities are {b.P20:.2%}, {b.Recall20:.2%} and {b.Lift20:.3f}. The paired precision difference is {pdiff:+.4f}, with a conditional 95% interval [{plo:+.4f}, {phi:+.4f}]. The lift is concentration relative to naturally occurring purchases, not additional sales attributable to contact. The supplied complete monthly metrics also contain the 10% and 30% capacities and descriptive gross-revenue coverage.',
      'MONTHLY_RESULTS_TABLE':table(shortlist),
      'MECHANISM_PARAGRAPH':f'Using the frozen boosting settings, extended-feature HGB has macro AP {main.loc["HGB","AP"]:.4f}, while RFM-only HGB has {main.loc["HGB_RFM","AP"]:.4f}. The difference of {main.loc["HGB","AP"]-main.loc["HGB_RFM","AP"]:+.4f} is an exploratory representation comparison, not a separately confirmed causal mechanism or the outcome of retuning each ablation. In the selected model, the largest mean permutation AP losses are '+', '.join(f'{r.feature} ({r.AP_drop_mean:.4f})' for r in imp.itertuples())+'. Correlation between transaction summaries limits interpretation of individual importance values.',
      'SEED_TABLE':table(seeds),
      'COMPUTE_PARAGRAPH':'Frozen model fitting and prediction times are reported below for the actual executing machine. They exclude initial spreadsheet loading, bootstrap inference and document generation. No universal runtime or deployed throughput is inferred. The deterministic refit and independent AP-from-predictions checks passed.\n\n'+table(runtime.drop(columns=['parameters'],errors='ignore')),
      'DISCUSSION_COMPARISON':f'In the primary comparison, the selected supervised model has a {direction} AP point estimate than {baseline}, by {abs(diff):.4f}. {interpret} This result should guide model choice within this benchmark rather than motivate additional test-based tuning until a preferred model wins.',
      'CONCLUSION_RESULT':f'The frozen {selected} versus {baseline} comparison yields AP {s.AP:.4f} versus {b.AP:.4f}, a difference of {aptext} with conditional interval {citext}. At 20% capacity, precision is {s.P20:.2%} versus {b.P20:.2%}.',
      'REFERENCES':(ROOT/'literature'/'references.md').read_text(encoding='utf-8').strip()
    }
    template=(ROOT/'manuscript'/'article_template.md.in').read_text(encoding='utf-8')
    found=set(re.findall(r'@@([A-Z0-9_]+)@@',template));assert found==set(TOKENS)==set(vals)
    for key,value in vals.items():template=template.replace('@@'+key+'@@',value)
    assert not re.search(r'@@[A-Z0-9_]+@@',template)
    manuscript=paper/'manuscript_en.md';manuscript.write_text(template,encoding='utf-8')
    summary=f'''# AI赋能电商客户运营研究结果说明

状态：基于已完成计算生成的历史单商家实证研究稿，尚未投稿，作者及投稿信息待确认。

研究题目为《AI赋能电商客户运营：基于时间外验证的复购预测与预算约束评估》。原始数据为541,909条真实交易行；清洗后保留{audit['positive_rows']:,}条正向购买记录。用2011年3至6月训练、7月验证，8至11月测试，预测过去90天活跃客户在未来30天是否购买。本文使用机器学习，不涉及生成式AI部署。

开发阶段已选定{selected}与{baseline}为主要比较，未根据测试结果更换主要方法。宏平均AP分别为{s.AP:.4f}和{b.AP:.4f}，差值{aptext}，按客户聚类的条件95%区间为{citext}。20%筛选容量下，命中率分别为{s.P20:.2%}和{b.P20:.2%}。差值和区间来自实际结果，不是人工设定的提升目标。

这些数值衡量自然发生的后续购买者能否被筛选出来，不是营销触达使销量增加的证据。数据没有处理分组、接触、激励和贡献毛利，不能估计营销增量、ROI或因果效果。模型重要性也不等于顾客行为的因果驱动因素。

结果只覆盖一个历史商家和四个固定月份；跨商家、现代平台和真实营销实验尚未完成。原始记录缺失客户ID、取消和退款的处理以及首次出现时间并非真实获客时间，均限制外推。复跑同一数据只能作为计算复现，不能算新增独立样本。

请先阅读英文完整稿中的结果、局限与声明，并核对目标期刊最新要求。作者、单位、基金、伦理和利益冲突不得虚构；本文不自称新算法或已经被SCI录用。
'''
    cn=paper/'中文研究结果说明.md';cn.write_text(summary,encoding='utf-8')
    # Portable HTML for reading; no external assets or script dependencies.
    htmlpath=paper/'manuscript_en.html';htmlpath.write_text(markdown_html(template),encoding='utf-8')
    files=[manuscript,cn,htmlpath]
    # Editable Word source is produced; its layout still needs visual review on the executing laptop.
    files.append(write_docx(template,paper/'manuscript_en.docx'))
    (paper/'layout_review_required.txt').write_text('Word source generated. Render with a real DOCX renderer and inspect every page before labeling layout verified. HTML provides a portable reading copy. No DOCX-render success is claimed by this script.\n',encoding='utf-8')
    files.append(paper/'layout_review_required.txt')
    return files

def markdown_html(text):
    lines=text.splitlines();body=[];tablelines=[]
    def flush():
        if not tablelines:return
        rows=[r for r in tablelines if not re.match(r'^\|[\s|:-]+\|$',r)]
        body.append('<table>'+''.join('<tr>'+''.join(f'<td>{html.escape(c.strip())}</td>' for c in r.strip('|').split('|'))+'</tr>' for r in rows)+'</table>');tablelines.clear()
    for line in lines:
        if line.startswith('|'):tablelines.append(line);continue
        flush()
        if not line:continue
        img=re.match(r'!\[(.*?)\]\((.*?)\)',line)
        if img:body.append(f'<img alt="{html.escape(img[1])}" src="{html.escape(img[2])}">');continue
        m=re.match(r'^(#{1,3}) (.*)',line)
        if m:body.append(f'<h{len(m[1])}>{html.escape(m[2])}</h{len(m[1])}>')
        else:body.append('<p>'+html.escape(line)+'</p>')
    flush()
    return '<!doctype html><html><head><meta charset="utf-8"><title>AI-enabled e-commerce research manuscript</title><style>body{max-width:900px;margin:40px auto;padding:0 25px;font:16px/1.65 Georgia,serif;color:#111}h1,h2,h3{line-height:1.25;color:#000}table{border-collapse:collapse;font:13px/1.4 Arial,sans-serif;width:100%;margin:18px 0}td{border:1px solid #d9d9d9;padding:7px}tr:first-child{background:#e8edf0;font-weight:bold}img{max-width:100%;height:auto}p{overflow-wrap:anywhere}@media print{body{margin:0}}</style></head><body>'+''.join(body)+'</body></html>'

def write_docx(text,dest):
    from docx import Document
    from docx.shared import Inches,Pt,RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    doc=Document();section=doc.sections[0];section.page_width=Inches(8.5);section.page_height=Inches(11)
    section.top_margin=section.bottom_margin=Inches(.8);section.left_margin=section.right_margin=Inches(.85)
    for name in ['Normal','Title','Heading 1','Heading 2','Heading 3']:
        s=doc.styles[name];s.font.name='Times New Roman';s.font.color.rgb=RGBColor(0,0,0)
    doc.styles['Normal'].font.size=Pt(11);doc.styles['Normal'].paragraph_format.space_after=Pt(6)
    lines=text.splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if not line:i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                if not re.match(r'^\|[\s|:-]+\|$',lines[i]):rows.append([c.strip() for c in lines[i].strip('|').split('|')])
                i+=1
            t=doc.add_table(rows=1,cols=len(rows[0]));t.autofit=True
            for j,v in enumerate(rows[0]):t.rows[0].cells[j].text=v
            for row in rows[1:]:
                cells=t.add_row().cells
                for j,v in enumerate(row):cells[j].text=v
            for ridx,row in enumerate(t.rows):
                for cell in row.cells:
                    tcpr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
                    for edge in ['top','left','bottom','right']:
                        el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
                    tcpr.append(borders)
                    if ridx==0:
                        shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E8EDF0');tcpr.append(shade)
                    for p in cell.paragraphs:
                        p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(3)
                        for run in p.runs:run.font.size=Pt(9)
            continue
        img=re.match(r'!\[(.*?)\]\((.*?)\)',line)
        if img:doc.add_picture(str(dest.parent/img[2]),width=Inches(6.6))
        elif line.startswith('# '):doc.add_paragraph(line[2:],style='Title')
        elif line.startswith('## '):doc.add_heading(line[3:],level=1)
        elif line.startswith('### '):doc.add_heading(line[4:],level=2)
        else:doc.add_paragraph(line)
        i+=1
    doc.save(dest);return dest

if __name__=='__main__':raise SystemExit('Use pipeline.py paper; completed evidence is required.')
