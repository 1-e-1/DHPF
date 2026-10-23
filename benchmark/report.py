"""Evidence-based checklist status. A queued job is never counted as completed."""
import csv
import html
import json
from collections import Counter,defaultdict
from pathlib import Path
from filelock import FileLock
import numpy as np
from .protocol import RESULTS,MODELS
from .suite import jobs
from .run import write_json


def _report():
    rows=[];expected=[]
    for group in ['toy','sequence','image','ablation','sensitivity','har']:
        for dataset,model,variant,overrides in jobs(group):
            for seed in range(5):expected.append(dict(group=group,dataset=dataset,model=model,variant=variant,seed=seed,name=f'{dataset}__{model}__{variant}__seed{seed}'))
    for row in expected:
        folder=RESULTS/'runs'/row['name'];row=row.copy()
        status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else dict(status='pending')
        row.update(status=status['status'],training_complete=(folder/'training_complete.json').exists(),distribution_evaluated=(folder/'metrics.json').exists(),specialized_evaluated=(folder/'image_metrics.json').exists() if row['group']=='image' else (folder/'downstream.json').exists() if row['dataset'] in ['sine','lorenz','har'] else True,postprocessed=(folder/'postprocess_complete.json').exists())
        if row['distribution_evaluated']:
            met=json.loads((folder/'metrics.json').read_text())
            for k,v in met.items():
                if isinstance(v,(int,float)) and not isinstance(v,bool):row[k]=v
            if row['dataset']=='har':
                row['conditional_mean_sw2_squared']=float(np.mean([v['sw2_squared'] for k,v in met.items() if k.isdigit()]))
        if (folder/'image_metrics.json').exists():
            met=json.loads((folder/'image_metrics.json').read_text())
            for k in ['FD_MNIST','frechet_inception_distance','inception_score_mean','inception_score_std']: 
                if k in met:row[k]=met[k]
        if (folder/'downstream.json').exists():
            met=json.loads((folder/'downstream.json').read_text())
            for origin in ['synthetic','real_matched']:
                for key in ['test_forecast_mse','test_macro_f1']:
                    if key in met[origin]:row[origin+'_'+key]=met[origin][key]
        if (folder/'metadata.json').exists():
            meta=json.loads((folder/'metadata.json').read_text());row.update(parameters=meta['parameters'],discriminator_parameters=meta['discriminator_parameters'])
        if (folder/'budget_final_metrics.json').exists():
            final=json.loads((folder/'budget_final_metrics.json').read_text());final=final['metrics'] if final.get('same_as_selected') else final
            if 'sw2_squared' in final:row['budget_final_sw2_squared']=final['sw2_squared']
        row['complete']=row['training_complete'] and row['distribution_evaluated'] and row['specialized_evaluated'] and row['postprocessed'] and (folder/'budget_final_metrics.json').exists()
        rows.append(row)
    def csvfile(path,records):
        if not records:return
        fields=list(dict.fromkeys(k for r in records for k in r))
        with path.open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    csvfile(RESULTS/'all_runs.csv',rows)
    grouped=defaultdict(list)
    for row in rows:grouped[(row['dataset'],row['model'],row['variant'])].append(row)
    summary=[]
    for (dataset,model,variant),rs in grouped.items():
        out=dict(dataset=dataset,model=model,variant=variant,expected_training_seeds=5,trained=sum(r['training_complete'] for r in rs),evaluated=sum(r['distribution_evaluated'] for r in rs),complete=sum(r['complete'] for r in rs),failed=sum(r['status']=='failed' for r in rs))
        metrics=['sw2_squared','w2_squared_1024_mean','valid_fraction','mode_coverage','mmd2_unbiased','power_rmse','autocorrelation_rmse','cross_correlation_rmse','FD_MNIST','frechet_inception_distance','inception_score_mean','synthetic_test_forecast_mse','real_matched_test_forecast_mse','synthetic_test_macro_f1','real_matched_test_macro_f1']
        for key in metrics:
            vals=[r[key] for r in rs if key in r];out[key+'_n']=len(vals)
            if vals:out[key+'_mean']=float(np.mean(vals));out[key+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        summary.append(out)
    csvfile(RESULTS/'summary.csv',summary)
    # Paired differences use identical training seeds, and retain incomplete pair counts.
    paired=[]
    lookup={(r['dataset'],r['model'],r['variant'],r['seed']):r for r in rows}
    for dataset in ['gmm8','sine']:
        for model in ['direct','augmented']:
            for seed in range(5):
                a=lookup[(dataset,'dhpf','base',seed)];b=lookup[(dataset,model,'base',seed)]
                if 'sw2_squared' in a and 'sw2_squared' in b:paired.append(dict(dataset=dataset,control=model,seed=seed,dhpf_minus_control_sw2_squared=a['sw2_squared']-b['sw2_squared']))
    csvfile(RESULTS/'E05_paired_differences.csv',paired)
    checks={}
    e01=RESULTS/'E01'/'checks.json';checks['E01']=dict(status='completed' if e01.exists() and json.loads(e01.read_text())['status']=='passed' else 'pending')
    mapping={'E02':['toy'],'E03':['image'],'E04':['sequence'],'E06':['ablation'],'E07':['sensitivity'],'E12':['har']}
    for eid,groups in mapping.items():
        selected=[r for r in rows if r['group'] in groups]
        if eid=='E02':selected=[r for r in selected if r['model'] in MODELS]
        if eid=='E04':selected=[r for r in selected if r['model'] in MODELS]
        if eid=='E06':selected += [r for r in rows if r['dataset'] in ['gmm8','sine'] and r['model']=='dhpf' and r['variant']=='base']
        if eid=='E07':selected += [r for r in rows if r['dataset'] in ['gmm8','sine'] and r['model']=='dhpf' and r['variant']=='base']
        done=sum(r['complete'] for r in selected);checks[eid]=dict(status='completed' if done==len(selected) else 'in_progress' if any(r['status']!='pending' for r in selected) else 'pending',expected=len(selected),complete=done,trained=sum(r['training_complete'] for r in selected),evaluated=sum(r['distribution_evaluated'] for r in selected))
    selected=[r for r in rows if r['dataset'] in ['gmm8','sine'] and r['model'] in ['direct','augmented','dhpf'] and r['variant']=='base']
    checks['E05']=dict(status='completed' if all(r['complete'] for r in selected) else 'in_progress',expected=30,complete=sum(r['complete'] for r in selected),paired_results=len(paired))
    for eid,file in [('E08','efficiency.json'),('E09','sampling_study.json'),('E10','mechanisms.npz')]:
        selected=[r for r in rows if r['variant']=='base' and (eid=='E08' or r['seed']==0) and (eid!='E10' or r['model']=='dhpf')]
        def artifact_ok(r):
            path=RESULTS/'runs'/r['name']/file
            if not path.exists():return False
            if eid=='E09':
                data=json.loads(path.read_text())
                return data['reference']['reference_verified'] is not False
            return True
        done=sum(artifact_ok(r) for r in selected)
        checks[eid]=dict(status='completed' if done==len(selected) else 'in_progress' if done else 'pending',expected=len(selected),complete=done,scope='all base runs' if eid=='E08' else 'fixed seed 0 checkpoints; sampling diagnostic uses 512 fixed initial states')
    status=RESULTS/'E11'/'status.json';checks['E11']=json.loads(status.read_text()) if status.exists() else dict(status='in_progress',complete=len(list((RESULTS/'E11').glob('*/metrics.json'))),expected=36)
    ledger=dict(overall='completed' if all(r['complete'] for r in rows) and all(v['status'].startswith('completed') for v in checks.values()) else 'in_progress',expected_training_runs=len(rows),trained=sum(r['training_complete'] for r in rows),distribution_evaluated=sum(r['distribution_evaluated'] for r in rows),fully_processed=sum(r['complete'] for r in rows),status_counts=dict(Counter(r['status'] for r in rows)),experiments=dict(sorted(checks.items())))
    write_json(RESULTS/'checklist_status.json',ledger)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [eid,d['status'],d.get('trained',''),d.get('complete',''),d.get('expected','')])+'</tr>' for eid,d in sorted(checks.items()))
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>DHPF 实验进度</title><style>body{{max-width:1100px;margin:40px auto;font:16px/1.6 system-ui;background:#f6f8fb;color:#182b42}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:9px 15px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff4d9;padding:16px}}a{{color:#0754ac}}</style><h1>DHPF-V3 实验清单执行记录</h1><p class="note">实际完成情况：训练 {ledger['trained']} / {len(rows)}，分布评价 {ledger['distribution_evaluated']}，完整后处理 {ledger['fully_processed']}。状态为 {ledger['overall']}。启动队列不代表实验完成；固定预算结束不代表收敛。</p><p>旧版 41 次诊断与本轮正式队列分开保存。主实验训练种子为 0–4，超参数只依据验证集选择。Drifting-raw 使用公开漂移目标的原始坐标版本，不宣称复现原论文的特征空间图像系统。</p><table><tr><th>实验</th><th>状态</th><th>训练完成</th><th>交付完成</th><th>预期</th></tr>{table}</table><p><a href="protocol.json">固定协议</a> · <a href="all_runs.csv">逐种子原值及状态</a> · <a href="summary.csv">均值与样本标准差</a> · <a href="E05_paired_differences.csv">机制对照配对差值</a> · <a href="E01/checks.json">E01 数值检查</a></p><p>E11 为单种子补充诊断，不用于五种子优势结论。E09、E10 采用预先指定的 seed 0 checkpoint；E09 的 512 个样本为数值诊断，不能与正式 10000 / 50000 样本指标混用。所有正式训练目前统一标注“固定预算结束，尚未证实收敛”。</p></html>'''
    (RESULTS/'report.html').write_text(body,encoding='utf-8')
    print(json.dumps({k:ledger[k] for k in ['overall','expected_training_runs','trained','distribution_evaluated','fully_processed']}),flush=True)


def report():
    RESULTS.mkdir(parents=True,exist_ok=True)
    with FileLock(str(RESULTS/'report.lock'),timeout=120):_report()


if __name__=='__main__':report()
