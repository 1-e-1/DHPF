import csv,html,json,hashlib
from pathlib import Path
from datetime import datetime
from docx import Document
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/timeseries_v2'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    status=read(OUT/'analysis_status.json');assert status['complete'] and status['original_hashes_unchanged']
    rows=[r for r in read(OUT/'summary.json') if r['variant']=='base' and r['model'] in ['dhpf','gan','ddpm_v','icfm','otcfm','drifting']]
    assert len(rows)==18 and sum(r['n'] for r in rows)==90
    body=['<h1>DHPF 时序生成研究结果</h1><p>2026-10-15 · 模型结构冻结 · 35个新增正式训练、14个验证集短训练全部完成。主对照涵盖3任务×6方法×5种子，共90个训练实例。</p>', '<p><b>研究结论：</b>Sine 自相关、Sine/Lorenz 零滞后通道相关存在局部优势；分布距离、谱幅值和下游预测未全面改善，HAR 未显示全面领先。所有结论按有限预算和本实现报告。证据复核为 partial / same-family / provisional；不等于独立复现或投稿录用保证。</p>']
    labels={'sw2_squared':'SW₂² ↓','mmd2_unbiased':'MMD² ↓','power_rmse':'谱RMSE ↓','autocorrelation_rmse':'ACF RMSE ↓','cross_correlation_rmse':'Corr₀ RMSE ↓','synthetic_test_forecast_mse':'TSTR MSE ↓','synthetic_test_macro_f1':'LR TSTR F1 ↑'}
    for dataset in ['sine','lorenz','har']:
        body.append('<h2>'+dataset+'</h2><div class="scroll"><table><tr><th>方法</th>'+''.join('<th>'+v+'</th>' for k,v in labels.items() if ('f1' not in k if dataset!='har' else 'forecast' not in k))+'</tr>')
        keys=[k for k in labels if ('f1' not in k if dataset!='har' else 'forecast' not in k)]
        for row in rows:
            if row['dataset']!=dataset:continue
            body.append('<tr><th>'+html.escape(row['model'])+'</th>'+''.join(f"<td>{row[k+'_mean']:.4f} ± {row[k+'_std']:.4f}</td>" for k in keys)+'</tr>')
        body.append('</table></div>')
    body.append('<h2>评价与复现边界</h2><p>误差为5训练种子的样本标准差。HAR先逐类计算分布统计，再六类等权平均。90项探索性主比较的Holm校正p值均为1；5种子符号检验最小双侧p=.0625。真实数据参照包含抽样和受试者偏移，不是数学下界。</p><p>逻辑回归探针的部分拟合达到迭代上限，已为全部方法另行完成float64闭式ridge探针；详见ridge_probe。旧epsilon/linear DDPM失稳结果完整保留，新DDPM-v使用相同网络和cosine/v参数化，不修改DHPF。raw Drifting不是官方完整时序模型复现。</p><h2>证据文件</h2><ul>')
    for p in ['summary.csv','all_runs.csv','per_class.csv','paired_comparisons.json','real_reference.json','inference_timing.json','EXPERIMENT_AUDIT.json','frozen_original.json','study_complete.json']:
        body.append(f'<li><a href="{p}">{p}</a></li>')
    body.append('</ul>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DHPF 时序研究结果</title><style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#24323c;line-height:1.7}h1{font-size:28px}h2{margin-top:32px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}th:first-child{text-align:left}tr:first-child{background:#eef3f6}.scroll{overflow:auto}a{color:#305d7b}</style>'+''.join(body)+'</html>'
    (OUT/'report.html').write_text(page,encoding='utf-8')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    content='''# 时序续篇实验结果

35个新增正式实验和14个短HPO全部完成，主对照90个训练实例。DHPF原16个benchmark文件SHA256不变；未新增图像训练。

研究判定：partial / same-family / provisional。局部统计优势成立为探索性观察，不支持全面领先。HAR分布与相关指标未胜主要基线，逻辑回归和ridge效用完整报告。所有90项原主比较统一Holm校正，无显著性结论。

完整证据：results/timeseries_v2/report.html、summary.csv、all_runs.csv、per_class.csv、paired_comparisons.json、ridge_probe、EXPERIMENT_AUDIT.json。原始训练配置、checkpoint、日志及样本在runs；初评和终评trace在.aris/traces/result-to-claim。

论文：DHPF_时序数据生成研究论文.docx；正文源、图源与数据构建脚本均已保留。Word视觉验收状态见paper/document_qa.json。
'''
    tracker='''# 实验进度

| 工作 | 状态 |
|---|---|
| 冻结原模型结构与代码 | 完成，16文件哈希一致 |
| 新基线公式和条件训练sanity | 完成，5类目标 |
| 新增HPO | 14/14 |
| 新增正式训练 | 35/35 |
| 三任务主表 | 90/90个训练实例 |
| HAR类内指标与条件公平性 | 完成 |
| 闭式ridge稳健性 | 30合成 + 1真实参照 |
| 统计、数值、计时、幅值诊断 | 完成 |
| 证据复核 | 完成，partial/same-family/provisional |
| 中文论文 | 完整正文、11公式、7表、5图；视觉验收见QA记录 |
'''
    for name,text in [('EXPERIMENT_RESULTS',content),('EXPERIMENT_TRACKER',tracker)]:
        for suffix in ['_'+stamp,'']:(ROOT/'refine-logs'/f'{name}{suffix}.md').write_text(text,encoding='utf-8')
    findings=ROOT/'findings.md'
    if not findings.exists():findings.write_text('# Research Findings\n',encoding='utf-8')
    with findings.open('a',encoding='utf-8') as f:f.write('\n## 2026-10-15 时序研究续篇\n\n初评和终评均为partial；完整补充后保留局部相关优势，拒绝整体SOTA和效率优势。未改DHPF。保留不利HAR结果和DDPM失败诊断。创新定位于具体共享源结构与训练流程组合，非各组件首创。最终门禁为same-family/provisional，审计warn来自逻辑回归迭代上限，已用闭式ridge复核；不等于独立验证。完整trace见.aris/traces/result-to-claim/2026-10-15_run02。\n')
    with (ROOT/'MANIFEST.md').open('a',encoding='utf-8') as f:
        for path,skill,description in [('results/timeseries_v2/report.html','result-to-claim','时序研究完整结果与边界'),('refine-logs/EXPERIMENT_RESULTS.md','run-experiment','训练完成与结果归档'),('DHPF_时序数据生成研究论文.docx','research-writing-skill + documents','完整中文论文'),('paper/figures','nature-figure','5组科研图及源代码/数据/审计')]:f.write(f'| {stamp} | {skill} | {path} | paper | {description} |\n')
    repro='''Reproduction (from the cloned repository root; historical artifacts required):
python -X utf8 -m research_ts.study
python -X utf8 -m research_ts.analyze
python -X utf8 -m research_ts.robust_probe
python -X utf8 -m research_ts.final_checks
python -X utf8 -m research_ts.figures
python -X utf8 -m research_ts.build_paper

study resumes/completes existing runs and will not retrain completed ones.
Do not rerun setup_study.py over the finished project: it is an initial setup tool.
Do not edit frozen benchmark/*.py. Analysis and document tools are outside the training-source fingerprint.
Original DHPF and non-DDPM synthetic runs: results/checklist_v1/runs.
New runs: results/timeseries_v2/runs. Dataset cache: data/checklist_v1.
Selected checkpoints use validation only. All negative results are retained.
Figures need the installed nature-figure audit helper; it does not change training.
The renderer is a temporary local document QA dependency, not a model dependency.
'''
    (OUT/'REPRODUCE.txt').write_text(repro,encoding='utf-8')
    print('Research report and reproducibility records written.')
if __name__=='__main__':main()
