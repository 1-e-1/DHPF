import hashlib, json, shutil
from pathlib import Path
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/timeseries_v2'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'benchmark').glob('*.py'))}
    freeze=OUT/'frozen_original.json'
    if freeze.exists():
        assert json.loads(freeze.read_text())['sha256']==hashes, 'Original benchmark changed'
    else:
        freeze.write_text(json.dumps({'created':datetime.now().isoformat(),'sha256':hashes,'constraint':'No changes to DHPF architecture, kernels, integration, training objective or original benchmark files.'},indent=2),encoding='utf-8')
    # An isolated copy of the existing trainer changes ONLY the DDPM schedule/target.
    source=(ROOT/'benchmark/run.py').read_text(encoding='utf-8')
    source=source.replace('from .','from benchmark.')
    source=source.replace('from benchmark.protocol import RESULTS, config, write_protocol','from benchmark.protocol import config, write_protocol\nRESULTS=ROOT/\'results\'/\'timeseries_v2\'\nfrom research_ts.diffusion import generate, diffusion_schedule')
    old='loss=(model(xt,t,c)-x).square().mean()'
    assert source.count(old)==1
    source=source.replace(old,'target=abar[idx,None].sqrt()*x-(1-abar[idx,None]).sqrt()*y\n                    loss=(model(xt,t,c)-target).square().mean()')
    source=source.replace("files=sorted((ROOT/'benchmark').glob('*.py'))","files=sorted((ROOT/'benchmark').glob('*.py'))+[ROOT/'research_ts'/n for n in ['runner.py','diffusion.py','study.py']]")
    source=source.replace('beta,alpha,abar,var=diffusion_schedule(device)','beta,alpha,abar,var=[z.float() for z in diffusion_schedule(device)]')
    source=source.replace("hashes={f.name:","hashes={str(f.relative_to(ROOT)):")
    source=source.replace("for f in files: shutil.copy2(f,path/f.name)","for f in files:\n            dst=path/f.relative_to(ROOT); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,dst)")
    (ROOT/'research_ts/runner.py').write_text(source,encoding='utf-8')
    claims=[{'id':'existing_sine_ACF','value':'0.052','source':'results/checklist_v1/summary.csv'}, {'id':'five_HAR_runs','value':'checkpoint_selection','source':'results/checklist_v1/runs/har__dhpf__base__seed0/metrics.json'}]
    (ROOT/'.aris/claims.json').write_text(json.dumps(claims,indent=2),encoding='utf-8')
    plan='''# DHPF 时序生成研究续篇：冻结方案

日期：2026-10-15。用户约束：不修改模型结构；以时序数据生成为主任务；完成研究与中文论文；图像仅作补充。

## 研究对象与主张
沿用共享移动源、径向与反对称切向核、二阶积分及终点 minibatch OT。生成时间 t 与序列时间 tau 严格区分。只研究固定长度多变量窗口生成，HAR 按活动类别条件生成；不宣称历史条件预测模型。

C1（待检验）：既有结构在部分时序依赖统计上存在可复现的优势，同时报告分布拟合和下游效用的代价。
C2（方法描述）：DHPF 构成结构化加速度场加终点分布学习的生成方法；不得把各已有组件、二阶动力学、Helmholtz 分解或 OT 单独宣称首创。

## 四个实验块
1. 基线审计：保留旧 epsilon/linear DDPM 的失败数据；检验 forward/reverse 公式与 v 参数化代数一致性。补充 cosine/v DDPM，沿用同一个 VectorNet，1000 步 ancestral 采样，不修改 DHPF 或增加训练期外的数据。
2. 主对照：复用 Sine/Lorenz 原 5 个种子的 GAN、I-CFM、OT-CFM、DHPF、raw Drifting；增加上述 DDPM-v 的 5 个种子。HAR 补齐 GAN、DDPM-v、I-CFM、OT-CFM、raw Drifting 的 5 个种子，复用 DHPF 5 个种子。合计 35 个新增正式训练。
3. 训练和评价：每个新增方法/任务使用 seed100、1000 updates、lr=[0.0002,0.001] 的验证集选择；正式 seed0..4、5000 updates、batch128；每1000 updates按验证 SW2²选择 checkpoint。HAR 固定原 subject-disjoint split，条件平均统计，TSTR macro-F1；Sine/Lorenz 2000生成样本与原测试集，TSTR ridge 96→32。模型参数按原配置匹配，报告残余差异；按 updates 而非计算时间匹配。所有实验单卡顺序执行，先 sanity。两候选 HPO 共14个短训练，不以测试结果调参。
4. 证据整合：复用原有机制、敏感性和数值实验；补充 real-v-real 参照、seed 配对差异和精确符号检验（5种子不足以支持双侧p<0.05，最小0.0625）、样本分布诊断。图像结果仅在讨论中说明适用边界，不新跑图像。

## 结果判定与边界
无论结果正负都写入论文。不得声称整体优于所有生成算法、SOTA、通用最优、移动源必要性、训练效率优势或已收敛。raw Drifting 是原坐标适配，不等于官方完整模型。旧失败 DDPM 不作优越性核心证据。仅一个真实数据集时明确跨领域外推尚未验证。固定结构限制下若优势消失，完成负面证据分析与方法研究论文，不通过更换结构或选择性删除结果挽救结论。

## 交付
冻结哈希、完整配置与原始日志/检查点/样本、可重建汇总 CSV/JSON、可编辑科研图、中文完整论文 Word、证据复核记录。正文采用实际观察结论；没有投稿目标则采用通用研究论文格式，作者信息由用户后续填写。
'''
    folder=ROOT/'refine-logs'; folder.mkdir(exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    for name,content in [('EXPERIMENT_PLAN',plan),('EXPERIMENT_TRACKER','# 实验进度\n\n原清单390项完成；DHPF源代码已冻结。新增35个正式实验和14个验证集短实验待执行。\n')]:
        (folder/f'{name}_{stamp}.md').write_text(content,encoding='utf-8'); (folder/f'{name}.md').write_text(content,encoding='utf-8')
    manifest=ROOT/'MANIFEST.md'
    if not manifest.exists(): manifest.write_text('# Research Output Manifest\n\n| Timestamp | Skill | File | Stage | Description |\n|---|---|---|---|---|\n',encoding='utf-8')
    with manifest.open('a',encoding='utf-8') as f:
        f.write(f'| {stamp} | experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | 冻结结构、时序续篇实验方案及时间戳副本 |\n')
        f.write(f'| {stamp} | experiment-plan | results/timeseries_v2/frozen_original.json | implementation | 原始代码 SHA256 |\n')
    print('Frozen original benchmark; isolated trainer and plan ready.')

if __name__=='__main__': main()
