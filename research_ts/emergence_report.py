"""Self-contained Chinese report of measured inference dynamics."""
import base64,html,json
from pathlib import Path
import pandas as pd
from research_ts.inference_emergence import OUT,write_json

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def pm(s,n=3):return f"{s['mean']:.{n}f} ± {s['std']:.{n}f}" if s['std'] is not None else f"{s['mean']:.{n}f}"
def table(headers,rows):
    return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def picture(name,caption):
    p=OUT/'figures'/name;mime='image/gif' if p.suffix=='.gif' else 'image/png'
    src='data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()
    return '<figure><img src="'+src+'" alt="'+html.escape(caption)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'

def main():
    s=read('summary.json');refine=read('refinement.json');rm=read('refinement_metrics.json');complete=read('complete.json')
    names={'sine':'Sine','lorenz':'Lorenz','har':'HAR：Walking'}
    text=[]
    text.append('<h1>DHPF 完整推理过程中的“涌现”诊断</h1><p class="date">2026-10-17 · 固定检查点 · 无新增训练</p>')
    text.append('<div class="finding"><b>结论：</b>生成样本群中出现了可复现的收缩—展开、协同旋转和协方差谱集中现象；但没有证据表明力子本身通过相互作用发生了自组织。力子的高度同步在未训练网络中已经存在。更准确的表述是：<b>训练得到的共享力场产生了有组织的群体输运动力学。</b></div>')
    text.append('<p>这里区分两个对象：<b>力子（场源）</b>是网络输出的中心、强度、核宽度和低秩旋转参数；<b>样本粒子</b>是从随机位置和速度出发、经积分后得到完整时序窗口的状态。下面的时间 t∈[0,1] 是生成时间，不是原始时序的观测时间。</p>')
    text.append('<h2>实际测试范围</h2><p>主诊断使用 Sine、Lorenz、HAR Walking 的 15 个既有检查点，每项任务 5 个训练种子。每次从相同的 256 组随机初态开始，记录 32 步显式中点积分的全部 33 个状态；另外以 129 个时刻查询全部 32 个力子。每个主检查点包含完整模型、未训练初始化、无力运动、冻结全部场源参数、仅径向项、仅无散项六种条件。另检查其余 5 个 HAR 活动的 seed 0，并进行步长细化。原模型源码与检查点哈希保持不变。</p>')
    text.append('<p>未训练模型按原训练种子重建初始化。所有关闭分量、冻结时间的条件都是同一检查点上的诊断干预，不是经过重新训练的替代模型。均值 ± 标准差以 5 个训练种子为重复单位，不能把 33 个时间点当成独立重复。</p>')
    text.append('<h2>1. 力子同步很明显，但不是训练后新产生的现象</h2>')
    rows=[]
    for d,r in s.items():
        rows.append([names[d],pm(r['sources']['full']['motion_alignment_mean']),pm(r['sources']['untrained']['motion_alignment_mean']),pm(r['sources']['full']['phase_shift_null_mean']),pm(r['sources']['full']['source_radius_ratio'],4)])
    text.append(table(['任务','训练后方向一致度','未训练方向一致度','独立时间错位对照','力子末端/初始分散半径'],rows))
    text.append('<p>方向一致度是各力子单位速度向量均值的模，范围为 0–1。训练后约 0.97–0.99，未训练模型约 0.99，二者都远高于独立时间错位后的约 0.18。训练并没有提高这一指标；共享时间编码足以解释主要的同步行为。力子末端相对初始的分散半径接近 1，也不支持“一次推理中力子持续聚成团”的描述。训练后的力子整体更紧凑，与推理过程中发生聚集是不同命题。</p>')
    sanity=read('sanity.json')['checks'];mx=max(r['solo_vs_batch_trajectory_max_abs_error'] for r in sanity)
    text.append(f'<p>因果检查进一步表明：改变一个力子的基础位置后，其余 31 个力子的输出差异为 <b>0</b>；同一个样本单独推理与批量推理的全过程最大坐标差为 <b>{mx:.2e}</b>，处于 float32 数值误差量级。当前代码没有“样本状态反馈给力子”或“力子之间相互推进”的通道，因此图上的同步不能解释成自主群体通信。</p>')
    text.append(picture('01_sources_and_structure.png','图 1：上排为力子方向一致度，下排为样本协方差 participation ratio 随完整推理时间的变化。实线为训练后的完整 DHPF，阴影为五个训练种子的样本标准差。时间错位虚线是 99 个独立相移对照的平均值；HAR 仅 Walking。'))
    text.append('<h2>2. 样本群出现了清晰的结构形成过程</h2>')
    rows=[]
    for d,r in s.items():
        rows.append([names[d],f"{r['initial_effective_dimension']:.2f}",pm(r['controls']['full']['effective_dimension'],2),f"{r['real_reference_effective_dimension']:.2f}",pm(r['stages']['minimum_spread_time']),pm(r['stages']['terminal_over_minimum_spread'],2)])
    text.append(table(['任务','初始有效维数','终点有效维数','真实窗口参考','最小分散度时刻','终点/最小分散度'],rows))
    text.append('<p>“有效维数”采用协方差 participation ratio：(tr C)² / tr(C²)，表示变化是否集中在少数方向，不是对真实流形维数的估计。Sine 与 Lorenz 从约 152.5 降至 7.81 和 2.30；HAR Walking 从约 209.1 降至 2.44。无力运动与未训练模型没有形成同程度的谱集中。Sine 在 t≈0.875 收缩最强，之后分散度回升到最小值的约 1.90 倍；Lorenz 对应约 4.17 倍。HAR 的收缩时刻在种子间差异较大，不能给它指定统一的转换时刻。</p>')
    text.append('<p>这些现象支持“训练形成了阶段性的分布输运”。但生成分布的有效维数低于对应真实数据参考，也提示过度压缩。不能仅由降维或投影上的聚集推断出新的吸引子、模态形成或更高生成质量。</p>')
    text.append('<h2>3. 高维旋转趋势通过了速度打乱和步长细化检查</h2>')
    rows=[]
    fine={r['dataset']:r['circulation_coherence'] for r in read('circulation_refinement.json')}
    for d,r in s.items():
        rows.append([names[d],pm(r['circulation_endpoint']['full']),pm(r['circulation_endpoint']['ballistic']),pm(r['circulation_shuffle_null']),f"{r['circulation_endpoint']['full']['values'][0]:.3f} → {fine[d]:.3f}"])
    text.append(table(['任务','完整模型环流一致度','无力对照','打乱速度对照均值','seed 0：K32 → K128'],rows))
    text.append('<p>环流一致度使用中心化位置与速度的反对称张量：‖Σᵢ(xᵢvᵢᵀ−vᵢxᵢᵀ)‖F / Σᵢ‖xᵢvᵢᵀ−vᵢxᵢᵀ‖F。它在完整高维空间计算，不依赖选择哪个二维投影。纯同向二维旋转的校验值为 1，纯径向运动为 0。该指标是在初步结果之后补充的探索性分析。</p>')
    text.append('<p>完整模型的终点环流一致度明显高于无力运动，也高于保留位置、速度各自分布但随机打乱二者配对的对照。细化积分后 seed 0 的数值接近，支持这一趋势不是单纯的粗步长假象。不过，环流一致度描述的是位置—速度的组织关系，不等价于已发现稳定涡旋、闭合轨道或极限环。冻结场源在 Sine 上也能产生更高的环流一致度，因此这不是动态场源独有的现象。</p>')
    text.append(picture('02_contraction_and_circulation.png','图 2：上排展示样本分散度的收缩与再展开；下排比较各诊断条件的终点环流一致度。点与误差线为五种子均值 ± 标准差；点线为各检查点速度打乱对照 95% 分位数的平均值，不是总体显著性阈值。Full/Init./Free/Frozen/Radial/Sol. 分别为完整、未训练、无力、冻结场源、仅径向、仅无散。阴影为标准差区间，可能延伸到非物理的负分散度范围；实际观测分散度均非负。'))
    text.append('<h2>4. 是否有利于生成质量，要单独判断</h2>')
    rows=[]
    for d,r in s.items():
        rows.append([names[d],pm(r['controls']['full']['acf_rmse']),pm(r['controls']['ballistic']['acf_rmse']),pm(r['controls']['frozen_t0']['acf_rmse']),pm(r['field']['effective_sources'],1),pm(r['field']['force_cancel'])])
    text.append(table(['任务','完整模型 ACF 误差','无力 ACF 误差','冻结场源 ACF 误差','有效贡献力子数','作用抵消比例'],rows))
    text.append('<p>Sine 与 Lorenz 的结构形成伴随 ACF 误差下降；HAR Walking 的 ACF 误差反而高于无力对照，不能把结构形成直接等同于性能改善。Sine 冻结场源时环流更强，但 ACF 误差高于完整模型，进一步说明“旋转更强”本身不是更好的优化目标。</p>')
    text.append('<p>按各力子对样本的作用模长归一化，平均约有 23–31 个力子参与贡献；它们的作用约有 28%–34% 相互抵消。这里的“有效贡献数”和“抵消比例”是全路径、全样本的描述量，支持分布式场作用，尚不足以证明力子间发生自组织。</p>')
    text.append('<h2>完整推理动画</h2>')
    text.append('<p>动画展示固定二维投影，而收缩指标在完整的 384/1152 维空间计算。样本向少数主方向集中时，二维投影可能变宽，同时全维总分散度下降；两种表现并不矛盾。</p>')
    text.append(picture('complete_inference.gif','图 3：固定展示 seed 0 的完整 33 帧推理。上排显示全部 32 个力子，下排显示全部 256 个样本；仅前 16 个索引绘制历史线以便辨认。每个面板的 PCA 基和坐标范围在全过程固定。不同面板的投影基不同，不能直接比较距离。蓝/橙表示径向系数的吸引/排斥符号，不代表合力方向。'))
    text.append('<h2>复核、适用范围与后续研究</h2>')
    text.append('<p>完整生成器的新增记录器与原积分器在 sanity 输入上的终点逐元素一致。K32 与 K128 的有效维数曲线在三个主任务 seed 0 上的相关系数为 0.9997、0.9992、0.9978；但终点相对 RMS 差异仍约为 4.7%、5.6%、10.7%，所以不能把细化检查当成连续解的精确证书。其余 HAR 活动只检查了一个种子：Walking、Upstairs、Downstairs、Sitting、Standing 的终点有效维数约为 2.7–3.8，Lying 约为 169.6，表明现象有条件依赖。</p>')
    classrows=[]
    classes=['Walking','Upstairs','Downstairs','Sitting','Standing','Lying']
    for r in rm:
        if r['dataset']=='har':classrows.append([classes[r['condition']],f"{r['coarse_final_effective_dimension']:.2f}",f"{r['fine_final_effective_dimension']:.2f}",f"{r['effective_dimension_curve_correlation']:.4f}"])
    text.append(table(['HAR 活动（仅 seed 0）','K32 终点有效维数','K128 终点有效维数','全过程曲线相关'],classrows))
    text.append('<p><b>论文中可采用的表述：</b>“DHPF 的已训练共享力场在推理过程中产生了可复现的样本协方差谱集中与协同旋转；这些动态结构随任务和条件改变，其强度不单调对应生成质量。”当前不宜写成“发现了力子自组织涌现”或“存在相变”。</p>')
    text.append('<p>若继续研究，优先比较相同终点目标下直接 MLP 加速度与 DHPF 的全过程指标，检验这些现象是否具有框架特异性；随后在不同初态随机种子和更多数据集上复核旋转与谱集中。若要检验相变，需要事先定义可调控制参数、序参量和有限规模检验；仅沿 t 观察一条陡峭曲线不够。</p>')
    text.append('<details><summary>全部主条件的终点指标（五种子均值 ± 标准差）</summary>')
    labels={'full':'完整 DHPF','untrained':'未训练','ballistic':'无力运动','frozen_t0':'冻结场源','radial_only':'仅径向','solenoidal_only':'仅无散'}
    rows=[]
    for d,r in s.items():
        for v,a in r['controls'].items():rows.append([names[d],labels[v],pm(a['sw2']),pm(a['acf_rmse']),pm(a['corr_rmse']),pm(a['power_rmse']),pm(a['effective_dimension'],2)])
    text.append(table(['任务','条件','SW²','ACF RMSE','Corr₀ RMSE','Power RMSE','有效维数'],rows))
    text.append('<p>每个条件使用 256 个样本和固定的 256 个真实测试窗口；SW² 用 128 个固定投影。该诊断样本数与论文主实验不同，不能把此表替换成论文主结果。速度打乱与时间错位各用 99 次技术重排，仅用于机制检查；没有把它们作为新的训练重复或宣称校正后的显著性。</p></details>')
    text.append('<p class="foot">原始全过程状态、逐时刻指标、检查点哈希、协议、脚本与绘图数据均保留在本次独立结果目录。此次工作没有训练更新，也没有修改论文结论。</p>')
    css='body{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;line-height:1.85;color:#243440;background:#f2f4f6;margin:0}main{max-width:1120px;margin:auto;background:white;padding:36px 46px}h1{font-size:27px}h2{font-size:21px;margin-top:34px;color:#244e68}.date,.foot{color:#687782;font-size:14px}.finding{background:#edf3f7;border-left:4px solid #305d7b;padding:18px}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}th,td{border-bottom:1px solid #dbe2e6;padding:9px;text-align:left;white-space:nowrap}th{background:#edf2f5}.tablewrap{overflow-x:auto}img{width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:14px;color:#536572}summary{cursor:pointer;font-weight:bold}details{padding:14px;background:#f7f9fa}@media(max-width:650px){main{padding:18px}h1{font-size:23px}}'
    document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DHPF 完整推理涌现诊断</title><style>'+css+'</style></head><body><main>'+''.join(text)+'</main></body></html>'
    (OUT/'report.html').write_text(document,encoding='utf-8')
    write_json(OUT/'report_metadata.json',dict(language='zh-CN',self_contained=True,training_updates=0,primary_checkpoints=15,primary_seeds=5,additional_conditions_seed0=5,report_bytes=len(document.encode('utf-8'))))
    print('Chinese report written:',OUT/'report.html')

if __name__=='__main__':main()
