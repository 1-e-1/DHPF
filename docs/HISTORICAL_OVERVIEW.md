> Historical overview. Refer to the root README for the current missing-artifact status.

# DHPF V3 experimental workspace

## 当前任务：执行论文实验清单（2026-10-13）

本轮按 `DHPF_V3_论文实验清单.docx` 的 E01–E12 执行，并按用户补充要求加入 Drifting。正式代码在 `benchmark/`，数据在 `data/checklist_v1/`，结果在 `results/checklist_v1/`。下文历史 41 次试验不计入本轮正式结果。

- 实时交付进度：[report.html](results/checklist_v1/report.html)；机器可读状态：[checklist_status.json](results/checklist_v1/checklist_status.json)。启动、训练完成、评价完成与全套完成分别记录。
- 固定协议：[protocol.json](results/checklist_v1/protocol.json)。主实验和关键消融均使用五个训练种子；超参数搜索只使用验证集；每次运行保留配置、代码快照、完整日志和 checkpoint。
- 正式队列共 390 次训练，另有等预算超参数搜索。E11 为 36 次单种子补充拟合诊断，不能作为五种子优势结论。
- Drifting 采用公开漂移目标的原始坐标适配，标为 **Drifting-raw**；不宣称复现其依赖特征编码器的完整图像系统。所有方法在固定预算结束后均明确标注尚未证实收敛。
- CPU / CUDA 续跑任务的进程与状态分别见 `worker_cpu.json`、`worker_cuda.json`；训练和后处理失败保存在 `errors/`，不会静默从汇总删除。

刷新汇总：`python -m benchmark.report`。若进程已结束，可按原配置继续：`python -u -m benchmark.worker --device cpu` 和 `python -u -m benchmark.worker --device cuda`。不要在同一设备的现有 worker 仍运行时重复启动；不要通过改写原配置来覆盖已完成运行。

## 实验定位修正（2026-10-13）

根据当前研究要求，DHPF 应作为生成算法/训练框架，与 GAN、DDPM、CFM 和 Drifting Models 做算法级比较；具体神经网络骨干和规模允许按任务选择。下列 41 次试验只作为既定小网络配置的实现检查、数值诊断和框架内部消融，不能作为 DHPF 框架对其他生成算法的性能评价。后续主实验应在验证集上分别选择各方法的合适网络与超参数，并比较生成质量、训练成本和采样成本。

## 历史实现检查与内部消融（2026-10-13）

已实际完成 **41 次正式训练**：36 次主对照、2 次细步长训练、2 次长训练、1 次场源分离诊断。另有 4 次各 10 更新的 CPU/GPU 热身，不计入正式结果。所有结果均保留，未挑选最优种子。

**目前证据：模型可训练，结构与梯度检查通过，但这组配置尚未拟合好细粒度分布，也未显示稳定优于直接加速度 MLP。** 这不是模型表达能力不成立的证明，也不是经过充分调参后的性能上限。

### 框架内对照：800 次更新、K=16、3 个种子

下表为独立 2048 样本、256 个投影方向上的 squared sliced-Wasserstein（SW2²），均值 ± 样本标准差，越低越好。完整原始数据在 [all_runs.csv](results/main/all_runs.csv)，汇总在 [summary.csv](results/main/summary.csv)。

| 数据分布 | 完整 DHPF | 仅径向 | 仅切向 | 二阶 MLP |
| --- | --- | --- | --- | --- |
| 八峰高斯 | 0.0409 ± 0.0014 | 0.0727 ± 0.0168 | 0.0828 ± 0.0034 | 0.0470 ± 0.0060 |
| 双环 | 0.0254 ± 0.0068 | 0.0238 ± 0.0037 | 0.0673 ± 0.0062 | 0.0181 ± 0.0050 |
| 双螺旋 | 0.0196 ± 0.0106 | 0.0176 ± 0.0072 | 0.0548 ± 0.0060 | 0.0190 ± 0.0036 |

完整 DHPF 相比同初态的无力自由运动基线有所改善，但不能仅由 SW2² 判断分布细节已恢复：八峰高斯中落在任一真实中心半径 0.36 内的比例平均仅 33.48%；双环中距任一圆环小于 0.18 的比例为 33.38%；双螺旋中距解析曲线小于 0.18 的比例为 48.01%。目标数据对应比例约为 99%。这些是几何邻域精度指标，不是完整的 precision/recall 度量。

主对照的场源模型为 36,933 个参数，MLP 为 35,026 个参数。当前 CPU 配置下，完整 DHPF 每次 800 更新约 22 秒，MLP 约 8 秒；本实验未显示训练速度收益，也不据此推断 GPU 或高维性能。主对照采用相同更新次数，不是相同计算预算。

![主对照：固定展示 seed 0，而非最优种子](results/main/comparison.png)

![训练目标曲线：3 个种子均值，40 步移动平均](results/main/training_curves.png)

### 连续模型与离散模型存在差异

同一主对照模型与同一初态，采样从 K=16 改为 K=64：

| 分布 | DHPF SW2²：16 → 64 步 | MLP SW2²：16 → 64 步 | DHPF 终点差异 RMSE |
| --- | --- | --- | --- |
| 八峰高斯 | 0.0409 → 0.0406 | 0.0470 → 0.0485 | 0.2603 |
| 双环 | 0.0254 → 0.0531 | 0.0181 → 0.0188 | 0.2815 |
| 双螺旋 | 0.0196 → 0.0493 | 0.0190 → 0.0190 | 0.2507 |

K=16 的部分低损失依赖离散过程，不能直接当作连续 ODE 的生成效果。另对每个模型/数据集的固定 seed 0 用 512 个相同初态运行 K=16/32/64/128/256，结果保存于 [diagnostics.json](results/main/diagnostics.json)。256 步只是更细的数值参照，并非解析真值；这些 512 样本指标不可直接与上面的 2048 样本指标混合比较。

补充试验在双环上将训练 K 改成 64，保留 800 更新、相同 seed 0 和数据序列：DHPF 在 K=64/256 采样的 SW2² 为 **0.01785 / 0.01788**，终点 RMSE 为 0.02069。步长敏感性明显减小，但圆环邻域精度仍只有 31.74%；同配置 MLP 的 SW2² 为 0.01570 / 0.01571。见 [细步长试验](results/fine_training/summary.csv)。该补充试验只有一个种子，不作统计显著性结论。

### 延长训练与场源聚集诊断

八峰高斯、seed 0、K=16、相同数据序列，从头训练 4000 更新后：

| 模型/条件 | K=16 SW2² | K=64 SW2² | 峰邻域精度 |
| --- | --- | --- | --- |
| DHPF，分离权重 0 | 0.03086 | 0.03652 | 37.45% |
| MLP | 0.03413 | 0.03373 | 37.65% |
| DHPF，分离权重 0.01 | 0.04329 | 0.05316 | 28.27% |

前两个模型的八峰仍主要呈圆环状。[长训练结果](results/long_training/summary.csv)。

关闭分离正则的长训练中，按 256 个阶段时刻统计，场源对距离中位数仅 **0.01771**，平均核宽度 **1.01746**。开启 V3 可选分离项（lambda=0.01、epsilon=0.05、每步独立采样 4 个时间点）后，场源对距离中位数增至 **0.51717**，但本次生成指标反而变差。因此“把场源分开就能恢复八个峰”的简单解释未获支持；此单种子试验也不能说明所有分离权重均无效。[分离试验](results/separation/summary.csv)。

![无分离正则：实际学习到的场源、样本路径和加速度场](results/long_training/gmm8__full__seed0/learned_dynamics.png)

![开启分离正则：场源更分散，但生成质量没有提升](results/separation/gmm8__full__seed0/learned_dynamics.png)

### 已完成的验证与下一阶段问题

无旋/无散误差在 float64 下约 1e-18；OT 穿过积分器的参数梯度与中心有限差分相对误差约 2.87e-10；批次独立性误差为 0。解析振子上显式中点法观测阶为 1.993、1.999。主结果完整性检查验证了所有 36 个组合、日志长度、有限值、源码快照 SHA256，以及各模型评估初态和目标样本完全一致。见 [数学检查](experiments/check_math.py)、[完整性检查结果](results/main/integrity_checks.json)。

下一阶段应优先分开研究：批量 OT 对细粒度多峰学习的影响、核宽度/场源初始化和时间编码频率的影响，以及较细积分下的训练效果。应加入标准 Flow Matching 或其他已知能拟合这些分布的参照，以检查实验任务与训练预算；本轮尚未包含这些参照，也没有进行高维试验。不要将当前指标直接写成论文中的性能优势结论。

原始两份论文未修改。检查点、配置、训练记录、独立样本与图均在各结果目录中。

## Implementation and initial protocol

This repository implements the supplied `DHPF_V3_基础论文_数学修订版.docx` as a first, unconditional 2-D experiment. It contains no robotics task or physical-time dynamics assumptions.

## Protocol fixed before the main run

- Data: online i.i.d. samples from 8 isotropic Gaussian modes, two concentric noisy rings, and two noisy spiral arms. Exact formulas are in `experiments/dhpf_experiment.py`; no finite training set is reused for evaluation.
- Generator input: independent standard-normal position and isotropic half-normal-radius velocity. Output: position at generation time 1; terminal velocity is free.
- Models: full DHPF, radial-only DHPF, tangential-only DHPF, and a direct position/time acceleration MLP. All are velocity-independent second-order systems.
- DHPF: 32 shared time-dependent sources, width 64, 2 residual blocks, 8 fixed Fourier frequencies, source coordinate bounds ±3.5, alpha/beta bounds ±2, kernel width in (0.25, 1.5), fixed kernel exponent 1.5. This is a smaller explicitly documented configuration than the draft's illustrative width 128/depth 3.
- Direct MLP: two hidden layers of width 176 with SiLU. Parameter counts are recorded, approximately matched but not identical. Its acceleration output is unrestricted; this structural difference is not concealed.
- Objective: exact minimum-cost permutation between CURRENT GENERATED ENDPOINTS and real samples; squared Euclidean ground cost, averaged over samples. No source-separation, velocity, acceleration or jerk penalties (lambda_sep = 0, permitted by V3).
- Integration: explicit midpoint, K=16 (32 field evaluations per sample). Source parameters for all stage times are batched before integration, preserving the exact time-only source definition.
- Optimization: 800 updates, batch 128, Adam lr=0.001, global gradient norm clipping at 10. Seeds 0, 1, 2. Dedicated data generators ensure identical training samples/prior draws across models for a given dataset and seed on the same device.
- Evaluation: 2048 independent generated and real samples; squared sliced-Wasserstein with 256 fixed directions; exact empirical W2 squared on the first 512 samples; 8-mode coverage and mode precision where applicable. Independent real-vs-real samples establish finite-sample context, not a strict lower bound. A ballistic no-force baseline uses the same initial states.
- Numerical check: resample the SAME initial states with K=64; report endpoint RMSE and sliced-Wasserstein. A coarse/fine discrepancy must be reported, not treated as successful continuous-model fitting.
- Report all 36 runs and across-seed sample standard deviations. No best-seed selection. These are preliminary equal-update experiments, not equal-compute comparisons, exhaustive tuning, or publication-level validation.
- Reinforcement learning Observation / Action / Reward and physical control timing: not applicable.

## Run

```powershell
python experiments/check_math.py
python experiments/dhpf_experiment.py --output results/main --train-steps 800 --device cpu --threads 2
python experiments/summarize.py --root results/main
python experiments/diagnose.py --root results/main
python experiments/verify_results.py

# Targeted follow-ups; seed zero is fixed, not selected by performance.
python experiments/dhpf_experiment.py --output results/fine_training --datasets rings --models full mlp --seeds 0 --train-steps 800 --integration-steps 64 --device cpu --threads 2
python experiments/dhpf_experiment.py --output results/long_training --datasets gmm8 --models full mlp --seeds 0 --train-steps 4000 --integration-steps 16 --device cpu --threads 2
python experiments/dhpf_experiment.py --output results/separation --datasets gmm8 --models full --seeds 0 --train-steps 4000 --integration-steps 16 --lambda-separation 0.01 --device cpu --threads 2
```

Each run saves a configuration, source SHA256, per-step training log, final checkpoint, independent samples, coarse/fine plots and metrics. Existing completed runs are skipped; use a NEW output directory for any changed configuration. Benchmark folders are warm-up checks, not scientific comparisons (CPU/GPU random streams differ).

## Interpretation limits

Minibatch OT is not an unbiased estimate of population W2 squared. A smaller training objective alone is not evidence of better generalization. Model quality must be judged on independent evaluation, mode coverage and refinement checks. Numerical timing is hardware/software/batch dependent, and does not establish broad speed superiority. Analytic structure checks do not prove distributional universality.
