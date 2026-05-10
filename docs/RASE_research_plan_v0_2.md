# RASE 研究文档 v0.2

**项目名**：RASE — Risk-Controlled Advantage Set Extraction  
**版本**：v0.2，基于 RASE Phase-0 结果更新  
**日期**：2026-05-10  
**范围声明**：本文档只讨论 RASE。RASE 是 offline generative / actor-based policy improvement 中的 action-level accepted-set extraction 问题；不包含 PCAR 的 offline-to-online action replacement，不包含 goal-conditioned stitching 的主线算法。

---

## 1. 一句话定位

RASE 研究的问题是：

> 在 offline RL 中，当我们从行为模型、IQL actor 或生成式 policy 中采样大量候选动作，并用离线 critic 选择高 Q 动作时，哪些动作可以被可信地视为 policy improvement，哪些只是 selection-induced false positive？

RASE 的目标不是提出一个新的 online fine-tuning 规则，也不是一开始就训练 flow policy；它首先要建立一个可检验的 offline diagnostic 与 accepted-set extraction 框架。

---

## 2. 与相邻课题的边界

### 2.1 RASE vs PCAR

PCAR 是 offline-to-online fine-tuning 中的 pairwise action replacement：用旧策略动作作为 anchor，只有候选动作被在线校准 margin 证明优于旧动作时才替换。RASE 不使用 online data、old-policy EMA、online margin 或 action replacement。RASE 当前阶段只研究 offline candidate selection 与 accepted set。

### 2.2 RASE vs Goal-Conditioned Certified Stitching

Certified stitching 研究的是 goal-conditioned offline RL 中的 subgoal / trajectory-level stitching：判断 `s -> z -> g` 是否可靠。RASE 当前研究的是普通 continuous-control offline RL 中的 single-state action candidate selection。AntMaze 后续可以作为 stitching-level extension，但不是当前 action-level 主证据。

### 2.3 RASE 与 Value-Calibrated Flow Policy 的关系

Value-Calibrated Flow Policy 的核心问题是“什么时候可以相信 critic 给出的 high-Q 动作并允许生成式 policy 向它移动”。RASE 可以看作该问题的前置诊断层：先证明 high-Q candidate selection 会诱发 optimism / false positives，再学习一个 risk-controlled accepted set，最后再考虑 flow actor distillation。

---

## 3. Phase-0 结果摘要

Phase-0 已完成：

```text
4 environments × 3 seeds × 3 candidate sources × 5 candidate-pool sizes
= 180 sweep rows
```

环境：

```text
halfcheetah-medium-replay-v2
hopper-medium-replay-v2
walker2d-medium-replay-v2
antmaze-umaze-v2
```

候选源：

```text
bc / iql / perturb
```

候选池大小：

```text
M = 1, 4, 16, 64, 256
```

### 3.1 主要发现

Phase-0 最强证据不是 conditional FPI 随 M 单调上升。实际结果显示，conditional FPI 多数下降，因为 M 增大后 predicted-positive coverage 也显著扩大，其中包含更多 FQE-positive 动作。

更稳定的主结论是：

> M 增大时，predicted advantage 上升明显快于 FQE advantage，导致 predicted–empirical gap 系统性扩大。

这支持 RASE 的核心动机：raw max-Q candidate selection 会产生 selection-induced optimism / winner's curse。

### 3.2 M=1 到 M=256 的平均变化

| Candidate source | Δ predicted adv | Δ FQE adv | Δ gap | Δ pred-positive rate | Δ unconditional FPI | Δ conditional FPI |
|---|---:|---:|---:|---:|---:|---:|
| BC | +5.020 | +2.249 | +2.771 | +0.423 | +0.091 | -0.118 |
| IQL | +4.051 | +1.818 | +2.233 | +0.321 | +0.057 | -0.089 |
| Perturb | +1.899 | +0.707 | +1.192 | +0.180 | -0.010 | -0.178 |

Interpretation:

1. BC / IQL candidate sources are the main evidence.
2. Perturb behaves like a conservative negative control.
3. Unconditional false-positive mass increases for BC/IQL even though conditional FPI decreases.

### 3.3 任务级观察

Locomotion replay tasks are the main Phase-0 evidence:

| Environment | BC gap Δ | IQL gap Δ | Perturb gap Δ |
|---|---:|---:|---:|
| halfcheetah-medium-replay-v2 | +3.212 | +2.604 | +1.783 |
| hopper-medium-replay-v2 | +4.099 | +3.028 | +1.458 |
| walker2d-medium-replay-v2 | +3.825 | +3.368 | +1.530 |

AntMaze action-level signal is weak. Its advantage scale is small and fixed global thresholds are inappropriate. AntMaze should be treated as a boundary case or moved to subgoal / trajectory-level RASE later.

---

## 4. Revised hypothesis

旧假设：

> Candidate pool 越大，conditional false-positive improvement rate 越高。

新假设：

> Candidate pool 越大，raw max-Q selection 会系统性放大 predicted–empirical value gap，并扩大 false-positive discovery mass。RASE 的目标是在保持 coverage 的同时控制 accepted-set false-positive risk。

这一修正非常重要。后续实验不再把 conditional FPI monotonicity 作为主判据，而是报告：

```text
predicted advantage vs M
FQE / rollout advantage vs M
predicted–empirical gap vs M
unconditional FPI mass vs M
risk-coverage / precision-at-coverage
```

---

## 5. RASE 方法定义

### 5.1 Candidate set

对每个 dataset state `s`，候选动作来自 proposal distribution：

\[
\mathcal A_M(s)=\{a_1,\ldots,a_M\},\quad a_i\sim q(a|s).
\]

当前 proposal 包括：

```text
BC policy
IQL policy
local perturbation around dataset action
```

后续可加入 flow / diffusion proposal，但 flow 不是当前阶段的必要模块。

### 5.2 Raw selection

\[
a^*_M(s)=\arg\max_{a\in\mathcal A_M(s)} Q_{IQL}(s,a).
\]

### 5.3 Pairwise predicted improvement

\[
\widehat A_{pair}(s,a^*)=Q_{IQL}(s,a^*)-Q_{IQL}(s,a_{data}).
\]

### 5.4 Empirical proxy improvement

当前使用 FQE proxy：

\[
A_{FQE}(s,a^*)=Q_{FQE}(s,a^*)-Q_{FQE}(s,a_{data}).
\]

下一步用 short-rollout diagnostic 检验 FQE proxy 是否可靠。

### 5.5 False-positive policy improvement

\[
FPI(s,a^*)=\mathbb 1[\widehat A_{pair}(s,a^*)>0]\cdot
\mathbb 1[A_{emp}(s,a^*)\le 0].
\]

其中 `A_emp` 可以是 FQE proxy，也可以是 rollout estimate。

### 5.6 RASE accepted set

RASE 不直接接受所有 high-Q actions，而是构造分数：

\[
Z(s,a)=\widehat A_{pair}(s,a)-\lambda_s R_{support}(s,a)-\lambda_u U_Q(s,a)-\lambda_c C_{consistency}(s,a).
\]

当前 v1 只使用：

\[
Z_{v1}=\widehat A_{pair}-\lambda_s\cdot \text{behavior\_NLL}.
\]

v2 代码新增：

```text
support_nll
kNN state-action support distance
IQL twin-Q disagreement
FQE twin-Q disagreement
action L2 distance to dataset action
```

并输出 composite score：

\[
Z_{v2}=\widehat A_{pair}-\lambda_s NLL-\lambda_{knn}D_{knn}-\lambda_q U_{IQL}-\lambda_f U_{FQE}.
\]

该 composite score 目前只用于 proxy alignment，不应在论文中宣称为最终算法。

---

## 6. 下一步实验计划

### Phase 0.5 — FQE label validation

目标：证明 FQE advantage 至少与 simulator short-rollout advantage 同方向相关。

任务优先级：

```text
hopper-medium-replay-v2
walker2d-medium-replay-v2
halfcheetah-medium-replay-v2
```

设置：

```text
candidate sources: BC / IQL
M: 1, 16, 64, 256
states: 256 or 512
rollout horizon: 50
rollout repeats: 3 to 5
continuation policy: IQL, optional BC
```

诊断：

```text
fqe_rollout_corr
pred_rollout_corr
rollout_fpi_rate
pred_rollout_gap_mean
fqe_rollout_gap_mean
```

Go 条件：

```text
FQE-rollout correlation is positive and stable.
M increases predicted-rollout gap.
RASE risk score improves rollout-positive precision at fixed coverage.
```

Stop / revise 条件：

```text
FQE and rollout labels disagree strongly.
FQE false positives are not reflected in rollout diagnostics.
```

### Phase 1 — Cross-fitted critic diagnostic

目标：确认 Phase-0 gap 不是同一 critic 的 in-sample artifact。

设置：

```text
K = 3 folds initially
train fold-specific IQL critics
candidate selection evaluated on held-out fold states
proposal source: BC / IQL
FQE evaluator: full FQE first, fold FQE optional later
```

核心比较：

```text
in-sample sweep vs out-of-fold sweep
M vs predicted gap
M vs FQE gap
M vs predicted–FQE gap
```

Go 条件：

```text
Out-of-fold critics still show selection-induced predicted–FQE gap.
Gap magnitude may shrink, but should not disappear.
```

### Phase 2 — Proxy alignment

目标：判断哪些 risk proxy 可以预测 FQE false positives / FQE-positive improvement。

记录 proxy：

```text
support_nll
kNN state-action distance
IQL twin-Q disagreement
FQE twin-Q disagreement
action L2 to data action
predicted pairwise gap
RASE score v1/v2
```

报告：

```text
AUROC / AUPRC for FQE-positive label
precision@coverage = 0.2 / 0.3 / 0.5 / 0.7
calibrated risk-coverage curve
```

Go 条件：

```text
RASE score improves precision@coverage over raw predicted gap.
At least one support/uncertainty proxy has AUROC > 0.60 on locomotion replay tasks.
```

### Phase 3 — Accepted-set extraction and actor distillation

目标：不再只诊断 selected candidates，而是训练 policy 拟合 risk-controlled accepted set。

候选方法：

```text
RASE-BC: supervised actor trained on accepted actions
RASE-Mixture: mixture policy over accepted candidates
RASE-WeightedBC: weights from calibrated accepted-set score
```

不要马上上 flow。先证明 accepted-set extraction 能改善 offline policy or diagnostic precision。

### Phase 4 — Flow / diffusion extension

只有 Phase 3 成立后再做。

问题：

```text
Does flow improve multi-modal accepted-set fidelity?
Does flow introduce interpolation false positives?
Does flow reduce inference or improve coverage compared with Gaussian / mixture actors?
```

---

## 7. 实验优先级与资源安排

### 最小下一轮

```text
3 envs: hopper / walker2d / halfcheetah medium-replay
3 seeds: 41 / 42 / 43
2 sources: BC / IQL
M: 1 / 16 / 64 / 256
```

运行顺序：

```text
1. Phase-0 sweep reuse existing checkpoints/results.
2. Phase-0.5 rollout diagnostic on BC source.
3. Proxy alignment on BC source.
4. Cross-fitted IQL on BC source.
5. Repeat IQL source only if BC source confirms the mechanism.
```

### 投稿基础版扩展

```text
D4RL locomotion: medium / medium-replay / medium-expert
AntMaze: moved to subgoal-level analysis or reported as limitation
Candidate sources: BC / IQL / flow proposal
Seeds: 5
```

---

## 8. 代码变更摘要

新增模块：

```text
rase/selection.py       reusable nested candidate selection
rase/rollout.py         short-horizon action replacement rollout diagnostic
rase/proxy.py           kNN support distance and proxy alignment
rase/metrics.py         AUROC/AUPRC/precision-at-coverage helpers
rase/crossfit.py        fold split utilities
run_rollout_diagnostic.py
run_proxy_alignment.py
run_crossfit_iql.py
```

新增脚本：

```text
scripts/run_phase05_rollout.sh
scripts/run_phase1_proxy.sh
scripts/run_phase1_crossfit.sh
```

新增输出：

```text
diagnostics/rollout_pairs_<source>_<continuation>.csv
diagnostics/rollout_summary_<source>_<continuation>.csv
diagnostics/proxy_details_<source>.csv
diagnostics/proxy_alignment_<source>.csv
diagnostics/calibrated_risk_coverage_<source>.csv
crossfit/crossfit_sweep_<source>.csv
```

---

## 9. 决策树

### 如果 rollout 支持 FQE

继续 Phase 1/2，把 FQE 作为主 proxy label，同时用 rollout 做少量强诊断图。

### 如果 rollout 不支持 FQE

停止使用 FQE FPI 作为主标签，改为：

```text
short-rollout labels for locomotion
FQE ensemble only as auxiliary proxy
```

### 如果 cross-fit 后 gap 消失

说明 Phase-0 主要来自 in-sample critic overfitting。RASE 应改成 cross-fitted / ensemble verification 方法，而不是直接用 raw IQL critic。

### 如果 proxy alignment 很弱

不要推进 actor distillation。先修 proxy：

```text
try kNN in learned representation
try ensemble IQL/FQE disagreement
try calibrated threshold per environment/source
```

### 如果 accepted-set precision 提升但 coverage 太低

RASE 可以作为 safety filter，但还不能作为完整 policy improvement。下一步要优化 coverage under risk constraint，而不是继续提高 threshold。

---

## 10. 当前论文主张草案

> Generative and actor-based offline policy improvement is often implemented as high-Q candidate selection. However, increasing the candidate pool size induces selection optimism: predicted advantage grows faster than empirical improvement, expanding the mass of false-positive improvements. RASE reframes offline policy improvement as risk-controlled advantage set extraction: instead of accepting every high-Q action, it extracts an accepted action set that optimizes coverage subject to calibrated false-positive risk.

中文表述：

> 生成式 offline RL 不应把 high-Q candidate selection 当成无约束 policy improvement。候选池越大，critic 越容易选中被高估的动作。RASE 将问题重写为 risk-controlled accepted-set extraction：在保持 coverage 的同时控制 false-positive policy improvement。

---

## 11. 当前结论

RASE 可以继续推进，但下一阶段不是直接做 flow policy，而是先完成三件事：

```text
1. rollout 验证 FQE label；
2. cross-fit 验证 selection optimism 不是 in-sample artifact；
3. proxy alignment 验证 support/uncertainty/consistency signal 是否能预测 false positives。
```

只有这三项成立后，才进入 accepted-set actor distillation 和 flow extension。

---

## Appendix A. Phase-0 summary figures

![Overall predicted-FQE gap vs M](/mnt/data/rase_analysis_prev/rase_phase0_analysis/plots/overall_gap_vs_M.png)

![Overall unconditional FPI vs M](/mnt/data/rase_analysis_prev/rase_phase0_analysis/plots/overall_unconditional_fpi_vs_M.png)

![Overall conditional FPI vs M](/mnt/data/rase_analysis_prev/rase_phase0_analysis/plots/overall_conditional_fpi_vs_M.png)

![Risk-coverage at M=256 for BC source](/mnt/data/rase_analysis_prev/rase_phase0_analysis/plots/risk_coverage_M256_bc.png)
