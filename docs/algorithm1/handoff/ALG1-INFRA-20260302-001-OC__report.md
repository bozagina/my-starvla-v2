# Algorithm1 全量工作汇报（重点覆盖 EXP002 / EXP003）

- 文档版本：v2（详细版）
- 汇总人：OC
- 汇总时间：2026-03-02（Asia/Shanghai）
- 对应汇总任务 EXP_ID：ALG1-INFRA-20260302-001-OC
- 重点覆盖实验：
  - ALG1-MASK-20260301-002-OC
  - ALG1-MASK-20260301-003-OC

---

## 0. 本文目的与范围

本文面向“后续整理/工作汇报/技术复盘”三类场景，目标是：

1. 把算法框架原理讲清楚（从输入到 loss 到推理闭环）。
2. 把为什么这样尝试讲清楚（每个尝试背后的假设与决策理由）。
3. 把尝试结果讲清楚（证据、指标、结论、保留风险）。

证据来源仅来自现有文档与已拉取远程结果：

- `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/mask_diagnosis_full_history.md`
- `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/progress_live.md`
- `/Users/bazinga/code/my-starvla/docs/algorithm1/path_a_llm_review_packet.md`
- `_remote_runs/*` 下 run 的 `config.yaml / metrics.jsonl / train.log / run_identity.txt`

说明：本项目是“本地开发、远程训练”，是否生效以远程 run 证据为准。

---

## 1. 我们的目标（项目级）

### 1.1 双目标

在 Path-A causal feedback 管线上同时达成：

- 目标 A：任务性能不退化（理想为提升）。
- 目标 B：soft-mask 选择性和可解释性增强（不再高熵近均匀）。

### 1.2 关键衡量指标

- mask 选择性：
  - `soft_mask_entropy`（越低越尖锐）
  - `soft_mask_topk_mass_32`（越高越集中）
  - `soft_mask_alpha_max_mean`（越高峰值越突出）
  - `soft_mask_attn_top1_mean_vis`
- feedback 有效性：
  - `feedback_mask_contrast_term / weighted`
  - `delta_action_alpha`
  - `delta_action_effective_norm_mean`
  - `delta_action_clip_saturation_frac`
- teacher 监督有效性（m_clip）：
  - `soft_mask_teacher_kl`
  - `soft_mask_teacher_loss_weighted`
  - `soft_mask_teacher_alpha_cos`
  - `soft_mask_teacher_confidence_mean`
  - `soft_mask_teacher_neg_*`
- 稳定性：
  - `debug/health/has_nonfinite`
  - `debug/action_health/has_nonfinite`

---

## 2. 算法框架原理（详细）

> 本节重点解释“我们到底在训练什么，信号如何流动，Path-A 如何进入动作预测”。

### 2.1 整体结构：双时刻感知 + 反馈动作修正

训练阶段使用两个时刻的视觉输入（`t` 与 `t+k`）构造“变化信号”，再把变化信号注入动作模型。

粗流程：

1. `t` 时刻输入经过 VLM，得到 task token / vision token / geometric token。
2. `t+k` 时刻输入再走一次（训练里通常 no-grad 抽取对照 token）。
3. 根据 token 变化构造残差信号（Path-A）。
4. soft-mask 对 token 变化进行筛选（强调与语言任务相关的 token）。
5. 将 feedback token 送入 action head：
   - 作为上下文条件参与 cross-attn；
   - 或通过 delta-action 分支作为受限纠偏项叠加到动作输出。
6. 联合主任务 loss 与辅助监督训练。

### 2.2 核心张量定义与形状

- `task_tokens`: `[B, K, H]`（常见 `K=32`, `H=4096`）
- `task_tokens_next`: `[B, K, H]`
- `action_chunk`: `[B, T, A]`（常见 `A=7`）
- `action_context`: `[B, A]`（由 prefix 聚合）
- `feedback_tokens`: `[B, Kf, H]`（常见 `Kf=4`）
- `context_task_tokens`: `[B, K+Kf, H]`

Path-A 的关键是让 `feedback_tokens` 对动作预测提供“下一步该怎么修正”的条件信息。

### 2.3 soft-mask 的数学流程（当前主实现）

输入：

- `language_queries q: [B, Lq, H]`
- `vision_tokens v: [B, Nv, H]`
- `geometric_tokens g: [B, Ng, H]`

处理流程：

1. 对 token 长度做对齐（最新版本支持范围保真重采样，而非单纯前缀截断）。
2. 按头拆分到 `num_heads`。
3. 计算 query-token 相似度：
   - `logits_vis = <q, v> * scale`
   - `logits_geo = <q, g> * scale`
4. 在 token 维 softmax，得到 query->token 分布。
5. query 聚合（如 `max`）+ head 聚合（如 `max`）。
6. channel 融合（`vision_only` / `geo_only` / `mean` / `mul` 等）。
7. 最终归一化得到 `alpha: [B, N]`。

应用方式：

- `alpha` 扩为 `[B, N, 1]`，与残差 token（如 `delta_tokens [B, N, H]`）逐 token 相乘。
- `alpha` 越尖锐，代表模型越聚焦少量关键 token。

回退机制：

- `force_uniform_mask=True` 或 soft-mask 构建失败时，回退为 `1/N`。

### 2.4 Path-A 注入到动作输出的机制

在 action head 中，反馈信息通过两条路径影响动作：

1. 上下文拼接：`context_task_tokens = cat([feedback_tokens, task_tokens], dim=1)`。
2. delta-action 残差：
   - `a_out = a_base + gate * clip(DeltaA(feedback))`
   - `clip` 用 `tanh` 限幅，`gate` 与调度/有效性相关。

这套设计意图是：

- 保留基础动作预测稳定性（`a_base`），
- 让反馈只做“有界纠偏”，避免无约束扰动。

### 2.5 损失函数组成（概念层）

总目标可理解为：

- `L_total = L_action + L_feedback_aux + L_mask_contrast + L_teacher(+...)`

其中：

- `L_action`: 主动作损失（核心优化目标）。
- `L_feedback_aux`: Path-A 辅助重构与方向项（约束反馈表示有意义）。
- `L_mask_contrast`: mask 对比项（增强有掩码与无掩码行为差异）。
- `L_teacher`（m_clip）：`KL(m_clip || alpha_pred)`，给 alpha 直接外部软监督。

### 2.6 m_clip teacher 监督原理

动机：仅靠动作损失对 mask 是“间接监督”，容易得到“可用但平坦”的局部最优。

做法：

1. 用 SigLIP 文本-视觉检索分布作为 teacher soft label（`m_clip`）。
2. 预测的 `alpha_pred` 与 `m_clip` 做 KL。
3. 通过 `confidence_floor` 等门控控制 teacher 样本权重。
4. 同时记录负控（shuffle text）指标验证 teacher 是否有真实区分能力。

核心价值：让 mask 本身有了直接训练目标，而不只通过动作 loss 间接学到。

### 2.7 训练-分析协作框架（实验追踪）

- EXP_ID 统一规范：`ALG1-<MODULE>-<YYYYMMDD>-<SEQ>-<OWNER>`。
- 若改训练 YAML，要求把 EXP_ID stamp 到 `run_id`。
- fetch 脚本输出 `run_identity.txt`，将远程 run 与本地改动强绑定。
- 所有重大动作写入 `progress_live.md`，形成可追溯证据链。

---

## 3. 为什么“先排 tower，再转监督”

### 3.1 初始疑问

日志里长期出现 SigLIP fallback，直觉上容易怀疑“视觉塔错配导致 mask 偏平”。

### 3.2 为什么必须先验证这个假设

如果根因真是 tower mismatch，那么继续做 loss/结构调参会在错误方向消耗大量算力。

### 3.3 实际验证与结论

通过 `tools/diagnose_mask_vision_compare.py` 做三分支对比：

- current
- llava_rebuild（官方 tower + 当前 projector）
- official ckpt（image-only 兼容 patch）

关键结果：

- `current_vs_llava_alpha_diff.alpha_cosine_mean ≈ 0.995620`
- `current_vs_official_alpha_diff.alpha_cosine_mean ≈ 0.995694`

结论：tower 来源替换对 alpha 几乎无影响。主矛盾转向“监督不足/目标可辨识性弱”。

---

## 4. ALG1-MASK-20260301-002-OC 详细复盘

> 该实验族的主线是：在可追溯前提下验证 directed 路径与 m_clip 监督是否真正有效。

### 4.1 阶段 A：运行可追溯与基线对齐

尝试内容：

- 清点 stash，确认 mask 相关 patch 集。
- 分析分支差异，确定 `codex/v4-1-delta-action-head` 为主分支。
- 修复 fetch 脚本对 EXP-suffixed 目录匹配，避免抓错 run。

为什么这样做：

- 如果 run 映射错误，后续全部结论都会失真。

结果：

- canonical run 与 EXP_ID 对齐恢复。
- 后续分析都基于 `run_identity.txt` 对齐的真实 run。

### 4.2 阶段 B：directed_self_cross 结构尝试

尝试内容：

- 对比 `similarity` vs `directed_self_cross`（matched early steps）。

为什么这样做：

- 结构改造（定向自注意 refinement）是潜在提升 mask 选择性的直观方向。

结果（早期窗口 19..199）：

- directed 路径确实启用成功（运行时指标非零）。
- 但选择性指标未提升（甚至略弱）：
  - entropy 略高
  - topk_mass_32 略低
  - attn_top1_vis 略低
- action_dit_loss 略有优势。

结论：

- directed 结构“可用但未显著改善选择性”，不能单独作为突破口。

### 4.3 阶段 C：接入 m_clip teacher 监督

尝试内容：

- 新增 teacher 配置与 KL 监督。
- 新增可靠性脚本 `validate_mclip_reliability.py`。

为什么这样做：

- 前面证据显示 mask 问题更像监督弱，需要给 alpha 直接目标。

结果（第一次，floor=0.02，run `...133009...`）：

- `teacher_ready=1`, `teacher_active=1`，但
- `sample_weight_mean=0`, `teacher_kl=0`, `teacher_loss_weighted=0`。

结论：

- teacher 分支“接线正确但门控关断”（非代码崩坏）。

### 4.4 阶段 D：修正 teacher 门控并复核

尝试内容：

- 将 `soft_mask_teacher_confidence_floor` 调到 `0.002`。

为什么这样做：

- 观测到 confidence 量级只有 ~0.005，原 floor=0.02 过高导致全样本被筛掉。

结果（run `...135730...`）：

- `sample_weight_mean=1.0`
- `teacher_kl≈0.089~0.091`
- `teacher_loss_weighted≈0.0018`
- `teacher_alpha_cos≈0.917~0.919`
- 稳定性正常（nonfinite=0）

结论：

- m_clip 监督从“无效”变为“有效贡献”。
- 新问题转为 teacher 分离度仍偏弱（如 `neg_top1_gap` 小）。

### 4.5 EXP002 总结

EXP002 的核心产出不是“最终最优指标”，而是三件决定后续方向的事实：

1. 实验追溯链打通（避免错 run 误判）。
2. directed 路径可用但并非主要增益来源。
3. m_clip teacher 已可稳定生效，监督增强路线成立。

---

## 5. ALG1-MASK-20260301-003-OC 详细复盘

> 该实验族主线：系统排查“为何仍低选择性”，找到并修复高概率实现问题，再做结构与诊断增强。

### 5.1 阶段 A：系统低选择性代码审计

尝试内容：

- 审计 soft-mask 计算细节。
- 重点检查 `soft_mask_score_norm=l2_only` 的归一化位置。

为什么这样做：

- EXP002 证明监督路线可行，但选择性提升仍受限，怀疑存在实现尺度问题“压平 logits”。

发现：

- 旧逻辑在 head split 前做 L2。
- 这会让每个 head 的有效幅度被缩小，导致注意力分布更平。

证据：

- 合成检验里 per-head 正确归一化的 logits std 约为旧方式的 4 倍（与 4 heads 对应）。

修复：

- 改为 head split 后对 `qh/vh/gh` 分别归一化（per-head L2）。

### 5.2 阶段 B：修复后远程 run 验证（run `...153623...`）

为什么这样做：

- 必须在远程对照下确认“不是偶然波动”，且不能牺牲动作主指标。

结果（对照旧 run `...135730...`，common steps `19..1139`）：

- `soft_mask_logits_std_vis`: `0.0406 -> 0.4165`（约 10.4x）
- `soft_mask_attn_top1_mean_vis`: `0.00232 -> 0.00657`
- `soft_mask_topk_mass_32`: `0.0712 -> 0.1105`
- `soft_mask_alpha_max_mean`: `0.00247 -> 0.00515`
- `soft_mask_entropy`: `6.2368 -> 6.1982`
- `action_dit_loss` 均值近似持平

进一步到 step `6499`：

- `soft_mask_entropy=6.1545`
- `soft_mask_topk_mass_32=0.1316`
- `soft_mask_alpha_max_mean=0.00567`
- `action_dit_loss=0.1055`

结论：

- 选择性提升显著且可持续，不是短期噪声。
- 该修复是当前最关键已验证增益点。

### 5.3 阶段 C：token-length 对齐修复

尝试内容：

- 将 `token_n=min(Nv,Ng)` 下的 prefix 截断替换为范围保真重采样。
- 同步修复 soft-mask 路径与 residual token-delta 路径。

为什么这样做：

- prefix 截断可能破坏跨模态覆盖，导致“只看前段 token”的偏置。

结果：

- 代码与诊断已接入（新增 mismatch/resample 指标）。
- 早期 EXP003 run（`...230839...`）中 `Nv==Ng==512`，暂未触发 mismatch 分支，尚不能评价该修复收益。

### 5.4 阶段 D：coverage 诊断增强

尝试内容：

- 新增 `alpha` 的前后半区、前后四分之一区域质量统计。

为什么这样做：

- 需要低成本监测是否仍存在前缀偏置，即使 mismatch 分支没被触发也能看 coverage 倾向。

早期结果（`...230839...` step<=199）：

- first_half≈0.5085, second_half≈0.4915
- front_quarter≈0.2577, rear_quarter≈0.2521

解读：

- 目前未见极端前缀挤压；但窗口过早，不能下最终结论。

### 5.5 阶段 E：EXP003 运行态确认

- `run_identity.txt` 正确映射到 `ALG1-MASK-20260301-003-OC`。
- 新增诊断字段已落地到 metrics。
- 稳定性正常（无 nonfinite / traceback）。

---

## 6. 关键尝试“为什么做 -> 结果如何”总表

| 尝试 | 为什么做 | 结果如何 | 决策 |
|---|---|---|---|
| Tower 替换诊断（current/llava/offical） | 排除“视觉塔错配”主因 | alpha 几乎不变（cos≈0.995~0.996） | 不再作为主攻方向 |
| directed_self_cross | 结构上增强 token 关系建模 | 早期窗口选择性未提升 | 保留但不作为唯一突破 |
| m_clip teacher 接入 | 给 mask 直接监督 | 首次门控关断（KL=0） | 定位为阈值问题 |
| teacher floor 下调到 0.002 | 解决 confidence 量级失配 | KL/weighted loss 非零，分支生效 | 继续优化 teacher 质量 |
| per-head L2 归一化修复 | 排查实现尺度压平问题 | 选择性指标显著持续提升 | 作为当前关键保留改动 |
| token 对齐 range-resample | 降低前缀截断偏置风险 | 代码已生效，待 mismatch 场景验证 | 持续观察 |
| coverage 诊断 | 监控分布覆盖偏置 | 早期未见极端偏置 | 长窗口持续跟踪 |

---

## 7. 当前阶段结论（可用于汇报）

1. 我们已把问题从“疑似视觉塔错配”准确收敛到“监督与实现细节”。
2. EXP002 完成了“可追溯链 + teacher 有效化 + 方向校准”。
3. EXP003 给出了当前最强实证增益：per-head L2 修复显著提升 mask 选择性且未见明显任务回归。
4. 接下来不是“再证明能不能跑”，而是“在保持当前增益基础上，压低后段 clip 饱和并提高 teacher 分离度”。

---

## 8. 仍需重点跟踪的问题

1. `delta_action_clip_saturation_frac` 后段偏高（可能限制反馈有效幅度）。
2. teacher reliability 中 `confidence_mean` 与 `neg_top1_gap` 仍偏弱。
3. token mismatch 分支在部分 run 尚未触发，需要更多样本覆盖验证。

---

## 9. 下一步建议（按优先级）

### P0（最近一轮）

1. 在保持 per-head L2 与 teacher floor=0.002 不变前提下，做最小 CFG A/B 降低 clip 饱和。
2. 固定窗口（2k/5k/10k）输出统一快照，持续验证 selectivity 与 action 是否同步健康。
3. 每次 fetch 后跑 `validate_mclip_reliability.py`，做标准化 PASS/FAIL 比较。

### P1（随后）

1. teacher 小步校准（temperature / smoothing）以提升 `neg_top1_gap`。
2. 若 mismatch 指标被触发且仍有偏置，再推进 blockwise alignment。

### P2（汇报工程化）

1. 固化“周报模板 + 自动指标对照脚本”，减少人工整理成本。
2. 输出短版（管理汇报）+ 长版（研发复盘）双文档。

---

## 10. 关键 run 与对应用途索引

- `..._20260301_111316__ALG1-MASK-20260301-002-OC`
  - similarity 基线（EXP002 对照基准）
- `..._20260301_113908__ALG1-MASK-20260301-002-OC`
  - directed_self_cross 早期验证
- `..._20260301_133009__ALG1-MASK-20260301-002-OC`
  - teacher 首次接入（门控关断案例）
- `..._20260301_135730__ALG1-MASK-20260301-002-OC`
  - teacher floor 调整后生效验证
- `..._20260301_153623__ALG1-MASK-20260301-002-OC`
  - per-head L2 修复后的关键增益 run（已扩展至 step 6499）
- `..._20260301_230839__ALG1-MASK-20260301-003-OC`
  - EXP003 新增对齐/coverage 诊断加载确认（早期快照）

---

## 11. 对外汇报建议话术（可直接使用）

- 我们已经完成根因分层：先证伪 tower 主因，再定位到监督与实现尺度问题。
- 目前最有效改动是 per-head L2 修复，已在远程长窗口显示稳定选择性收益。
- m_clip teacher 从“启用但无梯度”推进到“稳定非零监督”，方向正确。
- 下一阶段重点是“稳住已有收益 + 解决后段饱和与 teacher 分离度”。

