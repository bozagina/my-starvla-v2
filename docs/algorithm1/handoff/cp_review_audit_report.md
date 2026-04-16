# CP-REVIEW 审核报告

```
审核方:     CP-REVIEW 线程
被审核方:   CP-BUILD 线程
审核日期:   2026-04-15
审核规则:   evidence_only
总体裁决:   CONDITIONAL_PASS
```

---

## 1. Artifact Checklist

| # | Artifact | 预期位置 | 存在/完整 | 备注 |
|---|---|---|---|---|
| 1 | config diff — `framework.corrective_flow.*` | `starvla_train_pi_qwen25_corrective_flow.yaml` + CLI 参数（`run_cp_smoke_500step.sh`） | ✅ | YAML 含完整 `framework.corrective_flow` + `trainer.optional_loss_hooks.corrective_flow`；CLI 覆盖所有关键参数 |
| 2 | local smoke evidence | `progress_live.md` 2026-04-15T14:00 条目 | ✅ | AST parse / lint / YAML 检查全部 PASS 记录完整 |
| 3 | remote smoke evidence | `progress_live.md` 2026-04-15T20:55 条目 + `cp_current_round.yaml` `remote_evidence` 字段 | ✅ | run_id=`cp_smoke_500step_20260415_042824`, exit_code=0, 100 entries, mse=0.070034。metrics.jsonl 在远程服务器，本地不可直接验证（符合远程训练边界协议） |
| 4 | changed-file handoff note | `progress_live.md` 14:00 + 20:55 两条目 | ✅ | 全部变更文件逐一列出，BD-1~5 决策记录完整 |

---

## 2. CP-AH Findings（验收假设逐项判定）

### CP-AH-1（原始）：trigger=0 → region_loss=0

| 项 | 内容 |
|---|---|
| BUILD 声明 | PASS — `loss/region=0` in 38/100 steps；`debug/corrective_loss_region_trigger_filtered_count` avg=6.2 |
| 证据路径 | `progress_live.md` 20:55 条目；`cp_current_round.yaml` `acceptance_hypotheses.CP-AH-1` |
| 代码验证 | `QwenPI.py` L315-316 的 P0 门控（`trigger_label > 0.5` 显式 mask）在本模块审查范围内 |
| **CP-REVIEW 判定** | **PASS** — 38/100 步 `loss/region=0`，与 P0 激活率相符；debug counter 有效 |

### CP-AH-1-ext（诊断级）：CorrectiveFlowHead 隐式 P0

| 项 | 内容 |
|---|---|
| BUILD 声明 | PASS（诊断级）— `loss/corrective_flow_region` 前半段 0.077 → 后半段 0.037（↓52%） |
| RES 结论 | 采纳方案 B（保持 loss 逻辑不变）；CorrectiveFlowHead 不需要显式 P0 gate；trigger=0 样本经 base_loss 学习"零 velocity" 是正向正则 |
| 代码验证 | `corrective_flow_head.py` L186-190 确认无显式 P0 mask，方案 B 已实施 |
| debug metrics | `decomposed_metric_specs` 含 `corrective_flow_base_loss → loss/corrective_flow_base` 和 `corrective_flow_region_loss → loss/corrective_flow_region`，已在 `train_starvla.py` L1466-1467 确认 |
| **CP-REVIEW 判定** | **PASS（诊断级）** — RES 方案 B 分析逻辑严密；下降 52% 与隐式 P0 理论一致；debug metrics 已暴露可持续监控 |

### CP-AH-2：所有 loss 分量 finite 且非退化

| 项 | 内容 |
|---|---|
| BUILD 声明 | PASS — 12 个 loss 分量 500 步全部 finite，无 NaN/Inf |
| 证据 | `progress_live.md` 20:55：`loss/corrective_flow [0.082, 0.377]`，`loss/corrective_flow_base [0.074, 0.301]`，`loss/corrective_flow_region [0.017, 0.155]`；`loss/total` 12.19 → 1.60 (↓87%) |
| metrics.jsonl 状态 | 在远程服务器，100 条记录，exit_code=0；本地无法直接抽检——接受 BUILD 的 progress_live.md 文档证据 |
| **CP-REVIEW 判定** | **PASS** — 数值范围合理，下降趋势明确；三类 corrective_flow 分量均 finite 且非退化 |

### CP-AH-3：loss/consist finite 且下降趋势

| 项 | 内容 |
|---|---|
| BUILD 声明 | PASS — `loss/consist` 前半段 avg=9.3e-5 → 后半段 avg=5e-6（↓95%）；最后 5 步全为 0.0 |
| 证据 | `progress_live.md` 20:55 条目 |
| **CP-REVIEW 判定** | **PASS** — 95% 下降且末尾归零，P1 consist 约束有效收敛 |

### CP-AH-4：Success@LIBERO

| 项 | 内容 |
|---|---|
| BUILD 声明 | `BLOCKED_WAIT_REMOTE` — 需 LIBERO eval |
| **CP-REVIEW 判定** | **正确阻塞声明** — BUILD 未声明 PASS，符合执行纪律。AH-4 不计入本轮裁决 |

---

## 3. Freeze Constraint Verification

| 文件 | 是否被修改 | 验证方式 | 结论 |
|---|---|---|---|
| `starVLA/model/framework/a_module_interface.py` | 否 | `git diff HEAD` 返回空；最近 commit 为 `1f75f45`（pre-CP-BUILD） | ✅ FROZEN |
| `starVLA/model/modules/loss/optional_loss_utils.py` | 否 | `git diff HEAD` 返回空；最近 commit 同上 | ✅ FROZEN |
| `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py` | 否（仅 import） | `git diff HEAD` 空；CP-BUILD commit `69ced04` 不含此文件 | ✅ FROZEN |
| `starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py` | 否（仅 import） | 同上 | ✅ FROZEN |

---

## 4. Code Findings

### F-1 [MEDIUM] — `predict_action()` dtype 恢复逻辑泄漏

**文件**: `starVLA/model/framework/QwenPI.py`，predict_action 的 corrective_flow 块

**问题**:

```python
try:
    ami = self.a_module_interface
    ami_is_module = isinstance(ami, nn.Module)
    if ami_is_module:
        orig_ami_dtype = next(ami.parameters()).dtype
        ami.float()          # ← 转换为 float32

    self.corrective_flow_head.float()    # ← 转换为 float32

    a_predictions_cf = ami.predict(...)  # ← 若此处抛出 RuntimeError
    ...
    if ami_is_module:
        ami.to(dtype=orig_ami_dtype)     # ← 异常时跳过，ami 留在 float32
    self.corrective_flow_head.to(dtype=orig_cf_dtype)  # ← 同上
except RuntimeError as e:
    logger.warning(...)   # ← 无恢复逻辑
```

如果 `ami.predict()` 抛出 `RuntimeError`，`ami` 和 `corrective_flow_head` 将保持 float32，产生状态泄漏。后续 inference 调用时 `orig_ami_dtype` 会被错误地记录为 float32，无法自动恢复至 bf16。

**影响**: 生产 inference 时若遇 RuntimeError，可能导致模型持续以 float32 推理，增大显存占用。当前 500-step smoke 训练未触发此路径，因此不影响本轮验收结果。

**修复建议**:

```python
try:
    ...
    a_predictions_cf = ami.predict(...)
    ...
except RuntimeError as e:
    logger.warning("Corrective flow inference skipped (dtype): %s", e)
finally:
    if ami_is_module and 'orig_ami_dtype' in dir():
        ami.to(dtype=orig_ami_dtype)
    if 'orig_cf_dtype' in dir():
        self.corrective_flow_head.to(dtype=orig_cf_dtype)
```

---

### F-2 [MEDIUM] — 参考配置 `a_module.mode` 与实际 smoke train 不一致

**文件**: `starVLA/config/training/starvla_train_pi_qwen25_corrective_flow.yaml`

**问题**: 参考配置第 46 行设置 `a_module.mode: standalone`，但实际 500-step smoke train 使用 `--framework.a_module.mode lite`（见 `run_cp_smoke_500step.sh` L45）。

参考配置意图是文档化"生产级完整配置"，但 standalone 模式要求预先生成的 `a_outputs` 数据，而 lite 模式联合训练。两者训练语义差异显著——若未来使用参考配置直接启动训练，将以 standalone 模式运行，行为与 smoke train 不同。

**影响**: 文档误导性，可能导致未来实验配置错误。

**修复建议**: 将参考配置中 `a_module.mode` 改为 `lite`，或在文件顶部添加显式注释说明 standalone 与 lite 的区别及当前推荐模式。

---

### F-3 [LOW] — `trigger_prob` 形状在 forward/predict_action 路径不对称

**文件**: `starVLA/model/framework/QwenPI.py`

**观察**:
- `forward()` L475: `trigger_prob = torch.sigmoid(...).detach().unsqueeze(-1)` → `[B, 1]`，直接传入 `corrective_flow_head`
- `predict_action()` L573-579: `trigger_prob = torch.sigmoid(...)` → `[B]`，先用于 `.max()` 门控检查，再以 `trigger_prob.unsqueeze(-1).float()` 传入 head

两路最终传入 head 的形状一致（均为 `[B, 1]`），功能正确。但命名不对称可能引起误读。

**修复建议**: `predict_action()` 中获取 trigger_prob 时直接 unsqueeze，保持与 forward 一致的变量语义。非阻塞。

---

### F-4 [INFO] — `vl_hidden_dim` 硬编码依赖

**文件**: `starVLA/model/framework/QwenPI.py`，init 段

`CorrectiveFlowHead` 的 `vl_hidden_dim` 取自 `llm_hidden_size`（即 QwenVL 的 LLM 隐层维度）。当前 Qwen2.5-VL-3B 为 2048，与参考配置一致。无即时问题，但若切换更大 VLM，需同步检查此参数。信息记录，不需修复。

---

## 5. Overall Verdict

**CONDITIONAL_PASS**

- CP-AH-1~3 全部 PASS，远程 500-step smoke train 证据充分。
- 冻结约束全部满足，无上游语义篡改。
- RES 方案 B 分析合理，CP-AH-1-ext 诊断级监控已到位。
- 存在 F-1（dtype 状态泄漏）和 F-2（参考配置模式不一致）两个 MEDIUM 级问题，不影响当前 smoke 结果，但应在 v0 正式部署前修复。

---

## 6. Blocking Items（CONDITIONAL_PASS 条件）

在进入下一 gate（AH-5 LIBERO eval 或 v0 正式轮次）之前，需解决：

| # | 级别 | 文件 | 要求 |
|---|---|---|---|
| B-1 | MEDIUM | `QwenPI.py` — predict_action corrective flow 块 | 在 `except`/`finally` 中恢复 `ami` 和 `corrective_flow_head` 的原始 dtype |
| B-2 | MEDIUM | `starvla_train_pi_qwen25_corrective_flow.yaml` | 将 `a_module.mode` 改为 `lite`，或加注释明确说明参考配置与 smoke train 的模式差异 |

---

## 7. Recommendations（非阻塞）

| # | 建议 |
|---|---|
| R-1 | `predict_action()` 中 `trigger_prob.unsqueeze(-1)` 提前，与 `forward()` 保持命名对称（F-3） |
| R-2 | 在 `run_cp_smoke_500step.sh` 中增加 `--trainer.optional_loss_hooks.corrective_flow.noise_scale 0.05` 显式覆盖，与参考配置完全对齐，避免依赖代码默认值 |
| R-3 | 未来正式轮次中，将 metrics.jsonl 从服务器拉取至本地 `docs/algorithm1/handoff/remote_evidence/cp_smoke_500step_20260415_042824/` 存档，使 CP-REVIEW 可直接抽检数值而非依赖 progress_live.md 文字记录 |
| R-4 | OBS-2（correction_mask 有效监督密度 50%）已由 CP-RES RFC 记录（CP-R2）。若正式训练中 `loss/corrective_flow_region` 出现持续低值（< 0.01），需重新评估是否增大 `chunk_len` 或调整零填充策略 |

---

## 8. 后续行动

| 线程 | 动作 |
|---|---|
| **CP-BUILD** | 修复 B-1（dtype 恢复）和 B-2（参考配置模式），commit 后重新通知 CP-REVIEW |
| **CP-REVIEW** | 收到修复提交后执行 Quick-Pass 验证（仅检查 B-1/B-2，不重跑 smoke） |
| **远程** | AH-5 LIBERO eval（独立任务，不阻塞 B-1/B-2 修复） |

---

*报告写入时间: 2026-04-15 CP-REVIEW 线程*  
*审核轮次: CP-ROUND-BOOTSTRAP-WAIT-UPSTREAM Round 1*

---

## Round 2 — Quick-Pass

**日期**: 2026-04-15  
**范围**: 仅验证 B-1 / B-2 blocking items（无需重跑 smoke train）  
**总体裁决**: **PASS**

### B-1 验证：`QwenPI.py` predict_action dtype 恢复

| 检查点 | 预期 | 实际 | 结论 |
|---|---|---|---|
| `orig_cf_dtype` 在 `try` 块之前捕获 | `orig_cf_dtype = next(self.corrective_flow_head.parameters()).dtype`（try 外） | L559: `orig_cf_dtype = next(self.corrective_flow_head.parameters()).dtype`（try 块之前）| ✅ |
| `orig_ami_dtype` 初始化为 None，在 if 内赋值 | `orig_ami_dtype = None` 在 try 外，赋值在 `if ami_is_module` 内 | L558: `orig_ami_dtype = None`；L562: `orig_ami_dtype = next(ami.parameters()).dtype` | ✅ |
| `finally` 块无条件恢复 `corrective_flow_head` | `finally: self.corrective_flow_head.to(dtype=orig_cf_dtype)` | L585-588: `finally` 块正确实现 | ✅ |
| `finally` 块有守卫地恢复 `ami` | `if ami_is_module and orig_ami_dtype is not None` | L586: 守卫条件正确 | ✅ |

**B-1 Quick-Pass 判定：PASS** ✅

### B-2 验证：参考配置 `a_module.mode`

| 检查点 | 预期 | 实际 | 结论 |
|---|---|---|---|
| `a_module.mode` 为 `lite` | `mode: lite` | L46: `mode: lite` | ✅ |
| `standalone:` 子配置块已移除 | 无 `input_key`、`allow_pseudo_labels_fallback` 等字段 | 已全部移除（diff 确认 -9 行） | ✅ |

**B-2 Quick-Pass 判定：PASS** ✅

### 附加范围变更核查（非 B-1/B-2）

以下变更随 B-1/B-2 修复一并落地，均在 CP-BUILD 允许路径内，与 RES 分析要求一致：

| 文件 | 变更内容 | 来源 | 结论 |
|---|---|---|---|
| `corrective_flow_head.py` | `forward()` 返回类型 `tuple[T,T]` → `tuple[T,T,dict]`；新增 debug dict（base/region loss） | cp_res_ah1_analysis_response.md Q4 要求 | ✅ 在范围内 |
| `QwenPI.py forward()` | `cf_loss, _ = ...` → `cf_loss, _, cf_debug = ...`；debug dict 合入 output_dict | 同上 | ✅ 在范围内 |
| `train_starvla.py` | `decomposed_metric_specs` 新增 `corrective_flow_base_loss` / `corrective_flow_region_loss` 两条 | 同上；已在远程 smoke train 中运行 | ✅ 在范围内 |

### 附加观察

**OBS-QP-1 [INFO]**：`docs/algorithm1/handoff/research_notes/phase3_0b_vae_accerl_analysis.md` 含 LaTeX 公式格式变化（`_{...}` → `*{...}*` 及表格对齐调整），属非语义性格式改动，不影响内容正确性。此文件为 A-RES 产出，理论上不在 CP-BUILD 修改范围内，但变更无实质影响。

### Quick-Pass 整体裁决

> **PASS**

所有 Round 1 阻塞项均已正确修复，无新引入的 blocking 问题，附加范围变更与已声明的 RES 分析需求完全一致。CP-BUILD 交付物审核关闭。

*Quick-Pass 完成时间: 2026-04-15 CP-REVIEW 线程*

---

## Round 3 — CP-AH-4 Evidence Review & Plan C Code Audit

**日期**: 2026-04-16  
**范围**: CP-AH-4 LIBERO 评测结果、Plan A/C 消融证据、Plan C 代码变更独立审查  
**总体裁决**: **FAIL（CP-AH-4 仍未达标；推理修复代码 PASS，但训练层面损害未解决）**

### 3.1 审核背景

Round 2 对 CP-BUILD 的 One-Step Corrective Flow 实现发出 PASS（CP-AH-1~3 全通过，B-1/B-2 已修复）。CP-AH-4（Success@LIBERO）当时为 BLOCKED_WAIT_REMOTE。此后远程证据回传：

1. CP-AH-4 Baseline 评测完成：69.0%（138/200）
2. CP-AH-4 CF 原始推理评测完成：62.0%（124/200）→ FAIL（< 67.0%）
3. CP-RES 根因分析已交付（cp_res_ah4_failure_analysis_response.md）
4. Plan A 消融（CF 推理禁用）：62.0% → 确认训练损害
5. Plan C 推理修复（_cf_prev_chunk + region_gate threshold）：64.0% → 仍 FAIL

### 3.2 上游 A-output 稳定性（重新确认）

| 检查项 | 状态 | 证据 |
|---|---|---|
| A-REVIEW verdict | PASS_CONFIRMED (Round 3) | a_module_current_round.yaml |
| downstream_usability | USABLE | a_module_current_round.yaml L148 |
| a_module_interface.py 冻结 | 未修改 | `git diff HEAD` 返回空 |
| optional_loss_utils.py 冻结 | 未修改 | `git diff HEAD` 返回空 |
| a_output_contract 版本 | v0.9.3_frozen | cp_current_round.yaml L67 |

**结论**: 上游 A-output 稳定，CP-BUILD 未篡改上游语义。与 Round 1/2 结论一致。

### 3.3 CP-AH-4 LIBERO 评测结果审查

#### 3.3.1 评测协议一致性

| 项 | 协议要求 (cp_res_ah4_eval_protocol.md) | 实际执行 | 一致 |
|---|---|---|---|
| 起始 ckpt | 2d_vlm_no_geo_30k（两模型 bit-for-bit 相同） | ✅ 均从 30k ckpt 出发 | ✅ |
| 训练步数 | 3000 | ✅ 3000 | ✅ |
| a_module_mode | lite | ✅ lite | ✅ |
| CF 唯一差异 | corrective_flow_loss enabled/disabled | ✅ | ✅ |
| eval task_suite | libero_goal, 10 tasks | ✅ | ✅ |
| eval trials | 20/task | ✅ | ✅ |
| eval seed | 7 | ✅ | ✅ |

**结论**: 评测协议完全一致，结果可信。

#### 3.3.2 CP-AH-4 数值验证

| 模型 | 成功率 | 状态 |
|---|---|---|
| Baseline (33k) | 69.0% (138/200) | 参考 |
| CF original | 62.0% (124/200) | -7.0pp → FAIL |
| Plan A (CF 推理禁用) | 62.0% (124/200) | -7.0pp → 确认训练损害 |
| Plan C (推理修复) | 64.0% (128/200) | -5.0pp → 仍 FAIL |

阈值：CF ≥ Baseline - 2pp = 67.0%

**CP-AH-4 判定**: **FAIL** — 最佳变体 Plan C (64.0%) 仍低于阈值 3pp。

#### 3.3.3 Plan A 消融结论（独立验证）

Plan A = CF 训练模型 + 推理时通过 `trigger_threshold=1.1` 完全禁用 CF 修正。结果与原始 CF 相同（62.0%），证明：

- **CF 训练本身造成 base model -7pp 退化**（这是主因）
- 原始 CF 推理的正负影响恰好抵消（部分任务 +，部分任务 -，净效果 0pp）
- 推理路径修复（Plan C）提供 +2pp 正信号，说明 CF 推理在正确实现下确有价值

此结论与 CP-RES 的 H4/H5 假设判定一致，REVIEW 认可。

### 3.4 Plan C 代码变更审查

以下为 Round 2 PASS 后新增的代码变更（当前在 git working tree 中，未提交）：

#### 变更 1: `corrective_flow_head.py` — region_gate 阈值化

```diff
- region_gate = torch.sigmoid(region_logits).unsqueeze(-1)
+ region_gate = torch.clamp(
+     torch.sigmoid(region_logits) - 0.3, min=0.0
+ ).unsqueeze(-1)
```

| 检查点 | 判定 |
|---|---|
| 数学正确性 | ✅ sigmoid(x) < 0.3 时 gate=0，消除低置信区域的噪声修正 |
| 与训练路径一致性 | ⚠️ 训练路径 forward() 仍使用 `sigmoid(region_logits)` 无阈值作为权重——train/infer 不对称 |
| 功能影响 | ✅ 减少 H5 过修正，Plan C 验证有效（+2pp） |

**F-R3-1 [LOW]**: train/infer region_gate 处理不对称。训练时 `forward()` L186 用 `sigmoid(region_logits).unsqueeze(-1)` 作为 region_weight，推理时 `predict()` L221-223 用 `clamp(sigmoid - 0.3, 0)`。这不是 bug（训练目标不需要阈值），但意味着推理时的 effective correction magnitude 与训练时的 loss 权重不匹配。RES 方案 B 框架下可接受。

#### 变更 2: `QwenPI.py` — _cf_prev_chunk 推理语义对齐

```diff
+ self._cf_prev_chunk: Optional[torch.Tensor] = None
+ def reset_cf_state(self):
+     self._cf_prev_chunk = None
...
+ a_prev_for_cf = (
+     self._cf_prev_chunk.to(device=pred_actions.device)
+     if self._cf_prev_chunk is not None
+     else torch.zeros_like(pred_actions)
+ )
+ a_corrected = self.corrective_flow_head.predict(
+     vl_embs=base_hidden.float(),
+     a_base=a_prev_for_cf.float(),  # ← 使用前一 chunk
+     ...
+ )
+ self._cf_prev_chunk = pred_actions.detach().clone()
```

| 检查点 | 判定 |
|---|---|
| 语义对齐 | ✅ 推理现在用 prev_chunk 而非当前预测，与训练 a_prev 语义对齐 |
| 初始状态 | ✅ None → zeros_like，与训练 fallback 一致 |
| dtype 恢复 | ✅ finally 块正确恢复 cf_head 和 ami dtype（Round 2 已验证） |
| 参数命名 | ⚠️ `a_base=a_prev_for_cf` 参数名与实际语义不匹配（predict 签名是 a_base，传入的是 prev_chunk） |

**F-R3-2 [INFO]**: `corrective_flow_head.predict()` 参数名 `a_base` 不再反映其实际用途（现在传入的是 prev_chunk）。建议后续重构时重命名为 `a_input` 或 `a_prev`。非阻塞。

#### 变更 3: `QwenPI.py` — trigger_threshold 环境变量覆盖

```diff
+ _env_thresh = os.environ.get("STARVLA_CF_TRIGGER_THRESHOLD")
+ self.corrective_flow_trigger_threshold = float(_env_thresh) if _env_thresh else _cfg_thresh
```

| 检查点 | 判定 |
|---|---|
| 功能 | ✅ 允许 eval 时通过环境变量覆盖阈值（Plan A 用 1.1 禁用 CF） |
| 安全性 | ✅ 有日志记录覆盖行为 |

#### 变更 4: `websocket_policy_server.py` — reset RPC

```diff
+ elif mtype == "reset":
+     if hasattr(self._policy, "reset_cf_state"):
+         self._policy.reset_cf_state()
+     return {"status": "ok", "ok": True, "type": "reset", "request_id": req_id}
```

| 检查点 | 判定 |
|---|---|
| 服务端 | ✅ 正确实现，hasattr 防御性检查 |
| 客户端 | ❌ **未实现** — model2libero_interface.py 不在本 repo 中，未发送 reset 信号 |

**F-R3-3 [HIGH]**: 客户端 reset 未实现，`_cf_prev_chunk` 在 LIBERO eval 时跨 episode 泄漏。Plan C 的 64.0% 结果受此污染，真实效果可能更好（估计 +1~2pp）。这是 Plan C 评测不完整的根本原因。

#### 变更 5: `QwenPI.py` — state_dim 截断 + `train_starvla.py` decomposed metrics

| 检查点 | 判定 |
|---|---|
| state_dim 截断（forward + predict_action） | ✅ 与 LIBERO 8→7 截断需求一致 |
| decomposed_metric_specs +2 行 | ✅ 与 debug dict 对齐，已在 smoke train 中验证 |

### 3.5 CP-RES 根因分析审查

| RES 假设 | RES 判定 | REVIEW 独立验证 | REVIEW 判定 |
|---|---|---|---|
| H1 训练不足 | LIKELY 辅助因素 | loss/cf_base=0.007 非零，合理 | ✅ 同意 |
| H2 损失干扰 | UNLIKELY | CF loss/action < Baseline loss/action，排除 | ✅ 同意 |
| H3 trigger 域偏移 | LIKELY 放大器 | loss/region 高方差，合理 | ✅ 同意 |
| H4 a_prev/a_base 不匹配 | VERY LIKELY（首要） | 代码审查确认 forward vs predict 输入不同 | ✅ 同意，已在 Plan C 修复 |
| H5 过修正 | VERY LIKELY | T1/T7/T8 一致 -10pp + sigmoid(0)≈0.5 分析合理 | ✅ 同意，已在 Plan C 部分修复 |
| PAUSE 建议 | PAUSE | 合理：修复代码问题后重测，非方案缺陷 | ✅ 同意 |

**关键独立发现**: Plan A 证明训练损害（-7pp）是 CF-AH-4 FAIL 的主因，这不能通过推理修复解决。RES 正确识别了这一点并建议 PAUSE。REVIEW 认可 PAUSE 而非 PIVOT，因为：
- CF 在 T0 有 +35pp（Plan C），T9 有 +10pp（Plan C），证明 CF 推理在正确工作时有真实价值
- 训练损害需要训练层面的修复（gradient isolation / loss weight 调整），属于 CP-BUILD 下一轮任务

### 3.6 语义一致性检查（REVIEW 核心职责）

| 检查项 | 结果 |
|---|---|
| CP-BUILD 是否篡改上游 A-output 语义？ | ❌ 未篡改。a_module_interface.py、optional_loss_utils.py 均 FROZEN |
| CP-BUILD 是否偷偷重定义 delta_action 目标？ | ❌ 未重定义。velocity_gt = a_gt - a_prev 与 RES RFC 一致 |
| 训练/推理/验收指标是否一致？ | ⚠️ 部分不一致（见 F-R3-1），但在 RES 方案 B 框架下可接受 |
| 有无代码能跑但目标错的语义问题？ | ✅ H4（a_prev/a_base 不匹配）已在 Plan C 修复；H5（region_gate）已部分修复 |
| "可运行"是否等同于"正确"？ | ❌ 不等同。Plan A 证明代码能跑（smoke PASS）但训练目标存在副作用（-7pp） |

### 3.7 Findings List（按严重度排序）

| # | 严重度 | 类别 | 内容 | 状态 |
|---|---|---|---|---|
| F-R3-CRIT-1 | **CRITICAL** | 训练损害 | CF 联合训练导致 base model -7pp 退化（Plan A 证实），纯推理修复无法弥补 | 需训练层修复 |
| F-R3-HIGH-1 | **HIGH** | 评测不完整 | Plan C 评测缺少客户端 episode reset → _cf_prev_chunk 跨 episode 泄漏 → 64% 结果被污染 | 需补全 |
| F-R3-LOW-1 | LOW | train/infer 不对称 | region_gate 训练用 sigmoid，推理用 clamp(sigmoid-0.3, 0) | RES 方案 B 可接受 |
| F-R3-INFO-1 | INFO | 命名不一致 | corrective_flow_head.predict() 参数名 a_base 实际接收 prev_chunk | 建议重命名 |

### 3.8 Artifact Checklist（Round 3 新增证据）

| # | Artifact | 来源 | 完整性 |
|---|---|---|---|
| 1 | CP-AH-4 Baseline eval (69.0%) | cp_ah4_libero_results.md | ✅ 完整 (200 episodes, per-task) |
| 2 | CP-AH-4 CF eval (62.0%) | cp_ah4_libero_results.md | ✅ 完整 |
| 3 | Plan A 消融 (62.0%) | cp_ah4_libero_results.md | ✅ 完整 |
| 4 | Plan C 修复 eval (64.0%) | cp_ah4_libero_results.md | ⚠️ 不完整（缺客户端 reset） |
| 5 | CP-RES 根因分析 | cp_res_ah4_failure_analysis_response.md | ✅ 全面 |
| 6 | 训练损失对比 | cp_res_ah4_failure_analysis_request.md §3 | ✅ 完整 (600 entries each) |
| 7 | Plan C 代码变更 | git diff（本轮审查） | ✅ 代码 PASS |
| 8 | 冻结文件验证 | git diff HEAD（4 个冻结文件返回空） | ✅ FROZEN |

### 3.9 Overall Verdict（Round 3）

**CP-AH-4: FAIL**  
**Plan C 代码变更: PASS（推理修复逻辑正确）**  
**总体 CP v0 状态: BLOCKED — 需训练层修复后重测**

### 3.10 Residual Risks

| # | 风险 | 严重度 | 缓解策略 |
|---|---|---|---|
| RR-1 | CF 训练损害可能无法通过 loss weight 调整完全消除 | HIGH | 需 gradient isolation 实验（stop_grad on base model 或降低 corrective_flow_scale） |
| RR-2 | 客户端 reset 未实现，Plan C 真实上限未知 | MEDIUM | 补全客户端 reset 后重测 |
| RR-3 | 统计效力（20 trials, ±7pp 95% CI） | LOW | 评测协议已含 50-trial 仲裁条款 |

### 3.11 后续行动

| 线程 | 动作 | 优先级 |
|---|---|---|
| **CP-RES** | 设计训练层修复方案（gradient isolation / corrective_flow_scale 降低 / base model 参数冻结策略） | P0 |
| **CP-BUILD** | (1) 补全客户端 episode reset；(2) 实施 RES 训练层修复后重跑 CP-AH-4 | P0 |
| **CP-REVIEW** | 收到新 CP-AH-4 结果后执行 Round 4 审查 | 等待 |

---

*Round 3 完成时间: 2026-04-16 CP-REVIEW 线程*

---

## Round 4 — CP-AH-4 训练损害根因仲裁

**日期**: 2026-04-16  
**性质**: CP-RES 自我否定请求 → CP-REVIEW 独立仲裁  
**范围**: 梯度路径审查、实验公平性核查、统计显著性评估、RES 报告一致性验证  
**总体裁决**: **-7pp 主因是随机优化分歧（RNG divergence），辅因是梯度裁剪交互效应，不是 CF 梯度直接损害 VLM**

---

### Q1 判定：CF corrective_flow_loss 是否对 VLM 有梯度影响？

#### 逐行梯度路径追踪

审查 `QwenPI.forward()` L460-499，逐一确认 `corrective_flow_head.forward()` 的五个输入的 detach 状态：

| 输入 | 来源代码 | detach 状态 | VLM 梯度 |
|---|---|---|---|
| `vl_embs` | L488: `base_hidden.to(...).detach()` | **DETACHED** | ❌ 无 |
| `a_prev` | L473-477: GT actions + `torch.randn_like` | label tensor | ❌ 无 |
| `a_gt` | L423: `actions_target` | label tensor | ❌ 无 |
| `trigger_prob` | L486: `sigmoid(...).detach()` | **DETACHED** | ❌ 无 |
| `region_logits` | L487: `a_predictions_cf.region_logits.detach()` | **DETACHED** | ❌ 无 |

**结论：`corrective_flow_loss` 的反向传播仅更新 `CorrectiveFlowHead` 自身参数（~3M），不对 VLM backbone（~3B）产生任何梯度。**

#### L479-485 第二次 a_module_interface.predict() 路径

```python
pooled_for_cf = base_hidden.to(dtype=action_model_dtype).mean(dim=1)  # 未 detach
a_predictions_cf = self.a_module_interface.predict(pooled_hidden=pooled_for_cf, ...)
trigger_prob = ... .detach()     # 输出立即 detach
region_logits = ... .detach()    # 输出立即 detach
```

`pooled_for_cf` 未 detach，但输出全部 detach。此前向传播创建的计算图节点为"死端"——无任何 loss 通过此路径回传。在标准 PyTorch autograd 中，已有的 `a_loss`（通过 L450 的第一次 `_compute_optional_hook_outputs` 调用）梯度路径不受影响。

**判定：此路径无有效梯度贡献。**

#### 那 -7pp 的真实机制是什么？

我识别到两个间接机制：

**机制 A（主要）：RNG 状态分歧**

CF 模型的 forward 路径比 Baseline 多消耗以下随机数：

| 消耗点 | 代码位置 | 消耗量级 |
|---|---|---|
| `torch.randn_like(a_prev)` | QwenPI.py L477 | B × chunk_len × action_dim |
| CorrectiveFlowHead DiT dropout=0.1 | corrective_flow_head.py L81 | 4层 × 多个 attention/FFN |

关键时序：action_model（L446，flow-matching noise）在 CF 块（L460-499）**之前**执行。因此同一步内的 flow-matching noise 相同，但 CF 块消耗 RNG 后，**下一步**的 flow-matching noise 开始分歧。从 step 30002 起，两模型的优化轨迹完全不同。

**机制 B（次要）：全局梯度裁剪交互**

训练配置 `gradient_clipping: 1.0`（`starvla_train_pi_from30k.yaml` L81 + `ds_config.yaml` L21）。`train_starvla.py` L1709 确认：

```python
grad_norm = self.accelerator.clip_grad_norm_(
    self.model.parameters(),       # 包含 CF_HEAD 参数
    self.config.trainer.gradient_clipping,  # 1.0
)
```

全局梯度范数包含 CF head 参数的梯度。如果裁剪被触发（global_norm > 1.0）：

- Baseline 有效裁剪因子 = 1.0 / G_base
- CF 有效裁剪因子 = 1.0 / sqrt(G_base² + G_cf²)

CF 模型的 base 参数获得略小的有效梯度。但量级评估：CF head ~3M params vs 总模型 ~3B+，G_cf 相对 G_base 很小。以 G_base=5.0, G_cf=1.0 为例，差异约 2%。3000 步积累后效应有限但非零。

**Q1 最终判定：`corrective_flow_loss` 对 VLM 无直接梯度影响。-7pp 的主要机制是 RNG 分歧导致的优化轨迹差异，次要机制是全局梯度裁剪的微弱交互。**

---

### Q2 判定：实验配置公平性

逐项对比两个训练脚本（`run_cp_ah4_baseline_train.sh` vs `run_cp_ah4_cf_train.sh`）：

| 配置项 | Baseline | CF | 一致 |
|---|---|---|---|
| 起始 ckpt | `steps_30000_pytorch_model.pt` | 同一文件 | ✅ |
| `--seed` | 42 | 42 | ✅ |
| `--per_device_batch_size` | 4 | 4 | ✅ |
| `--gradient_accumulation_steps` | 4 | 4 | ✅ |
| `--num_processes` | 4 | 4 | ✅ |
| 有效 batch size | 64 | 64 | ✅ |
| `--trainer.max_train_steps` | 33000 | 33000 | ✅ |
| `--config_yaml` | `starvla_train_pi_from30k.yaml` | 同一文件 | ✅ |
| `--datasets.vla_data.data_mix` | `libero_all` | `libero_all` | ✅ |
| `--datasets.vla_data.correction_dataset_jsonl` | 同一 JSONL | 同一 JSONL | ✅ |
| `--framework.a_module.mode` | lite | lite | ✅ |
| a_loss/corrective_loss 全部权重 | 全部相同 | 全部相同 | ✅ |
| DeepSpeed 配置 | `deepspeed_zero2.yaml` | `deepspeed_zero2.yaml` | ✅ |
| CF 独有配置 | 无 | `corrective_flow.enabled=true, scale=0.5` | 预期差异 |

**显式公平性约束全部满足。** 但存在以下隐式混淆变量：

| # | 混淆变量 | 性质 | 影响评估 |
|---|---|---|---|
| C-1 | RNG 分歧（Q1 机制 A） | CF forward 多消耗随机数 → 后续步骤 flow-matching noise 不同 | **HIGH** — 导致不同优化轨迹 |
| C-2 | 梯度裁剪交互（Q1 机制 B） | CF 全局梯度范数略大 → base 参数有效 LR 微降 | LOW — ~2% 差异 |
| C-3 | GPU 内存布局 | CF 多 ~3M 参数 → 可能影响 ZeRO-2 分区 | NEGLIGIBLE |
| C-4 | 运行时间差异 | 不同时间戳运行 → 不同环境噪声 | NEGLIGIBLE |

**Q2 判定：显式配置公平，但 C-1（RNG 分歧）是显著隐式混淆变量。当前实验设计无法区分"CF 训练损害"与"随机优化差异"。**

---

### Q3 判定：-7pp gap 统计显著性

#### 整体 gap（200 episodes）

双比例 z 检验：

| 参数 | 值 |
|---|---|
| p₁ (Baseline) | 0.690 (138/200) |
| p₂ (CF) | 0.620 (124/200) |
| pooled p | 0.655 (262/400) |
| SE | sqrt(0.655 × 0.345 × (1/200 + 1/200)) = **0.0475** |
| z | (0.69 − 0.62) / 0.0475 = **1.47** |
| p-value (two-tailed) | **0.141** |
| 95% CI for gap | [-2.3pp, +16.3pp] |

**-7pp 在 α=0.05 水平下不显著。95% 置信区间包含零。**

#### Per-task 分析（各 20 episodes）

| Task | Baseline | CF | Gap | Fisher exact p | Bonferroni p_adj |
|---|---|---|---|---|---|
| T0 | 12/20 | 17/20 | +25pp | 0.177 | 1.0 |
| T1 | 20/20 | 18/20 | -10pp | 0.487 | 1.0 |
| T2 | 19/20 | 11/20 | -40pp | **0.008** | 0.08 |
| T3 | 7/20 | 6/20 | -5pp | 1.0 | 1.0 |
| T5 | 9/20 | 3/20 | -30pp | 0.041 | 0.41 |
| T7 | 20/20 | 18/20 | -10pp | 0.487 | 1.0 |
| T8 | 20/20 | 18/20 | -10pp | 0.487 | 1.0 |

T2 的 -40pp 在未校正下显著（p=0.008），但 Bonferroni 校正后 p_adj=0.08，边缘不显著。

#### 最小样本量估计

要在 α=0.05, power=0.80 下检出 7pp 的真实差异：

- 每组需要 n ≈ (z_α/2 + z_β)² × 2p(1-p) / δ²
- ≈ (1.96 + 0.84)² × 2 × 0.655 × 0.345 / 0.07²
- ≈ 7.84 × 0.452 / 0.0049
- ≈ **723 episodes per group（约 36 trials/task）**

当前 20 trials/task (200 episodes) 严重不足。评测协议中的 50-trial 仲裁条款（500 episodes）也不够。

**Q3 判定：当前 -7pp gap 不具备统计显著性（p=0.141）。200 episodes 的检验功效不足以检出此量级的真实差异。建议最低 36 trials/task（720 episodes）。**

---

### Q4 判定：两份 CP-RES 报告一致性

#### 报告时间线与信息差

| 报告 | 写作时间点 | 可用信息 | 核心结论 |
|---|---|---|---|
| 初始分析 (cp_res_ah4_failure_analysis_response.md) | Plan A 之前 | CF 62% vs Baseline 69% + 代码审查 | H4（推理路径缺陷）为首要根因 |
| Gap 分析 (cp_res_rfc_impl_gap_analysis.md) | Plan A/C 之后 | 全部四变体结果 | 5 个 gap 分析，但未明确修正 H4 首要判定 |

#### 矛盾点

1. **初始分析**: "失败的根因是可修复的代码问题（H4/H5），而非方案缺陷" → 暗示修复推理路径即可
2. **Plan A 实证**: Plan A = 62% = CF original → 推理路径净贡献为零，根因在训练阶段
3. **Gap 分析**: 未明确否定初始分析的"H4 首要根因"判定，而是平行列出 G1-G5 差距

初始分析的 H4 判定在 Plan A 结果面前**已被证伪**：H4（推理路径语义不匹配）对整体 -7pp 的净贡献为 0pp（Plan A = Plan original）。H4 修复（Plan C）提供了 +2pp，但这不能解释 -7pp 的来源。

Gap 分析没有明确做出此修正，而是转向列举实现差距。这不是矛盾，但是**不完整**——两份报告都没有直接回答"训练损害的物理机制是什么"。

#### 哪份更可信？

**Gap 分析更可信**（拥有完整四变体数据），但两份报告都没有正确识别训练损害的真实机制（RNG 分歧 + 梯度裁剪交互，而非 CF 梯度直接损害 VLM）。

**Q4 判定：初始分析的 H4 首要根因判定已被 Plan A 证伪。Gap 分析更完整但未修正此判定。两份报告均未识别 RNG 分歧这一真实机制。**

---

### 仲裁结论

**-7pp 训练损害是 (b) 随机优化差异，辅以 (a) 的微弱梯度裁剪交互。不是系统性架构效应，不是实验设计缺陷。**

| 候选 | 判定 | 理由 |
|---|---|---|
| (a) 系统性架构效应 | **排除** | CF loss 完全 detach，无梯度路径到 VLM |
| (b) 随机优化差异 | **首要** | CF forward 多消耗 RNG → flow-matching noise 分歧 → 不同优化轨迹；-7pp 不显著（p=0.141） |
| (b') 梯度裁剪交互 | **辅助** | 全局范数略增 → base 参数有效 LR 微降（~2%），3000 步积累有限 |
| (c) 实验设计缺陷 | **部分成立** | 样本量不足（200 episodes, power < 0.50）；RNG 耦合是隐式混淆变量 |

---

### 下一步建议：最小验证实验方案

#### 实验 1：消除 RNG 耦合的决定性实验（最优先）

目标：在 RNG 完全对齐的条件下对比 Baseline 和 CF 训练。

方案：在 CF forward 的 CF 块（L460-499）入口处保存 RNG 状态，出口处恢复，使得 CF 块的随机数消耗不影响后续步骤的 RNG。

```python
if self.corrective_flow_enabled:
    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    # ... CF block ...
    torch.random.set_rng_state(rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
```

然后以完全相同的配置训练 CF_v2（RNG 对齐），对比 CF_v2 vs Baseline。

如果 CF_v2 ≈ Baseline（69% ± 3pp），则确认 -7pp 纯为 RNG 分歧。
如果 CF_v2 仍然低于 Baseline，则存在梯度裁剪或其他未识别的机制。

**成本**：~2.5h 训练 + ~1.5h eval = 4h GPU 时间

#### 实验 2：增大样本量的统计验证

对现有 Baseline 33k ckpt 和 CF 33k ckpt 各加跑 2 个额外 seed（seed=8, seed=9），总计 3 × 200 = 600 episodes per model。

**成本**：~6h eval（无需训练）

#### 实验 3：梯度裁剪消融

如果实验 1 不能完全解释 gap，在 CF 训练中将 CF head 参数排除出梯度裁剪范围（单独裁剪 CF head 和 base model）：

```python
base_params = [p for n, p in model.named_parameters() if 'corrective_flow_head' not in n]
cf_params = [p for n, p in model.named_parameters() if 'corrective_flow_head' in n]
accelerator.clip_grad_norm_(base_params, 1.0)
accelerator.clip_grad_norm_(cf_params, 1.0)
```

**成本**：~2.5h 训练 + ~1.5h eval = 4h

#### 优先级排序

```
立即执行（无需架构改动）：
  P0: 实验 2 — 多 seed eval（6h，零训练成本，直接量化统计可信度）
  P1: 实验 1 — RNG 对齐训练（4h，决定性实验）

如 P0+P1 确认随机差异为主因：
  → CP-AH-4 当前 FAIL 判定需要降级为 INCONCLUSIVE
  → 重新设计 30-trial eval 基线

如 P1 仍有 gap：
  P2: 实验 3 — 梯度裁剪消融
```

---

*Round 4 仲裁完成时间: 2026-04-16 CP-REVIEW 线程*
