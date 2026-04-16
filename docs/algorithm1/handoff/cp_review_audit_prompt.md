# CP-REVIEW 审核提示

> 发起时间: 2026-04-15  
> 发起方: CP-BUILD  
> 目标方: CP-REVIEW  
> 审核规则: evidence_only（仅基于证据判定，不做代码风格/架构偏好评审）

---

## 你的角色

你是 **CP-REVIEW** 线程。CP-BUILD 已完成 One-Step Corrective Flow 的实现和远程
500-step smoke train 验证，状态为 `DONE_PENDING_REVIEW`。

你的任务是审核 CP-BUILD 的全部交付物，验证其是否满足 required_artifacts 和
acceptance_hypotheses，然后给出 PASS / CONDITIONAL_PASS / FAIL 判定。

---

## 审核清单

### A. 必读文档（按顺序）

1. **Manifest**: `docs/algorithm1/handoff/manifests/cp_current_round.yaml`
  - 关注 CP-BUILD thread 的 `status`, `acceptance_hypotheses`, `delivered_artifacts`, `blocking_conditions`
2. **Progress log（最新条目）**: `docs/algorithm1/handoff/progress_live.md`
  - 搜索 `2026-04-15 20:55` 找到 CP-BUILD 远程 smoke train 完成条目
  - 搜索 `2026-04-15 14:00` 找到 CP-BUILD 本地实现完成条目
3. **RES 分析文档**:
  - `docs/algorithm1/handoff/cp_res_ah1_analysis_request.md`（CP-BUILD → RES 的问题）
  - `docs/algorithm1/handoff/cp_res_ah1_analysis_response.md`（RES 的分析结论）
4. **原始 RFC**: `docs/algorithm1/handoff/cp_res_policy_rfc.md`
  - §6 定义了 CP-AH-1~4 验收假设

### B. 必审代码（4 个文件）

1. **新建** `starVLA/model/framework/corrective_flow_head.py`
  - 审核点:
    - `CorrectiveFlowHead` 架构是否符合 RFC（ActionEncoder + MetadataEncoder + DiT + ActionDecoder）
    - `forward()` 返回签名是否为 `tuple[Tensor, Tensor, dict]`
    - loss 计算逻辑（L186-196）: `base_loss + region_loss_weight * region_loss`
    - debug dict 包含 `corrective_flow_base_loss` 和 `corrective_flow_region_loss`
    - `predict()`: region-gated one-step correction `a_base + sigmoid(region) * velocity`
2. **修改** `starVLA/model/framework/QwenPI.py`
  - 审核点:
    - `__init_`_: 条件实例化 CorrectiveFlowHead（检查 `framework.corrective_flow.enabled`）
    - `forward()`: corrective_flow_loss 计算位置（在 optional_hook_outputs 之后）
    - `forward()`: debug dict 解包并合入 output_dict
    - `predict_action()`: eval 路径 bf16 修复（`isinstance(ami, nn.Module)` 判断 + 临时 float 转换）
    - 冻结约束: `a_module_interface.py` 未被修改
3. **修改** `starVLA/training/train_starvla.py`
  - 审核点:
    - `hook_specs` 中包含 `("corrective_flow", "corrective_flow_loss", "loss/corrective_flow")`
    - `decomposed_metric_specs` 中包含两个新条目:
      - `("corrective_flow_base_loss", "loss/corrective_flow_base", "corrective_flow")`
      - `("corrective_flow_region_loss", "loss/corrective_flow_region", "corrective_flow")`
4. **新建** `tools/handoff/run_cp_smoke_500step.sh`
  - 审核点:
    - 基于 `run_ah4_fusion_500step.sh` 模板
    - 正确的远程路径（repo root, conda env, model path, data path）
    - corrective_flow CLI 参数覆盖是否完整

### C. 配置审核

- **新建** `starVLA/config/training/starvla_train_pi_qwen25_corrective_flow.yaml`
  - 审核点: `framework.corrective_flow` 参数、`trainer.optional_loss_hooks.corrective_flow` 参数
  - 注意: 实际远程训练使用的是 `starvla_train_pi.yaml` + CLI 覆盖，此文件作为参考配置

### D. 冻结约束验证

以下文件 **必须未被修改**（检查 git status 或 diff）:

- `starVLA/model/framework/a_module_interface.py`
- `starVLA/model/modules/loss/optional_loss_utils.py`
- `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`（仅 import）
- `starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py`（仅 import）

---

## 验收假设审核

对每个 CP-AH，请验证 **证据是否支持 BUILD 声明的判定**:

### CP-AH-1 (original): trigger=0 → region_loss=0


| 项    | BUILD 声明                                                                                       |
| ---- | ---------------------------------------------------------------------------------------------- |
| 判定   | PASS                                                                                           |
| 证据   | `loss/region=0` in 38/100 steps; `debug/corrective_loss_region_trigger_filtered_count` avg=6.2 |
| 验证方式 | 确认 `metrics.jsonl` 中 P0 gate 生效                                                                |


### CP-AH-1-ext (diagnostic): CorrectiveFlowHead 隐式 P0


| 项      | BUILD 声明                                                                               |
| ------ | -------------------------------------------------------------------------------------- |
| 判定     | PASS (诊断级)                                                                             |
| 证据     | `loss/corrective_flow_region`: 前半段 0.077 → 后半段 0.037（↓52%）                             |
| RES 结论 | 方案 B 采纳，保持 loss 逻辑不变                                                                   |
| 验证方式   | 阅读 `cp_res_ah1_analysis_response.md` 确认 RES 结论合理；确认新 debug metrics 存在于 `metrics.jsonl` |


### CP-AH-2: 所有 loss 分量 finite 且非退化


| 项    | BUILD 声明                                                   |
| ---- | ---------------------------------------------------------- |
| 判定   | PASS                                                       |
| 证据   | 12 个 loss 分量 500 步全部 finite，无 NaN/Inf                      |
| 验证方式 | 抽检 `metrics.jsonl` 中任意 10 条记录，确认关键 loss key 存在且为 finite 数值 |


### CP-AH-3: loss/consist finite 且下降趋势


| 项    | BUILD 声明                                         |
| ---- | ------------------------------------------------ |
| 判定   | PASS                                             |
| 证据   | 前半段 avg=9.3e-5 → 后半段 avg=5e-6（↓95%）；最后 5 步全为 0.0 |
| 验证方式 | 确认 `metrics.jsonl` 中 `loss/consist` 值存在且非负       |


### CP-AH-4: Success@LIBERO


| 项    | BUILD 声明                       |
| ---- | ------------------------------ |
| 判定   | BLOCKED_WAIT_REMOTE            |
| 验证方式 | 确认此项未被 BUILD 声明为 PASS（正确的阻塞声明） |


---

## required_artifacts 检查表

请逐项确认以下交付物是否存在且完整:


| #   | Artifact                  | 预期位置                                                                                 |
| --- | ------------------------- | ------------------------------------------------------------------------------------ |
| 1   | config diff               | `starvla_train_pi_qwen25_corrective_flow.yaml` + CLI 参数（见 `run_cp_smoke_500step.sh`） |
| 2   | local smoke evidence      | `progress_live.md` 2026-04-15T14:00 条目（syntax/AST/lint/YAML all PASS）                |
| 3   | remote smoke evidence     | `progress_live.md` 2026-04-15T20:55 条目 + `cp_current_round.yaml` `remote_evidence`   |
| 4   | changed-file handoff note | `progress_live.md` 两个条目中的"修改了什么"                                                     |


---

## 期望输出

请给出结构化的审核报告，包含:

1. **Artifact checklist**: 每个 required_artifact 的存在/完整性判定
2. **CP-AH findings**: 每个验收假设的独立判定（PASS / FAIL / CONDITIONAL）
3. **Freeze constraint verification**: 冻结文件是否被修改
4. **Code findings**: 代码审核中发现的问题（如有）
  - CRITICAL: 影响正确性或安全性
  - HIGH: 影响可维护性或一致性
  - MEDIUM: 建议改进
  - LOW: 可选优化
5. **Overall verdict**: PASS / CONDITIONAL_PASS / FAIL
6. **Blocking items**: 如果 CONDITIONAL_PASS 或 FAIL，列出必须修复的项
7. **Recommendations**: 非阻塞性建议

请将审核报告写入: `docs/algorithm1/handoff/cp_review_audit_report.md`  
并更新 `cp_current_round.yaml` 中 CP-REVIEW 线程的状态和判定。