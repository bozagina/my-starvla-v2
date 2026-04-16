# CP-RES 分析响应：CP-AH-4 LIBERO 评测协议

```
status:     DELIVERED
author:     CP-RES thread
thread_id:  CP-RES
date:       2026-04-15
request_by: CP-BUILD
关联:       cp_res_policy_rfc.md §6, cp_review_audit_report.md
```

---

## Q1：训练协议设计

**结论：选项 (c)，混合数据训练，lite mode。**

- 不选 (a)：30k ckpt 无 A-module heads，直接 eval 等价于"纯 base policy"基线，与 CP-AH-4 定义的"A-only 基线"不符。
- 不选 (b)：仅 action loss 的模型不具备 A-module 语义，CorrectiveFlowHead 无法接收有意义的 trigger/region 信号。
- 选 (c) 理由：

```yaml
baseline_model:
  start_ckpt: 2d_vlm_no_geo_30k (30k steps)
  a_module_mode: lite
  enabled_losses: [action_loss, a_loss, corrective_loss]
  disabled_losses: [corrective_flow_loss]
  推理: trigger 激活后执行 A-module 元数据预测，不调用 CF head

cf_model:
  start_ckpt: same as baseline (bit-for-bit identical)
  a_module_mode: lite
  enabled_losses: [action_loss, a_loss, corrective_loss, corrective_flow_loss]
  corrective_flow_scale: 0.5
  推理: trigger 激活后调用 CorrectiveFlowHead 修正 action chunk
```

关键约束：两个模型从**完全相同的 30k ckpt** 出发，使用**完全相同的训练数据顺序**（固定 seed），唯一差异是 `corrective_flow_loss` 是否开启。

lite mode 选择理由：与 BD-3 决策、CP-BUILD smoke train、train/eval 一致性一致。

---

## Q2：训练步数

**推荐：3000 步，绝对最小 2000 步。**

| 参数 | 值 |
|------|-----|
| FASA 样本数 | 2000 |
| effective batch_size | 4 GPU × bs=4 × grad_accum=4 = 64 |
| 每步 FASA 消费 | ≈2-3 samples/step |
| 3000 步 FASA epochs | ≈ 3.75 |
| 时间估算 (4×H20) | ≈ 2.5 小时/模型 |

不建议 100k 步：FASA 2000 samples 会过度重复（~125 epochs），高风险覆盖 base skill。

---

## Q3：公平性约束

必须相同：起始 ckpt、seed、optimizer/lr/weight_decay、batch_size/grad_accum、训练步数、数据(LIBERO+FASA)、a_loss/corrective_loss 权重。

CF head 额外参数约 3M（占总参数 0.1%），影响可忽略。不建议 freeze base 参数。

---

## Q4：LIBERO eval 配置

```
task_suite: libero_goal      # 10 tasks
num_trials_per_task: 20      # 平衡统计效力与时间
seed: 7                      # eval_libero.py 默认
expected_state_dim: 7        # 匹配 30k ckpt
```

判定标准：
- **PASS**: CF success_rate >= baseline - 2pp（单次 20-trial run）
- **CONDITIONAL_PASS**: 差距在 [-5pp, -2pp]，升级到 50 trials
- **FAIL**: CF < baseline - 5pp

---

## Q5：state_dim 截断语义

截断 8→7 是语义正确的。LIBERO 的 8D state = [eef_pos(3), axisangle(3), gripper_qpos(2)]。截掉 gripper_qpos[1]（右指关节），因 Franka 平行夹爪对称联动，q_left 已充分表达开合状态。与 30k ckpt 训练时的 state_dim=7 一致，无分布偏移。

---

## Q6：风险评估

| # | 风险 | 严重度 | 缓解 |
|---|------|--------|------|
| R-1 | FASA 2000 samples 过少，trigger 信号噪声大 | MEDIUM | 监控 trigger rate，若 <5% 则 CF 几乎不生效 |
| R-2 | Base policy 遗忘（两模型同等受影响） | MEDIUM | 混合 LIBERO + FASA，保持 action_loss scale=1.0 |
| R-3 | CF head 未激活（trigger_prob < 0.5） | MEDIUM | eval 前抽查 trigger_prob 日志 |
| R-4 | 统计效力不足（20 trials ±7pp CI） | LOW | 两阶段协议：20 trials 初判 → 50 trials 仲裁 |
| R-5 | lite vs standalone 语义差异 | INFO | 已由 BD-3 固定 |

---

## 汇总协议

```yaml
protocol_id: cp_ah4_v0
gate: phase3_0b_cp_ah4_libero_eval

training:
  start_ckpt: /2025233147/zzq/SpatialVLA_llava3d/checkpoints_to_transfer/2d_vlm_no_geo_30k/
  steps: 3000
  seed: 42
  a_module_mode: lite

eval:
  task_suite: libero_goal
  num_trials_per_task: 20
  seed: 7
  expected_state_dim: 7

acceptance:
  PASS: cf_success_rate >= baseline_success_rate - 2pp
  CONDITIONAL_PASS: cf in [-5pp, -2pp], requires 50-trial confirmation
  FAIL: cf < baseline - 5pp
```
