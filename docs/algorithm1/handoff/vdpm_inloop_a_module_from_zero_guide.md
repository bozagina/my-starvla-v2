# StarVLA 新人从零指南：伪标签怎么来，A 模块怎么工作（含 VDPM in-loop）

本文面向第一次接触项目的同学。目标是讲清三件事：
- 伪标签（pseudo labels）是怎么生成的
- A 模块输出（`a_outputs`）是怎么构建的
- 训练时 A 模块是怎么接进主链路（包括 real in-loop VDPM）

文档中的代码路径优先使用服务器绝对路径：
`/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA`

---

## 0) 先统一“你看到的是哪套代码”

先在服务器执行：

```bash
cd /2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA
git branch --show-current
git rev-parse HEAD
```

当前常见现象：
- 你可能在 `codex/tmp-20260402-p0-audit-infra`
- 这个分支里通常有 `tools/build_vdpm_a_outputs.py`
- 但不一定有 `tools/build_fasa_dataset.py`

这不矛盾。原因是：
- **训练主链路（A-loss/in-loop）核心实现**在 `starVLA/model/framework/*` 和 `starVLA/training/train_starvla.py`
- **数据构建脚本**在不同分支/阶段被逐步引入

所以看“原理”和“训练接入”时，以主链路代码为准；看“伪标签构建脚本”时，允许参考历史权威脚本路径（后面会给）。

---

## 1) 一张总流程图（先建立全局感）

```text
原始/纠偏样本
  -> 生成 pseudo_labels（风险/触发/delta/区域先验）
  -> 生成 a_outputs（risk_pred/trigger_logit/delta_pred/region_logits/embed）
  -> 训练时 A 模块接口读取 a_outputs（offline 或 in-loop）
  -> QwenPI 计算分解损失（risk/trigger/delta/region/embed）
  -> trainer 记录 loss/* 指标并做门禁
```

---

## 2) 伪标签是怎么生成的

### 2.1 数据输入是什么

伪标签构建依赖每条样本至少包含：
- `action_chunk`（二维序列，T x action_dim）
- `state_t`
- `future_state`（或者可由 `horizon` 回填）
- `meta`（`dataset_name/trajectory_id/sample_step`）

### 2.2 核心计算逻辑（公式级）

参考（历史权威脚本）：
- `/2025233147/zzq_0317/codex_runs/p4_1_fasa_phase0_libero_remote_20260409/tools/build_fasa_dataset.py`

同逻辑本地镜像可见：
- `/Users/bazinga/code/my-starvla-v2/tools/build_fasa_dataset.py`

关键函数：
- `_compute_pseudo_labels`
- `_canonicalize_pseudo_labels`
- `_build_region_target_15`

生成规则（简化）是：
1. `delta_action_norm = ||a_t - a_{t-1}||`
2. 取 `delta_action_norm` 的 0.75 分位当阈值，得到 `correction_mask`
3. `risk_score = max(delta_action_norm)`
4. `trigger_label = max(correction_mask)`（有一个触发就记 1）
5. `affected_region_prior = delta_action_norm / sum(delta_action_norm)`（归一化）
6. 把区域信号重采样到固定长度 15，得到 `region_target_15`

最终 `pseudo_labels` 里常见字段：
- `trigger_label`
- `risk_score`
- `affected_region_prior`
- `correction_mask`
- `delta_action_norm`

---

## 3) A 输出（a_outputs）怎么构建

A 模块训练消费的 contract 固定是：
- `risk_pred`
- `trigger_logit`
- `delta_pred`
- `region_logits`（长度 15）
- `dynamic_embedding`（长度 16）
- `version`
- `source`

### 3.1 offline 构建（FASA）

说明（非常重要）：
- 在服务器当前常驻分支 `codex/tmp-20260402-p0-audit-infra`，`tools/build_fasa_a_outputs.py` 通常不存在。
- 当前服务器可直接执行的是 VDPM 统一构建脚本（见 3.2）。
- 如果你要读 FASA 规则化生成逻辑，可参考本地镜像脚本：
  `/Users/bazinga/code/my-starvla-v2/tools/build_fasa_a_outputs.py`

核心逻辑：
- `risk_pred = clip01(pseudo_labels.risk_score)`
- `trigger_logit = 2 * (clip01(trigger_label) - 0.5)`
- `delta_pred` 支持 scalar/list，list 时取有限值最大并截断到 `>=0`
- `region_logits` 由 `region_target_15`（或 prior/mask）转 logit
- `dynamic_embedding` 用 `state_t/future_state/horizon` 统计特征拼成 16 维

### 3.2 VDPM 构建（离线文件生成）

脚本：
- `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/tools/build_vdpm_a_outputs.py`

用途：
- 从外部预测或替代特征构建同一 contract 的 `a_outputs`
- 保证训练消费侧不需要改字段名

### 3.3 严格审计（强烈建议）

脚本（服务器当前分支可用）：
- `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/tools/a_outputs_audit.py`

补充：
- 一些分支会有 `tools/fasa_a_outputs_audit.py` 命名，但不是所有分支都有。

会检查：
- 字段是否齐全
- `region_logits` 是否长度 15
- `dynamic_embedding` 是否长度 16
- 数值是否 finite
- `delta_pred` 是否非负

---

## 4) A 模块在训练里怎么接入（重点）

核心代码：
- `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/starVLA/model/framework/a_module_interface.py`

### 4.1 两种总模式

`build_a_module_interface(...)` 支持：
- `lite`：模型内轻量 head 直接预测
- `standalone`：从样本 `a_outputs`（或 in-loop）读取

我们这条线主要用 `standalone`。

### 4.2 standalone 的三条来源路径

1. `offline + precomputed`
- 直接读样本里已有 `a_outputs`

2. `inloop + runtime_infer`
- 训练时按 `meta` 去 runtime jsonl 索引查找

3. `inloop + runtime_model`（real in-loop）
- 训练时真实调用 VDPM 模型推理
- 结果会带缓存，避免重复推理

### 4.3 real in-loop 的关键实现

同文件内嵌了 `_VDPM_WORKER_SCRIPT`，通过子进程长驻 worker 调 VDPM：
- 默认 Python：`/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/vdpm/.venv_vdpm_infer/bin/python`
- 默认 repo root：`/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/vdpm`

调用点类似：
- `model.inference(None, images=images.unsqueeze(0))`

这里的 `None` 是 VDPM 接口的占位参数（该入口主要用 `images`），不是“空跑”。真正输入是 `images`。

### 4.4 in-loop 审计指标（训练时会打）

你会看到这些指标：
- `inloop_calls`
- `inloop_success`
- `inloop_fail`
- `inloop_fallback_count`
- `vdpm_model_calls_total`
- `vdpm_cache_hit / vdpm_cache_miss / cache_hit_ratio`
- `vdpm_infer_latency_ms_mean / p95`

这些指标是判断“是不是真的 in-loop”最直接证据。

---

## 5) 损失怎么定义（QwenPI）

核心文件：
- `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/starVLA/model/framework/QwenPI.py`
- `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/starVLA/model/framework/optional_loss_utils.py`

### 5.1 target 从哪里来

`build_optional_hook_targets(...)` 负责把样本转成训练 target/mask：
- risk: `pseudo_labels.risk_score`
- trigger: `pseudo_labels.trigger_label`
- delta: `pseudo_labels.delta_action_norm`（支持 list/scalar）
- region: `pseudo_labels.correction_mask` + `affected_region_prior`
- embed: 优先 `a_outputs.dynamic_embedding`，否则 `pseudo.dynamic_embedding`

### 5.2 分解损失

`QwenPI` 中会计算并输出：
- `a_loss_risk`（MSE）
- `a_loss_trigger`（BCEWithLogits）
- `a_loss_embed`（MSE）
- `corrective_loss_delta`（MSE）
- `corrective_loss_region`（BCE + MSE 组合）

聚合关系：
- `a_loss = risk + trigger (+ embed, 看开关)`
- `corrective_loss = delta + region`

---

## 6) 训练日志里怎么确认“链路正常”

核心映射在：
- `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/starVLA/training/train_starvla.py`

关键指标落盘为：
- `loss/total`
- `loss/a_module`
- `loss/corrective`
- `loss/risk`
- `loss/trigger`
- `loss/delta`
- `loss/region`
- `loss/embed`

新人检查建议：
- `all_loss_finite=true`
- `loss/delta` 和 `loss/embed` 不是全 0
- `vdpm_model_calls_total > 0`（real in-loop 时）
- `fallback_to_precomputed_count` 符合你设定的策略

---

## 7) 新人最小上手流程（服务器）

### 7.1 先确认代码位置

```bash
cd /2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA
git branch --show-current
git rev-parse HEAD
```

### 7.2 看工具是否可用

```bash
python tools/build_vdpm_a_outputs.py --help
python tools/a_outputs_audit.py --help
```

如果你在本地补齐了 FASA 工具链，也可以额外验证：

```bash
python /Users/bazinga/code/my-starvla-v2/tools/build_fasa_a_outputs.py --help
```

### 7.3 训练侧关键开关（概念）

- `framework.a_module.mode=standalone`
- `framework.a_module.vdpm_mode=inloop`
- `framework.a_module.vdpm_source=runtime_model`
- `framework.a_module.vdpm_runtime_infer_enabled=true`
- `framework.a_module.vdpm_fallback_policy=strict_no_precomputed`（严格模式）

---

## 8) 最常见坑（你大概率会遇到）

1. 分支不对，脚本“看起来丢了”
- 先 `git branch --show-current`
- 再 `git ls-tree -r --name-only HEAD | rg 'build_fasa|build_vdpm|a_outputs_audit'`

2. 数据里 `delta_action_norm` 是 list，但你按 scalar 读
- 结果常见是 `loss/delta` 长期 0
- 正确做法是支持 list，取有限值聚合

3. in-loop 看起来开了，实际没调用模型
- 必看 `vdpm_model_calls_total`
- 只看 `inloop_calls` 不够

4. 合同没过还硬训
- 必须先跑审计
- `region_logits` 长度必须 15，`dynamic_embedding` 必须 16

---

## 9) 关键路径索引（服务器绝对路径）

代码：
- `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/starVLA/model/framework/a_module_interface.py`
- `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/starVLA/model/framework/QwenPI.py`
- `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/starVLA/model/framework/optional_loss_utils.py`
- `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/starVLA/training/train_starvla.py`

数据/构建工具：
- `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/tools/build_vdpm_a_outputs.py`
- `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/tools/a_outputs_audit.py`
- `/Users/bazinga/code/my-starvla-v2/tools/build_fasa_a_outputs.py`（本地镜像，便于理解 FASA 规则生成）

伪标签历史权威脚本（用于理解生成原理）：
- `/2025233147/zzq_0317/codex_runs/p4_1_fasa_phase0_libero_remote_20260409/tools/build_fasa_dataset.py`

文档：
- `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/docs/algorithm1/handoff/progress_live.md`

---

## 10) 一句话总结

这条线的本质是：
- 用稳定 contract 把“外部 A 预测”接入主训练
- 支持从 offline 过渡到 real in-loop
- 用严格审计 + 分解损失指标，证明训练真的在学到 `risk/trigger/delta/region/embed`，而不是只跑通表面流程。
