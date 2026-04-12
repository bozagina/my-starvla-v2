# Prompt for B Thread: Phase3.0R Fusion A-head Build

你是 B 线程实现负责人。请在以下代码树中执行 Phase3.0R 的最小可落地实现：

- 工作树：
  - `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410`

你必须先完整阅读：

- `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410/docs/starvla_retrofit/handoff/phase3_0r_vdpm_true_lite_a_head_research.md`
- `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410/starVLA/model/framework/a_module_interface.py`
- `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410/starVLA/model/framework/QwenPI.py`
- `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410/starVLA/model/framework/optional_loss_utils.py`
- `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410/starVLA/training/train_starvla.py`
- `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410/docs/algorithm1/handoff/progress_live.md`

## 你的任务

实现一个新的 `framework.a_module.mode=fusion` 路径，满足：

1. `VDPM runtime_model` 继续冻结并保持当前 in-loop `runtime_model` 机制
2. 主模型图内新增一个“真正 lite、非 MLP-only”的 trainable A-head
3. 融合输入必须至少包含：
   - `Qwen pooled_hidden`
   - `VDPM dynamic_embedding(16)`
4. 输出 contract 不允许破坏：
   - `risk_pred`
   - `trigger_logit`
   - `delta_pred`
   - `region_logits(len=15)`
   - `dynamic_embedding(len=16)`
5. 训练与日志键语义不允许破坏：
   - `loss/risk`
   - `loss/trigger`
   - `loss/delta`
   - `loss/region`
   - `loss/embed`
   - `loss/a_module`
   - `loss/corrective`
6. 旧模式 `lite/standalone` 不得回归
7. 不允许静默 fallback 冒充成功；任何 fallback 必须可审计

## 唯一推荐实现

先只实现主线，不要并行做多个版本：

- `CrossAttentionFusionAHead`

结构要求：

- `pooled_hidden[2048] -> q_proj -> q_token[256]`
- `dynamic_embedding[16] -> split 4x4 -> slot_proj -> 4 vdpm slots[256]`
- memory token 由：
  - `q_token`
  - `vdpm_global_token`
  - `4 vdpm slots`
  组成
- learned queries：
  - `1 global query`
  - `15 region queries`
  - `4 dynamic queries`
- `2-layer cross-attn decoder`
- heads：
  - `global -> risk_pred / trigger_logit / delta_pred`
  - `15 region queries -> region_logits_contract15`
  - `4 dynamic queries -> dynamic_embedding_pred16`
- 必须显式加入：
  - `region_logits_contract15 -> region_logits_train[chunk_len]`

注意：

- 主线只消费 `VDPM dynamic_embedding(16)` 作为 frozen VDPM 特征
- 不要把 frozen `risk/trigger/delta/region` 再喂回融合头做主输入

## 精确工程要求

你需要新增或改造：

### 1. 新增文件

- `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410/starVLA/model/framework/a_fusion_heads.py`

至少包含：

- `AModuleRuntimeBatch`
- `Region15ToChunkAdapter`
- `CrossAttentionFusionAHead`

### 2. 修改 `a_module_interface.py`

要求：

- 为 `AModulePredictions` 增加可选字段：
  - `dynamic_embedding_pred`
  - `region_logits_contract15`
- 在 `StandaloneAModuleInterface` 中增加 runtime feature collect 能力，复用现有：
  - `_extract_source`
  - runtime/cache/audit
- 在 `build_a_module_interface(...)` 中增加：
  - `mode=fusion`

重要：

- `lite`
- `standalone`

两条旧路径必须保持原行为。

### 3. 修改 `QwenPI.py`

要求：

- 在 `__init__` 中按 mode 决定是否创建 `self.a_fusion_head`
- 在 `_compute_optional_hook_outputs(...)` 中接入 fusion 路径
- 保持当前 target build 顺序：
  - 先 runtime collect / predict
  - 再 build optional targets
- 保持旧 `loss/*` 命名不变

### 4. 不要改以下受保护文件

- `/Users/bazinga/code/my-starvla-v2/starVLA/config/training/starvla_train_pi_qwen25.yaml`
- `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/p4_1_qwen25_longrun_authoritative_baseline.md`

启用新模式请使用：

- 新增非保护实验 yaml
- 或 CLI override

## 门禁要求

你至少要完成并交付以下最小证据：

### Train-50

必须满足：

- `all_loss_finite=true`
- `loss/risk|trigger|delta|region|embed` 全 present
- `max(loss/delta) > 0`
- `max(loss/embed) > 0`
- `debug/a_module_mode_fusion=1.0`

### Train-500

必须满足：

- `all_loss_finite=true`
- `nonzero_ratio_loss_delta=1.0`
- `nonzero_ratio_loss_embed=1.0`
- `freeze_assert.json.gate_pass=true`
- `noise_assert.json.gate_pass=true`
- `vdpm_model_calls_total > 0`
- `fallback_to_precomputed_count=0`

### Eval smoke

必须满足：

- `pass=true`
- `blocked=false`
- `traceback/valueerror/runtimeerror=0`

### AB compare vs 029

请生成：

- `ab_compare_fusion_vs_029.json`

判据不要用 raw loss 更小作为硬门禁。请沿用 029 的 gate-vector 风格，要求：

- `train_50_gate_pass=true`
- `train_500_gate_pass=true`
- `candidate_train_50_all_loss_finite=true`
- `candidate_train_500_all_loss_finite=true`
- `nonzero_ratio_loss_delta=1.0`
- `nonzero_ratio_loss_embed=1.0`
- `vdpm_model_calls_total > 0`
- `fallback_to_precomputed_count=0`
- `eval_smoke.pass=true`
- `eval_smoke` 无错误扫描命中
- 新增：
  - `debug/a_module_mode_fusion=1.0`

## 你需要特别注意的风险

1. 不要让 `region_logits(len=15)` 在训练里被隐式截断成 `chunk_len`
2. 不要把 fusion 头做成“大一点的 MLP”
3. 不要让 fallback 伪装成 fusion 成功
4. 不要破坏旧 `lite/standalone`
5. 如果新增 module 没有进入独立 lr group，请至少在交付里明确说明它当前落在哪个 param group，避免 silent base-lr

## 交付格式

你完成后必须给出：

1. 改动文件清单
2. 每个文件改了什么
3. `train_50` 门禁结果
4. `train_500` 门禁结果
5. `eval smoke` 结果
6. `ab_compare_fusion_vs_029.json` 结论
7. 风险与后续建议

## 记录要求

任何实质性修改或实验结论，都必须 append 到：

- `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410/docs/algorithm1/handoff/progress_live.md`

禁止重写旧记录。

