# Mask 诊断与后续开发总文档（截至 2026-03-01）

> 目标读者：后续接手该方向的大模型/工程师。  
> 目标用途：作为单一事实来源，快速恢复上下文、复现实验、继续开发。

---

## 0. 一页结论（TL;DR）

1. 当前训练中确实存在 `LLaVA3D vision tower path is empty or None, using SigLIP vision tower instead` 的 fallback。
2. 但把视觉分支替换为“官方 CLIP tower + 当前 mm_projector”后，mask 几乎不变（alpha cosine 约 0.996）。
3. 进一步接入官方 checkpoint（做 image-only 兼容 patch）后，mask 仍几乎不变（alpha cosine 约 0.9957）。
4. 因此，“语言/视觉来自不同模型导致 mask 偏平”不是主因。
5. 当前核心矛盾是：**希望 mask 具备强可解释性和尖锐选择性，但现有损失对 mask 的直接监督不足，导致平坦但可用的局部最优。**

---

## 1. 任务背景与问题定义

本轮排查聚焦 `Path-A causal feedback` 中 soft-mask 的可分性问题。用户连续提出的问题可归纳为：

1. mask 现在到底由什么决定（计算路径与回退条件）？
2. 偏平是“表征不可分”还是“监督太弱”？
3. 是否是 `force_uniform_mask` 或 soft-mask 构建失败导致退化成 `1/N`？
4. 是否是语言和视觉来自不同模型（LLaVA language + SigLIP/其他 vision）造成对齐失败？
5. 如果替换成其他模型，改动成本多大？

---

## 2. 关键代码地图（必须知道）

### 2.1 训练主流程与 mask 作用点

- `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py`
  - `_build_causal_feedback_tokens(...)`：生成 feedback token 的主入口。
  - `_build_soft_mask(...)`：计算 soft-mask `alpha`。
  - `_compute_causal_feedback_aux_loss(...)`：feedback 辅助损失。

### 2.2 视觉分支 fallback 逻辑

- `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py`
  - `get_image_features(...)` 内部控制 `base_model.encode_images` vs SigLIP fallback。
  - `_llava_vision_available` 状态在失败后会被置为 `False`。

### 2.3 新增诊断脚本（本轮核心产物）

- `/Users/bazinga/code/my-starvla/tools/diagnose_mask_vision_compare.py`
  - current 分支：按现有训练逻辑取 `vision/geometric/language_queries` 计算 mask；
  - llava rebuild 分支：用本地 LLaVA-3D repo 构建官方视觉塔，再接当前 mm_projector；
  - official ckpt 分支：直接加载官方 checkpoint（含兼容 patch）并计算可比 mask。

### 2.4 官方仓库参考位点

- `/Users/bazinga/code/my-starvla/LLaVA-3D/llava/model/llava_arch.py`
- `/Users/bazinga/code/my-starvla/LLaVA-3D/llava/model/builder.py`
- `/Users/bazinga/code/my-starvla/LLaVA-3D/llava/model/multimodal_encoder/clip_encoder.py`
- `/Users/bazinga/code/my-starvla/LLaVA-3D/llava/model/multimodal_encoder/video_encoder.py`

---

## 3. 当前配置快照（与结论强相关）

来源：训练配置 `..._v4_1_1.yaml`（用户当前使用）。

### 3.1 soft-mask 相关

- `soft_mask_enabled: true`
- `soft_mask_channel_mode: vision_only`
- `soft_mask_query_agg: max`
- `soft_mask_head_agg: max`
- `soft_mask_num_heads: 4`
- `soft_mask_logit_scale: 16`
- `soft_mask_temperature: 1.0`
- `soft_mask_score_norm: l2_only`
- `soft_mask_use_ema_inference: true`
- `soft_mask_use_ema_training: false`

### 3.2 feedback/监督强度相关

- `feedback_mask_contrast_enabled: true`
- `feedback_mask_contrast_weight: 0.03`（较低）
- `feedback_mask_contrast_start_step: 50`
- `causal_feedback_aux_weight: 0.05`
- `causal_feedback_aux_dir_weight: 0.1`
- `feedback_delta_action_alpha_mode: schedule`
- `feedback_delta_action_alpha_target: 0.1`
- `feedback_delta_action_last_layer_scale: 1e-3`
- `feedback_delta_action_clip: 0.05`

解释：这些配置整体偏“保守增益”，使 feedback 对主动作 loss 的影响有限。

---

## 4. mask 的严格定义与实现细节（必须理解）

### 4.1 最终 mask 形状

- 最终使用的 `alpha` 形状是 `[B, N]`，其中 `N = min(Nv, Ng)`（你当前常见为 `256`）。
- 应用时变为 `[B, N, 1]` 与 `delta_tokens [B, N, H]` 相乘。

### 4.2 形成过程（对应 `_build_soft_mask`）

1. 输入：
   - `language_queries q: [B, Lq, H]`
   - `vision_tokens v: [B, Nv, H]`
   - `geometric_tokens g: [B, Ng, H]`
2. 截断到同 token 数 `N` 后，做 head 拆分。
3. 计算相似度 logits：
   - `logits_vis = <q, v> * scale`
   - `logits_geo = <q, g> * scale`
4. 对 token 维 softmax，得到 query-to-token 注意力。
5. 先按 `query_agg` 聚合，再按 `head_agg` 聚合。
6. 按 `channel_mode` 融合 `alpha_vis/alpha_geo`，再归一化成概率分布。
7. 训练/推理可选 EMA 平滑。

### 4.3 uniform mask 回退条件

`_build_causal_feedback_tokens(...)` 中会回退 `1/N` 的情况：

1. 显式 `force_uniform_mask=True`；
2. soft-mask 返回 `None`（输入缺失/ndim 不合法/batch 或 hidden 不匹配等）；
3. soft-mask 返回形状与 `token_n` 不匹配。

补充：`force_uniform_mask=True` 主要用于构造 “unmasked baseline” 对比项，不是主训练常态。

---

## 5. 视觉 tower fallback 机制（现状）

在 `get_image_features(...)` 中，LLaVA 视觉路径被禁用的典型条件：

1. `vision_tower_path` 是空/None；
2. `vision_tower.load_model()` 失败；
3. `base_model.encode_images(...)` 抛异常。

一旦触发：

- 转为 SigLIP 路径；
- `_llava_vision_available = False`。

这解释了日志里的 fallback 提示，但不等于“它就是 mask 偏平根因”。

---

## 6. 诊断脚本改造历史（工程变更记录）

文件：`/Users/bazinga/code/my-starvla/tools/diagnose_mask_vision_compare.py`

### 6.1 初版

- 复现 current mask 统计；
- 输出 `vision_tower_info`。

### 6.2 加入 llava rebuild 分支

- 从本地 `LLaVA-3D` 读取官方 `mm_vision_tower`；
- 构建 CLIP vision tower；
- 用当前模型内 mm_projector 投影到 4096；
- 与 current 分支做 alpha diff。

### 6.3 加入 official ckpt 分支

- 参数：`--official-llava-ckpt`；
- 直接加载官方 checkpoint 并计算可比 mask。

### 6.4 兼容处理

- `torch_scatter` ABI 失败时：注入 stub `scatter_mean`；
- official 分支 `video_tower` 缺失时：
  - 注入 dummy video tower（提供 `video_tower.encode_pe` 和 `prompt_encoder`）；
  - patch `encode_images`/`encode_prompts`；
  - 记录 `official_patch_info`；
  - 报错时输出 `official_llava_error_traceback`。

---

## 7. 尝试与观察（按时间顺序）

### 7.1 尝试 A：验证是否已退化为均匀

观测：

- `llava_vision_available_flag=false`，`vision_tower_path=null`；
- mask 指标“高熵但非纯均匀”：
  - `entropy ~ 5.5428`（均匀 256 时是 `ln(256)=5.545177`）
  - `alpha_max_mean ~ 0.00507`（均匀是 `0.00390625`）
  - `topk_mass_32 ~ 0.1418`（均匀是 `0.125`）

结论：偏平，但不是硬退化成 `1/N`。

### 7.2 尝试 B：直接走当前模型 LLaVA encode

观测：

- 报 `NoneType object is not callable`；
- mm_projector 取值路径曾不稳定。

结论：只依赖当前 `base_vlm` 内残余信息不足以稳定复原视觉链路。

### 7.3 尝试 C：llava rebuild（官方视觉塔 + 当前 mm_projector）

观测：

- `resolved_vision_tower_name = openai/clip-vit-large-patch14-336`
- `raw_vision_feats_shape = [1,576,1024]`
- `projected_vision_feats_shape = [1,576,4096]`
- 与 current 差异极小（多次在以下区间）：
  - `alpha_cosine_mean ~ 0.9956 - 0.9967`
  - `alpha_l1_mean ~ 2.4e-4 - 2.8e-4`
  - `alpha_l2_rms ~ 3.1e-4 - 3.7e-4`

结论：替换视觉塔来源并未显著改变 mask。

### 7.4 尝试 D：official ckpt 分支第一次落地

观测：

- 脚本版本落后导致参数不识别；
- 修复后遇到 `torch_scatter` ABI 问题。

结论：先打通工程链路，再讨论算法结论。

### 7.5 尝试 E：official ckpt 分支第二次

观测：

- 错误转为 `'NoneType' object has no attribute 'video_tower'`。

根因：

- 官方 image path 仍隐式依赖 video tower 的接口。

结论：必须做 image-only 兼容 patch。

### 7.6 尝试 F：official 分支最终成功

关键观测（最终 JSON）：

- `official_patch_info.video_tower_missing = true`
- `official_patch_info.injected_dummy_video_tower = true`
- `official_patch_info.patched_encode_images = true`
- `official_patch_info.patched_encode_prompts = true`

mask 对比：

- `current_vs_llava_alpha_diff.alpha_cosine_mean = 0.995620`
- `current_vs_official_alpha_diff.alpha_cosine_mean = 0.995694`

结论：

- 即使 official 分支跑通，mask 与 current 分支依旧高度一致。

---

## 8. 定量结果总表（供后续回归比较）

基于最终一次 successful run：

| 指标 | current | llava_rebuild | official |
|---|---:|---:|---:|
| token_n | 256.0 | 256.0 | 256.0 |
| attn_top1_mean_vis | 0.00532368 | 0.00504932 | 0.00488507 |
| alpha_entropy | 5.5428238 | 5.5436287 | 5.5430450 |
| alpha_max_mean | 0.00506717 | 0.00469321 | 0.00451871 |
| topk_mass_32 | 0.1417938 | 0.1374233 | 0.1408636 |

差异：

- current vs llava_rebuild:
  - cosine `0.9956198`
  - l1 `2.7778e-4`
  - l2_rms `3.6635e-4`
- current vs official:
  - cosine `0.9956940`
  - l1 `2.7868e-4`
  - l2_rms `3.6333e-4`

解释口径：在该任务样本上，三者分布差异远小于“根因级差异”应有量级。

---

## 9. 证据链结论（最终口径）

1. fallback 现象为真，但不是主要矛盾；
2. 视觉塔来源替换后 mask 几乎不变，排除“tower 错配是主因”的假设；
3. 问题更接近：优化目标对 mask 的直接约束不足；
4. 当前实验更支持“监督太弱/太间接”，不支持“表征天然不可分”是主导解释。

---

## 10. 当前最核心矛盾与本质

### 10.1 核心矛盾

**想要可解释、尖锐、可区分的 mask；但训练信号主要考核动作效果，不直接考核 mask 本身。**

### 10.2 本质

是 `identifiability weak + indirect supervision` 的组合问题：

- mask 的自由度较高，但缺少强外部锚点；
- 反馈分支增益与对比权重偏小；
- 主任务可在“偏平 mask”下取得可接受损失，导致平坦解稳定存在。

---

## 11. 给后续接手者的工程事实与注意事项

### 11.1 不要误解的点

1. `llava_vision_available_flag=false` 不等于“mask 一定坏掉”；
2. 有 uniform fallback，但当前观测不是纯 uniform；
3. official 分支能跑通依赖 image-only patch（不是原生 video 全链路行为）。

### 11.2 当前诊断脚本边界

1. 主要在单图+单指令样本上验证；
2. 结论已强烈指向非 tower 根因，但仍建议扩展到小批量统计；
3. official 分支 patch 是“诊断便利性改造”，不是训练代码路径。

### 11.3 复现命令模板

```bash
python /Users/bazinga/code/my-starvla/tools/diagnose_mask_vision_compare.py \
  --run-config /Users/bazinga/code/my-starvla/starVLA/config/training/starvla_train_libero_mapanything_llava3d_ab_b_concat_cross_geometric_alg1_v4_1_1.yaml \
  --image /2025233147/zzq/SpatialVLA_llava3d/test/example.png \
  --instruction "What action should the robot take to pick the cup?" \
  --device cuda \
  --llava-repo /Users/bazinga/code/my-starvla/LLaVA-3D \
  --official-llava-ckpt /2025233147/zzq/mapAnythingLlava3dPi0.5/model_zoo/llava3d \
  --output-json /Users/bazinga/code/my-starvla/tools/mask_vision_compare_official.json
```

---

## 12. 后续开发建议（算法 + 重构视角）

### 12.1 优先级 P0（先做）

1. 先加“可批量统计”的诊断脚本入口（10-100 样本），输出均值/方差，而非单样本。
2. 在训练日志固定追踪以下指标（已有）：
   - `soft_mask_entropy`
   - `soft_mask_topk_mass_32`
   - `soft_mask_alpha_max_mean`
   - `feedback_mask_contrast_*`
   - `delta_action_*`
3. 设立回归阈值：若新改动使 `alpha_cosine(current,baseline)` 长期 >0.995 且主指标不变，则判定“改动无效”。

### 12.2 优先级 P1（核心算法）

1. 强化 mask 直接监督（比单纯换 tower 更关键）：
   - 提高或重构 `feedback_mask_contrast`；
   - 增加熵/稀疏正则；
   - 如可行，引入 token-level 或区域级外部监督（bbox/seg proxy）。
2. 结构升级：可试“多层有向 masked self-attn 生成 mask”：
   - token 拼接 `[L,V,G]`；
   - 用 block attention mask 控制方向（例如 `L->V/G`, `V->G`, `G->G`）；
   - 最终投影到 `[B,N]` mask。
3. 训练稳定策略：
   - 新旧 mask 混合 warmup（`alpha = (1-r) old + r new`）；
   - 分阶段拉高 feedback 分支权重，避免早期噪声主导。

### 12.3 优先级 P2（工程重构）

1. 将 mask 诊断统计抽成独立模块，避免重复实现（训练和脚本共用）。
2. 将 official 分支 patch 代码与主流程隔离到 `tools/` 私有 helper，避免误入训练链路。
3. 统一 mm_projector 查找逻辑，减少属性路径脆弱性。

---

## 13. 未来评估标准（建议）

“有效改动”建议同时满足：

1. `soft_mask_topk_mass_32` 稳定提升，且 `soft_mask_entropy` 稳定下降；
2. `feedback_mask_contrast_term` 有效工作，不只是可用标志；
3. 动作主指标不退化（或在统计上可接受）；
4. 关键是多种样本上成立，而非单样本偶然。

---

## 14. 已知风险与限制

1. 本结论基于当前配置与样本分布，跨数据域需复核；
2. official 分支是 image-only 兼容运行，不代表完整 video 功能对齐；
3. 如果后续大改 query 构造/融合层，本文结论需重新校验。

---

## 15. 最终判断（供接手者直接使用）

**当前最核心矛盾不是“视觉塔来源错配”，而是“mask 监督信号不足导致可解释性目标不可辨识”。**  
优先方向应是监督与目标函数设计，而不是继续在 tower 替换上投入主要精力。

---

## 16. 协作文档（新增）

后续接手请强制使用以下文档：

1. 实时进度日志（每次实质动作后追加）  
   `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/progress_live.md`
2. 项目级系统提示词与执行契约  
   `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/system_prompt_operating_contract.md`
3. 实验编号规范 + 命名约定  
   `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/experiment_id_and_naming_convention.md`
4. 新会话启动命令模板  
   `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/new_chat_bootstrap_command.md`
5. 本地开发/远程训练协作流程  
   `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/remote_training_workflow.md`
