# ALG1-INFRA-20260302-002-OC: LLaVA3D -> Qwen2.5-VL 替换基础设施评估

## 1) 结论先行

- 当前 `MapAnythingLlava3DPI` 不是“可直接换 base_vlm 字符串”的松耦合架构，而是与 `MapAnything + LLaVA3D` 深度绑定。
- 如果目标是保留现有 Path-A（soft-mask / token_delta_geo / teacher KL / 诊断体系），建议走“兼容层改造”路线，而不是直接切换到现有 `QwenPI` 系列框架。
- 预计是中到大规模改造，建议拆成 3 阶段（接口契约对齐 -> 训练链路打通 -> 远程 A/B 验证）。

## 2) 关键耦合点（现状证据）

### 2.1 框架入口与 VLM 选择

- `get_vlm_model` 通过 `base_vlm` 字符串匹配实现分流：若包含 `Qwen2.5-VL`，会返回 `_QWen_VL_Interface`。见：
  - `/Users/bazinga/code/my-starvla/starVLA/model/modules/vlm/__init__.py:24`
- 但 `MapAnythingLlava3DPI` 初始化时硬读取 `self.mapanythingllava3d_vlm_interface.model.language_model.model`。见：
  - `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:66`
- 现有 `_QWen_VL_Interface.model` 是 `Qwen2_5_VLForConditionalGeneration`，并无 `language_model.model` 层级。见：
  - `/Users/bazinga/code/my-starvla/starVLA/model/modules/vlm/QWen2_5.py:96`

### 2.2 Path-A 前向契约

`MapAnythingLlava3DPI.forward` 依赖专有接口/输出：

- 输入构造函数名固定：`build_mapanythingllava3d_inputs`。见：
  - `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:2053`
- 输出字段固定依赖：
  - `task_hidden_states`、`geometric_hidden_states`、`vision_hidden_states`、`vision_hidden_states_raw`、`language_queries`、`language_query_mask`。
  - 见：`/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:2088`

### 2.3 MapAnything-LLaVA3D 模型内部绑定

- 当前组合模型 `MapAnythingLlava3DForConditionalGeneration` 内部语言侧硬绑定 `LLaVA3DForCausalLMV2`。见：
  - `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py:86`
- `LLaVA3DForCausalLMV2` 直接引用 `LLaVA_3D` 代码路径和类。见：
  - `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_llava3d_v2.py:19`
- 处理器也引用 `LLaVA_3D.llava.constants` 的 `<image>`/索引语义。见：
  - `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/processing_mapanything_llava3d.py:28`

### 2.4 训练与脚本层的 MapAnything 命名耦合

- 训练构建日志读取 `cfg.framework.mapanything_llava3d.base_vlm`。见：
  - `/Users/bazinga/code/my-starvla/starVLA/training/train_starvla.py:118`
- runtime fingerprint 默认取 `mapanythingllava3d_vlm_interface`。见：
  - `/Users/bazinga/code/my-starvla/starVLA/training/train_starvla.py:328`
- action head 读取 `framework.mapanything_llava3d.{num_vl_layers, vl_hidden_dim}`。见：
  - `/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:485`
- 远程训练脚本仍硬编码 `MapAnythingLlava3DPI + --framework.mapanything_llava3d.*`。见：
  - `/Users/bazinga/code/my-starvla/_remote_runs/run_libero_train_alg1.sh:17`

## 3) 如果替换为 Qwen2.5-VL，需要改哪些层

## A. 接口与契约层（必须）

1. 新增“MapAnything 兼容 Qwen”接口（建议新增文件，不破坏旧接口）：
   - 目标：对 `MapAnythingLlava3DPI` 暴露同名方法与同形输出：
     - `build_mapanythingllava3d_inputs(...)`
     - `extract_task_tokens(...)`
     - `forward(...)` 返回上述 Path-A 依赖字段。
2. 修改 `get_vlm_model` 路由逻辑：
   - 不再仅依赖 `base_vlm` 字符串猜测。
   - 增加显式后端字段（例如 `framework.mapanything_llava3d.vlm_backend=qwen2_5_vl`）。

## B. VLM 组合模型层（必须）

1. 新建 `MapAnythingQwenVLForConditionalGeneration`（建议），复用 `MapAnything` 几何分支与 task-token 构造逻辑。
2. 将语言模型调用从 `LLaVA3DForCausalLMV2` 抽象为 HuggingFace 通用 causal-VLM 路径（Qwen2.5-VL）。
3. 明确并实现输出对齐：
   - `hidden_states`（供 action head 选层）
   - `task_hidden_states`
   - `geometric_hidden_states`
   - `vision_hidden_states` / `vision_hidden_states_raw`
   - `language_queries` / `language_query_mask`

## C. 输入处理与 token 语义层（必须）

1. 把 `MapAnythingLlava3DProcessor` 与 LLaVA 常量解耦：
   - `<image>` token、image token index、instruction mask 的生成逻辑，需要抽象成 backend-aware。
2. Qwen 的多图/视频输入（`process_vision_info`）需与当前多视角 batch 组织规则对齐。
3. 核对 `image token` 展开方式：
   - 目前 MapAnything 处理器按 patch 序列长度展开 `<image>`；Qwen chat-template 路径不一定同构，需做一致性桥接。

## D. 配置与训练脚本层（必须）

1. YAML schema 扩展：
   - 在 `framework.mapanything_llava3d` 下增加 qwen backend 所需字段（model id、processor 细项、是否启用几何分支等）。
2. `train_starvla.py` 与监控代码做兼容：
   - 读取 base_vlm、fingerprint、debug 字段时避免写死 mapanythingllava3d 接口名。
3. `_remote_runs/run_libero_train_alg1.sh` 增加后端分支参数注入：
   - 支持 `qwen2_5_vl` 路径，不再默认只注入 `framework.mapanything_llava3d.base_vlm`。

## E. 优化器分组 / 冻结策略（必须）

1. 现有学习率组大量写死在 `mapanythingllava3d_vlm_interface.model.*` 路径。
2. 换后端后需重建参数组映射（language / vision / geometric / fusion / action），否则会 silently 落回 base lr。

## F. 工具链与诊断层（高优先）

1. 下列工具当前是 LLaVA3D 绑定，需要补 qwen 版本或做后端分支：
   - `tools/diagnose_mask_vision_compare.py`
   - `tools/build_mapanything_llava3d_base.py`
2. 保证 Path-A 核心指标链不断：
   - `soft_mask_*`、`feedback_mask_contrast_*`、`delta_action_*`、teacher reliability。

## G. 检查点与迁移策略（必须决策）

1. 旧 checkpoint 不能直接无损加载到 Qwen 后端（语言/视觉参数空间和命名不同）。
2. 需要确定迁移策略：
   - 仅迁移 action head + 几何相关层（建议首选）。
   - 或从头训练（成本高）。

## 4) 两条可行路线

### 路线 1（推荐）：保留 Path-A，新增 Qwen 兼容后端

- 优点：保留当前算法资产（mask、teacher、诊断、回归指标）。
- 代价：中大改造，涉及接口、模型、处理器、脚本多层。

### 路线 2（快速）：切到现有 QwenPI/QwenAdapter 框架

- 优点：更快跑通 Qwen。
- 代价：基本不复用当前 ALG1 Path-A 资产，等于换研发主线。

## 5) 建议执行顺序（最小可执行）

1. 先做“契约适配 PoC”（不改算法）：
   - 目标：`MapAnythingLlava3DPI.forward` 在 qwen 后端下单步可跑、指标可打。
2. 再做“输入/输出一致性验证”：
   - 核对 hidden size、layer count、task token 形状、image token 对齐。
3. 最后上远程 smoke：
   - 仅 200~500 step，先验证无崩溃、指标可观测，再进入 A/B。

## 6) 风险与预估

- 最大风险不是“能否加载 Qwen”，而是“Path-A 关键中间量契约是否保持同分布”。
- 预计工作量：
  - P0 打通（可运行）：3~5 天
  - P1 对齐（可比实验）：5~10 天
  - P2 稳定（可用于主实验）：>10 天（取决于远程 A/B 迭代）

## 7) 当前建议

- 不建议直接把 `framework.mapanything_llava3d.base_vlm` 改成 `Qwen/Qwen2.5-VL-*` 后直接开跑；按当前代码会出现接口/属性不匹配问题。
- 建议先做“兼容层+契约适配”最小改造，再进入远程 smoke。
