# ALG1-INFRA-20260403-005-OC 远程 `/2025233147/zzq_0317/starVLA` 兼容性审计

- 审计日期: 2026-04-03
- 审计目标: 判断该远程“已调通版本”是否可直接作为 `my-starvla-v2` 当前 retrofit 开发基线。

## 1) 结论摘要

1. 该远程仓库可连通、可读取，且是独立 git repo（分支 `starVLA`）。
2. 不建议“直接替换”当前本地基线。
3. 建议作为“定向补丁来源（feature donor）”使用，优先导入 dataloader 与 Qwen2.5 VLM 小补丁。

## 2) 关键证据

### 2.1 远程路径与仓库状态

- 远程可连通 (`ssh myserver`)。
- 路径存在：`/2025233147/zzq_0317/starVLA`。
- git 状态：`starVLA...origin/starVLA`，ahead/behind=`0/0`，但有未提交修改与未跟踪文件。

未提交文件（核心）:
1. `examples/LIBERO/train_files/run_libero_train.sh`
2. `examples/LIBERO/train_files/starvla_cotrain_libero.yaml`
3. `starVLA/model/framework/__init__.py`
4. `starVLA/model/modules/vlm/QWen2_5.py`
5. `starVLA/model/modules/vlm/__init__.py`
6. `starVLA/config/training/starvla_train_pi.yaml`（untracked）

### 2.2 与当前本地框架差异（关键文件）

按文件对比（本地 vs 远程）:
1. `starVLA/training/train_starvla.py`: 1065 行 vs 427 行（差异很大）。
2. `starVLA/dataloader/gr00t_lerobot/datasets.py`: 2188 行 vs 2785 行（远程更大，含 action_mode/stats cache 体系）。
3. `examples/LIBERO/eval_files/eval_libero.py`: 389 行 vs 298 行（本地包含更多调试与输入检查）。
4. `starVLA/model/framework/MapAnythingLlava3DPI.py`: 本地存在，远程缺失。

### 2.3 当前 retrofit 主线兼容性

1. 我们当前线程依赖 `MapAnythingLlava3DPI` 的审计与插入点；远程路径缺该框架文件，直接切换会破坏当前主线。
2. 本地 `train_starvla.py` 已有更多稳定性保护与日志扩展（如更细粒度 metrics 路径与训练防护），远程版本更轻量。
3. 远程 `results/Checkpoints` 下仅见目录骨架，未发现可直接用于验收的 `metrics.jsonl/summary.jsonl/config.yaml` 完整证据包。

## 3) 可以借鉴的高价值改动（建议导入）

1. dataloader 统计缓存与 action_mode 体系（远程最近提交 `d73a046` 指向该方向）：
   - 文件: `starVLA/dataloader/gr00t_lerobot/datasets.py`
   - 价值: 提升统计缓存一致性、支持 `action_mode` 规范化与映射。
2. Qwen2.5 VLM 的可配置注意力实现:
   - `starVLA/model/modules/vlm/QWen2_5.py` 把 `attn_implementation` 改为从配置读取。
3. Qwen2.5 路径匹配增强:
   - `starVLA/model/modules/vlm/__init__.py` 增加 `qwen2.5` 小写路径匹配。

## 4) 不建议直接导入的改动

1. `run_libero_train.sh` 的 NCCL/网卡环境变量与绝对路径（高度机器相关）。
2. 直接替换完整 `train_starvla.py`（会回退本地已有训练稳定性/调试增强）。
3. 直接替换 eval 脚本（会丢失本地已有调试与 payload 校验增强）。

## 5) 最小接入建议

1. 保持当前 `my-starvla-v2` 为主基线。
2. 新建一次“定向迁移”小任务，仅迁移上述 3 个高价值点并逐项验证。
3. 迁移后按现有 gate 流程做本地 smoke + 远程证据验收。

建议 commit message:
- `[ALG1-INFRA-20260403-005-OC] audit remote zzq_0317 starVLA and define selective import strategy`
