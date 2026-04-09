# main 分支验证清单（P0 执行版）

- EXP_ID: `ALG1-INFRA-20260305-001-OC`
- 目的：优先完成 `main` 分支“3D VLM + action model”是否有效的证据闭环
- 适用：阶段1（先验证价值，再决定是否恢复阶段2扩展）

---

## 1. 必跑矩阵（统一预算）

| 格子ID | VLM | Geometric | Steps | 状态 | run_id | 结论 |
|---|---|---|---:|---|---|---|
| M00 | 2D | Off | 20k | TODO |  |  |
| M01 | 2D | On | 20k | TODO |  |  |
| M10 | 3D | Off | 20k | TODO |  |  |
| M11 | 3D | On | 20k | TODO |  |  |

---

## 2. 每格必须附带的证据

1. `run_identity.txt`
2. `config.yaml`
3. `metrics.jsonl`
4. `summary.jsonl`
5. `train.log`（或 `train.raw.log`）

没有以上五项，不进入最终对照表。

---

## 3. 每格固定输出指标

- 任务效果：`Success@LIBERO`
- 训练代理：`action_dit_loss`（固定窗口统计）
- 时延：`p95 latency`
- 稳定性：nonfinite/崩溃/异常中断

建议固定三个窗口：`2k / 5k / 10k`（或同等步数）

---

## 4. 判定门槛（建议）

- Gate-A（3D是否优于2D）：`Success@LIBERO` 提升 `>= +3pp`，且 `p95 latency` 不劣化超过 `20ms`。
- Gate-B（geo是否有效）：`with_geo - without_geo >= +2pp`。

硬回退判定（必须执行）：

- 若统一预算下同时满足：
  - `M10 < M00`（3D无geo劣于2D无geo）
  - `M11 < M01`（3D有geo劣于2D有geo）
- 则触发战略切换：
  - 当前主线切换为 `2D VLM + Path-A`；
  - 3D VLM 主线降级为储备方向（不占用当前主资源）。

---

## 5. 执行顺序（建议）

1. 先完成 `M00` 与 `M10`（先回答“2D vs 3D本体差异”）。
2. 再补 `M01` 与 `M11`（回答“几何注入是否放大优势”）。
3. 选一个关键格子复现实验（确认可重复）。
4. 形成阶段1结论，再决定阶段2是否恢复扩展。
5. 若触发硬回退判定，立即产出“2D基座Path-A迁移计划”（含里程碑与资源重排）。

---

## 6. 报告回填位置

- 详报章节：`H1/H5/J0/J3`
- 文件：`/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/ALG1-INFRA-20260305-001-OC__3d_vla_project_detail_v1.md`

---

## 7. 历史10k预结果记录（无 run_id）

建议 **放入报告**，但必须按“预观察”标记，不能用于最终分叉决策。

- 状态标签：`PRELIMINARY_WEAK_TRACEABILITY`
- 适用对象：仅限目前历史 checkpoint（3D with/without geo 的 10k 结果）
- 约束：在20k统一预算完整结果出来前，不触发 Gate-A / Gate-B / 硬回退判定

建议记录字段：

1. checkpoint 路径（绝对路径）
2. 对应配置快照（至少记录关键超参数）
3. 评测时间与评测脚本版本
4. 初步指标（Success / loss / latency）
5. 风险标签（无 run_id，弱可追溯）
6. 后续动作（继续训练到20k并复核）
