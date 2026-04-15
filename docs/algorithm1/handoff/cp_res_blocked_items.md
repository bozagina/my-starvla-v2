# CP-RES Blocked Items List

```
status:     DELIVERED
author:     CP-RES thread
thread_id:  CP-RES
round_id:   CP-ROUND-BOOTSTRAP-WAIT-UPSTREAM
gate:       phase3_0b_cp_research
date:       2026-04-15
```

---

## 结论摘要

**CP-RES 本轮无阻塞项。**  
上游 A-REVIEW 已发布 PASS_CONFIRMED + downstream_usability=USABLE，CP-RES 门控条件已满足。

---

## 1. 上游阻塞项

| 项 | 描述 | 状态 | 解封证据 |
|---|---|---|---|
| A-output contract 稳定性 | CP-RES 必须等待 A-REVIEW 确认下游可用性 | **UNBLOCKED** | `a_module_current_round.yaml`: A-REVIEW.downstream_usability=USABLE, reviewed_at 2026-04-15T00:30:00+08:00 |

---

## 2. 远程阻塞项

| 项 | 描述 | 状态 | 负责方 | 预计解锁时机 |
|---|---|---|---|---|
| CP-AH-4: `Success@LIBERO` eval | 需要 corrective policy 完成且在远程 GPU 服务器执行 LIBERO 评估 | **BLOCKED_WAIT_REMOTE** | OC（远程执行）| CP-BUILD 交付后 |

> **说明**：CP-AH-4 是端到端验收，必须有远程证据。在 CP-BUILD 产出可训练的 corrective policy 代码并在服务器完成评估之前，此项保持 BLOCKED_WAIT_REMOTE。

---

## 3. CP-BUILD 侧悬挂决策项（非阻塞 CP-RES，但阻塞 CP-BUILD 实现）

以下为 CP-RES 已识别但属于 CP-BUILD 决策范围的开放项，CP-BUILD 必须在其交付物中解决：

| # | 决策项 | 当前状态 | 阻塞方 |
|---|---|---|---|
| BD-1 | 推理时 action 混合机制（加法 / 条件替换 / 门控）| 未决 | CP-BUILD |
| BD-2 | 是否在 v0 中引入独立的 corrective action chunk 输出头 | 未决 | CP-BUILD |
| BD-3 | A-module inference mode：lite vs standalone | 未决（建议 lite）| CP-BUILD |
| BD-4 | 是否消费 `dynamic_embedding[16]` | 未决（建议否）| CP-BUILD |
| BD-5 | 是否消费 `region_target_15`（15-bin region）| 未决（建议否）| CP-BUILD |

---

## 4. CP-REVIEW 侧待解锁项

| 项 | 状态 | 解锁条件 |
|---|---|---|
| CP-RES 产出物审查 | 等待 CP-REVIEW 执行 | CP-REVIEW 需审核本 RFC 和 interface assumptions 是否自洽 |
| CP-BUILD 代码审查 | BLOCKED（CP-BUILD 未交付）| CP-BUILD 交付后 |
| CP-AH 全套验收 | BLOCKED（CP-BUILD 未交付 + 远程未验证）| CP-BUILD + 远程证据 |

---

## 5. CP-RES 内部完成声明

CP-RES 本轮所有必须产出物已交付：

| 工件 | 路径 | 状态 |
|---|---|---|
| Policy RFC | `docs/algorithm1/handoff/cp_res_policy_rfc.md` | DELIVERED |
| Interface Assumptions | `docs/algorithm1/handoff/cp_res_interface_assumptions.md` | DELIVERED |
| Blocked Items List | `docs/algorithm1/handoff/cp_res_blocked_items.md` | DELIVERED（本文件）|

**CP-RES 线程本轮任务完成，等待 CP-BUILD 基于本 RFC 启动实现。**
