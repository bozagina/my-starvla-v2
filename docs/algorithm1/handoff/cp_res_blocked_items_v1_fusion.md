# CP-RES Blocked Items List v1.1 — RTC Paradigm + Fusion Mode

```
status:     DELIVERED
author:     CP-RES thread
thread_id:  CP-RES
round_id:   CP-ROUND-FUSION-RESUME
gate:       phase3_0b_cp_fusion_resume
date:       2026-04-17
revision:   v1.1
supersedes: v0, v1
```

---

## 结论摘要

**CP-RES 无上游阻塞。** 已识别 CP-BUILD 工程任务和远程验证阻塞项。

---

## 1. 上游阻塞项 — 全部 UNBLOCKED

| 项 | 状态 | 证据 |
|---|------|------|
| A-output contract (fusion mode) | UNBLOCKED | A-REVIEW Round 5 PASS |
| VDPM in-loop 可用（训练时） | UNBLOCKED | T-B4 500step PASS |

---

## 2. 远程阻塞项

| 项 | 状态 | 负责方 |
|---|------|--------|
| CP-AH-4-v1: 多 seed Success@LIBERO | BLOCKED_WAIT_REMOTE | OC |
| CP-AH-5: RNG 对齐验证 | BLOCKED_WAIT_REMOTE | OC |

---

## 3. CP-BUILD 工程任务

| # | 任务 | 优先级 | 说明 |
|---|------|--------|------|
| ENG-1 | CF 训练路径 region_logits 统一为 fusion head 实时输出 | HIGH | §3.1 of RFC |
| ENG-3 | RNG 对齐（独立 Generator） | MEDIUM | §3.3 of RFC |
| ENG-4 | Episode reset 链路修复 | MEDIUM | §3.4 of RFC |
| ~~ENG-2~~ | ~~CF 推理路径 fusion 适配~~ | ~~HIGH~~ | **消除**（RTC 范式） |

---

## 4. BD 决策

| BD | 决策 | 状态 |
|----|------|------|
| BD-1 | pure additive: a_prev + velocity | **确定** |
| BD-3 | 训练 fusion，推理 CF 不依赖 A-module | **确定** |
| BD-6 | fusion head 实时输出 → 仅 loss 权重 | **确定** |
| BD-7 | 推荐 self-attn only | **CP-BUILD 决策** |
| BD-8 | trigger/region 仅训练 loss 权重 | **确定** |

---

## 5. CP-RES 完成声明

| 工件 | 路径 | 状态 |
|------|------|------|
| Policy RFC v1.1 | `docs/algorithm1/handoff/cp_res_policy_rfc_v1_fusion.md` | DELIVERED |
| Interface Assumptions v1.1 | `docs/algorithm1/handoff/cp_res_interface_assumptions_v1_fusion.md` | DELIVERED |
| Blocked Items v1.1 | `docs/algorithm1/handoff/cp_res_blocked_items_v1_fusion.md` | DELIVERED |
