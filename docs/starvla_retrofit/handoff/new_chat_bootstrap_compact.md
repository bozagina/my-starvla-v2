# New Chat Bootstrap (Retrofit Compact)

用途：以最小上下文成本启动 StarVLA retrofit 会话。

```text
请执行 StarVLA retrofit 紧凑启动流程，并严格遵守：

1) 先读取最小上下文包（按顺序）：
   - /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/context_pack_compact.md
   - /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/system_prompt_operating_contract.md
   - /Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/progress_live.md（仅尾部最近记录）
   - 并明确本仓仅走 Qwen2.5VL / Qwen3VL；MapAnything/LLaVA3D 已解耦到外部项目，不作为当前开发依赖

2) 输出不超过8条“当前状态与约束”总结。

3) 检查当前分支与工作区状态，并确认三层分支流转（tmp -> worktree -> final）。

4) 在首个代码改动前创建 EXP_ID（由你直接执行命令）：
   /Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh start \
     --module INFRA \
     --owner OC \
     --title "<one line task title>"

5) 进入当前 gate（默认按 P0->P1->P2->P3 顺序）。
   - 若信息不足，只加载当前 gate 相关源码；不要全量读仓库。

6) 每轮输出必须包含：
   - 修改了什么
   - 证据是什么
   - 得到什么结论
   - 下一步建议
   - 建议 commit message（[EXP_ID] one-line intent）

7) 每次重大修改后，必须追加更新：
   - /Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/progress_live.md

8) 本地开发、远程训练验证：
   - 远程是否生效只能依据用户提供的远程日志/配置/指标。

9) 开发前建议执行一键 readiness：
   - bash /Users/bazinga/code/my-starvla-v2/tools/handoff/pre_dev_readiness.sh

10) 若需要标准操作步骤，按清单执行：
   - /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/development_run_checklist.md

11) 若要并行开线程，先选线程专用文档：
   - /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/thread_prompts_and_checklists_index.md
   - /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/threads/
```
