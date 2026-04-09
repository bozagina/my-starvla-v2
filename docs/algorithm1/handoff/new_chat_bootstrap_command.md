# New Chat Bootstrap Command (Copy-Paste Template)

用途：每次开启新对话时，给新模型一段固定启动指令，让它自动执行既定规范流程。

---

## 1) 一键启动指令（复制到新对话第一条消息）

```text
请执行本项目会话启动流程，并严格遵守：

1) 先阅读以下文档（按顺序）：
   - /Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/mask_diagnosis_full_history.md
   - /Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/system_prompt_operating_contract.md
   - /Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/remote_training_workflow.md
   - /Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/experiment_id_and_naming_convention.md
   - /Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/progress_live.md

2) 阅读后先用不超过8条要点总结你理解到的当前状态与约束。

3) 然后先问我：本轮希望你负责哪个单一模块（MASK / FBLOSS / CFG / DIAG / DATA / INFRA）？
   在我确认模块前，不要开始大范围改代码。

4) 模块确认后，请自动创建一个标准化进度条目（EXP_ID）：
   - 由你（大模型）直接执行命令，不要让我手动执行：
     /Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh start \
       --module <CONFIRMED_MODULE> \
       --owner OC \
       --title "<one line task title>"
   - 或直接执行底层脚本：
     python /Users/bazinga/code/my-starvla-v2/tools/handoff/new_progress_entry.py \
       --module <CONFIRMED_MODULE> \
       --owner OC \
       --title "<one line task title>"
   - 执行后把生成的 EXP_ID 回报给我。
   - 只有在你无法执行命令（环境限制）时，才把命令回退给我手动执行。

5) 接着你再对该模块做深入分析，给出最小可执行计划并开始开发。

5.1) 如果本轮会改训练配置（yaml），请把 EXP_ID 写入 `run_id`（用于远程日志映射）：
   - 由你（大模型）执行：
     python /Users/bazinga/code/my-starvla-v2/tools/handoff/stamp_run_id_with_exp.py \
       --config <TRAIN_CONFIG_PATH> \
       --exp-id <GENERATED_EXP_ID> \
       --in-place

6) 每次重大修改后，必须追加更新：
   - /Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/progress_live.md

7) 每轮输出都要包含：
   - 修改了什么
   - 证据是什么
   - 得到什么结论
   - 下一步建议
   - 建议的 commit message（格式：`[EXP_ID] one-line intent`）

8) 牢记：我们是本地开发、远程训练。
   - 远程训练是否已应用改动，必须以我提供的远程日志/配置/指标为准。
   - 我会通过 tools 里的 fetch 脚本获取日志后再发给你分析。
```

---

## 1.1 终端辅助命令（仅手动兜底时使用）

在本地终端可直接用：

```bash
# 打印“新会话第一条消息”模板（复制粘贴到新对话）
/Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh prompt

# 在模型无法执行命令时，你手动补建标准化 EXP_ID 进度条目
/Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh start \
  --module MASK \
  --owner OC \
  --title "One line task title"
```

---

## 2) 我给模型确认模块时可用的短回复

示例：

- `本轮你负责 MASK 模块。请先创建 EXP_ID 并更新 progress，然后开始。`
- `本轮你负责 DIAG 模块，优先批量诊断脚本。`

---

## 3) 说明

1. 这份指令要求模型“先问模块再开发”，避免并行会话越界改动。
2. 通过 `new_progress_entry.py` 自动生成 `EXP_ID`，减少命名冲突。
3. 若模型未按流程执行，可直接提醒：  
   `请回到 new_chat_bootstrap_command.md 的流程，从第1步重来。`
