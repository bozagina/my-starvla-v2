# T2(INFRA): Schema/Validator 线程

- 线程定位: 把 T1 交付的样本字段固定成可机检 contract，减少“验收口径冲突”。
- 建议模块: `INFRA`

## 1) 启动命令

```bash
cd /Users/bazinga/code/my-starvla-v2
bash /Users/bazinga/code/my-starvla-v2/tools/handoff/pre_dev_readiness.sh
/Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh start \
  --module INFRA \
  --owner OC \
  --title "T2 schema + validator for retrofit builder"
```

## 2) 必读上下文（按顺序）

1. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/context_pack_compact.md`
2. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/system_prompt_operating_contract.md`
3. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/thread_prompts_and_checklists_index.md`
4. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/threads/thread_t2_schema_infra.md`
5. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/threads/thread_t1_data_builder.md`

## 3) 白名单文件

1. `/Users/bazinga/code/my-starvla-v2/tools/handoff/*.py`
2. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/*.md`
3. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/threads/*.md`

## 4) 线程专用 Prompt（复制即用）

```text
你是 T2(INFRA) 线程，目标是把 T1 builder 输出固化为 schema + validator，保证人审和机审一致。

要求:
1) 先读取 compact/context/线程索引 + T1 文档。
2) 不改模型逻辑，只做 contract 与校验基础设施。
3) 产出:
   - 样本 schema 文档
   - 本地 validator（可命令行运行）
   - 失败时可读错误信息
4) 每轮输出必须包含:
   - 修改了什么
   - 证据是什么
   - 得到什么结论
   - 下一步建议
   - 建议 commit message
```

## 5) Upstream Pass Checks

1. T1 至少给出字段草案（即使是 draft）。
2. 若 T1 未完成，先写 schema v0 并在 `progress_live.md` 标注假设项。
3. 不得阻塞 T4: 允许先提供“兼容模式校验”。

## 6) Checklist

1. 明确必填字段/可选字段/版本号。
2. 增加 shape 约束（例如 action chunk 的 `T` 与动作维度 `D`）。
3. 提供 validator CLI 与示例输入。
4. 输出与验收线程(T3)对齐的字段映射（避免指标口径冲突）。

## 7) Done 证据格式

1. 命令: `python <validator> --help` + 至少一次正样本/负样本验证。
2. 文档: schema 字段表与版本演进规则。
3. `progress_live.md` 记录“下游可直接引用的校验命令”。

## 8) 交接规则

1. 给 T3: 验收可复用的字段解释（防指标误读）。
2. 给 T4: trainer 插入点可依赖的字段存在性断言。
3. 若真实样本缺失，状态使用 `BLOCKED_WAIT_REMOTE`，并写清最小缺失工件。
