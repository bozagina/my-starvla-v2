# T4(INFRA): Trainer 最小插入点线程（A_module / corrective_policy）

- 线程定位: 在现有 `train_starvla.py` 上做最小插入，不做大规模重构。
- 建议模块: `INFRA`
- VLM 约束: 仅走 `Qwen2.5VL / Qwen3VL`，不引入 `MapAnything/LLaVA3D` 依赖。

## 1) 启动命令

```bash
cd /Users/bazinga/code/my-starvla-v2
bash /Users/bazinga/code/my-starvla-v2/tools/handoff/pre_dev_readiness.sh
/Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh start \
  --module INFRA \
  --owner OC \
  --title "T4 trainer insertion for A/corrective"
```

## 2) 必读上下文（按顺序）

1. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/context_pack_compact.md`
2. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/system_prompt_operating_contract.md`
3. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/thread_prompts_and_checklists_index.md`
4. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/threads/thread_t4_trainer_insertion.md`
5. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/threads/thread_t1_data_builder.md`

## 3) 白名单文件

1. `/Users/bazinga/code/my-starvla-v2/starVLA/training/train_starvla.py`
2. `/Users/bazinga/code/my-starvla-v2/starVLA/model/framework/QwenPI.py`
3. `/Users/bazinga/code/my-starvla-v2/starVLA/model/framework/QwenGR00T.py`
4. `/Users/bazinga/code/my-starvla-v2/starVLA/model/framework/QwenAdapter.py`
5. `/Users/bazinga/code/my-starvla-v2/starVLA/model/modules/vlm/QWen2_5.py`
6. `/Users/bazinga/code/my-starvla-v2/starVLA/model/modules/vlm/QWen3.py`
7. `/Users/bazinga/code/my-starvla-v2/starVLA/config/training/*.yaml`
8. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/*.md`

## 4) 线程专用 Prompt（复制即用）

```text
你是 T4(INFRA) 线程，目标是在 train_starvla 上找到并实现 A_module/corrective_policy 的最小插入点。

要求:
1) 只做最小可执行插入，不重写训练框架。
2) 优先插入点:
   - _train_step: loss 聚合与可选分支
   - eval_action_model: 可选 debug/eval 信息
   - _log_metrics: 新增指标命名空间
3) 兼容旧流程: 默认配置下行为不变。
4) 不做 Path-A 扩展，不做异步深耦合。
5) 每轮输出必须包含: 修改/证据/结论/下一步/commit message。
6) 如果发现需求依赖 MapAnything/LLaVA3D 文件，先回退到 Qwen 路径并在日志中标注阻塞点。
```

## 5) Upstream Pass Checks

1. T1 已提供 builder 字段草案（至少 action chunk + 元数据字段）。
2. T2 有 schema 约束或临时校验规则。
3. 若上游尚未稳定，只实现插入框架和开关，不绑死字段。

## 6) Checklist

1. 在 `_train_step` 增加可选 loss hook: `base + a_loss + corrective_loss`。
2. 在配置中加入开关（默认关闭），确保旧训练脚本不受影响。
3. 在 `eval_action_model` / `predict_action` 打通最小 debug 信息链路。
4. 指标 key 与 T3 验收线程对齐，避免后续解析冲突。

## 7) Done 证据格式

1. 命令证据: `rg -n "_train_step|eval_action_model|predict_action|Qwen|qwen"`。
2. 本地证据: 至少一次配置开关开/关的行为对比（日志级别即可）。
3. `progress_live.md` 给出“默认不变、开关启用后新增行为”。

## 8) 交接规则

1. 给 T3: 新增 metrics key 与解释。
2. 给 T1/T2: trainer 对输入字段的最小要求。
3. 如依赖远程跑通才可确认，先标记 `BLOCKED_WAIT_REMOTE` 并给出明确重试条件。
