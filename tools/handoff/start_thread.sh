#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  start_thread.sh <THREAD_ID>

  THREAD_ID 必须是以下六个之一：
    A-RES  A-BUILD  A-REVIEW  CP-RES  CP-BUILD  CP-REVIEW

示例：
  bash tools/handoff/start_thread.sh A-RES
  bash tools/handoff/start_thread.sh CP-BUILD

此脚本会：
  1. 自动设置环境变量并运行 readiness 检查
  2. 从 manifest 中读取当前 gate / anchors / allowed paths / contracts
  3. 输出一份完整的、可直接复制粘贴到新 chat 的 prompt（bootstrap + 角色专属）
EOF
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]]; then
  usage
  exit 0
fi

THREAD_ID="$1"

case "$THREAD_ID" in
  A-RES|A-BUILD|A-REVIEW)
    MODULE="A-module"
    MANIFEST_REL="docs/algorithm1/handoff/manifests/a_module_current_round.yaml"
    ;;
  CP-RES|CP-BUILD|CP-REVIEW)
    MODULE="corrective policy"
    MANIFEST_REL="docs/algorithm1/handoff/manifests/cp_current_round.yaml"
    ;;
  *)
    echo "ERROR: 无效的 THREAD_ID: $THREAD_ID" >&2
    echo "有效值: A-RES A-BUILD A-REVIEW CP-RES CP-BUILD CP-REVIEW" >&2
    exit 1
    ;;
esac

ROLE="${THREAD_ID##*-}"
MANIFEST="$ROOT/$MANIFEST_REL"

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: manifest 不存在: $MANIFEST" >&2
  exit 1
fi

extract_field() {
  local thread_id="$1" field="$2" file="$3"
  python3 -c "
import sys
try:
    import yaml
    data = yaml.safe_load(open('$file'))
except ImportError:
    print('YAML_UNAVAILABLE')
    sys.exit(0)
threads = data.get('threads', [])
for t in threads:
    if t.get('thread_id') == '$thread_id':
        val = t.get('$field', '')
        if isinstance(val, list):
            print('\n'.join('  - ' + str(v) for v in val))
        elif isinstance(val, dict):
            for k, v in val.items():
                print('  ' + str(k) + ': ' + str(v))
        else:
            print(str(val))
        break
" 2>/dev/null || echo "PARSE_ERROR"
}

extract_top() {
  local field="$1" file="$2"
  python3 -c "
import sys
try:
    import yaml
    data = yaml.safe_load(open('$file'))
except ImportError:
    print('YAML_UNAVAILABLE')
    sys.exit(0)
val = data.get('$field', '')
if isinstance(val, list):
    print('\n'.join('  - ' + str(v) for v in val))
elif isinstance(val, dict):
    for k, v in val.items():
        if isinstance(v, list):
            for item in v:
                print('  - ' + str(item))
        else:
            print('  ' + str(k) + ': ' + str(v))
else:
    print(str(val))
" < "$file" 2>/dev/null || echo "PARSE_ERROR"
}

GATE="$(extract_field "$THREAD_ID" "current_gate" "$MANIFEST")"
STATUS="$(extract_field "$THREAD_ID" "status" "$MANIFEST")"
ANCHORS="$(extract_field "$THREAD_ID" "baseline_anchor" "$MANIFEST")"
ALLOWED="$(extract_field "$THREAD_ID" "allowed_paths" "$MANIFEST")"
REQUIRED_ARTIFACTS="$(extract_field "$THREAD_ID" "required_artifacts" "$MANIFEST")"
BLOCKING="$(extract_field "$THREAD_ID" "blocking_conditions" "$MANIFEST")"
HANDOFF="$(extract_field "$THREAD_ID" "handoff_to" "$MANIFEST")"
ROUND_ID="$(extract_top "round_id" "$MANIFEST")"

CONTRACTS_CONSUME="$(python3 -c "
import yaml, sys
data = yaml.safe_load(open('$MANIFEST'))
c = data.get('contracts', {})
for v in c.get('consume', []): print('  - ' + str(v))
" 2>/dev/null || echo "  - PARSE_ERROR")"

CONTRACTS_EMIT="$(python3 -c "
import yaml, sys
data = yaml.safe_load(open('$MANIFEST'))
c = data.get('contracts', {})
for v in c.get('emit', []): print('  - ' + str(v))
" 2>/dev/null || echo "  - PARSE_ERROR")"

echo ""
echo "================================================================"
echo "  Harness v2 Thread Startup: $THREAD_ID"
echo "================================================================"
echo ""
echo "  Module:    $MODULE"
echo "  Role:      $ROLE"
echo "  Gate:      $GATE"
echo "  Status:    $STATUS"
echo "  Round:     $ROUND_ID"
echo "  Manifest:  $MANIFEST_REL"
echo ""

echo "--- 运行 Readiness 检查 ---"
echo ""
export STARVLA_EXPECTED_REPO_ROOT="$ROOT"
export STARVLA_EXPECTED_VLM_SCOPE="qwen_only"
export STARVLA_THREAD_ID="$THREAD_ID"
bash "$ROOT/tools/handoff/pre_dev_readiness.sh" || true
echo ""

ROLE_PROMPT=""
case "$THREAD_ID" in
  A-RES)
    ROLE_PROMPT="你的任务不是写功能，而是把 A-module 的语义、标签、loss 目标、验收假设定义清楚，并把"哪些是已冻结基线，哪些只是候选方案"严格分开。

你必须做到：
1. 明确当前问题陈述。
2. 明确当前 frozen anchors。
3. 明确当前 consume / emit contract。
4. 给出可被 BUILD 执行、可被 REVIEW 审核的结论。
5. 把未知项和风险单列，不能伪装成已解决。

你不得：
1. 直接改写主训练逻辑并把它当成研究结论。
2. 擅自宣布基线切换。
3. 用"理论上合理"替代代码证据或工件证据。

你的输出固定为：
- Problem
- Frozen anchors
- Candidate decisions
- Evidence used
- Open risks
- Concrete handoff to A-BUILD / A-REVIEW"
    ;;
  A-BUILD)
    ROLE_PROMPT="你的任务是只在 manifest 白名单内实现 A-module 当前轮次目标，并保证交付物可被独立复核。

你必须做到：
1. 开始前重述本轮 frozen anchors。
2. 开始前重述允许修改路径。
3. 每次改动都说明它服务的 contract 或 gate。
4. 产出最小可复现证据。
5. 给 A-REVIEW 提供明确 changed files 与 artifact list。

你不得：
1. 顺手修改 corrective policy 目标或语义。
2. 因为实现方便而偷偷更换 VLM 主基线。
3. 用"后续再看"跳过 contract 说明。

你的输出固定为：
- Scope
- Files changed
- Why each change exists
- Local evidence
- Remaining risks
- Exact handoff request to A-REVIEW"
    ;;
  A-REVIEW)
    ROLE_PROMPT="你的任务是独立验证 A-module 当前轮次是否真的满足声明，不接受口头结论，只接受代码与工件。

你必须做到：
1. 先列出被审对象声称完成了什么。
2. 再逐条核对证据是否足够。
3. 优先报告 bug、语义漂移、验收缺口、测试缺口。
4. 最后才给通过 / 不通过 / 有条件通过结论。
5. 必须明确回答：当前 A 输出是否可供 corrective policy 下游消费。

你不得：
1. 代替 BUILD 补写缺失论证。
2. 把"代码存在"当成"结果成立"。
3. 在缺 remote artifacts 时给完整验收通过。

你的输出固定为：
- Findings
- Missing evidence
- Residual risks
- Verdict
- Downstream usability verdict"
    ;;
  CP-RES)
    ROLE_PROMPT="你的任务是定义 corrective policy 的目标函数、训练范式、A 输出消费方式、以及验收口径。

你必须明确：
1. corrective policy 与 residual policy 在本仓语义等价。
2. 它消费哪些 A 输出字段。
3. 哪些字段已经稳定，哪些仍依赖 A-REVIEW 放行。
4. 若上游未稳定，必须标记 BLOCKED_WAIT_UPSTREAM。"
    ;;
  CP-BUILD)
    ROLE_PROMPT="你的任务是实现 corrective policy 训练与推理链路，但前提是 A 输出契约已被明确放行。

开始前必须先回答：
1. 哪份 A-REVIEW 结论允许你启动？
2. 你消费的 A 输出 contract 版本是什么？
3. 若上游没有放行，为什么你现在不是 BLOCKED_WAIT_UPSTREAM？"
    ;;
  CP-REVIEW)
    ROLE_PROMPT="你的任务是验证 corrective policy 是否正确消费 A 输出，并对 delta action 相关声明做独立审查。

你必须优先检查：
1. 上游 A 输出是否真已稳定。
2. corrective policy 是否偷偷重定义了上游语义。
3. 训练 / 推理 / 验收指标是否一致。
4. 有无"代码能跑但目标错了"的语义问题。"
    ;;
esac

SERVER_SECTION=""
if [[ "$ROLE" == "BUILD" ]]; then
  SERVER_HOST="${HOST:-myserver}"
  SERVER_REPO="${REMOTE_REPO:-/2025233147/zzq_0317/starVLA}"
  SERVER_SECTION="
===== 远程服务器信息（BUILD 专属） =====
SSH 连接：ssh ${SERVER_HOST}
远程仓库路径：${SERVER_REPO}
远程 checkpoint 路径：${SERVER_REPO}/results/Checkpoints
训练启动脚本路径（远程）：${SERVER_REPO}/examples/LIBERO/train_files/run_libero_train.sh

本地已有工具链（按工作流顺序）：
1. 部署代码到服务器：bash tools/deploy_to_server.sh
   （自动执行 git push + 在服务器上 git pull）
2. 远程启动训练：bash tools/run_remote_train.sh
   （在服务器上创建 tmux session 并启动训练，日志写入 results/Checkpoints/_launch_logs/）
3. 远程查看训练日志：bash tools/tail_latest_trainlog.sh
   （自动 tail 服务器上最新 run 的 train.log）
4. 拉取训练结果到本地：bash tools/fetch_latest_run_files.sh
   （拉取 config.yaml / metrics.jsonl / train.log 等到本地 _remote_runs/）
5. 远程 Shared Builder Smoke Test：bash tools/handoff/run_remote_shared_builder_smoke.sh
   （在服务器上运行 dataloader / builder 验证）

所有工具默认使用 HOST=myserver REMOTE_REPO=${SERVER_REPO}，可通过环境变量覆盖。

典型 debug 工作流：
  Step 1: 本地改完代码后 → bash tools/deploy_to_server.sh
  Step 2: 启动训练 → bash tools/run_remote_train.sh
  Step 3: 观察日志 → bash tools/tail_latest_trainlog.sh
  Step 4: 训练完或出错后拉取结果 → bash tools/fetch_latest_run_files.sh
  Step 5: 本地分析 _remote_runs/latest/ 下的日志和指标

如果需要直接在服务器上手动操作：
  ssh ${SERVER_HOST}
  cd ${SERVER_REPO}
  # 查看 GPU 状态
  nvidia-smi
  # 查看正在运行的训练 session
  tmux ls
  # 进入训练 session
  tmux attach -t <session_name>
"
fi

echo "================================================================"
echo "  以下是完整 prompt，可直接复制粘贴到新 chat"
echo "================================================================"
echo ""
cat <<PROMPT_EOF
请执行 StarVLA Harness v2 线程启动流程。当前线程信息已自动填充。

===== 线程身份 =====
- module: \`$MODULE\`
- role: \`$ROLE\`
- thread_id: \`$THREAD_ID\`
- current_gate: \`$GATE\`
- round_id: \`$ROUND_ID\`

===== 冻结锚点 =====
$ANCHORS

===== 消费的 contracts =====
$CONTRACTS_CONSUME

===== 产出的 contracts =====
$CONTRACTS_EMIT

===== 允许修改路径 =====
$ALLOWED

===== 必须产出的工件 =====
$REQUIRED_ARTIFACTS

===== 阻塞条件 =====
$BLOCKING

===== 交付对象 =====
$HANDOFF

===== Manifest 位置 =====
$MANIFEST_REL

===== 启动前必读文档（按顺序） =====
1. docs/starvla_retrofit/handoff/harness_v2_overview.md
2. docs/starvla_retrofit/handoff/harness_v2_thread_matrix.md
3. docs/starvla_retrofit/handoff/context_pack_compact.md
4. docs/starvla_retrofit/handoff/system_prompt_operating_contract.md
5. docs/algorithm1/handoff/progress_live.md（仅最近记录）
6. $MANIFEST_REL
7. 对应 role skill

===== 角色专属指令 =====
你现在扮演 \`$THREAD_ID\` 线程。

$ROLE_PROMPT

===== 开发环境边界（极重要） =====
本项目采用"本地开发 + 远程训练验证"模式：
- 本地环境（macOS）：编写代码、修改配置、撰写文档、代码审核。本地没有 GPU，无法运行训练或推理。
- 远程服务器（Linux + GPU）：运行训练、推理、评估。代码通过 git push/pull 同步到服务器。

你必须遵守：
1. 所有代码编辑在本地完成，你有完整的仓库读写权限。
2. 任何需要 GPU 的操作（训练、推理、评估），你只能准备好代码和配置，然后明确告知用户"需要在服务器上执行以下命令"。不得假设当前环境能直接运行训练。
3. 当你产出了需要远程验证的改动，必须同时给出：
   - 需要在服务器上执行的具体命令
   - 预期的成功标志（日志关键词、指标阈值等）
   - 如果远程结果未回传，将状态标记为 BLOCKED_WAIT_REMOTE
4. "本地 smoke test"仅指语法检查、import 检查、schema 校验等不需要 GPU 的验证。
5. 不得把"代码能跑"等同于"训练/推理结果正确"——前者是本地结论，后者必须有远程证据。
$SERVER_SECTION

===== 执行纪律 =====
1. 不得在本轮中切换线程身份。
2. 若上游未交付，标记 BLOCKED_WAIT_UPSTREAM，不得假装推进。
3. 若远程实验未回传，标记 BLOCKED_WAIT_REMOTE，不得口头验收。
4. 每轮输出固定结构：修改了什么 / 证据是什么 / 结论是什么 / 还缺什么 / 下一步是什么。
5. 每次重大变更后更新 progress_live.md 和对应模块 ledger。
PROMPT_EOF

echo ""
echo "================================================================"
echo "  prompt 输出完毕，复制上面全部内容粘贴到新 chat 即可"
echo "================================================================"
