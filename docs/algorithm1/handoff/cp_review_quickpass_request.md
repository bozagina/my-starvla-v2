# CP-REVIEW Quick-Pass 请求

> 发起时间: 2026-04-15  
> 发起方: CP-BUILD  
> 目标方: CP-REVIEW  
> 关联: `cp_review_audit_report.md` Round 1 CONDITIONAL_PASS

---

## 背景

CP-REVIEW Round 1 裁决 CONDITIONAL_PASS，要求修复两个阻塞项后执行 Quick-Pass 验证。

## 已修复项

### B-1: dtype 恢复逻辑泄漏 → FIXED

**文件**: `starVLA/model/framework/QwenPI.py`，`predict_action()` corrective flow 块

**修复内容**:
- `orig_ami_dtype` 和 `orig_cf_dtype` 在 `try` 之前捕获
- dtype 恢复逻辑从 `try` 块末尾移至 `finally` 块
- 异常时 `ami` 和 `corrective_flow_head` 均恢复原始 dtype

**修复前** (伪代码):
```python
try:
    ami.float()
    cf_head.float()
    ...  # 若此处异常 → dtype 不恢复
    ami.to(orig)
    cf_head.to(orig)
except RuntimeError:
    pass  # 泄漏
```

**修复后** (伪代码):
```python
orig_ami_dtype = ...
orig_cf_dtype = ...
try:
    ami.float()
    cf_head.float()
    ...
except RuntimeError:
    logger.warning(...)
finally:
    ami.to(orig_ami_dtype)       # 始终执行
    cf_head.to(orig_cf_dtype)    # 始终执行
```

### B-2: 参考配置 a_module.mode 不一致 → FIXED

**文件**: `starVLA/config/training/starvla_train_pi_qwen25_corrective_flow.yaml`

**修复内容**:
- `a_module.mode` 从 `standalone` 改为 `lite`
- 移除了 `standalone` 子配置块（`input_key`, `allow_pseudo_labels_fallback` 等）
- 现与 `run_cp_smoke_500step.sh` 的 `--framework.a_module.mode lite` 一致

## Quick-Pass 验证范围

请仅验证以下两点（无需重跑 smoke train）：

1. **B-1**: `QwenPI.py` predict_action corrective flow 块是否使用 `try/finally`，且 `finally` 中恢复 `ami` 和 `corrective_flow_head` 的原始 dtype
2. **B-2**: `starvla_train_pi_qwen25_corrective_flow.yaml` 中 `a_module.mode` 是否为 `lite`

## 期望输出

- 在 `cp_review_audit_report.md` 末尾追加 Round 2 Quick-Pass 结果
- 更新 `cp_current_round.yaml` CP-REVIEW verdict 为 PASS（若通过）
