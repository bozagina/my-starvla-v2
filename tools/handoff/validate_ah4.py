#!/usr/bin/env python3
"""AH-4 Acceptance Validation: check 500-step fusion training metrics."""
import json, math, statistics, sys

fpath = sys.argv[1] if len(sys.argv) > 1 else "metrics.jsonl"
with open(fpath) as f:
    lines = [json.loads(l) for l in f if l.strip()]

print("Total metric entries:", len(lines))

loss_keys = [
    "loss/total", "loss/action", "loss/a_module", "loss/corrective",
    "loss/trigger", "loss/risk", "loss/embed", "loss/delta", "loss/region",
]

print("=" * 80)
print("AH-4 ACCEPTANCE VALIDATION: 500-step Fusion Training")
print("=" * 80)

all_finite = True
nan_count = inf_count = 0
for r in lines:
    for k in loss_keys:
        v = r.get(k)
        if v is not None:
            if math.isnan(v): nan_count += 1; all_finite = False
            if math.isinf(v): inf_count += 1; all_finite = False

print("\n[1] Finiteness:", "PASS" if all_finite else "FAIL")
print("    NaN:", nan_count, " Inf:", inf_count)

print("\n[2] Non-degeneracy:")
all_nondegen = True
for k in loss_keys:
    vals = [r.get(k, 0) for r in lines]
    mn, mx = min(vals), max(vals)
    mean_v = statistics.mean(vals)
    std_v = statistics.stdev(vals) if len(vals) > 1 else 0
    always_zero = all(v == 0 for v in vals)
    constant = len(set(vals)) == 1

    if always_zero:
        status = "FAIL(zero)"
        all_nondegen = False
    elif constant:
        status = "WARN(const)"
        all_nondegen = False
    else:
        status = "PASS"

    print(f"    {k:25s} {status:15s} mean={mean_v:.4f} std={std_v:.4f} [{mn:.4f}, {mx:.4f}]")

print("\n[3] Trend (first 10 vs last 10 entries):")
for k in loss_keys:
    early = statistics.mean([r.get(k, 0) for r in lines[:10]])
    late = statistics.mean([r.get(k, 0) for r in lines[-10:]])
    trend = "down" if late < early else ("up" if late > early else "flat")
    pct = (late - early) / max(abs(early), 1e-8) * 100
    print(f"    {k:25s} early={early:.4f} late={late:.4f} {trend:5s} ({pct:+.1f}%)")

print("\n[4] Target coverage:")
for k in ["debug/a_loss_target_count", "debug/corrective_loss_target_count"]:
    vals = [r.get(k, 0) for r in lines]
    print(f"    {k}: mean={statistics.mean(vals):.1f} unique={sorted(set(vals))}")

print("\n" + "=" * 80)
verdict = "PASS" if (all_finite and all_nondegen) else "FAIL"
print(f"AH-4 OVERALL: {verdict}")
print(f"  Finiteness: {'PASS' if all_finite else 'FAIL'}")
print(f"  Non-degeneracy: {'PASS' if all_nondegen else 'FAIL'}")
print("=" * 80)

print("\nSample steps (logging_frequency=5):")
hdr = f"{'step':>5} | {'total':>8} | {'action':>8} | {'a_mod':>8} | {'corr':>8} | {'trig':>8} | {'risk':>8} | {'embed':>8} | {'delta':>8} | {'region':>8}"
print(hdr)
print("-" * len(hdr))
for idx in [0, 19, 39, 59, 79, 99]:
    if idx < len(lines):
        r = lines[idx]
        step = idx * 5
        print(f"{step:>5} | {r.get('loss/total',0):>8.4f} | {r.get('loss/action',0):>8.4f} | {r.get('loss/a_module',0):>8.4f} | {r.get('loss/corrective',0):>8.4f} | {r.get('loss/trigger',0):>8.4f} | {r.get('loss/risk',0):>8.4f} | {r.get('loss/embed',0):>8.4f} | {r.get('loss/delta',0):>8.4f} | {r.get('loss/region',0):>8.4f}")
if lines:
    r = lines[-1]
    step = (len(lines) - 1) * 5
    print(f"{step:>5} | {r.get('loss/total',0):>8.4f} | {r.get('loss/action',0):>8.4f} | {r.get('loss/a_module',0):>8.4f} | {r.get('loss/corrective',0):>8.4f} | {r.get('loss/trigger',0):>8.4f} | {r.get('loss/risk',0):>8.4f} | {r.get('loss/embed',0):>8.4f} | {r.get('loss/delta',0):>8.4f} | {r.get('loss/region',0):>8.4f}")
