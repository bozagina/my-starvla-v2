# RFC: P4.1 FASA Module Design (Future-Aware Structured A-Module)

- Thread: `P4.1-Research & Build Handoff`
- Status: `finalized RFC (build-ready)`
- Scope: standalone A-sidecar data pipeline, model architecture, loss design, training workflow, and acceptance criteria
- Compatibility target: `/Users/bazinga/code/my-starvla-v2/starVLA/model/framework/a_module_interface.py`
- Authoritative baseline: `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/p4_1_qwen25_longrun_authoritative_baseline.md`
- Mainline constraint: strictly preserve the current `Qwen2.5-VL` authoritative baseline and the standalone A consumer contract
- Non-goals:
  - no change to the authoritative `Qwen2.5-VL` planner backbone or its trainer entrypoint
  - no reintroduction of `Qwen3` as the accepted mainline
  - no `restore guard`, `fallback_to_lite`, or partial-freeze as the mainline solution
  - no breaking change to the frozen `a_outputs` payload contract

## 1) Executive Summary

This RFC upgrades the placeholder external A-module concept into `FASA`:

- `FASA` = `Future-Aware Structured A-Module`
- form factor: standalone asynchronous sidecar
- integration path: offline/precomputed `a_outputs`, consumed by the existing standalone A interface
- core role: detect and quantify plan-observation misalignment at execution time without modifying the main `Qwen2.5-VL` planner

The motivating failure mode is planner/executor frequency mismatch:

1. the planner emits an action chunk under one world state assumption
2. the environment evolves during chunk execution
3. local visual and physical reality can drift away from the planned execution manifold
4. the current system needs a structured corrective sidecar, not a second monolithic planner

`FASA` is therefore designed to:

1. read a short recent observation window plus the pending action chunk
2. estimate whether the current chunk is still safe to execute
3. localize which normalized future chunk bins are most correction-relevant
4. quantify how strong the correction should be
5. encode compact future-dynamics information in a fixed `16`-D latent

The mainline claim is not generic future awareness. The intended research position is:

- `VLASH`: future-state-aware asynchronous baseline
- `F2F-AP`: future-observation-aware asynchronous baseline
- `FASA`: future-aware structured corrective sidecar with a frozen external payload contract and retrofit-friendly integration

## 2) Hard Constraints Frozen by This RFC

### 2.1 Canonical External Payload

The canonical payload stored under `sample["a_outputs"]` is frozen as:

```json
{
  "risk_pred": 0.0,
  "trigger_logit": 0.0,
  "delta_pred": 0.0,
  "region_logits": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "dynamic_embedding": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "version": "fasa_v1",
  "source": "fasa/main"
}
```

### 2.2 Field Semantics

- `risk_pred`
  - scalar in `[0, 1]`
  - semantics: future-window task failure / unsafe execution risk
  - exported as a scalar payload value; internal model may use a logit and apply `sigmoid` before export
- `trigger_logit`
  - scalar pre-sigmoid intervention logit
  - positive means "stop trusting the current chunk and escalate / replan"
- `delta_pred`
  - scalar non-negative correction magnitude proxy
  - semantics: how strong a local residual correction should be when trigger does not fully seize control
- `region_logits`
  - fixed `15` independent logits over normalized future chunk bins
  - semantics: correction relevance per future normalized temporal bin
  - not a softmax distribution and not required to be mutually exclusive
- `dynamic_embedding`
  - exact `16`-D external latent
  - semantics: compact future-dynamics representation for diagnostics, retrieval, and future extensions
  - must be exported as `16` dimensions exactly; no train-time `128`-D output that is truncated at inference
- `version`
  - immutable schema tag, start with `fasa_v1`
- `source`
  - producer/provenance tag, examples: `fasa/main`, `fasa/interp15`, `fasa/guarded`

### 2.3 Compatibility Guarantees

This RFC preserves compatibility with the current standalone A consumer because:

1. `/Users/bazinga/code/my-starvla-v2/starVLA/model/framework/a_module_interface.py` already reads `sample["a_outputs"]`
2. the existing consumer directly uses `risk_pred`, `trigger_logit`, `delta_pred`, and `region_logits`
3. it validates `dynamic_embedding` length only when contract checks are enabled
4. it tolerates extra metadata fields such as `version` and `source`

Decision for `P4.1`:

- keep external `region_logits` fixed at `15`
- keep external `dynamic_embedding` fixed at `16`
- do not change the current standalone A consumer contract in this RFC

## 3) System Boundaries

### 3.1 What FASA Is

`FASA` is a separate trainable sidecar that produces `a_outputs` offline or in a decoupled runtime path.

The intended flow is:

1. collect or build a correction dataset with FASA supervision fields
2. train the FASA sidecar independently
3. use the trained sidecar to emit canonical `a_outputs`
4. let the current standalone A interface consume those outputs during mainline `Qwen2.5-VL` training or evaluation

### 3.2 What FASA Is Not

`FASA` is not:

- an in-graph replacement for the current planner
- a reason to enlarge the lite A-head inside `/Users/bazinga/code/my-starvla-v2/starVLA/model/framework/QwenPI.py`
- a justification to mutate the authoritative `Qwen2.5-VL` backbone
- a requirement for the mainline optional loss builder to understand all FASA-only auxiliary targets

### 3.3 Important Boundary on Targets

`future_state` and other FASA-only supervision fields are sidecar-training targets, not required inputs to the current mainline optional A-hook target builder.

This RFC explicitly does **not** require `/Users/bazinga/code/my-starvla-v2/starVLA/model/framework/optional_loss_utils.py` to become the target builder for all FASA training objectives.

Recommended split:

- mainline trainer:
  - continues to read `a_outputs`
  - continues to use the current optional A / corrective hook targets
- FASA trainer:
  - owns `future_state`, horizon-conditioned labels, and any auxiliary latent losses

## 4) Data Pipeline: Controlled Misalignment Collection

### 4.1 Goal

The FASA dataset should contain controlled, labeled spatiotemporal misalignment rather than only near-perfect successful trajectories.

This RFC recommends a DART-style wrapper to inject bounded perturbations during data collection and to backfill future-aligned supervision.

### 4.2 Required New Record Fields

Each collected correction-supervision record should support at least:

- `dataset_name`
- `trajectory_id`
- `step_index`
- `instruction`
- `base_action_chunk`
- `horizon_steps`
- `state_t`
- `future_state`
- `future_state_step`
- `perturbation_type`
- `perturbation_strength`
- `delay_span`
- `pseudo_labels`
  - `risk_score`
  - `trigger_label`
  - `delta_action_norm`
  - `correction_mask`
  - `affected_region_prior`
- optional `a_outputs` once FASA is trained and serving as the producer

### 4.3 Perturbation Strategy A: Action-Space Injection

Inject bounded action noise before environment execution.

Recommended default:

- probability: `p_action ~= 0.3`
- perturbation: `epsilon ~ N(0, sigma^2)` with task-specific clipping
- applied to the currently executed action, not to the saved base plan

Target construction:

1. obtain the expert or nominal recovery action under the perturbed state
2. define `delta_target = ||a_expert - a_executed||_2`
3. optionally normalize by action scale to stabilize cross-task comparison

This produces dense residual-correction supervision for `delta_pred`.

### 4.4 Perturbation Strategy B: Observation Delay / Freeze

Inject observation delay to simulate stale visual context.

Recommended default:

- probability: `p_delay ~= 0.2`
- sampled span: `k ~ Uniform{3, 4, 5}` frames
- use stale visual observations while the real environment keeps evolving

Target construction:

1. set `trigger_target = 1` during delay-induced misalignment windows when the stale observation materially invalidates the current chunk
2. map the affected chunk-relative interval to the canonical `15` normalized bins
3. build `region_target_15` as a multi-hot or soft bin target, not a one-of-15 classification target

### 4.5 Future-State Backfill Rule

`future_state` must be backfilled after the chosen horizon is reached.

This is mandatory.

It is **not** acceptable to record `future_state = env.get_true_state()` immediately at step `t` and claim it is the target for `t + H`.

Required collection rule:

1. when a training record is created at step `t`, store `horizon_steps = H`
2. add the record to a pending queue
3. when the rollout reaches `t + H`, read the true state and backfill:
   - `future_state`
   - `future_state_step = t + H`
4. only then finalize the record for FASA training

### 4.6 Risk Target Construction

`risk_score` should remain aligned with the current mainline semantics while being future-aware.

Recommended initial rule:

- binary or soft future risk within horizon `H`, computed from one or more of:
  - task failure within the future window
  - thresholded expert-recovery gap
  - thresholded state deviation
  - severe region activation over the future bins

Preferred first implementation:

```text
risk_score = clip(max(trigger_target, normalized_delta_target, future_failure_flag), 0, 1)
```

This keeps risk supervision simple and auditable.

## 5) Recommended FASA Architecture

### 5.1 Design Principle

Do not reuse the current lite A-head pattern of `pooled_hidden -> tiny MLPs` as the main research implementation.

Instead, use a structured sidecar that explicitly compares:

- the pending plan (`base_action_chunk`)
- the recent observed visual stream
- the current state context
- the selected horizon context

### 5.2 Inputs and Tensor Shapes

Recommended sidecar inputs:

- `recent_frames`: `[B, T_v, C, H, W]`
- `base_action_chunk`: `[B, L, D_a]`
- `action_pos`: `[B, L, 1]`
- `instruction_embedding`: `[B, D_t]`
- `state_t`: `[B, D_s]`
- `horizon_steps`: `[B]`
- optional runtime `latency_ms`: `[B, 1]`

Training-only auxiliary targets:

- `future_state`: `[B, D_f]`
- `region_target_15`: `[B, 15]`
- `trigger_target`: `[B]`
- `delta_target`: `[B]`
- `risk_score`: `[B]`

### 5.3 Mainline Backbone Recommendation

For the first implementation, the recommended sidecar backbone is:

1. a frozen or lightly trainable visual encoder over recent frames
   - recommended v1 choice: `DINOv2-S/B` patch features
2. a lightweight temporal adapter over frame tokens
   - temporal difference features and/or 1D temporal convolution
3. an action encoder for the pending action chunk
4. a small context encoder for instruction/state/horizon
5. a query-based fusion decoder

This keeps FASA independent from the `Qwen2.5-VL` planner internals while still allowing future upgrades.

### 5.4 Internal Blocks

#### Visual Path

1. encode each recent frame into patch or pooled visual tokens
2. build temporal residual features `V_t - V_{t-1}` where available
3. apply a lightweight temporal fusion block to obtain visual memory tokens

Output:

- `visual_memory`: `[B, N_v, D]`

#### Action Path

1. concatenate `base_action_chunk` with relative action-step position encodings
2. pass through an action MLP or shallow transformer

Output:

- `action_tokens`: `[B, L, D]`

#### Context Path

1. project `instruction_embedding`
2. project `state_t`
3. embed `horizon_steps`
4. optionally project `latency_ms`
5. fuse into one or a few context tokens

Output:

- `context_tokens`: `[B, N_c, D]`

### 5.5 Fusion and Decoder

Recommended v1 fusion:

1. let action tokens attend to visual memory and context tokens
2. use a small query decoder on top of the fused memory

Recommended decoder tokenization:

- `1` global query for `risk_pred`, `trigger_logit`, `delta_pred`
- `15` region queries, one per canonical normalized temporal bin
- `4` dynamic slot queries pooled into the final `16`-D external `dynamic_embedding`

This is preferred over directly projecting one pooled vector into all outputs.

### 5.6 Output Heads

Recommended output mapping:

- `global_query -> risk_logit -> risk_pred = sigmoid(risk_logit)`
- `global_query -> trigger_logit`
- `global_query -> delta_raw -> delta_pred = softplus(delta_raw)`
- `region_query_i -> scalar logit`, stacked into `region_logits[15]`
- `dynamic_slots -> pooled hidden -> internal latent h_dyn -> dynamic_embedding(16)`

Important:

- the exported `dynamic_embedding` must be exactly `16`-D
- if a higher-dimensional internal latent is desired for auxiliary losses, it must remain internal and distinct from the exported `dynamic_embedding`

## 6) Loss Design

### 6.1 Overview

The initial FASA objective should favor stability, auditability, and direct alignment with the frozen external contract.

Do not use a single exotic objective as the only training signal.

The recommended v1 formulation combines:

- calibrated risk prediction
- imbalanced trigger detection
- robust delta regression
- future-bin localization
- future-dynamics latent regularization

### 6.2 Risk Loss

Recommended formulation:

```text
risk_pred = sigmoid(risk_logit)
L_risk = BCEWithLogits(risk_logit, risk_score) + 0.5 * SmoothL1(risk_pred, risk_score)
```

Rationale:

- preserves probabilistic calibration
- remains stable when `risk_score` is a soft target in `[0, 1]`
- directly aligns with the existing mainline `risk_score` semantics

### 6.3 Trigger Loss

Recommended formulation:

```text
L_trigger = FocalBCE(trigger_logit, trigger_target; alpha=0.25, gamma=2.0)
```

Rationale:

- trigger positives are typically sparse
- focal loss reduces domination by easy negatives

### 6.4 Delta Loss

Recommended formulation:

```text
L_delta = SmoothL1(log1p(delta_pred), log1p(delta_target))
```

Rationale:

- residual magnitudes are often long-tailed
- `log1p` stabilizes scale while preserving ordering
- `SmoothL1` is more robust than plain MSE under DART-style perturbations

### 6.5 Region Losses

Recommended region target space:

- `region_target_15`: `[B, 15]`
- each bin indicates correction relevance, not a categorical class index

Recommended formulation:

```text
L_region_focal = FocalBCE(region_logits, region_target_15)
L_region_iou = 1 - SoftIoU(sigmoid(region_logits), region_target_15)
L_region_prior = MSE(sigmoid(region_logits), region_prior_15)
```

Optional smoothness regularizer:

```text
L_region_smooth = mean((p[:, 1:] - p[:, :-1])^2)
where p = sigmoid(region_logits)
```

Recommended combined region objective:

```text
L_region = 1.0 * L_region_focal + 0.5 * L_region_iou + 0.25 * L_region_prior + 0.1 * L_region_smooth
```

Rationale:

- focal term handles imbalance
- IoU term improves whole-bin shape alignment
- prior term aligns with current pseudo-label semantics
- smoothness discourages physically implausible bin flicker

### 6.6 Dynamic Embedding Losses

The exported `dynamic_embedding(16)` should become a real future latent, not a placeholder field.

Recommended v1 auxiliary objectives:

```text
L_dyn_nce = InfoNCE(z_t, z_pos, z_neg)
L_dyn_state = SmoothL1(W(z_t), future_state_enc)
```

Where:

- `z_t = dynamic_embedding(16)` or a small internal precursor projected from the dynamic slots
- `future_state_enc = MLP(future_state)`
- `z_pos` is a matching future-positive sample from the same trajectory / horizon
- `z_neg` are mismatched trajectories or wrong-horizon samples

Optional ablation only, not the mainline default:

- sliced Wasserstein or other OT-style distribution alignment losses

Reason for keeping OT-style loss optional:

1. it is more sensitive to batch construction
2. it complicates early debugging
3. it should not replace direct future-consistency supervision in v1

### 6.7 Recommended Total Objective

Initial default weights:

```text
L_total =
  1.0 * L_risk +
  2.0 * L_trigger +
  1.0 * L_delta +
  1.0 * L_region_focal +
  0.5 * L_region_iou +
  0.25 * L_region_prior +
  0.1 * L_region_smooth +
  0.2 * L_dyn_nce +
  0.1 * L_dyn_state
```

These are starting weights, not a claim of final optimality.

## 7) Training Workflow

### 7.1 Stage 0: Dataset Construction

Build the correction-supervision dataset with:

- controlled perturbation metadata
- horizon-aware target backfill
- canonical `15`-bin region targets
- future-state anchors for sidecar-only training

### 7.2 Stage 1: Dynamics Warmup

Train only the visual/action/context fusion and dynamic latent components with:

- `L_dyn_nce`
- `L_dyn_state`

Goal:

- stabilize a future-aware latent before full multitask training

### 7.3 Stage 2: Multitask FASA Training

Turn on all supervised heads:

- `risk`
- `trigger`
- `delta`
- `region`
- `dynamic embedding auxiliaries`

Recommended first setting:

- freeze the visual encoder or only tune a thin temporal adapter
- fully train the action/context encoders, decoder, and output heads

### 7.4 Stage 3: Optional Light Unfreezing

If region localization or trigger recall plateaus:

- unfreeze the last visual block or a small adapter subset
- do not convert this into a full planner retraining path

## 8) Deployment Workflow

### 8.1 Mainline-Compatible Deployment

The preferred initial deployment mode is offline or precomputed builder mode:

1. run the trained FASA sidecar over the target dataset
2. emit canonical `a_outputs`
3. attach them through the existing dataloader path
4. let the current standalone A consumer read them with no mainline command change

### 8.2 Optional Runtime Sidecar Mode

A future runtime mode is allowed, but not required for `P4.1` acceptance.

If enabled later:

- planner remains lower-frequency chunk planner
- FASA runs at a higher frequency than the planner, subject to hardware budget
- avoid claiming `100Hz+` as a hard guarantee in the RFC

Safer statement:

- target a higher update rate than the planner, e.g. `10-50 Hz` depending on hardware

## 9) Build Surfaces (Recommended File Split)

Recommended implementation surfaces:

- new sidecar model:
  - `starVLA/model/a_module/fasa_sidecar.py`
- new FASA losses:
  - `starVLA/model/a_module/fasa_losses.py`
- new sidecar trainer or training entrypoint:
  - `starVLA/training/train_fasa.py`
- new or extended dataset builder:
  - `tools/build_fasa_dataset.py` or a clearly-scoped extension of the current correction builder
- optional evaluation helpers:
  - `tools/eval_fasa.py`
  - `tools/visualize_fasa_region_bins.py`

This file split is recommended, not mandatory, but the implementation should preserve the system boundary:

- FASA training remains separate from the authoritative `Qwen2.5-VL` planner training path

## 10) Acceptance Criteria

### 10.1 Data Acceptance

The built dataset must satisfy:

1. all required payload-side supervision fields present for FASA training
2. `future_state` is backfilled from `t + H`, not recorded immediately at `t`
3. canonical `region_target_15` exists and is finite
4. all scalar targets are finite and auditable

### 10.2 Sidecar Output Acceptance

The FASA producer must emit payloads where:

1. `risk_pred` is finite
2. `trigger_logit` is finite
3. `delta_pred` is finite and non-negative
4. `region_logits` length is exactly `15`
5. `dynamic_embedding` length is exactly `16`
6. `version` and `source` are present

### 10.3 Mainline Integration Acceptance

When those outputs are consumed by the authoritative `Qwen2.5-VL` standalone path:

1. strict standalone contract remains enabled
2. `fallback_to_lite=false`
3. `strict_missing=true`
4. no shape mismatch on `region_logits` or `dynamic_embedding`
5. no fallback / restore / partial-freeze is introduced to justify success

### 10.4 Training Stability Acceptance

Minimum integration evidence package:

1. `5-step` smoke with strict standalone path passes
2. `50-step` smoke remains finite
3. no non-finite loss or silent fallback appears in logs
4. payload coverage and all-fields coverage are both near-complete

Long-run acceptance can follow once the standalone producer is stable.

## 11) Reference Pseudocode

### 11.1 Delayed Future-State Backfill Collector

```python
from collections import deque
import numpy as np


class FASARolloutCollector:
    def __init__(self, env, expert_policy, action_noise_prob=0.3, delay_prob=0.2, noise_std=0.05):
        self.env = env
        self.expert_policy = expert_policy
        self.p_action_noise = action_noise_prob
        self.p_delay = delay_prob
        self.noise_std = noise_std
        self.pending = deque()  # records waiting for t + H backfill
        self.obs_history = deque(maxlen=16)
        self.step_index = 0

    def _sample_horizon(self):
        return int(np.random.choice([1, 2, 4, 8]))

    def _inject_action_noise(self, base_action):
        executed = base_action.copy()
        info = {"perturbation_type": "none", "perturbation_strength": 0.0}
        if np.random.rand() < self.p_action_noise:
            noise = np.random.normal(0.0, self.noise_std, size=base_action.shape)
            executed = executed + noise
            info = {
                "perturbation_type": "action_noise",
                "perturbation_strength": float(np.linalg.norm(noise)),
            }
        return executed, info

    def _maybe_delay_observation(self, obs):
        # Placeholder: implementation may reuse a history frame for k steps.
        # Important part is to record the stale span and map it to region_target_15 later.
        return obs, {"delay_span": 0, "trigger_target": 0.0}

    def step(self, base_action_chunk, instruction, state_t):
        executed_action, noise_info = self._inject_action_noise(base_action_chunk[0])
        obs_next, reward, done, env_info = self.env.step(executed_action)
        obs_for_policy, delay_info = self._maybe_delay_observation(obs_next)

        horizon = self._sample_horizon()
        expert_action = self.expert_policy.get_action(self.env.get_state())
        delta_target = float(np.linalg.norm(expert_action - executed_action))

        record = {
            "step_index": self.step_index,
            "instruction": instruction,
            "base_action_chunk": np.asarray(base_action_chunk, dtype=np.float32),
            "state_t": np.asarray(state_t, dtype=np.float32),
            "horizon_steps": horizon,
            "perturbation_type": noise_info["perturbation_type"],
            "perturbation_strength": noise_info["perturbation_strength"],
            "delay_span": delay_info["delay_span"],
            "pseudo_labels": {
                "trigger_label": float(delay_info["trigger_target"]),
                "delta_action_norm": delta_target,
            },
        }
        self.pending.append(record)
        self._flush_ready_records()
        self.step_index += 1
        return obs_for_policy, reward, done, env_info

    def _flush_ready_records(self):
        finalized = []
        while self.pending:
            record = self.pending[0]
            target_step = record["step_index"] + int(record["horizon_steps"])
            if self.step_index < target_step:
                break
            record = self.pending.popleft()
            record["future_state_step"] = target_step
            record["future_state"] = np.asarray(self.env.get_state(), dtype=np.float32)
            # Build risk_score / region targets here using the finalized future window.
            finalized.append(record)
        return finalized
```

### 11.2 Sidecar Skeleton

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FASASidecar(nn.Module):
    def __init__(self, d_model=256, action_dim=7, state_dim=7, text_dim=1024):
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim + 1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.text_encoder = nn.Linear(text_dim, d_model)
        self.horizon_embed = nn.Embedding(32, d_model)

        self.visual_temporal = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)

        self.global_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.region_queries = nn.Parameter(torch.randn(1, 15, d_model))
        self.dynamic_queries = nn.Parameter(torch.randn(1, 4, d_model))

        self.risk_head = nn.Linear(d_model, 1)
        self.trigger_head = nn.Linear(d_model, 1)
        self.delta_head = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        self.region_head = nn.Linear(d_model, 1)

        self.dynamic_internal = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 64),
        )
        self.dynamic_out = nn.Linear(64, 16)
        self.future_state_proj = nn.Linear(16, 64)  # auxiliary path, not exported

    def forward(self, visual_tokens, base_action_chunk, action_pos, instruction_emb, state_t, horizon_steps):
        # visual_tokens: [B, T, D]
        visual_memory = self.visual_temporal(visual_tokens.transpose(1, 2)).transpose(1, 2)

        action_input = torch.cat([base_action_chunk, action_pos], dim=-1)
        action_tokens = self.action_encoder(action_input)

        context = (
            self.state_encoder(state_t)
            + self.text_encoder(instruction_emb)
            + self.horizon_embed(horizon_steps)
        ).unsqueeze(1)

        memory = torch.cat([visual_memory, context], dim=1)
        aligned_actions, _ = self.cross_attn(action_tokens, memory, memory)

        B = aligned_actions.shape[0]
        global_q = self.global_query.expand(B, -1, -1)
        region_q = self.region_queries.expand(B, -1, -1)
        dynamic_q = self.dynamic_queries.expand(B, -1, -1)

        global_tok, _ = self.cross_attn(global_q, aligned_actions, aligned_actions)
        region_tok, _ = self.cross_attn(region_q, aligned_actions, aligned_actions)
        dynamic_tok, _ = self.cross_attn(dynamic_q, aligned_actions, aligned_actions)

        global_tok = global_tok.squeeze(1)
        region_logits = self.region_head(region_tok).squeeze(-1)  # [B, 15]

        dyn_hidden = self.dynamic_internal(dynamic_tok.mean(dim=1))
        dynamic_embedding = self.dynamic_out(dyn_hidden)  # exactly [B, 16]

        risk_logit = self.risk_head(global_tok).squeeze(-1)
        trigger_logit = self.trigger_head(global_tok).squeeze(-1)
        delta_pred = self.delta_head(global_tok).squeeze(-1)

        return {
            "risk_logit": risk_logit,
            "risk_pred": torch.sigmoid(risk_logit),
            "trigger_logit": trigger_logit,
            "delta_pred": delta_pred,
            "region_logits": region_logits,
            "dynamic_embedding": dynamic_embedding,
            "dynamic_aux": dyn_hidden,
        }
```

### 11.3 Loss Skeleton

```python
import torch
import torch.nn.functional as F


def focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def soft_iou_loss(probs, targets, eps=1e-6):
    inter = (probs * targets).sum(dim=-1)
    union = (probs + targets - probs * targets).sum(dim=-1)
    return 1.0 - ((inter + eps) / (union + eps)).mean()


def compute_fasa_loss(preds, targets, future_state_encoder, temperature=0.1):
    risk_logit = preds["risk_logit"]
    risk_pred = preds["risk_pred"]
    trigger_logit = preds["trigger_logit"]
    delta_pred = preds["delta_pred"]
    region_logits = preds["region_logits"]
    z = preds["dynamic_embedding"]

    risk_score = targets["risk_score"]
    trigger_target = targets["trigger_target"]
    delta_target = targets["delta_target"]
    region_target_15 = targets["region_target_15"]
    region_prior_15 = targets["region_prior_15"]
    future_state = targets["future_state"]

    l_risk = F.binary_cross_entropy_with_logits(risk_logit, risk_score) + 0.5 * F.smooth_l1_loss(risk_pred, risk_score)
    l_trigger = focal_bce_with_logits(trigger_logit, trigger_target)
    l_delta = F.smooth_l1_loss(torch.log1p(delta_pred), torch.log1p(delta_target))

    region_prob = torch.sigmoid(region_logits)
    l_region_focal = focal_bce_with_logits(region_logits, region_target_15)
    l_region_iou = soft_iou_loss(region_prob, region_target_15)
    l_region_prior = F.mse_loss(region_prob, region_prior_15)
    l_region_smooth = ((region_prob[:, 1:] - region_prob[:, :-1]) ** 2).mean()

    future_state_enc = future_state_encoder(future_state)
    l_dyn_state = F.smooth_l1_loss(preds["dynamic_aux"], future_state_enc)

    # Placeholder: add an InfoNCE implementation once sampling utilities are wired.
    l_dyn_nce = z.new_zeros(())

    total = (
        1.0 * l_risk
        + 2.0 * l_trigger
        + 1.0 * l_delta
        + 1.0 * l_region_focal
        + 0.5 * l_region_iou
        + 0.25 * l_region_prior
        + 0.1 * l_region_smooth
        + 0.2 * l_dyn_nce
        + 0.1 * l_dyn_state
    )
    return total
```

## 12) Final Decisions Locked by This RFC

1. the accepted mainline remains the authoritative `Qwen2.5-VL` baseline
2. FASA is a standalone sidecar, not a mainline planner rewrite
3. the external `a_outputs` contract remains frozen
4. `dynamic_embedding` is exported as exactly `16` dimensions
5. `region_logits` are `15` independent correction-relevance logits, not a categorical class distribution
6. `future_state` supervision is horizon-backfilled and belongs to the FASA training path
7. OT-style losses may be evaluated later, but are not the only or required mainline objective in v1

