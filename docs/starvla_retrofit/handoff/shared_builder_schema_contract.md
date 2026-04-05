# StarVLA Retrofit Shared Builder Schema Contract (P2)

- Owner: INFRA(T2)
- Scope: Qwen2.5VL / Qwen3VL path only
- Status: active for `p1_shared_builder_v1`

## 1) Purpose

Fix the T1 builder output into a machine-checkable contract so T2/T3/T4 share the same acceptance criteria.

## 2) Schema Version

- Current: `p1_shared_builder_v1`
- Evolution rule:
1. Additive fields keep backward compatibility.
2. Breaking field/shape changes require a new `schema_version`.
3. Validator must be updated before trainer/eval acceptance switches to new version.

## 3) Contract Modes

Two modes are allowed and selected by `shared_builder_enabled`.

| Mode | `shared_builder_enabled` | Goal |
|---|---|---|
| Shared | `true` | Expose unified builder fields for retrofit path |
| Legacy | `false` | Preserve old training behavior |

## 4) Field Contract

### 4.1 Base Fields (both modes)

| Field | Type | Required | Constraint |
|---|---|---|---|
| `sample_keys` | `list[str]` | Yes | Must include `action,image,lang` |
| `action_shape` | `[T,D]` | Yes | `T>0,D>0`; usually `T=action_chunk_size`, `D=action_dim` |
| `state_shape` | `list[int] \| null` | Optional | Present if state is enabled |

### 4.2 Shared Mode Fields (`shared_builder_enabled=true`)

| Field | Type | Required | Constraint |
|---|---|---|---|
| `action_chunk_shape` | `[T,D]` | Yes | Must equal `action_shape` |
| `meta` | `dict` | Yes | Must include required meta keys |
| `meta.schema_version` | `str` | Yes | Must be `p1_shared_builder_v1` |
| `meta.action_chunk_len` | `int` | Yes | Must equal `action_chunk_shape[0]` |
| `meta.action_dim` | `int` | Yes | Must equal `action_chunk_shape[1]` |
| Shared-only sample keys | `obs,action_chunk,meta` | Yes | Must exist in `sample_keys` |

Required meta keys:
- `dataset_name`
- `trajectory_id`
- `sample_step`
- `action_keys`
- `state_keys`
- `video_keys`
- `action_chunk_len`
- `action_dim`
- `schema_version`

### 4.3 Legacy Mode Fields (`shared_builder_enabled=false`)

| Field | Type | Required | Constraint |
|---|---|---|---|
| `action_chunk_shape` | `null` | Yes | Must be null in strict mode |
| `meta` | `null` | Yes | Must be null in strict mode |
| Shared-only keys | absent | Yes | `obs,action_chunk,meta` should not appear |

## 5) Validator CLI

Path:
- `<REPO_ROOT>/tools/handoff/validate_shared_builder_schema.py`

Primary commands:

```bash
python <REPO_ROOT>/tools/handoff/validate_shared_builder_schema.py --help
```

```bash
python <REPO_ROOT>/tools/handoff/validate_shared_builder_schema.py \
  --payload-json /tmp/shared_payload.json \
  --mode auto \
  --expected-schema-version p1_shared_builder_v1 \
  --expected-action-chunk-len 16 \
  --expected-action-dim 7
```

```bash
python <REPO_ROOT>/tools/handoff/validate_shared_builder_schema.py \
  --smoke-log /path/to/shared_builder_smoke.log \
  --mode auto
```

Demo payload generation:

```bash
python <REPO_ROOT>/tools/handoff/validate_shared_builder_schema.py --demo shared_ok
python <REPO_ROOT>/tools/handoff/validate_shared_builder_schema.py --demo legacy_ok
python <REPO_ROOT>/tools/handoff/validate_shared_builder_schema.py --demo shared_bad
```

## 6) Mapping for T3 and T4

- For T3 acceptance:
1. Shared mode pass condition: shared keys exist, shape consistency holds, `meta.schema_version` matches.
2. Legacy mode pass condition: legacy keys stay stable, shared-only fields absent.

- For T4 trainer insertion:
1. If `shared_builder_enabled=true`, assert `action_chunk` exists and `meta.action_chunk_len/action_dim` are consistent.
2. If `shared_builder_enabled=false`, fallback to legacy `action` path without assuming `meta`.

## 7) Known Environment Notes

1. Remote smoke may need data root/backend override depending on server dataset layout.
2. Validator is data-source agnostic: it validates payload/log content only.

## 8) Trainer-Side Optional Assertion (T4 handshake)

`train_starvla.py` supports config-gated contract checks that are default-off:

```yaml
trainer:
  shared_builder_contract_check:
    enabled: false
    mode: auto
    expected_schema_version: p1_shared_builder_v1
    expected_action_chunk_len: null
    expected_action_dim: null
    require_state: false
    strict_legacy_no_extra: true
```

When enabled:
1. Contract checks run before forward in `_train_step`.
2. Violations fail fast with readable error.
3. Extra debug metrics are emitted:
   - `debug/shared_builder_contract_checked`
   - `debug/shared_builder_contract_is_shared`
   - `debug/shared_builder_contract_fail` (only on failure path)
