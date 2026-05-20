---
id: feature_migration
kind: skill
triggers:
  - feature migration
  - migrate feature
  - port feature
  - feature development
  - enable feature
  - enable UT
  - sliding window
  - SWA
  - local attention
---

# Feature Migration Skill

Strict rules for migrating a target-file feature into the CuteDSL HD256
kernels under `../flash_attn/cute/`.

## When to Use

| Signal | Action |
| ------ | ------ |
| feature migration, migrate feature, port feature | Load this skill plus `workflow.md` and `refactor.md`. |
| enable feature | Find the target-file implementation first, then rewrite only the needed subset into the current HD256 kernel. |
| SWA, local attention, sliding window | Treat the target forward/backward files as the source of truth for semantics, masks, scheduler interaction, and UT coverage. |
| enable UT | Open the relevant feature UT and run the full workflow gate before commit. |

## Target Map

Use the same target/current mapping as `refactor.md`.

| HD256 Area | Target File | Rule |
| ---------- | ----------- | ---- |
| Forward kernel | `../flash_attn/cute/flash_fwd_sm100.py` | Migrate forward features as an HD256 specialization of the forward target. |
| Backward entry | `../flash_attn/cute/flash_bwd_sm100.py` | Migrate entry-level backward feature handling from the backward target. |
| Backward dQ kernel | `../flash_attn/cute/flash_fwd_sm100.py` + `../flash_attn/cute/flash_bwd_sm100.py` | Follow forward-style structure where dQ recomputes forward values; follow backward target for gradient semantics. |
| Backward dK/dV kernel | `../flash_attn/cute/flash_bwd_sm100.py` | Migrate dK/dV feature behavior from the backward target. |

## Edit Scope

| File Type | Rule |
| --------- | ---- |
| Current HD256 kernel | Editable only for the feature being migrated. |
| Feature UT | Editable only to enable or add coverage for the migrated feature. |
| Other kernels | Do not edit. |
| Helper/shared files | Do not edit helper files, scheduler files, pipeline files, mask/block-info utilities, or other shared CuteDSL infrastructure. |
| Target files | Read-only. |

If the feature appears to require a helper or shared-file change, stop and ask
for explicit approval before editing outside the current kernel and feature UT.

## Migration Rules

| Area | Requirement |
| ---- | ----------- |
| Source of truth | Start from the target file's implementation of the relevant feature. Do not design a parallel local feature path from memory. |
| Rewrite style | Rewrite the target logic into the current kernel line by line as needed. Preserve target-style naming, order, indentation, comments, helper calls, layout construction, and control-flow shape where the role matches. |
| HD256 specialization | Keep only the needed HD256 and 2CTA subset. Do not copy unsupported target features or add generalized scaffolding. |
| No compatibility shim | Do not keep old and new paths side by side. Replace the current code with target-style feature logic. |
| No invented helper | Do not introduce local helper functions that do not exist in the target when equivalent target-style inline code is possible. |
| Divergence | Any remaining difference from the target must come from the current kernel's HD256/2CTA design or the requested feature boundary. |

## Validation

| Gate | Requirement |
| ---- | ----------- |
| Environment | Pass `workflow.md` W0 before testing. Runtime must import repo-local `flash_attn.cute`. |
| UT | Enable the feature's relevant UT coverage, then run monitored UT through `test.md`. Raw ad hoc pytest is not the gate. |
| Benchmark | After UT passes, run `benchmark.md`. Feature migration cannot lock or commit without benchmark. |
| Stress | After benchmark passes, run the finite `hd256_hang_stress.md` gate with `--max-iters 5000`. Feature migration cannot lock or commit without stress. |
| Hang | If a run hangs, immediately follow `hang_detect_fix.md`: kill the stale UT, reproduce narrowly, capture diagnostics, and fix before continuing. |
| Commit | Commit only after workflow W4 passes and W5 lock is explicit. Feature UT changes are allowed only for the migrated feature. |
