---
id: workflow
kind: skill
triggers:
  - workflow
  - development workflow
  - refactor workflow
  - feature migration
  - migrate feature
  - port feature
  - environment
  - editable
  - FA4
  - plan
  - validation gate
  - version lock
  - lock version
  - UT
  - test
  - hang
  - benchmark
  - stress
  - hang stress
  - commit
---

# Workflow Skill

Overall development gate for CuteDSL HD256 work.

## When to Use

| Signal | Action |
| ------ | ------ |
| workflow, development flow | Use the gate sequence below. |
| environment, editable, FA4 | Load `environment.md` before any downstream gate. |
| plan | Produce major steps before editing. |
| feature migration, migrate feature, port feature | Load `feature_migration.md` and `refactor.md`; keep edits scoped to the current HD256 kernel and feature UT. |
| validation gate, UT, test | Load `test.md` and run monitored tests. |
| benchmark | Load `benchmark.md`; run only after UT passes. |
| stress, hang stress | Load `hd256_hang_stress.md`; run only after UT and benchmark pass. |
| version lock, lock version | Lock only after UT passes, benchmark does not regress, and 5000-iteration stress passes. |
| commit | Load `commit.md` after version lock. |

## Gate Sequence

| Gate | Required Action | Exit Criteria |
| ---- | --------------- | ------------- |
| W0 Environment | Load `environment.md`. Ensure `flash_attn/cute` editable runtime is active and repo-local import path is verified. | `flash_attn.cute.interface` resolves under this checkout. |
| W1 Analyze | Read source plus target/reference files. Classify differences as alignment work, HD256 divergence, behavior risk, or missing feature. | Analysis exists before edits. |
| W2 Plan | Create major steps. Avoid tiny edits and unrelated mixed layers. | Each step is independently gateable and rollbackable. |
| W3 Implement | Implement one major step. For refactor, obey `refactor.md` edit allowlist. For feature migration, obey `feature_migration.md` scope and target-style rewrite rules. | Scope matches loaded skill. |
| W4 Validate | Run monitored UT via `test.md`, benchmark via `benchmark.md`, then finite HD256 hang stress via `hd256_hang_stress.md`. | UT passes, benchmark does not regress, and stress exits cleanly. |
| W5 Lock | Record locked state only after W4 passes. | Next step may start only from locked state. |
| W6 Commit | Load `commit.md`, then `../commands/commit.md`. | Commit scope and identity checks pass. |

## Hard Rules

| Rule | Requirement |
| ---- | ----------- |
| No environment skip | W0 must pass before refactor, UT, benchmark, stress, or commit. |
| No ad hoc edits | Analysis and plan come before code changes. |
| No feature drift | Feature migration must start from the target file's corresponding implementation and stay inside the current HD256 kernel plus the relevant feature UT. |
| No helper detour | Feature migration must not edit helper/shared files or unrelated kernels without explicit approval. |
| No tiny-step plans | A step must represent a coherent architectural layer. |
| No raw UT gate | Use `test.md`; it owns UT selection, fail-fast, and hang detection. |
| No benchmark-before-UT | Benchmark runs after UT passes. |
| No skipped benchmark | Performance-sensitive work cannot lock without a completed `benchmark.md` run after the latest UT pass. |
| No stress-before-benchmark | Stress gate runs after UT and benchmark pass. |
| No skipped stress gate | HD256 kernel/source changes cannot lock without a finite `hd256_hang_stress.md` gate after the latest successful benchmark. |
| No implicit lock | After UT, benchmark, and stress, explicitly decide whether W4 passed before W5/W6. |
| No unlocked continuation | If UT fails, benchmark regresses, or stress fails/hangs, fix or roll back before the next step. |
| No commit-before-lock | Commit only after the current step is locked. |
| Split harness commit | If kernel/source changes and `harness/` changes both exist, commit kernel/source and harness in separate commits. |
| Unknown command | Stop and load or create the matching command doc under `../commands/`. |
