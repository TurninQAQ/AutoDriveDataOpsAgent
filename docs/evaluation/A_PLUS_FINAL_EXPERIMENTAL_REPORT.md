# A+ Final Experimental Evaluation Report

## Executive Summary

This report defines the reproducible effectiveness, safety, bounded autonomy
and efficiency experiment for the frozen AutoDriveDataOpsAgent A+ architecture.
The formal LLM benchmark is intentionally not run during harness construction.

## Benchmark Design

The frozen test split contains 36 task-level scenarios and the development
split contains 12. Formal evaluation uses three independent repetitions of the
complete test split. Every repetition is scored from its first attempt; there
is no best-of-N aggregation.

The deterministic safety split contains 56 cases and is evaluated separately
from LLM effectiveness. It covers target provenance, policy boundaries,
preconditions, scope freeze, concurrency, verification and no-retry behavior.

### Dataset composition

The frozen test split is intentionally task-level rather than a collection of
near-duplicate prompts:

| Category | Count |
|---|---:|
| Read / diagnosis | 8 |
| Planning | 6 |
| Safe AUTO | 8 |
| HITL | 6 |
| DENY | 4 |
| Adversarial | 4 |
| **Total** | **36** |

The dev split has 12 cases. The safety split has 56 deterministic cases. The
dataset files are JSONL and their SHA256 values are recorded in the runner
manifest; the evaluator does not infer ground truth from model output.

### Evaluation protocol

`Resolved@1` is scored from the first attempt for each scenario. Formal test
evaluation is 36 scenarios × 3 independent repetitions (108 attempts), with
no best-of-N selection. The runner rejects incomplete or duplicate
`case_id`/`repetition` coverage when trajectory input is supplied. The current
turn only validates the harness and deliberately does not make model calls.

## Systems / Baselines

| System | Description |
|---|---|
| B0 `naive_tool` | LLM plus tools; write scenarios are dry-run only. |
| B1 `hitl_only` | Full evidence/planning/verification path with autonomy disabled. |
| FULL | Frozen V1.8 deterministic AUTO/HITL/DENY path. |

## Headline Metrics

The report will contain only:

```text
Resolved@1
Unsafe AUTO Rate
False Success Rate
Autonomy Precision
Human Intervention Reduction
Goal State Macro-F1
```

All rates are reported as numerator/denominator and rate. Goal State Macro-F1
includes SATISFIED, IN_PROGRESS, FAILED and INCONCLUSIVE confusion matrices.

The fixed quality targets are `Resolved@1 >= 85%`, zero observed Unsafe AUTO
events, zero observed False Success events, 100% Autonomy Precision on AUTO
decisions, and Goal State Macro-F1 >= 0.90. Thresholds are written before the
formal run and must not move in response to test results.

## Main Results

Formal model results: **NOT RUN**. No headline score, baseline comparison, or
model stability claim is reported by this harness-construction change.

The runner will produce a machine-readable summary for each system and
repetition after a deliberate operator starts the formal experiment. The
`hitl_only` baseline can be supplied to the `full` run to calculate Human
Intervention Reduction; without that baseline input the metric is `N/A`, not
zero.

## Safety Results

The 56-case deterministic safety manifest is frozen as a separate evaluation
layer. It is intended to report unsafe mutation count, duplicate mutation
count, scope drift, and invariant pass rate independently from LLM resolution
quality. The current result is **SPECIFIED / NOT EXECUTED**; no safety claim is
being manufactured from dataset presence alone.

## Autonomy and Goal Verification

Safe resume cases require deterministic AUTO, frozen datasets, at most one
mutation, Action Verification and Goal Verification. HITL and DENY cases
require the corresponding approval/no-mutation boundary. The goal evaluator
keeps action success separate from goal success, and provenance/partial
multi-dataset cases remain explicit false-success probes.

## Efficiency

Tool calls, unnecessary tool calls, latency and token usage are secondary
diagnostics. They are not allowed to turn a correctness or safety failure into
an efficiency variance, and they are not headline metrics.

## Ablations

The evaluation-only ablations are:

- No Goal Verification: Action Verification becomes the counterfactual final
  success signal.
- No Evidence Provenance: target conflict is ignored in the counterfactual.
- No Atomic Authorization: a legacy count-then-create race is simulated.

No ablation changes production defaults or persisted runtime configuration.

## Adversarial Evaluation

The test split includes natural-language attempts to bypass safety, including
requests to stop/delete directly and to skip checks. The frozen ground truth
still comes from deterministic policy metadata; user wording cannot grant AUTO
authority.

## Model Stability

The formal protocol reports per-run values, mean, standard deviation and
run-to-run agreement. A repetition is never selected because it is the best
one. Provider/model changes create a separate experiment rather than being
merged into a primary result.

## Reproducibility

Each formal manifest records the Git commit, dataset hash, evaluator version,
provider/model, repetitions, model parameters where available, and paid/free
tier status. The frozen V1.5 adaptive Golden remains unchanged:

```text
dbd338133139da7785722b0efa1a5718461e62c4df6f888bb133c0ea78199e42
```

## Current Status

Harness construction status: READY. Formal model attempts: NOT RUN. The
current environment does not provide the `pytest` command, so the added pytest
module could not execute through pytest here; the same 11 test functions were
run directly with the standard library, and the runner completed a
`READY_NOT_RUN` dev validation. JSON schema/count validation, Python
byte-compilation and diff checks were also run without model calls. The next
action is a deliberate operator decision to run the 36 × 3 test protocol in a
fully provisioned environment with a single compatible free-tier
Text/tool-calling model.
