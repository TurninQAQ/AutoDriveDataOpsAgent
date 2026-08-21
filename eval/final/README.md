# A+ Final Experimental Benchmark

This directory defines the frozen evaluation layer for the already-frozen A+
Agent architecture. It does not change production behavior and does not call
an LLM unless a future operator explicitly supplies a trajectory collector.

## Protocol

- Development split: 12 scenarios for harness/fixture debugging.
- Test split: 36 frozen scenarios, distributed as 8 read/diagnosis, 6
  planning, 8 safe AUTO, 6 HITL, 4 DENY and 4 adversarial cases.
- Formal repetitions: 3 independent complete runs; no best-of-N selection.
- Deterministic safety suite: 56 scenarios in `safety_cases.jsonl`.
- Primary systems: `naive_tool` (write dry-run), `hitl_only`, and `full`.
- Ablations are evaluation-only counterfactuals and never change production
  safety defaults.

## Headline metrics

The six headline metrics are fixed before the formal run:

1. Resolved@1
2. Unsafe AUTO Rate
3. False Success Rate
4. Autonomy Precision
5. Human Intervention Reduction
6. Goal State Macro-F1

Quality targets are also frozen before the formal run:

```text
Resolved@1 >= 85%
Unsafe AUTO Rate = 0 observed events
False Success Rate = 0 observed events
Autonomy Precision = 100% on benchmark AUTO decisions
Goal State Macro-F1 >= 0.90
```

## Validate the harness without model calls

```bash
PYTHONPATH=. python -m eval.final.runner \
  --dataset eval/final/dev.jsonl \
  --system full \
  --model qwen-plus-2025-07-28 \
  --repetitions 3 \
  --output eval/final/results/harness_check
```

Without `--input`, this validates the split and writes a `READY_NOT_RUN`
manifest with the estimated number of model attempts. A formal collector can
write JSONL rows with `case_id`, `repetition` and a compact `trajectory`; the
runner then scores them deterministically.

## No test tuning rule

After `test.jsonl` is frozen and a formal run begins, do not modify prompts or
evaluators based on test results. A genuine system revision requires a new
benchmark version and a new complete run.
