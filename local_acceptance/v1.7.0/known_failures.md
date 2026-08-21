# Known limitations / non-blocking items

- No new provider calls were made for V1.7. The prior qwen-plus-2025-07-28
  functional smoke remains PASS (4/4 functional, 1 bounded read-only efficiency
  variance) and is referenced without rewriting its artifact.
- qwen3.7-plus remains recorded as BLOCKED_FREE_TIER from the V1.6.4 403
  `AllocationQuota.FreeTierOnly` event. It was not retried.
- Formal qwen-plus benchmark remains deferred until A+ feature freeze.
- Existing local hardening E2E passed; bounded AUTO/HITL execution paths are
  covered by the new deterministic integration tests rather than a second
  external runtime fixture.
- The stale V1.6.1 test expectation for metadata-only diagnostic context was
  aligned with the already documented V1.6.3/V1.6.4 production contract; no
  production safety behavior was relaxed.
- `pytest -q` repository-wide collection remains blocked by the historical DAG
  fixture `/opt/airflow/config/datasets_config_test.yaml` being absent.
- `pytest -q tests` was stopped after 145 passed and 1 skipped at the known
  long-running GPU allocator test (`platform_core/services/gpu_allocator.py`);
  the V1.7 affected suite had already completed successfully.
