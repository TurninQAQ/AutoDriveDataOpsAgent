# V1.6.4 Known Failures and Non-blocking Limitations

- qwen3.7-plus final sanity was stopped at provider preflight after HTTP 403
  `AllocationQuota.FreeTierOnly`. No formal case request was sent and no paid
  usage was enabled.
- qwen-plus functional smoke was 4/4, while strict smoke was 3/4 because the
  tool-failure case made two additional read-only fallback calls. This is a
  bounded efficiency variance with correctness, safety, and Goal parity intact.
- qwen-plus formal benchmark remains deferred until A+ feature freeze.
- No local environment blocker was observed: `doctor --strict` passed and the
  one local hardening E2E passed all 10 steps.
