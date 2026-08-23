# Structured model provider

`providers/http_structured.py` implements an OpenAI-compatible structured
response adapter and `providers/qwen.py` supplies the Qwen/DashScope binding.
The request carries three explicitly separated channels:

1. Runtime Structured Context (authoritative deterministic state);
2. Operating Guidance (advisory principles);
3. Semantic Observation Context (`UNTRUSTED_EXTERNAL_DATA`).

The model returns one JSON AgentDecision proposal. It is parsed and then sent
through the existing `AgentDecisionIngressValidator`; malformed JSON/schema,
unknown variants, invalid goals, and invalid tool arguments become bounded
decision rejection. Transport timeout, rate limit, 5xx, and network failures
use explicit finite retry/timeouts and become provider-unavailable only after
the retry budget.

Credentials are referenced by environment-variable name. Secrets are never
placed in prompts, telemetry, event payloads, or logs. Tests use `httpx` local
MockTransport only; no paid model call is part of the regression suite.

## External smoke status

The production endpoint is configured by `AUTODRIVE_PROVIDER_ENDPOINT` and the
secret is supplied through the environment variable named by
`AUTODRIVE_PROVIDER_API_KEY_ENV` (normally `DASHSCOPE_API_KEY`). The current
review environment has no non-empty provider key, so no real Qwen/DashScope
request was sent. The result is **PENDING**, not PASS. The local adapter suite
independently covers valid structured output, malformed output, timeout, 429,
5xx, network failure, and bounded typed failure behavior.
