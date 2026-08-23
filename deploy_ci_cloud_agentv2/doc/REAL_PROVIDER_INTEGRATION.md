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
