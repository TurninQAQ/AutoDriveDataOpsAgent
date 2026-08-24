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

## Structured-output modes

ProviderConfig.structured_output_mode, configured with
AUTODRIVE_PROVIDER_STRUCTURED_MODE, accepts auto, json_schema, and json_object.
Known qwen3.7-plus models select the Bailian JSON Schema contract:

    response_format.type = json_schema
    json_schema.name = agent_decision
    json_schema.strict = true

The schema covers the V2 decision variants, GoalDescriptor goal kinds, and
canonical tool surface. It requires the initial GoalDescriptor and non-empty
call_id values. The local parser and Runtime-owned DecisionIngress remain
mandatory after model-side validation.

Older models such as qwen-plus-2025-07-28 retain json_object compatibility
mode. It permits at most one bounded model regeneration after a
schema-invalid response; it never repairs fields locally or bypasses
DecisionIngress. Strict json_schema mode fails closed without regeneration.

Credentials are referenced by environment-variable name. Secrets are never
placed in prompts, telemetry, event payloads, or logs. Tests use `httpx` local
MockTransport only; no paid model call is part of the regression suite.

## External smoke status

The production endpoint is configured by `AUTODRIVE_PROVIDER_ENDPOINT` and the
secret is supplied through the environment variable named by
`AUTODRIVE_PROVIDER_API_KEY_ENV` (normally `DASHSCOPE_API_KEY`). On 2026-08-24
the real validation process loaded the existing secret file only into its
process environment and used:

```text
provider: qwen
model: qwen-plus-2025-07-28
endpoint class: shared Beijing DashScope OpenAI-compatible
platform: localhost V2 in-process mock/simulated gateway
```

The production `QwenProvider` sent real requests to Bailian, parsed structured
AgentDecision proposals, and the Runtime/DecisionIngress boundary accepted
valid READ proposals. Historical compatibility-mode evidence includes a clean
single-READ run. The current strict-schema clean-restart evidence is recorded
separately below; it must not be merged with the historical result.

The model emitted some malformed proposals during bounded recovery. They were
rejected by the existing typed Provider/DecisionIngress path; no parser bypass,
WRITE approval, or mutation occurred. Queue results are normalized at the V2
platform boundary from the canonical queue-file shape (`active: null` or an
active object) into the strict platform queue result contract. The local
adapter suite independently covers valid structured output, malformed output,
timeout, 429, 5xx, network failure, and bounded typed failure behavior.

## Strict-schema clean-restart evidence

On 2026-08-24, after a clean Gateway restart, the deployment used:

    provider: qwen
    model: qwen3.7-plus-2026-05-26
    structured output: json_schema / strict=true
    endpoint class: shared Beijing DashScope OpenAI-compatible
    platform: localhost V2 in-process mock/simulated gateway

The first real request returned a valid SINGLE_TOOL_CALL(get_gpu_pool) with
valid GoalDescriptor and call ID. A fresh single-READ LangGraph run completed
through platform evidence, FINAL_CANDIDATE, and CompletionGate; platform
WRITE remained zero. A fresh multi-READ run emitted a valid
READ_TOOL_BATCH(get_gpu_pool, get_queue_state, search_knowledge) and all
three observations were normalized, but the subsequent model turn timed out
and the Runtime terminated safely as PROVIDER_UNAVAILABLE. A bounded
five-run sample produced one completed run and four provider-unavailable
terminations, with zero schema-invalid decisions and zero WRITE calls. This
is recorded as partial external reliability evidence; release readiness is not
claimed until the external model service is stable for the required sample.
