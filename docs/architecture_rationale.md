# Architecture Rationale - Banking AI Loan Inquiry Orchestration

## Architecture Overview

The solution uses a sequential three-agent pipeline with fail-safe defaults. Each step has a narrow responsibility, and customer-facing text is never shown until the final compliance gate passes.

```text
Customer Message
  |
  v
[Agent 1: guardrailCheck] -- block/escalate --> End
  |
  v
[Agent 2: inquiryParser] -- parse/API error --> End
  |
  v
[Agent 3: safetyChecks] -- retry once --> safetyChecks again
  |                                      |
  |                                      +-- still fail --> Human handoff
  v
Customer Response
```

## Model Selection Rationale

### Agent 1 - guardrailCheck: `llama-3.1-8b-instant` + deterministic rules

Task: return three booleans for banking relevance, sensitive PII, and distress escalation.

Why this design:

- The LLM handles varied customer phrasing.
- Deterministic rules override high-confidence signals such as SSNs, account/card-like numbers, loan keywords, and self-harm/distress phrases.
- This hybrid approach improves reliability on adversarial prompts where a lightweight classifier may be overly conservative.
- The lightweight model keeps latency and cost low.
- Model comparison showed the 70B model did not improve standalone guardrail accuracy, so the selected design uses the faster 8B model plus explicit policy checks for zero-tolerance cases.

### Agent 2 - inquiryParser: `llama-3.3-70b-versatile`

Task: extract intent, loan amount, employment status, risk estimate, and a compliant customer response.

Why this model:

- It handles the highest-reasoning part of the pipeline.
- It applies the simplified underwriting policy across incomplete and ambiguous inputs.
- It produces more stable JSON and more professional response language than the 8B model in this design.
- It is used only once per successful inquiry, so the added cost is contained.
- Model comparison showed the 70B model reached 100.0% on the evaluated structured inquiry cases, compared with 75.0% for the 8B model.

### Agent 3 - safetyChecks: `llama-3.1-8b-instant`

Task: enforce four final-output checks: no guarantees, no ECOA/fair-lending violation, no UDAAP issue, and exact disclaimer presence.

Why this model:

- It reviews a short generated response rather than a complex conversation.
- The task is closer to rule classification than open-ended reasoning.
- The final gate adds modest latency while reducing the chance of non-compliant output reaching a customer.
- Model comparison showed both tested models achieved 100.0% compliance-check accuracy, so the lower-latency 8B model was selected.

## Tradeoffs

| Tradeoff | Decision |
| --- | --- |
| Cost vs accuracy | Use the 70B model only for inference/generation; use lightweight checks before and after. |
| Latency vs safety | Keep the final compliance gate even though it adds a model call. |
| Reliability vs model flexibility | Add deterministic guardrail overrides for high-confidence policy signals. |
| Availability vs compliance | Fail closed on malformed outputs, API failures, and compliance failures. |

## Key Design Decisions

1. Sequential orchestration is required because Agent 3 reviews Agent 2's generated response.
2. Distress escalation is prioritized before non-banking blocking, so self-harm or urgent crisis language always routes to a human advisor.
3. Sensitive PII is blocked before downstream inference.
4. Compliance failures trigger one stricter retry, then human handoff.
5. The required disclaimer is enforced as an exact phrase for auditability.
