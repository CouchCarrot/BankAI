# Evaluation Report - Banking AI Loan Inquiry Orchestration

## Executive Summary

The Banking AI Loan Inquiry Orchestration system was evaluated using a custom Python evaluation framework. The final end-to-end pipeline passed all 26 evaluation cases, covering normal loan inquiries, ambiguous requests, PII injection, adversarial prompts, and compliance-focused edge cases.

| Metric | Result |
| --- | ---: |
| Total evaluation cases | 26 |
| Passed | 26 |
| Failed | 0 |
| Overall accuracy | 100.0% |
| Guardrail accuracy | 100.0% |
| Compliance pass rate | 100.0% |
| P50 latency | 1536.8ms |
| P95 latency | 1764.06ms |
| Average latency | 1311.1ms |

## Evaluation Dataset

| Category | Cases | Purpose |
| --- | ---: | --- |
| Normal loan inquiries | 8 | Validate mortgage, auto, personal, refinance, employment, and risk handling |
| Ambiguous requests | 4 | Validate vague banking requests and non-banking blocking |
| PII injection | 4 | Validate SSN, account number, card number, and masked-account handling |
| Adversarial prompts | 5 | Validate jailbreak, prompt injection, role-play attack, and encoding attack handling |
| Compliance-focused prompts | 5 | Validate ECOA edge cases, distress escalation, and self-harm escalation |

## End-to-End Pipeline Results

| Agent / Stage | Selected Model | Evaluation Focus | Result |
| --- | --- | --- | --- |
| guardrailCheck | `llama-3.1-8b-instant` + deterministic policy checks | Banking relevance, PII detection, distress escalation | 100.0% guardrail accuracy |
| inquiryParser | `llama-3.3-70b-versatile` | Intent classification, risk scoring, structured extraction, compliant response | 100.0% on evaluated final pipeline cases |
| safetyChecks | `llama-3.1-8b-instant` | Guarantee language, ECOA, UDAAP, required disclaimer | 100.0% compliance pass rate |
| End-to-end pipeline | Combined multi-agent workflow | Full routing, blocking, retry, and escalation behavior | 26/26 passed |

## Category Breakdown

| Category | Pass Rate | Notes |
| --- | ---: | --- |
| Normal loan inquiries | 100.0% | All standard loan types handled correctly |
| Ambiguous requests | 100.0% | Non-banking requests blocked; vague banking requests routed safely |
| PII injection | 100.0% | Sensitive PII blocked before downstream inference |
| Adversarial prompts | 100.0% | Prompt injection and jailbreak attempts did not bypass compliance controls |
| Compliance-focused prompts | 100.0% | ECOA, distress, and self-harm escalation scenarios handled correctly |

## Three-Model Comparison

The model comparison uses Groq-hosted models so the results are reproducible with one API key and not affected by external provider billing or quota issues. The comparison was run on a representative subset of cases for each agent.

| Model | Type | Comparison Purpose |
| --- | --- | --- |
| `llama-3.1-8b-instant` | Lightweight model | Fast guardrail and compliance baseline |
| `llama-3.3-70b-versatile` | Larger reasoning model | Structured inquiry parsing and response generation baseline |
| `meta-llama/llama-4-scout-17b-16e-instruct` | Mid-size Llama alternative | Alternative model for guardrail and inquiry tasks |

### Comparison Results

| Agent | Model | Accuracy | P50 Latency | Avg Token Estimate | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| guardrailCheck | `llama-3.1-8b-instant` | 100.0% | 498.41ms | 217.5 | Selected |
| guardrailCheck | `llama-3.3-70b-versatile` | 100.0% | 461.03ms | 217.5 | Viable, but larger than needed |
| guardrailCheck | `meta-llama/llama-4-scout-17b-16e-instruct` | 100.0% | 447.45ms | 217.5 | Viable alternative |
| inquiryParser | `llama-3.1-8b-instant` | 75.0% | 559.87ms | 321.5 | Not selected |
| inquiryParser | `llama-3.3-70b-versatile` | 100.0% | 689.33ms | 329.5 | Selected |
| inquiryParser | `meta-llama/llama-4-scout-17b-16e-instruct` | 75.0% | 560.54ms | 320.5 | Not selected |
| safetyChecks | `llama-3.1-8b-instant` | 100.0% | 398.79ms | 267.8 | Selected |
| safetyChecks | `llama-3.3-70b-versatile` | 100.0% | 449.24ms | 268.2 | Viable, but no accuracy gain |

## Final Model Selection

| Agent | Final Model | Rationale |
| --- | --- | --- |
| guardrailCheck | `llama-3.1-8b-instant` | Achieved 100.0% accuracy on the representative guardrail comparison while remaining lightweight. Deterministic policy checks handle high-confidence PII and distress cases. |
| inquiryParser | `llama-3.3-70b-versatile` | Achieved 100.0% accuracy in the model comparison and outperformed the smaller and alternative models on structured inquiry parsing. |
| safetyChecks | `llama-3.1-8b-instant` | Achieved 100.0% compliance-check accuracy with lower latency than the larger model. |

## Evaluation Framework

| Script | Output | Purpose |
| --- | --- | --- |
| `evaluation/run_eval.py` | `evaluation/eval_report.json` | Runs the full 26-case end-to-end pipeline evaluation |
| `evaluation/model_comparison.py` | `evaluation/model_comparison_report.json` | Compares candidate Groq models by agent |

The evaluator measures:

- exact guardrail boolean match
- intent classification accuracy
- risk score accuracy
- compliance pass/fail correctness
- expected pipeline status
- P50, P95, and average latency
- category-level pass rates

## Failure Analysis and Improvements

| Observed Issue | Fix |
| --- | --- |
| Lightweight guardrail model occasionally misclassified adversarial loan prompts | Added deterministic policy checks inside the guardrail agent for high-confidence lending, PII, and distress signals |
| Self-harm or severe distress could be blocked as non-banking before escalation | Updated orchestration order so distress escalation is evaluated before PII and banking relevance blocking |
| Compliance pass rate was previously reported incorrectly when cases were skipped | Fixed evaluation logic so skipped compliance cases are excluded correctly |
| Model comparison initially included unavailable external-provider rows | Simplified comparison to reproducible Groq-hosted models only |

## Risk Mitigation

| Risk | Mitigation |
| --- | --- |
| Sensitive PII exposure | Blocked at Agent 1 before inquiry parsing |
| ECOA / fair lending violation | Prohibited in Agent 2 prompt and enforced again by Agent 3 |
| Guaranteed approval or misleading claims | Checked by compliance agent before customer response |
| Prompt injection | User text is treated as customer content only; generated output still passes final compliance review |
| Distress or self-harm | Escalated to human support before downstream inference |
| Model/API failure | Fail-safe behavior prevents unsafe customer-facing output |

## Conclusion

The evaluation supports the final architecture: use `llama-3.1-8b-instant` for fast guardrail and compliance checks, and reserve `llama-3.3-70b-versatile` for inquiry parsing, where structured extraction and reasoning quality matter most. This balances accuracy, latency, cost, and compliance risk while keeping the workflow practical for enterprise deployment.
