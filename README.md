# Banking AI - Loan Inquiry Orchestration

A multi-agent AI system for automating initial retail banking loan inquiries with compliance, safety, and privacy controls.

## Architecture

```text
Customer Message
  |
  v
Agent 1: guardrailCheck
  Model: llama-3.1-8b-instant + deterministic rules
  Checks: banking relevance, sensitive PII, distress/escalation
  |
  v
Agent 2: inquiryParser
  Model: llama-3.3-70b-versatile
  Tasks: intent classification, field extraction, risk estimate, customer response
  |
  v
Agent 3: safetyChecks
  Model: llama-3.1-8b-instant
  Checks: guarantee language, ECOA/fair lending, UDAAP, required disclaimer
  |
  v
Customer Response or Human Escalation
```

## Model Selection Rationale

| Agent | Model | Reason |
| --- | --- | --- |
| Guardrail | llama-3.1-8b-instant + deterministic rules | Fast boolean classification, with rules for high-confidence PII, lending, and distress signals |
| Inference | llama-3.3-70b-versatile | Better suited for structured extraction, risk scoring, and compliant response generation |
| Compliance | llama-3.1-8b-instant | Low-latency rule review against a short generated response |

The design uses the larger model only where reasoning and generation quality matter most. Guardrail and compliance checks stay lightweight to reduce latency and cost.

## Setup

### 1. Clone or unzip the repository

```bash
cd banking-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

Create a `.env` file in the project root. Do not commit this file; use `.env.example` as the shareable template.

```text
GROQ_API_KEY=gsk_your_key_here
```

## Running the System

### Single inquiry

```bash
python orchestrator.py
```

Edit the `test_message` at the bottom of [orchestrator.py](orchestrator.py) to try different inputs.

### Programmatic usage

```python
import asyncio
from orchestrator import handle_loan_inquiry

result = asyncio.run(handle_loan_inquiry(
    "I want a $200,000 mortgage. I work full-time as a teacher."
))
print(result)
```

## Running Evaluations

From the project root:

```bash
python evaluation/run_eval.py
```

Or from the evaluation directory:

```bash
cd evaluation
python run_eval.py
```

The runner executes all 26 test cases and writes the full report to `evaluation/eval_report.json`.

### Model comparison

To compare candidate models for each agent:

```bash
python evaluation/model_comparison.py
```

This writes agent-level tradeoff results to `evaluation/model_comparison_report.json`, including accuracy, latency, parse failures, and approximate token usage.

The comparison runner uses three Groq-hosted models so the table can be reproduced with one working key. It compares the selected Llama models against a Llama 4 Scout alternative.

By default, it uses a representative subset of evaluation cases so it finishes quickly. To run the full dataset comparison:

```powershell
$env:FULL_MODEL_COMPARISON="true"
python .\evaluation\model_comparison.py
```

Add your Groq key to `.env`:

```text
GROQ_API_KEY=gsk_...
```

## Evaluation Dataset

| Category | Count | Description |
| --- | ---: | --- |
| normal | 8 | Standard loan inquiries |
| ambiguous | 4 | Vague or non-banking messages |
| pii_injection | 4 | Sensitive PII and masked-account cases |
| adversarial | 5 | Jailbreaks, injections, overrides |
| compliance | 5 | ECOA edge cases, distress, self-harm escalation |

## Pipeline Status Codes

| Status | Meaning |
| --- | --- |
| `PASS` | All agents passed |
| `PASS_AFTER_RETRY` | Compliance failed once and passed on retry |
| `BLOCKED_NOT_BANKING` | Not a retail banking or lending inquiry |
| `BLOCKED_PII_DETECTED` | Sensitive PII found in the message |
| `ESCALATED_DISTRESS` | Customer distress detected and routed to human support |
| `BLOCKED_COMPLIANCE_VIOLATION` | Compliance failed after retry |
| `INFERENCE_ERROR` | Agent 2 parsing failure |

## Key Design Decisions

1. Fail-safe defaults: parse or API errors use restrictive outcomes.
2. Hybrid guardrails: deterministic checks override obvious model mistakes for PII, distress, and banking relevance.
3. Retry on compliance failure: one stricter retry before hard-blocking a response.
4. Zero PII forwarding: PII-containing messages stop at Agent 1.
5. Compliance as final gate: Agent 3 reviews every generated customer-facing response.

## Deliverables

- Architecture rationale: [docs/architecture_rationale.md](docs/architecture_rationale.md)
- Evaluation report: [docs/evaluation_report.md](docs/evaluation_report.md)
- Executive email: [docs/executive_email.md](docs/executive_email.md)
