"""
Banking AI – Loan Inquiry Orchestration
Main orchestrator: routes messages through all three agents sequentially.
"""

import asyncio
import json
import time
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from agents.guardrail_agent import run_guardrail_check
from agents.inquiry_agent import run_inquiry_parser
from agents.compliance_agent import run_compliance_check


async def handle_loan_inquiry(customer_message: str) -> dict[str, Any]:
    """
    Full pipeline: Guardrail → Inquiry Parser → Compliance Check.
    Returns the final structured result or an escalation/block signal.
    """
    start = time.perf_counter()
    result: dict[str, Any] = {"customer_message": customer_message}

    # ── Agent 1: Guardrail Check ─────────────────────────────────────────────
    guardrail = await run_guardrail_check(customer_message)
    result["guardrail"] = guardrail

    if not guardrail.get("sentiment_check", True):
        result["pipeline_status"] = "ESCALATED_DISTRESS"
        result["customer_facing_message"] = (
            "I can see you may be going through a difficult situation. "
            "Let me connect you with a human advisor who can provide immediate assistance."
        )
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result

    if not guardrail.get("no_pii", True):
        result["pipeline_status"] = "BLOCKED_PII_DETECTED"
        result["customer_facing_message"] = (
            "For your security, please do not share sensitive personal information "
            "(account numbers, SSNs, card numbers) in this chat. "
            "A representative will assist you securely."
        )
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result

    if not guardrail.get("is_banking_related", False):
        result["pipeline_status"] = "BLOCKED_NOT_BANKING"
        result["customer_facing_message"] = (
            "I'm sorry, I can only assist with retail banking and loan inquiries. "
            "Please contact us for other concerns."
        )
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result

    # ── Agent 2: Inquiry Parser ──────────────────────────────────────────────
    inquiry = await run_inquiry_parser(customer_message)
    result["inquiry"] = inquiry

    if "error" in inquiry:
        result["pipeline_status"] = "INFERENCE_ERROR"
        result["customer_facing_message"] = (
            "We encountered an issue processing your request. "
            "Please try again or speak with an advisor."
        )
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result

    # ── Agent 3: Compliance Check ────────────────────────────────────────────
    compliance = await run_compliance_check(inquiry.get("summary_response", ""))
    result["compliance"] = compliance

    if not compliance.get("compliance_pass", False):
        # Retry once with a stricter prompt signal (fallback)
        retry_inquiry = await run_inquiry_parser(
            customer_message, force_compliant=True
        )
        retry_compliance = await run_compliance_check(
            retry_inquiry.get("summary_response", "")
        )
        if retry_compliance.get("compliance_pass", False):
            result["inquiry"] = retry_inquiry
            result["compliance"] = retry_compliance
            result["pipeline_status"] = "PASS_AFTER_RETRY"
        else:
            result["pipeline_status"] = "BLOCKED_COMPLIANCE_VIOLATION"
            result["customer_facing_message"] = (
                "Thank you for your inquiry. A loan specialist will review your "
                "request and contact you with compliant, personalised information."
            )
            result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            return result
    else:
        result["pipeline_status"] = "PASS"

    result["customer_facing_message"] = result["inquiry"].get("summary_response", "")
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return result


if __name__ == "__main__":
    test_message = (
        "Hi, I'm interested in a $250,000 mortgage. "
        "I work full-time as a software engineer. Can you help?"
    )
    output = asyncio.run(handle_loan_inquiry(test_message))
    print(json.dumps(output, indent=2))
