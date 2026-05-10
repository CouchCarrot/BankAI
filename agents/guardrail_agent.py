"""
Agent 1 – Intake Guardrails (guardrailCheck)
Uses Groq llama-3.1-8b-instant for low-latency, free boolean classification.
Returns: { is_banking_related, no_pii, sentiment_check }
"""

import asyncio
import json
import os
import re
import httpx

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"   # Fast, free – ideal for guardrails

SYSTEM_PROMPT = """You are a strict intake guardrail for a retail banking AI system.
Your ONLY job is to inspect the customer's message and return a JSON object with exactly three boolean fields.

DEFINITIONS:
- is_banking_related: true if the message relates to retail banking, loans, mortgages, credit, refinance, or financial products. false otherwise.
- no_pii: true if the message contains NO sensitive PII. false if it contains any of: full SSN, bank account numbers, credit/debit card numbers, CVV codes, passwords, routing numbers, Aadhaar/PAN identifiers, or unmasked government IDs. Partially masked values like ****1234 are acceptable (no_pii stays true).
- sentiment_check: true if the customer tone is normal/calm. false ONLY if the message contains threats, self-harm indicators, severe panic/distress, harassment, or urgent financial crisis requiring immediate human intervention.

OUTPUT: Return ONLY valid JSON. No explanations. No markdown. No extra fields.

Example:
{"is_banking_related": true, "no_pii": true, "sentiment_check": true}"""


def _rule_based_guardrail(customer_message: str) -> dict:
    """Deterministic checks for high-confidence policy signals."""
    text = customer_message.lower()

    banking_terms = [
        "loan", "mortgage", "auto", "vehicle", "car", "personal loan",
        "refinance", "credit", "interest rate", "rates", "borrow",
        "finance", "financing", "debt", "bill", "bills", "house",
        "lending", "underwriting", "approve", "approval", "guaranteed",
    ]
    non_lending_terms = ["stocks", "stock market", "cover letter"]

    pii_patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b(?:bank account|account number|account is)\D{0,20}\d{8,17}\b",
        r"\b(?:\d[ -]*?){13,19}\b",  # card-like/account-like numbers
        r"\b(?:cvv|cvc|security code|password|routing number|aadhaar|pan)\b",
    ]

    distress_terms = [
        "can't take it anymore", "what i'll do to myself", "hurt myself",
        "kill myself", "suicide", "self-harm", "desperate",
        "lose my house", "need help now", "can't pay my bills",
    ]

    is_banking_related = any(term in text for term in banking_terms)
    if any(term in text for term in non_lending_terms) and not is_banking_related:
        is_banking_related = False

    no_pii = not any(re.search(pattern, text) for pattern in pii_patterns)
    sentiment_check = not any(term in text for term in distress_terms)
    if not sentiment_check and any(term in text for term in ["bill", "debt", "house", "loan"]):
        is_banking_related = True

    return {
        "is_banking_related": is_banking_related,
        "no_pii": no_pii,
        "sentiment_check": sentiment_check,
    }


async def run_guardrail_check(customer_message: str) -> dict:
    rules = _rule_based_guardrail(customer_message)
    text = customer_message.lower()
    api_key = os.environ.get("GROQ_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 100,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": customer_message},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            for attempt in range(3):
                try:
                    resp = await client.post(GROQ_API_URL, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 429 or attempt == 2:
                        raise
                    await asyncio.sleep(10 * (attempt + 1))

        raw = data["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
    except (httpx.HTTPError, KeyError, json.JSONDecodeError):
        result = rules

    result.setdefault("is_banking_related", False)
    result.setdefault("no_pii", True)
    result.setdefault("sentiment_check", True)

    # High-confidence deterministic checks override model variability.
    if rules["is_banking_related"]:
        result["is_banking_related"] = True
    if any(term in text for term in ["stocks", "stock market", "cover letter"]):
        result["is_banking_related"] = False
    if not rules["no_pii"]:
        result["no_pii"] = False
    result["sentiment_check"] = rules["sentiment_check"]

    return result
