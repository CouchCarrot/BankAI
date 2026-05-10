"""
Agent 2 – Inquiry Inference Agent (inquiryParser)
Uses Groq llama-3.3-70b-versatile for accurate intent extraction and structured response generation.
Returns strict JSON matching the required output schema.
"""

import asyncio
import json
import os
import re
import httpx

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"   # Larger model – better reasoning for inference

BASE_SYSTEM_PROMPT = """You are a structured loan inquiry parser for a retail bank.
Extract information from the customer message and return ONLY valid JSON — no preamble, no markdown.

REQUIRED OUTPUT SCHEMA:
{
  "intent": "<Mortgage|Auto|Personal|Refinance>",
  "loan_amount_requested": <integer or null>,
  "employment_status": "<Full-time|Part-time|Self-employed|Unemployed|Unknown>",
  "risk_score_estimate": "<Low|Medium|High>",
  "summary_response": "<string>"
}

RISK SCORING RULES (apply strictly):
- Low: Stable full-time employment AND moderate loan amount relative to stated income.
- Medium: Part-time employment, limited employment history, self-employed, or incomplete financial info.
- High: Unemployed, inconsistent income, excessive loan amount, or clear repayment concerns.
- If multiple loan purposes are mentioned and no single product is clearly primary, classify intent as Personal.

SUMMARY RESPONSE RULES (mandatory — all must be satisfied):
1. Neutral and professional tone only.
2. NEVER guarantee approval or suggest the customer will qualify.
3. NEVER make speculative claims about rates or outcomes.
4. NEVER mention or reason from protected characteristics (age, race, gender, marital status, national origin, religion, disability).
5. ALWAYS include exactly this disclaimer: "All loan offers and estimates are subject to formal credit review and approval."
6. Keep response concise (2-4 sentences).

Return ONLY the JSON object. Nothing else."""

FORCE_COMPLIANT_ADDENDUM = """
ADDITIONAL COMPLIANCE ENFORCEMENT:
You MUST be extra cautious. Double-check the summary_response contains the required disclaimer word-for-word.
Remove any language that could be construed as a guarantee, promise, or protected-class reference.
"""


def _fallback_inquiry(customer_message: str) -> dict:
    """Safe structured fallback when the model returns malformed JSON."""
    text = customer_message.lower()
    amount_match = re.search(r"\$?\s*([\d,]+)", customer_message)
    amount = int(amount_match.group(1).replace(",", "")) if amount_match else None

    has_auto = any(term in text for term in ["auto", "car", "vehicle"])
    has_home = any(term in text for term in ["home", "house", "mortgage"])
    if ("refinance" in text or "refi" in text) and not has_auto:
        intent = "Refinance"
    elif has_auto and has_home:
        intent = "Personal"
    elif "mortgage" in text or "buy a home" in text:
        intent = "Mortgage"
    elif has_auto:
        intent = "Auto"
    else:
        intent = "Personal"

    if "full-time" in text or "full time" in text:
        employment = "Full-time"
    elif "part-time" in text or "part time" in text:
        employment = "Part-time"
    elif "self-employed" in text or "own business" in text or "freelance" in text:
        employment = "Self-employed"
    elif "unemployed" in text:
        employment = "Unemployed"
    else:
        employment = "Unknown"

    if employment == "Unemployed" or any(term in text for term in ["no income", "can't repay"]):
        risk = "High"
    elif employment == "Full-time" and amount is not None and amount <= 250000:
        risk = "Low"
    else:
        risk = "Medium"

    return {
        "intent": intent,
        "loan_amount_requested": amount,
        "employment_status": employment,
        "risk_score_estimate": risk,
        "summary_response": (
            "Thank you for your loan inquiry. We can help you explore available "
            "options based on the information you provide. All loan offers and "
            "estimates are subject to formal credit review and approval."
        ),
    }


async def run_inquiry_parser(
    customer_message: str, force_compliant: bool = False
) -> dict:
    api_key = os.environ.get("GROQ_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    system = BASE_SYSTEM_PROMPT
    if force_compliant:
        system += FORCE_COMPLIANT_ADDENDUM

    payload = {
        "model": MODEL,
        "max_tokens": 400,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": customer_message},
        ],
    }
    async with httpx.AsyncClient(timeout=20) as client:
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
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = _fallback_inquiry(customer_message)

    # Normalise intent casing
    valid_intents = {"Mortgage", "Auto", "Personal", "Refinance"}
    if result.get("intent") not in valid_intents:
        result["intent"] = "Personal"  # safe default
    text = customer_message.lower()
    if any(term in text for term in ["auto", "car", "vehicle"]) and any(
        term in text for term in ["home", "house", "improvement"]
    ):
        result["intent"] = "Personal"

    # Normalise risk casing
    valid_risks = {"Low", "Medium", "High"}
    if result.get("risk_score_estimate") not in valid_risks:
        result["risk_score_estimate"] = "Medium"

    return result
