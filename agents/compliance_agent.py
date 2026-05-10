"""
Agent 3 – Compliance Agent (safetyChecks)
Uses Groq llama-3.1-8b-instant for fast, deterministic compliance enforcement.
Returns: { compliance_pass: bool, violations: list[str] }

Acts as final enforcement layer — zero-tolerance for ECOA/UDAAP violations.
"""

import asyncio
import json
import os
import httpx

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"   # Fast & deterministic for rule enforcement

SYSTEM_PROMPT = """You are a strict banking compliance enforcement agent.
Your job is to review an AI-generated loan response and check it against four compliance rules.
Return ONLY valid JSON — no explanations, no markdown.

COMPLIANCE RULES:
1. NO_GUARANTEE: The response must NOT contain guaranteed approval language.
   Violations include: "guaranteed approval", "you will definitely qualify", "you are approved",
   "you will get the loan", "we guarantee", or any language that implies certain approval.

2. NO_ECOA_VIOLATION: The response must NOT reference or reason from protected characteristics.
   Protected: age, race, gender, marital status, national origin, religion, disability.
   This applies even if the customer voluntarily mentioned these attributes.

3. NO_UDAAP: The response must NOT contain misleading, deceptive, manipulative, or
   unsubstantiated financial claims or promises.

4. HAS_DISCLAIMER: The response MUST contain the exact phrase:
   "All loan offers and estimates are subject to formal credit review and approval."

OUTPUT SCHEMA:
{
  "compliance_pass": <true if ALL rules pass, false if ANY rule fails>,
  "violations": [<list of violated rule names as strings, empty if none>]
}

Possible violation names: "NO_GUARANTEE", "NO_ECOA_VIOLATION", "NO_UDAAP", "HAS_DISCLAIMER"

Return ONLY the JSON. Nothing else."""


async def run_compliance_check(summary_response: str) -> dict:
    if not summary_response:
        return {
            "compliance_pass": False,
            "violations": ["HAS_DISCLAIMER", "EMPTY_RESPONSE"],
        }

    api_key = os.environ.get("GROQ_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 150,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Review this response:\n\n{summary_response}"},
        ],
    }
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
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fail-safe: treat as violation
        return {"compliance_pass": False, "violations": ["PARSE_ERROR"]}

    result.setdefault("compliance_pass", False)
    result.setdefault("violations", [])
    return result
