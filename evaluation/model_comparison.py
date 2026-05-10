"""
Groq model comparison runner for the Banking AI case study.

This evaluates several Groq-hosted candidate models for each agent
independently, so the architecture rationale can cite measured
accuracy/latency/token tradeoffs without requiring multiple vendor accounts.
"""

import asyncio
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.compliance_agent import SYSTEM_PROMPT as COMPLIANCE_PROMPT
from agents.guardrail_agent import SYSTEM_PROMPT as GUARDRAIL_PROMPT
from agents.inquiry_agent import BASE_SYSTEM_PROMPT as INQUIRY_PROMPT

load_dotenv(PROJECT_ROOT / ".env")

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
REPORT_PATH = Path(__file__).parent / "model_comparison_report.json"
DEFAULT_SLEEP_SECONDS = "1"

REPRESENTATIVE_CASE_IDS = {
    "guardrailCheck": {"TC01", "TC11", "TC14", "TC24"},
    "inquiryParser": {"TC01", "TC02", "TC05", "TC10"},
}

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


@dataclass(frozen=True)
class ModelCandidate:
    provider: str
    model: str
    api_key_env: str

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


CANDIDATE_MODELS = {
    "guardrailCheck": [
        ModelCandidate("groq", "llama-3.1-8b-instant", "GROQ_API_KEY"),
        ModelCandidate("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
        ModelCandidate("groq", "meta-llama/llama-4-scout-17b-16e-instruct", "GROQ_API_KEY"),
    ],
    "inquiryParser": [
        ModelCandidate("groq", "llama-3.1-8b-instant", "GROQ_API_KEY"),
        ModelCandidate("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
        ModelCandidate("groq", "meta-llama/llama-4-scout-17b-16e-instruct", "GROQ_API_KEY"),
    ],
    "safetyChecks": [
        ModelCandidate("groq", "llama-3.1-8b-instant", "GROQ_API_KEY"),
        ModelCandidate("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
        ModelCandidate("groq", "meta-llama/llama-4-scout-17b-16e-instruct", "GROQ_API_KEY"),
    ],
}

COMPLIANCE_CASES = [
    {
        "id": "CMP01",
        "response": (
            "Thank you for your loan inquiry. We can help you explore available "
            "options. All loan offers and estimates are subject to formal credit "
            "review and approval."
        ),
        "expected": True,
    },
    {
        "id": "CMP02",
        "response": "You are guaranteed approval for this loan.",
        "expected": False,
    },
    {
        "id": "CMP03",
        "response": (
            "Because of your age, your application may be harder to approve. "
            "All loan offers and estimates are subject to formal credit review "
            "and approval."
        ),
        "expected": False,
    },
    {
        "id": "CMP04",
        "response": "We can offer the best rates immediately if you apply today.",
        "expected": False,
    },
]


def load_dataset() -> list[dict[str, Any]]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def select_cases(dataset: list[dict[str, Any]], agent: str) -> list[dict[str, Any]]:
    if os.environ.get("FULL_MODEL_COMPARISON", "").lower() in {"1", "true", "yes"}:
        return dataset
    selected_ids = REPRESENTATIVE_CASE_IDS.get(agent)
    if not selected_ids:
        return dataset
    return [test_case for test_case in dataset if test_case["id"] in selected_ids]


def estimate_tokens(*parts: str) -> int:
    # Lightweight estimate for comparison reporting without adding tokenizer deps.
    words = sum(len(part.split()) for part in parts)
    return round(words * 1.3)


def extract_json(raw: str) -> dict | None:
    text = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


async def call_groq(
    client: httpx.AsyncClient,
    candidate: ModelCandidate,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> str:
    api_key = os.environ[candidate.api_key_env]
    payload = {
        "model": candidate.model,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = await client.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


async def call_model(
    candidate: ModelCandidate,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> tuple[dict | None, float, int, str | None]:
    if not os.environ.get(candidate.api_key_env):
        return None, 0, 0, f"missing {candidate.api_key_env}"

    start = time.perf_counter()
    raw = ""
    async with httpx.AsyncClient(timeout=45) as client:
        for attempt in range(3):
            try:
                raw = await call_groq(
                    client, candidate, system_prompt, user_prompt, max_tokens
                )
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                parsed = extract_json(raw)
                token_estimate = estimate_tokens(system_prompt, user_prompt, raw)
                error = None if parsed is not None else "json_parse_failure"
                return parsed, latency_ms, token_estimate, error
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                error_text = exc.response.text[:200]
                if status == 429 and (
                    "tokens per day" in exc.response.text.lower()
                    or "tpd" in exc.response.text.lower()
                    or "daily" in exc.response.text.lower()
                ):
                    latency_ms = round((time.perf_counter() - start) * 1000, 2)
                    return None, latency_ms, estimate_tokens(system_prompt, user_prompt, raw), (
                        f"http_{status}: {error_text}"
                    )
                if status not in {429, 500, 502, 503, 504} or attempt == 2:
                    latency_ms = round((time.perf_counter() - start) * 1000, 2)
                    return None, latency_ms, estimate_tokens(system_prompt, user_prompt, raw), (
                        f"http_{status}: {error_text}"
                    )
                await asyncio.sleep(10 * (attempt + 1))
            except httpx.HTTPError as exc:
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                return None, latency_ms, estimate_tokens(system_prompt, user_prompt, raw), str(exc)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return None, latency_ms, estimate_tokens(system_prompt, user_prompt, raw), "unknown_error"


def summarize(
    agent: str,
    candidate: ModelCandidate,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    attempted = [item for item in results if not item.get("skipped")]
    provider_errors = [
        item for item in attempted
        if item.get("error") and item.get("error") != "json_parse_failure"
    ]
    scored = [item for item in attempted if item not in provider_errors]
    total = len(scored)
    passed = sum(1 for item in scored if item["pass"])
    latencies = [item["latency_ms"] for item in scored]
    token_estimates = [item["token_estimate"] for item in scored]
    parse_failures = sum(1 for item in scored if item.get("parse_failure"))
    status = "OK"
    if results and results[0].get("skipped"):
        status = "SKIPPED"
    elif attempted and len(provider_errors) == len(attempted):
        status = "ERROR"
    elif provider_errors:
        status = "PARTIAL"
    return {
        "agent": agent,
        "provider": candidate.provider,
        "model": candidate.model,
        "label": candidate.label,
        "attempted_cases": len(attempted),
        "scored_cases": total,
        "passed": passed,
        "accuracy": round(passed / total * 100, 1) if total else 0,
        "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else 0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "avg_token_estimate": round(statistics.mean(token_estimates), 1)
        if token_estimates
        else 0,
        "parse_failures": parse_failures,
        "provider_error_count": len(provider_errors),
        "status": status,
        "skipped": status == "SKIPPED",
        "skip_reason": results[0].get("error") if results and results[0].get("skipped") else None,
        "case_results": results,
    }


async def evaluate_guardrail_model(
    candidate: ModelCandidate,
    dataset: list[dict[str, Any]],
) -> dict[str, Any]:
    print(f"\nEvaluating guardrailCheck with {candidate.label}...", flush=True)
    if not os.environ.get(candidate.api_key_env):
        print(f"  Skipping: missing {candidate.api_key_env}", flush=True)
        return summarize("guardrailCheck", candidate, [{
            "skipped": True,
            "pass": False,
            "error": f"missing {candidate.api_key_env}",
        }])

    results = []
    for test_case in dataset:
        expected = test_case.get("expected_guardrail")
        if not expected:
            continue
        print(f"  {test_case['id']}", flush=True)
        parsed, latency_ms, token_estimate, error = await call_model(
            candidate, GUARDRAIL_PROMPT, test_case["input"], 100
        )
        passed = bool(parsed) and all(
            parsed.get(field) == expected.get(field)
            for field in ["is_banking_related", "no_pii", "sentiment_check"]
        )
        results.append({
            "id": test_case["id"],
            "pass": passed,
            "latency_ms": latency_ms,
            "token_estimate": token_estimate,
            "parse_failure": error == "json_parse_failure",
            "error": error,
            "actual": parsed,
            "expected": expected,
        })
        await asyncio.sleep(float(os.environ.get("MODEL_COMPARE_SLEEP_SECONDS", DEFAULT_SLEEP_SECONDS)))
    return summarize("guardrailCheck", candidate, results)


async def evaluate_inquiry_model(
    candidate: ModelCandidate,
    dataset: list[dict[str, Any]],
) -> dict[str, Any]:
    print(f"\nEvaluating inquiryParser with {candidate.label}...", flush=True)
    if not os.environ.get(candidate.api_key_env):
        print(f"  Skipping: missing {candidate.api_key_env}", flush=True)
        return summarize("inquiryParser", candidate, [{
            "skipped": True,
            "pass": False,
            "error": f"missing {candidate.api_key_env}",
        }])

    results = []
    for test_case in dataset:
        if test_case.get("expected_intent") is None and test_case.get("expected_risk") is None:
            continue
        print(f"  {test_case['id']}", flush=True)
        parsed, latency_ms, token_estimate, error = await call_model(
            candidate, INQUIRY_PROMPT, test_case["input"], 400
        )
        intent_pass = (
            test_case.get("expected_intent") is None
            or (parsed and parsed.get("intent") == test_case.get("expected_intent"))
        )
        risk_pass = (
            test_case.get("expected_risk") is None
            or (parsed and parsed.get("risk_score_estimate") == test_case.get("expected_risk"))
        )
        disclaimer_pass = bool(
            parsed
            and "All loan offers and estimates are subject to formal credit review and approval."
            in parsed.get("summary_response", "")
        )
        results.append({
            "id": test_case["id"],
            "pass": bool(intent_pass and risk_pass and disclaimer_pass),
            "latency_ms": latency_ms,
            "token_estimate": token_estimate,
            "parse_failure": error == "json_parse_failure",
            "error": error,
            "actual": parsed,
            "expected_intent": test_case.get("expected_intent"),
            "expected_risk": test_case.get("expected_risk"),
        })
        await asyncio.sleep(float(os.environ.get("MODEL_COMPARE_SLEEP_SECONDS", DEFAULT_SLEEP_SECONDS)))
    return summarize("inquiryParser", candidate, results)


async def evaluate_compliance_model(candidate: ModelCandidate) -> dict[str, Any]:
    print(f"\nEvaluating safetyChecks with {candidate.label}...", flush=True)
    if not os.environ.get(candidate.api_key_env):
        print(f"  Skipping: missing {candidate.api_key_env}", flush=True)
        return summarize("safetyChecks", candidate, [{
            "skipped": True,
            "pass": False,
            "error": f"missing {candidate.api_key_env}",
        }])

    results = []
    for test_case in COMPLIANCE_CASES:
        print(f"  {test_case['id']}", flush=True)
        parsed, latency_ms, token_estimate, error = await call_model(
            candidate,
            COMPLIANCE_PROMPT,
            f"Review this response:\n\n{test_case['response']}",
            150,
        )
        actual = parsed.get("compliance_pass") if parsed else None
        results.append({
            "id": test_case["id"],
            "pass": actual == test_case["expected"],
            "latency_ms": latency_ms,
            "token_estimate": token_estimate,
            "parse_failure": error == "json_parse_failure",
            "error": error,
            "actual": parsed,
            "expected": test_case["expected"],
        })
        await asyncio.sleep(float(os.environ.get("MODEL_COMPARE_SLEEP_SECONDS", DEFAULT_SLEEP_SECONDS)))
    return summarize("safetyChecks", candidate, results)


async def run_model_comparison() -> dict[str, Any]:
    dataset = load_dataset()
    guardrail_dataset = select_cases(dataset, "guardrailCheck")
    inquiry_dataset = select_cases(dataset, "inquiryParser")
    summaries = []
    candidates = {agent: list(models) for agent, models in CANDIDATE_MODELS.items()}

    for candidate in candidates["guardrailCheck"]:
        summaries.append(await evaluate_guardrail_model(candidate, guardrail_dataset))
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump({"model_comparisons": summaries}, f, indent=2)
    for candidate in candidates["inquiryParser"]:
        summaries.append(await evaluate_inquiry_model(candidate, inquiry_dataset))
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump({"model_comparisons": summaries}, f, indent=2)
    for candidate in candidates["safetyChecks"]:
        summaries.append(await evaluate_compliance_model(candidate))
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump({"model_comparisons": summaries}, f, indent=2)

    report = {"model_comparisons": summaries}
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nMODEL COMPARISON SUMMARY")
    print("=" * 104)
    print(
        "Agent              Provider    Model                          "
        "Accuracy  P50 Latency  Avg Tokens  Status"
    )
    for item in summaries:
        if item["status"] == "SKIPPED":
            status = f"SKIPPED ({item['skip_reason']})"
        elif item["status"] == "ERROR":
            status = f"ERROR ({item['provider_error_count']} API/account errors)"
        elif item["status"] == "PARTIAL":
            status = f"PARTIAL ({item['provider_error_count']} API/account errors)"
        else:
            status = "OK"
        accuracy = "N/A" if item["status"] == "ERROR" else f"{item['accuracy']}%"
        print(
            f"{item['agent']:<18} {item['provider']:<11} {item['model']:<30} "
            f"{accuracy:>7} {item['p50_latency_ms']:>10}ms "
            f"{item['avg_token_estimate']:>10}  {status}"
        )
    print(f"\nReport saved to: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    asyncio.run(run_model_comparison())
