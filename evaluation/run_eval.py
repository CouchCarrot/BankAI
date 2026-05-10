"""
Evaluation Runner – Banking AI Loan Inquiry Orchestration
Runs all test cases and produces a structured evaluation report.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator import handle_loan_inquiry

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
REPORT_PATH = Path(__file__).parent / "eval_report.json"


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def evaluate_guardrail(result: dict, expected: dict) -> dict:
    """Check exact boolean match for guardrail fields."""
    actual = result.get("guardrail", {})
    checks = {}
    for field in ["is_banking_related", "no_pii", "sentiment_check"]:
        exp_val = expected.get(field)
        act_val = actual.get(field)
        checks[field] = {
            "expected": exp_val,
            "actual": act_val,
            "pass": exp_val == act_val if exp_val is not None else True,
        }
    overall = all(c["pass"] for c in checks.values())
    return {"pass": overall, "checks": checks}


def evaluate_intent(result: dict, expected_intent: str | None) -> dict:
    if expected_intent is None:
        return {"pass": True, "skipped": True}
    actual = result.get("inquiry", {}).get("intent")
    return {"expected": expected_intent, "actual": actual, "pass": actual == expected_intent}


def evaluate_risk(result: dict, expected_risk: str | None) -> dict:
    if expected_risk is None:
        return {"pass": True, "skipped": True}
    actual = result.get("inquiry", {}).get("risk_score_estimate")
    return {"expected": expected_risk, "actual": actual, "pass": actual == expected_risk}


def evaluate_compliance(result: dict, expected_pass: bool | None) -> dict:
    if expected_pass is None:
        return {"pass": True, "skipped": True}
    actual = result.get("compliance", {}).get("compliance_pass")
    violations = result.get("compliance", {}).get("violations", [])
    return {
        "expected": expected_pass,
        "actual": actual,
        "violations": violations,
        "pass": actual == expected_pass,
    }


def evaluate_pipeline_status(result: dict, expected_status: str | None) -> dict:
    if expected_status is None:
        return {"pass": True, "skipped": True}
    actual = result.get("pipeline_status")
    return {"expected": expected_status, "actual": actual, "pass": actual == expected_status}


async def run_evaluation():
    dataset = load_dataset()
    results = []
    category_metrics = {}

    print(f"\n{'='*60}")
    print("Banking AI – Evaluation Runner")
    print(f"{'='*60}")
    print(f"Running {len(dataset)} test cases...\n")

    for tc in dataset:
        tc_id = tc["id"]
        category = tc["category"]
        description = tc["description"]

        print(f"[{tc_id}] {description}")

        start = time.perf_counter()
        try:
            pipeline_result = await handle_loan_inquiry(tc["input"])
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": tc_id,
                "category": category,
                "error": str(e),
                "pass": False,
            })
            continue

        await asyncio.sleep(float(os.environ.get("EVAL_SLEEP_SECONDS", "6")))

        # Run evaluations
        guardrail_eval = evaluate_guardrail(
            pipeline_result, tc.get("expected_guardrail", {})
        )
        intent_eval = evaluate_intent(pipeline_result, tc.get("expected_intent"))
        risk_eval = evaluate_risk(pipeline_result, tc.get("expected_risk"))
        compliance_eval = evaluate_compliance(
            pipeline_result, tc.get("expected_compliance_pass")
        )
        status_eval = evaluate_pipeline_status(
            pipeline_result, tc.get("expected_pipeline_status")
        )

        # Overall pass = all checks pass
        overall_pass = all([
            guardrail_eval["pass"],
            intent_eval["pass"],
            risk_eval["pass"],
            compliance_eval["pass"],
            status_eval["pass"],
        ])

        status_icon = "PASS" if overall_pass else "FAIL"
        print(f"  {status_icon} Latency: {latency_ms}ms | Status: {pipeline_result.get('pipeline_status')}")

        tc_result = {
            "id": tc_id,
            "category": category,
            "description": description,
            "pass": overall_pass,
            "latency_ms": latency_ms,
            "pipeline_status": pipeline_result.get("pipeline_status"),
            "evaluations": {
                "guardrail": guardrail_eval,
                "intent": intent_eval,
                "risk": risk_eval,
                "compliance": compliance_eval,
                "pipeline_status": status_eval,
            },
            "pipeline_result": pipeline_result,
        }
        results.append(tc_result)

        # Track category metrics
        if category not in category_metrics:
            category_metrics[category] = {"total": 0, "passed": 0, "latencies": []}
        category_metrics[category]["total"] += 1
        if overall_pass:
            category_metrics[category]["passed"] += 1
        category_metrics[category]["latencies"].append(latency_ms)

    # Compute summary metrics
    total = len(results)
    passed = sum(1 for r in results if r.get("pass", False))
    all_latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    sorted_latencies = sorted(all_latencies)
    p50 = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0
    p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0

    # Compliance-specific pass rate
    compliance_cases = [
        r for r in results
        if not r.get("evaluations", {}).get("compliance", {}).get("skipped", False)
    ]
    compliance_passed = sum(
        1 for r in compliance_cases
        if r.get("evaluations", {}).get("compliance", {}).get("pass", False)
    )

    # Guardrail accuracy
    guardrail_cases = [r for r in results if "evaluations" in r]
    guardrail_passed = sum(
        1 for r in guardrail_cases
        if r.get("evaluations", {}).get("guardrail", {}).get("pass", False)
    )

    summary = {
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "overall_accuracy": round(passed / total * 100, 1) if total else 0,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "avg_latency_ms": round(sum(all_latencies) / len(all_latencies), 1) if all_latencies else 0,
        "guardrail_accuracy": round(guardrail_passed / len(guardrail_cases) * 100, 1) if guardrail_cases else 0,
        "compliance_pass_rate": round(compliance_passed / len(compliance_cases) * 100, 1) if compliance_cases else 0,
        "category_breakdown": {
            cat: {
                "pass_rate": round(m["passed"] / m["total"] * 100, 1),
                "p50_latency_ms": sorted(m["latencies"])[len(m["latencies"]) // 2],
            }
            for cat, m in category_metrics.items()
        },
    }

    report = {"summary": summary, "test_cases": results}

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Cases:          {total}")
    print(f"Passed:               {passed} ({summary['overall_accuracy']}%)")
    print(f"Failed:               {total - passed}")
    print(f"P50 Latency:          {p50}ms")
    print(f"P95 Latency:          {p95}ms")
    print(f"Avg Latency:          {summary['avg_latency_ms']}ms")
    print(f"Guardrail Accuracy:   {summary['guardrail_accuracy']}%")
    print(f"Compliance Pass Rate: {summary['compliance_pass_rate']}%")
    print(f"\nReport saved to: {REPORT_PATH}")

    return report


if __name__ == "__main__":
    asyncio.run(run_evaluation())
