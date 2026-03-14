"""Dataset-driven evaluation runner for tool usage performance."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import time
from typing import Any, Dict, List, Tuple

from agent import ORCHESTRATOR


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    rows: List[Dict[str, Any]] = []
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif ext == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
    else:
        raise ValueError("Unsupported dataset format. Use .jsonl or .csv")
    return rows


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


async def evaluate_dataset(path: str) -> Dict[str, Any]:
    rows = _load_dataset(path)
    if not rows:
        return {
            "total": 0,
            "tool_usage_accuracy": 0.0,
            "tool_selection_correctness": 0.0,
            "must_use_precision": 0.0,
            "must_use_recall": 0.0,
            "must_use_f1": 0.0,
            "success_rate": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            "details": [],
        }

    usage_correct = 0
    selection_correct = 0
    success_count = 0
    tp = fp = tn = fn = 0
    latencies_ms: List[float] = []
    details: List[Dict[str, Any]] = []

    for idx, item in enumerate(rows):
        query = str(item.get("query", "")).strip()
        expected_tool = str(item.get("expected_tool", "none")).strip() or "none"
        must_use_tool = _to_bool(item.get("must_use_tool", False))

        start = time.perf_counter()
        state = await ORCHESTRATOR.run(
            user_message=query,
            history=[],
            session_id=f"eval-{idx}",
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(latency_ms)

        tool_decision = state.get("tool_decision") or {}
        selected_tool = tool_decision.get("tool_id") or "none"
        tool_result = state.get("tool_result") or {}
        used_tool = bool(selected_tool != "none")
        validation_passed = bool(state.get("validation_passed", False))
        fallback_used = bool((state.get("fallback_report") or {}).get("used", False))

        expected_used = must_use_tool or (expected_tool != "none")
        if used_tool == expected_used:
            usage_correct += 1
        if selected_tool == expected_tool:
            selection_correct += 1
        if validation_passed and not fallback_used:
            success_count += 1

        if expected_used and used_tool:
            tp += 1
        elif not expected_used and used_tool:
            fp += 1
        elif expected_used and not used_tool:
            fn += 1
        else:
            tn += 1

        details.append(
            {
                "query": query,
                "expected_tool": expected_tool,
                "predicted_tool": selected_tool,
                "must_use_tool": must_use_tool,
                "tool_execution_success": bool(tool_result.get("success", False)),
                "validation_passed": validation_passed,
                "fallback_used": fallback_used,
                "latency_ms": round(latency_ms, 2),
            }
        )

    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
    p50 = statistics.median(latencies_ms)
    p95_index = max(0, int(len(latencies_ms) * 0.95) - 1)
    p95 = sorted(latencies_ms)[p95_index]

    total = len(rows)
    return {
        "total": total,
        "tool_usage_accuracy": round(usage_correct / total, 4),
        "tool_selection_correctness": round(selection_correct / total, 4),
        "must_use_precision": round(precision, 4),
        "must_use_recall": round(recall, 4),
        "must_use_f1": round(f1, 4),
        "success_rate": round(success_count / total, 4),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tool-usage evaluation dataset.")
    parser.add_argument("--dataset", required=True, help="Path to .jsonl or .csv file")
    args = parser.parse_args()

    result = asyncio.run(evaluate_dataset(args.dataset))
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
