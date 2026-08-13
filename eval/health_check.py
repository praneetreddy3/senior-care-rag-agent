"""Lightweight regression check for automation tools (e.g. n8n, cron, CI).

Runs a small random sample of the eval set (not all 54 questions — that's too
slow/expensive to run on a schedule against a rate-limited free API) and
reports whether quality has regressed below the thresholds in `src/config.py`.
This is the same idea as `eval/run_eval.py`, just fast enough to call from an
HTTP endpoint on a timer.

Usage:
    python -m eval.health_check           # human-readable
    python -m eval.health_check --json     # machine-readable (what the API uses)
"""
from __future__ import annotations

import argparse
import json
import random
from statistics import mean

from eval.run_eval import faithfulness, keyword_coverage, load_eval
from src.agent import Agent
from src.config import settings


def run_health_check(agent: Agent | None = None, sample_size: int = 8, seed: int = 0) -> dict:
    """Run a fast sample of the eval set and check it against quality thresholds.

    Returns a dict with the sampled metrics, which thresholds were breached (if
    any), and an overall `pass` boolean — the shape an automation tool checks.
    """
    agent = agent or Agent()
    items = load_eval()
    sample = items if len(items) <= sample_size else random.Random(seed).sample(items, sample_size)

    hits, grounded, coverage, faith = [], [], [], []
    for item in sample:
        results = agent.retriever.retrieve(item["question"], top_k=4)
        sources = [c.source for c, _ in results]
        hits.append(1.0 if item["expected_source"] in sources else 0.0)

        state = agent.run(item["question"])
        answer = state.get("answer", "")
        grounded.append(1.0 if state.get("grounded") else 0.0)
        coverage.append(keyword_coverage(answer, item.get("answer_keywords", [])))
        context = "\n\n".join(d["text"] for d in state.get("relevant", []))
        f = faithfulness(item["question"], answer, context)
        if f is not None:
            faith.append(f)

    metrics = {
        "sample_size": len(sample),
        "hit_rate": round(mean(hits), 3) if hits else 0.0,
        "grounded_rate": round(mean(grounded), 3) if grounded else 0.0,
        "keyword_coverage": round(mean(coverage), 3) if coverage else 0.0,
        "faithfulness": round(mean(faith), 3) if faith else None,
    }

    failures = []
    if metrics["hit_rate"] < settings.eval_hit_rate_threshold:
        failures.append(
            f"hit_rate {metrics['hit_rate']} < threshold {settings.eval_hit_rate_threshold}"
        )
    if metrics["grounded_rate"] < settings.eval_grounded_threshold:
        failures.append(
            f"grounded_rate {metrics['grounded_rate']} < threshold {settings.eval_grounded_threshold}"
        )
    if metrics["faithfulness"] is not None and metrics["faithfulness"] < settings.eval_faithfulness_threshold:
        failures.append(
            f"faithfulness {metrics['faithfulness']} < threshold {settings.eval_faithfulness_threshold}"
        )

    return {"pass": not failures, "failures": failures, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_health_check(sample_size=args.sample_size)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    status = "PASS" if result["pass"] else "FAIL"
    print(f"\n=== Eval Health Check: {status} ===")
    for k, v in result["metrics"].items():
        print(f"{k:18}: {v}")
    if result["failures"]:
        print("\nFailures:")
        for f in result["failures"]:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
