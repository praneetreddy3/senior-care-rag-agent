"""Evaluation pipeline for the RAG agent.

Reports two families of metrics:

Retrieval quality (does the right document come back?):
  - Hit@k        : fraction of questions where the expected source is in top-k
  - MRR          : mean reciprocal rank of the expected source

Answer quality (is the generated answer good?):
  - Keyword coverage : fraction of expected keywords present in the answer
  - Faithfulness     : LLM-judge check that the answer is grounded (if LLM enabled)

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --k 4 --json eval/results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from src.agent import Agent
from src.config import ROOT, settings
from src.llm import generate

EVAL_FILE = Path(__file__).parent / "eval_set.json"


def load_eval() -> list[dict]:
    return json.loads(EVAL_FILE.read_text(encoding="utf-8"))


def retrieval_metrics(agent: Agent, items: list[dict], k: int) -> dict:
    hits, rr = [], []
    for item in items:
        results = agent.retriever.retrieve(item["question"], top_k=k)
        sources = [c.source for c, _ in results]
        expected = item["expected_source"]
        if expected in sources:
            hits.append(1.0)
            rr.append(1.0 / (sources.index(expected) + 1))
        else:
            hits.append(0.0)
            rr.append(0.0)
    return {"hit@k": mean(hits), "mrr": mean(rr), "k": k}


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    a = answer.lower()
    return mean(1.0 if kw.lower() in a else 0.0 for kw in keywords)


def faithfulness(question: str, answer: str, context: str) -> float | None:
    if not settings.llm_enabled:
        return None
    verdict = generate(
        prompt=(
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
            "Is the answer fully supported by the context and does it address the "
            "question? Reply strictly 'yes' or 'no'."
        ),
        temperature=0.0,
    )
    return 1.0 if verdict and verdict.strip().lower().startswith("y") else 0.0


def answer_metrics(agent: Agent, items: list[dict]) -> dict:
    coverage, faith, grounded = [], [], []
    for item in items:
        state = agent.run(item["question"])
        answer = state.get("answer", "")
        coverage.append(keyword_coverage(answer, item.get("answer_keywords", [])))
        grounded.append(1.0 if state.get("grounded") else 0.0)
        context = "\n\n".join(d["text"] for d in state.get("relevant", []))
        f = faithfulness(item["question"], answer, context)
        if f is not None:
            faith.append(f)
    out = {"keyword_coverage": mean(coverage) if coverage else 0.0}
    out["grounded_rate"] = mean(grounded) if grounded else 0.0  # agent's own verify
    out["faithfulness"] = mean(faith) if faith else None        # independent LLM judge
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    items = load_eval()
    agent = Agent()

    r = retrieval_metrics(agent, items, args.k)
    a = answer_metrics(agent, items)

    print("\n=== RAG Evaluation ===")
    print(f"Questions          : {len(items)}")
    print(f"LLM generation     : {'on' if settings.llm_enabled else 'off (extractive)'}")
    print(f"Embedding backend  : {settings.embedding_backend}")
    print("--- Retrieval ---")
    print(f"Hit@{r['k']}            : {r['hit@k']:.3f}")
    print(f"MRR               : {r['mrr']:.3f}")
    print("--- Answer ---")
    print(f"Keyword coverage  : {a['keyword_coverage']:.3f}")
    print(f"Grounded rate     : {a['grounded_rate']:.3f}  (agent self-verification)")
    if a["faithfulness"] is not None:
        print(f"Faithfulness      : {a['faithfulness']:.3f}  (independent LLM judge)")
    else:
        print("Faithfulness      : n/a (set LLM_API_KEY to enable LLM-judge)")

    results = {"retrieval": r, "answer": a, "n": len(items)}
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved results to {args.json}")


if __name__ == "__main__":
    main()
