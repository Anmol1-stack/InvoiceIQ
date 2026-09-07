"""Analyse the existing Week 4 model-run JSON files without changing the app.

Usage (from the repository root):
    py week4_submission/analyze_saved_results.py
    py week4_submission/analyze_saved_results.py --write

The script reads the three completed 25-question runs in evaluation/results.
It applies one transparent rule-based rubric to every model, calculates
retrieval metrics from the returned sources, and prints a comparable summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evaluation" / "results"
OUTPUT_FILE = Path(__file__).resolve().parent / "saved_results_metrics.json"

# Each question has one policy that must appear in the retrieved top-k set.
# The policy text is deliberately unchanged; this is evaluation metadata only.
EXPECTED_SOURCES = {
    "Q01": "payment_policy.txt", "Q02": "payment_policy.txt",
    "Q03": "payment_policy.txt", "Q04": "payment_policy.txt",
    "Q05": "termination_policy.txt", "Q06": "termination_policy.txt",
    "Q07": "termination_policy.txt", "Q08": "termination_policy.txt",
    "Q09": "termination_policy.txt", "Q10": "termination_policy.txt",
    "Q11": "payment_policy.txt", "Q12": "payment_policy.txt",
    "Q13": "payment_policy.txt", "Q14": "payment_policy.txt",
    "Q15": "termination_policy.txt", "Q16": "termination_policy.txt",
    "Q17": "termination_policy.txt", "Q18": "payment_policy.txt",
    "Q19": "payment_policy.txt", "Q20": "termination_policy.txt",
    "Q21": "termination_policy.txt", "Q22": "termination_policy.txt",
    "Q23": "payment_policy.txt", "Q24": "payment_policy.txt",
    "Q25": "termination_policy.txt",
}


def contains_all(text: str, *terms: str) -> bool:
    return all(term in text for term in terms)


def is_correct(question_id: str, answer: str) -> bool:
    """Return True only when the answer includes the required policy fact."""
    text = (answer or "").lower().strip()
    no_answer = ("no relevant information", "not found in the knowledge base")
    if any(phrase in text for phrase in no_answer):
        return False

    if question_id in {"Q01", "Q11", "Q12"}:
        return ("30" in text or "thirty" in text) and "invoice" in text
    if question_id == "Q02":
        return ("interest" in text or "charge" in text) and "late" in text
    if question_id in {"Q03", "Q13"}:
        return ("10" in text or "ten" in text) and "business" in text
    if question_id in {"Q04", "Q19", "Q14"}:
        return "undisputed" in text and ("pay" in text or "payable" in text) and (
            question_id != "Q14" or "no" in text[:50] or "cannot" in text
        )
    if question_id in {"Q05", "Q15"}:
        return ("30" in text or "thirty" in text) and "notice" in text
    if question_id in {"Q06", "Q20"}:
        return contains_all(text, "immediate", "termination") and (
            "15" in text or "fifteen" in text
        ) and ("breach" in text or "correct" in text)
    if question_id == "Q07":
        return ("no" in text[:50] or "does not" in text or "remain" in text) and "payment" in text
    if question_id == "Q08":
        return ("no" in text[:50] or "does not" in text or "continue" in text) and "confidential" in text
    if question_id == "Q09":
        return ("no" in text[:50] or "does not" in text or "continue" in text) and (
            "intellectual" in text or "ip" in text
        )
    if question_id in {"Q10", "Q17"}:
        return "confidential" in text and ("intellectual" in text or "ip" in text) and "payment" in text
    if question_id == "Q16":
        return ("15" in text or "fifteen" in text) and "day" in text
    if question_id == "Q18":
        return "july 1" in text or (("30" in text or "thirty" in text) and "june" in text)
    if question_id == "Q21":
        return ("no" in text[:50] or "cannot" in text or "remain" in text) and "payment" in text
    if question_id == "Q22":
        return ("no" in text[:50] or "cannot" in text) and "notice" in text and (
            "breach" in text or "immediate" in text
        )
    if question_id in {"Q23", "Q24"}:
        return any(phrase in text for phrase in ("not specified", "not stated", "no specific", "no mention"))
    if question_id == "Q25":
        return ("30" in text or "thirty" in text) and any(
            phrase in text for phrase in ("no maximum", "not stated", "does not state", "maximum")
        )
    raise ValueError(f"No rubric is configured for {question_id}")


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, round(0.95 * len(ordered)) - 1)]


def analyse_run(result_file: Path) -> dict:
    records = json.loads(result_file.read_text(encoding="utf-8"))
    if len(records) != 25:
        raise ValueError(f"{result_file.name} has {len(records)} records; expected 25")

    correct = relevance = grounded_errors = retrieved = 0
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    similarities: list[float] = []
    question_rows: list[dict] = []

    for row in records:
        question_id = row["id"]
        response = row.get("response", {})
        answer = response.get("answer", "")
        documents = response.get("retrieved_documents", [])
        answer_is_correct = is_correct(question_id, answer)
        expected_source = EXPECTED_SOURCES[question_id]
        source_rank = next(
            (index for index, document in enumerate(documents, 1)
             if document.get("source") == expected_source),
            None,
        )

        correct += answer_is_correct
        relevance += answer_is_correct  # narrow task set: core-answer relevance is the rubric.
        retrieved += source_rank is not None
        if source_rank:
            reciprocal_ranks.append(1 / source_rank)
        else:
            reciprocal_ranks.append(0.0)
        # A non-empty, rubric-failing answer despite retrieved evidence is a
        # reproducible proxy for an unsupported/contradictory claim.
        grounded_errors += bool(documents and answer.strip() and not answer_is_correct)
        latencies.append(float(row["latency_seconds"]))
        if documents:
            similarities.append(float(documents[0].get("similarity", 0.0)))
        question_rows.append({
            "id": question_id,
            "correct": answer_is_correct,
            "expected_source": expected_source,
            "expected_source_rank": source_rank,
            "top1_similarity": round(float(documents[0].get("similarity", 0.0)), 4) if documents else None,
            "latency_seconds": float(row["latency_seconds"]),
        })

    total = len(records)
    return {
        "model": result_file.stem,
        "source_file": str(result_file.relative_to(ROOT)).replace("\\", "/"),
        "questions": total,
        "correctness_accuracy_percent": round(100 * correct / total, 1),
        "answer_relevance_percent": round(100 * relevance / total, 1),
        "retrieval_recall_at_4_percent": round(100 * retrieved / total, 1),
        "retrieval_mrr_at_4": round(statistics.mean(reciprocal_ranks), 4),
        "mean_top1_similarity": round(statistics.mean(similarities), 4),
        "unsupported_or_contradictory_answer_rate_percent": round(100 * grounded_errors / total, 1),
        "average_latency_seconds": round(statistics.mean(latencies), 3),
        "p95_latency_seconds": round(percentile_95(latencies), 3),
        "minimum_latency_seconds": round(min(latencies), 3),
        "maximum_latency_seconds": round(max(latencies), 3),
        "question_rows": question_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="save JSON next to this script")
    args = parser.parse_args()

    files = sorted(
        path for path in RESULTS_DIR.glob("*.json")
        if "previous" not in path.stem
    )
    summaries = [analyse_run(path) for path in files]
    for summary in summaries:
        print(
            f"{summary['model']}: accuracy={summary['correctness_accuracy_percent']}%, "
            f"recall@4={summary['retrieval_recall_at_4_percent']}%, "
            f"MRR@4={summary['retrieval_mrr_at_4']}, "
            f"unsupported={summary['unsupported_or_contradictory_answer_rate_percent']}%, "
            f"avg latency={summary['average_latency_seconds']}s, "
            f"p95={summary['p95_latency_seconds']}s"
        )
    if args.write:
        OUTPUT_FILE.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"Saved {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
