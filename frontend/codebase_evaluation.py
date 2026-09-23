"""Evidence-based scoring and persistence for Model Evaluation's codebase suite.

The suite calls the existing InvoiceIQ application route.  This module contains
no model client, no code search, and no pre-computed outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CATEGORIES = (
    "Explanation",
    "Code Retrieval",
    "Dependency Understanding",
    "Bug Analysis",
    "Code Generation",
    "Refactoring",
    "RAG-based Question",
)


def validate_questions(questions: list[dict[str, Any]]) -> None:
    if not isinstance(questions, list):
        raise ValueError("The codebase dataset must be a JSON list.")
    required = {"id", "category", "question", "expected_behavior", "evaluation_type"}
    ids = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Each codebase dataset entry must be an object.")
        missing = required - set(question)
        if missing:
            raise ValueError(f"{question.get('id', 'question')} is missing: {', '.join(sorted(missing))}")
        if question["category"] not in CATEGORIES:
            raise ValueError(f"{question['id']} has an unsupported category")
        if question["id"] in ids:
            raise ValueError(f"Duplicate codebase evaluation id: {question['id']}")
        ids.add(question["id"])
    if {question["category"] for question in questions} != set(CATEGORIES):
        raise ValueError("The codebase dataset must include every one of the seven categories.")


def _contains(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def evaluate_answer(question: dict[str, Any], answer: str, response: dict[str, Any]) -> tuple[bool, str]:
    """Category-specific, deterministic checks against an actual model answer.

    Dataset criteria describe the requested behavior. The function only returns
    pass/fail after a live answer has been received; it never supplies an
    answer, score, or visualization value.
    """
    answer = (answer or "").strip()
    if not answer:
        return False, "No generated answer was returned by the application."

    required = question.get("required_terms", [])
    alternatives = question.get("any_terms", [])
    missing = [term for term in required if not _contains(answer, term)]
    has_alternative = not alternatives or any(_contains(answer, term) for term in alternatives)
    category = question["category"]

    if category == "Explanation":
        passed = not missing and has_alternative and len(answer.split()) >= 12
        rationale = "Explanation includes the required concept and relevant supporting detail."
    elif category == "Code Retrieval":
        passed = not missing and has_alternative
        rationale = "Answer identifies the requested code location/function evidence."
    elif category == "Dependency Understanding":
        passed = not missing and has_alternative
        rationale = "Answer identifies the required dependency/call relationship terms."
    elif category == "Bug Analysis":
        passed = not missing and has_alternative
        rationale = "Answer identifies both the problem and a corrective direction."
    elif category == "Code Generation":
        passed = not missing and has_alternative and ("def " in answer.lower() or "```" in answer)
        rationale = "Generated response contains code form and the requested implementation elements."
    elif category == "Refactoring":
        passed = not missing and has_alternative
        rationale = "Proposal addresses separation/reuse while retaining the requested behavior."
    elif category == "RAG-based Question":
        documents = response.get("retrieved_documents") or []
        expected_sources = set(question.get("expected_sources", []))
        returned_sources = {document.get("source") for document in documents}
        source_supported = bool(documents) and (not expected_sources or bool(expected_sources & returned_sources))
        passed = not missing and has_alternative and source_supported
        rationale = "Answer satisfies policy criteria and is supported by returned retrieval context."
        if not source_supported:
            rationale = "The required policy source was not present in the retrieved context."
    else:  # Dataset validation prevents this, but keep runtime behavior safe.
        return False, "Unsupported evaluation category."

    if passed:
        return True, rationale
    failures = []
    if missing:
        failures.append("missing required terms: " + ", ".join(missing))
    if alternatives and not has_alternative:
        failures.append("missing a relevant supporting term")
    if category == "Explanation" and len(answer.split()) < 12:
        failures.append("explanation is too brief")
    if category == "Code Generation" and not ("def " in answer.lower() or "```" in answer):
        failures.append("no code block or function definition found")
    return False, "; ".join(failures) or "Answer did not satisfy the category-specific criteria."


def load_results(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["results"]
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("results is not a list")
        return rows, payload.get("completed_at")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return [], None


def save_results(path: Path, rows: list[dict[str, Any]], completed_at: str) -> None:
    """Atomically persist only results returned by completed application calls."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"completed_at": completed_at, "results": rows}, indent=2), encoding="utf-8")
    temporary.replace(path)
