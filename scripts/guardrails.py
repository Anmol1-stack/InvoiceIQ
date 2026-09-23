"""Small, deterministic guardrails used by the optional InvoiceIQ safety views.

The checks deliberately live outside the retrieval and model implementations so
the existing RAG and non-RAG request paths keep their current behaviour.
"""

import re


MAX_QUESTION_LENGTH = 500
CONTROLLED_RESPONSE = (
    "I can only answer InvoiceIQ questions that are supported by the provided "
    "invoice and policy knowledge base."
)

SCOPE_TERMS = {
    "invoice", "invoices", "payment", "payments", "payable", "paid",
    "billing", "bill", "dispute", "disputed", "refund", "credit",
    "contract", "agreement", "termination", "terminate", "breach",
    "amendment", "renewal", "policy", "supplier", "customer",
    "late", "overdue", "due date", "payment terms", "confidentiality",
    "intellectual property", "invoiceiq",
}

BLOCKED_PATTERNS = (
    "ignore previous", "ignore the previous", "system prompt", "developer message",
    "jailbreak", "reveal your instructions", "act as", "bypass",
    "password", "api key", "secret key", "credit card number",
)


def controlled_result(reason):
    return {
        "allowed": False,
        "reason": reason,
        "response": CONTROLLED_RESPONSE + " " + reason,
    }


def check_input(question):
    """Return an allow/refuse decision before retrieval or model invocation."""
    text = (question or "").strip()
    lowered = text.lower()
    if not text:
        return controlled_result("Please enter an invoice or policy question.")
    if len(text) > MAX_QUESTION_LENGTH:
        return controlled_result(
            f"Please keep questions to {MAX_QUESTION_LENGTH} characters or fewer."
        )
    if any(pattern in lowered for pattern in BLOCKED_PATTERNS):
        return controlled_result("This request is not something InvoiceIQ can process.")
    if not any(term in lowered for term in SCOPE_TERMS):
        return controlled_result(
            "This question is outside InvoiceIQ's invoice, payment, contract, and policy scope."
        )
    return {"allowed": True, "reason": "Input is within InvoiceIQ scope."}


def check_output(answer, documents):
    """Conservative grounding check for a generated RAG answer.

    Passing requires retrieved evidence and either an explicit unavailable-detail
    response or meaningful answer/context word overlap. Numbers in an answer
    must also occur in the retrieved context, preventing invented deadlines or
    percentages from being accepted.
    """
    text = (answer or "").strip()
    context = " ".join(item.get("text", "") for item in documents).lower()
    if not text:
        return False, "The model returned no answer."
    if not context:
        return False, "No supporting knowledge-base content was retrieved."
    lowered = text.lower()
    unavailable = ("not specified", "not stated", "not available", "no relevant information")
    if any(phrase in lowered for phrase in unavailable):
        return True, "The answer appropriately states that the detail is unavailable."

    answer_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", lowered)
    number_words = {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
        "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "hundred", "thousand",
    }
    answer_number_words = set(re.findall(r"[a-z]+", lowered)) & number_words
    if (any(number not in context for number in answer_numbers) or
            any(word not in context for word in answer_number_words)):
        return False, "The answer contains a number not present in the retrieved information."

    words = set(re.findall(r"[a-z]{4,}", lowered))
    context_words = set(re.findall(r"[a-z]{4,}", context))
    overlap = words & context_words
    if len(overlap) < 3:
        return False, "The answer does not contain enough support from the retrieved information."
    return True, "Answer is grounded in the retrieved knowledge-base content."
