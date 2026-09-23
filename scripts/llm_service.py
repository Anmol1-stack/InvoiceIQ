import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8002
MODEL = os.environ.get("MODEL", "qwen2.5-coder:1.5b-instruct")
# The UI's batch evaluator permits 120 seconds.  End the upstream request a
# little earlier so this single-threaded service can return a useful error and
# accept the next request instead of remaining blocked behind Ollama.
GENERATION_TIMEOUT_SECONDS = float(os.environ.get("GENERATION_TIMEOUT_SECONDS", "105"))


def generate_answer(question, context, model, use_rag):

    if use_rag:
        prompt = f"""You are InvoiceIQ, a precise invoice and policy assistant.

Use only the provided context. Give the direct answer first, then only the
minimum supporting detail needed. Preserve all conditions, exceptions, and
negations exactly: do not confuse ordinary termination with immediate
termination, and do not turn a conditional rule into an unconditional rule.

If the context does not state a requested detail, say that it is not specified
in the provided policy. Do not guess or use general knowledge.

Context:
{context}

Question:
{question}

Answer in one or two concise sentences.
"""
    else:
        prompt = f"""You are InvoiceIQ, a helpful invoice and policy assistant.

Answer the question using your general knowledge. No knowledge-base context was
provided for this response, so do not claim that an answer comes from a company
policy or source. Be direct, and clearly state uncertainty when appropriate.

Question:
{question}

Answer in one or two concise sentences.
"""

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    request = urllib.request.Request(
        "http://host.docker.internal:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=GENERATION_TIMEOUT_SECONDS) as response:
        result = json.loads(response.read().decode("utf-8"))

    # Ollama supplies inference timings and token counters in its final JSON
    # response. Return them so the Week 4 dashboard can record real values
    # during a fresh evaluation rather than estimating token usage.
    metrics = {
        "prompt_tokens": result.get("prompt_eval_count"),
        "completion_tokens": result.get("eval_count"),
        "total_duration_ms": round(result.get("total_duration", 0) / 1_000_000, 3),
        "load_duration_ms": round(result.get("load_duration", 0) / 1_000_000, 3),
        "prompt_eval_duration_ms": round(
            result.get("prompt_eval_duration", 0) / 1_000_000,
            3
        ),
        "completion_eval_duration_ms": round(
            result.get("eval_duration", 0) / 1_000_000,
            3
        )
    }

    return result["response"], metrics, prompt


def send_json(handler, payload, status=200):
    response_bytes = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(response_bytes)))
    handler.end_headers()
    handler.wfile.write(response_bytes)


class LLMHandler(BaseHTTPRequestHandler):


    def do_POST(self):

        if self.path != "/generate":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        request_data = json.loads(body.decode("utf-8"))

        question = request_data["question"]
        context = request_data.get("context", "")
        model = request_data.get("model") or MODEL
        mode = request_data.get("mode", "rag")

        if mode not in ("rag", "non_rag"):
            send_json(self, {"error": "mode must be either 'rag' or 'non_rag'."}, 400)
            return

        try:
            answer, metrics, prompt = generate_answer(
                question, context, model, mode == "rag"
            )
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            send_json(
                self,
                {
                    "error": "LLM service is unavailable. Start Ollama and ensure the selected model is installed.",
                    "detail": str(error)
                },
                status=503
            )
            return

        response = {
            "question": question,
            "model": model,
            "mode": mode,
            "answer": answer,
            "metrics": metrics,
            "prompt": prompt,
        }

        send_json(self, response)

    def log_message(self, format, *args):
        print("LLM Service:", format % args)


print(f"LLM Service running on port {PORT}")

server = HTTPServer(("0.0.0.0", PORT), LLMHandler)
server.serve_forever()
