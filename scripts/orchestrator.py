import json
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from guardrails import CONTROLLED_RESPONSE, check_input, check_output

PORT = 8003


def call_service(url, data):

    payload = json.dumps(data).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def send_json(handler, payload, status=200):
    response_bytes = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(response_bytes)))
    handler.end_headers()
    handler.wfile.write(response_bytes)


class OrchestratorHandler(BaseHTTPRequestHandler):

    def do_POST(self):

        if self.path != "/ask":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        request_data = json.loads(body.decode("utf-8"))
        question = request_data["question"]
        model = request_data.get("model")
        guardrails_enabled = request_data.get("guardrails_enabled", False)
        # Keep RAG as the default so existing API clients continue to work,
        # while allowing the UI to explicitly request a model-only answer.
        mode = request_data.get("mode")
        if mode is None:
            use_rag = request_data.get("use_rag", True)
            mode = "rag" if use_rag else "non_rag"

        if mode not in ("rag", "non_rag"):
            send_json(
                self,
                {"error": "mode must be either 'rag' or 'non_rag'."},
                status=400
            )
            return

        use_rag = mode == "rag"

        # This is opt-in so existing API clients and the Week 3/4 workflows
        # retain their established behaviour. The Guardrails dashboard enables
        # it explicitly and therefore never sends an unsafe request to the LLM.
        if guardrails_enabled:
            input_check = check_input(question)
            if not input_check["allowed"]:
                send_json(
                    self,
                    {
                        "question": question,
                        "mode": mode,
                        "guardrail": {"status": "rejected", **input_check},
                        "retrieval_used": False,
                        "retrieved_documents": [],
                        "answer": input_check["response"],
                    },
                )
                return
            if not use_rag:
                refusal = (
                    "Guarded answers require RAG so InvoiceIQ can verify the "
                    "answer against its knowledge base."
                )
                send_json(self, {
                    "question": question, "mode": mode,
                    "guardrail": {"status": "rejected", "allowed": False, "reason": refusal},
                    "retrieval_used": False, "retrieved_documents": [],
                    "answer": CONTROLLED_RESPONSE + " " + refusal,
                })
                return

        print("\nQuestion:", question)
        print("Model:", model if model else "default")
        print("Mode:", mode)

        results = []
        context = ""

        if use_rag:
            retrieval_started = time.perf_counter()
            try:
                retrieval = call_service(
                    "http://retrieval:8001/retrieve",
                    {"question": question}
                )
            except Exception as error:
                send_json(
                    self,
                    {
                        "error": "Retrieval could not run. Check that Ollama and the embedding model are available.",
                        "detail": str(error)
                    },
                    status=503
                )
                return

            retrieval_latency_ms = round((time.perf_counter() - retrieval_started) * 1000, 3)

            results = retrieval["results"]

            if not results:
                answer = "No relevant information was found in the knowledge base for this question."
                guardrail = {}
                if guardrails_enabled:
                    answer = (
                        "I cannot provide an answer because sufficient supporting information "
                        "was not found in the InvoiceIQ knowledge base."
                    )
                    guardrail = {"status": "rejected", "allowed": False,
                                 "reason": "No sufficient knowledge-base evidence was retrieved."}
                send_json(
                    self,
                    {
                        "question": question,
                        "mode": mode,
                        "retrieval_used": True,
                        "retrieved_documents": [],
                        "answer": answer,
                        "guardrail": guardrail,
                    }
                )
                return

            context = "\n\n".join(result["text"] for result in results)

        try:
            llm_request = {
                "question": question,
                "context": context,
                "mode": mode
            }

            if model:
                llm_request["model"] = model

            llm_response = call_service(
                "http://llm:8002/generate",
                llm_request
            )
        except Exception as error:
            send_json(
                self,
                {
                    "error": "Answer generation could not run. Check that Ollama and the selected model are available.",
                    "detail": str(error)
                },
                status=503
            )
            return

        response = {
            "question": question,
            "model": llm_response.get("model", model),
            "mode": mode,
            "retrieval_used": use_rag,
            "retrieved_documents": results,
            "answer": llm_response["answer"],
            "llm_metrics": llm_response.get("metrics", {}),
            "prompt": llm_response.get("prompt"),
            "retrieval_metrics": {
                "duration_ms": retrieval_latency_ms if use_rag else None,
                "documents_retrieved": len(results),
            },
        }

        if guardrails_enabled:
            output_ok, output_reason = check_output(response["answer"], results)
            response["output_check"] = {
                "passed": output_ok,
                "reason": output_reason,
            }
            if not output_ok:
                response["answer"] = (
                    "I cannot provide an answer because it could not be verified "
                    "against the InvoiceIQ knowledge base."
                )
                response["guardrail"] = {
                    "status": "rejected", "allowed": False, "reason": output_reason
                }
            else:
                response["guardrail"] = {
                    "status": "passed", "allowed": True,
                    "reason": "Input and output checks passed."
                }

        send_json(self, response)

    def log_message(self, format, *args):
        print("Orchestrator:", format % args)


print(f"Orchestrator running on port {PORT}")

server = HTTPServer(("0.0.0.0", PORT), OrchestratorHandler)
server.serve_forever()
