import json
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

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

        print("\nQuestion:", question)
        print("Model:", model if model else "default")

        try:
            retrieval = call_service(
                "http://retrieval:8001/retrieve",
                {
                    "question": question
                }
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

        results = retrieval["results"]

        if not results:
            response = {
                "question": question,
                "retrieved_documents": [],
                "answer": "No relevant information was found in the knowledge base for this question."
            }

            send_json(self, response)
            return

        context = "\n\n".join(
            result["text"]
            for result in results
        )

        try:
            llm_response = call_service(
                "http://llm:8002/generate",
                {
                    "question": question,
                    "context": context,
                    "model": model
                }
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
            "retrieved_documents": results,
            "answer": llm_response["answer"],
            "llm_metrics": llm_response.get("metrics", {})
        }

        send_json(self, response)

    def log_message(self, format, *args):
        print("Orchestrator:", format % args)


print(f"Orchestrator running on port {PORT}")

server = HTTPServer(("0.0.0.0", PORT), OrchestratorHandler)
server.serve_forever()
