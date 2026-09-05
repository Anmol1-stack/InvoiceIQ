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

        retrieval = call_service(
            "http://retrieval:8001/retrieve",
            {
                "question": question
            }
        )

        results = retrieval["results"]

        if not results:
            response = {
                "question": question,
                "retrieved_documents": [],
                "answer": "No relevant information was found in the knowledge base for this question."
            }

            response_bytes = json.dumps(
                response,
                indent=2
            ).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()

            self.wfile.write(response_bytes)
            return

        context = "\n\n".join(
            result["text"]
            for result in results
        )

        llm_response = call_service(
            "http://llm:8002/generate",
            {
                "question": question,
                "context": context,
                "model": model
            }
        )

        response = {
            "question": question,
            "model": llm_response.get("model", model),
            "retrieved_documents": results,
            "answer": llm_response["answer"]
        }

        response_bytes = json.dumps(
            response,
            indent=2
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()

        self.wfile.write(response_bytes)

    def log_message(self, format, *args):
        print("Orchestrator:", format % args)


print(f"Orchestrator running on port {PORT}")

server = HTTPServer(("0.0.0.0", PORT), OrchestratorHandler)
server.serve_forever()
