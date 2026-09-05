import json
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8002
MODEL = os.environ.get("MODEL", "qwen2.5-coder:1.5b-instruct")


def generate_answer(question, context, model):

    prompt = f"""Answer the user's question using only the provided context.

Context:
{context}

Question:
{question}

Give a concise answer based on the context.
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

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"]


class LLMHandler(BaseHTTPRequestHandler):


    def do_POST(self):

        if self.path != "/generate":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        request_data = json.loads(body.decode("utf-8"))

        question = request_data["question"]
        context = request_data["context"]
        model = request_data.get("model", MODEL)

        answer = generate_answer(question, context, model)

        response = {
            "question": question,
            "model": model,
            "answer": answer
        }

        response_bytes = json.dumps(response).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()

        self.wfile.write(response_bytes)

    def log_message(self, format, *args):
        print("LLM Service:", format % args)


print(f"LLM Service running on port {PORT}")

server = HTTPServer(("0.0.0.0", PORT), LLMHandler)
server.serve_forever()
