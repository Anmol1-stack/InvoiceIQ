import json
import math
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8001
MODEL = "nomic-embed-text"

with open("data/embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)


def get_embedding(text):
    payload = json.dumps({
        "model": MODEL,
        "input": text
    }).encode("utf-8")

    request = urllib.request.Request(
        "http://host.docker.internal:11434/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["embeddings"][0]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot / (norm_a * norm_b)


def send_json(handler, payload, status=200):
    response_bytes = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(response_bytes)))
    handler.end_headers()
    handler.wfile.write(response_bytes)


class RetrievalHandler(BaseHTTPRequestHandler):

    def do_POST(self):

        if self.path != "/retrieve":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        request_data = json.loads(body.decode("utf-8"))
        question = request_data["question"]

        try:
            query_embedding = get_embedding(question)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            send_json(
                self,
                {
                    "error": "Embedding service is unavailable. Start Ollama and ensure nomic-embed-text is installed.",
                    "detail": str(error)
                },
                status=503
            )
            return

        results = []

        for chunk in chunks:
            similarity = cosine_similarity(
                query_embedding,
                chunk["embedding"]
            )

            results.append({
                "source": chunk["source"],
                "text": chunk["text"],
                "similarity": similarity
            })

        results.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        SIMILARITY_THRESHOLD = 0.60
        top_results = [
            result for result in results[:4]
            if result["similarity"] >= SIMILARITY_THRESHOLD
        ]

        response = {
            "question": question,
            "results": top_results,
            "relevant": len(top_results) > 0
        }

        send_json(self, response)

    def log_message(self, format, *args):
        print("Retrieval Service:", format % args)


print(f"Retrieval Service running on port {PORT}")

server = HTTPServer(("0.0.0.0", PORT), RetrievalHandler)
server.serve_forever()
