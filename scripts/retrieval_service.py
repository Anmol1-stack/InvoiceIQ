"""FAISS-backed retrieval service for the InvoiceIQ policy knowledge base."""

import hashlib
import json
import os
from pathlib import Path
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import faiss
import numpy as np

PORT = 8001
MODEL = "nomic-embed-text"
TOP_K = 4
SIMILARITY_THRESHOLD = 0.60
EMBEDDINGS_PATH = Path("data/embeddings.json")
INDEX_PATH = Path(os.environ.get(
    "FAISS_INDEX_PATH", "data/policy_chunks.faiss"
))
METADATA_PATH = Path(os.environ.get(
    "FAISS_METADATA_PATH", "data/policy_chunks.metadata.json"
))


def load_embedded_chunks():
    """Load the precomputed vectors used to build or refresh the FAISS index."""
    with EMBEDDINGS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise RuntimeError("No embedded policy chunks are available.")
    if any("embedding" not in chunk for chunk in chunks):
        raise RuntimeError("One or more policy chunks have no embedding.")

    return chunks


def source_fingerprint():
    """Refresh the persistent index only when its source embeddings change."""
    return hashlib.sha256(EMBEDDINGS_PATH.read_bytes()).hexdigest()


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32).copy()
    faiss.normalize_L2(vectors)
    return vectors


def build_or_load_index():
    """Return the persisted cosine-similarity FAISS index and metadata.

    Inner-product search over L2-normalized vectors is cosine similarity, so
    this preserves the prior retriever's relevance scores and threshold.
    """
    chunks = load_embedded_chunks()
    fingerprint = source_fingerprint()
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if INDEX_PATH.exists() and METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            if (
                metadata.get("fingerprint") == fingerprint
                and len(metadata.get("documents", [])) == len(chunks)
            ):
                index = faiss.read_index(str(INDEX_PATH))
                if index.ntotal == len(chunks):
                    return index, metadata["documents"]
        except (OSError, ValueError, RuntimeError):
            # Rebuild safely if the persisted index is partial or incompatible.
            pass

    vectors = normalize([chunk["embedding"] for chunk in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    documents = [
        {"source": chunk["source"], "text": chunk["text"]}
        for chunk in chunks
    ]

    faiss.write_index(index, str(INDEX_PATH))
    METADATA_PATH.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "embedding_model": MODEL,
                "dimensions": int(vectors.shape[1]),
                "documents": documents,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return index, documents


FAISS_INDEX, DOCUMENTS = build_or_load_index()
INDEX_LOCK = threading.Lock()


def get_embedding(text):
    payload = json.dumps({"model": MODEL, "input": text}).encode("utf-8")
    request = urllib.request.Request(
        "http://host.docker.internal:11434/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["embeddings"][0]


def search(query_embedding):
    query = normalize([query_embedding])
    if query.shape[1] != FAISS_INDEX.d:
        raise ValueError(
            "Query embedding dimension does not match the FAISS index. "
            "Regenerate data/embeddings.json with the configured embedding model."
        )

    with INDEX_LOCK:
        scores, ids = FAISS_INDEX.search(query, TOP_K)

    results = []
    for score, document_id in zip(scores[0], ids[0]):
        if document_id < 0 or score < SIMILARITY_THRESHOLD:
            continue
        document = DOCUMENTS[int(document_id)]
        results.append(
            {
                "source": document["source"],
                "text": document["text"],
                "similarity": float(score),
            }
        )
    return results


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
        request_data = json.loads(self.rfile.read(length).decode("utf-8"))
        question = request_data["question"]

        try:
            results = search(get_embedding(question))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            send_json(
                self,
                {
                    "error": "Embedding service is unavailable. Start Ollama and ensure nomic-embed-text is installed.",
                    "detail": str(error),
                },
                status=503,
            )
            return
        except ValueError as error:
            send_json(self, {"error": str(error)}, status=500)
            return

        send_json(
            self,
            {
                "question": question,
                "results": results,
                "relevant": bool(results),
            },
        )

    def log_message(self, format, *args):
        print("Retrieval Service:", format % args)


print(
    f"FAISS retrieval service running on port {PORT} "
    f"with {FAISS_INDEX.ntotal} indexed policy chunks"
)
server = HTTPServer(("0.0.0.0", PORT), RetrievalHandler)
server.serve_forever()
