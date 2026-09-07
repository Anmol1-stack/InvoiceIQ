import json
import math
import urllib.request

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5-coder:1.5b-instruct"

with open("data/embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)


def ollama_embed(text):
    payload = json.dumps({
        "model": EMBED_MODEL,
        "input": text
    }).encode("utf-8")

    request = urllib.request.Request(
        "http://localhost:11434/api/embed",
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
        return 0.0

    return dot / (norm_a * norm_b)


def retrieve(question, top_k=2):
    query_embedding = ollama_embed(question)

    results = []

    for chunk in chunks:
        score = cosine_similarity(
            query_embedding,
            chunk["embedding"]
        )

        results.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "text": chunk["text"],
            "score": score
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


def ask_ollama(prompt):
    payload = json.dumps({
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"]


question = "How long do we have to pay an invoice?"

results = retrieve(question, top_k=2)

context = "\n\n".join(
    f"[Source: {r['source']}]\n{r['text']}"
    for r in results
)

print("=" * 60)
print("QUESTION")
print("=" * 60)
print(question)

print("\n" + "=" * 60)
print("RETRIEVED INFORMATION")
print("=" * 60)

for r in results:
    print(f"\nSource: {r['source']}")
    print(f"Similarity: {r['score']:.4f}")
    print(f"Text: {r['text']}")

print("\n" + "=" * 60)
print("LLM RESPONSE WITHOUT RAG")
print("=" * 60)

no_rag_prompt = f"""
Answer the following question using your general knowledge.

Question:
{question}

Answer:
"""

print(ask_ollama(no_rag_prompt))

print("\n" + "=" * 60)
print("LLM RESPONSE WITH RAG")
print("=" * 60)

rag_prompt = f"""
Answer the question using ONLY the provided contract information.

Context:
{context}

Question:
{question}

Give a concise answer and mention the relevant source.
"""

print(ask_ollama(rag_prompt))
