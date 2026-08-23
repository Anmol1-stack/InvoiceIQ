import json
import math
import urllib.request

MODEL = "nomic-embed-text"

with open("data/embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)


def get_embedding(text):
    payload = json.dumps({
        "model": MODEL,
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

    return dot / (norm_a * norm_b)


questions = [
    "How long do we have to pay an invoice?",
    "Can confidential information be disclosed to third parties?",
    "How much notice is required to terminate the agreement?"
]


for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)
    print("=" * 70)

    query_embedding = get_embedding(question)

    results = []

    for chunk in chunks:
        similarity = cosine_similarity(
            query_embedding,
            chunk["embedding"]
        )

        results.append({
            "source": chunk["source"],
            "similarity": similarity,
            "text": chunk["text"]
        })

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    print("\nTOP RETRIEVED DOCUMENTS:")

    for result in results:
        print("\nSource:", result["source"])
        print("Similarity:", round(result["similarity"], 4))
        print("Text:", result["text"][:250], "...")


print("\n" + "=" * 70)
print("RETRIEVAL TEST COMPLETE")
print("=" * 70)
