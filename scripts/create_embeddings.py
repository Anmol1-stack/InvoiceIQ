import json
import urllib.request

MODEL = "nomic-embed-text"

with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

payload = json.dumps({
    "model": MODEL,
    "input": texts
}).encode("utf-8")

request = urllib.request.Request(
    "http://localhost:11434/api/embed",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(request) as response:
    result = json.loads(response.read().decode("utf-8"))

embeddings = result["embeddings"]

for chunk, embedding in zip(chunks, embeddings):
    chunk["embedding"] = embedding
    chunk["embedding_dimension"] = len(embedding)

with open("data/embeddings.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"Created embeddings for {len(chunks)} chunks")
print(f"Embedding dimension: {len(embeddings[0])}")
print("Saved to data/embeddings.json")
