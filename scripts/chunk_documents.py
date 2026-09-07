import json
from pathlib import Path

SOURCE_DIR = Path("knowledge_base")
OUTPUT_FILE = Path("data/chunks.json")

CHUNK_SIZE = 80
OVERLAP = 20

chunks = []

for file_path in sorted(SOURCE_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8").strip()
    words = text.split()

    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])

        chunks.append({
            "chunk_id": f"{file_path.stem}_{chunk_id}",
            "source": file_path.name,
            "text": chunk_text
        })

        if end == len(words):
            break

        start = end - OVERLAP
        chunk_id += 1

OUTPUT_FILE.parent.mkdir(exist_ok=True)
OUTPUT_FILE.write_text(
    json.dumps(chunks, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Created {len(chunks)} chunks")
print(f"Saved to {OUTPUT_FILE}")
