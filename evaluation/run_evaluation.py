import json
import os
import time
import urllib.request
import urllib.error

QUESTIONS_FILE = "evaluation/questions.json"
RESULTS_DIR = "evaluation/results"
API_URL = "http://localhost:8000/ask"

MODEL_NAME = os.environ.get("EVAL_MODEL", "qwen2.5-coder:1.5b-instruct")
OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    MODEL_NAME.replace(":", "_").replace("-", "_") + ".json"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

results = []

print("=" * 70)
print(f"MODEL: {MODEL_NAME}")
print(f"QUESTIONS: {len(questions)}")
print("=" * 70)

for i, item in enumerate(questions, 1):
    qid = item["id"]
    question = item["question"]
    expected = item["expected"]

    print(f"[{i}/{len(questions)}] {qid}: {question}")

    payload = json.dumps({
        "question": question
    }).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))

        latency = time.perf_counter() - start

        results.append({
            "id": qid,
            "question": question,
            "expected": expected,
            "response": data,
            "latency_seconds": round(latency, 3)
        })

        print(f"    OK - {latency:.2f}s")

    except Exception as e:
        latency = time.perf_counter() - start

        results.append({
            "id": qid,
            "question": question,
            "expected": expected,
            "error": str(e),
            "latency_seconds": round(latency, 3)
        })

        print(f"    ERROR - {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print("=" * 70)
print(f"Saved: {OUTPUT_FILE}")
print(f"Completed entries: {len(results)}")
print("=" * 70)
