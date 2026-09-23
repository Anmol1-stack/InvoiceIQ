import json
import sys
import statistics

if len(sys.argv) != 2:
    print("Usage: python evaluation/analyze_results.py <results_file>")
    sys.exit(1)

file = sys.argv[1]

with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Manual semantic evaluation:
# CORRECT = substantively matches the expected answer.
# INCORRECT = contradicts, omits a necessary key fact, answers
# a different question, or makes an unsupported claim.

decisions = {
    "Q01": "correct",
    "Q02": "correct",
    "Q03": "correct",
    "Q04": "correct",
    "Q05": "correct",
    "Q06": "correct",
    "Q07": "incorrect",
    "Q08": "incorrect",
    "Q09": "correct",
    "Q10": "correct",
    "Q11": "incorrect",
    "Q12": "correct",
    "Q13": "correct",
    "Q14": "incorrect",
    "Q15": "correct",
    "Q16": "correct",
    "Q17": "incorrect",
    "Q18": "correct",
    "Q19": "correct",
    "Q20": "incorrect",
    "Q21": "correct",
    "Q22": "incorrect",
    "Q23": "correct",
    "Q24": "correct",
    "Q25": "incorrect",
}

correct = sum(decisions.get(x["id"]) == "correct" for x in data)
incorrect = sum(decisions.get(x["id"]) == "incorrect" for x in data)

latencies = [
    x["latency_seconds"]
    for x in data
    if "latency_seconds" in x
]

similarities = []

for x in data:
    response = x.get("response", {})
    for doc in response.get("retrieved_documents", []):
        if doc.get("similarity") is not None:
            similarities.append(doc["similarity"])
            break

accuracy = (correct / len(data)) * 100 if data else 0

print("=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"File: {file}")
print(f"Questions evaluated: {len(data)}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {accuracy:.2f}%")

if latencies:
    print(f"Average latency: {statistics.mean(latencies):.3f}s")
    print(f"Minimum latency: {min(latencies):.3f}s")
    print(f"Maximum latency: {max(latencies):.3f}s")

if similarities:
    print(f"Average top-1 similarity: {statistics.mean(similarities):.4f}")

print()
print("QUESTION-LEVEL RESULTS")
print("-" * 60)

for x in data:
    qid = x["id"]
    result = decisions.get(qid, "unscored").upper()
    print(f"{qid}: {result}")

print("=" * 60)
