import ollama, re

MODEL = "qwen2.5-coder:1.5b-instruct"

def generate(prompt, tagged=False):
    if tagged:
        prompt += (
            "\n\nWrap the function in [PYTHON][/PYTHON] tags "
            "and the test cases in [TESTS][/TESTS] tags."
        )

    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return resp["message"]["content"]

def extract(text, tag):
    m = re.search(rf"\[{tag}\](.*?)\[/{tag}\]", text, re.DOTALL)

    if m:
        return m.group(1).strip()

    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)

    return m.group(1).strip() if m else None

if __name__ == "__main__":
    prompts = [
        "Write a function to check if a number is prime and test it.",
        "Write a class in Python to manage a student's grade book.",
        "Generate a Python script to scrape product names and prices from Flipkart.",
    ]

    for p in prompts:
        print("=" * 60)
        print(p)
        print("=" * 60)
        print(generate(p))
