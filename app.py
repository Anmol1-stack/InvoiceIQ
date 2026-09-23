import ollama, re, math
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
    out = generate(
        "Write a function to check if a number is prime and test it.",
        tagged=True
    )

    print(out)

    code = extract(out, "PYTHON")
    tests = extract(out, "TESTS")

    ns = {}

    exec(code, ns)

    if tests:
        try:
            exec(tests, ns)
            print("\n[PASS] all generated tests passed")

        except AssertionError as e:
            print(
                f"\n[FAIL] generated test failed - "
                f"model produced a bad assertion: {e}"
            )

        except Exception as e:
            print(
                f"\n[ERROR] generated code errored: "
                f"{type(e).__name__}: {e}"
            )
