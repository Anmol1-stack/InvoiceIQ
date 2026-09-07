# Showing the completed Week 4 dashboard

## 1. Start the backend after the metric update

From the repository root, rebuild and start the existing services so `llm_service.py` exposes Ollama token/timing metrics:

```powershell
docker compose up --build
```

Keep this terminal open. Ollama must also be running locally with the three required models available.

## Before running model questions: start Ollama

The Docker services call Ollama through `host.docker.internal:11434`. Start the Ollama application/service on Windows, or run the following in a separate terminal if Ollama is installed:

```powershell
ollama serve
```

Then make sure the embedding model and all three comparison models are available:

```powershell
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:1.5b-instruct
ollama pull llama3.2:3b
ollama pull deepseek-coder:1.3b
```

If Ollama is not running, InvoiceIQ now displays a clear `503` error explaining that retrieval cannot reach the embedding model. It does not silently show an empty or misleading answer.

## 2. Start the Streamlit dashboard locally

Use a second terminal from the repository root:

```powershell
py -3.11 -m pip install -r frontend\requirements.txt
py -3.11 -m streamlit run frontend\app.py
```

Running Streamlit locally is intentional: it lets the dashboard sample the local Ollama process for CPU/RAM and, when NVIDIA tooling is available, host GPU usage.

## 3. What to show the evaluator

In the sidebar, select **Week 4 — Final Submission**.

1. **Overview** — maps every evaluator exercise to its evidence.
2. **Model Results** — show the completed fair comparison of the three saved 25-question runs.
3. **RAG Evidence** — open Q01, Q22, Q10, Q24, and Q23 to show the relationship between retrieval, context, and answer quality.
4. **Repository Analysis** — explain that current RAG is policy-text-only, then walk through the multi-file request flow.
5. **Fresh Instrumented Run** — run all three models. The dashboard automatically saves the complete result under `evaluation/results/week4_live_instrumented_<timestamp>.json`; it also provides a download button for the same evidence.

## Important interpretation notes

- The existing saved runs correctly show accuracy, retrieval, hallucination-proxy, and latency but do **not** contain old token/resource values.
- Fresh runs collect token counts from Ollama's response. CPU/RAM are sampled only if `psutil` can see a local Ollama process.
- GPU values require `nvidia-smi`; otherwise the dashboard displays `N/A` instead of inventing a number.
