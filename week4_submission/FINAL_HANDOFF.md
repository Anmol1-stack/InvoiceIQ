# Week 4 final handoff

This project is ready to zip and send. The only item that cannot be included in a ZIP is the locally installed Ollama models; your teammate already has those models.

## Your teammate's final run

1. Extract the project ZIP.
2. Start Ollama and confirm all four entries appear in `ollama list`:
   - `nomic-embed-text`
   - `qwen2.5-coder:1.5b-instruct`
   - `llama3.2:3b`
   - `deepseek-coder:1.3b`
3. From the project root, run `docker compose up --build -d`.
4. Start the dashboard: `py -3.11 -m streamlit run frontend\app.py`.
5. Open the URL printed by Streamlit, normally `http://localhost:8501`.
6. Open **Week 4 — Final Submission** in the sidebar.
7. Show the evaluator the first four tabs: Overview, Model Results, RAG Evidence, and Repository Analysis.
8. In **Fresh Instrumented Run**, click **Run All 3 Models — 75 Instrumented Questions** and wait for completion.
9. Confirm the saved file message. It is automatically written to `evaluation/results/week4_live_instrumented_<timestamp>.json`.
10. Use the displayed fresh-run comparison table and downloaded JSON as final evidence for Exercise 3 and Exercise 4.

## What is already in the ZIP

- The unchanged Week 3 InvoiceIQ RAG application.
- The 25-question evaluation dataset.
- Three prior model result files and a verified comparison table.
- The complete six-exercise Week 4 report and RAG/repository analysis.
- The separate, presentation-ready Week 4 dashboard page.
- Live token, latency, CPU/RAM, and optional GPU measurement support.

The final live run is intentionally performed on the teammate's machine, because its Ollama models and hardware determine the token/resource measurements.
