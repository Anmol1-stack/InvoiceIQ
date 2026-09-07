# Week 4 submission supplement

This folder was added without modifying the Week 3 application or existing Week 4 implementation. It turns the existing model runs into a reproducible, clearly scoped submission pack.

Contents:

- `WEEK4_REPORT.md` — responses to all six exercises and the quantified results.
- `RAG_CASE_STUDIES.md` — question → retrieved context → model-response analysis.
- `REPOSITORY_ANALYSIS.md` — the repository-level code-understanding experiment.
- `analyze_saved_results.py` — standard-library-only analysis of the existing saved model responses.
- `saved_results_metrics.json` — dashboard-ready summary of the completed legacy runs.
- `DASHBOARD_GUIDE.md` — exact steps for presenting the new separate Week 4 UI.
- `FINAL_HANDOFF.md` — teammate checklist for the final model run after unzipping.

To regenerate the metrics from the repository root:

```powershell
py week4_submission/analyze_saved_results.py --write
```

The saved data does not contain Ollama token counters or container CPU/GPU/RAM samples. Those metrics are therefore explicitly recorded as unavailable rather than invented. The report states the instrumentation needed for a future live re-run.
