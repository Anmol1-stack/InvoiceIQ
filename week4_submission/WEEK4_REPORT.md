# InvoiceIQ — Week 4 Evaluation Report

## Scope and reproducibility

This report evaluates the **same InvoiceIQ application** built in Week 3. No existing application, prompt, knowledge-base, chunk, embedding, service, or UI file was edited for this submission supplement.

Evidence sources are the existing evaluation set at `evaluation/questions.json` and the completed 25-question runs in `evaluation/results/`. The results below can be recalculated by `analyze_saved_results.py` when Python is available.

## Exercise 1 — Models compared

All three models use the same prompt in `scripts/llm_service.py`, the same `nomic-embed-text` query embedding, 25 stored knowledge-base chunks, cosine-similarity retrieval, threshold (0.60), top-4 context, and application API.

| Model | Stored result file | Questions |
| --- | --- | ---: |
| Qwen 2.5 Coder 1.5B Instruct | `qwen2.5_coder_1.5b_instruct.json` | 25 |
| LLaMA 3.2 3B | `llama3.2_3b.json` | 25 |
| DeepSeek Coder 1.3B | `deepseek_coder_1.3b.json` | 25 |

## Exercise 2 — Evaluation dataset

The existing dataset has 25 representative InvoiceIQ questions. It covers payment dates and late payment (Q01–Q02, Q11–Q12, Q18, Q23–Q24), disputes (Q03–Q04, Q13–Q14, Q19), termination/surviving obligations (Q05–Q10, Q15–Q17, Q20–Q22, Q25), and deliberately unanswerable/underspecified questions (Q23–Q25). The exact same question IDs were run against every model.

## Exercise 3 — Metrics and quantitative results

### Metric definitions

| Metric | Calculation |
| --- | --- |
| Correctness / accuracy | Percentage of 25 answers meeting the question-specific policy rubric in `analyze_saved_results.py`. The rubric requires core facts such as the deadline, amount, condition, and negation where applicable. |
| Answer relevance | Percentage of answers satisfying the same core-answer rubric. For this tightly scoped factual set, a response that lacks the core policy fact is not relevant enough to count. |
| Retrieval quality: Recall@4 | Percentage of questions where the expected policy file appears among the four returned chunks. |
| Retrieval quality: MRR@4 | Mean of `1 / rank` for the first returned chunk from the expected policy file; zero if it is absent. |
| Top-1 similarity | Mean cosine similarity of the first returned chunk. It measures semantic retrieval confidence, not answer correctness. |
| Unsupported/contradictory answer rate | Percentage of questions with retrieved context and a non-empty answer that fails the policy rubric. This is a reproducible proxy for hallucination/contradiction; it is conservative and should be manually audited for formal research. |
| Response latency | Wall-clock seconds recorded around each `/ask` API request. Average, 95th percentile, minimum, and maximum are reported. |
| Test-pass rate | Not applicable: InvoiceIQ answers policy questions; it does not generate executable code or tests. |
| Token usage and CPU/GPU/RAM | Not available in the existing stored result files. The current LLM service returns only answer/model, and the historic runs did not sample Ollama or Docker resource counters. Values are therefore **not invented**. |

### Results from the existing completed runs

| Model | Accuracy / relevance | Recall@4 | MRR@4 | Mean top-1 similarity | Unsupported / contradictory rate | Avg latency | P95 latency | Min–max latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek Coder 1.3B | 76.0% (19/25) | 100.0% | 0.9400 | 0.7955 | 24.0% (6/25) | 1.530 s | 2.216 s | 0.454–2.515 s |
| Qwen 2.5 Coder 1.5B | 64.0% (16/25) | 100.0% | 0.9400 | 0.7955 | 36.0% (9/25) | 2.088 s | 2.653 s | 1.102–4.374 s |
| LLaMA 3.2 3B | 64.0% (16/25) | 100.0% | 0.9400 | 0.7955 | 36.0% (9/25) | 2.437 s | 3.407 s | 1.039–4.458 s |

Retrieval values are identical because the retrieval system does not change with the answer model. The query embeddings, chunk store, similarity calculation, threshold, and top-4 limit remain fixed, as required for a fair model comparison.

### Resource/token instrumentation in the dashboard for a fresh live run

The new **Week 4 — Final Submission** dashboard page performs the following during the same 25-question run:

1. It returns Ollama's `prompt_eval_count`, `eval_count`, and inference timings through the existing API response.
2. When Streamlit is run locally, it samples the Ollama process CPU/RAM with `psutil`; NVIDIA GPU utilisation/memory is sampled through `nvidia-smi` when available.
3. It stores per-question samples alongside each answer and lets the user download a JSON evidence file. Do not compare runs made under different hardware load.

## Exercise 4 — Evidence-based analysis

DeepSeek Coder is the best model in the existing evidence: it has the highest factual accuracy (76.0%), the lowest unsupported/contradictory-answer rate (24.0%), and the lowest average and P95 latency (1.530 s and 2.216 s). It is therefore both more accurate and faster in this test set.

Qwen and LLaMA tie on accuracy (64.0%) and unsupported-answer rate (36.0%), but Qwen is faster: 2.088 s average versus 2.437 s, and 2.653 s P95 versus 3.407 s. For the observed data, Qwen is the better choice of those two.

This is **not** evidence that DeepSeek uses fewer resources: no CPU/GPU/RAM or token measurements were recorded. It only establishes a quality–latency result on this host and this data. The 100% retrieval recall and identical 0.7955 mean top-1 similarity also demonstrate that answer quality is not determined by retrieval alone; the answer model can still omit or contradict retrieved facts.

## Exercise 5 — RAG-pipeline analysis

See `RAG_CASE_STUDIES.md`. The selected cases explicitly record the question, retrieved evidence, answer, and analysis of retrieval quality → context quality → response quality.

## Exercise 6 — Repository/codebase understanding

See `REPOSITORY_ANALYSIS.md`. The current RAG index contains only policy text, so it cannot answer multi-file implementation questions. A manual static tracing exercise identifies the actual application call path and shows the exact limitation to address in a later code-aware RAG iteration.

## Conclusion

Week 3's RAG application is retained and Week 4 has a comparative dataset, three model runs, quantitative analysis, RAG evidence, and a repository-understanding assessment. The only metrics that cannot honestly be reported from current evidence are token and resource consumption; the collection method above makes that gap explicit and reproducible.
