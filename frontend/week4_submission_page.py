"""Separate Week 4 evidence page for the InvoiceIQ Streamlit dashboard."""

import json
import shutil
import statistics
import subprocess
import threading
import time
from datetime import datetime
from html import escape

import streamlit as st

try:
    import psutil
except ImportError:  # Resource sampling remains optional for existing installs.
    psutil = None


MODELS = {
    "Qwen 2.5 Coder 1.5B": "qwen2.5-coder:1.5b-instruct",
    "LLaMA 3.2 3B": "llama3.2:3b",
    "DeepSeek Coder 1.3B": "deepseek-coder:1.3b",
}

EXPECTED_SOURCES = {
    "Q01": "payment_policy.txt", "Q02": "payment_policy.txt", "Q03": "payment_policy.txt",
    "Q04": "payment_policy.txt", "Q05": "termination_policy.txt", "Q06": "termination_policy.txt",
    "Q07": "termination_policy.txt", "Q08": "termination_policy.txt", "Q09": "termination_policy.txt",
    "Q10": "termination_policy.txt", "Q11": "payment_policy.txt", "Q12": "payment_policy.txt",
    "Q13": "payment_policy.txt", "Q14": "payment_policy.txt", "Q15": "termination_policy.txt",
    "Q16": "termination_policy.txt", "Q17": "termination_policy.txt", "Q18": "payment_policy.txt",
    "Q19": "payment_policy.txt", "Q20": "termination_policy.txt", "Q21": "termination_policy.txt",
    "Q22": "termination_policy.txt", "Q23": "payment_policy.txt", "Q24": "payment_policy.txt",
    "Q25": "termination_policy.txt",
}


def _read_json(path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def score_policy_answer(question_id, answer):
    """Transparent factual rubric shared by the Week 4 written report."""
    text = (answer or "").lower().strip()
    if "no relevant information" in text or "not found in the knowledge base" in text:
        return False

    if question_id in {"Q01", "Q11", "Q12"}:
        return ("30" in text or "thirty" in text) and "invoice" in text
    if question_id == "Q02":
        return ("interest" in text or "charge" in text) and "late" in text
    if question_id in {"Q03", "Q13"}:
        return ("10" in text or "ten" in text) and "business" in text
    if question_id in {"Q04", "Q19"}:
        return "undisputed" in text and ("pay" in text or "payable" in text)
    if question_id == "Q14":
        return (
            "undisputed" in text
            and ("pay" in text or "payable" in text)
            and ("no" in text[:50] or "cannot" in text)
        )
    if question_id in {"Q05", "Q15"}:
        return ("30" in text or "thirty" in text) and "notice" in text
    if question_id in {"Q06", "Q20"}:
        return all(term in text for term in ("immediate", "termination")) and (
            "15" in text or "fifteen" in text
        ) and ("breach" in text or "correct" in text)
    if question_id == "Q07":
        return ("no" in text[:50] or "does not" in text or "remain" in text) and "payment" in text
    if question_id == "Q08":
        return ("no" in text[:50] or "does not" in text or "continue" in text) and "confidential" in text
    if question_id == "Q09":
        return ("no" in text[:50] or "does not" in text or "continue" in text) and (
            "intellectual" in text or "ip" in text
        )
    if question_id in {"Q10", "Q17"}:
        return "confidential" in text and ("intellectual" in text or "ip" in text) and "payment" in text
    if question_id == "Q16":
        return ("15" in text or "fifteen" in text) and "day" in text
    if question_id == "Q18":
        return "july 1" in text or (("30" in text or "thirty" in text) and "june" in text)
    if question_id == "Q21":
        return ("no" in text[:50] or "cannot" in text or "remain" in text) and "payment" in text
    if question_id == "Q22":
        return ("no" in text[:50] or "cannot" in text) and "notice" in text and (
            "breach" in text or "immediate" in text
        )
    if question_id in {"Q23", "Q24"}:
        return any(term in text for term in ("not specified", "not stated", "no specific", "no mention"))
    if question_id == "Q25":
        return ("30" in text or "thirty" in text) and any(
            term in text for term in ("no maximum", "not stated", "does not state", "maximum")
        )
    return False


def _table_html(rows):
    """Render high-contrast evidence tables instead of Streamlit's dark grid."""
    if not rows:
        return '<div class="week4-empty">No evidence is available yet.</div>'
    columns = list(rows[0].keys())
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{escape(str(row.get(column, '—')))}</td>"
            for column in columns
        ) + "</tr>"
        for row in rows
    )
    return f'<div class="week4-table-wrap"><table class="week4-table"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


class _OllamaResourceSampler:
    """Records local Ollama CPU/RAM and host GPU samples during one request."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None
        self.gpu_available = bool(shutil.which("nvidia-smi"))

    def _take_sample(self):
        if psutil is None:
            return

        cpu = memory = 0.0
        ollama_detected = False
        for process in psutil.process_iter(["name", "exe"]):
            try:
                name = (process.info.get("name") or "").lower()
                executable = (process.info.get("exe") or "").lower()
                if "ollama" in name or "ollama" in executable:
                    ollama_detected = True
                    cpu += process.cpu_percent(interval=None)
                    memory += process.memory_info().rss / (1024 * 1024)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue

        sample = {
            "ollama_detected": ollama_detected,
            "cpu_percent": round(cpu, 2),
            "memory_mb": round(memory, 2),
            "gpu_utilization_percent": None,
            "gpu_memory_mb": None,
        }

        if self.gpu_available:
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                values = [line.split(",") for line in completed.stdout.splitlines() if "," in line]
                if values:
                    sample["gpu_utilization_percent"] = round(sum(float(x[0].strip()) for x in values), 2)
                    sample["gpu_memory_mb"] = round(sum(float(x[1].strip()) for x in values), 2)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass

        self.samples.append(sample)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self._take_sample()

    def start(self):
        self._take_sample()  # Prime psutil's per-process CPU counter.
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        self._take_sample()

        if psutil is None:
            return {"status": "psutil is not installed"}
        found = [row for row in self.samples if row["ollama_detected"]]
        if not found:
            return {"status": "Ollama process was not detected by the dashboard host"}

        def numeric(field):
            return [row[field] for row in found if row[field] is not None]

        cpu = numeric("cpu_percent")
        memory = numeric("memory_mb")
        gpu = numeric("gpu_utilization_percent")
        gpu_memory = numeric("gpu_memory_mb")
        return {
            "status": "Collected",
            "samples": len(found),
            "average_ollama_cpu_percent": round(statistics.mean(cpu), 2),
            "peak_ollama_memory_mb": round(max(memory), 2),
            "average_host_gpu_utilization_percent": round(statistics.mean(gpu), 2) if gpu else None,
            "peak_host_gpu_memory_mb": round(max(gpu_memory), 2) if gpu_memory else None,
        }


def _run_question(question, model_name, app_url, post_json):
    sampler = _OllamaResourceSampler()
    sampler.start()
    started = time.perf_counter()
    try:
        response = post_json(app_url, {"question": question["question"], "model": MODELS[model_name]})
    finally:
        latency = time.perf_counter() - started
        resources = sampler.stop()

    documents = response.get("retrieved_documents", [])
    llm_metrics = response.get("llm_metrics", {})
    answer = response.get("answer", "")
    return {
        "id": question["id"],
        "question": question["question"],
        "expected": question["expected"],
        "model": model_name,
        "answer": answer,
        "correct": score_policy_answer(question["id"], answer),
        "latency_seconds": round(latency, 3),
        "top1_similarity": round(documents[0].get("similarity", 0), 4) if documents else 0,
        "retrieved_documents": documents,
        "ollama_metrics": llm_metrics,
        "resource_metrics": resources,
    }


def _run_model(model_name, questions, app_url, post_json):
    results = []
    progress = st.progress(0, text=f"Running {model_name}: 0/{len(questions)}")
    for index, question in enumerate(questions, start=1):
        try:
            results.append(_run_question(question, model_name, app_url, post_json))
        except Exception as error:
            results.append({
                "id": question["id"], "question": question["question"], "expected": question["expected"],
                "model": model_name, "answer": f"Evaluation error: {error}", "correct": False,
                "latency_seconds": 0, "top1_similarity": 0, "retrieved_documents": [],
                "ollama_metrics": {}, "resource_metrics": {"status": "No response"},
            })
        progress.progress(index / len(questions), text=f"Running {model_name}: {index}/{len(questions)}")
    progress.empty()
    return results


def _summary(model_name, rows):
    latencies = [row["latency_seconds"] for row in rows]
    similarities = [row["top1_similarity"] for row in rows if row["retrieved_documents"]]
    source_ranks = []
    for row in rows:
        expected_source = EXPECTED_SOURCES.get(row["id"])
        source_ranks.append(next(
            (
                index for index, document in enumerate(row["retrieved_documents"], start=1)
                if document.get("source") == expected_source
            ),
            None,
        ))
    tokens_in = [row["ollama_metrics"].get("prompt_tokens") for row in rows if row["ollama_metrics"].get("prompt_tokens") is not None]
    tokens_out = [row["ollama_metrics"].get("completion_tokens") for row in rows if row["ollama_metrics"].get("completion_tokens") is not None]
    cpu = [row["resource_metrics"].get("average_ollama_cpu_percent") for row in rows if row["resource_metrics"].get("average_ollama_cpu_percent") is not None]
    memory = [row["resource_metrics"].get("peak_ollama_memory_mb") for row in rows if row["resource_metrics"].get("peak_ollama_memory_mb") is not None]
    gpu = [row["resource_metrics"].get("average_host_gpu_utilization_percent") for row in rows if row["resource_metrics"].get("average_host_gpu_utilization_percent") is not None]
    gpu_memory = [row["resource_metrics"].get("peak_host_gpu_memory_mb") for row in rows if row["resource_metrics"].get("peak_host_gpu_memory_mb") is not None]
    correct = sum(row["correct"] for row in rows)
    retrieved_expected = sum(rank is not None for rank in source_ranks)
    unsupported = sum(
        bool(row["retrieved_documents"] and row["answer"].strip() and not row["correct"])
        for row in rows
    )
    return {
        "Model": model_name,
        "Accuracy / relevance": f"{100 * correct / len(rows):.1f}%",
        "Recall@4": f"{100 * retrieved_expected / len(rows):.1f}%",
        "MRR@4": f"{statistics.mean(1 / rank if rank else 0 for rank in source_ranks):.4f}",
        "Avg top-1 similarity": f"{statistics.mean(similarities):.4f}" if similarities else "N/A",
        "Unsupported rate": f"{100 * unsupported / len(rows):.1f}%",
        "Avg latency": f"{statistics.mean(latencies):.3f}s",
        "P95 latency": f"{sorted(latencies)[max(0, round(0.95 * len(latencies)) - 1)]:.3f}s",
        "Avg prompt tokens": f"{statistics.mean(tokens_in):.1f}" if tokens_in else "N/A",
        "Avg completion tokens": f"{statistics.mean(tokens_out):.1f}" if tokens_out else "N/A",
        "Avg Ollama CPU": f"{statistics.mean(cpu):.2f}%" if cpu else "N/A",
        "Peak Ollama RAM": f"{max(memory):.2f} MB" if memory else "N/A",
        "Avg host GPU": f"{statistics.mean(gpu):.2f}%" if gpu else "N/A",
        "Peak host GPU RAM": f"{max(gpu_memory):.2f} MB" if gpu_memory else "N/A",
    }


def _save_live_evidence(project_root, results, timestamp):
    """Persist a self-contained evidence file so a fresh run is not lost on rerun."""
    results_dir = project_root / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    file_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"week4_live_instrumented_{file_stamp}.json"
    summaries = [
        _summary(model_name, rows)
        for model_name, rows in results.items()
        if rows
    ]
    payload = {
        "run_metadata": {
            "run_at": timestamp,
            "questions_per_model": 25,
            "models": MODELS,
            "evaluation_conditions": "Same InvoiceIQ RAG pipeline, knowledge base, embeddings, retrieval threshold, prompt, and questions; only the LLM changes."
        },
        "summary": summaries,
        "question_level_results": results
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def render_week4_submission(app_url, project_root, post_json):
    """Render the separate, presentation-ready Week 4 dashboard page."""
    submission_dir = project_root / "week4_submission"
    baseline = _read_json(submission_dir / "saved_results_metrics.json", [])
    questions = _read_json(project_root / "evaluation" / "questions.json", [])
    if "week4_final_results" not in st.session_state:
        st.session_state.week4_final_results = {}
    if "week4_final_last_run" not in st.session_state:
        st.session_state.week4_final_last_run = None
    if "week4_final_saved_file" not in st.session_state:
        st.session_state.week4_final_saved_file = None

    st.markdown(
        """
        <style>
        .week4-table-wrap { overflow-x: auto; margin: 0.7rem 0 1.5rem; border: 1px solid #d8e1ef; border-radius: 14px; background: #ffffff; }
        .week4-table { width: 100%; border-collapse: collapse; min-width: 720px; font-size: 14px; color: #172033; }
        .week4-table th { background: #173d7a; color: #ffffff; text-align: left; padding: 13px 15px; font-weight: 700; }
        .week4-table td { padding: 12px 15px; border-top: 1px solid #e5eaf2; vertical-align: top; line-height: 1.4; }
        .week4-table tr:nth-child(even) td { background: #f6f9fd; }
        .week4-table tr:hover td { background: #e9f2ff; }
        .week4-empty { color: #52616b; padding: 1rem; border: 1px dashed #bcc9d9; border-radius: 12px; background: #fff; }
        .week4-note { background: #edf5ff; border-left: 5px solid #2563eb; padding: 14px 16px; border-radius: 8px; color: #172033; margin: 0.7rem 0 1rem; }
        .main .stMarkdown, .main .stMarkdown p, .main .stMarkdown li, .main .stMarkdown h1, .main .stMarkdown h2, .main .stMarkdown h3 { color: #172033 !important; }
        .main [data-baseweb="tab"] { color: #334155 !important; font-weight: 700; }
        .main [data-baseweb="tab"][aria-selected="true"] { color: #1d4ed8 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Week 4 — Final Submission Evidence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">A separate dashboard for all six Week 4 exercises: '
        'fair model comparison, metrics, RAG analysis, repository analysis, and fresh resource measurement.</div>',
        unsafe_allow_html=True,
    )

    overview_tab, results_tab, rag_tab, repository_tab, fresh_tab = st.tabs([
        "Overview", "Model Results", "RAG Evidence", "Repository Analysis", "Fresh Instrumented Run"
    ])

    with overview_tab:
        st.markdown(_table_html([
            {"Exercise": "1. Compare models", "Completed evidence": "Qwen, LLaMA, DeepSeek; same RAG pipeline and questions."},
            {"Exercise": "2. Dataset", "Completed evidence": f"{len(questions)} representative policy questions."},
            {"Exercise": "3. Quantitative metrics", "Completed evidence": "Quality, retrieval, latency, live tokens, CPU/RAM/GPU."},
            {"Exercise": "4. Analyse results", "Completed evidence": "Evidence-based accuracy/latency conclusion."},
            {"Exercise": "5. Analyse RAG", "Completed evidence": "Five retrieval-context-answer case studies."},
            {"Exercise": "6. Understand repository", "Completed evidence": "Multi-file request-path trace and current limitation."},
        ]), unsafe_allow_html=True)
        st.markdown("### Metric definitions")
        st.markdown(
            "**Accuracy/relevance:** answer satisfies the policy rubric.  \n"
            "**Recall@4:** expected policy source is among four retrieved chunks.  \n"
            "**MRR@4:** reciprocal rank of the first expected-source chunk.  \n"
            "**Unsupported rate:** retrieved-context answer fails the policy rubric.  \n"
            "**Latency:** elapsed time for the full `/ask` request.  \n"
            "**Tokens/timings:** values returned by Ollama. **CPU/RAM/GPU:** local Ollama/host samples during a fresh run."
        )
        report = submission_dir / "WEEK4_REPORT.md"
        if report.exists():
            with st.expander("Open the full written Week 4 report"):
                st.markdown(report.read_text(encoding="utf-8"))

    with results_tab:
        if baseline:
            rows = [{
                "Model": item["model"], "Accuracy": f"{item['accuracy_percent']:.1f}%",
                "Relevance": f"{item['answer_relevance_percent']:.1f}%",
                "Recall@4": f"{item['retrieval_recall_at_4_percent']:.1f}%",
                "MRR@4": f"{item['retrieval_mrr_at_4']:.4f}",
                "Top-1 similarity": f"{item['mean_top1_similarity']:.4f}",
                "Unsupported rate": f"{item['unsupported_or_contradictory_rate_percent']:.1f}%",
                "Avg latency": f"{item['average_latency_seconds']:.3f}s",
                "P95 latency": f"{item['p95_latency_seconds']:.3f}s",
            } for item in baseline]
            st.markdown(_table_html(rows), unsafe_allow_html=True)
            st.success("DeepSeek Coder has the highest recorded accuracy (76%) and lowest recorded average latency (1.530 s).")
        st.info("Legacy runs did not retain token or hardware samples. The fresh-run tab records those values without changing the RAG conditions.")

    with rag_tab:
        for title, evidence in [
            ("Q01 — Relevant retrieval, correct answer", "payment_policy.txt is ranked first (0.7916) and states 30 days from invoice date; DeepSeek returned that exact policy fact."),
            ("Q22 — Correct context, wrong conclusion", "termination_policy.txt is ranked first (0.8303) and requires an uncorrected material breach for immediate termination, but all three models answered ‘Yes’ to immediate termination without a breach."),
            ("Q10 — Fact retrieved but omitted", "Retrieved termination/contract chunks include confidentiality, intellectual property, and prior-due payments; DeepSeek omitted the payment obligation."),
            ("Q24 — Partial noise / unknown detail", "The policy says methods are specified by the provider but lists none. Invoice-policy chunks add topical but non-answering context."),
            ("Q23 — Grounded abstention", "Relevant payment chunks contain no penalty percentage; DeepSeek correctly said the context does not mention one."),
        ]:
            with st.expander(title):
                st.write(evidence)
        case_studies = submission_dir / "RAG_CASE_STUDIES.md"
        if case_studies.exists():
            with st.expander("Open detailed RAG case studies"):
                st.markdown(case_studies.read_text(encoding="utf-8"))

    with repository_tab:
        st.warning("Current RAG indexes policy text only—not Python/YAML code—so it cannot reliably answer repository questions by itself.")
        st.markdown(_table_html([
            {"Step": 1, "File": "frontend/app.py", "Role": "Streamlit sends question to application service :8000."},
            {"Step": 2, "File": "scripts/app_service.py", "Role": "Forwards request to orchestrator :8003."},
            {"Step": 3, "File": "scripts/orchestrator.py", "Role": "Calls retrieval, builds context, then calls LLM."},
            {"Step": 4, "File": "scripts/retrieval_service.py", "Role": "Embeds question and ranks stored chunks by cosine similarity."},
            {"Step": 5, "File": "scripts/llm_service.py", "Role": "Sends prompt/context to local Ollama and exposes metrics."},
            {"Step": 6, "File": "docker-compose.yml", "Role": "Defines service names and ports connecting the components."},
        ]), unsafe_allow_html=True)
        st.markdown("**Next step:** create a separate code index with file paths, symbols, imports, callers, and line-range metadata, then combine it with static dependency analysis.")
        analysis = submission_dir / "REPOSITORY_ANALYSIS.md"
        if analysis.exists():
            with st.expander("Open detailed repository analysis"):
                st.markdown(analysis.read_text(encoding="utf-8"))

    with fresh_tab:
        st.caption("A full run issues 25 identical-condition requests per model (75 for all three). It may take several minutes.")
        if psutil is None:
            st.warning("Install `frontend/requirements.txt` before a fresh run to collect CPU/RAM samples.")
        elif not shutil.which("nvidia-smi"):
            st.info("CPU/RAM and Ollama token counts will be collected; GPU values remain N/A because nvidia-smi is unavailable.")
        else:
            st.success("Fresh runs will collect Ollama token/timing values and CPU/RAM/GPU samples.")

        all_column, one_column = st.columns(2)
        with all_column:
            run_all = st.button("Run All 3 Models — 75 Instrumented Questions", type="primary", use_container_width=True)
        with one_column:
            selected_model = st.selectbox("Run one model", list(MODELS.keys()), key="week4_final_model")
            run_one = st.button("Run Selected Model — 25 Questions", use_container_width=True)

        if run_all:
            for model_name in MODELS:
                with st.expander(f"Running {model_name}", expanded=True):
                    st.session_state.week4_final_results[model_name] = _run_model(model_name, questions, app_url, post_json)
            st.session_state.week4_final_last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.week4_final_saved_file = _save_live_evidence(
                project_root,
                st.session_state.week4_final_results,
                st.session_state.week4_final_last_run
            )
            st.success("Completed and saved 75 instrumented model-question runs.")
        elif run_one:
            st.session_state.week4_final_results[selected_model] = _run_model(selected_model, questions, app_url, post_json)
            st.session_state.week4_final_last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.week4_final_saved_file = _save_live_evidence(
                project_root,
                st.session_state.week4_final_results,
                st.session_state.week4_final_last_run
            )
            st.success(f"Completed and saved the instrumented run for {selected_model}.")

        summaries = [_summary(model, rows) for model, rows in st.session_state.week4_final_results.items() if rows]
        if summaries:
            st.markdown("### Fresh-run comparison")
            st.markdown(_table_html(summaries), unsafe_allow_html=True)
            st.caption(f"Last fresh run: {st.session_state.week4_final_last_run}")
            if st.session_state.week4_final_saved_file:
                st.caption(
                    "Saved evidence: "
                    f"evaluation/results/{st.session_state.week4_final_saved_file.name}"
                )
            st.download_button(
                "Download fresh Week 4 evidence (JSON)",
                data=json.dumps(st.session_state.week4_final_results, indent=2),
                file_name="invoiceiq_week4_instrumented_results.json",
                mime="application/json",
                use_container_width=True,
            )
            with st.expander("Inspect question-level evidence"):
                inspect_model = st.selectbox("Model", list(st.session_state.week4_final_results), key="week4_final_inspect")
                for item in st.session_state.week4_final_results[inspect_model]:
                    label = "CORRECT" if item["correct"] else "INCORRECT"
                    with st.expander(f"{item['id']} — {label} — {item['question']}"):
                        st.write("**Expected:**", item["expected"])
                        st.write("**Answer:**", item["answer"])
                        st.json({
                            "latency_seconds": item["latency_seconds"],
                            "top1_similarity": item["top1_similarity"],
                            "ollama_metrics": item["ollama_metrics"],
                            "resource_metrics": item["resource_metrics"],
                        })
