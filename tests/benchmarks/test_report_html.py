from __future__ import annotations

from dspy_repl.benchmarks.report_html import render_html_report  # type: ignore[import-not-found]


def test_render_html_report_contains_sections(tmp_path) -> None:  # type: ignore[no-untyped-def]
    analysis = {
        "derived_rows": [
            {
                "run_id": "r1",
                "dataset": "oolong_trec",
                "engine": "sql",
                "avg_score": 0.8,
                "success_rate": 0.9,
                "avg_latency_seconds": 10.0,
                "avg_iterations": 4.0,
                "score_per_second": 0.08,
                "failure_rate": 0.1,
                "timeout_rate": 0.0,
            }
        ],
        "niah_context_rows": [],
        "insights": ["Best quality: sql ..."],
        "run_configs": [{"run_id": "r1", "dataset": "oolong_trec", "config": {"model": "test-model"}}],
        "per_sample_results": [],
        "trajectories": [],
        "per_engine_trajectory_stats": [],
    }
    output = tmp_path / "report.html"
    render_html_report(analysis, output)
    text = output.read_text(encoding="utf-8")

    assert "Benchmark Analytics Report" in text
    assert "Key Insights" in text
    assert "chart-quality" in text
    assert "metrics-table" in text
    assert "chart-niah" not in text


def test_render_html_report_handles_niah_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    analysis = {
        "derived_rows": [
            {
                "run_id": "r1",
                "dataset": "s_niah",
                "engine": "python",
                "avg_score": 0.7,
                "success_rate": 0.8,
                "avg_latency_seconds": 20.0,
                "avg_iterations": 5.0,
                "score_per_second": 0.035,
                "failure_rate": 0.1,
                "timeout_rate": 0.05,
            }
        ],
        "niah_context_rows": [
            {
                "run_id": "r1",
                "dataset": "s_niah",
                "engine": "python",
                "context_length": 8192,
                "avg_score": 0.9,
                "success_rate": 1.0,
                "avg_latency_seconds": 2.0,
                "avg_iterations": 2.0,
            }
        ],
        "insights": ["test insight"],
        "run_configs": [{"run_id": "r1", "dataset": "s_niah", "config": {"model": "test-model"}}],
        "per_sample_results": [],
        "trajectories": [],
        "per_engine_trajectory_stats": [],
    }
    output = tmp_path / "report_niah.html"
    render_html_report(analysis, output)
    text = output.read_text(encoding="utf-8")
    assert "S-NIAH Score Degradation by Context Length" in text
