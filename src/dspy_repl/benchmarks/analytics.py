from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def discover_run_dirs(base_dir: Path, latest: int | None = None) -> list[Path]:
    if not base_dir.exists():
        return []
    run_dirs = sorted([path for path in base_dir.iterdir() if path.is_dir() and (path / "summary.json").exists()])
    if latest is None or latest <= 0:
        return run_dirs
    return run_dirs[-latest:]


def load_run_artifacts(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {summary_path}")

    payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "summary": _read_json(summary_path),
        "warnings": [],
    }

    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        payload["results"] = _read_jsonl(results_path)
    else:
        payload["results"] = []
        payload["warnings"].append("results.jsonl is missing")

    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        payload["run_config"] = _read_json(run_config_path)
    else:
        payload["run_config"] = {}
        payload["warnings"].append("run_config.json is missing")

    per_engine_path = run_dir / "per_engine_trajectory_stats.json"
    if per_engine_path.exists():
        payload["per_engine_trajectory_stats"] = _read_json(per_engine_path)
    else:
        payload["per_engine_trajectory_stats"] = {}
        payload["warnings"].append("per_engine_trajectory_stats.json is missing")

    niah_context_path = run_dir / "summary_by_context_length.json"
    if niah_context_path.exists():
        payload["summary_by_context_length"] = _read_json(niah_context_path)
    else:
        payload["summary_by_context_length"] = {"rows": []}

    return payload


def normalize_run_rows(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary = run_payload["summary"]
    trajectory_stats: dict[str, Any] = run_payload.get("per_engine_trajectory_stats", {})
    run_id = str(summary.get("run_id") or run_payload.get("run_id") or "unknown")
    dataset = str(summary.get("dataset", "unknown"))
    rows: list[dict[str, Any]] = []

    for row in summary.get("rows", []):
        engine = str(row.get("engine", "unknown"))
        engine_traj = trajectory_stats.get(engine, {})
        normalized = {
            "run_id": run_id,
            "run_dir": run_payload.get("run_dir"),
            "dataset": dataset,
            "engine": engine,
            "success_rate": float(row.get("success_rate", 0.0)),
            "num_success": int(row.get("num_success", 0)),
            "num_total": int(row.get("num_total", 0)),
            "avg_score": float(row.get("avg_score", 0.0) or 0.0),
            "avg_latency_seconds": float(row.get("avg_latency_seconds", 0.0) or 0.0),
            "avg_iterations": float(row.get("avg_iterations", 0.0) or 0.0),
            "failed_samples": int(engine_traj.get("failed_samples", 0)),
            "timeout_samples": int(engine_traj.get("timeout_samples", 0)),
            "avg_error_steps": float(engine_traj.get("avg_error_steps", 0.0)),
            "warnings": list(run_payload.get("warnings", [])),
        }
        rows.append(normalized)
    return rows


def normalize_niah_context_rows(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary = run_payload.get("summary_by_context_length", {})
    run_id = str(run_payload.get("summary", {}).get("run_id") or run_payload.get("run_id") or "unknown")
    dataset = str(run_payload.get("summary", {}).get("dataset", "unknown"))
    rows: list[dict[str, Any]] = []
    for row in summary.get("rows", []):
        rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "engine": str(row.get("engine", "unknown")),
                "context_length": int(row.get("context_length", 0)),
                "success_rate": float(row.get("success_rate", 0.0)),
                "avg_score": float(row.get("avg_score", 0.0)),
                "avg_latency_seconds": float(row.get("avg_latency_seconds", 0.0)),
                "avg_iterations": float(row.get("avg_iterations", 0.0)),
            }
        )
    return rows


def derive_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    derived: list[dict[str, Any]] = []
    for row in rows:
        latency = float(row.get("avg_latency_seconds", 0.0))
        iterations = float(row.get("avg_iterations", 0.0))
        score = float(row.get("avg_score", 0.0))
        value = dict(row)
        value["score_per_second"] = (score / latency) if latency > 0 else 0.0
        value["score_per_iteration"] = (score / iterations) if iterations > 0 else 0.0
        total = int(row.get("num_total", 0))
        failures = int(row.get("failed_samples", 0))
        timeouts = int(row.get("timeout_samples", 0))
        value["failure_rate"] = (failures / total) if total > 0 else 0.0
        value["timeout_rate"] = (timeouts / total) if total > 0 else 0.0
        derived.append(value)
    return derived


def _normalize_per_sample_results(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary = run_payload.get("summary", {})
    run_id = str(summary.get("run_id") or run_payload.get("run_id") or "unknown")
    dataset = str(summary.get("dataset", "unknown"))
    results = run_payload.get("results", [])
    rows: list[dict[str, Any]] = []
    for row in results:
        rows.append(
            {
                "run_id": run_id,
                "run_dir": str(run_payload.get("run_dir", "")),
                "dataset": dataset,
                "engine": str(row.get("engine", "unknown")),
                "sample_id": str(row.get("sample_id", "")),
                "task_name": str(row.get("task_name", "")),
                "answer": row.get("answer"),
                "expected": row.get("expected"),
                "score": float(row.get("score", 0.0) or 0.0),
                "iterations": int(row.get("iterations", 0) or 0),
                "elapsed_seconds": float(row.get("elapsed_seconds", 0.0) or 0.0),
                "success": bool(row.get("success", False)),
                "error": row.get("error"),
                "trajectory_path": row.get("trajectory_path"),
                "trajectory_diagnostics": row.get("trajectory_diagnostics", {}),
            }
        )
    return rows


def _normalize_per_engine_trajectory_stats(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary = run_payload.get("summary", {})
    run_id = str(summary.get("run_id") or run_payload.get("run_id") or "unknown")
    dataset = str(summary.get("dataset", "unknown"))
    stats_by_engine = run_payload.get("per_engine_trajectory_stats", {})
    rows: list[dict[str, Any]] = []
    for engine, value in stats_by_engine.items():
        stats = dict(value) if isinstance(value, dict) else {}
        rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "engine": str(engine),
                "total_samples": int(stats.get("total_samples", 0) or 0),
                "failed_samples": int(stats.get("failed_samples", 0) or 0),
                "timeout_samples": int(stats.get("timeout_samples", 0) or 0),
                "avg_steps": float(stats.get("avg_steps", 0.0) or 0.0),
                "avg_llm_call_mentions": float(stats.get("avg_llm_call_mentions", 0.0) or 0.0),
                "avg_sql_statement_estimate": float(stats.get("avg_sql_statement_estimate", 0.0) or 0.0),
                "avg_error_steps": float(stats.get("avg_error_steps", 0.0) or 0.0),
                "avg_code_chars": float(stats.get("avg_code_chars", 0.0) or 0.0),
                "avg_output_chars": float(stats.get("avg_output_chars", 0.0) or 0.0),
                "avg_latency_seconds_success_only": float(stats.get("avg_latency_seconds_success_only", 0.0) or 0.0),
                "avg_iterations_success_only": float(stats.get("avg_iterations_success_only", 0.0) or 0.0),
            }
        )
    return sorted(rows, key=lambda row: (row["run_id"], row["engine"]))


def _load_selected_trajectories(run_payload: dict[str, Any], max_trajectories: int = 40) -> list[dict[str, Any]]:
    run_dir = Path(str(run_payload.get("run_dir", "")))
    candidates = _normalize_per_sample_results(run_payload)

    def _interestingness(row: dict[str, Any]) -> tuple[int, int, int]:
        diagnostics = row.get("trajectory_diagnostics", {}) or {}
        error_steps = int(diagnostics.get("error_steps", 0) or 0)
        steps = int(diagnostics.get("steps", 0) or 0)
        not_success = 0 if row.get("success") else 1
        has_error = 1 if row.get("error") else 0
        return (
            not_success * 3 + has_error * 2 + (1 if error_steps > 0 else 0),
            error_steps,
            steps,
        )

    selected = sorted(candidates, key=_interestingness, reverse=True)[:max_trajectories]
    output: list[dict[str, Any]] = []

    for row in selected:
        rel_path = row.get("trajectory_path")
        if not rel_path:
            continue
        trajectory_file = run_dir / str(rel_path)
        if not trajectory_file.exists():
            continue
        try:
            payload = _read_json(trajectory_file)
        except json.JSONDecodeError:
            continue
        output.append(
            {
                "run_id": row.get("run_id"),
                "dataset": row.get("dataset"),
                "engine": row.get("engine"),
                "sample_id": row.get("sample_id"),
                "task_name": row.get("task_name"),
                "success": row.get("success"),
                "score": row.get("score"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "error": row.get("error"),
                "trajectory": payload.get("trajectory", []),
            }
        )

    return output


def generate_insights(derived_rows: list[dict[str, Any]]) -> list[str]:
    if not derived_rows:
        return ["No benchmark rows were found in the selected run directories."]

    insights: list[str] = []
    best_score = max(derived_rows, key=lambda row: (row.get("avg_score", 0.0), row.get("success_rate", 0.0)))
    insights.append(
        f"Best quality: {best_score['engine']} on {best_score['dataset']} (score={best_score['avg_score']:.3f}, success={best_score['success_rate']:.1%})."
    )

    fastest = [row for row in derived_rows if row.get("success_rate", 0.0) > 0]
    if fastest:
        fastest_row = min(fastest, key=lambda row: row.get("avg_latency_seconds", 0.0) or 10**9)
        insights.append(
            f"Fastest successful engine: {fastest_row['engine']} on {fastest_row['dataset']} ({fastest_row['avg_latency_seconds']:.2f}s average latency)."
        )

    tradeoff = max(
        derived_rows,
        key=lambda row: (row.get("avg_score", 0.0) - (row.get("avg_latency_seconds", 0.0) / 1000.0)),
    )
    insights.append(
        f"Best quality-latency tradeoff: {tradeoff['engine']} on {tradeoff['dataset']} (score_per_second={tradeoff['score_per_second']:.4f})."
    )

    unstable = [
        row
        for row in derived_rows
        if row.get("timeout_rate", 0.0) > 0.10
        or row.get("failure_rate", 0.0) > 0.20
        or row.get("avg_error_steps", 0.0) > 1.0
    ]
    if unstable:
        names = ", ".join(sorted({f"{row['engine']}@{row['dataset']}" for row in unstable}))
        insights.append(f"Stability warning: high failure/timeout/error-step rates detected for {names}.")

    if any(row.get("warnings") for row in derived_rows):
        insights.append("Some runs have partial artifacts; report sections were generated with graceful fallbacks.")

    return insights


def analyze_runs(run_dirs: list[Path], max_trajectories: int = 40) -> dict[str, Any]:
    runs = [load_run_artifacts(path) for path in run_dirs]
    normalized_rows: list[dict[str, Any]] = []
    niah_context_rows: list[dict[str, Any]] = []
    run_configs: list[dict[str, Any]] = []
    per_sample_results: list[dict[str, Any]] = []
    per_engine_trajectory_stats: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for run in runs:
        normalized_rows.extend(normalize_run_rows(run))
        niah_context_rows.extend(normalize_niah_context_rows(run))
        run_configs.append(
            {
                "run_id": run.get("run_id"),
                "dataset": run.get("summary", {}).get("dataset", "unknown"),
                "config": run.get("run_config", {}),
            }
        )
        per_sample_results.extend(_normalize_per_sample_results(run))
        per_engine_trajectory_stats.extend(_normalize_per_engine_trajectory_stats(run))
        trajectories.extend(_load_selected_trajectories(run, max_trajectories=max_trajectories))

    derived_rows = derive_metrics(normalized_rows)
    insights = generate_insights(derived_rows)
    return {
        "run_dirs": [str(path) for path in run_dirs],
        "rows": normalized_rows,
        "derived_rows": derived_rows,
        "niah_context_rows": niah_context_rows,
        "run_configs": run_configs,
        "per_sample_results": per_sample_results,
        "per_engine_trajectory_stats": per_engine_trajectory_stats,
        "trajectories": trajectories,
        "insights": insights,
    }
