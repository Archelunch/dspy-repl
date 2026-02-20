from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

ENGINE_COLORS: dict[str, str] = {
    "python": "#14b8a6",
    "js": "#f59e0b",
    "haskell": "#8b5cf6",
    "scheme": "#f43f5e",
    "sql": "#0ea5e9",
}


def _engine_color(engine: str) -> str:
    return ENGINE_COLORS.get(engine, "#64748b")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return escape(str(value))


def _aggregate_by_engine(derived_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in derived_rows:
        grouped.setdefault(str(row["engine"]), []).append(row)

    result: list[dict[str, Any]] = []
    for engine, rows in grouped.items():
        count = len(rows)
        result.append(
            {
                "engine": engine,
                "avg_score": sum(float(row.get("avg_score", 0.0)) for row in rows) / count if count else 0.0,
                "success_rate": sum(float(row.get("success_rate", 0.0)) for row in rows) / count if count else 0.0,
                "avg_latency_seconds": sum(float(row.get("avg_latency_seconds", 0.0)) for row in rows) / count
                if count
                else 0.0,
                "score_per_second": sum(float(row.get("score_per_second", 0.0)) for row in rows) / count
                if count
                else 0.0,
            }
        )
    return sorted(result, key=lambda row: row["engine"])


def _build_table_html(derived_rows: list[dict[str, Any]]) -> str:
    headers = [
        ("run_id", "Run"),
        ("dataset", "Dataset"),
        ("engine", "Engine"),
        ("avg_score", "Avg Score"),
        ("success_rate", "Success Rate"),
        ("avg_latency_seconds", "Avg Latency (s)"),
        ("avg_iterations", "Avg Iterations"),
        ("score_per_second", "Score/sec"),
        ("failure_rate", "Failure Rate"),
        ("timeout_rate", "Timeout Rate"),
    ]

    head = "".join(
        f'<th onclick="sortTable(\'{key}\')" data-key="{key}" scope="col">{label}</th>' for key, label in headers
    )
    body_rows: list[str] = []
    for row in derived_rows:
        score = float(row.get("avg_score", 0.0))
        success_rate = float(row.get("success_rate", 0.0))
        score_class = "metric-good" if score >= 0.8 else "metric-warn" if score >= 0.5 else "metric-bad"
        success_class = "metric-good" if success_rate >= 0.8 else "metric-warn" if success_rate >= 0.5 else "metric-bad"
        body_rows.append(
            "<tr class='metrics-row'>"
            f"<td>{_safe_text(row.get('run_id', ''))}</td>"
            f"<td>{_safe_text(row.get('dataset', ''))}</td>"
            f"<td><span class='engine-dot' style='background:{_engine_color(str(row.get('engine', '')))}'></span>{_safe_text(row.get('engine', ''))}</td>"
            f"<td class='{score_class}'>{score:.4f}</td>"
            f"<td class='{success_class}'>{success_rate:.2%}</td>"
            f"<td>{float(row.get('avg_latency_seconds', 0.0)):.2f}</td>"
            f"<td>{float(row.get('avg_iterations', 0.0)):.2f}</td>"
            f"<td>{float(row.get('score_per_second', 0.0)):.4f}</td>"
            f"<td>{float(row.get('failure_rate', 0.0)):.2%}</td>"
            f"<td>{float(row.get('timeout_rate', 0.0)):.2%}</td>"
            "</tr>"
        )
    return f"""
<div class="table-wrap">
<table id="metrics-table" class="data-table">
  <thead><tr>{head}</tr></thead>
  <tbody>
    {"".join(body_rows)}
  </tbody>
</table>
</div>
"""


def _build_run_configs_html(run_configs: list[dict[str, Any]]) -> str:
    if not run_configs:
        return "<p class='muted'>No run configuration artifacts were found.</p>"

    cards: list[str] = []
    for row in run_configs:
        config = row.get("config", {}) or {}
        engines = ", ".join(str(value) for value in config.get("engines", []))
        cards.append(
            f"""
<details class="run-config-card">
  <summary>
    <span>{_safe_text(row.get("run_id", "unknown"))}</span>
    <span class="dataset-pill">{_safe_text(row.get("dataset", "unknown"))}</span>
  </summary>
  <div class="run-config-grid">
    <div><strong>Model</strong><div>{_safe_text(config.get("model", "n/a"))}</div></div>
    <div><strong>Temperature</strong><div>{_safe_text(config.get("temperature", "n/a"))}</div></div>
    <div><strong>Max Iterations</strong><div>{_safe_text(config.get("max_iterations", "n/a"))}</div></div>
    <div><strong>Timeout (s)</strong><div>{_safe_text(config.get("engine_timeout_seconds", "n/a"))}</div></div>
    <div><strong>Max LLM Calls</strong><div>{_safe_text(config.get("max_llm_calls", "n/a"))}</div></div>
    <div><strong>Max Samples</strong><div>{_safe_text(config.get("max_samples", "n/a"))}</div></div>
  </div>
  <div class="engines-row"><strong>Engines:</strong> {_safe_text(engines or "n/a")}</div>
</details>
"""
        )
    return "".join(cards)


def _build_trajectory_stats_html(stats_rows: list[dict[str, Any]]) -> str:
    if not stats_rows:
        return "<p class='muted'>No per-engine trajectory statistics found.</p>"

    header = (
        "<tr>"
        "<th>Run</th><th>Dataset</th><th>Engine</th><th>Avg Steps</th><th>Avg Error Steps</th>"
        "<th>Avg Code Chars</th><th>Avg Output Chars</th><th>LLM Mentions</th></tr>"
    )
    body: list[str] = []
    for row in stats_rows:
        body.append(
            "<tr>"
            f"<td>{_safe_text(row.get('run_id', ''))}</td>"
            f"<td>{_safe_text(row.get('dataset', ''))}</td>"
            f"<td><span class='engine-dot' style='background:{_engine_color(str(row.get('engine', '')))}'></span>{_safe_text(row.get('engine', ''))}</td>"
            f"<td>{float(row.get('avg_steps', 0.0)):.2f}</td>"
            f"<td>{float(row.get('avg_error_steps', 0.0)):.2f}</td>"
            f"<td>{float(row.get('avg_code_chars', 0.0)):.0f}</td>"
            f"<td>{float(row.get('avg_output_chars', 0.0)):.0f}</td>"
            f"<td>{float(row.get('avg_llm_call_mentions', 0.0)):.2f}</td>"
            "</tr>"
        )
    return f"""
<div class="table-wrap">
<table class="data-table">
  <thead>{header}</thead>
  <tbody>{"".join(body)}</tbody>
</table>
</div>
"""


def _build_per_sample_results_html(
    per_sample_results: list[dict[str, Any]], trajectories: list[dict[str, Any]], engines: list[str]
) -> str:
    if not per_sample_results:
        return "<p class='muted'>No per-sample rows were found.</p>"

    trajectory_by_key: dict[str, dict[str, Any]] = {}
    for row in trajectories:
        key = f"{row.get('run_id')}::{row.get('engine')}::{row.get('sample_id')}"
        trajectory_by_key[key] = row

    filters = "".join(f"<option value='{_safe_text(engine)}'>{_safe_text(engine)}</option>" for engine in engines)
    rows: list[str] = []
    for index, row in enumerate(per_sample_results):
        score = float(row.get("score", 0.0) or 0.0)
        status_badge = (
            "<span class='status-badge ok'>success</span>"
            if row.get("success")
            else "<span class='status-badge bad'>failed</span>"
        )
        key = f"{row.get('run_id')}::{row.get('engine')}::{row.get('sample_id')}"
        trajectory = trajectory_by_key.get(key, {})
        steps = trajectory.get("trajectory", []) if isinstance(trajectory.get("trajectory", []), list) else []
        detail_id = f"traj-{index}"
        preview = ""
        if steps:
            preview = _safe_text(str(steps[0].get("reasoning", "")))[:160]
            if len(preview) == 160:
                preview += "..."
        steps_html: list[str] = []
        for step_index, step in enumerate(steps, start=1):
            steps_html.append(
                f"""
<div class="step-block">
  <h5>Step {step_index}</h5>
  <div class="step-section">
    <div class="step-label">Reasoning</div>
    <blockquote>{_safe_text(step.get("reasoning", ""))}</blockquote>
  </div>
  <div class="step-section">
    <div class="step-label">Code</div>
    <pre>{_safe_text(step.get("code", ""))}</pre>
  </div>
  <div class="step-section">
    <div class="step-label">Output</div>
    <pre>{_safe_text(step.get("output", ""))}</pre>
  </div>
</div>
"""
            )

        details_html = (
            f"<details id='{detail_id}'><summary>View trajectory ({len(steps)} steps)</summary>{''.join(steps_html)}</details>"
            if steps
            else "<div class='muted'>Trajectory not included in report payload.</div>"
        )
        rows.append(
            f"""
<tr class="sample-row" data-engine="{_safe_text(row.get("engine", ""))}" data-status="{"success" if row.get("success") else "failed"}">
  <td>{_safe_text(row.get("run_id", ""))}</td>
  <td>{_safe_text(row.get("dataset", ""))}</td>
  <td><span class="engine-dot" style="background:{_engine_color(str(row.get("engine", "")))}"></span>{_safe_text(row.get("engine", ""))}</td>
  <td>{_safe_text(row.get("sample_id", ""))}</td>
  <td>{score:.3f}</td>
  <td>{float(row.get("elapsed_seconds", 0.0)):.2f}</td>
  <td>{int(row.get("iterations", 0))}</td>
  <td>{status_badge}</td>
  <td>{_safe_text(row.get("error", "") or "")}</td>
  <td>{_safe_text(preview)}</td>
</tr>
<tr class="sample-detail-row" data-engine="{_safe_text(row.get("engine", ""))}" data-status="{"success" if row.get("success") else "failed"}">
  <td colspan="10">{details_html}</td>
</tr>
"""
        )
    return f"""
<div class="sample-controls">
  <label for="engine-filter">Engine</label>
  <select id="engine-filter" onchange="filterSamples()">
    <option value="">All</option>
    {filters}
  </select>
  <label for="status-filter">Status</label>
  <select id="status-filter" onchange="filterSamples()">
    <option value="">All</option>
    <option value="success">Success</option>
    <option value="failed">Failed</option>
  </select>
</div>
<div class="table-wrap">
<table id="samples-table" class="data-table">
  <thead>
    <tr>
      <th>Run</th><th>Dataset</th><th>Engine</th><th>Sample</th><th>Score</th><th>Latency (s)</th><th>Iterations</th><th>Status</th><th>Error</th><th>Trajectory Preview</th>
    </tr>
  </thead>
  <tbody>{"".join(rows)}</tbody>
</table>
</div>
"""


def _js_array(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _render_charts(derived_rows: list[dict[str, Any]], niah_context_rows: list[dict[str, Any]]) -> str:
    by_engine = _aggregate_by_engine(derived_rows)
    engines = [row["engine"] for row in by_engine]
    avg_scores = [row["avg_score"] for row in by_engine]
    success_rates = [row["success_rate"] for row in by_engine]
    avg_latency = [row["avg_latency_seconds"] for row in by_engine]
    failure_rates = [float(row.get("failure_rate", 0.0)) for row in by_engine]
    timeout_rates = [float(row.get("timeout_rate", 0.0)) for row in by_engine]
    per_engine_colors = [_engine_color(engine) for engine in engines]

    scatter_x = [float(row.get("avg_latency_seconds", 0.0)) for row in derived_rows]
    scatter_y = [float(row.get("avg_score", 0.0)) for row in derived_rows]
    scatter_text = [f"{row.get('run_id')} | {row.get('dataset')} | {row.get('engine')}" for row in derived_rows]
    scatter_colors = [_engine_color(str(row.get("engine", ""))) for row in derived_rows]

    niah_by_engine: dict[str, list[dict[str, Any]]] = {}
    for row in niah_context_rows:
        niah_by_engine.setdefault(str(row.get("engine", "unknown")), []).append(row)

    niah_traces = []
    for engine, rows in sorted(niah_by_engine.items()):
        rows_sorted = sorted(rows, key=lambda value: int(value.get("context_length", 0)))
        niah_traces.append(
            {
                "x": [int(value.get("context_length", 0)) for value in rows_sorted],
                "y": [float(value.get("avg_score", 0.0)) for value in rows_sorted],
                "mode": "lines+markers",
                "name": engine,
                "line": {"color": _engine_color(engine)},
                "marker": {"color": _engine_color(engine)},
            }
        )

    max_latency = max(avg_latency) if avg_latency else 1.0
    radar_quality = avg_scores
    radar_success = success_rates
    radar_speed = [1.0 - (value / max_latency if max_latency > 0 else 0.0) for value in avg_latency]
    radar_stability = [1.0 - value for value in failure_rates]
    radar_timeout = [1.0 - value for value in timeout_rates]
    niah_script = ""
    if niah_traces:
        niah_script = f"""
  const niahTraces = {_js_array(niah_traces)};
  Plotly.newPlot('chart-niah', niahTraces, {{
    ...darkLayout,
    title: 'S-NIAH Score Degradation by Context Length',
    xaxis: {{ title: 'Context Length (tokens)' }},
    yaxis: {{ title: 'Average Score' }}
  }});
"""

    return f"""
<script>
  const engines = {_js_array(engines)};
  const avgScores = {_js_array(avg_scores)};
  const successRates = {_js_array(success_rates)};
  const avgLatency = {_js_array(avg_latency)};
  const failureRates = {_js_array(failure_rates)};
  const timeoutRates = {_js_array(timeout_rates)};
  const engineColors = {_js_array(per_engine_colors)};
  const radarQuality = {_js_array(radar_quality)};
  const radarSuccess = {_js_array(radar_success)};
  const radarSpeed = {_js_array(radar_speed)};
  const radarStability = {_js_array(radar_stability)};
  const radarTimeout = {_js_array(radar_timeout)};

  const darkLayout = {{
    paper_bgcolor: '#0f172a',
    plot_bgcolor: '#1e293b',
    font: {{ color: '#e2e8f0' }},
    margin: {{ l: 50, r: 24, t: 48, b: 48 }},
    xaxis: {{ gridcolor: '#334155', zerolinecolor: '#334155' }},
    yaxis: {{ gridcolor: '#334155', zerolinecolor: '#334155' }}
  }};

  Plotly.newPlot('chart-quality', [
    {{ type: 'bar', x: engines, y: avgScores, name: 'Avg Score', marker: {{ color: engineColors }} }},
    {{ type: 'bar', x: engines, y: successRates, name: 'Success Rate', marker: {{ color: '#94a3b8' }} }}
  ], {{
    ...darkLayout,
    title: 'Quality Overview by Engine',
    barmode: 'group',
    yaxis: {{ ...darkLayout.yaxis, tickformat: '.0%' }}
  }});

  const latencyOrder = avgLatency
    .map((value, idx) => [value, engines[idx], engineColors[idx]])
    .sort((a, b) => a[0] - b[0]);

  Plotly.newPlot('chart-latency', [{{
    type: 'bar',
    orientation: 'h',
    y: latencyOrder.map(v => v[1]),
    x: latencyOrder.map(v => v[0]),
    text: latencyOrder.map(v => `${{v[0].toFixed(2)}}s`),
    textposition: 'outside',
    marker: {{ color: latencyOrder.map(v => v[2]) }},
    name: 'Avg Latency'
  }}], {{
    ...darkLayout,
    title: 'Average Latency by Engine (Lower is Better)',
    xaxis: {{ ...darkLayout.xaxis, title: 'Latency (s)' }},
    yaxis: {{ ...darkLayout.yaxis, automargin: true }}
  }});

  Plotly.newPlot('chart-frontier', [{{
    type: 'scatter',
    mode: 'markers',
    x: {_js_array(scatter_x)},
    y: {_js_array(scatter_y)},
    text: {_js_array(scatter_text)},
    marker: {{ size: 10, color: {_js_array(scatter_colors)} }},
    hovertemplate: '%{{text}}<br>latency=%{{x:.2f}}<br>score=%{{y:.3f}}<extra></extra>'
  }}], {{
    ...darkLayout,
    title: 'Score vs Latency (Efficiency Frontier)',
    xaxis: {{ title: 'Average Latency (s)' }},
    yaxis: {{ title: 'Average Score' }}
  }});

  Plotly.newPlot('chart-radar', engines.map((engine, i) => {{
    return {{
      type: 'scatterpolar',
      r: [radarQuality[i], radarSuccess[i], radarSpeed[i], radarStability[i], radarTimeout[i]],
      theta: ['Quality', 'Success', 'Speed', 'Stability', 'Timeout Robustness'],
      fill: 'toself',
      opacity: 0.45,
      name: engine,
      line: {{ color: engineColors[i] }}
    }};
  }}), {{
    ...darkLayout,
    title: 'Engine Performance Profile',
    polar: {{
      bgcolor: '#1e293b',
      radialaxis: {{
        visible: true,
        range: [0, 1],
        gridcolor: '#334155',
        linecolor: '#334155'
      }},
      angularaxis: {{
        gridcolor: '#334155',
        linecolor: '#334155'
      }}
    }}
  }});
  {niah_script}
</script>
"""


def render_html_report(analysis: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    insights = analysis.get("insights", [])
    derived_rows = analysis.get("derived_rows", [])
    niah_context_rows = analysis.get("niah_context_rows", [])
    run_configs = analysis.get("run_configs", [])
    per_sample_results = analysis.get("per_sample_results", [])
    trajectories = analysis.get("trajectories", [])
    trajectory_stats_rows = analysis.get("per_engine_trajectory_stats", [])

    top = max(derived_rows, key=lambda row: float(row.get("avg_score", 0.0)), default=None)
    avg_success = (
        sum(float(row.get("success_rate", 0.0)) for row in derived_rows) / len(derived_rows) if derived_rows else 0.0
    )
    avg_latency = (
        sum(float(row.get("avg_latency_seconds", 0.0)) for row in derived_rows) / len(derived_rows)
        if derived_rows
        else 0.0
    )
    all_engines = sorted({str(row.get("engine", "")) for row in derived_rows if row.get("engine")})
    total_rows = len(derived_rows)

    insight_html = "".join(f"<div class='insight-item'>{_safe_text(text)}</div>" for text in insights)
    table_html = _build_table_html(derived_rows)
    run_config_html = _build_run_configs_html(run_configs)
    trajectory_stats_html = _build_trajectory_stats_html(trajectory_stats_rows)
    per_sample_html = _build_per_sample_results_html(per_sample_results, trajectories, all_engines)
    charts_js = _render_charts(derived_rows, niah_context_rows)
    niah_chart_div = "<div id='chart-niah' class='chart'></div>" if niah_context_rows else ""

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Benchmark Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --surface: #1e293b;
      --surface-soft: #172033;
      --border: #334155;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --ok: #22c55e;
      --bad: #ef4444;
      --warn: #f59e0b;
      --python: #14b8a6;
      --js: #f59e0b;
      --haskell: #8b5cf6;
      --scheme: #f43f5e;
      --sql: #0ea5e9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 0;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    .container {{ max-width: 1480px; margin: 0 auto; padding: 24px; }}
    h1, h2, h3 {{ margin: 0 0 10px 0; }}
    h1 {{ font-size: 30px; }}
    h2 {{ margin-top: 28px; font-size: 20px; }}
    p, .muted {{ color: var(--muted); }}
    .header {{ display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 12px; margin-bottom: 10px; }}
    .header-meta {{ font-size: 13px; color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0 20px; }}
    .card {{
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      background: linear-gradient(180deg, var(--surface), var(--surface-soft));
      box-shadow: 0 3px 12px rgba(2, 6, 23, 0.24);
    }}
    .card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }}
    .card .value {{ margin-top: 8px; font-size: 26px; font-weight: 700; }}
    .insights-grid {{ display: grid; gap: 8px; }}
    .insight-item {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      background: rgba(30, 41, 59, 0.7);
    }}
    .engine-dot {{ width: 10px; height: 10px; display: inline-block; border-radius: 9999px; margin-right: 8px; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 14px; }}
    .chart {{ height: 360px; border: 1px solid var(--border); border-radius: 14px; background: var(--surface); }}
    .table-wrap {{ border: 1px solid var(--border); border-radius: 14px; overflow: auto; background: var(--surface); }}
    .data-table {{ border-collapse: collapse; width: 100%; min-width: 980px; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid var(--border); padding: 10px; text-align: left; font-size: 13px; vertical-align: top; }}
    .data-table th {{ position: sticky; top: 0; cursor: pointer; background: #111b2d; z-index: 1; }}
    .metric-good {{ color: var(--ok); font-weight: 700; }}
    .metric-warn {{ color: var(--warn); font-weight: 700; }}
    .metric-bad {{ color: var(--bad); font-weight: 700; }}
    .run-config-card {{
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-bottom: 10px;
      background: var(--surface);
    }}
    .run-config-card summary {{
      list-style: none;
      cursor: pointer;
      padding: 10px 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 600;
    }}
    .dataset-pill {{
      border: 1px solid var(--border);
      border-radius: 9999px;
      padding: 2px 8px;
      font-size: 12px;
      color: var(--muted);
    }}
    .run-config-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; padding: 0 12px 10px; }}
    .engines-row {{ padding: 0 12px 12px; color: var(--muted); }}
    .sample-controls {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin: 8px 0 12px;
      flex-wrap: wrap;
    }}
    .sample-controls label {{ color: var(--muted); font-size: 13px; }}
    .sample-controls select {{
      background: var(--surface);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 8px;
    }}
    .status-badge {{ border-radius: 9999px; font-size: 12px; padding: 2px 8px; font-weight: 700; text-transform: uppercase; }}
    .status-badge.ok {{ background: rgba(34, 197, 94, 0.18); color: #86efac; }}
    .status-badge.bad {{ background: rgba(239, 68, 68, 0.18); color: #fca5a5; }}
    .sample-detail-row > td {{ background: #121d31; }}
    .step-block {{ border: 1px solid var(--border); border-radius: 10px; margin: 8px 0; padding: 10px; }}
    .step-block h5 {{ margin: 0 0 8px; font-size: 13px; color: var(--muted); }}
    .step-section {{ margin-bottom: 8px; }}
    .step-label {{ font-size: 12px; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }}
    blockquote {{ margin: 0; border-left: 3px solid var(--border); padding: 8px 10px; background: #111827; }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #0b1220;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      font-size: 12px;
      color: #cbd5e1;
    }}
  </style>
</head>
<body>
  <div class="container">
  <div class="header">
    <div>
      <h1>Benchmark Analytics Report</h1>
      <p>Generated: {generated_at}</p>
    </div>
    <div class="header-meta">Runs: {_safe_text(", ".join(sorted({str(row.get("run_id", "")) for row in derived_rows})) or "n/a")}</div>
  </div>

  <div class="cards">
    <div class="card"><div class="label">Rows analyzed</div><div class="value">{total_rows}</div></div>
    <div class="card"><div class="label">Average success rate</div><div class="value">{avg_success:.2%}</div></div>
    <div class="card"><div class="label">Average latency</div><div class="value">{avg_latency:.2f}s</div></div>
    <div class="card"><div class="label">Top score</div><div class="value">{_safe_text((top["engine"] + " @ " + top["dataset"]) if top else "n/a")}</div></div>
  </div>

  <h2>Run Configuration</h2>
  {run_config_html}

  <h2>Key Insights</h2>
  <div class="insights-grid">{insight_html}</div>

  <h2>Engine Comparison</h2>
  {table_html}

  <h2>Charts</h2>
  <div class="chart-grid">
    <div id="chart-quality" class="chart"></div>
    <div id="chart-latency" class="chart"></div>
    <div id="chart-frontier" class="chart"></div>
    <div id="chart-radar" class="chart"></div>
    {niah_chart_div}
  </div>

  <h2>Trajectory Statistics</h2>
  {trajectory_stats_html}

  <h2>Per-Sample Results and Trajectories</h2>
  {per_sample_html}

  <script>
    function sortTable(key) {{
      const table = document.getElementById('metrics-table');
      if (!table) return;
      const headers = Array.from(table.querySelectorAll('thead th'));
      const idx = headers.findIndex(h => h.dataset.key === key);
      if (idx < 0) return;
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      const asc = !(table.dataset.sortKey === key && table.dataset.sortDir === 'asc');
      rows.sort((a, b) => {{
        const va = a.children[idx].innerText;
        const vb = b.children[idx].innerText;
        const na = parseFloat(va.replace('%',''));
        const nb = parseFloat(vb.replace('%',''));
        if (!Number.isNaN(na) && !Number.isNaN(nb)) return asc ? na - nb : nb - na;
        return asc ? va.localeCompare(vb) : vb.localeCompare(va);
      }});
      rows.forEach(r => tbody.appendChild(r));
      table.dataset.sortKey = key;
      table.dataset.sortDir = asc ? 'asc' : 'desc';
    }}

    function filterSamples() {{
      const engine = document.getElementById('engine-filter')?.value || '';
      const status = document.getElementById('status-filter')?.value || '';
      const rows = Array.from(document.querySelectorAll('#samples-table tbody tr'));
      rows.forEach((row) => {{
        const rowEngine = row.dataset.engine || '';
        const rowStatus = row.dataset.status || '';
        const showEngine = !engine || rowEngine === engine;
        const showStatus = !status || rowStatus === status;
        row.style.display = showEngine && showStatus ? '' : 'none';
      }});
    }}
  </script>
  {charts_js}
</div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
