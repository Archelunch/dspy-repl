from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from dspy_repl.benchmarks.analytics import analyze_runs, discover_run_dirs
from dspy_repl.benchmarks.report_html import render_html_report


def _parse_run_dirs(raw: str | None) -> list[Path]:
    if not raw:
        return []
    return [Path(value.strip()) for value in raw.split(",") if value.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate benchmark analytics HTML report.")
    parser.add_argument("--run-dir", default=None, help="Path to one benchmark run directory.")
    parser.add_argument("--run-dirs", default=None, help="Comma-separated benchmark run directories.")
    parser.add_argument("--latest", type=int, default=None, help="Use latest N run directories under --base-dir.")
    parser.add_argument("--base-dir", default="benchmark_results", help="Base benchmark results directory.")
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=40,
        help="Maximum number of trajectory files to embed into report for trajectory viewer.",
    )
    parser.add_argument("--output", default=None, help="Output HTML file path.")
    return parser


def resolve_input_run_dirs(args: argparse.Namespace) -> list[Path]:
    run_dirs: list[Path] = []
    if args.run_dir:
        run_dirs.append(Path(args.run_dir))
    run_dirs.extend(_parse_run_dirs(args.run_dirs))
    if args.latest is not None:
        run_dirs.extend(discover_run_dirs(Path(args.base_dir), latest=args.latest))

    # Preserve order, remove duplicates.
    unique: list[Path] = []
    seen: set[str] = set()
    for run_dir in run_dirs:
        key = str(run_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(run_dir)
    return unique


def default_output_path(base_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base_dir / "reports" / f"report_{stamp}.html"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_dirs = resolve_input_run_dirs(args)
    if not run_dirs:
        parser.error("No run directories selected. Use --run-dir, --run-dirs, or --latest.")

    analysis = analyze_runs(run_dirs, max_trajectories=max(0, int(args.max_trajectories)))
    output = Path(args.output) if args.output else default_output_path(Path(args.base_dir))
    report_path = render_html_report(analysis, output)
    print(f"HTML report saved to: {report_path}")


if __name__ == "__main__":
    main()
