from __future__ import annotations

import json
import shutil
import sys
import threading
import time
import warnings
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, cast

from dspy_repl import SQLRLM, HaskellRLM, JavaScriptRLM, SchemeRLM
from dspy_repl.benchmarks.config import BenchmarkConfig, Language
from dspy_repl.benchmarks.logging_utils import log_event, setup_benchmark_logger, setup_runtime_logger
from dspy_repl.compat import dspy

ScoreFn = Callable[[str, str], float]
TaskLoaderFn = Callable[[BenchmarkConfig], list[dict[str, Any]]]
_PYDANTIC_SERIALIZER_WARNING_RE = r".*Pydantic serializer warnings:.*"


@dataclass(frozen=True)
class TaskResult:
    sample_id: str
    task_name: str
    engine: Language
    answer: str
    iterations: int
    elapsed_seconds: float
    success: bool
    expected: str | None = None
    score: float | None = None
    trajectory: list[dict[str, Any]] | None = None
    sql_profile: dict[str, float] | None = None
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_name": self.task_name,
            "engine": self.engine,
            "answer": self.answer,
            "expected": self.expected,
            "score": self.score,
            "iterations": self.iterations,
            "elapsed_seconds": self.elapsed_seconds,
            "success": self.success,
            "error": self.error,
            "sql_profile": self.sql_profile,
            "trajectory": self.trajectory or [],
        }


def _ensure_sibling_dspy_package() -> None:
    """
    Add sibling adapters/dspy directory to import path in local monorepo checkouts.
    """
    for parent in Path(__file__).resolve().parents:
        dspy_root = parent / "dspy"
        if (dspy_root / "benchmarks" / "reporting.py").exists():
            if str(dspy_root) not in sys.path:
                sys.path.insert(0, str(dspy_root))
            return


def _artifact_helpers() -> tuple[
    Callable[[], str],
    Callable[[str, str], Path],
    Callable[[Path, dict[str, Any], list[dict[str, Any]]], dict[str, Any]],
]:
    _ensure_sibling_dspy_package()
    from benchmarks.reporting import make_run_dir, make_run_id, write_run_artifacts  # type: ignore[import-not-found]

    return make_run_id, make_run_dir, write_run_artifacts


def _sanitize_task_for_worker(task: dict[str, Any]) -> dict[str, Any]:
    # score_fn callables are not reliably picklable across process boundaries.
    return {
        "id": task.get("id"),
        "name": task.get("name"),
        "signature": task.get("signature"),
        "inputs": task.get("inputs"),
        "expected": task.get("expected"),
        "description": task.get("description"),
    }


def _check_prerequisites() -> dict[Language, bool]:
    return {
        "python": shutil.which("deno") is not None,
        "scheme": shutil.which("guile") is not None,
        "haskell": shutil.which("ghci") is not None,
        "sql": True,
        "js": shutil.which("node") is not None,
    }


def _normalize_trajectory(raw_trajectory: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_trajectory, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw_trajectory:
        payload = _to_jsonable(item)
        if isinstance(payload, dict):
            normalized.append(payload)
        else:
            normalized.append({"value": payload})
    return normalized


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        with _suppress_known_pydantic_serializer_warning():
            try:
                dumped = value.model_dump(mode="json")  # type: ignore[call-arg]
                return _to_jsonable(dumped)
            except Exception:
                return str(value)
    return str(value)


@contextmanager
def _suppress_known_pydantic_serializer_warning() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_PYDANTIC_SERIALIZER_WARNING_RE,
            category=UserWarning,
        )
        yield


def resolve_runnable_languages(requested: tuple[Language, ...], logger) -> list[Language]:
    available = _check_prerequisites()
    runnable: list[Language] = []
    for language in requested:
        if available.get(language, False):
            runnable.append(language)
            continue
        tool_name = {"python": "deno", "scheme": "guile", "haskell": "ghci", "sql": "sqlite", "js": "node"}[language]
        log_event(logger, "language_skipped_missing_dependency", language=language, dependency=tool_name)
    if not runnable:
        raise RuntimeError("No runnable languages remain after prerequisite checks.")
    return runnable


def _configure_lm(config: BenchmarkConfig) -> None:
    lm = dspy.LM(  # type: ignore[attr-defined]
        config.model.model,
        temperature=config.model.temperature,
        max_tokens=config.model.max_tokens,
    )
    dspy.configure(lm=lm)  # type: ignore[attr-defined]


def _build_engine(language: Language, signature: str, config: BenchmarkConfig) -> Any:
    if language == "python":
        return dspy.RLM(  # type: ignore[attr-defined]
            signature,
            max_iterations=config.run.max_iterations,
            max_llm_calls=config.run.max_llm_calls,
            verbose=config.run.verbose,
        )
    if language == "scheme":
        return SchemeRLM(
            signature,
            max_iterations=config.run.max_iterations,
            max_llm_calls=config.run.max_llm_calls,
            verbose=config.run.verbose,
        )
    if language == "haskell":
        return HaskellRLM(
            signature,
            max_iterations=config.run.max_iterations,
            max_llm_calls=config.run.max_llm_calls,
            verbose=config.run.verbose,
        )
    if language == "sql":
        return SQLRLM(
            signature,
            max_iterations=config.run.max_iterations,
            max_llm_calls=config.run.max_llm_calls,
            verbose=config.run.verbose,
        )
    if language == "js":
        return JavaScriptRLM(
            signature,
            max_iterations=config.run.max_iterations,
            max_llm_calls=config.run.max_llm_calls,
            verbose=config.run.verbose,
        )
    raise ValueError(f"Unknown language: {language}")


def _run_engine_with_timeout(
    *,
    engine: Any,
    inputs: dict[str, Any],
    timeout_seconds: int,
    language: Language,
    sample_id: str,
) -> Any:
    if timeout_seconds <= 0:
        return engine(**inputs)

    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}
    finished = threading.Event()

    def _invoke() -> None:
        try:
            result_box["result"] = engine(**inputs)
        except BaseException as exc:  # pragma: no cover - depends on runtime errors
            error_box["error"] = exc
        finally:
            finished.set()

    worker = threading.Thread(target=_invoke, name=f"engine-runner-{language}-{sample_id}", daemon=True)
    worker.start()
    if not finished.wait(timeout_seconds):
        shutdown_fn = getattr(engine, "shutdown", None)
        if callable(shutdown_fn):
            try:
                shutdown_fn()
            except Exception:
                pass
        raise TimeoutError(f"Language '{language}' timed out after {timeout_seconds}s on sample '{sample_id}'.")
    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("result")


def execute_task(
    *,
    language: Language,
    task: dict[str, Any],
    config: BenchmarkConfig,
    score_fn: ScoreFn | None = None,
) -> TaskResult:
    sample_id = str(task.get("id", task.get("name", "unknown")))
    task_name = str(task.get("name", sample_id))
    signature = str(task["signature"])
    inputs = cast(dict[str, Any], task["inputs"])
    expected = task.get("expected")
    try:
        with _suppress_known_pydantic_serializer_warning():
            engine = _build_engine(language, signature, config)
            start = time.time()
            result = _run_engine_with_timeout(
                engine=engine,
                inputs=inputs,
                timeout_seconds=config.run.engine_timeout_seconds,
                language=language,
                sample_id=sample_id,
            )
        elapsed_seconds = round(time.time() - start, 2)

        output_field = next(iter(dspy.Signature(signature).output_fields.keys()))  # type: ignore[attr-defined]
        answer = getattr(result, output_field, str(result))
        trajectory = _normalize_trajectory(getattr(result, "trajectory", []))

        score: float | None = None
        if expected is not None and score_fn is not None:
            score = score_fn(str(answer), str(expected))

        sql_profile = getattr(engine, "last_sql_profile", None) if language == "sql" else None
        return TaskResult(
            sample_id=sample_id,
            task_name=task_name,
            engine=language,
            answer=str(answer),
            iterations=len(trajectory),
            elapsed_seconds=elapsed_seconds,
            success=True,
            expected=str(expected) if expected is not None else None,
            score=score,
            trajectory=trajectory,
            sql_profile=sql_profile if isinstance(sql_profile, dict) else None,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime failures
        elapsed_seconds = float(config.run.engine_timeout_seconds) if isinstance(exc, TimeoutError) else 0.0
        return TaskResult(
            sample_id=sample_id,
            task_name=task_name,
            engine=language,
            answer="",
            iterations=0,
            elapsed_seconds=elapsed_seconds,
            success=False,
            expected=str(expected) if expected is not None else None,
            score=None,
            trajectory=[],
            sql_profile=None,
            error=str(exc),
        )


def _run_task_worker(
    *,
    language: Language,
    task: dict[str, Any],
    config: BenchmarkConfig,
    score_fn: ScoreFn | None = None,
) -> TaskResult:
    setup_runtime_logger(verbose=config.run.verbose)
    _configure_lm(config)
    return execute_task(language=language, task=task, config=config, score_fn=score_fn)


def _run_task_across_languages(
    *,
    task: dict[str, Any],
    languages: list[Language],
    config: BenchmarkConfig,
    logger,
    executor: ProcessPoolExecutor | None,
    score_fn: ScoreFn | None = None,
) -> list[TaskResult]:
    if not config.parallel.enabled:
        return [execute_task(language=language, task=task, config=config, score_fn=score_fn) for language in languages]

    if executor is None:
        raise RuntimeError("Parallel execution is enabled but no process executor was provided.")

    sample_id = str(task.get("id", task.get("name", "unknown")))
    max_workers = config.parallel.max_workers if config.parallel.max_workers is not None else len(languages)
    log_event(
        logger,
        "task_parallel_dispatch",
        sample_id=sample_id,
        languages=languages,
        max_workers=max_workers,
    )

    futures_by_lang: dict[Language, Future[TaskResult]] = {
        language: executor.submit(_run_task_worker, language=language, task=task, config=config, score_fn=score_fn)
        for language in languages
    }
    ordered_results: dict[Language, TaskResult] = {}
    worker_timeout_s = float(config.run.engine_timeout_seconds) + 5.0
    for language, future in futures_by_lang.items():
        try:
            ordered_results[language] = future.result(timeout=worker_timeout_s)
        except FutureTimeoutError:
            future.cancel()
            ordered_results[language] = TaskResult(
                sample_id=sample_id,
                task_name=str(task.get("name", sample_id)),
                engine=language,
                answer="",
                iterations=0,
                elapsed_seconds=float(config.run.engine_timeout_seconds),
                success=False,
                expected=str(task.get("expected")) if task.get("expected") is not None else None,
                error=f"Worker timeout after {worker_timeout_s:.1f}s",
            )
        except Exception as exc:  # pragma: no cover - process/runtime failures
            ordered_results[language] = TaskResult(
                sample_id=sample_id,
                task_name=str(task.get("name", sample_id)),
                engine=language,
                answer="",
                iterations=0,
                elapsed_seconds=0.0,
                success=False,
                expected=str(task.get("expected")) if task.get("expected") is not None else None,
                error=f"Worker failure: {exc}",
            )
    log_event(
        logger,
        "task_parallel_completed",
        sample_id=sample_id,
        completed=len(ordered_results),
    )
    return [ordered_results[language] for language in languages if language in ordered_results]


def summarize_results(results: list[TaskResult]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[TaskResult]] = {}
    for result in results:
        grouped.setdefault(result.engine, []).append(result)
    for engine, rows in grouped.items():
        ok_rows = [row for row in rows if row.success]
        scores = [row.score for row in ok_rows if row.score is not None]
        summary[engine] = {
            "total": float(len(rows)),
            "successes": float(len(ok_rows)),
            "success_rate": (len(ok_rows) / len(rows)) if rows else 0.0,
            "avg_elapsed_seconds": (sum(row.elapsed_seconds for row in ok_rows) / len(ok_rows) if ok_rows else 0.0),
            "avg_iterations": (sum(row.iterations for row in ok_rows) / len(ok_rows) if ok_rows else 0.0),
            "avg_score": (sum(scores) / len(scores)) if scores else 0.0,
        }
    return summary


def run_benchmark(
    *,
    config: BenchmarkConfig,
    dataset_name: str,
    task_loader: TaskLoaderFn,
    score_fn: ScoreFn | None = None,
) -> Path:
    make_run_id, make_run_dir, write_run_artifacts = _artifact_helpers()
    run_id = make_run_id()
    run_dir = make_run_dir(config.artifacts.save_dir, run_id)
    logger = setup_benchmark_logger(run_dir=run_dir, verbose=config.run.verbose)

    log_event(logger, "benchmark_start", run_id=run_id, config=config.to_dict(), dataset=dataset_name)
    _configure_lm(config)
    languages = resolve_runnable_languages(config.run.languages, logger)
    tasks = [_sanitize_task_for_worker(task) for task in task_loader(config)]
    log_event(logger, "dataset_loaded", dataset=dataset_name, sample_count=len(tasks), languages=languages)

    results: list[TaskResult] = []
    incremental_path = run_dir / "incremental_results.jsonl"

    max_workers = config.parallel.max_workers if config.parallel.max_workers is not None else len(languages)
    with ProcessPoolExecutor(max_workers=max_workers) if config.parallel.enabled else nullcontext() as executor:
        process_executor = cast(ProcessPoolExecutor | None, executor)
        for task_index, task in enumerate(tasks, start=1):
            task_id = str(task.get("id", f"task_{task_index}"))
            log_event(
                logger,
                "task_start",
                task_index=task_index,
                total_tasks=len(tasks),
                sample_id=task_id,
                task_name=task.get("name"),
                languages=languages,
            )
            task_results = _run_task_across_languages(
                task=task,
                languages=languages,
                config=config,
                logger=logger,
                executor=process_executor,
                score_fn=score_fn,
            )
            for result in task_results:
                results.append(result)
                log_event(
                    logger,
                    "task_result",
                    sample_id=result.sample_id,
                    language=result.engine,
                    success=result.success,
                    elapsed_seconds=result.elapsed_seconds,
                    iterations=result.iterations,
                    score=result.score,
                    error=result.error,
                )
                if config.artifacts.incremental_save:
                    with incremental_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(result.to_record(), ensure_ascii=False) + "\n")

    run_config = {
        "run_id": run_id,
        "dataset": dataset_name,
        "dataset_name": config.dataset.dataset_name,
        "dataset_split": config.dataset.dataset_split,
        "max_samples": config.dataset.max_samples,
        "seed": config.dataset.seed,
        "sample_id": config.dataset.sample_id,
        "model": config.model.model,
        "temperature": config.model.temperature,
        "max_tokens": config.model.max_tokens,
        "engines": languages,
        "max_iterations": config.run.max_iterations,
        "max_llm_calls": config.run.max_llm_calls,
        "engine_timeout_seconds": config.run.engine_timeout_seconds,
        "verbose": config.run.verbose,
    }
    summary_payload = write_run_artifacts(run_dir, run_config, [result.to_record() for result in results])
    aggregate = summarize_results(results)
    log_event(
        logger,
        "benchmark_completed",
        run_id=run_id,
        result_count=len(results),
        summary_rows=len(summary_payload.get("rows", [])),
        aggregate=aggregate,
        run_dir=str(run_dir),
    )
    return run_dir
