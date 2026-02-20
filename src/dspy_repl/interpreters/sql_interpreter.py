"""
Local interpreter for SQL code execution using SQLite.

This module provides SQLInterpreter, which runs SQL statements in an isolated
SQLite database. It implements the CodeInterpreter protocol and mirrors the
stateful REPL behavior used by other interpreters.
"""

from __future__ import annotations

import inspect
import json
import keyword
import os
import queue
import re
import sqlite3
import threading
import time
from typing import Any, Callable

from dspy_repl.core.code_interpreter import CodeInterpreterError, FinalOutput

__all__ = ["SQLInterpreter", "FinalOutput", "CodeInterpreterError"]


class SQLInterpreter:
    """Stateful SQL interpreter backed by SQLite."""

    def __init__(
        self,
        db_path: str = ":memory:",
        tools: dict[str, Callable[..., str]] | None = None,
        output_fields: list[dict] | None = None,
        connection_factory: Callable[[str], sqlite3.Connection] | None = None,
        auto_indexes: bool = True,
        index_candidates: tuple[str, ...] | None = None,
        max_display_rows: int = 200,
        statement_timeout_seconds: float = 30.0,
        tool_timeout_seconds: float = 45.0,
        preload_sql: str | None = None,
        skip_variable_tables: set[str] | None = None,
    ) -> None:
        self.db_path = db_path
        self.tools = dict(tools) if tools else {}
        self.output_fields = output_fields
        self.connection_factory = connection_factory or sqlite3.connect
        self._preload_sql = preload_sql
        self._skip_variable_tables: set[str] = set(skip_variable_tables) if skip_variable_tables else set()
        self.auto_indexes = auto_indexes
        self.index_candidates = index_candidates or (
            "id",
            "key",
            "name",
            "city",
            "state",
            "country",
            "date",
            "time",
            "timestamp",
            "category",
            "type",
            "status",
            "user_id",
            "item_id",
            "query",
        )
        self.max_display_rows = max(1, int(max_display_rows))

        self._conn: sqlite3.Connection | None = None
        self._owner_thread: int | None = None
        self._tables_created: set[str] = set()
        self._final_output: Any | None = None
        self._registered_tools: set[str] = set()
        self._registered_submit = False
        self._table_generation = 0
        self.statement_timeout_seconds = statement_timeout_seconds
        self.tool_timeout_seconds = tool_timeout_seconds
        self._profile: dict[str, float] = {
            "table_load_seconds": 0.0,
            "query_exec_seconds": 0.0,
            "format_seconds": 0.0,
            "describe_seconds": 0.0,
            "execute_calls": 0.0,
        }

    def _check_thread_ownership(self) -> None:
        current_thread = threading.current_thread().ident
        if self._owner_thread is None:
            self._owner_thread = current_thread
        elif self._owner_thread != current_thread:
            raise RuntimeError(
                "SQLInterpreter is not thread-safe and cannot be shared across threads. "
                "Create a separate interpreter instance for each thread."
            )

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = self.connection_factory(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA temp_store = MEMORY")
            self._conn.execute("PRAGMA cache_size = -64000")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._run_preload_sql()
            self._register_builtin_functions()
            self._register_tool_functions()
        return self._conn

    def _discover_existing_tables(self) -> None:
        """Register all user tables from sqlite_master into _tables_created."""
        assert self._conn is not None
        cursor = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        for row in cursor:
            self._tables_created.add(row[0])
        self._table_generation += 1

    def _run_preload_sql(self) -> None:
        if not self._preload_sql:
            return
        assert self._conn is not None
        existing = self._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()
        if existing and existing[0] > 0:
            self._discover_existing_tables()
            return
        sql = self._preload_sql
        if os.path.isfile(sql):
            with open(sql) as f:
                sql = f.read()
        self._conn.executescript(sql)
        self._conn.commit()
        self._discover_existing_tables()

    def start(self) -> None:
        self._ensure_connection()

    def shutdown(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = None
        self._owner_thread = None
        self._tables_created.clear()
        self._final_output = None
        self._registered_tools.clear()
        self._registered_submit = False
        self._table_generation = 0
        self._profile = {
            "table_load_seconds": 0.0,
            "query_exec_seconds": 0.0,
            "format_seconds": 0.0,
            "describe_seconds": 0.0,
            "execute_calls": 0.0,
        }

    def __enter__(self) -> SQLInterpreter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code, variables)

    def _register_builtin_functions(self) -> None:
        conn = self._ensure_connection()
        if not self._registered_submit:
            conn.create_function("SUBMIT", 1, self._submit)
            self._registered_submit = True

    def _register_tool_functions(self) -> None:
        conn = self._ensure_connection()
        for name, fn in self.tools.items():
            arity = self._extract_fixed_arity(fn)
            conn.create_function(name, arity, self._wrap_tool(fn))
            self._registered_tools.add(name)

    def _extract_fixed_arity(self, fn: Callable[..., Any]) -> int:
        sig = inspect.signature(fn)
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        if has_varargs:
            return -1

        positional = [
            p
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional)

    @staticmethod
    def _coerce_tool_result(result: Any) -> Any:
        if result is None:
            return ""
        if isinstance(result, (str, int, float, bytes)):
            return result
        if isinstance(result, bool):
            return int(result)
        return json.dumps(result, ensure_ascii=False)

    def _wrap_tool(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any) -> Any:
            try:
                result = self._call_tool_with_timeout(fn, args)
                return self._coerce_tool_result(result)
            except Exception as e:
                raise sqlite3.OperationalError(f"Tool '{getattr(fn, '__name__', 'tool')}' failed: {e}") from e

        return wrapped

    def _call_tool_with_timeout(self, fn: Callable[..., Any], args: tuple[Any, ...]) -> Any:
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        cancel_event = threading.Event()

        def worker() -> None:
            try:
                params = inspect.signature(fn).parameters
                if "cancel_event" in params:
                    result = fn(*args, cancel_event=cancel_event)
                elif "_cancel_event" in params:
                    result = fn(*args, _cancel_event=cancel_event)
                else:
                    result = fn(*args)
                result_queue.put((True, result))
            except Exception as exc:
                result_queue.put((False, exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(self.tool_timeout_seconds)
        if thread.is_alive():
            cancel_event.set()
            raise TimeoutError(f"Tool call timed out after {self.tool_timeout_seconds:.1f}s.")
        ok, payload = result_queue.get_nowait()
        if ok:
            return payload
        raise payload

    def _submit(self, json_output: Any) -> str:
        try:
            if isinstance(json_output, (dict, list)):
                parsed = json_output
            elif json_output is None:
                parsed = {}
            else:
                parsed = json.loads(str(json_output))
        except Exception as e:
            raise sqlite3.OperationalError(f"SUBMIT expects JSON text, got: {json_output!r}") from e

        self._final_output = parsed
        return "SUBMITTED"

    def _quote_ident(self, name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    def _validate_identifier(self, name: str) -> None:
        if not name.isidentifier() or keyword.iskeyword(name):
            raise CodeInterpreterError(f"Invalid variable name for SQL table: '{name}'")

    def _infer_sql_type(self, value: Any) -> str:
        if isinstance(value, bool):
            return "INTEGER"
        if isinstance(value, int):
            return "INTEGER"
        if isinstance(value, float):
            return "REAL"
        if value is None:
            return "TEXT"
        return "TEXT"

    def _normalize_cell(self, value: Any) -> Any:
        if isinstance(value, bool):
            return int(value)
        if value is None or isinstance(value, (int, float, str, bytes)):
            return value
        return json.dumps(value, ensure_ascii=False)

    def _try_parse_json_string(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        candidate = value.strip()
        if not candidate:
            return value
        if not (
            (candidate.startswith("{") and candidate.endswith("}"))
            or (candidate.startswith("[") and candidate.endswith("]"))
        ):
            return value
        try:
            return json.loads(candidate)
        except Exception:
            return value

    def _maybe_create_indexes(self, cursor: sqlite3.Cursor, table_name: str, columns: list[str]) -> None:
        if not self.auto_indexes:
            return
        table_ident = self._quote_ident(table_name)
        for col in columns:
            if col not in self.index_candidates:
                continue
            if col in {"idx", "key"}:
                continue
            idx_name = self._quote_ident(f"ix_{table_name}_{col}")
            col_ident = self._quote_ident(col)
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_ident} ({col_ident})")

    def _ensure_variable_table(self, name: str, raw_value: Any) -> None:
        self._validate_identifier(name)
        if name in self._tables_created:
            return
        if name in self._skip_variable_tables:
            return

        value = self._try_parse_json_string(raw_value)
        conn = self._ensure_connection()
        cursor = conn.cursor()
        table = self._quote_ident(name)

        if isinstance(value, (str, int, float, bool)) or value is None:
            value_type = self._infer_sql_type(value)
            cursor.execute(f"CREATE TABLE {table} (value {value_type})")
            cursor.execute(f"INSERT INTO {table} (value) VALUES (?)", (self._normalize_cell(value),))
            self._tables_created.add(name)
            self._table_generation += 1
            return

        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            dict_items = [item for item in value if isinstance(item, dict)]
            cols: list[str] = []
            for item in dict_items:
                for key in item.keys():
                    key_s = str(key)
                    if key_s not in cols:
                        cols.append(key_s)

            col_types: dict[str, str] = dict.fromkeys(cols, "TEXT")
            for col in cols:
                for item in dict_items:
                    val = item.get(col)
                    inferred = self._infer_sql_type(val)
                    if col_types[col] == "TEXT":
                        continue
                    col_types[col] = inferred
                for item in dict_items:
                    if item.get(col) is not None:
                        col_types[col] = self._infer_sql_type(item.get(col))
                        break

            col_defs = ", ".join(f"{self._quote_ident(col)} {col_types.get(col, 'TEXT')}" for col in cols)
            cursor.execute(f"CREATE TABLE {table} (idx INTEGER PRIMARY KEY, {col_defs})")
            self._maybe_create_indexes(cursor, name, cols)

            placeholders = ", ".join("?" for _ in cols)
            col_idents = ", ".join(self._quote_ident(col) for col in cols)
            rows = [
                [idx] + [self._normalize_cell(item.get(col)) for col in cols] for idx, item in enumerate(dict_items)
            ]
            cursor.executemany(f"INSERT INTO {table} (idx, {col_idents}) VALUES (?, {placeholders})", rows)
            self._tables_created.add(name)
            self._table_generation += 1
            return

        if isinstance(value, list):
            item_type = "TEXT"
            for item in value:
                if item is not None:
                    item_type = self._infer_sql_type(item)
                    break
            cursor.execute(f"CREATE TABLE {table} (idx INTEGER PRIMARY KEY, value {item_type})")
            rows = [(idx, self._normalize_cell(item)) for idx, item in enumerate(value)]
            cursor.executemany(f"INSERT INTO {table} (idx, value) VALUES (?, ?)", rows)
            self._tables_created.add(name)
            self._table_generation += 1
            return

        if isinstance(value, dict):
            items = list(value.items())
            if items and all(isinstance(v, dict) for _, v in items):
                nested_cols: list[str] = []
                for _, nested in items:
                    for col in nested.keys():
                        col_s = str(col)
                        if col_s not in nested_cols:
                            nested_cols.append(col_s)
                col_defs = ", ".join(f"{self._quote_ident(c)} TEXT" for c in nested_cols)
                cursor.execute(f"CREATE TABLE {table} (key TEXT PRIMARY KEY, {col_defs})")
                self._maybe_create_indexes(cursor, name, nested_cols)
                placeholders = ", ".join("?" for _ in nested_cols)
                col_idents = ", ".join(self._quote_ident(c) for c in nested_cols)
                rows = [
                    [str(key)] + [self._normalize_cell(nested.get(col)) for col in nested_cols] for key, nested in items
                ]
                cursor.executemany(f"INSERT INTO {table} (key, {col_idents}) VALUES (?, {placeholders})", rows)
            else:
                cursor.execute(f"CREATE TABLE {table} (key TEXT PRIMARY KEY, value TEXT)")
                rows = [(str(key), self._normalize_cell(val)) for key, val in items]
                cursor.executemany(f"INSERT INTO {table} (key, value) VALUES (?, ?)", rows)
            self._tables_created.add(name)
            self._table_generation += 1
            return

        cursor.execute(f"CREATE TABLE {table} (value TEXT)")
        cursor.execute(f"INSERT INTO {table} (value) VALUES (?)", (json.dumps(value, ensure_ascii=False),))
        self._tables_created.add(name)
        self._table_generation += 1

    def _inject_variables(self, variables: dict[str, Any]) -> None:
        if not variables:
            return
        if all(name in self._tables_created for name in variables):
            return
        conn = self._ensure_connection()
        before = len(self._tables_created)
        for name, value in variables.items():
            self._ensure_variable_table(name, value)
        if len(self._tables_created) > before:
            conn.commit()

    _CHECK_RE = re.compile(r"CHECK\s*\(((?:[^()]*|\([^()]*\))*)\)", re.IGNORECASE)

    def describe_tables(self, table_names: list[str] | None = None) -> list[dict[str, Any]]:
        started = time.perf_counter()
        conn = self._ensure_connection()
        cursor = conn.cursor()
        target_names = set(table_names) if table_names else set(self._tables_created)
        descriptions: list[dict[str, Any]] = []

        for name in sorted(target_names):
            if name not in self._tables_created:
                continue
            ident = self._quote_ident(name)
            cols = cursor.execute(f"PRAGMA table_info({ident})").fetchall()
            col_defs = [f"{row['name']} {row['type'] or 'TEXT'}" for row in cols]
            row_count = cursor.execute(f"SELECT COUNT(*) AS c FROM {ident}").fetchone()

            fks = cursor.execute(f"PRAGMA foreign_key_list({ident})").fetchall()
            fk_info = [f"{row['from']} -> {row['table']}.{row['to']}" for row in fks]

            check_constraints: list[str] = []
            create_row = cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()
            if create_row and create_row["sql"]:
                check_constraints = self._CHECK_RE.findall(create_row["sql"])

            desc: dict[str, Any] = {
                "name": name,
                "columns": col_defs,
                "rows": int(row_count["c"]) if row_count else 0,
            }
            if fk_info:
                desc["foreign_keys"] = fk_info
            if check_constraints:
                desc["checks"] = check_constraints
            descriptions.append(desc)
        self._profile["describe_seconds"] += time.perf_counter() - started
        return descriptions

    def table_generation(self) -> int:
        return self._table_generation

    def get_profile(self) -> dict[str, float]:
        return dict(self._profile)

    def _split_sql_statements(self, code: str) -> list[str]:
        statements: list[str] = []
        buf: list[str] = []
        in_single_quote = False
        in_double_quote = False
        in_line_comment = False
        in_block_comment = False
        i = 0
        n = len(code)

        while i < n:
            ch = code[i]
            nxt = code[i + 1] if i + 1 < n else ""

            if in_line_comment:
                buf.append(ch)
                if ch == "\n":
                    in_line_comment = False
                i += 1
                continue

            if in_block_comment:
                buf.append(ch)
                if ch == "*" and nxt == "/":
                    buf.append(nxt)
                    i += 2
                    in_block_comment = False
                    continue
                i += 1
                continue

            if in_single_quote:
                buf.append(ch)
                if ch == "'" and nxt == "'":
                    buf.append(nxt)
                    i += 2
                    continue
                if ch == "'":
                    in_single_quote = False
                i += 1
                continue

            if in_double_quote:
                buf.append(ch)
                if ch == '"' and nxt == '"':
                    buf.append(nxt)
                    i += 2
                    continue
                if ch == '"':
                    in_double_quote = False
                i += 1
                continue

            if ch == "-" and nxt == "-":
                buf.append(ch)
                buf.append(nxt)
                i += 2
                in_line_comment = True
                continue

            if ch == "/" and nxt == "*":
                buf.append(ch)
                buf.append(nxt)
                i += 2
                in_block_comment = True
                continue

            if ch == "'":
                buf.append(ch)
                in_single_quote = True
                i += 1
                continue

            if ch == '"':
                buf.append(ch)
                in_double_quote = True
                i += 1
                continue

            if ch == ";":
                statement = "".join(buf).strip()
                if statement:
                    statements.append(statement)
                buf = []
                i += 1
                continue

            buf.append(ch)
            i += 1

        tail = "".join(buf).strip()
        if tail:
            statements.append(tail)
        return statements

    def _fetch_preview_rows(self, cursor: sqlite3.Cursor) -> tuple[list[sqlite3.Row], int, bool]:
        batch = cursor.fetchmany(self.max_display_rows + 1)
        if len(batch) <= self.max_display_rows:
            return batch, len(batch), False

        rows = batch[: self.max_display_rows]
        total_rows = len(batch)
        while True:
            more = cursor.fetchmany(1000)
            if not more:
                break
            total_rows += len(more)
        return rows, total_rows, True

    def _format_select_rows(
        self,
        cursor: sqlite3.Cursor,
        rows: list[sqlite3.Row],
        total_rows: int,
        truncated: bool,
    ) -> str:
        columns = [col[0] for col in (cursor.description or [])]
        if not columns:
            return "(no columns)"

        str_rows = [[("" if v is None else str(v)) for v in row] for row in rows]
        widths = [len(col) for col in columns]
        for row in str_rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        header = "| " + " | ".join(columns[i].ljust(widths[i]) for i in range(len(columns))) + " |"
        sep = "|-" + "-|-".join("-" * widths[i] for i in range(len(columns))) + "-|"
        body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |" for row in str_rows]
        summary = f"({total_rows} rows)"
        if truncated:
            summary = f"(showing {len(rows)} of {total_rows} rows; add LIMIT for targeted inspection)"
        return "\n".join([header, sep] + body + [summary])

    @staticmethod
    def _strip_leading_sql_comments(text: str) -> str:
        """Remove leading ``-- …`` and ``/* … */`` comments from a SQL string."""
        s = text.lstrip()
        while s:
            if s.startswith("--"):
                newline = s.find("\n")
                if newline == -1:
                    return ""
                s = s[newline + 1 :].lstrip()
            elif s.startswith("/*"):
                end = s.find("*/")
                if end == -1:
                    return ""
                s = s[end + 2 :].lstrip()
            else:
                break
        return s

    def _statement_is_select(self, statement: str) -> bool:
        upper = self._strip_leading_sql_comments(statement).upper()
        return upper.startswith("SELECT") or upper.startswith("WITH") or upper.startswith("PRAGMA")

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        self._check_thread_ownership()
        overall_started = time.perf_counter()
        conn = self._ensure_connection()
        self._register_builtin_functions()
        self._register_tool_functions()

        self._final_output = None
        load_started = time.perf_counter()
        self._inject_variables(variables or {})
        self._profile["table_load_seconds"] += time.perf_counter() - load_started

        statements = self._split_sql_statements(code)
        if not statements:
            return None

        outputs: list[str] = []
        cursor = conn.cursor()

        try:
            for statement in statements:
                statement_started = time.perf_counter()
                timeout_seconds = self.statement_timeout_seconds

                def _progress_handler(
                    _statement_started: float = statement_started,
                    _timeout_seconds: float = timeout_seconds,
                ) -> int:
                    if (time.perf_counter() - _statement_started) > _timeout_seconds:
                        return 1
                    return 0

                conn.set_progress_handler(_progress_handler, 10_000)
                exec_started = time.perf_counter()
                cursor.execute(statement)
                self._profile["query_exec_seconds"] += time.perf_counter() - exec_started
                if self._statement_is_select(statement):
                    fmt_started = time.perf_counter()
                    rows, total_rows, truncated = self._fetch_preview_rows(cursor)
                    outputs.append(self._format_select_rows(cursor, rows, total_rows, truncated))
                    self._profile["format_seconds"] += time.perf_counter() - fmt_started
                else:
                    affected = cursor.rowcount if cursor.rowcount is not None and cursor.rowcount >= 0 else 0
                    outputs.append(f"OK ({affected} rows affected)")
                    # Commit each non-SELECT statement so tool UDFs that read through
                    # separate SQLite connections observe the latest writes immediately.
                    conn.commit()
                conn.set_progress_handler(None, 0)
            conn.commit()
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e).lower():
                raise CodeInterpreterError(
                    f"SQLite statement timed out after {self.statement_timeout_seconds:.1f}s."
                ) from e
            raise CodeInterpreterError(f"SQLite error: {e}") from e
        except sqlite3.DatabaseError as e:
            raise CodeInterpreterError(f"Database error: {e}") from e
        finally:
            conn.set_progress_handler(None, 0)
            self._profile["execute_calls"] += 1
            _ = overall_started

        if self._final_output is not None:
            return FinalOutput(self._final_output)
        return "\n".join(outputs)
