from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable

from pydantic import Field

from dspy_repl.compat import dspy, ensure_signature, get_active_lm, translate_field_type
from dspy_repl.core.base_rlm import BaseReplRLM
from dspy_repl.core.code_interpreter import CodeInterpreter
from dspy_repl.core.repl_types import REPLEntry, REPLHistory, REPLVariable
from dspy_repl.interpreters import load_sql_interpreter

if TYPE_CHECKING:
    from dspy_repl.compat import Signature


class SQLREPLEntry(REPLEntry):
    def format(self, index: int, max_output_chars: int = 10_000) -> str:
        reasoning_line = f"Reasoning: {self.reasoning}\n" if self.reasoning else ""
        return (
            f"=== Step {index + 1} ===\n"
            f"{reasoning_line}"
            f"Code:\n```sql\n{self.code}\n```\n"
            f"{self.format_output(self.output, max_output_chars)}"
        )


class SQLREPLHistory(REPLHistory):
    entries: list[SQLREPLEntry] = Field(default_factory=list)

    def format(self) -> str:
        if not self.entries:
            return "You have not interacted with the SQL REPL environment yet."
        return "\n".join(
            entry.format(index=i, max_output_chars=self.max_output_chars) for i, entry in enumerate(self.entries)
        )

    def append(self, *, reasoning: str = "", code: str, output: str) -> SQLREPLHistory:
        new_entry = SQLREPLEntry(reasoning=reasoning, code=code, output=output)
        return SQLREPLHistory(entries=list(self.entries) + [new_entry], max_output_chars=self.max_output_chars)


SQL_ACTION_INSTRUCTIONS_TEMPLATE = """You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a SQLite REPL environment. Write SQL and it will be executed. You will see output, then write more SQL based on what you learned. This is an iterative process.

Available:
- Input variables may be preloaded as SQL tables or provided as plain-text context (see variables_info).
- The database may contain pre-existing tables with FOREIGN KEY and CHECK constraints. Respect insert ordering: parent tables before child tables.
- llm_query(prompt) - query a sub-LLM for semantic analysis
- llm_query_batched(prompt1, prompt2, ...) - query multiple prompts concurrently, returns JSON array string
- SELECT SUBMIT(json_object(...)) - submit final output when done. SUBMIT MUST be called as SELECT SUBMIT(...), never bare SUBMIT(...).

IMPORTANT: This is ITERATIVE.
1. READ VARIABLES_INFO FIRST - it shows table schemas, FK relationships, CHECK constraints, and row counts.
2. ITERATE - run small SQL snippets and inspect outputs.
3. VERIFY BEFORE SUBMITTING - if results look wrong, revise.
4. USE llm_query FOR SEMANTICS - SQL finds structure; llm_query handles semantic interpretation.
5. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run.

SYNTAX RULES:
- This is SQLite. Use json_extract(), json_each(), COALESCE() — NOT jsonExtract or camelCase variants.
- To submit: SELECT SUBMIT(json_object('key1', val1, 'key2', val2)) — always wrap in SELECT.
- Output ONLY valid SQL in the code block. No extra brackets, markup or non-SQL text.

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT with your output."""


class SQLRLM(BaseReplRLM):
    _RESERVED_TOOL_NAMES = frozenset({"llm_query", "llm_query_batched", "SUBMIT"})
    _CODE_FENCE_PATTERN = re.compile(r"^```(?:sql)?\s*\n(.*)\n```\s*$", re.DOTALL)

    def __init__(
        self,
        signature: type[Signature] | str,
        max_iterations: int = 20,
        max_llm_calls: int = 50,
        max_output_chars: int = 10_000,
        verbose: bool = False,
        tools: list[Callable] | None = None,
        sub_lm: dspy.LM | None = None,
        interpreter: CodeInterpreter | None = None,
        db_path: str = ":memory:",
        preload_sql: str | None = None,
        skip_variable_tables: set[str] | None = None,
        tool_timeout_seconds: float = 600.0,
        statement_timeout_seconds: float = 600.0,
    ) -> None:
        self.signature = ensure_signature(signature)
        self.sub_lm = sub_lm
        self._db_path = db_path
        self._preload_sql = preload_sql
        self._skip_variable_tables = set(skip_variable_tables) if skip_variable_tables else set()
        self._tool_timeout_seconds = tool_timeout_seconds
        self._statement_timeout_seconds = statement_timeout_seconds
        self._variables_info_cache_key: tuple[Any, ...] | None = None
        self._variables_info_cache_value: str | None = None
        self.last_sql_profile: dict[str, float] = {}
        self._run_profile: dict[str, float] = {}
        super().__init__(
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            max_output_chars=max_output_chars,
            verbose=verbose,
            tools=tools,
            interpreter=interpreter,
        )
        action_sig, extract_sig = self._build_signatures()
        self.generate_action = dspy.Predict(action_sig)
        self.extract = dspy.Predict(extract_sig)

    def _new_history(self) -> SQLREPLHistory:
        return SQLREPLHistory(max_output_chars=self.max_output_chars)

    def _format_output(self, output: str) -> str:
        return output or "(no output - did your SQL return rows?)"

    def _missing_outputs_hint(self, output_field_names: list[str], missing: set[str]) -> str:
        pairs = ", ".join(f"'{name}', {name}" for name in output_field_names)
        return f"[Error] Missing output fields: {sorted(missing)}. Use SELECT SUBMIT(json_object({pairs}))"

    def _create_interpreter(self, execution_tools: dict[str, Callable]) -> CodeInterpreter:
        interpreter_cls = load_sql_interpreter()
        return interpreter_cls(
            db_path=self._db_path,
            preload_sql=self._preload_sql,
            skip_variable_tables=self._skip_variable_tables or None,
            tools=execution_tools,
            output_fields=self._get_output_fields_info(),
            tool_timeout_seconds=self._tool_timeout_seconds,
            statement_timeout_seconds=self._statement_timeout_seconds,
        )

    @staticmethod
    def _format_table_line(table: dict[str, Any]) -> str:
        columns = ", ".join(table.get("columns", []))
        rows = table.get("rows", 0)
        line = f"- {table['name']} ({columns}) -- {rows} rows"
        fks = table.get("foreign_keys", [])
        if fks:
            line += f"\n  FKs: {', '.join(fks)}"
        checks = table.get("checks", [])
        if checks:
            line += f"\n  CHECKs: {'; '.join(checks)}"
        return line

    def _variables_info_for_prompt(self, repl: CodeInterpreter, variables: list[REPLVariable]) -> str:
        generation = -1
        if hasattr(repl, "table_generation"):
            try:
                generation = int(repl.table_generation())
            except Exception:
                generation = -1

        cache_key = (generation, tuple((v.name, v.total_length) for v in variables))
        if cache_key == self._variables_info_cache_key and self._variables_info_cache_value is not None:
            return self._variables_info_cache_value

        all_table_info: list[dict[str, Any]] = []
        if hasattr(repl, "describe_tables"):
            try:
                all_table_info = repl.describe_tables()
            except Exception:
                all_table_info = []

        if not all_table_info:
            return "\n\n".join(v.format() for v in variables)

        by_name = {entry["name"]: entry for entry in all_table_info}
        var_names = {v.name for v in variables}
        skip_names = self._skip_variable_tables

        input_var_lines: list[str] = []
        for var in variables:
            if var.name in skip_names:
                input_var_lines.append(f"- {var.name}: {var.preview}")
                continue
            table = by_name.pop(var.name, None)
            if not table:
                input_var_lines.append(f"- {var.name} (table not found)")
                continue
            input_var_lines.append(self._format_table_line(table))

        schema_lines: list[str] = []
        for name in sorted(by_name):
            if name in var_names:
                continue
            schema_lines.append(self._format_table_line(by_name[name]))

        sections: list[str] = []
        if input_var_lines:
            sections.append("Input variables:\n" + "\n".join(input_var_lines))
        if schema_lines:
            sections.append("Database tables:\n" + "\n".join(schema_lines))

        formatted = "\n\n".join(sections) if sections else "\n\n".join(v.format() for v in variables)
        self._variables_info_cache_key = cache_key
        self._variables_info_cache_value = formatted
        return formatted

    def _make_llm_tools(self, max_workers: int = 8) -> dict[str, Callable]:
        state = {"call_count": 0}
        lock = threading.Lock()

        def _check_and_increment(n: int = 1) -> None:
            with lock:
                if state["call_count"] + n > self.max_llm_calls:
                    raise RuntimeError(
                        f"LLM call limit exceeded: {state['call_count']} + {n} > {self.max_llm_calls}. "
                        "Use SQL for aggregation instead of making more LLM calls."
                    )
                state["call_count"] += n

        def _query_lm(prompt: str) -> str:
            target_lm = get_active_lm(self.sub_lm)
            if target_lm is None:
                raise RuntimeError("No LM configured. Use dspy.configure(lm=...) or pass sub_lm to SQLRLM.")
            response = target_lm(prompt)
            if isinstance(response, list) and response:
                item = response[0]
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
                return str(item)
            return str(response)

        def llm_query(prompt: str) -> str:
            if not prompt:
                raise ValueError("prompt cannot be empty")
            _check_and_increment(1)
            return _query_lm(prompt)

        def llm_query_batched(*prompts: str) -> str:
            if not prompts:
                return "[]"
            _check_and_increment(len(prompts))
            results: dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {executor.submit(_query_lm, p): i for i, p in enumerate(prompts)}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = f"[ERROR] {e}"
            ordered = [results[i] for i in range(len(prompts))]
            return json.dumps(ordered, ensure_ascii=False)

        return {"llm_query": llm_query, "llm_query_batched": llm_query_batched}

    def _build_signatures(self) -> tuple[Signature, Signature]:
        inputs_str = ", ".join(f"`{n}`" for n in self.signature.input_fields)
        final_output_pairs = ", ".join(f"'{name}', {name}" for name in self.signature.output_fields.keys())
        output_fields = "\n".join(f"- {translate_field_type(n, f)}" for n, f in self.signature.output_fields.items())
        task_instructions = f"{self.signature.instructions}\n\n" if self.signature.instructions else ""
        tool_docs = self._format_tool_docs(self._user_tools)

        action_sig = (
            dspy.Signature(
                {},
                task_instructions
                + SQL_ACTION_INSTRUCTIONS_TEMPLATE.format(
                    inputs=inputs_str,
                    final_output_names=final_output_pairs,
                    output_fields=output_fields,
                    max_llm_calls=self.max_llm_calls,
                )
                + tool_docs,
            )
            .append(
                "variables_info", dspy.InputField(desc="Table schemas and metadata available in SQL REPL"), type_=str
            )
            .append(
                "repl_history", dspy.InputField(desc="Previous SQL REPL executions and outputs"), type_=SQLREPLHistory
            )
            .append(
                "iteration",
                dspy.InputField(desc="Current iteration number (1-indexed) out of max_iterations"),
                type_=str,
            )
            .append(
                "reasoning",
                dspy.OutputField(desc="Think step-by-step: what do you know? What remains? Plan your next SQL action."),
                type_=str,
            )
            .append("code", dspy.OutputField(desc="SQL code to execute. Use markdown code block format."), type_=str)
        )

        extract_instructions = """Based on the REPL trajectory, extract the final outputs now.

Review your trajectory to see what information you gathered and what values you computed, then provide the final outputs."""
        extended_task_instructions = (
            "The trajectory was generated with the following objective: \n" + task_instructions + "\n"
            if task_instructions
            else ""
        )
        extract_sig = dspy.Signature(
            {**self.signature.output_fields}, extended_task_instructions + extract_instructions
        )
        extract_sig = extract_sig.prepend(
            "repl_history", dspy.InputField(desc="Your SQL REPL interactions so far"), type_=SQLREPLHistory
        )
        extract_sig = extract_sig.prepend(
            "variables_info",
            dspy.InputField(desc="Table schemas and metadata available in SQL REPL"),
            type_=str,
        )
        return action_sig, extract_sig

    def _execute_iteration(
        self,
        repl: CodeInterpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
    ) -> dspy.Prediction | REPLHistory:
        variables_info = self._variables_info_for_prompt(repl, variables)
        prompt_started = time.perf_counter()
        action = self.generate_action(
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iterations}",
        )
        self._run_profile["prompt_seconds"] += time.perf_counter() - prompt_started
        self._log_iteration(iteration, action)

        try:
            code = self._strip_code_fences(action.code)
            result = repl.execute(code, variables=dict(input_args))
        except Exception as e:
            result = f"[Error] {e}"
        return self._process_execution_result(action, result, history, output_field_names)

    async def _aexecute_iteration(
        self,
        repl: CodeInterpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
    ) -> dspy.Prediction | REPLHistory:
        variables_info = self._variables_info_for_prompt(repl, variables)
        prompt_started = time.perf_counter()
        action = await self.generate_action.acall(
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iterations}",
        )
        self._run_profile["prompt_seconds"] += time.perf_counter() - prompt_started
        self._log_iteration(iteration, action)

        try:
            code = self._strip_code_fences(action.code)
            result = repl.execute(code, variables=dict(input_args))
        except Exception as e:
            result = f"[Error] {e}"
        return self._process_execution_result(action, result, history, output_field_names)

    def _sync_last_profile(self, repl: CodeInterpreter) -> None:
        interpreter_profile: dict[str, float] = {}
        if hasattr(repl, "get_profile"):
            try:
                interpreter_profile = repl.get_profile()
            except Exception:
                interpreter_profile = {}
        self.last_sql_profile = {**interpreter_profile, **self._run_profile}

    def forward(self, **input_args: Any) -> dspy.Prediction:
        self._validate_inputs(input_args)
        self._variables_info_cache_key = None
        self._variables_info_cache_value = None
        self._run_profile = {"prompt_seconds": 0.0}
        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools) as repl:
            history = self._new_history()
            for iteration in range(self.max_iterations):
                result = self._execute_iteration(repl, variables, history, iteration, input_args, output_field_names)
                if isinstance(result, dspy.Prediction):
                    self._sync_last_profile(repl)
                    return result
                history = result
            variables_info = self._variables_info_for_prompt(repl, variables)
            self._sync_last_profile(repl)
            return self._extract_fallback(variables_info, history, output_field_names)

    async def aforward(self, **input_args: Any) -> dspy.Prediction:
        self._validate_inputs(input_args)
        self._variables_info_cache_key = None
        self._variables_info_cache_value = None
        self._run_profile = {"prompt_seconds": 0.0}
        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools) as repl:
            history = self._new_history()
            for iteration in range(self.max_iterations):
                result = await self._aexecute_iteration(
                    repl, variables, history, iteration, input_args, output_field_names
                )
                if isinstance(result, dspy.Prediction):
                    self._sync_last_profile(repl)
                    return result
                history = result
            variables_info = self._variables_info_for_prompt(repl, variables)
            self._sync_last_profile(repl)
            return await self._aextract_fallback(variables_info, history, output_field_names)
