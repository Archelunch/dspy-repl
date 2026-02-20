from __future__ import annotations

import logging
import re
import warnings
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import pydantic

from dspy_repl.compat import Module, Prediction, Tool, parse_value
from dspy_repl.core.code_interpreter import SIMPLE_TYPES, CodeInterpreter, CodeInterpreterError, FinalOutput
from dspy_repl.core.repl_types import REPLEntry, REPLHistory, REPLVariable

logger = logging.getLogger(__name__)


class BaseReplRLM(Module):
    """Shared non-Python REPL loop for DSPy-compatible RLM modules."""

    _RESERVED_TOOL_NAMES: frozenset[str] = frozenset()
    _CODE_FENCE_PATTERN: re.Pattern[str]
    _PYDANTIC_SERIALIZER_WARNING_RE = r".*Pydantic serializer warnings:.*"

    def __init__(
        self,
        *,
        max_iterations: int,
        max_llm_calls: int,
        max_output_chars: int,
        verbose: bool,
        tools: list[Callable] | None,
        interpreter: CodeInterpreter | None,
    ) -> None:
        super().__init__()
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.max_output_chars = max_output_chars
        self.verbose = verbose
        self._interpreter = interpreter
        self._user_tools = self._normalize_tools(tools)
        self._validate_tools(self._user_tools)

    @property
    def tools(self) -> dict[str, Tool]:
        return dict(self._user_tools)

    def _normalize_tools(self, tools: list[Callable] | None) -> dict[str, Tool]:
        if not tools:
            return {}
        if isinstance(tools, dict):
            raise TypeError(
                "tools must be a list, not a dict. "
                "Change tools={'name': func} to tools=[func] "
                "(tool names are inferred from function names, or use dspy.Tool(func, name='custom_name'))"
            )

        normalized: dict[str, Tool] = {}
        for item in tools:
            tool = item if isinstance(item, Tool) else Tool(item)
            if not tool.name:
                raise ValueError("Tool name cannot be empty")
            normalized[tool.name] = tool
        return normalized

    def _validate_tools(self, tools: dict[str, Tool]) -> None:
        for name in tools:
            if name in self._RESERVED_TOOL_NAMES:
                raise ValueError(f"Tool name '{name}' conflicts with built-in sandbox function")

    def _format_tool_docs(self, tools: dict[str, Tool]) -> str:
        if not tools:
            return ""
        lines = ["\nAdditional tools available:"]
        for tool in tools.values():
            params = []
            for arg_name, arg_schema in (tool.args or {}).items():
                arg_type = arg_schema.get("type", "Any")
                params.append(f"{arg_name}: {arg_type}")
            params_str = ", ".join(params)
            sig_str = f"{tool.name}({params_str})"
            desc = (tool.desc or "No description").replace("\n", "  ")
            lines.append(f"- `{sig_str}` - {desc}")
        return "\n".join(lines)

    def _strip_code_fences(self, code: Any) -> str:
        if not isinstance(code, str):
            return ""
        code = code.strip()
        match = self._CODE_FENCE_PATTERN.match(code)
        if match:
            return match.group(1)
        return code

    def _get_output_fields_info(self) -> list[dict[str, str]]:
        fields: list[dict[str, str]] = []
        for name, field in self.signature.output_fields.items():
            annotation = getattr(field, "annotation", str)
            field_info: dict[str, str] = {"name": name}
            if annotation in SIMPLE_TYPES:
                field_info["type"] = annotation.__name__
            fields.append(field_info)
        return fields

    def _build_variables(self, **input_args: Any) -> list[REPLVariable]:
        variables = []
        for name, value in input_args.items():
            field_info = self.signature.input_fields.get(name)
            variables.append(REPLVariable.from_value(name, value, field_info=field_info))
        return variables

    def _validate_inputs(self, input_args: dict[str, Any]) -> None:
        missing = set(self.signature.input_fields.keys()) - set(input_args.keys())
        if missing:
            raise ValueError(f"Missing required inputs: {sorted(missing)}")

    def _prepare_execution_tools(self) -> dict[str, Callable]:
        execution_tools = self._make_llm_tools()
        execution_tools.update({name: tool.func for name, tool in self._user_tools.items()})
        return execution_tools

    def _inject_execution_context(self, interpreter: CodeInterpreter, execution_tools: dict[str, Callable]) -> None:
        interpreter.tools.update(execution_tools)
        if hasattr(interpreter, "output_fields"):
            interpreter.output_fields = self._get_output_fields_info()
        if hasattr(interpreter, "_tools_registered"):
            interpreter._tools_registered = False

    @contextmanager
    def _interpreter_context(self, execution_tools: dict[str, Callable]) -> Iterator[CodeInterpreter]:
        if self._interpreter is not None:
            self._inject_execution_context(self._interpreter, execution_tools)
            yield self._interpreter
        else:
            repl = self._create_interpreter(execution_tools)
            try:
                yield repl
            finally:
                repl.shutdown()

    def _format_output(self, output: str) -> str:
        return output

    def _variables_info_for_prompt(self, repl: CodeInterpreter, variables: list[REPLVariable]) -> str:
        del repl
        return "\n\n".join(variable.format() for variable in variables)

    def _process_final_output(
        self,
        result: FinalOutput,
        output_field_names: list[str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        raw_output = result.output
        if not isinstance(raw_output, dict):
            return (
                None,
                f"[Error] FINAL returned {type(raw_output).__name__}, expected dict with fields: {output_field_names}",
            )

        missing = set(output_field_names) - set(raw_output.keys())
        if missing:
            return None, self._missing_outputs_hint(output_field_names, missing)

        parsed_outputs: dict[str, Any] = {}
        type_errors = []
        for name in output_field_names:
            field = self.signature.output_fields[name]
            annotation = getattr(field, "annotation", str)
            try:
                parsed_outputs[name] = parse_value(raw_output[name], annotation)
            except (ValueError, pydantic.ValidationError) as e:
                type_errors.append(
                    f"{name}: expected {annotation.__name__ if hasattr(annotation, '__name__') else annotation}, "
                    f"got {type(raw_output[name]).__name__}: {e}"
                )
        if type_errors:
            return None, "[Type Error] " + "; ".join(type_errors)
        return parsed_outputs, None

    def _process_execution_result(
        self,
        pred: Any,
        result: Any,
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        code = self._strip_code_fences(pred.code)

        if isinstance(result, str) and result.startswith("[Error]"):
            output = self._format_output(result)
            return history.append(reasoning=pred.reasoning, code=code, output=output)

        if isinstance(result, FinalOutput):
            parsed_outputs, error = self._process_final_output(result, output_field_names)
            if error:
                return history.append(reasoning=pred.reasoning, code=code, output=error)

            final_history = history.append(reasoning=pred.reasoning, code=code, output=f"FINAL: {parsed_outputs}")
            assert parsed_outputs is not None
            return Prediction(
                **parsed_outputs,
                trajectory=self._serialize_trajectory(final_history),
                final_reasoning=pred.reasoning,
            )

        if isinstance(result, list):
            output = "\n".join(map(str, result))
        else:
            output = str(result) if result else ""
        output = self._format_output(output)
        if self.verbose:
            logger.info(REPLEntry.format_output(output, self.max_output_chars))
        return history.append(reasoning=pred.reasoning, code=code, output=output)

    def _execute_iteration(
        self,
        repl: CodeInterpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        variables_info = self._variables_info_for_prompt(repl, variables)
        with self._suppress_known_pydantic_serializer_warning():
            action = self.generate_action(
                variables_info=variables_info,
                repl_history=history,
                iteration=f"{iteration + 1}/{self.max_iterations}",
            )
        self._log_iteration(iteration, action)
        try:
            code = self._strip_code_fences(action.code)
            if not code:
                result = "[Error] Model returned empty SQL/REPL code for this iteration."
            else:
                result = repl.execute(code, variables=dict(input_args))
        except (CodeInterpreterError, SyntaxError) as e:
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
    ) -> Prediction | REPLHistory:
        variables_info = self._variables_info_for_prompt(repl, variables)
        with self._suppress_known_pydantic_serializer_warning():
            action = await self.generate_action.acall(
                variables_info=variables_info,
                repl_history=history,
                iteration=f"{iteration + 1}/{self.max_iterations}",
            )
        self._log_iteration(iteration, action)
        try:
            code = self._strip_code_fences(action.code)
            if not code:
                result = "[Error] Model returned empty SQL/REPL code for this iteration."
            else:
                result = repl.execute(code, variables=dict(input_args))
        except (CodeInterpreterError, SyntaxError) as e:
            result = f"[Error] {e}"
        return self._process_execution_result(action, result, history, output_field_names)

    def _extract_fallback(
        self,
        variables_info: str,
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction:
        logger.warning("%s reached max iterations, using extract to get final output", self.__class__.__name__)
        with self._suppress_known_pydantic_serializer_warning():
            extract_pred = self.extract(variables_info=variables_info, repl_history=history)
        return Prediction(
            trajectory=self._serialize_trajectory(history),
            final_reasoning="Extract forced final output",
            **{name: getattr(extract_pred, name) for name in output_field_names},
        )

    async def _aextract_fallback(
        self,
        variables_info: str,
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction:
        logger.warning("%s reached max iterations, using extract to get final output", self.__class__.__name__)
        with self._suppress_known_pydantic_serializer_warning():
            extract_pred = await self.extract.acall(variables_info=variables_info, repl_history=history)
        return Prediction(
            trajectory=self._serialize_trajectory(history),
            final_reasoning="Extract forced final output",
            **{name: getattr(extract_pred, name) for name in output_field_names},
        )

    def forward(self, **input_args: Any) -> Prediction:
        self._validate_inputs(input_args)
        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools) as repl:
            history = self._new_history()
            for iteration in range(self.max_iterations):
                result = self._execute_iteration(repl, variables, history, iteration, input_args, output_field_names)
                if isinstance(result, Prediction):
                    return result
                history = result
            variables_info = self._variables_info_for_prompt(repl, variables)
            return self._extract_fallback(variables_info, history, output_field_names)

    async def aforward(self, **input_args: Any) -> Prediction:
        self._validate_inputs(input_args)
        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools) as repl:
            history = self._new_history()
            for iteration in range(self.max_iterations):
                result = await self._aexecute_iteration(
                    repl, variables, history, iteration, input_args, output_field_names
                )
                if isinstance(result, Prediction):
                    return result
                history = result
            variables_info = self._variables_info_for_prompt(repl, variables)
            return await self._aextract_fallback(variables_info, history, output_field_names)

    def _log_iteration(self, iteration: int, action: Any) -> None:
        if self.verbose:
            logger.info(
                "%s iteration %s/%s\nReasoning: %s\nCode:\n%s",
                self.__class__.__name__,
                iteration + 1,
                self.max_iterations,
                action.reasoning,
                action.code,
            )

    @contextmanager
    def _suppress_known_pydantic_serializer_warning(self) -> Iterator[None]:
        """
        Suppress known non-fatal warnings emitted by upstream LM response serialization.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=self._PYDANTIC_SERIALIZER_WARNING_RE,
                category=UserWarning,
            )
            yield

    def _serialize_trajectory(self, history: REPLHistory) -> list[dict[str, Any]]:
        trajectory: list[dict[str, Any]] = []
        for entry in history:
            try:
                payload = entry.model_dump(mode="json")
            except Exception:
                payload = {
                    "reasoning": getattr(entry, "reasoning", ""),
                    "code": getattr(entry, "code", ""),
                    "output": str(getattr(entry, "output", "")),
                }
            if isinstance(payload, dict):
                trajectory.append(payload)
            else:
                trajectory.append({"value": str(payload)})
        return trajectory

    @abstractmethod
    def _new_history(self) -> REPLHistory:
        """Create a fresh language-specific history instance."""

    @abstractmethod
    def _missing_outputs_hint(self, output_field_names: list[str], missing: set[str]) -> str:
        """Message for missing final outputs."""

    @abstractmethod
    def _create_interpreter(self, execution_tools: dict[str, Callable]) -> CodeInterpreter:
        """Create language-specific interpreter."""

    @abstractmethod
    def _make_llm_tools(self, max_workers: int = 8) -> dict[str, Callable]:
        """Create internal language-specific llm query tools."""
