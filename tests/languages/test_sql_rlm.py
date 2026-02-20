from __future__ import annotations

import asyncio

from dspy_repl.core.code_interpreter import CodeInterpreterError, FinalOutput
from dspy_repl.languages.sql_rlm import SQLRLM
from tests.helpers.mock_interpreter import MockInterpreter
from tests.helpers.predictor_factory import make_mock_predictor


def test_sql_code_fence_stripping(lm_context) -> None:
    with lm_context([{"output": "x"}]):
        rlm = SQLRLM("context -> output", interpreter=MockInterpreter())
        code = "```sql\nSELECT 1;\n```"
        assert rlm._strip_code_fences(code) == "SELECT 1;"


def test_sql_single_iteration_submit(lm_context) -> None:
    mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
    with lm_context([{"answer": "42"}]):
        rlm = SQLRLM("context -> answer", interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [{"reasoning": "done", "code": "SELECT SUBMIT(json_object('answer', '42'))"}]
        )
        result = rlm(context="test")
        assert result.answer == "42"
        assert len(result.trajectory) == 1


def test_sql_error_recovery(lm_context) -> None:
    mock = MockInterpreter(responses=[CodeInterpreterError("bad sql"), FinalOutput({"answer": "ok"})])
    with lm_context([{"answer": "ok"}]):
        rlm = SQLRLM("context -> answer", interpreter=mock, max_iterations=3)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "try", "code": "SELECT * FROM missing_table;"},
                {"reasoning": "recover", "code": "SELECT SUBMIT(json_object('answer', 'ok'));"},
            ]
        )
        result = rlm(context="test")
        assert result.answer == "ok"
        assert len(result.trajectory) == 2


def test_sql_fallback_extract_after_max_iterations(lm_context) -> None:
    mock = MockInterpreter(responses=["rows", "rows"])
    with lm_context([{"answer": "fallback"}]):
        rlm = SQLRLM("context -> answer", interpreter=mock, max_iterations=2)
        rlm.generate_action = make_mock_predictor([{"reasoning": "explore", "code": "SELECT * FROM context LIMIT 1;"}])
        rlm.extract = make_mock_predictor([{"answer": "fallback"}])
        result = rlm(context="test")
        assert result.answer == "fallback"
        assert result.final_reasoning == "Extract forced final output"


def test_sql_async_forward(lm_context) -> None:
    mock = MockInterpreter(responses=[FinalOutput({"answer": "async"})])
    with lm_context([{"answer": "async"}]):
        rlm = SQLRLM("context -> answer", interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [{"reasoning": "done", "code": "SELECT SUBMIT(json_object('answer', 'async'));"}]
        )
        result = asyncio.run(rlm.aforward(context="test"))
        assert result.answer == "async"


def test_sql_profile_sync_when_interpreter_exposes_profile(lm_context) -> None:
    class ProfiledMockInterpreter(MockInterpreter):
        def get_profile(self):
            return {"table_load_seconds": 0.2, "query_exec_seconds": 0.3}

    mock = ProfiledMockInterpreter(responses=[FinalOutput({"answer": "42"})])
    with lm_context([{"answer": "42"}]):
        rlm = SQLRLM("context -> answer", interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [{"reasoning": "done", "code": "SELECT SUBMIT(json_object('answer', '42'));"}]
        )
        _ = rlm(context="test")
        assert rlm.last_sql_profile["table_load_seconds"] == 0.2
        assert rlm.last_sql_profile["query_exec_seconds"] == 0.3


def test_sql_recovers_when_action_code_is_none(lm_context) -> None:
    mock = MockInterpreter(responses=[FinalOutput({"answer": "ok"})])
    with lm_context([{"answer": "ok"}]):
        rlm = SQLRLM("context -> answer", interpreter=mock, max_iterations=3)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "bad action", "code": None},
                {"reasoning": "recover", "code": "SELECT SUBMIT(json_object('answer', 'ok'));"},
            ]
        )
        result = rlm(context="test")
        assert result.answer == "ok"
        assert len(result.trajectory) == 2
        assert result.trajectory[0]["output"].startswith("[Error] Model returned empty")
