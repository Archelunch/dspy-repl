"""Tests for SQLRLM with preload_sql, skip_variable_tables, and enriched prompt info."""

from __future__ import annotations

from dspy_repl.interpreters.sql_interpreter import SQLInterpreter
from dspy_repl.languages.sql_rlm import SQLRLM
from tests.helpers.predictor_factory import make_mock_predictor

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE authors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE books (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author_id TEXT NOT NULL REFERENCES authors(id),
    genre TEXT NOT NULL CHECK(genre IN ('fiction','nonfiction','poetry')),
    pages INTEGER CHECK(pages > 0)
);
"""


class TestVariablesInfoForPrompt:
    def test_schema_tables_visible_in_prompt(self, lm_context) -> None:
        interp = SQLInterpreter(preload_sql=SCHEMA)
        with lm_context([{"answer": "42"}]):
            rlm = SQLRLM("query -> answer", interpreter=interp)
            variables = rlm._build_variables(query="find all fiction books")
            info = rlm._variables_info_for_prompt(interp, variables)

            assert "authors" in info
            assert "books" in info
            assert "query" in info

    def test_fk_info_in_prompt(self, lm_context) -> None:
        interp = SQLInterpreter(preload_sql=SCHEMA)
        with lm_context([{"answer": "42"}]):
            rlm = SQLRLM("query -> answer", interpreter=interp)
            variables = rlm._build_variables(query="test")
            info = rlm._variables_info_for_prompt(interp, variables)

            assert "author_id -> authors.id" in info

    def test_check_info_in_prompt(self, lm_context) -> None:
        interp = SQLInterpreter(preload_sql=SCHEMA)
        with lm_context([{"answer": "42"}]):
            rlm = SQLRLM("query -> answer", interpreter=interp)
            variables = rlm._build_variables(query="test")
            info = rlm._variables_info_for_prompt(interp, variables)

            assert "CHECK" in info or "fiction" in info

    def test_skip_variable_shows_as_plain_text(self, lm_context) -> None:
        interp = SQLInterpreter(
            preload_sql=SCHEMA,
            skip_variable_tables={"query"},
        )
        with lm_context([{"answer": "42"}]):
            rlm = SQLRLM(
                "query -> answer",
                interpreter=interp,
                skip_variable_tables={"query"},
            )
            variables = rlm._build_variables(query="find fiction books")
            info = rlm._variables_info_for_prompt(interp, variables)

            assert "find fiction books" in info
            assert "Database tables:" in info
            assert "authors" in info
            assert "books" in info

    def test_separates_input_and_database_sections(self, lm_context) -> None:
        interp = SQLInterpreter(preload_sql=SCHEMA)
        with lm_context([{"answer": "42"}]):
            rlm = SQLRLM("query -> answer", interpreter=interp)
            variables = rlm._build_variables(query="test")
            info = rlm._variables_info_for_prompt(interp, variables)

            assert "Input variables:" in info
            assert "Database tables:" in info


class TestSQLRLMPassthrough:
    def test_create_interpreter_with_preload(self, lm_context) -> None:
        with lm_context([{"answer": "42"}]):
            rlm = SQLRLM(
                "query -> answer",
                db_path=":memory:",
                preload_sql=SCHEMA,
                skip_variable_tables={"query"},
            )
            execution_tools = rlm._prepare_execution_tools()
            interp = rlm._create_interpreter(execution_tools)
            try:
                tables = interp.describe_tables()
                names = {t["name"] for t in tables}
                assert "authors" in names
                assert "books" in names
            finally:
                interp.shutdown()

    def test_full_forward_with_preloaded_schema(self, lm_context) -> None:
        """End-to-end: SQLRLM creates its own interpreter with preload_sql and
        the LLM (mocked) can interact with the pre-created tables."""
        with lm_context([{"answer": "42"}]):
            rlm = SQLRLM(
                "query -> answer",
                preload_sql=SCHEMA,
                skip_variable_tables={"query"},
            )
            rlm.generate_action = make_mock_predictor(
                [
                    {
                        "reasoning": "Insert data and query",
                        "code": (
                            "INSERT INTO authors (id, name) VALUES ('a1', 'Tolkien');\n"
                            "INSERT INTO books (id, title, author_id, genre, pages) "
                            "VALUES ('b1', 'The Hobbit', 'a1', 'fiction', 310);\n"
                            "SELECT SUBMIT(json_object('answer', "
                            "(SELECT title FROM books WHERE author_id = 'a1')));"
                        ),
                    }
                ]
            )
            result = rlm(query="find fiction books")
            assert result.answer == "The Hobbit"

    def test_fk_violation_surfaces_as_error(self, lm_context) -> None:
        """When the LLM inserts with a bad FK, it gets an error and can recover."""
        with lm_context([{"answer": "recovered"}]):
            rlm = SQLRLM(
                "query -> answer",
                preload_sql=SCHEMA,
                skip_variable_tables={"query"},
                max_iterations=3,
            )
            rlm.generate_action = make_mock_predictor(
                [
                    {
                        "reasoning": "try bad FK",
                        "code": (
                            "INSERT INTO books (id, title, author_id, genre, pages) "
                            "VALUES ('b1', 'Orphan', 'bad_id', 'fiction', 100);"
                        ),
                    },
                    {
                        "reasoning": "fix: create author first",
                        "code": (
                            "INSERT INTO authors (id, name) VALUES ('a1', 'Fixed');\n"
                            "INSERT INTO books (id, title, author_id, genre, pages) "
                            "VALUES ('b1', 'Fixed Book', 'a1', 'fiction', 100);\n"
                            "SELECT SUBMIT(json_object('answer', 'recovered'));"
                        ),
                    },
                ]
            )
            result = rlm(query="test")
            assert result.answer == "recovered"
            assert len(result.trajectory) == 2
