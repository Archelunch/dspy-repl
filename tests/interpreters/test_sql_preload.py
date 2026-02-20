"""Tests for SQLInterpreter preload_sql, describe_tables FK/CHECK enrichment, and skip_variable_tables."""

from __future__ import annotations

import os
import tempfile

import pytest

from dspy_repl.interpreters.sql_interpreter import SQLInterpreter

SAMPLE_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE authors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    bio TEXT
);

CREATE TABLE books (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author_id TEXT NOT NULL REFERENCES authors(id),
    genre TEXT NOT NULL CHECK(genre IN ('fiction','nonfiction','poetry')),
    pages INTEGER CHECK(pages > 0)
);

CREATE TABLE reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL REFERENCES books(id),
    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    body TEXT
);

CREATE INDEX idx_books_author ON books(author_id);
CREATE INDEX idx_reviews_book ON reviews(book_id);
"""


class TestPreloadSqlString:
    def test_tables_are_created(self) -> None:
        with SQLInterpreter(preload_sql=SAMPLE_SCHEMA) as interp:
            tables = interp.describe_tables()
            names = {t["name"] for t in tables}
            assert names == {"authors", "books", "reviews"}

    def test_tables_are_empty(self) -> None:
        with SQLInterpreter(preload_sql=SAMPLE_SCHEMA) as interp:
            tables = interp.describe_tables()
            for t in tables:
                assert t["rows"] == 0

    def test_can_insert_and_query(self) -> None:
        with SQLInterpreter(preload_sql=SAMPLE_SCHEMA) as interp:
            result = interp.execute(
                "INSERT INTO authors (id, name) VALUES ('a1', 'Alice');"
                "INSERT INTO books (id, title, author_id, genre, pages) VALUES ('b1', 'Test Book', 'a1', 'fiction', 200);"
                "SELECT title FROM books WHERE author_id = 'a1';"
            )
            assert "Test Book" in result

    def test_fk_enforcement(self) -> None:
        with SQLInterpreter(preload_sql=SAMPLE_SCHEMA) as interp:
            from dspy_repl.core.code_interpreter import CodeInterpreterError

            with pytest.raises(CodeInterpreterError, match="FOREIGN KEY"):
                interp.execute(
                    "INSERT INTO books (id, title, author_id, genre, pages) "
                    "VALUES ('b1', 'Orphan', 'nonexistent', 'fiction', 100);"
                )

    def test_check_enforcement(self) -> None:
        with SQLInterpreter(preload_sql=SAMPLE_SCHEMA) as interp:
            from dspy_repl.core.code_interpreter import CodeInterpreterError

            interp.execute("INSERT INTO authors (id, name) VALUES ('a1', 'Alice');")
            with pytest.raises(CodeInterpreterError, match="CHECK"):
                interp.execute(
                    "INSERT INTO books (id, title, author_id, genre, pages) "
                    "VALUES ('b1', 'Bad Genre', 'a1', 'romance', 100);"
                )

    def test_table_generation_incremented(self) -> None:
        with SQLInterpreter(preload_sql=SAMPLE_SCHEMA) as interp:
            interp.start()
            assert interp.table_generation() >= 1


class TestPreloadSqlFile:
    def test_loads_from_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
            f.write(SAMPLE_SCHEMA)
            f.flush()
            path = f.name
        try:
            with SQLInterpreter(preload_sql=path) as interp:
                tables = interp.describe_tables()
                names = {t["name"] for t in tables}
                assert names == {"authors", "books", "reviews"}
        finally:
            os.unlink(path)

    def test_non_sql_extension_treated_as_raw(self) -> None:
        with SQLInterpreter(preload_sql="CREATE TABLE t (x TEXT);") as interp:
            tables = interp.describe_tables()
            assert len(tables) == 1
            assert tables[0]["name"] == "t"


class TestPreloadPersistentDb:
    def test_second_open_skips_ddl_registers_tables(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            with SQLInterpreter(db_path=db_path, preload_sql=SAMPLE_SCHEMA) as interp:
                interp.execute("INSERT INTO authors (id, name) VALUES ('a1', 'Alice');")
                tables_first = {t["name"] for t in interp.describe_tables()}
                assert tables_first == {"authors", "books", "reviews"}

            with SQLInterpreter(db_path=db_path, preload_sql=SAMPLE_SCHEMA) as interp2:
                tables_second = {t["name"] for t in interp2.describe_tables()}
                assert tables_second == {"authors", "books", "reviews"}
                result = interp2.execute("SELECT name FROM authors WHERE id = 'a1';")
                assert "Alice" in result
        finally:
            os.unlink(db_path)


class TestDescribeTablesForeignKeys:
    def test_fk_info_present(self) -> None:
        with SQLInterpreter(preload_sql=SAMPLE_SCHEMA) as interp:
            tables = interp.describe_tables()
            by_name = {t["name"]: t for t in tables}

            books = by_name["books"]
            assert "foreign_keys" in books
            assert any("author_id -> authors.id" in fk for fk in books["foreign_keys"])

            reviews = by_name["reviews"]
            assert "foreign_keys" in reviews
            assert any("book_id -> books.id" in fk for fk in reviews["foreign_keys"])

    def test_no_fks_omitted(self) -> None:
        with SQLInterpreter(preload_sql=SAMPLE_SCHEMA) as interp:
            tables = interp.describe_tables()
            by_name = {t["name"]: t for t in tables}
            authors = by_name["authors"]
            assert "foreign_keys" not in authors

    def test_check_constraints_present(self) -> None:
        with SQLInterpreter(preload_sql=SAMPLE_SCHEMA) as interp:
            tables = interp.describe_tables()
            by_name = {t["name"]: t for t in tables}

            books = by_name["books"]
            assert "checks" in books
            checks_joined = " ".join(books["checks"])
            assert "genre" in checks_joined.lower() or "fiction" in checks_joined.lower()

            reviews = by_name["reviews"]
            assert "checks" in reviews
            checks_joined = " ".join(reviews["checks"])
            assert "rating" in checks_joined.lower()

    def test_no_checks_omitted(self) -> None:
        with SQLInterpreter(preload_sql="CREATE TABLE simple (id TEXT PRIMARY KEY, val TEXT);") as interp:
            tables = interp.describe_tables()
            assert "checks" not in tables[0]

    def test_describe_specific_table(self) -> None:
        with SQLInterpreter(preload_sql=SAMPLE_SCHEMA) as interp:
            tables = interp.describe_tables(["books"])
            assert len(tables) == 1
            assert tables[0]["name"] == "books"
            assert "foreign_keys" in tables[0]


class TestSkipVariableTables:
    def test_skipped_variables_dont_create_tables(self) -> None:
        with SQLInterpreter(
            preload_sql="CREATE TABLE items (id TEXT PRIMARY KEY, val TEXT);",
            skip_variable_tables={"query", "config"},
        ) as interp:
            interp.execute(
                "INSERT INTO items (id, val) VALUES ('i1', 'test');",
                variables={"query": "find all items", "config": {"batch_size": 10}},
            )
            tables = interp.describe_tables()
            names = {t["name"] for t in tables}
            assert "items" in names
            assert "query" not in names
            assert "config" not in names

    def test_non_skipped_variables_still_work(self) -> None:
        with SQLInterpreter(
            preload_sql="CREATE TABLE items (id TEXT PRIMARY KEY);",
            skip_variable_tables={"theme"},
        ) as interp:
            result = interp.execute(
                "SELECT value FROM data WHERE idx = 0;",
                variables={"theme": "dark fantasy", "data": [10, 20, 30]},
            )
            assert "10" in result

    def test_empty_skip_set_is_noop(self) -> None:
        with SQLInterpreter(skip_variable_tables=set()) as interp:
            result = interp.execute(
                "SELECT value FROM x;",
                variables={"x": "hello"},
            )
            assert "hello" in result


class TestBackwardsCompatibility:
    def test_default_behavior_unchanged(self) -> None:
        with SQLInterpreter() as interp:
            result = interp.execute(
                "SELECT value FROM greeting;",
                variables={"greeting": "hello world"},
            )
            assert "hello world" in result

    def test_preload_none_is_noop(self) -> None:
        with SQLInterpreter(preload_sql=None) as interp:
            tables = interp.describe_tables()
            assert tables == []

    def test_describe_tables_still_works_for_variable_tables(self) -> None:
        with SQLInterpreter() as interp:
            interp.execute(
                "SELECT COUNT(*) FROM users;",
                variables={"users": [{"name": "Alice"}, {"name": "Bob"}]},
            )
            tables = interp.describe_tables()
            assert len(tables) == 1
            assert tables[0]["name"] == "users"
            assert tables[0]["rows"] == 2
            assert "foreign_keys" not in tables[0]
            assert "checks" not in tables[0]


class TestPreloadWithTools:
    def test_tools_work_with_preloaded_schema(self) -> None:
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        with SQLInterpreter(
            preload_sql="CREATE TABLE people (id TEXT PRIMARY KEY, name TEXT NOT NULL);",
            tools={"greet": greet},
        ) as interp:
            interp.execute("INSERT INTO people (id, name) VALUES ('p1', 'Alice');")
            result = interp.execute("SELECT greet(name) AS msg FROM people WHERE id = 'p1';")
            assert "Hello, Alice!" in result
