"""
Example: Using SQLRLM with a pre-loaded SQL schema.

This demonstrates the new features:
  1. preload_sql    -- Pre-create tables with FKs and CHECKs before the LLM starts.
  2. db_path        -- Persist the database to a file for incremental work.
  3. skip_variable_tables -- Pass string inputs as prompt context, not as SQL tables.
  4. describe_tables -- Now returns foreign_keys and checks metadata.

Without these features, the LLM would need to spend iterations writing CREATE TABLE
DDL, and it wouldn't see the schema in its prompt. Now the schema is ready from
iteration 1, and the LLM sees table structures including FK relationships.
"""

from dspy_repl.interpreters.sql_interpreter import SQLInterpreter

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE departments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    budget REAL CHECK(budget >= 0)
);

CREATE TABLE employees (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    department_id TEXT NOT NULL REFERENCES departments(id),
    role TEXT NOT NULL CHECK(role IN ('engineer','manager','designer','analyst')),
    salary REAL CHECK(salary > 0)
);

CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    lead_id TEXT NOT NULL REFERENCES employees(id),
    status TEXT NOT NULL CHECK(status IN ('active','completed','cancelled'))
);
"""


def example_interpreter_direct() -> None:
    """Use SQLInterpreter directly with preload_sql."""
    print("=== SQLInterpreter with preload_sql ===\n")

    with SQLInterpreter(preload_sql=SCHEMA) as interp:
        tables = interp.describe_tables()
        for t in tables:
            print(f"Table: {t['name']}")
            print(f"  Columns: {', '.join(t['columns'])}")
            print(f"  Rows: {t['rows']}")
            if t.get("foreign_keys"):
                print(f"  FKs: {', '.join(t['foreign_keys'])}")
            if t.get("checks"):
                print(f"  CHECKs: {'; '.join(t['checks'])}")
            print()

        interp.execute(
            "INSERT INTO departments (id, name, budget) VALUES ('d1', 'Engineering', 500000);"
            "INSERT INTO employees (id, name, department_id, role, salary) "
            "VALUES ('e1', 'Alice', 'd1', 'engineer', 120000);"
            "INSERT INTO projects (id, name, lead_id, status) "
            "VALUES ('p1', 'Alpha', 'e1', 'active');"
        )

        result = interp.execute(
            "SELECT e.name, d.name AS dept, p.name AS project "
            "FROM employees e "
            "JOIN departments d ON e.department_id = d.id "
            "JOIN projects p ON p.lead_id = e.id;"
        )
        print("Query result:")
        print(result)
        print()

        from dspy_repl.core.code_interpreter import CodeInterpreterError

        print("Attempting FK violation (insert employee with bad department)...")
        try:
            interp.execute(
                "INSERT INTO employees (id, name, department_id, role, salary) "
                "VALUES ('e2', 'Bob', 'nonexistent', 'manager', 100000);"
            )
        except CodeInterpreterError as e:
            print(f"  Caught: {e}\n")

        print("Attempting CHECK violation (invalid role)...")
        try:
            interp.execute(
                "INSERT INTO employees (id, name, department_id, role, salary) "
                "VALUES ('e2', 'Bob', 'd1', 'intern', 50000);"
            )
        except CodeInterpreterError as e:
            print(f"  Caught: {e}\n")


def example_skip_variable_tables() -> None:
    """Demonstrate skip_variable_tables preventing useless table creation."""
    print("=== skip_variable_tables ===\n")

    with SQLInterpreter(
        preload_sql=SCHEMA,
        skip_variable_tables={"query_text", "config"},
    ) as interp:
        interp.execute(
            "INSERT INTO departments (id, name, budget) VALUES ('d1', 'Engineering', 500000);",
            variables={
                "query_text": "Find all active projects led by engineers",
                "config": {"max_results": 10, "sort": "name"},
            },
        )

        tables = interp.describe_tables()
        table_names = {t["name"] for t in tables}
        print(f"Tables present: {sorted(table_names)}")
        print(f"  'query_text' as table? {'query_text' in table_names}")
        print(f"  'config' as table? {'config' in table_names}")
        print()


def example_persistent_db() -> None:
    """Demonstrate file-based persistence."""
    import os
    import tempfile

    print("=== Persistent DB ===\n")

    db_path = os.path.join(tempfile.gettempdir(), "example_preload.db")
    if os.path.exists(db_path):
        os.unlink(db_path)

    with SQLInterpreter(db_path=db_path, preload_sql=SCHEMA) as interp:
        interp.execute(
            "INSERT INTO departments (id, name, budget) VALUES ('d1', 'Engineering', 500000);"
            "INSERT INTO employees (id, name, department_id, role, salary) "
            "VALUES ('e1', 'Alice', 'd1', 'engineer', 120000);"
        )
        print(f"Session 1: inserted 1 department + 1 employee into {db_path}")

    with SQLInterpreter(db_path=db_path, preload_sql=SCHEMA) as interp2:
        tables = interp2.describe_tables()
        for t in tables:
            print(f"  Session 2 sees: {t['name']} ({t['rows']} rows)")

        result = interp2.execute("SELECT name FROM employees;")
        print(f"\n  Employees from session 2:\n{result}")

    os.unlink(db_path)
    print()


def example_sqlrlm_usage() -> None:
    """Show how SQLRLM constructor accepts the new parameters.

    NOTE: This example only demonstrates the API shape.
    Running it requires a configured LM (e.g., dspy.configure(lm=...)).
    """
    print("=== SQLRLM API (constructor only, no LLM call) ===\n")

    print(
        "SQLRLM(\n"
        '    "query -> answer",\n'
        '    db_path="data/my_project.db",\n'
        '    preload_sql="schema.sql",\n'
        '    skip_variable_tables={"query", "config"},\n'
        "    tools=[my_tool_fn],\n"
        "    max_iterations=15,\n"
        ")\n"
    )
    print("The LLM sees in its prompt from iteration 1:")
    print("  Input variables:")
    print('  - query: "Find all active projects..."')
    print("  - config: {max_results: 10}")
    print()
    print("  Database tables:")
    print("  - departments (id TEXT, name TEXT, budget REAL) -- 0 rows")
    print("    CHECKs: budget >= 0")
    print("  - employees (id TEXT, name TEXT, department_id TEXT, role TEXT, salary REAL) -- 0 rows")
    print("    FKs: department_id -> departments.id")
    print("    CHECKs: role IN ('engineer','manager','designer','analyst'); salary > 0")
    print("  - projects (id TEXT, name TEXT, lead_id TEXT, status TEXT) -- 0 rows")
    print("    FKs: lead_id -> employees.id")
    print("    CHECKs: status IN ('active','completed','cancelled')")
    print()


if __name__ == "__main__":
    example_interpreter_direct()
    example_skip_variable_tables()
    example_persistent_db()
    example_sqlrlm_usage()
