"""Modular non-Python REPL engines for DSPy."""

from dspy_repl.languages.haskell_rlm import HaskellRLM
from dspy_repl.languages.js_rlm import JavaScriptRLM
from dspy_repl.languages.scheme_rlm import SchemeRLM
from dspy_repl.languages.sql_rlm import SQLRLM

__all__ = ["SchemeRLM", "SQLRLM", "HaskellRLM", "JavaScriptRLM"]
__version__ = "0.4.0"
