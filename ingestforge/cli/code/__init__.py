"""Code analysis commands package.

Provides code analysis and documentation tools:
- analyze: Analyze code structure and patterns
- map: Generate code maps and dependency graphs

Usage:
    from ingestforge.cli.code import code_app
"""

from __future__ import annotations

from ingestforge.cli.code.main import app as code_app

__all__ = ["code_app"]
