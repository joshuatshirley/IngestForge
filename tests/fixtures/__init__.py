"""
Fixture modules for IngestForge tests.

This package provides specialized fixtures for testing different
components of the IngestForge system.

Modules
-------
- documents: Sample document generators (PDF, DOCX, PPTX, HTML, etc.)
- mocks: Reusable mock objects and responses
- agents: Mock LLM implementations for testing agents
"""

from tests.fixtures.agents import MockLLM

__all__ = ["MockLLM"]
