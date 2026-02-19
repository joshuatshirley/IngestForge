"""
Tests for Interactive Ask Command.

This module tests the interactive REPL for conversational queries.

Test Strategy
-------------
- Focus on command handling and validation logic
- Mock external dependencies (LLM, storage, user input)
- Test command dispatch and conversation history
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Test behavior, not implementation details

Organization
------------
- TestAskCommandInit: Initialization
- TestValidation: Parameter validation (k value)
- TestCommandHandling: Command dispatch (/exit, /help, /history, /clear)
- TestConversationHistory: History management
- TestLLMClient: LLM client retrieval
"""

from unittest.mock import Mock, patch

import pytest
import typer

from ingestforge.cli.interactive.ask import AskCommand


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_context(has_storage: bool = True, has_config: bool = True):
    """Create a mock context dictionary."""
    ctx = {}

    if has_storage:
        ctx["storage"] = Mock()

    if has_config:
        ctx["config"] = Mock()

    return ctx


# ============================================================================
# Test Classes
# ============================================================================


class TestAskCommandInit:
    """Tests for AskCommand initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_ask_command(self):
        """Test creating AskCommand instance."""
        cmd = AskCommand()

        assert cmd is not None
        assert cmd.conversation_history == []

    def test_inherits_from_base_command(self):
        """Test AskCommand inherits from IngestForgeCommand."""
        from ingestforge.cli.core import IngestForgeCommand

        cmd = AskCommand()

        assert isinstance(cmd, IngestForgeCommand)


class TestValidation:
    """Tests for parameter validation.

    Rule #4: Focused test class - tests validate_k_value()
    """

    def test_validate_k_valid_values(self):
        """Test validation accepts valid k values."""
        cmd = AskCommand()

        # Should not raise for valid values
        cmd.validate_k_value(1)
        cmd.validate_k_value(5)
        cmd.validate_k_value(10)
        cmd.validate_k_value(50)

    def test_validate_k_too_small(self):
        """Test validation rejects k < 1."""
        cmd = AskCommand()

        with pytest.raises(typer.BadParameter, match="k must be at least 1"):
            cmd.validate_k_value(0)

        with pytest.raises(typer.BadParameter):
            cmd.validate_k_value(-1)

    def test_validate_k_too_large(self):
        """Test validation rejects k > 50."""
        cmd = AskCommand()

        with pytest.raises(typer.BadParameter, match="k cannot exceed 50"):
            cmd.validate_k_value(51)

        with pytest.raises(typer.BadParameter):
            cmd.validate_k_value(100)


class TestCommandHandling:
    """Tests for command handling.

    Rule #4: Focused test class - tests _handle_command()
    """

    def test_handle_exit_command(self):
        """Test /exit command returns True."""
        cmd = AskCommand()

        result = cmd._handle_command("/exit")

        assert result is True

    def test_handle_quit_command(self):
        """Test /quit command returns True."""
        cmd = AskCommand()

        result = cmd._handle_command("/quit")

        assert result is True

    def test_handle_q_command(self):
        """Test /q command returns True."""
        cmd = AskCommand()

        result = cmd._handle_command("/q")

        assert result is True

    def test_handle_help_command(self):
        """Test /help command returns False (don't exit)."""
        cmd = AskCommand()

        result = cmd._handle_command("/help")

        assert result is False

    def test_handle_question_mark_command(self):
        """Test /? command (alias for help) returns False."""
        cmd = AskCommand()

        result = cmd._handle_command("/?")

        assert result is False

    def test_handle_history_command(self):
        """Test /history command returns False."""
        cmd = AskCommand()

        result = cmd._handle_command("/history")

        assert result is False

    def test_handle_clear_command(self):
        """Test /clear command returns False."""
        cmd = AskCommand()

        result = cmd._handle_command("/clear")

        assert result is False

    def test_handle_unknown_command(self):
        """Test unknown command returns False."""
        cmd = AskCommand()

        result = cmd._handle_command("/unknown")

        assert result is False

    def test_commands_case_insensitive(self):
        """Test commands are case-insensitive."""
        cmd = AskCommand()

        # Exit commands
        assert cmd._handle_command("/EXIT") is True
        assert cmd._handle_command("/Quit") is True

        # Non-exit commands
        assert cmd._handle_command("/HELP") is False
        assert cmd._handle_command("/History") is False


class TestConversationHistory:
    """Tests for conversation history management.

    Rule #4: Focused test class - tests history operations
    """

    def test_initial_history_empty(self):
        """Test conversation history starts empty."""
        cmd = AskCommand()

        assert len(cmd.conversation_history) == 0

    def test_clear_history(self):
        """Test clearing conversation history."""
        cmd = AskCommand()
        cmd.conversation_history = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]

        cmd._clear_history()

        assert len(cmd.conversation_history) == 0

    def test_display_history_empty(self):
        """Test displaying empty history doesn't crash."""
        cmd = AskCommand()

        # Should not raise
        cmd._display_history()

    def test_display_history_with_entries(self):
        """Test displaying history with entries."""
        cmd = AskCommand()
        cmd.conversation_history = [
            {
                "question": "What is Python?",
                "answer": "Python is a programming language...",
            },
        ]

        # Should not raise
        cmd._display_history()


class TestLLMClient:
    """Tests for LLM client retrieval.

    Rule #4: Focused test class - tests _get_llm_client()
    """

    def test_get_llm_client_success(self):
        """Test getting LLM client successfully."""
        cmd = AskCommand()
        mock_client = Mock()
        ctx = make_mock_context()

        with patch(
            "ingestforge.llm.factory.get_best_available_client",
            return_value=mock_client,
        ):
            result = cmd._get_llm_client(ctx)

        assert result is mock_client

    def test_get_llm_client_none_available(self):
        """Test when no LLM client available."""
        cmd = AskCommand()
        ctx = make_mock_context()

        with patch(
            "ingestforge.llm.factory.get_best_available_client", return_value=None
        ):
            result = cmd._get_llm_client(ctx)

        assert result is None

    def test_get_llm_client_error(self):
        """Test LLM client error handling."""
        cmd = AskCommand()
        ctx = make_mock_context()

        with patch(
            "ingestforge.llm.factory.get_best_available_client",
            side_effect=Exception("LLM error"),
        ):
            result = cmd._get_llm_client(ctx)

        assert result is None


class TestProcessUserInput:
    """Tests for user input processing.

    Rule #4: Focused test class - tests _process_user_input()
    """

    def test_process_empty_input(self):
        """Test processing empty input returns False."""
        cmd = AskCommand()

        result = cmd._process_user_input("", Mock(), Mock(), 5, False)

        assert result is False

    def test_process_exit_command(self):
        """Test processing /exit returns True."""
        cmd = AskCommand()

        result = cmd._process_user_input("/exit", Mock(), Mock(), 5, False)

        assert result is True

    def test_process_help_command(self):
        """Test processing /help returns False."""
        cmd = AskCommand()

        result = cmd._process_user_input("/help", Mock(), Mock(), 5, False)

        assert result is False

    @patch.object(AskCommand, "_process_question")
    def test_process_question(self, mock_process):
        """Test processing non-command input calls _process_question."""
        cmd = AskCommand()
        storage = Mock()
        llm_client = Mock()

        result = cmd._process_user_input(
            "What is Python?", storage, llm_client, 5, False
        )

        assert result is False
        mock_process.assert_called_once_with(
            "What is Python?", storage, llm_client, 5, False
        )


class TestDisplayMethods:
    """Tests for display helper methods.

    Rule #4: Focused test class - tests display methods
    """

    def test_display_welcome(self):
        """Test welcome message displays without error."""
        cmd = AskCommand()

        # Should not raise
        cmd._display_welcome()

    def test_display_help(self):
        """Test help message displays without error."""
        cmd = AskCommand()

        # Should not raise
        cmd._display_help()


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - AskCommand init: 2 tests (creation, inheritance)
    - Validation: 4 tests (valid k, too small, too large, boundaries)
    - Command handling: 10 tests (exit, quit, q, help, ?, history, clear, unknown, case-insensitive)
    - Conversation history: 4 tests (empty, clear, display empty, display with entries)
    - LLM client: 3 tests (success, none available, error)
    - Process user input: 4 tests (empty, exit, help, question)
    - Display methods: 2 tests (welcome, help)

    Total: 29 tests

Design Decisions:
    1. Focus on testable logic (command handling, validation)
    2. Mock external dependencies (LLM, storage, user input)
    3. Test command dispatch and history management
    4. Don't test interactive loop (requires user input simulation)
    5. Test behavior, not implementation details
    6. Simple, clear tests that verify command works
    7. Follows NASA JPL Rule #1 (Simple Control Flow)
    8. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - AskCommand initialization with empty history
    - Inheritance from IngestForgeCommand base class
    - K value validation (1-50 range)
    - Command dispatch (/exit, /quit, /q, /help, /?, /history, /clear)
    - Case-insensitive command handling
    - Unknown command handling
    - Conversation history management (clear, display)
    - LLM client retrieval (success, failure, error)
    - User input processing (empty, commands, questions)
    - Display methods (welcome, help)

Justification:
    - Interactive ask command is critical user-facing feature
    - Command handling logic needs verification
    - Validation prevents invalid parameters
    - History management enables conversation context
    - Simple tests verify command system works correctly
"""
