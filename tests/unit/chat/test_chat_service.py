"""Comprehensive GWT unit tests for ChatService.

Conversational Query Mode.
"""

import pytest
from unittest.mock import MagicMock, patch
from ingestforge.chat.service import ChatService, ChatMessage, ChatResponse
from ingestforge.storage.base import SearchResult


@pytest.fixture
def mock_llm():
    client = MagicMock()
    # Mock response object
    res = MagicMock()
    res.text = "Mocked Response"
    client.generate.return_value = res
    return client


@pytest.fixture
def chat_service(mock_llm):
    with patch("ingestforge.chat.service.load_config"), patch(
        "ingestforge.llm.factory.get_llm_client", return_value=mock_llm
    ):
        service = ChatService()
        service._retriever = MagicMock()
        return service


# =============================================================================
# CHAT TURN TESTS
# =============================================================================


def test_chat_standalone_query(chat_service, mock_llm):
    """GIVEN a user query with no history
    WHEN chat_service.chat is called
    THEN it returns a response without rewriting (or identical rewrite).
    """
    history = []
    user_query = "What is quantum gravity?"

    # Mock search results
    chat_service.retriever.search.return_value = [
        SearchResult(
            chunk_id="1",
            content="Gravity is...",
            score=0.9,
            document_id="doc1",
            section_title="",
            chunk_type="",
            source_file="",
            word_count=10,
        )
    ]

    response = chat_service.chat(history, user_query)

    assert isinstance(response, ChatResponse)
    assert response.answer == "Mocked Response"
    assert len(response.sources) == 1
    # Verify rewrite was called
    assert mock_llm.generate.called


def test_query_contextualization(chat_service, mock_llm):
    """GIVEN a follow-up query "What about him?" and history about "Einstein"
    WHEN _rewrite_query is called
    THEN it uses the LLM to create a standalone query.
    """
    history = [
        ChatMessage(role="user", content="Who is Albert Einstein?"),
        ChatMessage(role="ai", content="He was a physicist."),
    ]
    follow_up = "What about him?"

    # Mock LLM to return the contextualized version
    res = MagicMock()
    res.text = "Tell me more about Albert Einstein's life."
    mock_llm.generate.return_value = res

    rewritten = chat_service._rewrite_query(history, follow_up)

    assert "Albert Einstein" in rewritten
    # Verify prompt contained history
    args, _ = mock_llm.generate.call_args
    assert "Who is Albert Einstein?" in args[0]


def test_generate_answer_no_sources(chat_service, mock_llm):
    """GIVEN no retrieved sources
    WHEN _generate_answer is called
    THEN it returns a helpful "no info" message.
    """
    # LLM says it doesn't know
    res = MagicMock()
    res.text = "I don't know."
    mock_llm.generate.return_value = res

    answer = chat_service._generate_answer([], "Query", [])
    assert "I don't know" in answer
