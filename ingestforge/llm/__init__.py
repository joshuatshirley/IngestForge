"""
LLM Provider Integrations.

This module provides a unified interface to various LLM providers for text
generation, question answering, and summarization tasks.

Architecture Position
---------------------
    CLI (outermost)
      └── **Feature Modules** (you are here)
            └── Shared (patterns, interfaces, utilities)
                  └── Core (innermost)

LLM Role in Pipeline

    ┌─────────────────────────────────────────────────────────────────┐
    │                      Query Processing                            │
    │  Retrieve chunks → Build context → LLM generates answer         │
    └─────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │       LLM Provider        │
                    │  (Claude/OpenAI/Ollama)   │
                    └───────────────────────────┘

Supported Providers
-------------------
**Cloud Providers** (require API keys)
- ClaudeClient: Anthropic Claude models (claude-sonnet-4-20250514, opus)
- OpenAIClient: OpenAI GPT models (gpt-4, gpt-3.5-turbo)
- GeminiClient: Google Gemini models (gemini-pro)

**Local Providers** (run on your machine)
- OllamaClient: Ollama-served models (llama2, mistral, etc.)
- LlamaCppClient: Direct llama.cpp integration

Provider Selection
------------------
Providers are selected via configuration:

    llm:
      provider: claude         # claude, openai, gemini, ollama, llamacpp
      model: claude-sonnet-4-20250514
      api_key: ${ANTHROPIC_API_KEY}

Or programmatically:

    from ingestforge.llm import get_llm_client
    client = get_llm_client(config)

Key Components
--------------
**LLMClient (Base Class)**
    Abstract interface all providers implement:
    - generate(): Text generation
    - chat(): Multi-turn conversation
    - embed(): Get embeddings (if supported)

**LLMError**
    Base exception for LLM-related errors.

**RateLimitError**
    Raised when API rate limits are hit.
    The retry decorator handles automatic backoff.

Factory Function
----------------
    from ingestforge.llm import get_llm_client

    # Automatically selects based on config
    client = get_llm_client(config)

    # Generate text
    response = client.generate(
        prompt="Summarize this text: ...",
        max_tokens=500,
    )

Usage Example
-------------
    from ingestforge.llm import get_llm_client

    # Get client from config
    client = get_llm_client(config)

    # Simple generation
    response = client.generate("Explain quantum computing in simple terms")
    print(response)

    # With system prompt
    response = client.generate(
        prompt="What are the key points?",
        system="You are a helpful research assistant.",
        context=retrieved_chunks,
    )

    # Multi-turn chat
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "What is RAG?"},
    ]
    response = client.chat(messages)

Retry and Rate Limiting
-----------------------
All LLM clients use the @llm_retry decorator from core.retry:
- Automatic retry on transient errors
- Exponential backoff with jitter
- Configurable retry count and delays

    # Built into all clients - no manual handling needed
    response = client.generate(prompt)  # Retries automatically

Configuration Examples
----------------------
    # Claude (cloud)
    llm:
      provider: claude
      model: claude-sonnet-4-20250514
      api_key: ${ANTHROPIC_API_KEY}

    # Ollama (local)
    llm:
      provider: ollama
      model: llama2
      base_url: http://localhost:11434

    # OpenAI (cloud)
    llm:
      provider: openai
      model: gpt-4
      api_key: ${OPENAI_API_KEY}
"""

from ingestforge.llm.base import LLMClient, LLMError, RateLimitError
from ingestforge.llm.factory import get_llm_client

__all__ = ["LLMClient", "LLMError", "RateLimitError", "get_llm_client"]
