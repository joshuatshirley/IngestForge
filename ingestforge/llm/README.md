# LLM Module

## Purpose

LLM client integrations for Gemini, Claude, OpenAI, Ollama, and llama.cpp. Provides unified interface with automatic retry, rate limiting, and error handling.

## Architecture Context

```
┌─────────────────────────────────────────┐
│   query/ - Query pipeline               │
│            ↓                             │
│   llm/ - Generate answers               │  ← You are here
│   (Gemini, Claude, OpenAI, local)       │
│            ↓                             │
│   Answer with citations                 │
└─────────────────────────────────────────┘
```

## Key Components

| Component | Purpose | Provider |
|-----------|---------|----------|
| `base.py` | LLMClient abstract interface | All |
| `gemini.py` | GeminiClient | Google Gemini |
| `claude.py` | ClaudeClient | Anthropic Claude |
| `openai.py` | OpenAIClient | OpenAI GPT |
| `ollama.py` | OllamaClient | Ollama (local) |
| `llamacpp.py` | LlamaCppClient | llama.cpp (local) |
| `factory.py` | get_llm_client - auto-select | Factory |

## LLMClient Interface

All providers implement:

```python
class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> str
        """Generate text from prompt."""

    @abstractmethod
    def is_available(self) -> bool
        """Check if client is configured and ready."""
```

## Supported Providers

### Gemini (Default)

```python
from ingestforge.llm import get_llm_client

config.llm.default_provider = "gemini"
config.llm.gemini.api_key = "your-key"
config.llm.gemini.model = "gemini-1.5-flash"

client = get_llm_client(config)
answer = client.generate("Explain quantum computing")
```

### Claude

```python
config.llm.default_provider = "claude"
config.llm.claude.api_key = "your-key"
config.llm.claude.model = "claude-3-haiku-20240307"

client = get_llm_client(config)
answer = client.generate("Explain quantum computing")
```

### OpenAI

```python
config.llm.default_provider = "openai"
config.llm.openai.api_key = "your-key"
config.llm.openai.model = "gpt-4o-mini"

client = get_llm_client(config)
answer = client.generate("Explain quantum computing")
```

### Ollama (Local)

```python
config.llm.default_provider = "ollama"
config.llm.ollama.model = "llama3:latest"
config.llm.ollama.url = "http://localhost:11434"

client = get_llm_client(config)
answer = client.generate("Explain quantum computing")
```

### llama.cpp (Local)

```python
config.llm.default_provider = "llamacpp"
config.llm.llamacpp.model_path = "/path/to/model.gguf"
config.llm.llamacpp.n_ctx = 2048
config.llm.llamacpp.n_gpu_layers = 35

client = get_llm_client(config)
answer = client.generate("Explain quantum computing")
```

## Configuration

```yaml
# config.yaml
llm:
  default_provider: gemini  # gemini, claude, openai, ollama, llamacpp

  gemini:
    model: gemini-1.5-flash
    api_key: ${GEMINI_API_KEY}

  claude:
    model: claude-3-haiku-20240307
    api_key: ${ANTHROPIC_API_KEY}

  openai:
    model: gpt-4o-mini
    api_key: ${OPENAI_API_KEY}

  ollama:
    model: llama3:latest
    url: http://localhost:11434

  llamacpp:
    model_path: ""
    n_ctx: 2048
    n_gpu_layers: 35
```

## Automatic Retry

All clients use `@llm_retry` decorator from `core/retry.py`:

```python
# Automatic retry with exponential backoff
# 3 attempts, 2s base delay, 30s max
@llm_retry
def generate(self, prompt):
    return self.client.generate(prompt)
```

## Error Handling

```python
from ingestforge.llm import get_llm_client, LLMError, RateLimitError

client = get_llm_client(config)

try:
    answer = client.generate("...")
except RateLimitError:
    print("Rate limit hit - retry later")
except LLMError as e:
    print(f"LLM error: {e}")
```

## Usage Examples

### Example 1: RAG Answer Generation

```python
# Generate answer with context
def generate_answer(query, chunks):
    # Build context from chunks
    context = "\n\n".join([
        f"[{i+1}] {chunk.content}"
        for i, chunk in enumerate(chunks)
    ])

    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

    client = get_llm_client(config)
    answer = client.generate(prompt, max_tokens=500)
    return answer

# Use it
chunks = retriever.search("quantum computing", top_k=5)
answer = generate_answer("What is quantum computing?", chunks)
```

### Example 2: Provider Fallback

```python
providers = ["gemini", "claude", "openai"]

for provider in providers:
    try:
        config.llm.default_provider = provider
        client = get_llm_client(config)

        if client.is_available:
            answer = client.generate(prompt)
            break
    except Exception as e:
        logger.warning(f"{provider} failed: {e}")
        continue
else:
    raise RuntimeError("All LLM providers failed")
```

## Dependencies

### Required
None - all providers are optional

### Optional
- `google-generativeai` - Gemini
- `anthropic` - Claude
- `openai` - OpenAI
- `requests` - Ollama
- `llama-cpp-python` - llama.cpp

### Installation

```bash
# Cloud providers
pip install google-generativeai anthropic openai

# Local providers
pip install requests  # Ollama
pip install llama-cpp-python  # llama.cpp
```

## Testing

```bash
pytest tests/test_llm_*.py -v
```

## References

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - System overview
- [ADR-006: LLM Retry](../../docs/architecture/ADR-006-llm-retry-consolidation.md)
