# LLM Adapter for ReActEngine

The LLM Adapter (`llm_adapter.py`) connects existing `LLMClient` implementations to the `ReActEngine`'s `ThinkFunction` Protocol, enabling LLM-powered autonomous reasoning.

## Overview

This adapter bridges the gap between IngestForge's LLM abstraction layer and the ReAct reasoning engine, allowing any LLM provider (Claude, OpenAI, Ollama, etc.) to power autonomous agent behavior.

## Architecture

```
┌─────────────────┐
│   LLMClient     │  (Ollama, Claude, OpenAI, etc.)
│   (llm/base.py) │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ LLMThinkAdapter │  Converts task + history → ReAct prompt
│                 │  Parses LLM response → (thought, action, input)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  ReActEngine    │  Executes Reason-Act-Observe loop
│ react_engine.py │
└─────────────────┘
```

## ReAct Format

The adapter expects LLM responses in this format:

```
Thought: [reasoning about what to do next]
Action: [tool_name or "FINISH"]
Action Input: {"param1": "value1", "param2": "value2"}
```

### Example Response

```
Thought: I need to find information about solar panels.
Action: search_documents
Action Input: {"query": "solar panel efficiency"}
```

### Completion Signal

When the task is complete:

```
Thought: I now have enough information to answer.
Action: FINISH
Action Input: {}
```

## Usage

### Basic Example

```python
from ingestforge.agent.llm_adapter import create_llm_think_adapter
from ingestforge.agent.react_engine import ReActEngine, SimpleTool
from ingestforge.llm.ollama import OllamaClient

# 1. Create LLM client
llm = OllamaClient(model="qwen2.5:14b")

# 2. Create adapter
adapter = create_llm_think_adapter(llm_client=llm)

# 3. Create ReAct engine with adapter
engine = ReActEngine(think_fn=adapter)

# 4. Register tools
search_tool = SimpleTool(
    name="search",
    description="Search knowledge base",
    _fn=lambda query: f"Results for {query}",
)
engine.register_tool(search_tool)

# 5. Run agent
result = engine.run("Find information about X")
print(result.final_answer)
```

### With Custom Configuration

```python
from ingestforge.llm.base import GenerationConfig

# Configure generation parameters
config = GenerationConfig(
    max_tokens=1000,
    temperature=0.7,  # Higher for more creative reasoning
    stop_sequences=["Observation:"],  # Stop before observation
)

adapter = create_llm_think_adapter(llm_client=llm, config=config)
```

### With Different LLM Providers

```python
# Ollama (local)
from ingestforge.llm.ollama import OllamaClient
llm = OllamaClient(model="qwen2.5:14b")

# Claude (API)
from ingestforge.llm.claude import ClaudeClient
llm = ClaudeClient(api_key="...", model="claude-3-5-sonnet-20241022")

# OpenAI (API)
from ingestforge.llm.openai import OpenAIClient
llm = OpenAIClient(api_key="...", model="gpt-4o")

# All work the same way with the adapter
adapter = create_llm_think_adapter(llm_client=llm)
```

## Implementation Details

### Prompt Building

The adapter builds prompts that:

1. Include system instructions explaining the ReAct format
2. List available tools
3. Provide conversation history (last 10 steps)
4. Request the next thought/action

### Response Parsing

The adapter uses regex to extract:

- **Thought**: Captures text after `Thought:` until next section
- **Action**: Captures tool name after `Action:`
- **Action Input**: Parses JSON dict after `Action Input:`

### Error Handling

The adapter handles:

- **Empty responses**: Returns default thought with no action
- **Missing fields**: Provides sensible defaults
- **Invalid JSON**: Returns empty dict for action_input
- **LLM errors**: Logs error and returns error thought

### Safety Bounds

Following NASA JPL Commandments:

- `MAX_PROMPT_LENGTH = 16000`: Prevents token overflow
- `MAX_HISTORY_STEPS = 10`: Limits history to recent context
- Parameter validation on initialization
- Graceful degradation on parse failures

## Testing

Comprehensive tests in `tests/unit/agent/test_llm_adapter.py`:

```bash
pytest tests/unit/agent/test_llm_adapter.py -v
```

Test coverage includes:

- Adapter creation and configuration
- ReAct format parsing (complete, partial, malformed)
- Prompt building with history and tools
- Error handling and edge cases
- Integration with ReActEngine

## 
- **Rule #1**: Max 3 nesting levels, early returns
- **Rule #2**: Fixed upper bounds (MAX_PROMPT_LENGTH, MAX_HISTORY_STEPS)
- **Rule #4**: All functions <60 lines
- **Rule #7**: Parameter validation before operations
- **Rule #9**: Complete type hints throughout

## Related Components

- `react_engine.py`: Core ReAct loop implementation
- `llm/base.py`: LLM client interface
- `tool_registry.py`: Tool management for agents
- Task 2.1.2: Knowledge base tools (next step)

## Example: Full Research Agent

```python
from ingestforge.agent.llm_adapter import create_llm_think_adapter
from ingestforge.agent.react_engine import ReActEngine, SimpleTool
from ingestforge.llm.ollama import OllamaClient

# Create LLM and adapter
llm = OllamaClient(model="qwen2.5:14b")
adapter = create_llm_think_adapter(llm)

# Create engine
engine = ReActEngine(think_fn=adapter, max_iterations=20)

# Register research tools
engine.register_tool(SimpleTool(
    name="search_knowledge_base",
    description="Search indexed documents",
    _fn=lambda query: search_chunks(query),
))

engine.register_tool(SimpleTool(
    name="analyze_source",
    description="Analyze a specific source",
    _fn=lambda source_id: analyze_document(source_id),
))

# Run autonomous research
result = engine.run(
    "Research the impact of climate change on agriculture "
    "and provide a summary with citations"
)

# Get detailed results
for step in result.steps:
    print(f"Thought: {step.thought}")
    print(f"Action: {step.action}")
    print(f"Observation: {step.observation}\n")

print(f"Final Answer: {result.final_answer}")
```

## Future Enhancements

Planned improvements (see DEV_QUEUE.md):

- **Tool descriptions**: Include tool descriptions in prompts
- **Few-shot examples**: Add example ReAct traces
- **Retry logic**: Retry on parsing failures
- **Structured output**: Force JSON mode for reliable parsing
- **Token counting**: Estimate tokens before generation
