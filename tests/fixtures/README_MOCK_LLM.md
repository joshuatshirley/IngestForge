# MockLLM Usage Guide

The `MockLLM` class provides a deterministic LLM client for testing ReAct agents and other components that depend on LLM responses.

## Features

- **Deterministic responses**: Queue responses in advance for predictable testing
- **Call tracking**: Track all prompts and responses for verification
- **Rule #5 compliance**: Raises `AssertionError` if called more times than responses configured
- **Full LLMClient interface**: Compatible with all code expecting an `LLMClient`

## Basic Usage

```python
from tests.fixtures.agents import MockLLM

# Create mock and configure responses
llm = MockLLM
llm.set_responses([
    "Thought: I should search\nAction: search",
    "Thought: Found answer\nFinal Answer: 42"
])

# Use like a real LLM
response1 = llm.generate("What is the task?")
response2 = llm.generate("Continue")

# Verify call history
assert llm.call_count == 2
assert llm.call_history[0] == "What is the task?"
```

## Testing ReAct Agents

```python
from ingestforge.agent.react_engine import ReActEngine, SimpleTool
from tests.fixtures.agents import MockLLM

def test_agent_completes_task:
    # Setup mock LLM
    llm = MockLLM
    llm.set_responses([
        "I need to search\nAction: search",
        "Found answer\nFinal Answer: result"
    ])

    # Create thinking function that uses mock
    def think_fn(task, history, tools):
        prompt = f"Task: {task}"
        response = llm.generate(prompt)

        if "Final Answer:" in response:
            answer = response.split("Final Answer:")[1].strip
            return (answer, None, {})

        if "Action:" in response:
            action = response.split("Action:")[1].strip
            thought = response.split("Action:")[0].strip
            return (thought, action, {"query": "test"})

        return (response, None, {})

    # Create and run agent
    engine = ReActEngine(think_fn=think_fn)

    search_tool = SimpleTool(
        name="search",
        description="Search",
        _fn=lambda query: "Found info"
    )
    engine.register_tool(search_tool)

    result = engine.run("Find information")

    # Verify
    assert result.success is True
    assert llm.call_count == 2
```

## Key Methods

### Configuration

- `set_responses(sequence: List[str])` - Set response sequence
- `set_model_name(name: str)` - Set model name for testing
- `set_availability(available: bool)` - Set availability status

### Generation

- `generate(prompt: str) -> str` - Get next response
- `generate_with_context(system_prompt, user_prompt, context) -> str` - Generate with context

### Tracking

- `call_count: int` - Number of calls made
- `get_last_prompt -> str` - Get most recent prompt
- `call_history: List[str]` - All prompts (copy)
- `response_history: List[str]` - All responses (copy)

### Response Management

- `get_remaining_count -> int` - Responses remaining
- `has_responses_remaining -> bool` - Check if responses available
- `reset` - Clear history, restart sequence (preserves responses)
- `clear_responses` - Clear everything

## Validation (Rule #5)

The MockLLM enforces strict response counting:

```python
llm = MockLLM
llm.set_responses(["only one"])

llm.generate("first")  # OK
llm.generate("second")  # AssertionError: MockLLM called 2 times but only 1 responses configured
```

This helps catch tests that don't properly configure responses.

## Best Practices

### 1. Configure exact number of responses needed

```python
# Good: Exact count
llm.set_responses(["r1", "r2"])  # Agent makes 2 calls

# Bad: Will fail on 3rd call
llm.set_responses(["r1"])  # Agent makes 2 calls -> AssertionError
```

### 2. Reset between test runs

```python
def test_multiple_runs:
    llm = MockLLM

    # First run
    llm.set_responses(["response 1"])
    result1 = run_agent(llm)

    # Second run - reconfigure
    llm.set_responses(["response 2"])
    result2 = run_agent(llm)

    # Verify both
    assert llm.call_count == 2  # Accumulates
```

### 3. Verify call history

```python
def test_correct_prompts:
    llm = MockLLM
    llm.set_responses(["answer"])

    agent.run(llm, "task")

    # Verify prompts contain expected content
    assert "task" in llm.call_history[0]
    assert len(llm.call_history) == 1
```

### 4. Check remaining responses

```python
def test_with_safety_check:
    llm = MockLLM
    llm.set_responses(["r1", "r2", "r3"])

    while llm.has_responses_remaining:
        response = llm.generate("prompt")
        process(response)

    # No AssertionError - stopped at right time
```

## Error Messages

The MockLLM provides helpful error messages:

```python
# Empty sequence
llm.set_responses([])
# ValueError: Response sequence cannot be empty

# Wrong type
llm.set_responses(["valid", 123])
# ValueError: Response at index 1 must be a string, got <class 'int'>

# Exhausted responses
llm.set_responses(["only one"])
llm.generate("1")
llm.generate("2")
# AssertionError: MockLLM called 2 times but only 1 responses configured.
#                 Use set_responses to provide more responses.
```

## Advanced: Multi-turn conversations

```python
def test_conversation:
    llm = MockLLM
    llm.set_responses([
        "Thought: Need more info\nAction: search",
        "Thought: Still unclear\nAction: analyze",
        "Thought: Got it\nFinal Answer: 42"
    ])

    agent = create_agent(llm)
    result = agent.run("Complex task")

    # Verify conversation
    assert llm.call_count == 3
    assert result.success is True

    # Check each turn
    for i, prompt in enumerate(llm.call_history):
        assert f"iteration {i}" in prompt or "task" in prompt.lower
```

## Comparison with unittest.mock

The MockLLM is specialized for LLM testing:

```python
# unittest.mock - generic, manual setup
from unittest.mock import Mock

mock_llm = Mock
mock_llm.generate.side_effect = ["r1", "r2"]
mock_llm.model_name = "mock"
# Need to track calls manually

# MockLLM - specialized, built-in features
from tests.fixtures.agents import MockLLM

llm = MockLLM
llm.set_responses(["r1", "r2"])
# Automatic call tracking, validation, history
```

## Files

- `tests/fixtures/agents.py` - MockLLM implementation
- `tests/unit/fixtures/test_mock_llm.py` - Unit tests (48 tests)
- `tests/unit/agent/test_mock_llm_integration.py` - Integration tests (11 tests)

## NASA JPL Compliance

The MockLLM follows all 10 commandments:

1. **Simple control flow** - No complex nesting
2. **Fixed bounds** - No infinite loops
3. **Memory management** - Returns copies to prevent mutation
4. **Small functions** - All methods <60 lines
5. **Assertions** - Raises AssertionError on exhaustion (defensive coding)
6. **Minimal scope** - Private attributes, narrow scopes
7. **Parameter validation** - Validates all inputs
8. **No abstractions** - Simple, direct implementation
9. **Type hints** - Complete type annotations
10. **Testable** - 100% test coverage

## Support

For questions or issues, see:
- Implementation: `tests/fixtures/agents.py`
- Tests: `tests/unit/fixtures/test_mock_llm.py`
- Examples: `tests/unit/agent/test_mock_llm_integration.py`
