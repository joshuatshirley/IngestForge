# Debate Adapter for Multi-Agent Fact-Checker

The debate adapter connects LLM clients to the adversarial fact-checking system, enabling LLM-powered debate participants.

## Overview

The `debate_adapter.py` module provides:
- `create_proponent_function` - Creates an LLM-powered proponent
- `create_critic_function` - Creates an LLM-powered critic
- Response parsing and error handling
- Structured debate argument formatting

## Quick Start

```python
from ingestforge.agent import (
    DebateOrchestrator,
    Claim,
    create_proponent_function,
    create_critic_function,
)
from ingestforge.llm import get_llm_client

# Get LLM client
llm_client = get_llm_client

# Create debate functions
proponent_fn = create_proponent_function(llm_client)
critic_fn = create_critic_function(llm_client)

# Create orchestrator
orchestrator = DebateOrchestrator(
    proponent_fn=proponent_fn,
    critic_fn=critic_fn,
    max_rounds=3,
)

# Verify a claim
claim = Claim(
    content="AI will transform healthcare by 2030",
    source="user_input",
)

result = orchestrator.verify(claim)

print(f"Status: {result.status.value}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Rounds: {result.rounds_count}")
```

## LLM Response Format

The adapter expects LLMs to respond in this structured format:

```
Position: [Your stance on the claim]
Evidence: [Supporting or refuting evidence]
Confidence: [0.0-1.0]
```

### Example Proponent Response

```
Position: The claim is well-supported by available research.
Evidence: Multiple peer-reviewed studies from 2023-2024 show AI
diagnostic systems achieving 95%+ accuracy in radiology and pathology.
Confidence: 0.85
```

### Example Critic Response

```
Position: The claim lacks sufficient empirical support.
Evidence: Current AI systems require extensive validation and
regulatory approval processes that may extend beyond 2030.
Confidence: 0.70
```

## Architecture

```
CLI Command
    ↓
DebateOrchestrator
    ├── Proponent Function (LLM-powered)
    │   └── LLMClient.generate
    └── Critic Function (LLM-powered)
        └── LLMClient.generate
    ↓
VerificationResult
    ↓
VerificationDisplay (Rich UI)
```

## Functions

### `create_proponent_function(llm_client: LLMClient) -> DebateFunction`

Creates a debate function that defends claims.

**Args:**
- `llm_client`: LLM client instance

**Returns:**
- Callable that takes `(prompt, history)` and returns `DebateArgument`

**Example:**
```python
proponent_fn = create_proponent_function(llm_client)
arg = proponent_fn("Defend: Climate change is urgent", [])
```

### `create_critic_function(llm_client: LLMClient) -> DebateFunction`

Creates a debate function that critiques claims.

**Args:**
- `llm_client`: LLM client instance

**Returns:**
- Callable that takes `(prompt, history)` and returns `DebateArgument`

**Example:**
```python
critic_fn = create_critic_function(llm_client)
arg = critic_fn("Critique: Climate change is urgent", [])
```

## Error Handling

The adapter provides robust error handling:

1. **LLM Errors**: Returns fallback argument with error message
2. **Parse Failures**: Uses defaults for missing fields
3. **Invalid Confidence**: Defaults to 0.6
4. **Empty Responses**: Returns "No response provided" argument

All errors are logged but don't crash the debate.

## Testing

See `tests/unit/agent/test_debate_adapter.py` for comprehensive tests.

**Run tests:**
```bash
pytest tests/unit/agent/test_debate_adapter.py -v
```

**Test coverage:**
- Function creation and validation
- LLM error handling
- Response parsing (complete, partial, malformed)
- Field extraction (position, evidence, confidence)
- Fallback argument creation
- Full debate cycle integration

## Integration with CLI

The debate command uses this adapter:

```bash
ingestforge argument debate "AI will replace human workers"
```

This triggers:
1. CLI creates LLM client
2. Adapter creates proponent and critic functions
3. DebateOrchestrator runs multi-round debate
4. VerificationDisplay shows results

## Configuration

### Max Rounds

Control debate length:
```python
orchestrator = DebateOrchestrator(
    proponent_fn=proponent_fn,
    critic_fn=critic_fn,
    max_rounds=5,  # Up to 10 rounds
)
```

### LLM Configuration

Customize LLM behavior:
```python
from ingestforge.llm.base import GenerationConfig

config = GenerationConfig(
    max_tokens=1000,
    temperature=0.8,  # More creative
)

# Adapter uses this internally
```

## Prompt Engineering

The adapter constructs prompts with:
- System instructions (role definition)
- Base prompt (from orchestrator)
- Recent history (last 2 arguments)
- Format instructions

This ensures consistent, high-quality debate arguments.

## Limitations

1. **Context Window**: Long debates may exceed token limits
2. **Format Adherence**: LLMs may not always follow format exactly
3. **Language Support**: Currently English-only
4. **Hallucination**: LLMs may generate plausible but false evidence

## Related Modules

- `fact_checker.py` - Core debate orchestration
- `verification_ui.py` - Result display
- `llm_adapter.py` - ReAct engine LLM adapter (different pattern)

## NASA JPL Compliance

This module follows all NASA JPL Commandments:
- Rule #1: Max 3 nesting levels, early returns
- Rule #2: Fixed upper bounds (MAX_POSITION_LENGTH, etc.)
- Rule #4: All functions <60 lines
- Rule #7: Parameter validation before operations
- Rule #9: Complete type hints

## Future Enhancements

- Multi-language support
- Evidence source validation
- Confidence calibration
- Debate strategy customization
- Citation extraction
