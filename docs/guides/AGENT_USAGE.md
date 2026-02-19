# Autonomous Agent Usage Guide

The IngestForge agent uses the ReAct (Reasoning + Acting) framework to perform multi-step research tasks autonomously.

## Overview

The agent:
1. Receives a task from you
2. Reasons about what to do
3. Selects and executes tools
4. Observes results
5. Repeats until complete

## Quick Start

```bash
# Basic research task
ingestforge agent run "What are the main topics in my knowledge base?"

# With output file
ingestforge agent run "Summarize all documents" --output summary.md

# Limit iterations
ingestforge agent run "Find related concepts" --max-steps 5
```

## Available Commands

### `agent run`

Execute a research task.

```bash
ingestforge agent run "TASK" [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--max-steps N` | Maximum iterations (default: 10) |
| `--output FILE` | Save report to file |
| `--format FORMAT` | Report format: markdown, html, json |
| `--provider NAME` | LLM provider: llamacpp, ollama, claude, openai |
| `--model NAME` | Override model name |
| `--domain-aware` | Enable domain-specific tool filtering |
| `--no-grammar` | Disable grammar constraint (debugging) |
| `--quiet` | Suppress progress output |

**Examples:**
```bash
# Research with specific provider
ingestforge agent run "Find CVE vulnerabilities" --provider llamacpp

# Domain-aware (auto-select relevant tools)
ingestforge agent run "Legal precedents for IP" --domain-aware

# Generate HTML report
ingestforge agent run "Project overview" --output report.html --format html
```

### `agent tools`

List available tools.

```bash
ingestforge agent tools
```

Output shows tool names, categories, and descriptions.

### `agent status`

Check agent configuration.

```bash
ingestforge agent status
```

Shows:
- Registered tools count
- LLM availability
- Model information

### `agent test-llm`

Test LLM connectivity.

```bash
ingestforge agent test-llm
ingestforge agent test-llm --provider ollama
```

## Available Tools

### Knowledge Base Tools

| Tool | Description |
|------|-------------|
| `search_knowledge_base` | Search your ingested documents |
| `ingest_document` | Add a new document to the knowledge base |
| `get_chunk_details` | Get full content of a specific chunk |

### Web Research Tools

| Tool | Description |
|------|-------------|
| `search_web` | Search the internet for information |

### Domain Discovery Tools

| Tool | Description |
|------|-------------|
| `discover_arxiv` | Search arXiv for academic papers |
| `discover_cve` | Search CVE database for vulnerabilities |
| `discover_law` | Search legal databases |
| `discover_medical` | Search medical literature |

### Utility Tools

| Tool | Description |
|------|-------------|
| `echo` | Echo a message (for testing) |
| `format` | Format text output |
| `describe_figure` | Describe an image/figure in a chunk |

## How the Agent Works

### ReAct Loop

Each iteration follows this pattern:

```
Thought: I need to search for information about X
Action: search_knowledge_base
Action Input: {"query": "X", "top_k": 5}
Observation: Found 5 results: [...]

Thought: I found relevant information, let me search for more
Action: search_web
Action Input: {"query": "X latest research"}
Observation: Web results: [...]

Thought: I have enough information to answer
Action: FINISH
Action Input: {}
```

### Grammar-Constrained Generation

When using local LLMs (llama-cpp), the agent uses GBNF grammar to guarantee valid output format. This:
- Eliminates parsing errors
- Ensures tool names are valid
- Produces properly formatted JSON

To disable (for debugging):
```bash
ingestforge agent run "task" --no-grammar
```

### Domain-Aware Tool Selection

With `--domain-aware`, the agent:
1. Analyzes your task
2. Detects relevant domains (legal, medical, cyber, etc.)
3. Prioritizes domain-specific tools

```bash
# Auto-selects discover_cve for cyber tasks
ingestforge agent run "Find vulnerabilities in OpenSSL" --domain-aware
```

## Configuration

### LLM Settings

In `ingestforge.yaml`:

```yaml
llm:
  default_provider: llamacpp
  llamacpp:
    model_path: .data/models/llama-3-8b-instruct.Q4_K_M.gguf
    n_ctx: 8192
    n_gpu_layers: -1
```

### Recommended Models

| Model | Size | Use Case |
|-------|------|----------|
| Llama 3 8B | 4.7GB | Good balance |
| Qwen 2.5 14B | 8.9GB | Best quality |
| Qwen 2.5 3B | 2.0GB | Fast, lightweight |
| Phi-2 | 1.6GB | Very fast |

### Resource Monitoring

The agent monitors system resources and will pause if:
- RAM usage > 85% (sustained)
- VRAM usage > 90% (sustained)
- CPU usage > 95% (sustained)

This prevents system overload during long-running tasks.

## Best Practices

### Writing Good Tasks

**Good tasks:**
```bash
# Specific and actionable
"Find all documents about machine learning and summarize key concepts"

# Clear scope
"Search the knowledge base for Python best practices"

# Defined output
"Create a list of all mentioned authors with their contributions"
```

**Poor tasks:**
```bash
# Too vague
"Tell me about stuff"

# No clear goal
"Research everything"
```

### Iteration Limits

- Start with `--max-steps 5` for testing
- Increase to 10-20 for complex research
- Use `--quiet` for long-running tasks

### Output Formats

```bash
# Markdown (default) - good for reading
--format markdown

# HTML - good for sharing
--format html

# JSON - good for processing
--format json
```

## Troubleshooting

### "Unknown tool" Errors

Usually indicates parsing issues. Try:
1. Use a larger model
2. Ensure grammar is enabled (default for llama-cpp)
3. Simplify your task

### Agent Not Finishing

If the agent loops without finishing:
1. Reduce `--max-steps`
2. Make task more specific
3. Check if required information exists in knowledge base

### Slow Performance

1. Use GPU acceleration: `n_gpu_layers: -1`
2. Use a smaller model
3. Reduce `--max-steps`

### Memory Issues

```bash
# Reduce GPU layers
# In ingestforge.yaml:
llamacpp:
  n_gpu_layers: 20  # Partial GPU offload
```

## Example Workflows

### Research Paper Review

```bash
# 1. Ingest papers
ingestforge ingest papers/ --recursive

# 2. Get overview
ingestforge agent run "What papers do I have and what are their main topics?"

# 3. Deep dive
ingestforge agent run "Compare methodologies across all papers" --max-steps 15

# 4. Generate report
ingestforge agent run "Write a literature review summary" --output review.md
```

### Security Audit

```bash
# Domain-aware search
ingestforge agent run "Find security vulnerabilities related to web authentication" \
  --domain-aware --output audit.md
```

### Multi-Source Research

```bash
# Combine knowledge base + web
ingestforge agent run "Research latest developments in quantum computing, \
  combining my documents with current web sources" --max-steps 20
```

## API Usage

For programmatic access:

```python
from ingestforge.agent.llm_adapter import create_llm_think_adapter
from ingestforge.agent.react_engine import create_engine
from ingestforge.agent.tool_registry import create_registry, register_builtin_tools
from ingestforge.llm.factory import get_llm_client

# Setup
config = load_config
llm_client = get_llm_client(config)

# Create adapter with grammar
adapter = create_llm_think_adapter(
    llm_client=llm_client,
    use_grammar=True,  # Enable for local LLMs
)

# Create engine
engine = create_engine(think_fn=adapter, max_iterations=10)

# Register tools
registry = create_registry
register_builtin_tools(registry)
for name in registry.tool_names:
    tool = registry.get_as_protocol(name)
    if tool:
        engine.register_tool(tool)

# Run
result = engine.run("Your research task")
print(result.final_answer)
```

## See Also

- [Getting Started](GETTING_STARTED.md) - Initial setup
- [Configuration](../configuration.md) - All config options
- [CLI Reference](../cli.md) - All commands
