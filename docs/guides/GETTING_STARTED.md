# Getting Started with IngestForge

IngestForge is a document ingestion and knowledge management system with AI-powered search and an autonomous research agent.

## Installation

### Prerequisites

- Python 3.10+
- 8GB+ RAM recommended
- GPU optional (improves local LLM performance)

### Quick Install

```bash
# Clone and install
git clone https://github.com/yourusername/ingestforge.git
cd ingestforge
pip install -e ".[dev]"

# Verify installation
ingestforge --version
```

### Optional Dependencies

```bash
# For local LLM support (llama-cpp)
pip install llama-cpp-python

# For GPU acceleration
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# For audio transcription
pip install faster-whisper

# For OCR capabilities
pip install easyocr
# Or for better accuracy: pip install paddleocr
```

## Your First Project

### 1. Initialize a Project

```bash
# Create a new project
ingestforge init my-research

# Navigate to project
cd my-research

# Check status
ingestforge status
```

This creates:
```
my-research/
├── ingestforge.yaml    # Configuration
├── .data/
│   ├── storage/        # Vector database
│   └── models/         # Local LLM models (optional)
└── documents/          # Place your documents here
```

### 2. Ingest Documents

```bash
# Single file
ingestforge ingest document.pdf

# Directory (recursive)
ingestforge ingest documents/ --recursive

# YouTube video
ingestforge ingest "https://www.youtube.com/watch?v=VIDEO_ID"

# Skip errors and continue
ingestforge ingest folder/ -r --skip-errors
```

Supported formats:
- PDF, DOCX, TXT, MD
- HTML, EPUB
- Images (with OCR)
- YouTube videos (transcripts)

### 3. Query Your Knowledge Base

```bash
# Ask a question
ingestforge query "What are the main findings?"

# Get more context
ingestforge query "Explain the methodology" --top-k 10

# Search only (no LLM)
ingestforge query "machine learning" --no-llm
```

### 4. Interactive Mode

```bash
# Start interactive session
ingestforge interactive
```

The interactive menu provides:
- Document ingestion
- Knowledge base queries
- Study tools (flashcards, quizzes)
- Export options
- Agent mode

## Using the Autonomous Agent

The agent can perform multi-step research tasks autonomously.

### Basic Usage

```bash
# Run a research task
ingestforge agent run "Find all information about climate change impacts"

# Limit iterations
ingestforge agent run "Summarize recent papers" --max-steps 5

# Save report
ingestforge agent run "Research quantum computing" --output report.md
```

### Available Agent Tools

```bash
# List all tools
ingestforge agent tools
```

Tools include:
- `search_knowledge_base` - Search your documents
- `search_web` - Search the internet
- `ingest_document` - Add new documents
- `discover_arxiv` - Find academic papers
- `discover_cve` - Find security vulnerabilities
- And more...

### Test LLM Connection

```bash
# Verify LLM is working
ingestforge agent test-llm
```

## Configuration

### LLM Providers

Edit `ingestforge.yaml`:

```yaml
llm:
  # Use local model (recommended)
  default_provider: llamacpp
  llamacpp:
    model_path: .data/models/your-model.gguf
    n_ctx: 8192
    n_gpu_layers: -1  # All layers on GPU

  # Or use cloud providers
  # default_provider: claude
  # claude:
  #   api_key: ${ANTHROPIC_API_KEY}
  #   model: claude-3-sonnet-20240229
```

### Storage Options

```yaml
storage:
  backend: chromadb  # Default, good for most uses
  # backend: jsonl   # Simple file-based storage
  path: .data/storage
```

### Chunking Settings

```yaml
chunking:
  strategy: semantic  # Smart paragraph splitting
  max_size: 512       # Tokens per chunk
  overlap: 50         # Token overlap between chunks
```

## Common Workflows

### Research Workflow

```bash
# 1. Initialize project
ingestforge init research-project
cd research-project

# 2. Ingest papers
ingestforge ingest papers/ --recursive

# 3. Use agent for deep research
ingestforge agent run "What are the key findings across all papers?"

# 4. Export findings
ingestforge export markdown findings.md --include-citations
```

### Study Workflow

```bash
# 1. Ingest study materials
ingestforge ingest textbook.pdf lectures/

# 2. Generate study aids
ingestforge study quiz "Chapter 3" --count 20
ingestforge study flashcards "Key concepts" --count 50

# 3. Practice in interactive mode
ingestforge interactive
```

### Literary Analysis

```bash
# 1. Ingest texts
ingestforge ingest hamlet.txt macbeth.txt

# 2. Analyze
ingestforge lit themes "Hamlet"
ingestforge lit character "Macbeth" --character "Lady Macbeth"
ingestforge lit symbols "Hamlet" --output symbols.md
```

## Troubleshooting

### "Model not found"

Download a GGUF model:
```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir .data/models/
```

### "Out of memory"

1. Use a smaller model (3B or 7B parameters)
2. Reduce `n_gpu_layers` in config
3. Set `n_gpu_layers: 0` for CPU-only

### "ChromaDB connection failed"

```bash
# Reset storage
rm -rf .data/storage
ingestforge init --force
```

### Import Errors

```bash
# Reinstall
pip install -e .

# Check dependencies
pip check
```

## Next Steps

- [Agent Usage Guide](AGENT_USAGE.md) - Deep dive into the autonomous agent
- [Configuration Reference](../configuration.md) - All configuration options
- [CLI Reference](../cli.md) - Complete command reference
- [API Documentation](../API.md) - REST API for integrations

## Getting Help

```bash
# Command help
ingestforge --help
ingestforge agent --help
ingestforge agent run --help
```

For issues: [GitHub Issues](https://github.com/yourusername/ingestforge/issues)
