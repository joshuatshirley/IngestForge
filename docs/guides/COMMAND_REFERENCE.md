# IngestForge CLI Command Reference

Complete reference for all IngestForge commands.

**Version:** 1.2
**Total Commands:** 23+
**Command Groups:** 8

---

## Agent Commands

Group: `ingestforge agent`

### `ingestforge agent run`
**Purpose:** Run autonomous research agent

**Usage:**
```bash
ingestforge agent run "TASK" [OPTIONS]
```

**Options:**
- `--max-steps N` - Maximum iterations (default: 10)
- `--output FILE` - Save report to file
- `--format FORMAT` - Report format: markdown, html, json
- `--provider NAME` - LLM provider: llamacpp, ollama, claude, openai
- `--model NAME` - Override model name
- `--domain-aware` - Enable domain-specific tool filtering
- `--no-grammar` - Disable grammar constraint (debugging)
- `--quiet` - Suppress progress output

**Examples:**
```bash
ingestforge agent run "Summarize all documents"
ingestforge agent run "Find security issues" --domain-aware
ingestforge agent run "Research topic X" --output report.md --max-steps 15
```

---

### `ingestforge agent tools`
**Purpose:** List available agent tools

**Usage:**
```bash
ingestforge agent tools [OPTIONS]
```

**Options:**
- `-p, --project PATH` - Project directory

---

### `ingestforge agent status`
**Purpose:** Show agent status and capabilities

**Usage:**
```bash
ingestforge agent status [OPTIONS]
```

**Options:**
- `-p, --project PATH` - Project directory
- `--provider NAME` - Test specific provider

---

### `ingestforge agent test-llm`
**Purpose:** Test LLM connectivity

**Usage:**
```bash
ingestforge agent test-llm [OPTIONS]
```

**Options:**
- `-p, --project PATH` - Project directory
- `--provider NAME` - LLM provider to test

**Examples:**
```bash
ingestforge agent test-llm
ingestforge agent test-llm --provider ollama
```

---

## Core Commands

### `ingestforge status`
**Purpose:** Display project status and statistics

**Usage:**
```bash
ingestforge status [OPTIONS]
```

**Options:**
- `-p, --project PATH` - Project directory

**Example:**
```bash
ingestforge status
ingestforge status --project /path/to/project
```

---

### `ingestforge init`
**Purpose:** Initialize new IngestForge project

**Usage:**
```bash
ingestforge init NAME [OPTIONS]
```

**Options:**
- `-p, --project PATH` - Parent directory for project
- `--with-sample` - Include sample document
- `--mobile` - Optimize for mobile/low-resource environments

**Examples:**
```bash
ingestforge init my_project
ingestforge init research_docs --with-sample
ingestforge init mobile_kb --mobile
```

---

### `ingestforge query`
**Purpose:** Search knowledge base and generate AI-powered answers

**Usage:**
```bash
ingestforge query "QUESTION" [OPTIONS]
```

**Options:**
- `-p, --project PATH` - Project directory
- `-k` - Number of chunks to retrieve (default: 5)
- `--no-llm` - Skip LLM generation, show only search results

**Examples:**
```bash
ingestforge query "What is machine learning?"
ingestforge query "Explain RAG architecture" --k 10
ingestforge query "Python best practices" --no-llm
```

---

### `ingestforge ingest`
**Purpose:** Process documents through ingestion pipeline

**Usage:**
```bash
ingestforge ingest PATH [OPTIONS]
```

**Options:**
- `-p, --project PATH` - Project directory
- `-r, --recursive` - Process directories recursively
- `--skip-errors` - Continue on errors

**Examples:**
```bash
ingestforge ingest documents/paper.pdf
ingestforge ingest research/ --recursive
ingestforge ingest data/ -r --skip-errors
```

---

## Literary Analysis Commands

Group: `ingestforge lit`

### `ingestforge lit themes`
**Purpose:** Extract and analyze major themes

**Usage:**
```bash
ingestforge lit themes "WORK" [OPTIONS]
```

**Options:**
- `-p, --project PATH` - Project directory
- `-o, --output FILE` - Save analysis to file

**Examples:**
```bash
ingestforge lit themes "Hamlet"
ingestforge lit themes "1984" --output themes.md
```

---

### `ingestforge lit character`
**Purpose:** Analyze character development

**Usage:**
```bash
ingestforge lit character "WORK" [OPTIONS]
```

**Options:**
- `-c, --character NAME` - Specific character to analyze
- `-p, --project PATH` - Project directory
- `-o, --output FILE` - Save analysis to file

**Examples:**
```bash
ingestforge lit character "Romeo and Juliet"
ingestforge lit character "Hamlet" --character "Ophelia"
ingestforge lit character "Macbeth" -c "Lady Macbeth" -o analysis.md
```

---

### `ingestforge lit symbols`
**Purpose:** Identify and analyze symbolic patterns

**Usage:**
```bash
ingestforge lit symbols "WORK" [OPTIONS]
```

**Options:**
- `-p, --project PATH` - Project directory
- `-o, --output FILE` - Save analysis to file

**Examples:**
```bash
ingestforge lit symbols "The Great Gatsby"
ingestforge lit symbols "Moby Dick" --output symbols.md
```

---

### `ingestforge lit outline`
**Purpose:** Generate structural outline

**Usage:**
```bash
ingestforge lit outline "WORK" [OPTIONS]
```

**Options:**
- `-d, --detailed` - Include detailed scene-by-scene breakdown
- `-p, --project PATH` - Project directory
- `-o, --output FILE` - Save outline to file

**Examples:**
```bash
ingestforge lit outline "Pride and Prejudice"
ingestforge lit outline "The Odyssey" --detailed
ingestforge lit outline "1984" -d -o outline.md
```

---

## Research Commands

Group: `ingestforge research`

### `ingestforge research audit`
**Purpose:** Audit knowledge base quality and coverage

**Usage:**
```bash
ingestforge research audit [OPTIONS]
```

**Options:**
- `-p, --project PATH` - Project directory
- `-d, --detailed` - Include detailed per-source analysis
- `-o, --output FILE` - Save audit report

**Examples:**
```bash
ingestforge research audit
ingestforge research audit --detailed
ingestforge research audit -d -o audit_report.md
```

---

### `ingestforge research verify`
**Purpose:** Verify source citations and references

**Usage:**
```bash
ingestforge research verify [OPTIONS]
```

**Options:**
- `-p, --project PATH` - Project directory
- `-m, --show-missing` - Show chunks without citations
- `-o, --output FILE` - Save verification report

**Examples:**
```bash
ingestforge research verify
ingestforge research verify --show-missing
ingestforge research verify -m -o verify_report.md
```

---

## Study Commands

Group: `ingestforge study`

### `ingestforge study quiz`
**Purpose:** Generate quiz questions from knowledge base

**Usage:**
```bash
ingestforge study quiz "TOPIC" [OPTIONS]
```

**Options:**
- `-n, --count N` - Number of questions (default: 5)
- `-d, --difficulty LEVEL` - Question difficulty (easy/medium/hard)
- `-p, --project PATH` - Project directory
- `-o, --output FILE` - Save quiz to JSON file

**Examples:**
```bash
ingestforge study quiz "Python programming"
ingestforge study quiz "Machine Learning" --count 10 --difficulty hard
ingestforge study quiz "History" -n 15 -o quiz.json
```

---

### `ingestforge study flashcards`
**Purpose:** Generate flashcard sets for memorization

**Usage:**
```bash
ingestforge study flashcards "TOPIC" [OPTIONS]
```

**Options:**
- `-n, --count N` - Number of flashcards (default: 10)
- `-t, --type TYPE` - Card type (definition/concept/fact/process)
- `-p, --project PATH` - Project directory
- `-o, --output FILE` - Save flashcards to JSON file

**Examples:**
```bash
ingestforge study flashcards "Biology terms"
ingestforge study flashcards "Machine Learning" --count 20 --type concept
ingestforge study flashcards "Chemistry" -n 30 -t definition -o cards.json
```

---

## Interactive Commands

Group: `ingestforge interactive`

### `ingestforge interactive ask`
**Purpose:** Start interactive query session (REPL mode)

**Usage:**
```bash
ingestforge interactive ask [OPTIONS]
```

**Options:**
- `-p, --project PATH` - Project directory
- `-k` - Number of chunks to retrieve per query (default: 5)
- `--no-history` - Disable conversation history

**Examples:**
```bash
ingestforge interactive ask
ingestforge interactive ask --k 10
ingestforge interactive ask --no-history
```

**In the REPL:**
- Type questions naturally
- `/history` - View conversation history
- `/clear` - Clear conversation history
- `/help` - Show help
- `/exit` or `/quit` - Exit interactive mode
- `Ctrl+C` - Exit

---

## Export Commands

Group: `ingestforge export`

### `ingestforge export markdown`
**Purpose:** Export knowledge base to Markdown format with streaming support for large datasets

**Usage:**
```bash
ingestforge export markdown OUTPUT [OPTIONS]
```

**Options:**
- `-p, --project PATH` - Project directory
- `-q, --query TEXT` - Filter chunks by search query
- `-n, --limit N` - Limit number of chunks
- `--no-grouping` - Don't group chunks by source
- `-c, --include-citations` - Include citations section at end

**Features:**
- Streaming export prevents memory issues with large corpora (EXPORT-001.1)
- Citation tracking for research notes and bibliography generation (EXPORT-001.2)
- Grouped or sequential output formats

**Examples:**
```bash
ingestforge export markdown output.md
ingestforge export markdown ml_docs.md --query "machine learning"
ingestforge export markdown sample.md --limit 100
ingestforge export markdown sequential.md --no-grouping
ingestforge export markdown research_notes.md --include-citations
```

---

### `ingestforge export json`
**Purpose:** Export knowledge base to JSON format

**Usage:**
```bash
ingestforge export json OUTPUT [OPTIONS]
```

**Options:**
- `-p, --project PATH` - Project directory
- `-q, --query TEXT` - Filter chunks by search query
- `-n, --limit N` - Limit number of chunks
- `--compact` - Use compact formatting (no indentation)

**Examples:**
```bash
ingestforge export json output.json
ingestforge export json ml_docs.json --query "machine learning"
ingestforge export json sample.json --limit 100
ingestforge export json compact.json --compact
```

---

## Command Categories

### Autonomous Agent
- `agent run` - Run research tasks autonomously
- `agent tools` - List available tools
- `agent status` - Check agent configuration
- `agent test-llm` - Test LLM connectivity

### Document Processing
- `init` - Initialize project
- `ingest` - Process documents
- `status` - Check status

### Knowledge Access
- `query` - Search and answer questions
- `interactive ask` - Conversational queries

### Content Analysis
- `lit themes` - Theme analysis
- `lit character` - Character analysis
- `lit symbols` - Symbol analysis
- `lit outline` - Structure analysis

### Quality & Research
- `research audit` - Quality audit
- `research verify` - Citation verification

### Learning Tools
- `study quiz` - Generate quizzes
- `study flashcards` - Create flashcards

### Data Export
- `export markdown` - Export to Markdown
- `export json` - Export to JSON

---

## Global Options

All commands support:
- `-p, --project PATH` - Specify project directory
- `--help` - Show command help

---

## Quick Start Examples

**Basic Workflow:**
```bash
# 1. Initialize project
ingestforge init my_research

# 2. Ingest documents
cd my_research
ingestforge ingest documents/ --recursive

# 3. Query knowledge base
ingestforge query "What are the main findings?"

# 4. Generate study materials
ingestforge study quiz "Research methods" --count 10
ingestforge study flashcards "Key concepts" --count 20

# 5. Export for sharing
ingestforge export markdown knowledge_base.md
```

**Literary Analysis Workflow:**
```bash
# 1. Initialize and ingest literary texts
ingestforge init shakespeare
cd shakespeare
ingestforge ingest hamlet.pdf macbeth.pdf

# 2. Analyze different aspects
ingestforge lit themes "Hamlet"
ingestforge lit character "Hamlet" --character "Ophelia"
ingestforge lit symbols "Macbeth"
ingestforge lit outline "Hamlet" --detailed

# 3. Save analyses
ingestforge lit themes "Hamlet" -o hamlet_themes.md
ingestforge lit character "Macbeth" -c "Lady Macbeth" -o lady_macbeth.md
```

**Research & Quality Assurance:**
```bash
# 1. Audit knowledge base
ingestforge research audit --detailed -o audit.md

# 2. Verify citations
ingestforge research verify --show-missing -o citations.md

# 3. Export verified content
ingestforge export json verified_data.json
```

**Interactive Exploration:**
```bash
# Start interactive session
ingestforge interactive ask

# In the REPL:
You: What is quantum computing?
[Answer displayed with citations]

You: How does it compare to classical computing?
[Answer using conversation context]

You: /history
[Shows conversation history]

You: /exit
```

---

## Command Design Principles

All commands follow the code quality guidelines:

1. ✅ **Simple Control Flow** - Max 2 levels nesting
2. ✅ **Fixed Loop Bounds** - No infinite loops
3. ✅ **Memory Management** - Efficient resource usage
4. ✅ **Small Functions** - Max 60 lines per function
5. ✅ **Defensive Programming** - Comprehensive validation
6. ✅ **Minimal Scope** - Local variables, clear boundaries
7. ✅ **Parameter Checking** - All inputs validated
8. ✅ **Clear Abstractions** - No obscure patterns
9. ✅ **Type Safety** - 100% type hints
10. ✅ **Static Analysis** - Pylint/mypy compliant

---

## Command Structure

```
ingestforge/
├── Agent (4)
│   ├── agent run
│   ├── agent tools
│   ├── agent status
│   └── agent test-llm
├── Core Commands (4)
│   ├── status
│   ├── init
│   ├── query
│   └── ingest
├── Literary (4)
│   ├── lit themes
│   ├── lit character
│   ├── lit symbols
│   └── lit outline
├── Research (2)
│   ├── research audit
│   └── research verify
├── Study (2)
│   ├── study quiz
│   └── study flashcards
├── Interactive (1)
│   └── interactive ask
└── Export (2)
    ├── export markdown
    └── export json
```

**Total:** 19 primary commands + 8 command groups = 27+ commands

---

**For more information:** Run `ingestforge --help` or `ingestforge <command> --help`
