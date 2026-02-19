# CLI Reference

> Complete guide to all IngestForge commands

---

## Installation

```bash
# Install with all features
pip install -e ".[full]"

# Or minimal install
pip install -e .

# Verify installation
ingestforge --help
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| **Core** | |
| `init` | Initialize a new project |
| `ingest` | Process a single document |
| `add` | Fetch and ingest a URL |
| `watch` | Auto-process new files |
| `query` | Search the corpus |
| `ask` | Interactive Q&A mode |
| `status` | Show statistics |
| **Study Materials** | |
| `export` | Export research notes |
| `flashcards` | Generate Anki flashcards |
| `glossary` | Generate term glossary |
| `overview` | Generate corpus overview |
| `quiz` | Generate self-assessment quiz |
| **Research & Analysis** | |
| `explain` | Get simple explanation of a concept |
| `compare` | Compare two concepts side-by-side |
| `connect` | Find how two concepts are linked |
| `scholars` | Identify key authors and contributions |
| `timeline` | Chronological development of ideas |
| **Argument Building** | |
| `debate` | Analyze pro/con arguments |
| `support` | Find evidence supporting a claim |
| `counter` | Find counterarguments to a claim |
| `conflicts` | Find contradictions between sources |
| `gaps` | Identify uncovered research areas |
| **Citation & Writing** | |
| `quote` | Find quotable passages |
| `cite` | Generate formatted citations |
| `bibliography` | Generate full bibliography |
| `thesis` | Evaluate a thesis against sources |
| `draft` | Generate draft paragraphs with citations |
| **Export & Discovery** | |
| `concept-map` | Generate Mermaid concept map |
| `folder-export` | Export organized study materials folder |
| `discover` | Search academic papers and educational resources |
| `prerequisites` | Detect prerequisite concepts |
| **Utilities** | |
| `cache` | Manage query cache |
| `model` | Manage local LLM models |
| `serve` | Start REST API server |
| `reset` | Clear all indexed data |

---

## Interactive Menu

**The easiest way to use IngestForge is to run it with no arguments:**

```bash
ingestforge
```

This launches an interactive menu that guides you through all available features.

### Menu Options

When you run `ingestforge` without arguments, you'll see:

```
╭───────────────────────────────────────╮
│         IngestForge Menu              │
├───────────────────────────────────────┤
│  [1] Initialize Project               │
│  [2] Ingest Document                  │
│  [3] Add URL                          │
│  [4] Query Corpus                     │
│  [5] Interactive Q&A                  │
│  [6] Generate Study Materials  →      │
│  [7] Export & Reports          →      │
│  [8] View Status                      │
│  [9] Settings & Utilities      →      │
│  [0] Exit                             │
╰───────────────────────────────────────╯
```

### Study Materials Submenu

Selecting option `[6]` opens:

```
╭───────────────────────────────────────╮
│      Study Materials                  │
├───────────────────────────────────────┤
│  [1] Flashcards (Anki-compatible)     │
│  [2] Glossary of Key Terms            │
│  [3] Self-Assessment Quiz             │
│  [4] Concept Explanation              │
│  [5] Compare Concepts                 │
│  [0] Back to Main Menu                │
╰───────────────────────────────────────╯
```

### Export & Reports Submenu

Selecting option `[7]` opens:

```
╭───────────────────────────────────────╮
│      Export & Reports                 │
├───────────────────────────────────────┤
│  [1] Export Research Notes            │
│  [2] Corpus Overview                  │
│  [0] Back to Main Menu                │
╰───────────────────────────────────────╯
```

### Settings & Utilities Submenu

Selecting option `[9]` opens:

```
╭───────────────────────────────────────╮
│      Settings & Utilities             │
├───────────────────────────────────────┤
│  [1] Cache Management                 │
│  [2] Model Management                 │
│  [3] Start API Server                 │
│  [4] Watch Folder                     │
│  [5] Reset Data                       │
│  [0] Back to Main Menu                │
╰───────────────────────────────────────╯
```

### Navigation

- **Number keys**: Select menu options
- **Enter**: Confirm selection
- **0**: Go back or exit
- **Ctrl+C**: Exit at any time

### Benefits of Interactive Mode

1. **No memorization required** - All options visible at once
2. **Guided workflow** - Menus prompt for required inputs
3. **Discoverable** - Find features you didn't know existed
4. **Quick access** - Faster than typing long commands
5. **Safe** - Destructive actions require confirmation

### When to Use CLI vs Interactive

| Use Interactive Menu | Use CLI Commands |
|---------------------|------------------|
| First-time exploration | Scripts & automation |
| Quick one-off tasks | Repeatable workflows |
| Discovering features | CI/CD pipelines |
| Teaching/demos | Power users |

---

## Commands

### init

Initialize a new IngestForge project in the current directory.

```bash
ingestforge init [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--name, -n` | `my-knowledge-base` | Project name |
| `--path, -p` | Current directory | Project directory |
| `--with-sample, -s` | `false` | Include a sample document |

**Examples:**

```bash
# Basic initialization
ingestforge init

# Named project with sample
ingestforge init --name "quantum-research" --with-sample

# Initialize in specific directory
ingestforge init --path ./my-project --name "thesis-research"
```

**What it creates:**

```
./
├── config.yaml              # Project configuration
├── .ingest/
│   ├── pending/             # Drop documents here
│   ├── processing/          # Currently processing
│   └── completed/           # Successfully processed
└── .data/
    ├── chunks/              # JSONL storage
    ├── chromadb/            # Vector database
    └── index/               # BM25 index
```

---

### ingest

Process a single document and add it to the index.

```bash
ingestforge ingest <FILE_PATH> [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `FILE_PATH` | Yes | Path to document to process |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--verbose, -v` | `false` | Show detailed progress |

**Supported formats:**

| Format | Extensions | Notes |
|--------|------------|-------|
| PDF | `.pdf` | Native text or OCR for scanned |
| HTML | `.html`, `.htm` | Web pages |
| EPUB | `.epub` | E-books |
| Word | `.docx` | Microsoft Word |
| PowerPoint | `.pptx` | Shapes, tables, speaker notes |
| Text | `.txt`, `.md` | Plain text, Markdown |
| Images | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.gif` | OCR (Tesseract or EasyOCR) |

**Examples:**

```bash
# Process a PDF
ingestforge ingest research-paper.pdf

# Process with verbose output
ingestforge ingest textbook.pdf --verbose

# Process a web page (saved HTML)
ingestforge ingest saved-article.html

# Process an image (requires Tesseract or EasyOCR)
ingestforge ingest whiteboard-photo.jpg
```

**Output:**

```
✓ Processed: research-paper.pdf
  Chunks created: 47
  Chunks indexed: 47
  Time: 12.3s
```

---

### add

Fetch a URL and ingest it directly.

```bash
ingestforge add <URL>
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `URL` | Yes | Web page URL to fetch and process |

**Examples:**

```bash
# Add a web article
ingestforge add https://example.com/article-about-ml

# Add Wikipedia page
ingestforge add https://en.wikipedia.org/wiki/Machine_learning
```

**Output:**

```
Fetching: https://example.com/article-about-ml
Done! Introduction to Machine Learning
  Chunks created: 12
```

**Notes:**

- Automatically extracts main content (removes navigation, ads)
- Preserves author, date, and source URL for citations
- Requires internet connection

---

### watch

Watch the pending folder and automatically process new documents.

```bash
ingestforge watch [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--existing/--no-existing` | `--existing` | Process existing files on startup |

**Examples:**

```bash
# Start watching (process existing files first)
ingestforge watch

# Start watching without processing existing
ingestforge watch --no-existing
```

**Output:**

```
Watching .ingest/pending/
Press Ctrl+C to stop.

→ Processing: new-paper.pdf
  ✓ Done: 23 chunks indexed
→ Processing: article.html
  ✓ Done: 8 chunks indexed
```

**Notes:**

- Monitors `.ingest/pending/` for new files
- Moves processed files to `.ingest/completed/`
- Press `Ctrl+C` to stop

---

### query

Search the indexed corpus.

```bash
ingestforge query <QUERY_TEXT> [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `QUERY_TEXT` | Yes | Search query |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--top, -k` | `5` | Number of results to return |
| `--verbose, -v` | `false` | Show full content (not truncated) |

**Examples:**

```bash
# Basic search
ingestforge query "What is backpropagation?"

# Get more results
ingestforge query "neural network architecture" --top 10

# Show full content
ingestforge query "gradient descent" --verbose
```

**Output:**

```
Results for: What is backpropagation?

1. Neural Network Training
   Score: 0.8734 | [Smith 2023, Ch.4, p.82]
   Backpropagation is an algorithm for training neural networks by computing...

2. Deep Learning Fundamentals
   Score: 0.8521 | [Jones 2022, §3.2]
   The backpropagation algorithm works by propagating errors backward...

3. Machine Learning Overview
   Score: 0.7892 | [Wikipedia: Backpropagation]
   In machine learning, backpropagation is a method used to calculate...
```

---

### ask

Interactive question-answering mode.

```bash
ingestforge ask [QUESTION]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `QUESTION` | No | Optional single question (exits after answering) |

**Examples:**

```bash
# Interactive mode
ingestforge ask

# Single question mode
ingestforge ask "What are the key concepts?"
```

**Interactive session:**

```
IngestForge Q&A
Type your questions. Press Ctrl+C to exit.

> What is machine learning?

1. Introduction
   Machine learning is a subset of artificial intelligence that enables...
   [Smith, Ch.1]

2. ML Basics
   At its core, machine learning involves algorithms that learn patterns...
   [Jones, p.12]

> How does it differ from traditional programming?

1. Programming Paradigms
   Traditional programming requires explicit rules, while ML learns from data...
   [Smith, Ch.2]
```

---

### status

Show pipeline statistics and corpus information.

```bash
ingestforge status
```

**Output:**

```
┌──────────────────────────────────┐
│       IngestForge Status         │
├─────────────────┬────────────────┤
│ Metric          │ Value          │
├─────────────────┼────────────────┤
│ Project         │ my-research    │
│ Total Documents │ 15             │
│ Total Chunks    │ 342            │
│ Total Embeddings│ 342            │
│ Pending         │ 2              │
│ In Progress     │ 0              │
│ Completed       │ 15             │
│ Failed          │ 0              │
│ Last Updated    │ 2026-01-30     │
└─────────────────┴────────────────┘
```

---

### export

Export query results as markdown research notes with citations.

```bash
ingestforge export <QUERY_TEXT> [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `QUERY_TEXT` | Yes | Query to find relevant content |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | `notes.md` | Output file path |
| `--top, -k` | `5` | Number of results to include |

**Examples:**

```bash
# Basic export
ingestforge export "machine learning basics"

# Custom output file
ingestforge export "neural networks" --output nn-notes.md

# More comprehensive notes
ingestforge export "deep learning" --top 15 --output deep-learning.md
```

**Output file format:**

```markdown
# Research Notes: machine learning basics

Generated: 2026-01-30

## Key Findings

### 1. Introduction to ML
Machine learning is a subset of artificial intelligence...

**Source:** [Smith 2023, Ch.1, p.5]

### 2. Types of Learning
There are three main types: supervised, unsupervised...

**Source:** [Jones 2022, §2.1]

---

## Sources

1. Smith, J. (2023). *Machine Learning Fundamentals*. p.5
2. Jones, A. (2022). *AI Handbook*. Section 2.1
```

---

### flashcards

Generate Anki-compatible flashcards from indexed content.

```bash
ingestforge flashcards [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | `flashcards.csv` | Output CSV file path |
| `--max, -m` | `100` | Maximum number of cards |
| `--definitions/--no-definitions` | `--definitions` | Include term definitions |
| `--concepts/--no-concepts` | `--concepts` | Include concept explanations |
| `--questions/--no-questions` | `--questions` | Include Q&A cards |

**Examples:**

```bash
# Generate all types of flashcards
ingestforge flashcards

# Only definitions, max 50 cards
ingestforge flashcards --max 50 --no-concepts --no-questions

# Custom output
ingestforge flashcards --output ml-cards.csv
```

**Output format (Anki CSV):**

```
Define: Machine Learning<TAB>A subset of AI that enables computers to learn from data without being explicitly programmed.<br><br>[Source: [Smith 2023]]<TAB>definition
What is gradient descent?<TAB>An optimization algorithm used to minimize the loss function by iteratively moving toward the minimum.<br><br>[Source: [Jones 2022]]<TAB>question
```

**Importing to Anki:**

1. Open Anki
2. File → Import
3. Select the CSV file
4. Set field separator to "Tab"
5. Map fields: Front, Back, Tags

---

### glossary

Generate a glossary of key terms from indexed content.

```bash
ingestforge glossary [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | `glossary.md` | Output file path |
| `--min, -m` | `2` | Minimum term occurrences |

**Examples:**

```bash
# Generate glossary
ingestforge glossary

# Require more occurrences (filters rare terms)
ingestforge glossary --min 5

# Custom output
ingestforge glossary --output terms.md
```

**Output format:**

```markdown
# Glossary

## A

### Algorithm
A step-by-step procedure for solving a problem or performing a computation.

*Sources: [Smith 2023], [Jones 2022]*

### Artificial Intelligence
The simulation of human intelligence by machines.

*Sources: [Smith 2023, Ch.1]*

## B

### Backpropagation
An algorithm for training neural networks by computing gradients.

*Sources: [Deep Learning, Ch.6]*
```

---

### overview

Generate an overview of all indexed content.

```bash
ingestforge overview [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | `overview.md` | Output file path |

**Examples:**

```bash
# Generate overview
ingestforge overview

# Custom output
ingestforge overview --output corpus-summary.md
```

**Output format:**

```markdown
# Research Overview

## Corpus Statistics

- **Total chunks indexed:** 342
- **Documents processed:** 15
- **Date range:** 2020-2024

## Sources

| Document | Chunks | Type |
|----------|--------|------|
| ML Fundamentals | 89 | PDF |
| AI Handbook | 67 | PDF |
| Wikipedia articles | 45 | HTML |

## Key Themes

1. **Machine Learning Basics** - Foundations and terminology
2. **Neural Networks** - Architecture and training
3. **Applications** - Real-world use cases

## Suggested Queries

- "What is the difference between supervised and unsupervised learning?"
- "How do neural networks learn?"
- "What are common ML applications?"

## Next Steps

- [ ] Review the glossary for key terms
- [ ] Explore themes in depth
- [ ] Generate study materials
```

---

### compare

Compare two concepts side-by-side based on your indexed sources.

```bash
ingestforge compare <CONCEPT_A> <CONCEPT_B> [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `CONCEPT_A` | Yes | First concept to compare |
| `CONCEPT_B` | Yes | Second concept to compare |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file (prints to console if omitted) |
| `--top, -k` | `10` | Number of results to search per concept |

**Examples:**

```bash
# Compare two concepts
ingestforge compare "supervised learning" "unsupervised learning"

# Save to file
ingestforge compare "Python" "JavaScript" --output comparison.md

# Search deeper
ingestforge compare "CNN" "RNN" --top 20
```

**Output format:**

```markdown
# Comparison: supervised learning vs unsupervised learning

*Based on your indexed sources*

## Similarities

### 1. Machine Learning
**supervised learning:** Both use algorithms to learn patterns...
*[Smith 2023, Ch.1]*

**unsupervised learning:** Both process training data...
*[Jones 2022, p.45]*

## Key Differences

| Aspect | supervised learning | unsupervised learning |
|--------|--------------------|-----------------------|
| labels | requires labeled data | uses unlabeled data |
| output | predicts known categories | discovers patterns |

## Unique to supervised learning

- Classification and regression tasks
  *[Deep Learning, Ch.5]*

## Unique to unsupervised learning

- Clustering and dimensionality reduction
  *[ML Handbook, §3.2]*

---

## Sources

- Smith 2023
- Jones 2022
- Deep Learning
```

---

### quiz

Generate a self-assessment quiz from indexed content.

```bash
ingestforge quiz [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | `quiz.md` | Output markdown file |
| `--questions, -n` | `20` | Number of questions to generate |
| `--topic, -t` | None | Focus on specific topic |
| `--mc/--no-mc` | `--mc` | Include multiple choice questions |
| `--sa/--no-sa` | `--sa` | Include short answer questions |
| `--tf/--no-tf` | `--tf` | Include true/false questions |

**Examples:**

```bash
# Generate quiz with all question types
ingestforge quiz

# Focus on a specific topic
ingestforge quiz --topic "neural networks" --questions 30

# Only multiple choice
ingestforge quiz --no-sa --no-tf --questions 15

# Custom output
ingestforge quiz --output ml-quiz.md
```

**Output format:**

```markdown
# Self-Assessment Quiz

**Topic:** neural networks
**Total Questions:** 20

*Generated from your indexed sources*

---

## Questions

### Multiple Choice

**1.** What is a neural network?

   A. A biological brain structure
   B. A computational model inspired by biological neurons
   C. A type of database
   D. A programming language

**2.** Which activation function is commonly used in hidden layers?

   A. Linear
   B. ReLU
   C. Softmax
   D. Identity

### True or False

**3.** True or False: Neural networks require labeled data for training.

**4.** True or False: Deep learning uses neural networks with multiple layers.

### Short Answer

**5.** Define: Backpropagation

**6.** What is the purpose of the loss function?

---

## Answer Key

*Scroll down to check your answers*

<details>
<summary>Click to reveal answers</summary>

### Multiple Choice Answers

**1.** B. A computational model inspired by biological neurons
   *Source: [Deep Learning, Ch.1]*

**2.** B. ReLU
   *Source: [Neural Networks Guide, p.34]*

### True/False Answers

**3.** False
   Unsupervised learning doesn't require labeled data.
   *Source: [ML Handbook]*

**4.** True
   *Source: [Deep Learning, Ch.1]*

### Short Answer Answers

**5.** Backpropagation is an algorithm for training neural networks by computing gradients of the loss function.
   *Source: [Deep Learning, Ch.6]*

</details>

---

*Quiz generated by IngestForge*
```

---

### explain

Get a simple explanation of a concept from your indexed sources.

```bash
ingestforge explain <CONCEPT> [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `CONCEPT` | Yes | Concept to explain |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file (prints to console if omitted) |
| `--verbose, -v` | `false` | Include more details |

**Examples:**

```bash
# Get explanation
ingestforge explain "gradient descent"

# Save to file
ingestforge explain "backpropagation" --output bp-explained.md

# Verbose output
ingestforge explain "neural networks" --verbose
```

**Output format:**

```markdown
# Explain: gradient descent

## Simple Explanation

Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models. It works by iteratively adjusting parameters in the direction that reduces error.

## Detailed Explanation

**What is gradient descent?**
- Gradient descent is an iterative optimization algorithm for finding the minimum of a function. *([Deep Learning, Ch.4])*
- The algorithm calculates the gradient (slope) and moves in the opposite direction. *([ML Fundamentals, p.89])*

**Key Points:**
- Used to train neural networks and other ML models
- Learning rate controls the step size
- Can get stuck in local minima

**Examples:**
- For example, training a linear regression model uses gradient descent to find optimal weights. *([Statistics 101, §3.2])*

## Related Concepts

- Backpropagation
- Learning rate
- Loss function
- Stochastic gradient descent

---

## Sources

- Deep Learning, Ch.4
- ML Fundamentals, p.89
- Statistics 101
```

---

### cache

Manage the query cache.

```bash
ingestforge cache <ACTION>
```

**Arguments:**

| Argument | Values | Description |
|----------|--------|-------------|
| `ACTION` | `stats`, `clear` | Action to perform |

**Examples:**

```bash
# View cache statistics
ingestforge cache stats

# Clear the cache
ingestforge cache clear
```

**Stats output:**

```
┌────────────────────────────────┐
│    Query Cache Statistics      │
├─────────────┬──────────────────┤
│ Metric      │ Value            │
├─────────────┼──────────────────┤
│ Entries     │ 47               │
│ Max Size    │ 1000             │
│ Hits        │ 234              │
│ Misses      │ 89               │
│ Evictions   │ 12               │
│ Hit Rate    │ 72.4%            │
└─────────────┴──────────────────┘
```

---

### model

Manage local LLM models for offline processing.

```bash
ingestforge model <ACTION> [NAME]
```

**Arguments:**

| Argument | Values | Description |
|----------|--------|-------------|
| `ACTION` | `list`, `download`, `info` | Action to perform |
| `NAME` | Model name | Required for `download` and `info` |

**Examples:**

```bash
# List available models
ingestforge model list

# Download a model
ingestforge model download phi-2

# Get model info
ingestforge model info phi-2
```

**List output:**

```
Available Models for Download:

┌──────────┬───────────────────────────┬────────────────┐
│ Name     │ Description               │ Status         │
├──────────┼───────────────────────────┼────────────────┤
│ phi-2    │ Microsoft Phi-2 (2.7B)    │ Not downloaded │
│ mistral  │ Mistral 7B Instruct       │ ✓ Downloaded   │
│ llama3   │ Llama 3 8B                │ Not downloaded │
└──────────┴───────────────────────────┴────────────────┘
```

---

### serve

Start the REST API server.

```bash
ingestforge serve [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--host, -h` | `0.0.0.0` | Host to bind |
| `--port, -p` | `8000` | Port to bind |
| `--reload` | `false` | Enable auto-reload (development) |

**Examples:**

```bash
# Start server
ingestforge serve

# Custom port
ingestforge serve --port 3000

# Development mode with auto-reload
ingestforge serve --reload
```

**Output:**

```
Starting IngestForge API server
Running at http://0.0.0.0:8000
API docs at http://0.0.0.0:8000/docs
```

---

### debate

Analyze pro/con arguments on a topic from your sources.

```bash
ingestforge debate <TOPIC> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file |
| `--top, -k` | `30` | Number of results to search |

**Examples:**

```bash
ingestforge debate "Should AI be regulated?"
ingestforge debate "Is remote work productive?" --output debate.md
```

---

### support

Find supporting evidence for a claim.

```bash
ingestforge support <CLAIM> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file |
| `--top, -k` | `20` | Number of results to search |

**Examples:**

```bash
ingestforge support "AI increases productivity"
ingestforge support "Climate change affects agriculture" --output evidence.md
```

---

### counter

Find counterarguments to a claim.

```bash
ingestforge counter <CLAIM> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file |
| `--top, -k` | `20` | Number of results to search |

**Examples:**

```bash
ingestforge counter "AI will eliminate most jobs"
```

---

### conflicts

Find contradictions between sources on a topic.

```bash
ingestforge conflicts <TOPIC> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file |
| `--top, -k` | `30` | Number of results to search |

**Examples:**

```bash
ingestforge conflicts "economic growth"
```

---

### gaps

Identify research gaps and missing coverage.

```bash
ingestforge gaps <TOPIC> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file |
| `--top, -k` | `30` | Number of results to search |

**Examples:**

```bash
ingestforge gaps "AI ethics"
ingestforge gaps "machine learning" --output gaps.md
```

---

### scholars

Identify key authors and contributors on a topic.

```bash
ingestforge scholars <TOPIC> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file |
| `--top, -k` | `30` | Number of results to search |

**Examples:**

```bash
ingestforge scholars "quantum computing"
```

---

### timeline

Build a chronological timeline from sources.

```bash
ingestforge timeline <TOPIC> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file |
| `--top, -k` | `30` | Number of results to search |

**Examples:**

```bash
ingestforge timeline "artificial intelligence"
```

---

### connect

Find connections between two concepts.

```bash
ingestforge connect <CONCEPT_A> <CONCEPT_B> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file |
| `--top, -k` | `20` | Number of results per concept |

**Examples:**

```bash
ingestforge connect "inflation" "unemployment"
```

---

### quote

Find quotable passages from your sources.

```bash
ingestforge quote <TOPIC> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file |
| `--top, -k` | `20` | Number of results to search |

**Examples:**

```bash
ingestforge quote "automation anxiety"
```

---

### cite

Generate a formatted citation for a source.

```bash
ingestforge cite <QUERY> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--style, -s` | `apa` | Citation style: apa, mla, chicago, harvard, ieee, bibtex |

**Examples:**

```bash
ingestforge cite "Smith 2023" --style apa
ingestforge cite "machine learning" --style mla
```

---

### thesis

Evaluate a thesis statement against your sources.

```bash
ingestforge thesis <THESIS_TEXT> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file |
| `--top, -k` | `20` | Number of results to search |

**Examples:**

```bash
ingestforge thesis "AI will fundamentally transform the labor market"
```

---

### draft

Generate draft paragraphs with inline citations.

```bash
ingestforge draft <TOPIC> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file |
| `--top, -k` | `20` | Number of results to search |

**Examples:**

```bash
ingestforge draft "effects of inflation on consumer spending"
ingestforge draft "neural networks" --output draft.md
```

---

### bibliography

Generate a bibliography of all indexed sources.

```bash
ingestforge bibliography [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--style, -s` | `apa` | Citation style: apa, mla, chicago, harvard, ieee, bibtex |
| `--output, -o` | None | Output file path |

**Examples:**

```bash
ingestforge bibliography --style apa
ingestforge bibliography --style bibtex --output refs.bib
```

---

### concept-map

Generate a concept map from your indexed sources as a Mermaid diagram.

```bash
ingestforge concept-map [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file |
| `--max, -m` | `30` | Maximum concepts to include |

**Examples:**

```bash
ingestforge concept-map
ingestforge concept-map --max 20 --output map.md
```

---

### folder-export

Export all research materials as an organized folder structure.

```bash
ingestforge folder-export <OUTPUT_DIR>
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `OUTPUT_DIR` | Yes | Output directory for export |

**Examples:**

```bash
ingestforge folder-export ./my_study_package
```

**Output structure:**

```
my_study_package/
├── 00_START_HERE.md
├── 01_overview.md
├── 02_glossary.md
├── 03_concept_map.md
├── 04_study_notes/
├── 05_flashcards.csv
├── 06_quiz.md
├── 07_reading_list.md
└── bibliography.bib
```

---

### discover

Search academic papers and educational resources on a topic.

```bash
ingestforge discover <TOPIC> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--academic/--no-academic` | `--academic` | Search academic papers (arXiv, Semantic Scholar) |
| `--educational/--no-educational` | `--educational` | Search educational resources |
| `--max, -m` | `10` | Maximum results per source |

**Examples:**

```bash
ingestforge discover "quantum computing"
ingestforge discover "machine learning" --no-educational
ingestforge discover "calculus" --no-academic --max 20
```

---

### prerequisites

Detect prerequisite concepts for a topic from your sources.

```bash
ingestforge prerequisites <TOPIC> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | None | Output markdown file |
| `--top, -k` | `30` | Number of results to search |

**Examples:**

```bash
ingestforge prerequisites "machine learning"
ingestforge prerequisites "quantum computing" --output prereqs.md
```

---

### reset

Reset all processed data (destructive).

```bash
ingestforge reset [OPTIONS]
```

**Options:**

| Option | Required | Description |
|--------|----------|-------------|
| `--confirm, -y` | Yes | Confirm the reset |

**Examples:**

```bash
# Will prompt for confirmation
ingestforge reset

# Confirmed reset
ingestforge reset --confirm
```

**Warning:** This permanently deletes:
- All indexed chunks
- All embeddings
- All search indexes
- Processing history

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `INGESTFORGE_CONFIG` | Path to config file (default: `./config.yaml`) |
| `INGESTFORGE_OCR_ENGINE` | OCR engine preference: `auto`, `tesseract`, `easyocr` |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error (file not found, processing failed, etc.) |

---

## Tips

### Processing Large Documents

```bash
# Process in background with watch
ingestforge watch &

# Drop files into pending folder
cp large-textbook.pdf .ingest/pending/

# Monitor progress
ingestforge status
```

### Batch Processing

```bash
# Process multiple files
for f in papers/*.pdf; do
    ingestforge ingest "$f"
done
```

### Combining with Other Tools

```bash
# Export notes and open in editor
ingestforge export "topic" --output notes.md && code notes.md

# Generate flashcards and count
ingestforge flashcards && wc -l flashcards.csv
```

---

*Last updated: 2026-02-03*
