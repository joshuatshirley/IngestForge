# IngestForge Quick Start

## Installation

```bash
# Clone repository
git clone https://github.com/joshuatshirley/IngestForge.git
cd IngestForge

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Verify Installation

```bash
# Run verification script
python verify_setup.py

# Test CLI
ingestforge --help

# Check version
python -c "import ingestforge; print(ingestforge.__version__)"
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ingestforge --cov-report=term-missing

# Run specific test file
pytest tests/unit/cli/core/test_ingest.py -v
```

### Type Checking

```bash
# Check all code
mypy ingestforge/ --strict

# Check specific module
mypy ingestforge/cli/core/ --strict
```

### Code Quality

```bash
# Validate code quality guidelines
python validate_commandments.py

# Lint code
pylint ingestforge/

# Format check (if using black)
black --check ingestforge/
```

## Project Structure

```
IngestForge/
├── ingestforge/           # Main package
│   ├── cli/              # CLI commands (16 groups, 47 commands)
│   ├── core/             # Core infrastructure (Pipeline, Security, Config)
│   ├── storage/          # Storage backends (ChromaDB, Postgres)
│   └── llm/              # Unified LLM provider interfaces
├── tests/                # GWT-based unit and integration tests
├── docs/                 # Technical documentation and guides
└── .github/workflows/    # CI/CD pipelines (CI, Lint, Security)
```

## Available Commands

### Core Operations
- `ingestforge ingest` - Process documents
- `ingestforge chunk` - Split documents (planned)
- `ingestforge enrich` - Enrich metadata (planned)
- `ingestforge embed` - Generate embeddings (planned)
- `ingestforge store` - Store in database (planned)
- `ingestforge retrieve` - Search and retrieve (planned)
- `ingestforge query` - AI-powered Q&A (planned)

### Analysis Tools
- `ingestforge analyze scan` - Scan documents (planned)
- `ingestforge analyze structure` - Extract structure (planned)
- `ingestforge analyze quality` - Quality check (planned)

### Configuration
- `ingestforge init` - Initialize project
- `ingestforge status` - Show status
- `ingestforge config init` - Setup config (planned)
- `ingestforge config list` - List settings (planned)

### Additional Features
- `ingestforge lit` - Literary analysis (planned)
- `ingestforge research` - Research tools (planned)
- `ingestforge study` - Study aids (planned)
- `ingestforge citation` - Citation tools (planned)
- `ingestforge code` - Code analysis (planned)
- `ingestforge export` - Export data (planned)
- `ingestforge transform` - Transform data (planned)
- `ingestforge workflow` - Automation (planned)
- `ingestforge maintenance` - Maintenance (planned)
- `ingestforge monitor` - Monitoring (planned)
- `ingestforge index` - Index management (planned)
- `ingestforge interactive` - Interactive mode (planned)

## Configuration

### Project Initialization

```bash
# Create new project
ingestforge init my-project

# Navigate to project
cd my-project

# Check status
ingestforge status
```

### Configuration File

Default config at `.ingestforge/config.yaml`:

```yaml
project:
  name: my-project
  version: "1.2.0"

storage:
  backend: chromadb
  path: .ingestforge/storage

llm:
  provider: ollama
  model: llama2

chunking:
  strategy: semantic
  size: 512
  overlap: 50
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code quality standards (code quality guidelines)
- Development setup
- Testing requirements
- Pull request process

## Getting Help

```bash
# General help
ingestforge --help

# Command help
ingestforge [command] --help

# Subcommand help
ingestforge [group] [subcommand] --help
```

## Common Tasks

### Add New Command

1. Create command file in appropriate group
2. Implement command following code quality guidelines
3. Add tests in `tests/unit/cli/[group]/`
4. Update group's `__init__.py`
5. Run validation: `python validate_commandments.py`
6. Run tests: `pytest tests/ -v`

### Run Quality Checks

```bash
# All checks
python validate_commandments.py && \
pytest tests/ --cov=ingestforge && \
mypy ingestforge/ --strict

# Or use pre-commit (if configured)
pre-commit run --all-files
```

## Troubleshooting

### Import Errors

```bash
# Reinstall package
pip install -e .

# Check installation
python -c "import ingestforge"
```

### Test Failures

```bash
# Run with verbose output
pytest tests/ -vv

# Run with debugging
pytest tests/ --pdb
```

### Type Errors

```bash
# Check specific file
mypy ingestforge/cli/core/ingest.py --strict

# Show error context
mypy ingestforge/ --strict --show-error-context
```

## Resources

- **Documentation:** `docs/`
- **Examples:** `examples/` (Implementation tracked in Tasks 310-314)
- **Issue Tracker:** GitHub Issues
- **Discussions:** GitHub Discussions

## License

MIT License - see [LICENSE](LICENSE) for details
