"""
Main configuration class for IngestForge.

This module provides the Config dataclass that aggregates all sub-configs
and handles initialization, validation, path management, and YAML parsing.

Architecture Context
--------------------
Configuration sits at the Core layer and is consumed by every other module.
The Config object is typically created once at startup and passed through
the dependency chain to all components that need settings.

    User's config.yaml
           ↓
    load_config() → Config object
           ↓
    Passed to: Pipeline, Storage, Retriever, LLM clients, etc.

Configuration Hierarchy
-----------------------
The Config dataclass aggregates several nested configs:

    Config
    ├── ProjectConfig      # Project name, directories
    ├── IngestConfig       # File watching, supported formats
    ├── SplitConfig        # PDF chapter splitting
    ├── ChunkingConfig     # Semantic chunking parameters
    ├── EnrichmentConfig   # Embeddings, entities, questions
    ├── StorageConfig      # Backend selection (jsonl, chromadb)
    ├── RetrievalConfig    # Search settings, hybrid weights
    ├── LLMConfig          # Provider, model, API keys
    └── ServerConfig       # API server settings

Environment Variables
---------------------
Secrets and deployment-specific values use ${VAR_NAME} syntax:

    llm:
      api_key: ${ANTHROPIC_API_KEY}
      model: ${LLM_MODEL:claude-sonnet-4-20250514}  # with default

The expand_env_vars() function recursively processes all string values.

Key Design Decisions
--------------------
1. **Dataclasses over dicts**: Type safety and IDE autocompletion.
2. **Defaults for everything**: Zero-config operation is possible.
3. **Validation in __post_init__**: Catch config errors early at load time.
4. **Path expansion**: All path configs become absolute Path objects.

Usage Example
-------------
    # Load from default location (./config.yaml)
    config = load_config()

    # Load from specific path
    config = load_config(config_path=Path("custom_config.yaml"))

    # Access nested settings
    model = config.enrichment.embedding_model
    top_k = config.retrieval.top_k

    # Paths are ready to use
    pending_dir = config.pending_path  # Returns absolute Path
"""

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from ingestforge.core.config.base import IngestConfig, ProjectConfig, SplitConfig
from ingestforge.core.config.chunking import ChunkingConfig, RefinementConfig
from ingestforge.core.config.enrichment import EnrichmentConfig
from ingestforge.core.config.features import (
    APIConfig,
    DoctrineAPIConfig,
    FeatureAnalysisConfig,
    LiteraryConfig,
    OCRConfig,
    RedactionConfig,
    ResearchConfig,
    WebSearchConfig,
)
from ingestforge.core.config.llm import LlamaCppConfig, LLMConfig, LLMProviderConfig
from ingestforge.core.config.retrieval import (
    AuthorityConfig,
    HybridConfig,
    ParentDocConfig,
    RetrievalConfig,
)
from ingestforge.core.config.storage import (
    ChromaDBConfig,
    PostgresConfig,
    StorageConfig,
)


@dataclass
class Config:
    """Main IngestForge configuration."""

    # Use __slots__ for memory efficiency (dataclasses support this via slots=True in 3.10+)
    # Note: Can't use __slots__ directly with field(default_factory=...) in older Python
    # Instead we add validation assertions in __post_init__

    project: ProjectConfig = field(default_factory=ProjectConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    api: APIConfig = field(default_factory=APIConfig)
    literary: LiteraryConfig = field(default_factory=LiteraryConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    feature_analysis: FeatureAnalysisConfig = field(
        default_factory=FeatureAnalysisConfig
    )
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    performance_mode: str = "balanced"  # quality, balanced, speed, mobile

    # Runtime paths (set after loading)
    _base_path: Path = field(default_factory=Path.cwd, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Validates:
        - Chunking size constraints
        - Hybrid weight normalization
        - Critical path validation (data_dir not root)
        - Nested config type assertions
        """
        # JPL #5: Assertions for nested config types
        assert isinstance(self.project, ProjectConfig), "project must be ProjectConfig"
        assert isinstance(self.ingest, IngestConfig), "ingest must be IngestConfig"
        assert isinstance(
            self.chunking, ChunkingConfig
        ), "chunking must be ChunkingConfig"
        assert isinstance(self.storage, StorageConfig), "storage must be StorageConfig"
        assert isinstance(
            self.retrieval, RetrievalConfig
        ), "retrieval must be RetrievalConfig"
        assert isinstance(self.llm, LLMConfig), "llm must be LLMConfig"

        # Validate chunking constraints
        if self.chunking.min_size >= self.chunking.max_size:
            raise ValueError("chunking.min_size must be less than chunking.max_size")

        # Validate critical paths - data_dir must not be root or empty
        assert self.project.data_dir not in (
            "/",
            "\\",
            "",
        ), f"data_dir must not be root or empty: {self.project.data_dir}"
        assert self.project.ingest_dir not in (
            "/",
            "\\",
            "",
        ), f"ingest_dir must not be root or empty: {self.project.ingest_dir}"

        # Validate performance_mode
        valid_modes = {"quality", "balanced", "speed", "mobile"}
        assert (
            self.performance_mode.lower() in valid_modes
        ), f"performance_mode must be one of {valid_modes}, got: {self.performance_mode}"

        # Normalize hybrid weights
        if (
            self.retrieval.hybrid.bm25_weight + self.retrieval.hybrid.semantic_weight
            != 1.0
        ):
            total = (
                self.retrieval.hybrid.bm25_weight
                + self.retrieval.hybrid.semantic_weight
            )
            if total == 0:
                # Both weights zero: default to equal split
                self.retrieval.hybrid.bm25_weight = 0.5
                self.retrieval.hybrid.semantic_weight = 0.5
            else:
                self.retrieval.hybrid.bm25_weight /= total
                self.retrieval.hybrid.semantic_weight /= total

    @property
    def data_path(self) -> Path:
        """Get absolute path to data directory."""
        return self._base_path / self.project.data_dir

    @property
    def ingest_path(self) -> Path:
        """Get absolute path to ingest directory."""
        return self._base_path / self.project.ingest_dir

    @property
    def pending_path(self) -> Path:
        """Get path to pending documents directory."""
        if self.ingest.pending_path_override:
            override = Path(self.ingest.pending_path_override)
            # If relative, resolve against base_path; if absolute, use as-is
            if override.is_absolute():
                return override
            return self._base_path / override
        return self.ingest_path / "pending"

    @property
    def processing_path(self) -> Path:
        """Get path to processing directory."""
        return self.ingest_path / "processing"

    @property
    def completed_path(self) -> Path:
        """Get path to completed documents directory."""
        return self.ingest_path / "completed"

    @property
    def chunks_path(self) -> Path:
        """Get path to chunks storage."""
        return self.data_path / "chunks"

    @property
    def chromadb_path(self) -> Path:
        """Get path to ChromaDB storage."""
        return self._base_path / self.storage.chromadb.persist_directory

    def ensure_directories(self) -> None:
        """Create all required directories."""
        directories = [
            self.pending_path,
            self.processing_path,
            self.completed_path,
            self.data_path / "chunks",
            self.data_path / "embeddings",
            self.data_path / "index",
        ]

        if self.storage.backend == "chromadb":
            directories.append(self.chromadb_path)

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if key.startswith("_"):
                continue
            result[key] = value
        return result

    @staticmethod
    def _filter_fields(cls_type: Any, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Filter dict to only keys that match dataclass fields, handling None."""
        if not data:
            return {}
        valid_keys = {f.name for f in fields(cls_type)}
        return {k: v for k, v in data.items() if k in valid_keys}

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], base_path: Optional[Path] = None
    ) -> "Config":
        """Create Config from dictionary."""
        # Import here to avoid circular dependency
        from ingestforge.core.config_loaders import expand_env_vars

        data = expand_env_vars(data)

        simple_configs = cls._parse_simple_configs(data)
        storage = cls._parse_storage_config(data)
        retrieval = cls._parse_retrieval_config(data)
        llm = cls._parse_llm_config(data)
        api = APIConfig(**data.get("api", {}))
        ocr = OCRConfig(**cls._filter_fields(OCRConfig, data.get("ocr")))
        research = cls._parse_research_config(data)
        feature_analysis = cls._parse_feature_analysis_config(data)
        redaction = RedactionConfig(
            **cls._filter_fields(RedactionConfig, data.get("redaction"))
        )

        config = cls(
            **simple_configs,
            storage=storage,
            retrieval=retrieval,
            llm=llm,
            api=api,
            ocr=ocr,
            research=research,
            feature_analysis=feature_analysis,
            redaction=redaction,
            performance_mode=data.get("performance_mode", "balanced"),
        )

        if base_path:
            config._base_path = base_path

        return config

    @classmethod
    def _parse_simple_configs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse simple nested configs without complex nesting."""
        return {
            "project": ProjectConfig(
                **cls._filter_fields(ProjectConfig, data.get("project"))
            ),
            "ingest": IngestConfig(
                **cls._filter_fields(IngestConfig, data.get("ingest"))
            ),
            "split": SplitConfig(**cls._filter_fields(SplitConfig, data.get("split"))),
            "chunking": ChunkingConfig(
                **cls._filter_fields(ChunkingConfig, data.get("chunking"))
            ),
            "refinement": RefinementConfig(
                **cls._filter_fields(RefinementConfig, data.get("refinement"))
            ),
            "enrichment": EnrichmentConfig(
                **cls._filter_fields(EnrichmentConfig, data.get("enrichment"))
            ),
        }

    @classmethod
    def _parse_storage_config(cls, data: Dict[str, Any]) -> StorageConfig:
        """Parse storage config with nested chromadb and postgres configs."""
        storage_data = data.get("storage") or {}
        chromadb = ChromaDBConfig(
            **cls._filter_fields(ChromaDBConfig, storage_data.get("chromadb"))
        )
        postgres = PostgresConfig(
            **cls._filter_fields(PostgresConfig, storage_data.get("postgres"))
        )
        return StorageConfig(
            backend=storage_data.get("backend", "chromadb"),
            compression=storage_data.get("compression", False),
            chromadb=chromadb,
            postgres=postgres,
        )

    @classmethod
    def _parse_retrieval_config(cls, data: Dict[str, Any]) -> RetrievalConfig:
        """Parse retrieval config with nested hybrid, parent_doc, and authority configs."""
        retrieval_data = data.get("retrieval") or {}
        hybrid = HybridConfig(
            **cls._filter_fields(HybridConfig, retrieval_data.get("hybrid"))
        )
        parent_doc_data = retrieval_data.get("parent_doc") or {}
        parent_doc = ParentDocConfig(
            **cls._filter_fields(ParentDocConfig, parent_doc_data)
        )
        authority_data = retrieval_data.get("authority") or {}
        authority = AuthorityConfig(
            enabled=authority_data.get("enabled", False),
            default_level=authority_data.get("default_level", 4),
            patterns=authority_data.get("patterns", []),
        )
        return RetrievalConfig(
            strategy=retrieval_data.get("strategy", "hybrid"),
            top_k=retrieval_data.get("top_k", 10),
            rerank=retrieval_data.get("rerank", True),
            rerank_model=retrieval_data.get(
                "rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            hybrid=hybrid,
            parent_doc=parent_doc,
            authority=authority,
        )

    @classmethod
    def _parse_llm_config(cls, data: Dict[str, Any]) -> LLMConfig:
        """Parse LLM config with multiple provider configs."""
        llm_data = data.get("llm") or {}
        gemini = LLMProviderConfig(
            **cls._filter_fields(
                LLMProviderConfig,
                llm_data.get("gemini") or {"model": "gemini-1.5-flash"},
            )
        )
        claude = LLMProviderConfig(
            **cls._filter_fields(
                LLMProviderConfig,
                llm_data.get("claude") or {"model": "claude-3-haiku-20240307"},
            )
        )
        openai = LLMProviderConfig(
            **cls._filter_fields(
                LLMProviderConfig, llm_data.get("openai") or {"model": "gpt-4o-mini"}
            )
        )
        ollama_data = llm_data.get("ollama", {})
        ollama = LLMProviderConfig(
            model=ollama_data.get("model", "qwen2.5:14b"),
            url=ollama_data.get("url", "http://localhost:11434"),
        )
        llamacpp_data = llm_data.get("llamacpp", {})
        llamacpp = LlamaCppConfig(
            model_path=llamacpp_data.get("model_path", ""),
            n_ctx=llamacpp_data.get("n_ctx", 8192),
            n_gpu_layers=llamacpp_data.get("n_gpu_layers", 0),
            n_threads=llamacpp_data.get("n_threads", 0),
            auto_gpu_layers=llamacpp_data.get("auto_gpu_layers", True),
        )
        return LLMConfig(
            default_provider=llm_data.get("default_provider", "llamacpp"),
            gemini=gemini,
            claude=claude,
            openai=openai,
            ollama=ollama,
            llamacpp=llamacpp,
        )

    @classmethod
    def _parse_research_config(cls, data: Dict[str, Any]) -> ResearchConfig:
        """Parse research config with nested web_search config."""
        research_data = data.get("research") or {}
        web_search_data = research_data.get("web_search") or {}
        web_search = WebSearchConfig(
            **cls._filter_fields(WebSearchConfig, web_search_data)
        )
        return ResearchConfig(
            web_search=web_search,
            auto_process=research_data.get("auto_process", True),
            max_sources_per_session=research_data.get("max_sources_per_session", 50),
            default_scrape_delay_min=research_data.get("default_scrape_delay_min", 1.0),
            default_scrape_delay_max=research_data.get("default_scrape_delay_max", 3.0),
        )

    @classmethod
    def _parse_feature_analysis_config(
        cls, data: Dict[str, Any]
    ) -> FeatureAnalysisConfig:
        """Parse feature analysis config with nested doctrine_api config."""
        fa_data = data.get("feature_analysis") or {}
        doctrine_data = fa_data.get("doctrine_api") or {}
        doctrine_api = DoctrineAPIConfig(
            url=doctrine_data.get("url", "http://localhost:8000"),
            enabled=doctrine_data.get("enabled", True),
            timeout_seconds=doctrine_data.get("timeout_seconds", 30),
            top_k=doctrine_data.get("top_k", 5),
        )
        return FeatureAnalysisConfig(
            doctrine_api=doctrine_api,
            max_code_results=fa_data.get("max_code_results", 20),
            max_story_results=fa_data.get("max_story_results", 10),
            max_generated_stories=fa_data.get("max_generated_stories", 5),
        )
