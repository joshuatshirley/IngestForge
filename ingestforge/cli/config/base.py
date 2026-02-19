"""Base class for config commands.

Provides shared functionality for configuration management.

Follows Commandments #4 (Small Functions), #6 (Smallest Scope),
and #9 (Type Safety).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import json

from ingestforge.cli.core.command_base import BaseCommand


class ConfigCommand(BaseCommand):
    """Base class for config commands."""

    def get_config_path(self, project: Optional[Path] = None) -> Path:
        """Get configuration file path.

        Checks for ingestforge.yaml first (preferred), then falls back to
        legacy .ingestforge/config.json.

        Args:
            project: Project directory

        Returns:
            Path to config file
        """
        base_dir = project if project else Path.cwd()

        # Prefer ingestforge.yaml (new format)
        yaml_path = base_dir / "ingestforge.yaml"
        if yaml_path.exists():
            return yaml_path

        # Also check for .yaml extension
        yml_path = base_dir / "ingestforge.yml"
        if yml_path.exists():
            return yml_path

        # Fall back to legacy JSON config
        json_path = base_dir / ".ingestforge" / "config.json"
        if json_path.exists():
            return json_path

        # Default to YAML path (will trigger default config)
        return yaml_path

    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file.

        Supports both YAML and JSON formats.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If config cannot be loaded
        """
        if not config_path.exists():
            return self.get_default_config()

        try:
            with config_path.open("r", encoding="utf-8") as f:
                # Detect format by extension
                if config_path.suffix in (".yaml", ".yml"):
                    import yaml

                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)

            # Normalize YAML config to match expected structure
            return self._normalize_config(config)

        except Exception as e:
            raise ValueError(f"Cannot load config: {e}")

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize YAML config to match expected display format.

        Args:
            config: Raw configuration dictionary

        Returns:
            Normalized configuration dictionary
        """
        # If already in expected format, return as-is
        if "llm" in config and "provider" in config.get("llm", {}):
            return config

        # Convert from ingestforge.yaml format
        normalized = {
            "version": config.get("project", {}).get("version", "1.0"),
            "llm": {},
            "embedding": {},
            "storage": {},
            "chunking": {},
            "retrieval": {},
        }

        # LLM settings
        llm_config = config.get("llm", {})
        normalized["llm"]["provider"] = llm_config.get("default_provider", "llamacpp")

        # Get model from provider-specific config
        provider = normalized["llm"]["provider"]
        if provider == "llamacpp":
            llamacpp = llm_config.get("llamacpp", {})
            model_path = llamacpp.get("model_path", "")
            # Extract model name from path
            model_name = Path(model_path).stem if model_path else "unknown"
            normalized["llm"]["model"] = model_name
            normalized["llm"]["n_ctx"] = llamacpp.get("n_ctx", 8192)

        # Embedding settings
        embedding_config = config.get("embedding", {})
        normalized["embedding"]["provider"] = embedding_config.get(
            "provider", "sentence-transformers"
        )
        normalized["embedding"]["model"] = embedding_config.get(
            "model", "all-MiniLM-L6-v2"
        )

        # Storage settings
        storage_config = config.get("storage", {})
        normalized["storage"]["backend"] = storage_config.get("backend", "chromadb")
        chromadb = storage_config.get("chromadb", {})
        normalized["storage"]["path"] = chromadb.get(
            "persist_directory", ".data/chromadb"
        )

        # Chunking settings
        chunking_config = config.get("chunking", {})
        normalized["chunking"]["size"] = chunking_config.get("chunk_size", 512)
        normalized["chunking"]["overlap"] = chunking_config.get("chunk_overlap", 50)
        normalized["chunking"]["strategy"] = chunking_config.get("strategy", "semantic")

        # Retrieval settings
        retrieval_config = config.get("retrieval", {})
        normalized["retrieval"]["top_k"] = retrieval_config.get("top_k", 5)
        normalized["retrieval"]["score_threshold"] = retrieval_config.get(
            "score_threshold", 0.7
        )
        normalized["retrieval"]["rerank"] = retrieval_config.get("rerank", False)

        return normalized

    def save_config(self, config_path: Path, config: Dict[str, Any]) -> None:
        """Save configuration to file.

        Args:
            config_path: Path to config file
            config: Configuration dictionary

        Raises:
            ValueError: If config cannot be saved
        """
        try:
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Save config
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        except Exception as e:
            raise ValueError(f"Cannot save config: {e}")

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "version": "1.0",
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimensions": 1536,
            },
            "storage": {
                "backend": "chromadb",
                "path": ".ingestforge/storage",
                "collection": "documents",
            },
            "chunking": {
                "size": 1000,
                "overlap": 100,
                "strategy": "simple",
            },
            "retrieval": {
                "top_k": 5,
                "score_threshold": 0.7,
                "rerank": True,
            },
            "processing": {
                "batch_size": 10,
                "parallel": True,
                "max_workers": 4,
            },
        }

    def validate_config_structure(
        self, config: Dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Validate configuration structure.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors: list[str] = []
        required_sections = [
            "llm",
            "embedding",
            "storage",
            "chunking",
            "retrieval",
        ]

        # Check required sections
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")

        # Validate LLM config
        if "llm" in config:
            llm = config["llm"]
            if "provider" not in llm:
                errors.append("Missing llm.provider")
            if "model" not in llm:
                errors.append("Missing llm.model")

        # Validate embedding config
        if "embedding" in config:
            embedding = config["embedding"]
            if "provider" not in embedding:
                errors.append("Missing embedding.provider")
            if "model" not in embedding:
                errors.append("Missing embedding.model")

        # Validate storage config
        if "storage" in config:
            storage = config["storage"]
            if "backend" not in storage:
                errors.append("Missing storage.backend")

        return (len(errors) == 0, errors)

    def get_config_value(self, config: Dict[str, Any], key: str) -> Optional[Any]:
        """Get configuration value by key.

        Args:
            config: Configuration dictionary
            key: Key in dot notation (e.g., "llm.model")

        Returns:
            Configuration value or None
        """
        parts = key.split(".")
        current = config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def set_config_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set configuration value by key.

        Args:
            config: Configuration dictionary
            key: Key in dot notation (e.g., "llm.model")
            value: Value to set

        Raises:
            ValueError: If key is invalid
        """
        parts = key.split(".")

        if len(parts) == 0:
            raise ValueError("Invalid key")

        # Navigate to parent
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set value
        current[parts[-1]] = value

    def format_config_display(self, config: Dict[str, Any]) -> str:
        """Format configuration for display.

        Args:
            config: Configuration dictionary

        Returns:
            Formatted configuration string
        """
        from rich.syntax import Syntax

        config_json = json.dumps(config, indent=2, ensure_ascii=False)
        return Syntax(config_json, "json", theme="monokai")

    def create_config_summary(self, config: Dict[str, Any]) -> list[str]:
        """Create summary of key configuration values.

        Args:
            config: Configuration dictionary

        Returns:
            List of summary lines
        """
        lines = [
            "[bold]Configuration Summary[/bold]",
            "",
            f"Version: {config.get('version', 'unknown')}",
            "",
            "[bold cyan]LLM:[/bold cyan]",
            f"  Provider: {config.get('llm', {}).get('provider', 'unknown')}",
            f"  Model: {config.get('llm', {}).get('model', 'unknown')}",
            "",
            "[bold cyan]Embedding:[/bold cyan]",
            f"  Provider: {config.get('embedding', {}).get('provider', 'unknown')}",
            f"  Model: {config.get('embedding', {}).get('model', 'unknown')}",
            "",
            "[bold cyan]Storage:[/bold cyan]",
            f"  Backend: {config.get('storage', {}).get('backend', 'unknown')}",
            f"  Path: {config.get('storage', {}).get('path', 'unknown')}",
            "",
            "[bold cyan]Chunking:[/bold cyan]",
            f"  Size: {config.get('chunking', {}).get('size', 'unknown')}",
            f"  Overlap: {config.get('chunking', {}).get('overlap', 'unknown')}",
        ]

        return lines
