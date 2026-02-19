"""PostgreSQL schema migrations.

This package contains SQL migration files for the PostgreSQL storage backend.
Each migration is a Python module with an `up` function that applies the migration.

Migration naming: v{version}_{description}.py
Example: v002_add_library_index.py
"""

from pathlib import Path
from typing import List, Tuple

MIGRATIONS_DIR = Path(__file__).parent


def get_available_migrations() -> List[Tuple[int, str, Path]]:
    """Get list of available migrations.

    Returns:
        List of (version, description, path) tuples sorted by version
    """
    migrations = []
    for path in MIGRATIONS_DIR.glob("v*.py"):
        if path.name == "__init__.py":
            continue
        # Parse version from filename: v001_description.py
        name = path.stem
        version_str = name.split("_")[0][1:]  # Remove 'v' prefix
        try:
            version = int(version_str)
            description = "_".join(name.split("_")[1:])
            migrations.append((version, description, path))
        except ValueError:
            continue

    return sorted(migrations, key=lambda x: x[0])
