"""CLI commands package.

This package contains all IngestForge CLI command implementations.
Each command is a separate module following the command pattern:

- Command class extending IngestForgeCommand
- Typer wrapper function for CLI integration
- All commands follow code quality guidelines (small functions, type safety, etc.)

Commands:
- status: Display project status
- init: Initialize new project
- query: Search and answer questions
- ingest: Process documents
- preview: Preview chunk with context
- mark: Mark chunks as read/unread (ORG-001)
- tag: Add/remove tags from chunks (ORG-002)
- tags: List all tags (ORG-002)
- bookmark: Save/remove bookmarks (ORG-003)
- bookmarks: List all bookmarks (ORG-003)
- annotate: Add annotations to chunks (ORG-004)
- annotations: Manage annotations (ORG-004)
- link: Create manual links between entities ()
- links: List manual links ()
- unlink: Delete manual links ()
- auth-wizard: Interactive API key setup (UX-018)
- reset: Reset project and clear data (DEAD-D04)
- api: Start REST API server (TICKET-504)
- demo: Try pre-loaded sample documents ()

Usage:
    from ingestforge.cli.commands import status, init, query, ingest, preview, mark, tag
"""

from __future__ import annotations

# Import command wrappers (not classes, to avoid circular imports)
from ingestforge.cli.commands.status import command as status_command
from ingestforge.cli.commands.init import command as init_command
from ingestforge.cli.commands.query import command as query_command
from ingestforge.cli.commands.ingest import command as ingest_command
from ingestforge.cli.commands.preview import command as preview_command
from ingestforge.cli.commands.mark import command as mark_command
from ingestforge.cli.commands.tag import command as tag_command
from ingestforge.cli.commands.tag import list_tags_command as tags_command
from ingestforge.cli.commands.bookmark import command as bookmark_command
from ingestforge.cli.commands.bookmark import (
    list_bookmarks_command as bookmarks_command,
)
from ingestforge.cli.commands.annotate import command as annotate_command
from ingestforge.cli.commands.annotate import annotations_app
from ingestforge.cli.commands.link import command as link_command
from ingestforge.cli.commands.link import list_links_command as links_command
from ingestforge.cli.commands.link import delete_link_command as unlink_command
from ingestforge.cli.commands.auth_wizard import command as auth_wizard_command
from ingestforge.cli.commands.reset import reset_command
from ingestforge.cli.commands.api import api_app
from ingestforge.cli.commands.demo import demo_command

__all__ = [
    "status_command",
    "init_command",
    "query_command",
    "ingest_command",
    "preview_command",
    "mark_command",
    "tag_command",
    "tags_command",
    "bookmark_command",
    "bookmarks_command",
    "annotate_command",
    "annotations_app",
    "link_command",
    "links_command",
    "unlink_command",
    "auth_wizard_command",
    "reset_command",
    "api_app",
    "demo_command",
]
