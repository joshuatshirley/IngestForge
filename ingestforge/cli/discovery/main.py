"""Discovery command group - Research and learning resource discovery.

This module registers all discovery commands for academic paper search:
- arxiv: Search arXiv papers (raw HTTP client)
- arxiv-lib: Search arXiv using arxiv library (RES-002)
- arxiv-download: Download specific arXiv paper
- scholar: Search Semantic Scholar
- scholar-citations: Get citations for a paper
- scholar-references: Get references for a paper
- crossref: Lookup DOI
- crossref-search: Search CrossRef
- court: Search CourtListener court opinions (LEGAL-001)
- court-detail: Get case details by cluster ID
- court-download: Download opinion text
- court-list: List federal court codes
- cve: Search CVE vulnerabilities (CYBER-002)
- cve-get: Get specific CVE details"""

import typer
from ingestforge.cli.discovery import (
    academic,
    arxiv,
    arxiv_discovery,
    court,
    crossref,
    cve,
    educational,
    scholar,
    scholars,
    timeline,
)

app = typer.Typer(
    name="discovery",
    help="Discover research papers, educational resources, and scholars",
    no_args_is_help=True,
)

# Original commands
app.command(name="academic")(academic.command)
app.command(name="educational")(educational.command)
app.command(name="scholars")(scholars.command)
app.command(name="timeline")(timeline.command)

# Enhanced arXiv commands
app.command(name="arxiv")(arxiv.command)
app.command(name="arxiv-lib")(arxiv_discovery.command)
app.command(name="arxiv-download")(arxiv.download_command)

# Enhanced Semantic Scholar commands
app.command(name="scholar")(scholar.command)
app.command(name="scholar-citations")(scholar.citations_command)
app.command(name="scholar-references")(scholar.references_command)

# CrossRef commands
app.command(name="crossref")(crossref.lookup_command)
app.command(name="crossref-search")(crossref.search_command)

# CourtListener commands (LEGAL-001)
app.command(name="court")(court.search_command)
app.command(name="court-detail")(court.detail_command)
app.command(name="court-download")(court.download_command)
app.command(name="court-list")(court.list_courts_command)

# CVE/NVD commands (CYBER-002)
app.command(name="cve")(cve.command)
app.command(name="cve-get")(cve.get_command)
