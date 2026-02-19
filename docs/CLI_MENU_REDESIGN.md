# IngestForge CLI Menu Redesign

## Executive Summary

- **Current**: 144 features, 31 missing from menu, 3-4 clicks average
- **Proposed**: 100% coverage, 1-2 clicks for common tasks, 60% click reduction

## Current Problems

1. **31 features not accessible from menu** (Organization, Legal, CVE, etc.)
2. **High click depth** - Common tasks require 3-4 clicks
3. **Cognitive overload** - 6 pillars with overlapping categories
4. **No quick access** - Must navigate menus for simple operations

## Proposed Solution

### Quick Bar (Always Visible)

Single-key shortcuts for the 8 most common operations:

```
╔═══════════════════════════════════════════════════════════════════╗
║  [Q]uery  [I]ngest  [A]gent  [S]tatus  [F]lash  [E]xport  [G]UI  ║
╚═══════════════════════════════════════════════════════════════════╝
```

| Key | Action | Current Clicks | New |
|-----|--------|----------------|-----|
| Q | Query knowledge base | 3 | 1 |
| I | Ingest documents | 3 | 1 |
| A | Agent research | 4 | 1 |
| S | Project status | 2 | 1 |
| F | Flashcards/Study | 4 | 1 |
| E | Quick export | 4 | 1 |
| G | Launch GUI | 3 | 1 |
| ? | Help/Commands | N/A | 1 |

### Simplified Pillars (4 instead of 6)

```
┌─────────────────────────────────────────────────────────────────┐
│  [1] INGEST & ORGANIZE                                          │
│      Documents, Libraries, Tags, Bookmarks, Transform           │
│                                                                  │
│  [2] SEARCH & DISCOVER                                          │
│      Query, Academic, Legal, Security, Agent                    │
│                                                                  │
│  [3] ANALYZE & UNDERSTAND                                       │
│      Comprehension, Arguments, Literary, Code, Fact Check       │
│                                                                  │
│  [4] CREATE & EXPORT                                            │
│      Study Tools, Writing, Citations, Export                    │
│                                                                  │
│  [0] SYSTEM                                                      │
│      Config, LLM, Storage, Maintenance, API                     │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Menu Structure

### [1] INGEST & ORGANIZE

```
1.1  Add Documents
     ├─ [1] File/Folder      ingest <path>
     ├─ [2] YouTube Video    ingest <url>
     ├─ [3] Audio File       ingest <audio>
     └─ [4] Batch Import     workflow batch

1.2  Manage Documents
     ├─ [1] List All         (inline display)
     ├─ [2] Preview Chunk    preview <id>
     ├─ [3] Delete Document  (select from list)
     ├─ [4] Re-ingest        (select + reprocess)
     └─ [5] Clear All        (confirm + clear)

1.3  Organize (NEW!)
     ├─ [1] Tags             tag, tags
     ├─ [2] Bookmarks        bookmark, bookmarks
     ├─ [3] Annotations      annotate, annotations
     └─ [4] Mark Read/Unread mark

1.4  Libraries
     ├─ [1] List Libraries   (inline)
     ├─ [2] Query Library    query --library
     ├─ [3] Move Document    (select + move)
     └─ [4] Delete Library   (select + delete)

1.5  Transform
     ├─ [1] Clean Text       transform clean (with new refiners!)
     ├─ [2] Split Document   transform split
     ├─ [3] Merge/Dedupe     transform merge
     ├─ [4] Filter Chunks    transform filter
     └─ [5] Enrich Metadata  transform enrich
```

### [2] SEARCH & DISCOVER

```
2.1  Query Knowledge Base
     ├─ [1] Quick Query      query <question>
     ├─ [2] Library Query    query --library <lib>
     └─ [3] Interactive      interactive ask

2.2  Academic Research
     ├─ [1] Semantic Scholar discovery scholar
     ├─ [2] arXiv Search     discovery arxiv
     ├─ [3] arXiv Download   discovery arxiv-download
     ├─ [4] CrossRef/DOI     discovery crossref
     ├─ [5] Citations        discovery scholar-citations
     ├─ [6] References       discovery scholar-references
     └─ [7] Find Researchers discovery scholars

2.3  Legal Research (NEW!)
     ├─ [1] Court Opinions   discovery court
     ├─ [2] Case Details     discovery court-detail
     ├─ [3] Download Opinion discovery court-download
     └─ [4] Jurisdictions    discovery court-list

2.4  Security Research (NEW!)
     ├─ [1] CVE Search       discovery cve
     └─ [2] CVE Details      discovery cve-get

2.5  Agent Research
     ├─ [1] Run Agent        agent run
     ├─ [2] Agent Status     agent status
     ├─ [3] Available Tools  agent tools
     └─ [4] Test LLM         agent test-llm
```

### [3] ANALYZE & UNDERSTAND

```
3.1  Comprehension
     ├─ [1] Explain Concept  comprehension explain
     ├─ [2] Compare Concepts comprehension compare
     └─ [3] Connect Ideas    comprehension connect

3.2  Arguments
     ├─ [1] Find Conflicts   argument conflicts
     ├─ [2] Counter Args     argument counter
     ├─ [3] Debate Analysis  argument debate
     ├─ [4] Knowledge Gaps   argument gaps
     └─ [5] Find Support     argument support

3.3  Literary
     ├─ [1] Story Arc        lit arc
     ├─ [2] Characters       lit character
     ├─ [3] Themes           lit themes
     ├─ [4] Symbols          lit symbols
     └─ [5] Outline          lit outline

3.4  Content Analysis
     ├─ [1] Topics           analyze topics
     ├─ [2] Entities         analyze entities
     ├─ [3] Relationships    analyze relationships (NEW!)
     ├─ [4] Connections      analyze connections (NEW!)
     ├─ [5] Timeline         analyze timeline (NEW!)
     ├─ [6] Similarity       analyze similarity
     ├─ [7] Duplicates       analyze duplicates
     └─ [8] Knowledge Graph  analyze knowledge-graph

3.5  Code Analysis
     ├─ [1] Analyze Code     code analyze
     ├─ [2] Explain Code     code explain
     ├─ [3] Document Code    code document
     └─ [4] Code Map         code map

3.6  Fact Checking
     ├─ [1] Contradictions   analyze contradictions
     ├─ [2] Evidence Links   analyze evidence
     └─ [3] Agent Verify     agent run (verify mode)

3.7  Stored Analyses
     ├─ [1] List All         analysis list
     ├─ [2] Search           analysis search
     ├─ [3] Show Details     analysis show
     ├─ [4] For Document     analysis for-document
     ├─ [5] Refresh          analysis refresh
     ├─ [6] Delete           analysis delete
     └─ [7] Statistics       analysis stats
```

### [4] CREATE & EXPORT

```
4.1  Study Tools
     ├─ [1] Flashcards       study flashcards
     ├─ [2] Quiz             study quiz
     ├─ [3] Study Notes      study notes
     ├─ [4] Glossary         study glossary
     └─ [5] Spaced Review    study review

4.2  Academic Writing
     ├─ [1] Draft            writing draft
     ├─ [2] Outline          writing outline
     ├─ [3] Paraphrase       writing paraphrase
     ├─ [4] Find Quotes      writing quote
     ├─ [5] Rewrite          writing rewrite
     ├─ [6] Simplify         writing simplify
     └─ [7] Thesis Help      writing thesis

4.3  Citations
     ├─ [1] Extract          citation extract
     ├─ [2] Format           citation format
     ├─ [3] Bibliography     citation bibliography
     ├─ [4] Validate         citation validate
     ├─ [5] Citation Graph   citation graph
     ├─ [6] Check in Doc     writing cite check (NEW!)
     ├─ [7] Format in Doc    writing cite format (NEW!)
     └─ [8] Insert           writing cite insert (NEW!)

4.4  Export
     ├─ [1] Markdown         export markdown
     ├─ [2] JSON             export json
     ├─ [3] PDF              export pdf
     ├─ [4] Outline Doc      export outline (NEW!)
     ├─ [5] RAG Context      export context
     ├─ [6] Study Package    export folder-export
     ├─ [7] Knowledge Graph  export knowledge-graph
     ├─ [8] Create Package   export pack (NEW!)
     ├─ [9] Import Package   export unpack (NEW!)
     └─ [0] Package Info     export info (NEW!)

4.5  Research Summary (NEW!)
     └─ [1] Multi-Agent Sum  research summarize
```

### [0] SYSTEM

```
0.1  Configuration
     ├─ [1] Show Config      config show
     ├─ [2] List All         config list
     ├─ [3] Set Value        config set
     ├─ [4] Validate         config validate
     └─ [5] Reset Defaults   config reset

0.2  LLM Settings
     ├─ [1] Select Model     (interactive selector)
     ├─ [2] Test Connection  agent test-llm
     └─ [3] API Key Setup    auth-wizard

0.3  Storage
     ├─ [1] Health Check     storage health
     ├─ [2] Statistics       storage stats
     └─ [3] Migrate          storage migrate

0.4  Index Management
     ├─ [1] Index Info       index info
     ├─ [2] List Indexes     index list
     ├─ [3] Rebuild          index rebuild
     └─ [4] Delete           index delete

0.5  Maintenance
     ├─ [1] Backup           maintenance backup
     ├─ [2] Restore          maintenance restore
     ├─ [3] Cleanup          maintenance cleanup
     └─ [4] Optimize         maintenance optimize

0.6  Monitoring
     ├─ [1] Diagnostics      monitor diagnostics
     ├─ [2] Health Check     monitor health
     ├─ [3] View Logs        monitor logs
     └─ [4] Metrics          monitor metrics

0.7  API Server
     ├─ [1] Start            api start
     ├─ [2] Stop             api stop
     ├─ [3] Status           api status
     └─ [4] Documentation    api docs

0.8  Workflow
     ├─ [1] Batch Ops        workflow batch
     ├─ [2] Pipeline         workflow pipeline
     └─ [3] Schedule         workflow schedule

0.9  System Info
     ├─ [1] Doctor           doctor
     ├─ [2] Status           status
     ├─ [3] Init Project     init
     └─ [4] Reset Project    reset
```

## Implementation Plan

### Phase 1: Quick Bar (High Impact, Low Effort)
1. Add quick bar display above main menu
2. Implement single-key shortcuts (Q, I, A, S, F, E, G, ?)
3. Bypass menu navigation for these keys

### Phase 2: Add Missing Features
1. Add Organization submenu (tags, bookmarks, annotations, mark)
2. Add Legal Research submenu (court, court-detail, etc.)
3. Add Security Research submenu (cve, cve-get)
4. Add missing Analysis commands (connections, timeline, relationships)
5. Add missing Export commands (pack, unpack, info, outline)
6. Add missing Citation commands (cite check, cite format, cite insert)

### Phase 3: Restructure Pillars
1. Consolidate 6 pillars into 4 + System
2. Update menu definitions
3. Update handler mappings
4. Test all paths

### Phase 4: Enhance Transform
1. Update `transform clean` to use TextCleanerRefiner
2. Add flags: --group-paragraphs, --clean-bullets, --clean-prefix-postfix
3. Add element classification preview

## Feature Coverage Summary

| Category | Features | Current Menu | After Redesign |
|----------|----------|--------------|----------------|
| Core | 8 | 6 | 8 |
| Agent | 4 | 1 | 4 |
| Organization | 9 | 0 | 9 |
| Analysis | 17 | 12 | 17 |
| Argument | 5 | 5 | 5 |
| Comprehension | 3 | 3 | 3 |
| Literary | 5 | 5 | 5 |
| Code | 4 | 4 | 4 |
| Discovery | 18 | 11 | 18 |
| Research | 3 | 2 | 3 |
| Study | 5 | 5 | 5 |
| Writing | 10 | 7 | 10 |
| Citation | 5 | 5 | 5 |
| Export | 10 | 6 | 10 |
| Transform | 5 | 5 | 5 |
| System | 23 | 20 | 23 |
| **TOTAL** | **144** | **113 (78%)** | **144 (100%)** |

## Metrics

| Metric | Current | After | Improvement |
|--------|---------|-------|-------------|
| Feature coverage | 78% | 100% | +22% |
| Avg clicks (common tasks) | 3.5 | 1.4 | -60% |
| Max click depth | 4 | 3 | -25% |
| Pillars | 6 | 5 | -17% cognitive load |
| Quick access items | 1 | 8 | +700% |
