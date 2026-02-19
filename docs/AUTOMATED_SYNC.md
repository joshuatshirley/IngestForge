# IngestForge Automated Sync

Automated synchronization of your codebase and ADO work items with IngestForge.

## Quick Start

### Manual Run

```bash
# Run the automated sync (runs ado_sync.py + ingests changes)
.\Auto-Sync.bat

# Or with PowerShell directly
.\scripts\automated-sync.ps1

# Options
.\scripts\automated-sync.ps1 -Force        # Full re-sync
.\scripts\automated-sync.ps1 -SkipADOSync  # Only IngestForge sync
.\scripts\automated-sync.ps1 -Verbose      # Detailed output
```

### Schedule Automated Runs

```powershell
# Run as Administrator

# Daily at 2 AM
.\scripts\Setup-ScheduledTask.ps1 -Schedule Daily -Time "02:00"

# Weekly on Sunday at 3 AM
.\scripts\Setup-ScheduledTask.ps1 -Schedule Weekly -Time "03:00" -DayOfWeek Sunday

# Every 4 hours
.\scripts\Setup-ScheduledTask.ps1 -Schedule Hourly -Interval 4

# Remove the scheduled task
.\scripts\Setup-ScheduledTask.ps1 -Remove
```

## Architecture

The automation integrates with your existing `ado_sync.py` script:

```
┌─────────────────────────────────────────────────────────────┐
│                    Automated Sync Flow                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: ado_sync.py (C:\...\AIE\ado_sync.py)               │
│     ├── Git clone/pull AIE repository → repo\               │
│     ├── Git clone/pull AIE Wiki → wiki\                     │
│     ├── Export ADO work items → ado_export\                 │
│     │   ├── Features                                        │
│     │   ├── User Stories                                    │
│     │   ├── Bugs                                            │
│     │   └── Tasks                                           │
│     └── Build _hierarchy.json index                         │
│                                                              │
│  Step 2: Sync Salesforce Code                               │
│     └── ingestforge sync repo\src -p *.cls -p *.trigger     │
│                                                              │
│  Step 3: Sync Wiki                                          │
│     └── ingestforge sync wiki\ -p *.md                      │
│                                                              │
│  Step 4: Sync ADO Work Items                                │
│     └── ingestforge sync ado_export\ -p *.md                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## How Incremental Sync Works

The sync only processes files that have changed:

| File Status | Action |
|-------------|--------|
| **New** | Ingest (create chunks + embeddings) |
| **Changed** | Delete old chunks, re-ingest |
| **Deleted** | Remove chunks from database |
| **Unchanged** | Skip (fast!) |

State is tracked in `.data/sync_state.json`.

## Prerequisites

### Python Dependencies for ado_sync.py

```bash
pip install azure-devops msrest
```

### ADO PAT Token

The `ado_sync.py` uses a Personal Access Token. Set it as environment variable:

```powershell
$env:ADO_PAT = "your-pat-token-here"
```

## File Locations

| Item | Path |
|------|------|
| ADO Sync Script | `C:\Users\joshu\..Project\Projects\AIE\ado_sync.py` |
| Git Repository | `C:\Users\joshu\..Project\Projects\AIE\repo\` |
| Wiki | `C:\Users\joshu\..Project\Projects\AIE\wiki\` |
| ADO Export | `C:\Users\joshu\..Project\Projects\AIE\ado_export\` |
| IngestForge | `C:\Users\joshu\..Project\Projects\IngestForge\` |
| Sync State | `.data\sync_state.json` |
| Sync Log | `.data\automated-sync.log` |

## Logs

View recent sync activity:

```powershell
# Last 50 lines
Get-Content .data\automated-sync.log -Tail 50

# Follow live
Get-Content .data\automated-sync.log -Wait -Tail 20
```

## Troubleshooting

### ado_sync.py Fails

```bash
# Check Python dependencies
pip install azure-devops msrest

# Test manually
cd C:\Users\joshu\..Project\Projects\AIE
python ado_sync.py
```

### Scheduled Task Not Running

```powershell
# Check task status
Get-ScheduledTask -TaskName "IngestForge-AutoSync"

# View last run info
Get-ScheduledTaskInfo -TaskName "IngestForge-AutoSync"

# Run manually to test
Start-ScheduledTask -TaskName "IngestForge-AutoSync"
```

### Reset Sync State

If sync gets confused, reset and start fresh:

```bash
ingestforge sync-reset --yes
.\Auto-Sync.bat -Force
```

## Commands After Sync

Once synced, use these commands to query your data:

```bash
# Search code
ingestforge code-search "TriggerDispatcher"
ingestforge code-search "SOQL" --type Selector

# Search work items
ingestforge story-search "enlistment workflow"
ingestforge story-search "bug" --type Bug

# Cross-reference
ingestforge trace-story 29232       # Find code for a story
ingestforge reverse-story AccountsTriggerHandler  # Find stories for code
ingestforge story-gaps --package aie-enlistment   # Find undocumented code
```
