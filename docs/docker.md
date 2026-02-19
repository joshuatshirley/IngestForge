# Docker Deployment Guide

This guide covers running IngestForge with Docker, including data management, backup/restore procedures, and production deployment.

## Quick Start

```bash
# Start the API server
docker-compose up -d

# Run CLI commands
docker-compose run --rm cli ingest /app/documents/myfile.pdf
docker-compose run --rm cli query "What is this about?"
docker-compose run --rm cli status

# Stop services
docker-compose down
```

## Services

### API Server (`ingestforge`)

The main service runs the FastAPI server on port 8000.

```bash
# Start in foreground (see logs)
docker-compose up ingestforge

# Start in background
docker-compose up -d ingestforge

# View logs
docker-compose logs -f ingestforge

# Restart
docker-compose restart ingestforge
```

### CLI Runner (`cli`)

Run one-off CLI commands without starting the API server.

```bash
# Initialize a new project
docker-compose run --rm cli init --name myproject

# Ingest a document
docker-compose run --rm cli ingest /app/documents/research.pdf

# Query the corpus
docker-compose run --rm cli query "summarize the main findings"

# Check status
docker-compose run --rm cli status

# Run doctor diagnostics
docker-compose run --rm cli doctor
```

### Development (`dev`)

Hot-reload development environment with source code mounted.

```bash
# Start development server
docker-compose --profile dev up dev

# Run tests inside container
docker-compose --profile dev run --rm dev pytest tests/ -v
```

## Data Persistence

IngestForge uses two Docker volumes to persist data:

| Volume | Path | Description |
|--------|------|-------------|
| `ingestforge-data` | `/app/.data` | Processed chunks, embeddings, ChromaDB indexes |
| `ingestforge-pending` | `/app/.ingest/pending` | Documents waiting to be processed |

### Viewing Volume Contents

```bash
# List data files
docker run --rm -v ingestforge-data:/data alpine ls -la /data

# List pending files
docker run --rm -v ingestforge-pending:/pending alpine ls -la /pending

# Check volume sizes
docker system df -v | grep ingestforge
```

### Adding Documents

Mount a local folder to import documents:

```bash
# Method 1: Mount documents folder (configured in docker-compose.yml)
# Place files in ./documents/ folder, then:
docker-compose run --rm cli ingest /app/documents/myfile.pdf

# Method 2: Copy file directly to pending volume
docker cp myfile.pdf $(docker-compose ps -q cli):/app/.ingest/pending/

# Method 3: One-off mount
docker run --rm \
  -v ingestforge-data:/app/.data \
  -v ingestforge-pending:/app/.ingest/pending \
  -v /path/to/local/file.pdf:/input/file.pdf:ro \
  ingestforge ingestforge ingest /input/file.pdf
```

## Backup & Restore

### Creating Backups

```bash
# Create a timestamped backup
docker-compose --profile backup run --rm backup

# Backups are saved to ./backups/ directory
ls -la backups/
# ingestforge-backup-20240115_143022.tar.gz
```

### Automated Backup Script

Create `scripts/backup.sh`:

```bash
#!/bin/bash
# Run this from project root
cd "$(dirname "$0")/.."

# Create backup
docker-compose --profile backup run --rm backup

# Keep only last 7 backups
cd backups && ls -t *.tar.gz | tail -n +8 | xargs -r rm
```

Add to crontab for daily backups:
```bash
# Run at 2 AM daily
0 2 * * * /path/to/ingestforge/scripts/backup.sh
```

### Restoring from Backup

**Warning**: Restore overwrites all existing data.

```bash
# Stop services first
docker-compose down

# List available backups
docker-compose --profile restore run --rm restore

# Restore specific backup
docker-compose --profile restore run --rm restore /backups/ingestforge-backup-20240115_143022.tar.gz

# Start services
docker-compose up -d
```

### Manual Backup (Alternative)

```bash
# Export volumes to tar
docker run --rm \
  -v ingestforge-data:/data:ro \
  -v ingestforge-pending:/pending:ro \
  -v $(pwd)/backups:/backup \
  alpine tar -czf /backup/manual-backup.tar.gz -C / data pending

# Import from tar
docker run --rm \
  -v ingestforge-data:/data \
  -v ingestforge-pending:/pending \
  -v $(pwd)/backups:/backup:ro \
  alpine sh -c "rm -rf /data/* /pending/* && tar -xzf /backup/manual-backup.tar.gz -C /"
```

### Migrating Data Between Hosts

```bash
# On source host: export volumes
docker run --rm \
  -v ingestforge-data:/data:ro \
  -v ingestforge-pending:/pending:ro \
  ubuntu tar -czf - -C / data pending > ingestforge-export.tar.gz

# Transfer file to target host
scp ingestforge-export.tar.gz user@newhost:/path/to/

# On target host: import volumes
cat ingestforge-export.tar.gz | docker run --rm -i \
  -v ingestforge-data:/data \
  -v ingestforge-pending:/pending \
  ubuntu tar -xzf - -C /
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `LOG_LEVEL` | Logging verbosity | INFO |
| `INGESTFORGE_LLM_PROVIDER` | LLM to use | gemini |

## Production Deployment

### Using docker-compose.prod.yml

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  ingestforge:
    restart: always
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 1G
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"
```

Deploy:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Reverse Proxy with Nginx

```nginx
upstream ingestforge {
    server localhost:8000;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/ssl/certs/example.com.crt;
    ssl_certificate_key /etc/ssl/private/example.com.key;

    location / {
        proxy_pass http://ingestforge;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Health Monitoring

```bash
# Check health endpoint
curl http://localhost:8000/health

# Docker health status
docker inspect --format='{{.State.Health.Status}}' ingestforge-api
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs ingestforge

# Check if port is in use
lsof -i :8000

# Rebuild image
docker-compose build --no-cache ingestforge
```

### Out of disk space

```bash
# Check Docker disk usage
docker system df

# Clean up unused resources
docker system prune -a --volumes
```

### Permission errors

```bash
# Fix volume permissions
docker run --rm -v ingestforge-data:/data alpine chown -R 1000:1000 /data
docker run --rm -v ingestforge-pending:/pending alpine chown -R 1000:1000 /pending
```

### Reset everything

```bash
# Stop and remove containers, volumes, and images
docker-compose down -v --rmi all

# Start fresh
docker-compose up --build
```
