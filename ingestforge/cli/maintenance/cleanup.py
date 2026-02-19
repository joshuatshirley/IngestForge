"""Cleanup command - Clean up temporary files and old data.

Cleans temporary files, caches, old data, and completed job records.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Optional
import typer

from ingestforge.cli.maintenance.base import MaintenanceCommand
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
DEFAULT_JOB_RETENTION_DAYS = 7
MAX_JOB_RETENTION_DAYS = 365


class CleanupCommand(MaintenanceCommand):
    """Clean up temporary files and old data."""

    def execute(
        self,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        temp: bool = True,
        cache: bool = True,
        logs: bool = False,
        jobs: bool = False,
        job_days: int = DEFAULT_JOB_RETENTION_DAYS,
        force: bool = False,
    ) -> int:
        """Clean up project files.

        Args:
            project: Project directory
            output: Output file for report
            temp: Clean temporary files
            cache: Clean cache files
            logs: Clean log files
            jobs: Clean old job records
            job_days: Job retention days (default 7)
            force: Skip confirmation

        Returns:
            0 on success, 1 on error
        """
        try:
            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)
            project_path = ctx["project_dir"]

            # Validate job_days (Rule #7)
            if job_days < 1 or job_days > MAX_JOB_RETENTION_DAYS:
                self.print_error(
                    f"Job retention must be 1-{MAX_JOB_RETENTION_DAYS} days"
                )
                return 1

            # Confirm cleanup
            if not force:
                if not self._confirm_cleanup(temp, cache, logs, jobs):
                    self.print_info("Cleanup cancelled")
                    return 0

            # Perform cleanup
            results = self._perform_cleanup(
                project_path, temp, cache, logs, jobs, job_days
            )

            # Display results
            summary = self.create_maintenance_summary(results, "Cleanup")
            self.console.print(summary)

            # Save report if requested
            if output:
                self.save_maintenance_report(output, results, "Cleanup")

            return 0

        except Exception as e:
            return self.handle_error(e, "Cleanup failed")

    def _confirm_cleanup(self, temp: bool, cache: bool, logs: bool, jobs: bool) -> bool:
        """Confirm cleanup operation.

        Args:
            temp: Clean temp files
            cache: Clean cache
            logs: Clean logs
            jobs: Clean old job records

        Returns:
            True if confirmed
        """
        items = []
        if temp:
            items.append("temporary files")
        if cache:
            items.append("cache")
        if logs:
            items.append("logs")
        if jobs:
            items.append("old job records")

        msg = f"Clean up {', '.join(items)}?"
        return typer.confirm(msg)

    def _perform_cleanup(
        self,
        project: Path,
        temp: bool,
        cache: bool,
        logs: bool,
        jobs: bool,
        job_days: int,
    ) -> dict[str, Any]:
        """Perform cleanup operations.

        Args:
            project: Project directory
            temp: Clean temp files
            cache: Clean cache
            logs: Clean logs
            jobs: Clean old job records
            job_days: Job retention days

        Returns:
            Cleanup results
        """
        items_cleaned = 0
        space_freed = 0
        errors: list[str] = []
        jobs_cleaned = 0

        # Clean temp files
        if temp:
            temp_results = self._clean_temp_files(project)
            items_cleaned += temp_results["count"]
            space_freed += temp_results["size"]
            errors.extend(temp_results["errors"])

        # Clean cache
        if cache:
            cache_results = self._clean_cache(project)
            items_cleaned += cache_results["count"]
            space_freed += cache_results["size"]
            errors.extend(cache_results["errors"])

        # Clean logs
        if logs:
            log_results = self._clean_logs(project)
            items_cleaned += log_results["count"]
            space_freed += log_results["size"]
            errors.extend(log_results["errors"])

        # Clean old job records (DEAD-D07)
        if jobs:
            job_results = self._clean_jobs(project, job_days)
            jobs_cleaned = job_results["count"]
            errors.extend(job_results["errors"])

        result = {
            "items_cleaned": items_cleaned,
            "space_freed": self.format_size(space_freed),
            "errors": errors,
        }

        if jobs:
            result["jobs_cleaned"] = jobs_cleaned

        return result

    def _clean_temp_files(self, project: Path) -> dict[str, Any]:
        """Clean temporary files.

        Args:
            project: Project directory

        Returns:
            Cleanup results
        """
        temp_dir = project / ".ingestforge" / "temp"

        if not temp_dir.exists():
            return {"count": 0, "size": 0, "errors": []}

        size = self.calculate_directory_size(temp_dir)
        count = self.delete_directory_contents(temp_dir)

        return {"count": count, "size": size, "errors": []}

    def _clean_cache(self, project: Path) -> dict[str, Any]:
        """Clean cache files.

        Args:
            project: Project directory

        Returns:
            Cleanup results
        """
        cache_dir = project / ".ingestforge" / "cache"

        if not cache_dir.exists():
            return {"count": 0, "size": 0, "errors": []}

        size = self.calculate_directory_size(cache_dir)
        count = self.delete_directory_contents(cache_dir)

        return {"count": count, "size": size, "errors": []}

    def _clean_logs(self, project: Path) -> dict[str, Any]:
        """Clean log files.

        Args:
            project: Project directory

        Returns:
            Cleanup results
        """
        log_files = self.get_log_files(project)

        if not log_files:
            return {"count": 0, "size": 0, "errors": []}

        size = sum(f.stat().st_size for f in log_files)
        count = 0

        for log_file in log_files:
            try:
                log_file.unlink()
                count += 1
            except Exception as e:
                logger.debug(f"Failed to delete log file {log_file}: {e}")

        return {"count": count, "size": size, "errors": []}

    def _clean_jobs(self, project: Path, retention_days: int) -> dict[str, Any]:
        """Clean old job records from queue database.

        DEAD-D07: Wire SQLiteJobQueue cleanup.

        Args:
            project: Project directory
            retention_days: Days to retain completed jobs

        Returns:
            Cleanup results
        """
        import asyncio
        from ingestforge.core.jobs import create_job_queue

        db_path = project / ".data" / "jobs.db"

        if not db_path.exists():
            return {"count": 0, "errors": []}

        try:
            queue = create_job_queue(db_path)
            older_than = timedelta(days=retention_days)
            count = asyncio.run(queue.cleanup(older_than))
            return {"count": count, "errors": []}

        except Exception as e:
            logger.warning(f"Failed to clean job queue: {e}")
            return {"count": 0, "errors": [str(e)]}


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for report"
    ),
    temp: bool = typer.Option(True, "--temp/--no-temp", help="Clean temporary files"),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Clean cache files"),
    logs: bool = typer.Option(False, "--logs", help="Clean log files"),
    jobs: bool = typer.Option(False, "--jobs", help="Clean old job queue records"),
    job_days: int = typer.Option(
        DEFAULT_JOB_RETENTION_DAYS,
        "--job-days",
        help=f"Days to retain job records (default: {DEFAULT_JOB_RETENTION_DAYS})",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Clean up temporary files and old data.

    Removes temporary files, cache, and optionally logs and job records
    to free up disk space and improve performance.

    Default: Cleans temp and cache (preserves logs and jobs)

    Examples:
        # Clean temp and cache (with confirmation)
        ingestforge maintenance cleanup

        # Clean without confirmation
        ingestforge maintenance cleanup --force

        # Clean everything including logs and jobs
        ingestforge maintenance cleanup --logs --jobs --force

        # Clean only temp files
        ingestforge maintenance cleanup --no-cache

        # Clean old job records (older than 7 days)
        ingestforge maintenance cleanup --jobs --force

        # Clean jobs older than 30 days
        ingestforge maintenance cleanup --jobs --job-days 30 --force

        # Clean with report
        ingestforge maintenance cleanup -o cleanup.md

        # Specific project
        ingestforge maintenance cleanup -p /path/to/project
    """
    cmd = CleanupCommand()
    exit_code = cmd.execute(project, output, temp, cache, logs, jobs, job_days, force)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
