"""Unit tests for SRS-004 due_check module.

Tests the lightweight due cards notification system.
"""
import tempfile
import time
import gc
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta


from ingestforge.study.due_check import (
    count_due_cards,
    get_due_notification,
    show_due_notification,
)


class TestCountDueCards:
    """Test count_due_cards function."""

    def test_no_database(self):
        """Test behavior when database doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            due_count, total_count = count_due_cards(project_dir)

            assert due_count == 0
            assert total_count == 0

    def test_empty_database(self):
        """Test behavior with empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            db_path = project_dir / ".data" / "reviews.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create empty database with schema
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE cards (
                    card_id TEXT PRIMARY KEY,
                    front TEXT,
                    back TEXT,
                    topic TEXT,
                    next_review TEXT
                )
            """
            )
            conn.commit()
            cursor.close()
            conn.close()

            due_count, total_count = count_due_cards(project_dir)

            assert due_count == 0
            assert total_count == 0

            # Allow Windows to release file locks
            gc.collect()
            time.sleep(0.1)

    def test_due_cards(self):
        """Test counting due cards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            db_path = project_dir / ".data" / "reviews.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create database with cards
            now = datetime.now()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE cards (
                    card_id TEXT PRIMARY KEY,
                    front TEXT,
                    back TEXT,
                    topic TEXT,
                    next_review TEXT
                )
            """
            )

            # Add 3 due cards (past dates)
            for i in range(3):
                cursor.execute(
                    "INSERT INTO cards VALUES (?, ?, ?, ?, ?)",
                    (
                        f"due_{i}",
                        "Q",
                        "A",
                        "topic",
                        (now - timedelta(days=1)).isoformat(),
                    ),
                )

            # Add 2 future cards
            for i in range(2):
                cursor.execute(
                    "INSERT INTO cards VALUES (?, ?, ?, ?, ?)",
                    (
                        f"future_{i}",
                        "Q",
                        "A",
                        "topic",
                        (now + timedelta(days=1)).isoformat(),
                    ),
                )

            # Add 1 card with NULL next_review (should be counted as due)
            cursor.execute(
                "INSERT INTO cards VALUES (?, ?, ?, ?, NULL)",
                ("null_review", "Q", "A", "topic"),
            )

            conn.commit()
            cursor.close()
            conn.close()

            due_count, total_count = count_due_cards(project_dir)

            assert due_count == 4  # 3 past + 1 NULL
            assert total_count == 6

            # Allow Windows to release file locks
            gc.collect()
            time.sleep(0.1)


class TestGetDueNotification:
    """Test get_due_notification function."""

    def test_no_due_cards(self):
        """Test notification when no cards are due."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            notification = get_due_notification(project_dir)

            assert notification is None

    def test_singular_notification(self):
        """Test notification for 1 due card."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            db_path = project_dir / ".data" / "reviews.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create database with 1 due card
            now = datetime.now()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE cards (
                    card_id TEXT PRIMARY KEY,
                    front TEXT,
                    back TEXT,
                    topic TEXT,
                    next_review TEXT
                )
            """
            )
            cursor.execute(
                "INSERT INTO cards VALUES (?, ?, ?, ?, ?)",
                ("card_1", "Q", "A", "topic", (now - timedelta(days=1)).isoformat()),
            )
            conn.commit()
            cursor.close()
            conn.close()

            notification = get_due_notification(project_dir)

            assert notification is not None
            assert "1 card is due" in notification
            assert "ingestforge study review due" in notification

            # Allow Windows to release file locks
            gc.collect()
            time.sleep(0.1)

    def test_plural_notification(self):
        """Test notification for multiple due cards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            db_path = project_dir / ".data" / "reviews.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create database with 5 due cards
            now = datetime.now()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE cards (
                    card_id TEXT PRIMARY KEY,
                    front TEXT,
                    back TEXT,
                    topic TEXT,
                    next_review TEXT
                )
            """
            )
            for i in range(5):
                cursor.execute(
                    "INSERT INTO cards VALUES (?, ?, ?, ?, ?)",
                    (
                        f"card_{i}",
                        "Q",
                        "A",
                        "topic",
                        (now - timedelta(days=1)).isoformat(),
                    ),
                )
            conn.commit()
            cursor.close()
            conn.close()

            notification = get_due_notification(project_dir)

            assert notification is not None
            assert "5 cards are due" in notification

            # Allow Windows to release file locks
            gc.collect()
            time.sleep(0.1)


class TestShowDueNotification:
    """Test show_due_notification function."""

    def test_quiet_mode(self, capsys):
        """Test that quiet mode suppresses output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            show_due_notification(project_dir, quiet=True)

            captured = capsys.readouterr()
            assert captured.out == ""

    def test_no_notification_when_no_cards(self, capsys):
        """Test no output when no cards are due."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            show_due_notification(project_dir, quiet=False)

            captured = capsys.readouterr()
            assert captured.out == ""
