"""SQLite state manager for tracking processed music tracks."""

import sqlite3
from pathlib import Path

from resonate.models import ProcessingResult


class StateManager:
    """Manages SQLite database state for processed tracks."""

    def __init__(self, sqlite_path: str = "data/state.sqlite") -> None:
        """Initialize StateManager with database path and ensure DB schema exists."""
        self.sqlite_path = Path(sqlite_path)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a new SQLite database connection."""
        return sqlite3.connect(self.sqlite_path)

    def init_db(self) -> None:
        """Initialize SQLite database tables and parent directories."""
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_tracks (
                    rating_key TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    artist TEXT NOT NULL,
                    mapped_mood TEXT,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS track_lyrics (
                    artist TEXT NOT NULL,
                    title TEXT NOT NULL,
                    lyrics_text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (artist, title)
                )
                """
            )
            conn.commit()

    def is_track_processed(self, rating_key: str) -> bool:
        """Check if a track with the given rating key has been processed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM processed_tracks WHERE rating_key = ?",
                (rating_key,),
            )
            return cursor.fetchone() is not None

    def get_processed_keys(self) -> set[str]:
        """Retrieve set of all processed track rating keys."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT rating_key FROM processed_tracks")
            return {row[0] for row in cursor.fetchall()}

    def save_result(self, result: ProcessingResult) -> None:
        """Save a single track processing result to the database."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_tracks
                (rating_key, title, artist, mapped_mood, confidence, source, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime(?, 'unixepoch'))
                """,
                (
                    result.rating_key,
                    result.title,
                    result.artist,
                    result.mapped_mood,
                    result.confidence,
                    result.source,
                    result.timestamp,
                ),
            )
            conn.commit()

    def save_results_batch(self, results: list[ProcessingResult]) -> None:
        """Save multiple track processing results using a bulk transaction."""
        if not results:
            return
        data = [
            (
                r.rating_key,
                r.title,
                r.artist,
                r.mapped_mood,
                r.confidence,
                r.source,
                r.timestamp,
            )
            for r in results
        ]
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO processed_tracks
                (rating_key, title, artist, mapped_mood, confidence, source, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime(?, 'unixepoch'))
                """,
                data,
            )
            conn.commit()

    def get_stats(self) -> dict[str, int]:
        """Get summary statistics of processed tracks from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM processed_tracks")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM processed_tracks WHERE mapped_mood IS NOT NULL")
            mapped = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM processed_tracks WHERE mapped_mood IS NULL")
            unmapped = cursor.fetchone()[0]

            return {
                "total_processed": total,
                "mapped": mapped,
                "unmapped": unmapped,
            }

    def get_cached_lyrics(self, artist: str, title: str) -> dict[str, str] | None:
        """Retrieve cached lyrics for a given artist and track title."""
        if not artist or not title:
            return None
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT lyrics_text, source FROM track_lyrics "
                "WHERE LOWER(TRIM(artist)) = LOWER(TRIM(?)) "
                "AND LOWER(TRIM(title)) = LOWER(TRIM(?))",
                (artist, title),
            )
            row = cursor.fetchone()
            if row:
                return {"lyrics_text": row[0], "source": row[1]}
            return None

    def save_cached_lyrics(self, artist: str, title: str, lyrics_text: str, source: str) -> None:
        """Save lyrics to cache database."""
        if not artist or not title or not lyrics_text:
            return
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO track_lyrics (artist, title, lyrics_text, source, fetched_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (artist.strip(), title.strip(), lyrics_text, source),
            )
            conn.commit()

