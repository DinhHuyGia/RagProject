"""SQLite metadata storage for uploaded and indexed documents."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    filename: str
    stored_path: str
    page_count: int
    chunk_count: int
    status: str
    created_at: str
    chunk_ids: list[str]


class DocumentRegistry:
    """Persist document metadata separately from the vector database."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    document_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (document_id, chunk_id),
                    FOREIGN KEY (document_id)
                        REFERENCES documents(document_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
                    ON document_chunks(document_id);
                """
            )

    def add_document(
        self,
        *,
        document_id: str,
        filename: str,
        stored_path: str,
        page_count: int,
        chunk_ids: list[str],
        status: str = "indexed",
    ) -> DocumentRecord:
        created_at = datetime.now(UTC).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    filename,
                    stored_path,
                    page_count,
                    chunk_count,
                    status,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    filename,
                    stored_path,
                    page_count,
                    len(chunk_ids),
                    status,
                    created_at,
                ),
            )
            connection.executemany(
                """
                INSERT INTO document_chunks (
                    document_id,
                    chunk_id,
                    position
                )
                VALUES (?, ?, ?)
                """,
                (
                    (document_id, chunk_id, position)
                    for position, chunk_id in enumerate(chunk_ids)
                ),
            )

        return DocumentRecord(
            document_id=document_id,
            filename=filename,
            stored_path=stored_path,
            page_count=page_count,
            chunk_count=len(chunk_ids),
            status=status,
            created_at=created_at,
            chunk_ids=list(chunk_ids),
        )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    document_id,
                    filename,
                    stored_path,
                    page_count,
                    chunk_count,
                    status,
                    created_at
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

            if row is None:
                return None

            chunk_rows = connection.execute(
                """
                SELECT chunk_id
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY position
                """,
                (document_id,),
            ).fetchall()

        return self._record_from_rows(row, chunk_rows)

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    document_id,
                    filename,
                    stored_path,
                    page_count,
                    chunk_count,
                    status,
                    created_at
                FROM documents
                ORDER BY created_at DESC
                """
            ).fetchall()

            records = []
            for row in rows:
                chunk_rows = connection.execute(
                    """
                    SELECT chunk_id
                    FROM document_chunks
                    WHERE document_id = ?
                    ORDER BY position
                    """,
                    (row["document_id"],),
                ).fetchall()
                records.append(self._record_from_rows(row, chunk_rows))

        return records

    def delete_document(self, document_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
        return cursor.rowcount > 0

    @staticmethod
    def _record_from_rows(
        row: sqlite3.Row,
        chunk_rows: list[sqlite3.Row],
    ) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            stored_path=row["stored_path"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            status=row["status"],
            created_at=row["created_at"],
            chunk_ids=[chunk_row["chunk_id"] for chunk_row in chunk_rows],
        )
