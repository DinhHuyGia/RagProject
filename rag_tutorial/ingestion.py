from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from rag_tutorial.shared import create_text_splitter

LoaderFactory = Callable[[str], object]


LOADERS: dict[str, LoaderFactory] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".md": lambda path: TextLoader(path, encoding="utf-8"),
}

@dataclass
class IngestionResult:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    chunk_ids: list[str]


class UnsupportedDocumentError(ValueError):
    pass

class DocumentIngestionService:
    def __init__(
        self,
        vectorstore: Chroma,
        *,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):
        self.vectorstore = vectorstore
        self.splitter = create_text_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def ingest_file(
        self,
        file_path: Path,
        *,
        original_filename: str | None = None,
        document_id: str | None = None,
    ) -> IngestionResult:
        extension = file_path.suffix.lower()

        if extension not in LOADERS:
            raise UnsupportedDocumentError(
                f"Unsupported document type: {extension}"
            )

        document_id = document_id or str(uuid4())
        filename = original_filename or file_path.name

        loader = LOADERS[extension](str(file_path))
        documents = loader.load()
        documents = [
            document
            for document in documents
            if document.page_content.strip()
        ]

        if not documents:
            raise ValueError("The document contains no extractable text.")

        self._normalize_metadata(
            documents=documents,
            document_id=document_id,
            filename=filename,
        )

        chunks = self.splitter.split_documents(documents)

        chunk_ids: list[str] = []

        for position, chunk in enumerate(chunks):
            chunk_id = f"{document_id}:{position}"

            chunk.metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_position": position,
                }
            )

            chunk_ids.append(chunk_id)

        self.vectorstore.add_documents(
            documents=chunks,
            ids=chunk_ids,
        )

        return IngestionResult(
            document_id=document_id,
            filename=filename,
            page_count=len(documents),
            chunk_count=len(chunks),
            chunk_ids=chunk_ids,
        )

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete indexed chunks for one uploaded document."""
        if chunk_ids:
            self.vectorstore.delete(ids=chunk_ids)

    @staticmethod
    def _normalize_metadata(
        documents: list[Document],
        document_id: str,
        filename: str,
    ) -> None:
        for document in documents:
            page_index = document.metadata.get("page")

            document.metadata.update(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "source": filename,
                }
            )

            if isinstance(page_index, int):
                document.metadata["page_number"] = page_index + 1
