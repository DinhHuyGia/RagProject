import re
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import chromadb
import pytest
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from reportlab.pdfgen import canvas

from rag_tutorial.ingestion import DocumentIngestionService


class DeterministicKeywordEmbeddings(Embeddings):
    """Small offline embeddings with predictable exact-keyword similarity."""

    vocabulary = (
        "orchid",
        "launch",
        "friday",
        "harbor",
        "budget",
        "monday",
        "amber",
        "inventory",
        "cobalt",
        "authorization",
    )

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        return [
            float(keyword in tokens)
            for keyword in cls.vocabulary
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def real_vectorstore() -> Iterator[Chroma]:
    client = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False),
    )
    vectorstore = Chroma(
        client=client,
        collection_name=f"ingestion_test_{uuid4().hex}",
        embedding_function=DeterministicKeywordEmbeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )

    try:
        yield vectorstore
    finally:
        vectorstore.delete_collection()


def test_real_txt_ingestion_retrieval_and_deletion(
    workspace_temp_path: Path,
    real_vectorstore: Chroma,
) -> None:
    target_path = workspace_temp_path / "orchid.txt"
    target_path.write_text(
        (
            "Project Orchid launch ceremony happens Friday at noon.\n\n"
            "The operations team will meet beside the east entrance and "
            "complete the release checklist before guests arrive."
        ),
        encoding="utf-8",
    )
    distractor_path = workspace_temp_path / "harbor.txt"
    distractor_path.write_text(
        "Project Harbor budget review happens Monday in the finance room.",
        encoding="utf-8",
    )

    ingestion = DocumentIngestionService(
        real_vectorstore,
        chunk_size=15,
        chunk_overlap=0,
    )
    target = ingestion.ingest_file(
        target_path,
        original_filename="orchid-notes.txt",
        document_id="orchid-document",
    )
    distractor = ingestion.ingest_file(
        distractor_path,
        original_filename="harbor-notes.txt",
        document_id="harbor-document",
    )

    assert target.document_id == "orchid-document"
    assert target.filename == "orchid-notes.txt"
    assert target.page_count == 1
    assert target.chunk_count >= 2
    assert target.chunk_ids == [
        f"orchid-document:{position}"
        for position in range(target.chunk_count)
    ]

    stored = real_vectorstore.get(ids=target.chunk_ids)
    assert set(stored["ids"]) == set(target.chunk_ids)
    assert len(stored["metadatas"]) == target.chunk_count

    metadata_by_position = sorted(
        stored["metadatas"],
        key=lambda metadata: metadata["chunk_position"],
    )
    for position, metadata in enumerate(metadata_by_position):
        assert metadata["document_id"] == "orchid-document"
        assert metadata["filename"] == "orchid-notes.txt"
        assert metadata["source"] == "orchid-notes.txt"
        assert metadata["chunk_id"] == f"orchid-document:{position}"
        assert metadata["chunk_position"] == position
        assert "page_number" not in metadata

    retriever = real_vectorstore.as_retriever(search_kwargs={"k": 3})
    matches = retriever.invoke("When is the Project Orchid launch?")

    assert matches
    assert matches[0].metadata["document_id"] == "orchid-document"
    assert "Orchid" in matches[0].page_content

    ingestion.delete_chunks(target.chunk_ids)

    assert real_vectorstore.get(
        where={"document_id": "orchid-document"}
    )["ids"] == []

    remaining_matches = retriever.invoke("When is the Project Orchid launch?")
    assert remaining_matches
    assert all(
        document.metadata["document_id"] != "orchid-document"
        for document in remaining_matches
    )
    assert {
        document.metadata["document_id"]
        for document in remaining_matches
    } == {distractor.document_id}


def test_real_pdf_ingestion_preserves_page_metadata(
    workspace_temp_path: Path,
    real_vectorstore: Chroma,
) -> None:
    pdf_path = workspace_temp_path / "operations.pdf"
    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(
        72,
        720,
        "The Amber inventory report is stored in the north archive.",
    )
    pdf.showPage()
    pdf.drawString(
        72,
        720,
        "The Cobalt authorization code is recorded on the second page.",
    )
    pdf.save()

    ingestion = DocumentIngestionService(
        real_vectorstore,
        chunk_size=200,
        chunk_overlap=0,
    )
    result = ingestion.ingest_file(
        pdf_path,
        original_filename="operations-manual.pdf",
        document_id="operations-document",
    )

    assert result.page_count == 2
    assert result.chunk_count == 2

    stored = real_vectorstore.get(ids=result.chunk_ids)
    metadata_by_page = sorted(
        stored["metadatas"],
        key=lambda metadata: metadata["page_number"],
    )
    assert [
        metadata["page_number"]
        for metadata in metadata_by_page
    ] == [1, 2]
    assert all(
        metadata["document_id"] == "operations-document"
        for metadata in metadata_by_page
    )
    assert all(
        metadata["filename"] == "operations-manual.pdf"
        for metadata in metadata_by_page
    )

    retriever = real_vectorstore.as_retriever(search_kwargs={"k": 1})
    matches = retriever.invoke("Where is the Cobalt authorization code?")

    assert len(matches) == 1
    assert matches[0].metadata["page_number"] == 2
    assert "Cobalt authorization code" in matches[0].page_content
