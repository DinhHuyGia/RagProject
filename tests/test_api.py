from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_tutorial.adaptive_rag import (
    AdaptiveAnswer,
    AdaptiveRAG,
    SourceReference,
)
from rag_tutorial.api import (
    AppServices,
    AppSettings,
    create_app,
)
from rag_tutorial.document_registry import DocumentRegistry
from rag_tutorial.ingestion import IngestionResult


class FakeIngestionService:
    def __init__(self) -> None:
        self.indexed_chunks: set[str] = set()
        self.deleted_chunks: list[str] = []

    def ingest_file(
        self,
        file_path: Path,
        *,
        original_filename: str | None = None,
        document_id: str | None = None,
    ) -> IngestionResult:
        if file_path.read_bytes() == b"FAIL_INDEX":
            raise RuntimeError("Simulated indexing failure.")

        assert document_id is not None
        chunk_ids = [f"{document_id}:0"]
        self.indexed_chunks.update(chunk_ids)
        return IngestionResult(
            document_id=document_id,
            filename=original_filename or file_path.name,
            page_count=1,
            chunk_count=1,
            chunk_ids=chunk_ids,
        )

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        self.deleted_chunks.extend(chunk_ids)
        self.indexed_chunks.difference_update(chunk_ids)


class FakeRAG:
    def __init__(self, registry: DocumentRegistry) -> None:
        self.registry = registry
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        question: str,
        strategy: str = "auto",
        indexes: list[str] | None = None,
    ) -> AdaptiveAnswer:
        resolved_indexes = AdaptiveRAG.normalize_indexes(indexes)
        self.calls.append(
            {
                "question": question,
                "strategy": strategy,
                "indexes": resolved_indexes,
            }
        )

        document = self.registry.list_documents()[0]
        return AdaptiveAnswer(
            question=question,
            answer="The uploaded document contains the requested answer.",
            strategy="simple",
            indexes=resolved_indexes,
            routing_reason="Selected by the offline test double.",
            retrieval_queries=[question],
            sources=[
                SourceReference(
                    document_id=document.document_id,
                    filename=document.filename,
                    page_number=1,
                )
            ],
        )


class FailingDocumentRegistry(DocumentRegistry):
    def add_document(self, **kwargs: Any):
        raise RuntimeError("Simulated registry failure.")


@pytest.fixture
def api_client(
    workspace_temp_path: Path,
) -> Iterator[tuple[TestClient, AppSettings, dict[str, Any]]]:
    settings = AppSettings(
        upload_directory=workspace_temp_path / "uploads",
        document_database=workspace_temp_path / "documents.sqlite3",
        max_upload_size=32,
    )
    dependencies: dict[str, Any] = {}

    def service_factory(resolved_settings: AppSettings) -> AppServices:
        registry = DocumentRegistry(resolved_settings.document_database)
        ingestion = FakeIngestionService()
        rag = FakeRAG(registry)
        dependencies.update(
            {
                "registry": registry,
                "ingestion": ingestion,
                "rag": rag,
            }
        )
        return AppServices(
            vectorstore=None,
            ingestion=ingestion,
            document_registry=registry,
            rag=rag,
        )

    application = create_app(
        settings=settings,
        service_factory=service_factory,
    )
    with TestClient(application) as client:
        yield client, settings, dependencies


def test_upload_ask_and_delete_lifecycle(
    api_client: tuple[TestClient, AppSettings, dict[str, Any]],
) -> None:
    client, settings, dependencies = api_client

    upload_response = client.post(
        "/documents",
        files={
            "file": (
                "notes.txt",
                b"The launch date is Friday.",
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 201
    uploaded = upload_response.json()
    assert uploaded["filename"] == "notes.txt"
    assert uploaded["page_count"] == 1
    assert uploaded["chunk_count"] == 1
    assert uploaded["status"] == "indexed"

    document_id = uploaded["document_id"]
    stored_files = list(settings.upload_directory.iterdir())
    assert len(stored_files) == 1
    assert stored_files[0].name == f"{document_id}.txt"

    list_response = client.get("/documents")
    assert list_response.status_code == 200
    assert list_response.json() == [uploaded]

    ask_response = client.post(
        "/rag/ask",
        json={
            "question": "When is the launch date?",
            "indexes": ["chunks"],
        },
    )
    assert ask_response.status_code == 200
    answer = ask_response.json()
    assert answer["indexes"] == ["chunks"]
    assert answer["sources"] == [
        {
            "document_id": document_id,
            "filename": "notes.txt",
            "page_number": 1,
        }
    ]
    assert dependencies["rag"].calls[0]["indexes"] == ["chunks"]

    delete_response = client.delete(f"/documents/{document_id}")
    assert delete_response.status_code == 204
    assert client.get("/documents").json() == []
    assert list(settings.upload_directory.iterdir()) == []
    assert dependencies["ingestion"].indexed_chunks == set()
    assert dependencies["ingestion"].deleted_chunks == [f"{document_id}:0"]


def test_ask_requires_an_uploaded_document(
    api_client: tuple[TestClient, AppSettings, dict[str, Any]],
) -> None:
    client, _, dependencies = api_client

    response = client.post(
        "/rag/ask",
        json={"question": "What does the document say?"},
    )

    assert response.status_code == 409
    assert response.json()["detail"].startswith("Upload at least one")
    assert dependencies["rag"].calls == []


@pytest.mark.parametrize(
    ("filename", "content", "expected_status"),
    [
        ("program.exe", b"content", 415),
        ("empty.txt", b"", 400),
        ("large.txt", b"x" * 33, 413),
    ],
)
def test_upload_validation(
    api_client: tuple[TestClient, AppSettings, dict[str, Any]],
    filename: str,
    content: bytes,
    expected_status: int,
) -> None:
    client, settings, dependencies = api_client

    response = client.post(
        "/documents",
        files={"file": (filename, content, "application/octet-stream")},
    )

    assert response.status_code == expected_status
    assert dependencies["registry"].list_documents() == []
    assert not settings.upload_directory.exists()


def test_indexing_failure_removes_the_stored_file(
    api_client: tuple[TestClient, AppSettings, dict[str, Any]],
) -> None:
    client, settings, dependencies = api_client

    response = client.post(
        "/documents",
        files={"file": ("broken.txt", b"FAIL_INDEX", "text/plain")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "The document could not be indexed."
    assert dependencies["registry"].list_documents() == []
    assert list(settings.upload_directory.iterdir()) == []


def test_registry_failure_rolls_back_chunks_and_file(
    workspace_temp_path: Path,
) -> None:
    settings = AppSettings(
        upload_directory=workspace_temp_path / "uploads",
        document_database=workspace_temp_path / "documents.sqlite3",
    )
    registry = FailingDocumentRegistry(settings.document_database)
    ingestion = FakeIngestionService()

    def service_factory(_: AppSettings) -> AppServices:
        return AppServices(
            vectorstore=None,
            ingestion=ingestion,
            document_registry=registry,
            rag=FakeRAG(registry),
        )

    application = create_app(
        settings=settings,
        service_factory=service_factory,
    )
    with TestClient(application) as client:
        response = client.post(
            "/documents",
            files={"file": ("notes.txt", b"content", "text/plain")},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "The document was indexed but could not be registered."
    )
    assert ingestion.indexed_chunks == set()
    assert len(ingestion.deleted_chunks) == 1
    assert list(settings.upload_directory.iterdir()) == []


def test_invalid_index_is_rejected_by_the_api(
    api_client: tuple[TestClient, AppSettings, dict[str, Any]],
) -> None:
    client, _, _ = api_client

    response = client.post(
        "/rag/ask",
        json={
            "question": "What does the document say?",
            "indexes": ["summaries"],
        },
    )

    assert response.status_code == 422
