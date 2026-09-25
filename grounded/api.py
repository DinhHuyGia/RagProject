from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import (
    APIRouter,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from pydantic import BaseModel, Field

from grounded.adaptive_rag import (
    AdaptiveAnswer,
    AdaptiveRAG,
    IndexName,
    RequestedStrategy,
)
from grounded.document_registry import DocumentRecord, DocumentRegistry
from grounded.ingestion import DocumentIngestionService
from grounded.shared import create_document_vectorstore

DEFAULT_UPLOAD_DIRECTORY = Path("data/uploads")
DEFAULT_DOCUMENT_DATABASE = Path("data/documents.sqlite3")
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024


@dataclass(frozen=True)
class AppSettings:
    upload_directory: Path = DEFAULT_UPLOAD_DIRECTORY
    document_database: Path = DEFAULT_DOCUMENT_DATABASE
    max_upload_size: int = MAX_UPLOAD_SIZE


@dataclass
class AppServices:
    vectorstore: Any
    ingestion: Any
    document_registry: Any
    rag: Any


ServiceFactory = Callable[[AppSettings], AppServices]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    strategy: RequestedStrategy = "auto"
    indexes: list[IndexName] | None = Field(
        default=None,
        max_length=1,
    )


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    status: str
    created_at: str

    @classmethod
    def from_record(cls, record: DocumentRecord) -> "DocumentResponse":
        return cls(
            document_id=record.document_id,
            filename=record.filename,
            page_count=record.page_count,
            chunk_count=record.chunk_count,
            status=record.status,
            created_at=record.created_at,
        )


def build_services(settings: AppSettings) -> AppServices:
    vectorstore = create_document_vectorstore()
    ingestion = DocumentIngestionService(vectorstore)
    document_registry = DocumentRegistry(settings.document_database)
    rag = AdaptiveRAG(
        chunk_retriever=vectorstore.as_retriever(
            search_kwargs={"k": 8},
        )
    )
    return AppServices(
        vectorstore=vectorstore,
        ingestion=ingestion,
        document_registry=document_registry,
        rag=rag,
    )


router = APIRouter()


@router.post("/rag/ask", response_model=AdaptiveAnswer)
def ask_rag(body: AskRequest, request: Request) -> AdaptiveAnswer:
    registry: DocumentRegistry = request.app.state.document_registry
    if not registry.list_documents():
        raise HTTPException(
            status_code=409,
            detail="Upload at least one document before asking a question.",
        )

    rag: AdaptiveRAG = request.app.state.rag

    try:
        return rag.invoke(
            question=body.question,
            strategy=body.strategy,
            indexes=body.indexes,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail="The RAG pipeline is unavailable.",
        ) from error


@router.post("/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> DocumentResponse:
    settings: AppSettings = request.app.state.settings
    original_filename = file.filename or "document"
    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {extension}",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=(
                "The file exceeds the "
                f"{settings.max_upload_size // (1024 * 1024)} MB limit."
            ),
        )

    document_id = str(uuid4())

    settings.upload_directory.mkdir(parents=True, exist_ok=True)
    stored_path = settings.upload_directory / f"{document_id}{extension}"
    stored_path.write_bytes(content)

    try:
        result = request.app.state.ingestion.ingest_file(
            stored_path,
            original_filename=original_filename,
            document_id=document_id,
        )
    except ValueError as error:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception as error:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="The document could not be indexed.",
        ) from error

    try:
        record = request.app.state.document_registry.add_document(
            document_id=result.document_id,
            filename=result.filename,
            stored_path=str(stored_path.resolve()),
            page_count=result.page_count,
            chunk_ids=result.chunk_ids,
        )
    except Exception as error:
        with suppress(Exception):
            request.app.state.ingestion.delete_chunks(result.chunk_ids)
        with suppress(OSError):
            stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="The document was indexed but could not be registered.",
        ) from error

    return DocumentResponse.from_record(record)


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(request: Request) -> list[DocumentResponse]:
    records = request.app.state.document_registry.list_documents()
    return [DocumentResponse.from_record(record) for record in records]


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, request: Request) -> Response:
    registry: DocumentRegistry = request.app.state.document_registry
    record = registry.get_document(document_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    settings: AppSettings = request.app.state.settings
    upload_root = settings.upload_directory.resolve()
    stored_path = Path(record.stored_path).resolve()

    try:
        stored_path.relative_to(upload_root)
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail="The stored document path is invalid.",
        ) from error

    try:
        request.app.state.ingestion.delete_chunks(record.chunk_ids)
        stored_path.unlink(missing_ok=True)
        registry.delete_document(document_id)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="The document could not be completely deleted.",
        ) from error

    return Response(status_code=204)


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    return {
        "status": "ready",
        "rag_loaded": request.app.state.rag is not None,
    }


def create_app(
    *,
    settings: AppSettings | None = None,
    service_factory: ServiceFactory = build_services,
) -> FastAPI:
    resolved_settings = settings or AppSettings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        services = service_factory(resolved_settings)
        application.state.settings = resolved_settings
        application.state.vectorstore = services.vectorstore
        application.state.ingestion = services.ingestion
        application.state.document_registry = services.document_registry
        application.state.rag = services.rag

        yield

        application.state.rag = None
        application.state.ingestion = None
        application.state.document_registry = None
        application.state.vectorstore = None

    application = FastAPI(
        title="Adaptive RAG API",
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()
