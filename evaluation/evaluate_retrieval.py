import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from grounded.ingestion import DocumentIngestionService
from grounded.shared import create_embeddings

EVALUATION_DOCUMENTS = (
    (
        Path("evaluation/fixtures/apple_website_analysis.pdf"),
        "eval-apple-website",
    ),
    (
        Path("evaluation/fixtures/Apple 10K.pdf"),
        "eval-apple-2025-10k",
    ),
)

DATASET_PATH = Path(
    "evaluation/apple_website_analysis_questions.jsonl"
)
TOP_K = 8


def normalize_text(text: str) -> str:
    """Make PDF and evidence whitespace comparable."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from error

    return records


def build_retriever():
    missing_paths = [
        path
        for path, _document_id in EVALUATION_DOCUMENTS
        if not path.is_file()
    ]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Evaluation document(s) not found: {missing}")

    vectorstore = Chroma(
        collection_name="apple_multi_document_evaluation",
        embedding_function=create_embeddings(),
    )

    ingestion = DocumentIngestionService(
        vectorstore,
        chunk_size=600,
        chunk_overlap=100,
    )

    for pdf_path, document_id in EVALUATION_DOCUMENTS:
        result = ingestion.ingest_file(
            pdf_path,
            original_filename=pdf_path.name,
            document_id=document_id,
        )
        print(
            f"Ingested {result.filename}: "
            f"{result.page_count} pages, "
            f"{result.chunk_count} chunks "
            f"(document_id={result.document_id})."
        )

    return vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )


def is_relevant(
    document: Document,
    expected_evidence: list[str],
) -> bool:
    """A chunk is relevant if it contains any expected evidence."""
    document_text = normalize_text(document.page_content)

    return any(
        normalize_text(evidence) in document_text
        for evidence in expected_evidence
    )


def calculate_precision_at_k(
    documents: list[Document],
    expected_evidence: list[str],
    k: int,
) -> tuple[float, list[bool]]:
    top_documents = documents[:k]

    relevance = [
        is_relevant(document, expected_evidence)
        for document in top_documents
    ]

    relevant_count = sum(relevance)

    # Missing results count as non-relevant.
    precision = relevant_count / k

    return precision, relevance


def calculate_recall_at_k(
    documents: list[Document],
    expected_evidence: list[str],
    k: int,
) -> tuple[float, list[bool]]:
    """Measure how many distinct evidence items appear in the top K."""
    if not expected_evidence:
        raise ValueError(
            "Recall requires at least one expected evidence item."
        )

    document_texts = [
        normalize_text(document.page_content)
        for document in documents[:k]
    ]

    evidence_found = [
        any(
            normalize_text(evidence) in document_text
            for document_text in document_texts
        )
        for evidence in expected_evidence
    ]

    recall = sum(evidence_found) / len(expected_evidence)

    return recall, evidence_found


def calculate_source_recall_at_k(
    documents: list[Document],
    expected_sources: list[str],
    k: int,
) -> tuple[float, list[bool]]:
    """Measure how many expected source files appear in the top K."""
    if not expected_sources:
        raise ValueError(
            "Source recall requires at least one expected source."
        )

    retrieved_sources = {
        document.metadata.get("filename")
        for document in documents[:k]
    }
    sources_found = [
        source in retrieved_sources for source in expected_sources
    ]

    return sum(sources_found) / len(expected_sources), sources_found


def main() -> None:
    dataset = load_dataset(DATASET_PATH)
    retriever = build_retriever()

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    source_recall_scores: list[float] = []
    total_evidence = 0
    total_evidence_found = 0
    total_sources = 0
    total_sources_found = 0

    for record in dataset:
        # Precision is not meaningful for these records because they
        # intentionally have no relevant evidence.
        if not record["answerable"]:
            continue

        question = record["user_input"]
        expected_evidence = record["reference_contexts"]
        expected_sources = record.get("expected_sources", [])

        documents = retriever.invoke(question)

        precision, relevance = calculate_precision_at_k(
            documents=documents,
            expected_evidence=expected_evidence,
            k=TOP_K,
        )
        recall, evidence_found = calculate_recall_at_k(
            documents=documents,
            expected_evidence=expected_evidence,
            k=TOP_K,
        )

        source_result = None
        if expected_sources:
            source_result = calculate_source_recall_at_k(
                documents=documents,
                expected_sources=expected_sources,
                k=TOP_K,
            )
            source_recall, sources_found = source_result
            source_recall_scores.append(source_recall)
            total_sources += len(expected_sources)
            total_sources_found += sum(sources_found)

        precision_scores.append(precision)
        recall_scores.append(recall)
        total_evidence += len(expected_evidence)
        total_evidence_found += sum(evidence_found)

        relevant_count = sum(relevance)
        found_count = sum(evidence_found)

        print(f"\n{record['id']}")
        print(f"Question: {question}")
        print(
            f"Precision@{TOP_K}: {precision:.2f} "
            f"({relevant_count}/{TOP_K})"
        )
        print(
            f"Recall@{TOP_K}: {recall:.2f} "
            f"({found_count}/{len(expected_evidence)} evidence items)"
        )
        if source_result is not None:
            source_recall, sources_found = source_result
            print(
                f"Source Recall@{TOP_K}: {source_recall:.2f} "
                f"({sum(sources_found)}/{len(expected_sources)} sources)"
            )
            for source, found in zip(
                expected_sources,
                sources_found,
                strict=True,
            ):
                label = "found" if found else "missing"
                print(f"  Source {label}: {source}")

        for evidence, found in zip(
            expected_evidence,
            evidence_found,
            strict=True,
        ):
            label = "found" if found else "missing"
            print(f"  Evidence {label}: {evidence}")

        for rank, (document, relevant) in enumerate(
            zip(documents[:TOP_K], relevance, strict=True),
            start=1,
        ):
            label = "relevant" if relevant else "not relevant"
            chunk_id = document.metadata.get("chunk_id")
            page = document.metadata.get("page_number")
            source = document.metadata.get("filename")

            print(
                f"  {rank}. {label}; "
                f"source={source}; chunk={chunk_id}; page={page}"
            )

    print("\n=== Overall result ===")
    print(f"Questions evaluated: {len(precision_scores)}")
    print(
        f"Mean Precision@{TOP_K}: "
        f"{mean(precision_scores):.3f}"
    )
    print(
        f"Mean Recall@{TOP_K}: "
        f"{mean(recall_scores):.3f}"
    )
    print(
        f"Evidence Recall@{TOP_K}: "
        f"{total_evidence_found / total_evidence:.3f} "
        f"({total_evidence_found}/{total_evidence})"
    )
    if source_recall_scores:
        print(
            f"Mean Source Recall@{TOP_K}: "
            f"{mean(source_recall_scores):.3f}"
        )
        print(
            f"Source Recall@{TOP_K}: "
            f"{total_sources_found / total_sources:.3f} "
            f"({total_sources_found}/{total_sources})"
        )


if __name__ == "__main__":
    main()
