import pytest
from langchain_core.documents import Document

from evaluation.evaluate_retrieval import (
    calculate_source_recall_at_k,
)


def test_source_recall_counts_distinct_expected_files() -> None:
    documents = [
        Document(
            page_content="Website evidence",
            metadata={"filename": "apple_website_analysis.pdf"},
        ),
        Document(
            page_content="More website evidence",
            metadata={"filename": "apple_website_analysis.pdf"},
        ),
        Document(
            page_content="10-K evidence",
            metadata={"filename": "Apple 10K.pdf"},
        ),
    ]

    recall, sources_found = calculate_source_recall_at_k(
        documents,
        ["apple_website_analysis.pdf", "Apple 10K.pdf"],
        k=3,
    )

    assert recall == 1.0
    assert sources_found == [True, True]


def test_source_recall_respects_top_k() -> None:
    documents = [
        Document(
            page_content="Website evidence",
            metadata={"filename": "apple_website_analysis.pdf"},
        ),
        Document(
            page_content="10-K evidence",
            metadata={"filename": "Apple 10K.pdf"},
        ),
    ]

    recall, sources_found = calculate_source_recall_at_k(
        documents,
        ["apple_website_analysis.pdf", "Apple 10K.pdf"],
        k=1,
    )

    assert recall == 0.5
    assert sources_found == [True, False]


def test_source_recall_requires_expected_sources() -> None:
    with pytest.raises(ValueError, match="at least one expected source"):
        calculate_source_recall_at_k([], [], k=8)
