import pytest

from evaluation.scoring import (
    get_reference,
    score_unanswerable_response,
    summarize_by_answerability,
)


def test_get_reference_prefers_atomic_claims() -> None:
    record = {
        "id": "example",
        "reference": "Broad reference.",
        "reference_claims": [
            "First required claim.",
            "Second required claim.",
        ],
    }

    assert get_reference(record) == (
        "First required claim. Second required claim."
    )


def test_unanswerable_response_requires_refusal_and_topic() -> None:
    record = {
        "id": "unsupported-load-time",
        "refusal_topic_terms": ["page load time"],
    }

    score, refusal_detected, topic_coverage = (
        score_unanswerable_response(
            "The document does not state its average page-load time.",
            record,
        )
    )

    assert score == 1.0
    assert refusal_detected is True
    assert topic_coverage == 1.0


def test_unanswerable_response_rejects_unsupported_direct_answer() -> None:
    record = {
        "id": "unsupported-load-time",
        "refusal_topic_terms": ["page load time"],
    }

    score, refusal_detected, topic_coverage = (
        score_unanswerable_response(
            "The average page-load time is two seconds.",
            record,
        )
    )

    assert score == 0.0
    assert refusal_detected is False
    assert topic_coverage == 1.0


def test_summaries_are_separated_by_answerability() -> None:
    answerable = {
        "answerable": True,
        "context_precision": 0.8,
        "context_recall": 1.0,
        "faithfulness": 1.0,
        "answer_relevancy": 0.7,
        "factual_correctness": 0.9,
        "refusal_correctness": None,
    }
    unanswerable = {
        "answerable": False,
        "context_precision": None,
        "context_recall": None,
        "faithfulness": None,
        "answer_relevancy": None,
        "factual_correctness": None,
        "refusal_correctness": 1.0,
    }

    summary, counts = summarize_by_answerability(
        [answerable, unanswerable]
    )

    assert summary["answerable"]["answer_relevancy"] == pytest.approx(
        0.7
    )
    assert summary["unanswerable"]["refusal_correctness"] == 1.0
    assert counts["answerable"]["answer_relevancy"] == 1
    assert counts["unanswerable"]["refusal_correctness"] == 1
