import pytest

from evaluation.evaluate_ragas_comparison import (
    calculate_agentic_metrics,
    calculate_row_delta,
    calculate_summary_delta,
    select_records,
    summarize_agentic_metrics,
)


def test_select_records_filters_by_id_then_limit() -> None:
    records = [
        {"id": "first"},
        {"id": "second"},
    ]

    assert select_records(records, limit=1) == [{"id": "first"}]
    assert select_records(records, question_id="second") == [
        {"id": "second"}
    ]


def test_select_records_rejects_unknown_id_and_invalid_limit() -> None:
    records = [{"id": "first"}]

    with pytest.raises(ValueError, match="No evaluation question"):
        select_records(records, question_id="missing")

    with pytest.raises(ValueError, match="positive integer"):
        select_records(records, limit=0)


def test_calculate_summary_delta_is_langgraph_minus_langchain() -> None:
    langchain = {
        "answerable": {
            "context_recall": 0.75,
            "answer_relevancy": None,
        },
        "unanswerable": {"refusal_correctness": 0.5},
    }
    langgraph = {
        "answerable": {
            "context_recall": 1.0,
            "answer_relevancy": None,
        },
        "unanswerable": {"refusal_correctness": 1.0},
    }

    assert calculate_summary_delta(langchain, langgraph) == {
        "answerable": {
            "context_recall": 0.25,
            "answer_relevancy": None,
        },
        "unanswerable": {"refusal_correctness": 0.5},
    }


def test_calculate_row_delta_uses_applicable_metrics() -> None:
    langchain = {
        "answerable": False,
        "refusal_correctness": 0.0,
    }
    langgraph = {
        "answerable": False,
        "refusal_correctness": 1.0,
    }

    assert calculate_row_delta(langchain, langgraph) == {
        "refusal_correctness": 1.0
    }


def test_calculate_agentic_metrics_tracks_tasks_retries_and_sources() -> None:
    record = {
        "expected_sources": [
            "apple_website_analysis.pdf",
            "Apple 10K.pdf",
        ]
    }
    details = {
        "available_sources": [
            {
                "document_id": "website-document",
                "filename": "apple_website_analysis.pdf",
            },
            {
                "document_id": "10k-document",
                "filename": "Apple 10K.pdf",
            },
        ],
        "retrieval_plan": {
            "tasks": [
                {
                    "task_id": "evidence-1",
                    "status": "satisfied",
                    "attempts": 1,
                    "source_document_id": "website-document",
                },
                {
                    "task_id": "evidence-2",
                    "status": "satisfied",
                    "attempts": 2,
                    "source_document_id": "10k-document",
                },
            ]
        },
        "task_retry_events": [
            {
                "task_id": "evidence-2",
                "retry_type": "targeted",
                "status_before": "insufficient",
            }
        ],
        "pipeline_llm_call_count": 7,
        "pipeline_llm_call_counts": {
            "retrieval_planner": 1,
            "evidence_task_grader": 3,
            "task_query_rewriter": 1,
            "context_grader": 1,
            "answer_generator": 1,
        },
    }

    metrics = calculate_agentic_metrics(record, details)

    assert metrics["task_count"] == 2
    assert metrics["task_completion_rate"] == 1.0
    assert metrics["average_attempts_per_task"] == 1.5
    assert metrics["retried_task_count"] == 1
    assert metrics["total_task_retries"] == 1
    assert metrics["targeted_retry_accuracy"] == 1.0
    assert metrics["unnecessary_retry_count"] == 0
    assert metrics["source_routing_accuracy"] == 1.0
    assert metrics["source_routing_precision"] == 1.0
    assert metrics["source_routing_recall"] == 1.0
    assert metrics["pipeline_llm_call_count"] == 7


def test_summarize_agentic_metrics_uses_task_weighted_rates() -> None:
    rows = [
        {
            "agentic_metrics": {
                "task_count": 2,
                "satisfied_task_count": 2,
                "failed_task_count": 0,
                "unresolved_task_count": 0,
                "all_tasks_satisfied": True,
                "average_attempts_per_task": 1.5,
                "retried_task_count": 1,
                "total_task_retries": 1,
                "retry_event_count": 1,
                "targeted_retry_count": 1,
                "correct_retry_count": 1,
                "unnecessary_retry_count": 0,
                "source_routing_accuracy": 1.0,
                "source_routing_precision": 1.0,
                "source_routing_recall": 1.0,
                "pipeline_llm_call_count": 7,
                "elapsed_seconds": 4.0,
            }
        },
        {
            "agentic_metrics": {
                "task_count": 1,
                "satisfied_task_count": 0,
                "failed_task_count": 1,
                "unresolved_task_count": 0,
                "all_tasks_satisfied": False,
                "average_attempts_per_task": 3.0,
                "retried_task_count": 1,
                "total_task_retries": 2,
                "retry_event_count": 2,
                "targeted_retry_count": 1,
                "correct_retry_count": 1,
                "unnecessary_retry_count": 1,
                "source_routing_accuracy": None,
                "source_routing_precision": None,
                "source_routing_recall": None,
                "pipeline_llm_call_count": 5,
                "elapsed_seconds": 2.0,
            }
        },
    ]

    summary = summarize_agentic_metrics(rows)

    assert summary["task_completion_rate"] == pytest.approx(2 / 3)
    assert summary["question_completion_rate"] == 0.5
    assert summary["average_attempts_per_task"] == 2.0
    assert summary["total_task_retries"] == 3
    assert summary["targeted_retry_accuracy"] == pytest.approx(2 / 3)
    assert summary["unnecessary_retry_rate"] == pytest.approx(1 / 3)
    assert summary["source_routing_accuracy"] == 1.0
    assert summary["source_routing_sample_count"] == 1
    assert summary["average_pipeline_llm_calls"] == 6.0
    assert summary["average_elapsed_seconds"] == 3.0
