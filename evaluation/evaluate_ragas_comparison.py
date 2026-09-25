import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.documents import Document
from openai import AsyncOpenAI

from grounded.adaptive_rag import AdaptiveRAG, RequestedStrategy
from grounded.langgraph_rag import LangGraphRAG
from grounded.shared import get_api_key, get_base_url

if __package__:
    from .compare_rag import STRATEGIES
    from .evaluate_retrieval import (
        DATASET_PATH,
        TOP_K,
        build_retriever,
        load_dataset,
    )
    from .ragas_compat import install_vertexai_import_shim
    from .scoring import (
        get_reference,
        score_unanswerable_response,
        summarize_by_answerability,
    )
else:
    from compare_rag import STRATEGIES
    from evaluate_retrieval import (
        DATASET_PATH,
        TOP_K,
        build_retriever,
        load_dataset,
    )
    from ragas_compat import install_vertexai_import_shim
    from scoring import (
        get_reference,
        score_unanswerable_response,
        summarize_by_answerability,
    )

install_vertexai_import_shim()

from ragas.embeddings.base import embedding_factory  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

REPORT_PATH = Path("evaluation/report/ragas_comparison_results.json")
SYSTEM_NAMES = ("langchain", "langgraph")
ANSWERABLE_SCORE_NAMES = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "factual_correctness",
)
UNANSWERABLE_SCORE_NAMES = ("refusal_correctness",)


@dataclass
class PipelineRun:
    response: str
    documents: list[Document]
    details: dict[str, Any]
    elapsed_seconds: float


@dataclass
class RagasMetrics:
    context_precision: ContextPrecision
    context_recall: ContextRecall
    faithfulness: Faithfulness
    answer_relevancy: AnswerRelevancy
    factual_correctness: FactualCorrectness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AdaptiveRAG and LangGraphRAG with RAGAS."
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="simple",
        help="Retrieval strategy used by both systems (default: simple).",
    )
    parser.add_argument(
        "--question-id",
        help="Evaluate only the record with this ID.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N selected records.",
    )
    return parser.parse_args()


def select_records(
    records: list[dict],
    *,
    question_id: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Filter records for targeted or lower-cost comparison runs."""
    selected = records

    if question_id:
        selected = [
            record for record in selected if record["id"] == question_id
        ]
        if not selected:
            raise ValueError(
                f"No evaluation question found with id: {question_id}"
            )

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer.")
        selected = selected[:limit]

    return selected


def create_ragas_metrics() -> RagasMetrics:
    client = AsyncOpenAI(
        api_key=get_api_key(),
        base_url=get_base_url(),
        timeout=120,
        max_retries=3,
    )
    evaluator_model = os.getenv(
        "RAGAS_EVALUATOR_MODEL",
        os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5-mini"),
    )
    embedding_model = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "text-embedding-3-small",
    )
    evaluator_llm = llm_factory(
        evaluator_model,
        provider="openai",
        client=client,
        max_tokens=4096,
    )
    evaluator_embeddings = embedding_factory(
        "openai",
        model=embedding_model,
        client=client,
    )

    return RagasMetrics(
        context_precision=ContextPrecision(llm=evaluator_llm),
        context_recall=ContextRecall(llm=evaluator_llm),
        faithfulness=Faithfulness(llm=evaluator_llm),
        answer_relevancy=AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
        factual_correctness=FactualCorrectness(
            llm=evaluator_llm,
            mode="f1",
            atomicity="high",
            coverage="high",
        ),
    )


def run_langchain(
    rag: AdaptiveRAG,
    question: str,
    strategy: RequestedStrategy,
) -> PipelineRun:
    started_at = perf_counter()
    result = rag.invoke(question=question, strategy=strategy)

    # AdaptiveAnswer exposes source metadata, not full Documents. Repeat the
    # deterministic retrieval to retain the exact contexts for RAGAS.
    documents = rag.retrieve(
        queries=result.retrieval_queries,
        indexes=result.indexes,
    )
    return PipelineRun(
        response=result.answer,
        documents=documents,
        details={
            "resolved_strategy": result.strategy,
            "routing_reason": result.routing_reason,
            "retrieval_queries": result.retrieval_queries,
            "retrieval_retries": 0,
            "retrieval_attempts": 1,
        },
        elapsed_seconds=perf_counter() - started_at,
    )


def run_langgraph(
    rag: LangGraphRAG,
    question: str,
    strategy: RequestedStrategy,
) -> PipelineRun:
    started_at = perf_counter()
    result = rag.invoke(question=question, strategy=strategy)
    retry_count = result["retry_count"]
    retrieval_plan = result["retrieval_plan"]
    llm_call_counts = result.get("llm_call_counts", {})
    return PipelineRun(
        response=result["answer"],
        documents=result["documents"],
        details={
            "resolved_strategy": result["strategy"],
            "routing_reason": result["routing_reason"],
            "available_sources": [
                source.model_dump()
                for source in result.get("available_sources", [])
            ],
            "retrieval_plan": retrieval_plan.model_dump(),
            "retrieval_queries": result["retrieval_queries"],
            "retrieval_scopes": [
                scope.model_dump()
                for scope in result["retrieval_scopes"]
            ],
            "retrieval_retries": retry_count,
            "retrieval_attempts": retry_count + 1,
            "context_sufficient": result["context_sufficient"],
            "context_grade_reason": result["context_grade_reason"],
            "all_evidence_tasks_satisfied": result.get(
                "all_evidence_tasks_satisfied"
            ),
            "evidence_grade_reason": result.get(
                "evidence_grade_reason"
            ),
            "task_retry_events": [
                event.model_dump()
                for event in result.get("task_retry_events", [])
            ],
            "pipeline_llm_call_counts": llm_call_counts,
            "pipeline_llm_call_count": sum(
                llm_call_counts.values()
            ),
        },
        elapsed_seconds=perf_counter() - started_at,
    )


def calculate_agentic_metrics(
    record: dict,
    details: dict[str, Any],
) -> dict[str, Any]:
    """Calculate deterministic metrics from final agentic graph state."""
    tasks = details["retrieval_plan"]["tasks"]
    task_count = len(tasks)
    satisfied_count = sum(
        task["status"] == "satisfied" for task in tasks
    )
    failed_count = sum(task["status"] == "failed" for task in tasks)
    unresolved_count = task_count - satisfied_count - failed_count
    total_attempts = sum(task["attempts"] for task in tasks)
    total_task_retries = sum(
        max(task["attempts"] - 1, 0) for task in tasks
    )
    retried_task_count = sum(task["attempts"] > 1 for task in tasks)

    retry_events = details.get("task_retry_events", [])
    correct_retry_count = sum(
        event["status_before"] == "insufficient"
        for event in retry_events
    )
    targeted_retry_count = sum(
        event["retry_type"] == "targeted" for event in retry_events
    )
    unnecessary_retry_count = sum(
        event["status_before"] == "satisfied"
        for event in retry_events
    )
    retry_event_count = len(retry_events)

    source_by_id = {
        source["document_id"]: source["filename"]
        for source in details.get("available_sources", [])
    }
    selected_sources = sorted(
        {
            source_by_id[task["source_document_id"]]
            for task in tasks
            if task.get("source_document_id") in source_by_id
        }
    )
    expected_sources = sorted(record.get("expected_sources", []))

    if expected_sources:
        expected_set = set(expected_sources)
        selected_set = set(selected_sources)
        source_matches = len(expected_set & selected_set)
        source_routing_recall = source_matches / len(expected_set)
        source_routing_precision = (
            source_matches / len(selected_set)
            if selected_set
            else 0.0
        )
        source_routing_accuracy = float(
            expected_set == selected_set
        )
    else:
        source_routing_recall = None
        source_routing_precision = None
        source_routing_accuracy = None

    return {
        "task_count": task_count,
        "satisfied_task_count": satisfied_count,
        "failed_task_count": failed_count,
        "unresolved_task_count": unresolved_count,
        "task_completion_rate": (
            satisfied_count / task_count if task_count else 0.0
        ),
        "all_tasks_satisfied": (
            task_count > 0 and satisfied_count == task_count
        ),
        "average_attempts_per_task": (
            total_attempts / task_count if task_count else 0.0
        ),
        "retried_task_count": retried_task_count,
        "total_task_retries": total_task_retries,
        "retry_event_count": retry_event_count,
        "targeted_retry_count": targeted_retry_count,
        "correct_retry_count": correct_retry_count,
        "targeted_retry_accuracy": (
            correct_retry_count / retry_event_count
            if retry_event_count
            else None
        ),
        "unnecessary_retry_count": unnecessary_retry_count,
        "unnecessary_retry_rate": (
            unnecessary_retry_count / retry_event_count
            if retry_event_count
            else 0.0
        ),
        "expected_sources": expected_sources,
        "selected_sources": selected_sources,
        "source_routing_accuracy": source_routing_accuracy,
        "source_routing_precision": source_routing_precision,
        "source_routing_recall": source_routing_recall,
        "pipeline_llm_call_count": details.get(
            "pipeline_llm_call_count",
            0,
        ),
        "pipeline_llm_call_counts": details.get(
            "pipeline_llm_call_counts",
            {},
        ),
    }


def summarize_agentic_metrics(rows: list[dict]) -> dict[str, Any]:
    """Aggregate per-question agentic metrics across LangGraph runs."""
    metrics = [row["agentic_metrics"] for row in rows]
    task_count = sum(metric["task_count"] for metric in metrics)
    satisfied_count = sum(
        metric["satisfied_task_count"] for metric in metrics
    )
    failed_count = sum(
        metric["failed_task_count"] for metric in metrics
    )
    unresolved_count = sum(
        metric["unresolved_task_count"] for metric in metrics
    )
    total_attempts = sum(
        metric["average_attempts_per_task"] * metric["task_count"]
        for metric in metrics
    )
    retry_event_count = sum(
        metric["retry_event_count"] for metric in metrics
    )
    correct_retry_count = sum(
        metric["correct_retry_count"] for metric in metrics
    )
    unnecessary_retry_count = sum(
        metric["unnecessary_retry_count"] for metric in metrics
    )
    source_scores = [
        metric["source_routing_accuracy"]
        for metric in metrics
        if metric["source_routing_accuracy"] is not None
    ]
    source_precision_scores = [
        metric["source_routing_precision"]
        for metric in metrics
        if metric["source_routing_precision"] is not None
    ]
    source_recall_scores = [
        metric["source_routing_recall"]
        for metric in metrics
        if metric["source_routing_recall"] is not None
    ]

    return {
        "question_count": len(metrics),
        "task_count": task_count,
        "satisfied_task_count": satisfied_count,
        "failed_task_count": failed_count,
        "unresolved_task_count": unresolved_count,
        "task_completion_rate": (
            satisfied_count / task_count if task_count else 0.0
        ),
        "question_completion_rate": (
            sum(metric["all_tasks_satisfied"] for metric in metrics)
            / len(metrics)
            if metrics
            else 0.0
        ),
        "average_attempts_per_task": (
            total_attempts / task_count if task_count else 0.0
        ),
        "retried_task_count": sum(
            metric["retried_task_count"] for metric in metrics
        ),
        "total_task_retries": sum(
            metric["total_task_retries"] for metric in metrics
        ),
        "retry_event_count": retry_event_count,
        "targeted_retry_count": sum(
            metric["targeted_retry_count"] for metric in metrics
        ),
        "targeted_retry_accuracy": (
            correct_retry_count / retry_event_count
            if retry_event_count
            else None
        ),
        "unnecessary_retry_count": unnecessary_retry_count,
        "unnecessary_retry_rate": (
            unnecessary_retry_count / retry_event_count
            if retry_event_count
            else 0.0
        ),
        "source_routing_accuracy": (
            sum(source_scores) / len(source_scores)
            if source_scores
            else None
        ),
        "source_routing_precision": (
            sum(source_precision_scores) / len(source_precision_scores)
            if source_precision_scores
            else None
        ),
        "source_routing_recall": (
            sum(source_recall_scores) / len(source_recall_scores)
            if source_recall_scores
            else None
        ),
        "source_routing_sample_count": len(source_scores),
        "total_pipeline_llm_calls": sum(
            metric.get("pipeline_llm_call_count", 0)
            for metric in metrics
        ),
        "average_pipeline_llm_calls": (
            sum(
                metric.get("pipeline_llm_call_count", 0)
                for metric in metrics
            )
            / len(metrics)
            if metrics
            else 0.0
        ),
        "average_elapsed_seconds": (
            sum(metric.get("elapsed_seconds", 0.0) for metric in metrics)
            / len(metrics)
            if metrics
            else 0.0
        ),
    }


async def score_pipeline_run(
    record: dict,
    run: PipelineRun,
    metrics: RagasMetrics,
) -> dict:
    question = record["user_input"]
    reference = get_reference(record)
    contexts = [document.page_content for document in run.documents]

    if record["answerable"]:
        precision = await metrics.context_precision.ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=contexts,
        )
        recall = await metrics.context_recall.ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=contexts,
        )
        faithfulness = await metrics.faithfulness.ascore(
            user_input=question,
            response=run.response,
            retrieved_contexts=contexts,
        )
        relevancy = await metrics.answer_relevancy.ascore(
            user_input=question,
            response=run.response,
        )
        correctness = await metrics.factual_correctness.ascore(
            response=run.response,
            reference=reference,
        )
        scores = {
            "context_precision": float(precision.value),
            "context_recall": float(recall.value),
            "faithfulness": float(faithfulness.value),
            "answer_relevancy": float(relevancy.value),
            "factual_correctness": float(correctness.value),
            "refusal_correctness": None,
            "refusal_detected": None,
            "refusal_topic_coverage": None,
        }
    else:
        refusal_score, refusal_detected, topic_coverage = (
            score_unanswerable_response(run.response, record)
        )
        scores = {
            "context_precision": None,
            "context_recall": None,
            "faithfulness": None,
            "answer_relevancy": None,
            "factual_correctness": None,
            "refusal_correctness": refusal_score,
            "refusal_detected": refusal_detected,
            "refusal_topic_coverage": topic_coverage,
        }

    row = {
        "id": record["id"],
        "category": record["category"],
        "answerable": record["answerable"],
        "user_input": question,
        "reference": reference,
        "response": run.response,
        "retrieved_contexts": contexts,
        "elapsed_seconds": run.elapsed_seconds,
        **run.details,
        **scores,
    }

    if "retrieval_plan" in run.details:
        agentic_metrics = calculate_agentic_metrics(
            record,
            run.details,
        )
        agentic_metrics["elapsed_seconds"] = run.elapsed_seconds
        row["agentic_metrics"] = agentic_metrics

    return row


def calculate_summary_delta(
    langchain_summary: dict[str, dict[str, float | None]],
    langgraph_summary: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    """Calculate LangGraph minus LangChain for applicable metrics."""
    delta: dict[str, dict[str, float | None]] = {}
    for group, graph_metrics in langgraph_summary.items():
        delta[group] = {}
        for metric, graph_score in graph_metrics.items():
            chain_score = langchain_summary[group][metric]
            if graph_score is None or chain_score is None:
                delta[group][metric] = None
            else:
                delta[group][metric] = graph_score - chain_score
    return delta


def calculate_row_delta(
    langchain_row: dict,
    langgraph_row: dict,
) -> dict[str, float | None]:
    """Calculate per-question LangGraph minus LangChain scores."""
    metric_names = (
        ANSWERABLE_SCORE_NAMES
        if langchain_row["answerable"]
        else UNANSWERABLE_SCORE_NAMES
    )
    delta = {}
    for metric in metric_names:
        chain_score = langchain_row[metric]
        graph_score = langgraph_row[metric]
        if chain_score is None or graph_score is None:
            delta[metric] = None
        else:
            delta[metric] = graph_score - chain_score
    return delta


def print_question_scores(
    langchain_row: dict,
    langgraph_row: dict,
) -> None:
    metric_names = (
        ANSWERABLE_SCORE_NAMES
        if langchain_row["answerable"]
        else UNANSWERABLE_SCORE_NAMES
    )
    print("  Scores (LangChain | LangGraph | delta):")
    for metric in metric_names:
        chain_score = langchain_row[metric]
        graph_score = langgraph_row[metric]
        if chain_score is None or graph_score is None:
            print(f"    {metric}: n/a")
            continue
        delta = graph_score - chain_score
        print(
            f"    {metric}: {chain_score:.3f} | "
            f"{graph_score:.3f} | {delta:+.3f}"
        )


def print_summary(
    system_name: str,
    summary: dict[str, dict[str, float | None]],
    counts: dict[str, dict[str, int]],
) -> None:
    print(f"\n=== {system_name} ===")
    for group, group_metrics in summary.items():
        print(f"{group}:")
        for metric, score in group_metrics.items():
            count = counts[group][metric]
            if score is None:
                print(f"  {metric}: n/a")
            else:
                print(f"  {metric}: {score:.3f} (n={count})")


def print_agentic_summary(summary: dict[str, Any]) -> None:
    """Print the deterministic LangGraph agentic metrics."""
    print("\n=== LangGraph agentic behavior ===")
    print(
        "Tasks satisfied: "
        f"{summary['satisfied_task_count']}/{summary['task_count']} "
        f"({summary['task_completion_rate']:.3f})"
    )
    print(
        "Questions with every task satisfied: "
        f"{summary['question_completion_rate']:.3f}"
    )
    print(
        "Average attempts per task: "
        f"{summary['average_attempts_per_task']:.3f}"
    )
    print(
        "Average pipeline LLM calls: "
        f"{summary['average_pipeline_llm_calls']:.3f}"
    )
    print(
        "Average LangGraph runtime: "
        f"{summary['average_elapsed_seconds']:.3f}s"
    )
    print(f"Total task retries: {summary['total_task_retries']}")
    retry_accuracy = summary["targeted_retry_accuracy"]
    print(
        "Targeted retry accuracy: "
        + (
            f"{retry_accuracy:.3f}"
            if retry_accuracy is not None
            else "n/a (no retries)"
        )
    )
    print(
        "Unnecessary retries: "
        f"{summary['unnecessary_retry_count']} "
        f"(rate={summary['unnecessary_retry_rate']:.3f})"
    )
    source_accuracy = summary["source_routing_accuracy"]
    print(
        "Exact source-routing accuracy: "
        + (
            f"{source_accuracy:.3f} "
            f"(n={summary['source_routing_sample_count']})"
            if source_accuracy is not None
            else "n/a (no expected_sources labels)"
        )
    )


async def main() -> None:
    args = parse_args()
    strategy: RequestedStrategy = args.strategy
    records = select_records(
        load_dataset(DATASET_PATH),
        question_id=args.question_id,
        limit=args.limit,
    )
    retriever = build_retriever()
    langchain_rag = AdaptiveRAG(chunk_retriever=retriever)
    langgraph_rag = LangGraphRAG(chunk_retriever=retriever)
    metrics = create_ragas_metrics()
    results = {name: [] for name in SYSTEM_NAMES}

    for record in records:
        print(f"\n=== {record['id']} ===")
        print("Running and scoring LangChain...")
        langchain_row = await score_pipeline_run(
            record,
            run_langchain(
                langchain_rag,
                record["user_input"],
                strategy,
            ),
            metrics,
        )
        results["langchain"].append(langchain_row)

        print("Running and scoring LangGraph...")
        langgraph_row = await score_pipeline_run(
            record,
            run_langgraph(
                langgraph_rag,
                record["user_input"],
                strategy,
            ),
            metrics,
        )
        results["langgraph"].append(langgraph_row)

        print(f"  Expected:  {langchain_row['reference']}")
        print(f"  LangChain: {langchain_row['response']}")
        print(f"  LangGraph: {langgraph_row['response']}")
        print(
            "  LangGraph retries: "
            f"{langgraph_row['retrieval_retries']}"
        )
        agentic = langgraph_row["agentic_metrics"]
        print(
            "  Evidence tasks: "
            f"{agentic['satisfied_task_count']}/"
            f"{agentic['task_count']} satisfied; "
            f"average attempts={agentic['average_attempts_per_task']:.2f}"
        )
        if agentic["source_routing_accuracy"] is not None:
            print(
                "  Exact source routing: "
                f"{agentic['source_routing_accuracy']:.0f}"
            )
        print_question_scores(langchain_row, langgraph_row)

    systems = {}
    for name in SYSTEM_NAMES:
        summary, counts = summarize_by_answerability(results[name])
        systems[name] = {
            "summary": summary,
            "metric_sample_counts": counts,
            "results": results[name],
        }

    agentic_summary = summarize_agentic_metrics(
        results["langgraph"]
    )
    systems["langgraph"]["agentic_summary"] = agentic_summary

    delta = calculate_summary_delta(
        systems["langchain"]["summary"],
        systems["langgraph"]["summary"],
    )
    per_question_comparison = [
        {
            "id": langchain_row["id"],
            "answerable": langchain_row["answerable"],
            "score_delta": calculate_row_delta(
                langchain_row,
                langgraph_row,
            ),
            "langchain_response": langchain_row["response"],
            "langgraph_response": langgraph_row["response"],
            "langgraph_retrieval_retries": langgraph_row[
                "retrieval_retries"
            ],
            "langgraph_agentic_metrics": langgraph_row[
                "agentic_metrics"
            ],
        }
        for langchain_row, langgraph_row in zip(
            results["langchain"],
            results["langgraph"],
            strict=True,
        )
    ]
    report = {
        "strategy": strategy,
        "top_k": TOP_K,
        "questions_evaluated": len(records),
        "answerable_questions": sum(
            record["answerable"] for record in records
        ),
        "unanswerable_questions": sum(
            not record["answerable"] for record in records
        ),
        "delta_definition": "langgraph_minus_langchain",
        "summary_delta": delta,
        "agentic_summary": agentic_summary,
        "per_question_comparison": per_question_comparison,
        "systems": systems,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_summary(
        "LangChain AdaptiveRAG",
        systems["langchain"]["summary"],
        systems["langchain"]["metric_sample_counts"],
    )
    print_summary(
        "LangGraphRAG",
        systems["langgraph"]["summary"],
        systems["langgraph"]["metric_sample_counts"],
    )
    print_agentic_summary(agentic_summary)

    print("\n=== LangGraph - LangChain score delta ===")
    for group, group_metrics in delta.items():
        print(f"{group}:")
        for metric, score in group_metrics.items():
            if score is None:
                print(f"  {metric}: n/a")
            else:
                print(f"  {metric}: {score:+.3f}")
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
