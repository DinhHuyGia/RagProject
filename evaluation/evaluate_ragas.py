import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI

from grounded.adaptive_rag import AdaptiveRAG
from grounded.shared import (
    format_docs,
    get_api_key,
    get_base_url,
)

if __package__:
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

REPORT_PATH = Path("evaluation/report/ragas_results.json")


def select_records(records: list[dict]) -> list[dict]:
    """Apply an optional question limit for lower-cost test runs."""
    configured_limit = os.getenv("RAGAS_EVALUATION_LIMIT")

    if configured_limit is None:
        return records

    try:
        limit = int(configured_limit)
    except ValueError as error:
        raise ValueError(
            "RAGAS_EVALUATION_LIMIT must be a positive integer."
        ) from error

    if limit < 1:
        raise ValueError(
            "RAGAS_EVALUATION_LIMIT must be a positive integer."
        )

    return records[:limit]


async def main() -> None:
    records = select_records(load_dataset(DATASET_PATH))
    retriever = build_retriever()

    # Reuse the project's answer prompt and Azure model configuration.
    rag = AdaptiveRAG(chunk_retriever=retriever)

    client = AsyncOpenAI(
        api_key=get_api_key(),
        base_url=get_base_url(),
        timeout=120,
        max_retries=3,
    )

    evaluator_model = os.getenv(
        "RAGAS_EVALUATOR_MODEL",
        os.getenv(
            "AZURE_OPENAI_CHAT_DEPLOYMENT",
            "gpt-5-mini",
        ),
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

    context_precision = ContextPrecision(llm=evaluator_llm)
    context_recall = ContextRecall(llm=evaluator_llm)
    faithfulness = Faithfulness(llm=evaluator_llm)
    answer_relevancy = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    factual_correctness = FactualCorrectness(
        llm=evaluator_llm,
        mode="f1",
        atomicity="high",
        coverage="high",
    )

    results = []

    for record in records:
        question = record["user_input"]
        reference = get_reference(record)

        # This first Ragas run evaluates the simple retrieval strategy.
        documents = rag.retrieve(
            queries=[question],
            indexes=["chunks"],
        )
        retrieved_contexts = [
            document.page_content
            for document in documents
        ]

        response = rag.answer_chain.invoke(
            {
                "question": question,
                "context": format_docs(documents),
            }
        )

        if record["answerable"]:
            relevancy_result = await answer_relevancy.ascore(
                user_input=question,
                response=response,
            )
            correctness_result = await factual_correctness.ascore(
                response=response,
                reference=reference,
            )
            precision_result = await context_precision.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
            recall_result = await context_recall.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
            faithfulness_result = await faithfulness.ascore(
                user_input=question,
                response=response,
                retrieved_contexts=retrieved_contexts,
            )
            precision_score = float(precision_result.value)
            recall_score = float(recall_result.value)
            faithfulness_score = float(faithfulness_result.value)
            relevancy_score = float(relevancy_result.value)
            correctness_score = float(correctness_result.value)
            refusal_score = None
            refusal_detected = None
            refusal_topic_coverage = None
        else:
            # Retrieval metrics have no meaningful gold-context
            # denominator for intentionally unanswerable questions.
            precision_score = None
            recall_score = None
            faithfulness_score = None
            relevancy_score = None
            correctness_score = None
            (
                refusal_score,
                refusal_detected,
                refusal_topic_coverage,
            ) = score_unanswerable_response(response, record)

        row = {
            "id": record["id"],
            "category": record["category"],
            "answerable": record["answerable"],
            "user_input": question,
            "reference": reference,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
            "context_precision": precision_score,
            "context_recall": recall_score,
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score,
            "factual_correctness": correctness_score,
            "refusal_correctness": refusal_score,
            "refusal_detected": refusal_detected,
            "refusal_topic_coverage": refusal_topic_coverage,
        }
        results.append(row)

        print(f"\n{record['id']}")
        print(f"  Expected answer: {reference}")
        print(f"  Generated answer: {response}")
        if record["answerable"]:
            print(
                f"  Context precision: "
                f"{row['context_precision']:.3f}"
            )
            print(
                f"  Context recall:    "
                f"{row['context_recall']:.3f}"
            )
            print(
                f"  Faithfulness:      "
                f"{row['faithfulness']:.3f}"
            )
            print(
                f"  Answer relevancy:  "
                f"{row['answer_relevancy']:.3f}"
            )
            print(
                f"  Factual correctness: "
                f"{row['factual_correctness']:.3f}"
            )
        else:
            print("  Context precision: n/a (unanswerable)")
            print("  Context recall:    n/a (unanswerable)")
            print("  Faithfulness:      n/a (unanswerable)")
            print("  Answer relevancy:  n/a (unanswerable)")
            print("  Factual correctness: n/a (unanswerable)")
            print(
                f"  Refusal correctness: "
                f"{row['refusal_correctness']:.3f}"
            )
            print(
                f"  Refusal detected: "
                f"{row['refusal_detected']}"
            )
            print(
                f"  Refusal topic coverage: "
                f"{row['refusal_topic_coverage']:.3f}"
            )

    summary, metric_sample_counts = summarize_by_answerability(
        results
    )

    report = {
        "strategy": "simple",
        "top_k": TOP_K,
        "questions_evaluated": len(results),
        "answerable_questions": sum(
            row["answerable"] for row in results
        ),
        "unanswerable_questions": sum(
            not row["answerable"] for row in results
        ),
        "summary": summary,
        "metric_sample_counts": metric_sample_counts,
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Answerable-question metrics ===")

    for metric, score in summary["answerable"].items():
        sample_count = metric_sample_counts["answerable"][metric]
        if score is None:
            print(f"{metric}: n/a")
        else:
            print(f"{metric}: {score:.3f} (n={sample_count})")

    print("\n=== Unanswerable-question metrics ===")

    for metric, score in summary["unanswerable"].items():
        sample_count = metric_sample_counts["unanswerable"][metric]
        if score is None:
            print(f"{metric}: n/a")
        else:
            print(f"{metric}: {score:.3f} (n={sample_count})")

    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
