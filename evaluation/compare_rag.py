import argparse

from grounded.adaptive_rag import AdaptiveRAG, RequestedStrategy
from grounded.langgraph_rag import LangGraphRAG

if __package__:
    from .evaluate_retrieval import build_retriever
else:
    from evaluate_retrieval import build_retriever


STRATEGIES: tuple[RequestedStrategy, ...] = (
    "auto",
    "simple",
    "multi_query",
    "decomposition",
    "step_back",
    "hyde",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one question through AdaptiveRAG and LangGraphRAG."
        )
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to send to both RAG implementations.",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="auto",
        help="Retrieval strategy shared by both runs (default: auto).",
    )
    return parser.parse_args()


def print_list(label: str, values: list[str]) -> None:
    print(f"{label}:")
    for value in values:
        print(f"  - {value}")


def main() -> None:
    args = parse_args()
    question = args.question.strip()
    strategy: RequestedStrategy = args.strategy

    if not question:
        raise ValueError("--question must not be empty.")

    retriever = build_retriever()
    langchain_rag = AdaptiveRAG(chunk_retriever=retriever)
    langgraph_rag = LangGraphRAG(chunk_retriever=retriever)

    print("\nRunning LangChain AdaptiveRAG...")
    langchain_result = langchain_rag.invoke(
        question=question,
        strategy=strategy,
    )

    print("Running LangGraphRAG...")
    langgraph_result = langgraph_rag.invoke(
        question=question,
        strategy=strategy,
    )

    print("\n=== Shared input ===")
    print(f"Question: {question}")
    print(f"Requested strategy: {strategy}")

    print("\n=== LangChain AdaptiveRAG ===")
    print(f"Resolved strategy: {langchain_result.strategy}")
    print(f"Routing reason: {langchain_result.routing_reason}")
    print_list(
        "Retrieval queries",
        langchain_result.retrieval_queries,
    )
    print(f"Answer: {langchain_result.answer}")

    print("\n=== LangGraphRAG ===")
    print(f"Resolved strategy: {langgraph_result['strategy']}")
    print(f"Routing reason: {langgraph_result['routing_reason']}")
    retrieval_plan = langgraph_result["retrieval_plan"]
    print(f"Retrieval plan reason: {retrieval_plan.reason}")
    print(
        "Requires multiple sources: "
        f"{retrieval_plan.requires_multiple_sources}"
    )
    print("Planned retrieval tasks:")
    for position, task in enumerate(retrieval_plan.tasks, start=1):
        source = task.source_hint or "any relevant source"
        print(f"  {position}. {task.subquestion}")
        print(f"     Query: {task.search_query}")
        print(f"     Source hint: {source}")
        print(
            "     Document filter: "
            f"{task.source_document_id or 'none'}"
        )
        print(f"     Status: {task.status}")
        print(f"     Retrieval attempts: {task.attempts}")
        print(
            "     Evidence chunks: "
            f"{', '.join(task.evidence_chunk_ids) or 'none'}"
        )
        print(
            "     Evidence grade: "
            f"{task.grade_reason or 'not graded'}"
        )
    print_list(
        "Final retrieval queries",
        langgraph_result["retrieval_queries"],
    )
    print(f"Retrieval retries: {langgraph_result['retry_count']}")
    print(
        "Context sufficient: "
        f"{langgraph_result['context_sufficient']}"
    )
    print(
        "Context grade reason: "
        f"{langgraph_result['context_grade_reason']}"
    )
    print(f"Answer: {langgraph_result['answer']}")


if __name__ == "__main__":
    main()
