"""Section 5: fuse rankings from several generated retrieval queries."""

from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from grounded.shared import (
    build_blog_resources,
    create_base_rag_prompt,
    create_llm,
    format_docs,
)

RAG_FUSION_TEMPLATE = """Generate four distinct search queries that preserve
the intent of the following question. Return one query per line.

Question: {question}"""


def reciprocal_rank_fusion(results: list[list], k: int = 60):
    scores: dict[str, float] = {}
    for documents in results:
        for rank, document in enumerate(documents):
            key = dumps(document)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + k)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    print("RAG Fusion ranked documents:")
    for index, (serialized, score) in enumerate(ranked, start=1):
        document = loads(serialized)
        preview = document.page_content.replace("\n", " ")[:160]
        print(f"{index}. score={score:.5f} | {preview}")
    return [loads(serialized) for serialized, _ in ranked]


def build_rag_fusion_chain():
    resources = build_blog_resources("section_05_rag_fusion")
    llm = create_llm()
    generate_queries = (
        ChatPromptTemplate.from_template(RAG_FUSION_TEMPLATE)
        | llm
        | StrOutputParser()
        | (lambda text: [line.strip() for line in text.splitlines() if line.strip()])
    )

    def retrieve(input_data: dict):
        queries = generate_queries.invoke(input_data)
        print("Generated RAG Fusion queries:")
        for index, query in enumerate(queries, start=1):
            print(f"{index}. {query}")
        results = [resources.retriever.invoke(query) for query in queries]
        return reciprocal_rank_fusion(results)

    return (
        {
            "context": RunnableLambda(retrieve) | format_docs,
            "question": RunnableLambda(lambda data: data["question"]),
        }
        | create_base_rag_prompt()
        | llm
        | StrOutputParser()
    )


def main() -> None:
    chain = build_rag_fusion_chain()
    question = "Compare task decomposition and self-reflection in LLM agents."
    print(chain.invoke({"question": question}))


if __name__ == "__main__":
    main()

