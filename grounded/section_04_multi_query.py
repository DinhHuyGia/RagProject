"""Section 4: retrieve with multiple LLM-generated query perspectives."""

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

MULTI_QUERY_TEMPLATE = """You are an AI language model assistant. Generate five
different versions of the user's question for vector retrieval. The variations
should provide different perspectives while preserving the original intent.
Return one query per line.

Original question: {question}"""


def get_unique_union(document_lists: list[list]):
    serialized = [dumps(doc) for documents in document_lists for doc in documents]
    return [loads(doc) for doc in dict.fromkeys(serialized)]


def build_multi_query_rag_chain():
    resources = build_blog_resources("section_04_multi_query")
    llm = create_llm()
    query_prompt = ChatPromptTemplate.from_template(MULTI_QUERY_TEMPLATE)
    generate_queries = (
        query_prompt
        | llm
        | StrOutputParser()
        | (lambda text: [line.strip() for line in text.splitlines() if line.strip()])
    )

    def retrieve(input_data: dict):
        queries = generate_queries.invoke(input_data)
        print("Generated queries:")
        for index, query in enumerate(queries, start=1):
            print(f"{index}. {query}")
        return get_unique_union(
            [resources.retriever.invoke(query) for query in queries]
        )

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
    chain = build_multi_query_rag_chain()
    print(chain.invoke({"question": "How do agents think?"}))


if __name__ == "__main__":
    main()

