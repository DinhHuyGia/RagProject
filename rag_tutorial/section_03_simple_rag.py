"""Section 3: the simplest retrieval-augmented generation chain."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag_tutorial.shared import (
    build_blog_resources,
    create_base_rag_prompt,
    create_llm,
    format_docs,
)


def build_simple_rag_chain():
    resources = build_blog_resources("section_03_simple_rag")
    prompt = create_base_rag_prompt()
    llm = create_llm()

    return (
        {
            "context": resources.retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def main() -> None:
    chain = build_simple_rag_chain()
    print(chain.invoke("What is task decomposition?"))


if __name__ == "__main__":
    main()

