"""Section 11: search embedded summaries and return their original chunks."""

import uuid

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.stores import InMemoryByteStore

from rag_tutorial.shared import (
    create_base_rag_prompt,
    create_embeddings,
    create_llm,
    format_docs,
    load_multi_representation_documents,
)


def build_summary_retriever():
    parent_documents = load_multi_representation_documents()
    llm = create_llm()
    summarize = (
        {"doc": lambda document: document.page_content}
        | ChatPromptTemplate.from_template(
            "Summarize the following document accurately and concisely:\n\n{doc}"
        )
        | llm
        | StrOutputParser()
    )
    summaries = summarize.batch(
        parent_documents,
        config={"max_concurrency": 5},
    )

    id_key = "doc_id"
    parent_ids = [str(uuid.uuid4()) for _ in parent_documents]
    summary_documents = [
        Document(
            page_content=summary,
            metadata={
                id_key: parent_ids[index],
                "source": parent_documents[index].metadata.get(
                    "source",
                    "Unknown",
                ),
            },
        )
        for index, summary in enumerate(summaries)
    ]

    vectorstore = Chroma(
        collection_name="section_11_summaries",
        embedding_function=create_embeddings(),
    )
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=InMemoryByteStore(),
        id_key=id_key,
        search_kwargs={"k": 4},
    )
    retriever.vectorstore.add_documents(summary_documents)
    retriever.docstore.mset(list(zip(parent_ids, parent_documents)))
    return retriever, llm


def build_summary_rag_chain():
    retriever, llm = build_summary_retriever()
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | create_base_rag_prompt()
        | llm
        | StrOutputParser()
    )


def main() -> None:
    chain = build_summary_rag_chain()
    question = "What are the main challenges in building autonomous agents?"
    print(chain.invoke(question))


if __name__ == "__main__":
    main()
