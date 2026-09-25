"""Section 11b: search atomic propositions and return original chunks."""

import uuid

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.stores import InMemoryByteStore
from pydantic import BaseModel, Field

from grounded.shared import (
    create_base_rag_prompt,
    create_embeddings,
    create_llm,
    format_docs,
    load_multi_representation_documents,
)


class PropositionList(BaseModel):
    """Atomic, self-contained facts extracted from one parent chunk."""

    propositions: list[str] = Field(
        ...,
        description=(
            "Five to fifteen self-contained factual statements. Each statement "
            "must express one fact and contain no unsupported information."
        ),
    )


PROPOSITION_TEMPLATE = """Extract 5 to 15 useful propositions from this document.

Requirements:
- Give each proposition exactly one main fact.
- Make every proposition understandable without surrounding context.
- Replace vague pronouns with their specific subject.
- Preserve important names, dates, numbers, and technical terms.
- Do not add unsupported information or duplicate propositions.

Document:

{doc}"""


def build_proposition_retriever():
    parent_documents = load_multi_representation_documents()
    llm = create_llm()
    extract_propositions = (
        {"doc": lambda document: document.page_content}
        | ChatPromptTemplate.from_template(PROPOSITION_TEMPLATE)
        | llm.with_structured_output(PropositionList)
    )
    results = extract_propositions.batch(
        parent_documents,
        config={"max_concurrency": 5},
    )

    id_key = "doc_id"
    parent_ids = [str(uuid.uuid4()) for _ in parent_documents]
    proposition_documents = []
    for parent_id, parent_document, result in zip(
        parent_ids,
        parent_documents,
        results,
    ):
        for proposition in result.propositions:
            proposition_documents.append(
                Document(
                    page_content=proposition,
                    metadata={
                        id_key: parent_id,
                        "source": parent_document.metadata.get(
                            "source",
                            "Unknown",
                        ),
                    },
                )
            )

    if not proposition_documents:
        raise RuntimeError("The LLM did not generate any propositions.")

    vectorstore = Chroma(
        collection_name="section_11b_propositions",
        embedding_function=create_embeddings(),
    )
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=InMemoryByteStore(),
        id_key=id_key,
        search_kwargs={"k": 8},
    )
    retriever.vectorstore.add_documents(proposition_documents)
    retriever.docstore.mset(list(zip(parent_ids, parent_documents)))
    return retriever, llm


def build_proposition_rag_chain():
    retriever, llm = build_proposition_retriever()
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
    chain = build_proposition_rag_chain()
    question = "What type of memory persists across agent sessions?"
    print(chain.invoke(question))


if __name__ == "__main__":
    main()
