import re
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from rag_tutorial.section_05_rag_fusion import reciprocal_rank_fusion
from rag_tutorial.shared import create_llm, format_docs

StrategyName = Literal[
    "simple",
    "multi_query",
    "decomposition",
    "step_back",
    "hyde",
]

RequestedStrategy = Literal[
    "auto",
    "simple",
    "multi_query",
    "decomposition",
    "step_back",
    "hyde",
]

IndexName = Literal["chunks"]

class RouteDecision(BaseModel):
    strategy: StrategyName
    reason: str

class SourceReference(BaseModel):
    document_id: str | None
    filename: str
    page_number: int | None

class AdaptiveAnswer(BaseModel):
    question: str
    answer: str
    strategy: StrategyName
    indexes: list[IndexName]
    routing_reason: str
    retrieval_queries: list[str]
    sources: list[SourceReference]

STRATEGY_PROMPTS = {
    "multi_query": """
Generate four different search queries that preserve the original intent.
Return one query per line.

Question: {question}
""",
    "decomposition": """
Break the question into three independently searchable sub-questions.
Return one sub-question per line.

Question: {question}
""",
    "step_back": """
Write one broader background question that would help answer the original.
Return only the question.

Question: {question}
""",
    "hyde": """
Write a short hypothetical passage that directly answers the question.
The passage will only be used for semantic retrieval.

Question: {question}
""",
}

ROUTER_PROMPT = """
Choose the best retrieval strategy.

Strategies:
- simple: direct and specific questions
- multi_query: ambiguous questions needing multiple phrasings
- decomposition: multi-part or comparison questions
- step_back: narrow questions needing broader background
- hyde: questions whose wording may not occur in the documents

Question: {question}
"""

class AdaptiveRAG:
    def __init__(self, chunk_retriever: Any):
        self.llm = create_llm()
        self.chunk_retriever = chunk_retriever

        self.router = (
            ChatPromptTemplate.from_template(ROUTER_PROMPT)
            | self.llm.with_structured_output(RouteDecision)
        )

        self.transformers = {
            name: (
                ChatPromptTemplate.from_template(template)
                | self.llm
                | StrOutputParser()
            )
            for name, template in STRATEGY_PROMPTS.items()
        }

        answer_prompt = ChatPromptTemplate.from_template("""
    You are a concise document question-answering assistant.

Answer the user's exact question using only the retrieved context.

Requirements:
- Begin with the direct answer.
- Address every part of the question.
- Answer only what the user asked.
- Use the fewest sentences necessary.
- If one sentence fully answers the question, stop after that sentence.
- Include only facts needed to answer the question.
- Do not summarize unrelated parts of the context.
- Do not add supporting details or general background unless the question
  explicitly asks for them.
- Prefer one to three concise sentences.
- For multi-part questions, answer each part clearly.
- If the context does not contain enough information, say exactly
  what requested information the document does not provide, then stop.
- Do not use outside knowledge.

Question:
{question}

Retrieved context:
{context}

Direct answer:
        """
        )

        self.answer_chain = (
            answer_prompt
            | self.llm
            | StrOutputParser()
        )


    def get_retriever(self, index_name: IndexName):
        if index_name != "chunks":
            raise ValueError(
                "The document API only supports the uploaded-document "
                "chunks index."
            )
        return self.chunk_retriever

    @staticmethod
    def normalize_indexes(
        indexes: list[IndexName] | None,
    ) -> list[IndexName]:
        if indexes is None:
            return ["chunks"]
        if indexes != ["chunks"]:
            raise ValueError(
                "The document API only supports indexes=['chunks']."
            )
        return ["chunks"]
    
    def build_queries(
        self,
        question: str,
        strategy: StrategyName,
    ) -> list[str]:
        if strategy == "simple":
            return [question]

        generated = self.transformers[strategy].invoke(
            {"question": question}
        ).strip()

        if strategy in {"step_back", "hyde"}:
            transformed = [generated]
        else:
            transformed = []
            for line in generated.splitlines():
                cleaned = re.sub(
                    r"^\s*(?:[-*]|\d+[.)])\s*",
                    "",
                    line,
                ).strip()

                if cleaned:
                    transformed.append(cleaned)

        return list(dict.fromkeys([question, *transformed]))

    def retrieve(
        self,
        queries: list[str],
        indexes: list[IndexName],
        document_id: str | None = None,
    ) -> list[Document]:
        ranked_results = []

        for index_name in indexes:
            retriever = self.get_retriever(index_name)

            if document_id is not None:
                vectorstore = getattr(retriever, "vectorstore", None)

                if vectorstore is not None:
                    search_kwargs = dict(
                        getattr(retriever, "search_kwargs", {})
                    )
                    existing_filter = search_kwargs.get("filter")
                    document_filter = {"document_id": document_id}

                    if existing_filter:
                        search_kwargs["filter"] = {
                            "$and": [existing_filter, document_filter]
                        }
                    else:
                        search_kwargs["filter"] = document_filter

                    retriever = vectorstore.as_retriever(
                        search_type=getattr(
                            retriever,
                            "search_type",
                            "similarity",
                        ),
                        search_kwargs=search_kwargs,
                    )
    
            for query in queries:
                documents = retriever.invoke(query)

                # Keep filtering strict even for retriever implementations
                # that cannot push a metadata filter into their vector store.
                if document_id is not None:
                    documents = [
                        document
                        for document in documents
                        if document.metadata.get("document_id")
                        == document_id
                    ]

                ranked_results.append(documents)

        return reciprocal_rank_fusion(ranked_results)[:8]
    
    def invoke(
        self,
        question: str,
        strategy: RequestedStrategy = "auto",
        indexes: list[IndexName] | None = None,
    ) -> AdaptiveAnswer:
        resolved_indexes = self.normalize_indexes(indexes)

        if strategy == "auto":
            decision = self.router.invoke({"question": question})
        else:
            decision = RouteDecision(
                strategy=strategy,
                reason="The API caller selected this strategy.",
            )

        queries = self.build_queries(
            question,
            decision.strategy,
        )

        documents = self.retrieve(
            queries,
            resolved_indexes,
        )

        answer = self.answer_chain.invoke(
            {
                "question": question,
                "context": format_docs(documents),
            }
        )

        source_map = {}

        for document in documents:
            metadata = document.metadata

            key = (
                metadata.get("document_id"),
                metadata.get("page_number"),
            )

            source_map[key] = SourceReference(
                document_id=metadata.get("document_id"),
                filename=str(metadata.get("filename", "Unknown")),
                page_number=metadata.get("page_number"),
            )

        sources = list(source_map.values())

        return AdaptiveAnswer(
            question=question,
            answer=answer,
            strategy=decision.strategy,
            indexes=resolved_indexes,
            routing_reason=decision.reason,
            retrieval_queries=queries,
            sources=sources,
        )



