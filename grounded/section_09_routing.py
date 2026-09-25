"""Section 9: logical routing and embedding-based semantic routing."""

from typing import Literal

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

from grounded.shared import create_embeddings, create_llm


class RouteQuery(BaseModel):
    """Route a programming question to its documentation source."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="The documentation source most relevant to the question.",
    )


def build_logical_router():
    llm = create_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Route each programming question to Python, JavaScript, or Go docs.",
            ),
            ("human", "{question}"),
        ]
    )
    router = prompt | llm.with_structured_output(RouteQuery)

    def choose_route(result: RouteQuery) -> str:
        routes = {
            "python_docs": "chain for python_docs",
            "js_docs": "chain for js_docs",
            "golang_docs": "chain for golang_docs",
        }
        return routes[result.datasource]

    return router | RunnableLambda(choose_route)


PHYSICS_TEMPLATE = """You are a concise physics professor. If you do not know
the answer, say so.

Question: {query}"""

MATH_TEMPLATE = """You are a mathematician. Break difficult problems into
parts, solve the parts, and combine them.

Question: {query}"""


def cosine_similarity_to_many(query_embedding, prompt_embeddings):
    query = np.asarray(query_embedding)
    prompts = np.asarray(prompt_embeddings)
    denominator = np.linalg.norm(prompts, axis=1) * np.linalg.norm(query)
    if np.any(denominator == 0):
        raise ValueError("Semantic routing received a zero embedding vector.")
    return np.dot(prompts, query) / denominator


def build_semantic_router():
    llm = create_llm()
    embeddings = create_embeddings()
    templates = [PHYSICS_TEMPLATE, MATH_TEMPLATE]
    template_embeddings = embeddings.embed_documents(templates)

    def select_prompt(input_data: dict):
        query_embedding = embeddings.embed_query(input_data["query"])
        similarities = cosine_similarity_to_many(
            query_embedding,
            template_embeddings,
        )
        selected = templates[int(similarities.argmax())]
        print("Using MATH" if selected == MATH_TEMPLATE else "Using PHYSICS")
        return PromptTemplate.from_template(selected)

    return (
        {"query": RunnablePassthrough()}
        | RunnableLambda(select_prompt)
        | llm
        | StrOutputParser()
    )


def main() -> None:
    logical_router = build_logical_router()
    javascript_question = "Why is my JavaScript array map callback failing?"
    print(logical_router.invoke({"question": javascript_question}))

    semantic_router = build_semantic_router()
    print(semantic_router.invoke("What is a black hole?"))


if __name__ == "__main__":
    main()

