"""Section 6: decompose a question, answer its parts, then synthesize."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from grounded.shared import (
    build_blog_resources,
    create_base_rag_prompt,
    create_llm,
    format_docs,
)

DECOMPOSITION_TEMPLATE = """Break the following question into three independent
sub-questions that can be answered separately. Return one sub-question per line.

Question: {question}"""

SYNTHESIS_TEMPLATE = """Use these question-and-answer pairs to answer the
original question.

{context}

Original question: {question}"""


def format_qa_pairs(questions: list[str], answers: list[str]) -> str:
    pairs = [
        f"Question {index}: {question}\nAnswer {index}: {answer}"
        for index, (question, answer) in enumerate(
            zip(questions, answers),
            start=1,
        )
    ]
    return "\n\n".join(pairs)


def ask_decomposition_rag(question: str) -> str:
    resources = build_blog_resources("section_06_decomposition")
    llm = create_llm()
    generate_questions = (
        ChatPromptTemplate.from_template(DECOMPOSITION_TEMPLATE)
        | llm
        | StrOutputParser()
        | (lambda text: [line.strip() for line in text.splitlines() if line.strip()])
    )
    answer_sub_question = create_base_rag_prompt() | llm | StrOutputParser()

    sub_questions = generate_questions.invoke({"question": question})
    answers = []
    print("Decomposed questions:")
    for index, sub_question in enumerate(sub_questions, start=1):
        print(f"{index}. {sub_question}")
        documents = resources.retriever.invoke(sub_question)
        answers.append(
            answer_sub_question.invoke(
                {
                    "context": format_docs(documents),
                    "question": sub_question,
                }
            )
        )

    synthesis_chain = (
        ChatPromptTemplate.from_template(SYNTHESIS_TEMPLATE)
        | llm
        | StrOutputParser()
    )
    return synthesis_chain.invoke(
        {
            "context": format_qa_pairs(sub_questions, answers),
            "question": question,
        }
    )


def main() -> None:
    question = "What are the main components of an autonomous LLM agent system?"
    print(ask_decomposition_rag(question))


if __name__ == "__main__":
    main()

