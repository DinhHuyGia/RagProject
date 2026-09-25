"""Section 7: retrieve for both the original and a broader step-back question."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda

from grounded.shared import build_blog_resources, create_llm, format_docs

STEP_BACK_EXAMPLES = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "What can the members of The Police do?",
    },
    {
        "input": "Jan Sindel was born in what country?",
        "output": "What is Jan Sindel's personal history?",
    },
]

RESPONSE_TEMPLATE = """Answer the question using the relevant supplied context.
Do not contradict the context. Ignore context that is unrelated.

# Original-question context
{normal_context}

# Step-back context
{step_back_context}

# Original question
{question}
"""


def build_step_back_chain():
    resources = build_blog_resources("section_07_step_back")
    llm = create_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=STEP_BACK_EXAMPLES,
    )
    step_back_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Create a broader step-back question that is easier to answer.",
            ),
            few_shot_prompt,
            ("human", "{question}"),
        ]
    )
    generate_step_back = step_back_prompt | llm | StrOutputParser()

    def retrieve_step_back(input_data: dict):
        question = generate_step_back.invoke(input_data)
        print("Step-back question:", question)
        return resources.retriever.invoke(question)

    return (
        {
            "normal_context": (
                RunnableLambda(lambda data: data["question"])
                | resources.retriever
                | format_docs
            ),
            "step_back_context": RunnableLambda(retrieve_step_back) | format_docs,
            "question": RunnableLambda(lambda data: data["question"]),
        }
        | ChatPromptTemplate.from_template(RESPONSE_TEMPLATE)
        | llm
        | StrOutputParser()
    )


def main() -> None:
    chain = build_step_back_chain()
    question = "What is task decomposition for LLM agents?"
    print(chain.invoke({"question": question}))


if __name__ == "__main__":
    main()

