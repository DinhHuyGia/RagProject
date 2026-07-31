"""Section 10: construct a YouTube transcript query and use it in RAG."""

import datetime

from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field
from pytubefix import YouTube

from rag_tutorial.shared import (
    create_base_rag_prompt,
    create_embeddings,
    create_llm,
    create_text_splitter,
    format_docs,
)

DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=pbAd8O1Lvm4"


class TutorialSearch(BaseModel):
    """Structured search over tutorial-video transcripts and metadata."""

    content_search: str = Field(
        ...,
        description=(
            "Semantic transcript query preserving the user's intent. Do not add "
            "unmentioned products or technologies. Maximum 12 words."
        ),
    )
    title_search: str = Field(
        ...,
        description="Title keywords from the user's question. Maximum 6 words.",
    )
    min_view_count: int | None = Field(
        None,
        description="Inclusive minimum views; only when explicitly requested.",
    )
    max_view_count: int | None = Field(
        None,
        description="Exclusive maximum views; only when explicitly requested.",
    )
    earliest_publish_date: datetime.date | None = Field(
        None,
        description="Inclusive earliest date; only when explicitly requested.",
    )
    latest_publish_date: datetime.date | None = Field(
        None,
        description="Exclusive latest date; only when explicitly requested.",
    )
    min_length_sec: int | None = Field(
        None,
        description="Inclusive minimum seconds; only when explicitly requested.",
    )
    max_length_sec: int | None = Field(
        None,
        description="Exclusive maximum seconds; only when explicitly requested.",
    )

    def pretty_print(self) -> None:
        for name, field_info in type(self).model_fields.items():
            value = getattr(self, name)
            if value is not None and value != field_info.default:
                print(f"{name}: {value}")


def date_to_ordinal(value: datetime.date | datetime.datetime | None) -> int:
    if isinstance(value, datetime.datetime):
        return value.date().toordinal()
    if isinstance(value, datetime.date):
        return value.toordinal()
    return 0


def load_youtube_documents(video_url: str = DEFAULT_VIDEO_URL):
    documents = YoutubeLoader.from_youtube_url(
        video_url,
        add_video_info=False,
    ).load()
    if not documents:
        raise RuntimeError("No YouTube transcript documents were loaded.")

    metadata = {
        "video_url": video_url,
        "title": "Unknown",
        "description": "Unknown",
        "view_count": 0,
        "thumbnail_url": "Unknown",
        "publish_date": "Unknown",
        "publish_date_ordinal": 0,
        "length": 0,
        "author": "Unknown",
    }
    try:
        video = YouTube(video_url)
        publish_date = video.publish_date
        metadata.update(
            {
                "title": video.title or "Unknown",
                "description": video.description or "Unknown",
                "view_count": video.views or 0,
                "thumbnail_url": video.thumbnail_url or "Unknown",
                "publish_date": (
                    publish_date.isoformat() if publish_date else "Unknown"
                ),
                "publish_date_ordinal": date_to_ordinal(publish_date),
                "length": video.length or 0,
                "author": video.author or "Unknown",
            }
        )
    except Exception as error:
        print(f"Warning: enriched YouTube metadata was unavailable: {error}")

    for document in documents:
        document.metadata.update(metadata)
    return documents


def build_chroma_filter(search: TutorialSearch):
    conditions = []
    comparisons = (
        ("view_count", "$gte", search.min_view_count),
        ("view_count", "$lt", search.max_view_count),
        ("length", "$gte", search.min_length_sec),
        ("length", "$lt", search.max_length_sec),
        (
            "publish_date_ordinal",
            "$gte",
            search.earliest_publish_date.toordinal()
            if search.earliest_publish_date
            else None,
        ),
        (
            "publish_date_ordinal",
            "$lt",
            search.latest_publish_date.toordinal()
            if search.latest_publish_date
            else None,
        ),
    )
    for metadata_field, operator, value in comparisons:
        if value is not None:
            conditions.append({metadata_field: {operator: value}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def build_query_analyzer(llm):
    system_message = """Convert the question into a structured tutorial-video
search. Preserve its meaning, do not invent technologies, and only add metadata
filters explicitly requested by the user."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_message), ("human", "{question}")]
    )
    return prompt | llm.with_structured_output(TutorialSearch)


def build_youtube_rag_chain(video_url: str = DEFAULT_VIDEO_URL):
    llm = create_llm()
    analyzer = build_query_analyzer(llm)
    documents = load_youtube_documents(video_url)
    splits = create_text_splitter().split_documents(documents)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=create_embeddings(),
        collection_name="section_10_youtube_tutorials",
    )

    def retrieve(question: str):
        search = analyzer.invoke({"question": question})
        print("Constructed query:")
        search.pretty_print()
        where_filter = build_chroma_filter(search)
        if where_filter:
            return vectorstore.similarity_search(
                search.content_search,
                k=4,
                filter=where_filter,
            )
        return vectorstore.similarity_search(search.content_search, k=4)

    return (
        {
            "context": RunnableLambda(retrieve) | format_docs,
            "question": RunnablePassthrough(),
        }
        | create_base_rag_prompt()
        | llm
        | StrOutputParser()
    )


def main() -> None:
    chain = build_youtube_rag_chain()
    print(chain.invoke("How do I build RAG from scratch?"))


if __name__ == "__main__":
    main()
