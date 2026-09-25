"""Lazy shared setup for the independently runnable RAG examples."""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Web loaders use this header when requesting public source pages.
os.environ.setdefault("USER_AGENT", "Grounded/1.0")


DEFAULT_EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
DEFAULT_CHAT_DEPLOYMENT = "gpt-5-mini"

BLOG_URLS = (
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/",
)
LANGCHAIN_DOCS_URL = (
    "https://python.langchain.com/docs/use_cases/query_analysis/techniques/"
    "routing#routing-to-multiple-indexes"
)

BASE_RAG_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""


@dataclass
class BlogRagResources:
    """Documents and retrieval objects shared by the blog-based examples."""

    documents: list[Document]
    splits: list[Document]
    vectorstore: Chroma
    retriever: Any


@lru_cache(maxsize=1)
def get_api_key() -> str:
    """Read an Azure/OpenAI key without storing it in source code."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your Azure OpenAI / Foundry API key: ")
    if not api_key:
        raise RuntimeError("An Azure OpenAI API key is required.")
    return api_key


def get_base_url() -> str:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    if not endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT environment variable is required."
        )
    return endpoint.rstrip("/") + "/openai/v1/"


def create_embeddings() -> OpenAIEmbeddings:
    """Create the configured embedding client only when requested."""
    return OpenAIEmbeddings(
        model=os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            DEFAULT_EMBEDDING_DEPLOYMENT,
        ),
        base_url=get_base_url(),
        api_key=get_api_key(),
    )


def create_llm() -> ChatOpenAI:
    """Create the configured chat model only when requested."""
    return ChatOpenAI(
        model=os.getenv(
            "AZURE_OPENAI_CHAT_DEPLOYMENT",
            DEFAULT_CHAT_DEPLOYMENT,
        ),
        base_url=get_base_url(),
        api_key=get_api_key(),
        temperature=0,
    )


def create_text_splitter(
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def load_blog_documents(include_langchain_docs: bool = True) -> list[Document]:
    """Load blog pages while applying HTML filters only where appropriate."""
    blog_loader = WebBaseLoader(
        web_paths=BLOG_URLS,
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        },
    )
    documents = blog_loader.load()

    if include_langchain_docs:
        docs_loader = WebBaseLoader(
            LANGCHAIN_DOCS_URL,
            continue_on_failure=True,
        )
        documents.extend(docs_loader.load())

    documents = [doc for doc in documents if doc.page_content.strip()]
    if not documents:
        raise RuntimeError("No source documents were loaded.")
    return documents


def build_blog_resources(
    collection_name: str,
    *,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    search_k: int = 4,
) -> BlogRagResources:
    """Load, split, embed, and index the shared blog sources."""
    documents = load_blog_documents()
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    splits = splitter.split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
    return BlogRagResources(documents, splits, vectorstore, retriever)


def load_multi_representation_documents(
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Load moderately sized parent chunks for summary/proposition indexes."""
    documents = load_blog_documents()
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(documents)


def format_docs(documents: list[Document]) -> str:
    return "\n\n".join(document.page_content for document in documents)


def create_base_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(BASE_RAG_TEMPLATE)

def create_document_vectorstore() -> Chroma:
    persist_directory = Path("data/chroma")
    persist_directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name="uploaded_documents",
        embedding_function=create_embeddings(),
        persist_directory=str(persist_directory),
    )
