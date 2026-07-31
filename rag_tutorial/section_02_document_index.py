"""Section 2: load, split, embed, and retrieve source documents."""

from rag_tutorial.shared import build_blog_resources


def main() -> None:
    resources = build_blog_resources("section_02_documents")
    results = resources.retriever.invoke("What is task decomposition?")

    print(f"Loaded documents: {len(resources.documents)}")
    print(f"Indexed chunks: {len(resources.splits)}")
    print(f"Retrieved chunks: {len(results)}")

    for index, document in enumerate(results, start=1):
        preview = document.page_content.replace("\n", " ")[:300]
        print(f"\n{index}. Source: {document.metadata.get('source', 'Unknown')}")
        print(preview)


if __name__ == "__main__":
    main()

