"""Command-line launcher for the independently runnable RAG tutorial sections."""

import argparse
import runpy

SECTIONS = {
    "1": "rag_tutorial.section_01_embedding_similarity",
    "2": "rag_tutorial.section_02_document_index",
    "3": "rag_tutorial.section_03_simple_rag",
    "4": "rag_tutorial.section_04_multi_query",
    "5": "rag_tutorial.section_05_rag_fusion",
    "6": "rag_tutorial.section_06_decomposition",
    "7": "rag_tutorial.section_07_step_back",
    "8": "rag_tutorial.section_08_hyde",
    "9": "rag_tutorial.section_09_routing",
    "10": "rag_tutorial.section_10_query_construction",
    "11": "rag_tutorial.section_11_summary_indexing",
    "11b": "rag_tutorial.section_11b_proposition_indexing",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one isolated section of the RAG code-along."
    )
    parser.add_argument(
        "section",
        choices=SECTIONS,
        help="Section number to run (1-11 or 11b).",
    )
    args = parser.parse_args()
    runpy.run_module(SECTIONS[args.section], run_name="__main__")


if __name__ == "__main__":
    main()

