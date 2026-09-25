"""Command-line launcher for standalone retrieval examples."""

import argparse
import runpy

SECTIONS = {
    "1": "grounded.section_01_embedding_similarity",
    "2": "grounded.section_02_document_index",
    "3": "grounded.section_03_simple_rag",
    "4": "grounded.section_04_multi_query",
    "5": "grounded.section_05_rag_fusion",
    "6": "grounded.section_06_decomposition",
    "7": "grounded.section_07_step_back",
    "8": "grounded.section_08_hyde",
    "9": "grounded.section_09_routing",
    "10": "grounded.section_10_query_construction",
    "11": "grounded.section_11_summary_indexing",
    "11b": "grounded.section_11b_proposition_indexing",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a standalone Grounded retrieval example."
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

