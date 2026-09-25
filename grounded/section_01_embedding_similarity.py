"""Section 1: compare two texts with embedding cosine similarity."""

import numpy as np

from grounded.shared import create_embeddings


def cosine_similarity(vector_a, vector_b) -> float:
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return float(np.dot(vector_a, vector_b) / denominator)


def main() -> None:
    embeddings = create_embeddings()
    question_vector = embeddings.embed_query("What kinds of pets do I like?")
    document_vector = embeddings.embed_query("My favorite pet is a cat.")

    print("Embedding length:", len(question_vector))
    print("Cosine similarity:", cosine_similarity(question_vector, document_vector))


if __name__ == "__main__":
    main()

