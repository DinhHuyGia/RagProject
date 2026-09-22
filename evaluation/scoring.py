import re
from statistics import mean

ANSWERABLE_METRICS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "factual_correctness",
]
UNANSWERABLE_METRICS = ["refusal_correctness"]
REFUSAL_MARKERS = (
    "does not provide",
    "does not state",
    "does not identify",
    "does not specify",
    "does not mention",
    "does not contain",
    "do not provide",
    "not provided",
    "not stated",
    "not identified",
    "not specified",
    "not mentioned",
    "information is missing",
    "information is unavailable",
    "insufficient information",
    "not enough information",
    "cannot determine",
    "could not find",
)


def get_reference(record: dict) -> str:
    """Build a reference from atomic claims when they are available."""
    claims = record.get("reference_claims")

    if claims:
        if not all(
            isinstance(claim, str) and claim.strip()
            for claim in claims
        ):
            raise ValueError(
                f"Invalid reference_claims for record {record['id']}."
            )

        return " ".join(claim.strip() for claim in claims)

    return record["reference"]


def normalize_for_matching(text: str) -> str:
    """Normalize punctuation and whitespace for refusal checks."""
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def score_unanswerable_response(
    response: str,
    record: dict,
) -> tuple[float, bool, float]:
    """Score whether an answer refuses and names the requested topic."""
    normalized_response = normalize_for_matching(response)
    refusal_detected = any(
        marker in normalized_response
        for marker in REFUSAL_MARKERS
    )

    topic_terms = record.get("refusal_topic_terms", [])
    if not topic_terms:
        raise ValueError(
            f"Unanswerable record {record['id']} requires "
            "refusal_topic_terms."
        )

    matched_terms = sum(
        normalize_for_matching(term) in normalized_response
        for term in topic_terms
    )
    topic_coverage = matched_terms / len(topic_terms)
    score = float(refusal_detected and topic_coverage == 1.0)

    return score, refusal_detected, topic_coverage


def summarize_metrics(
    results: list[dict],
    metric_names: list[str],
) -> tuple[dict[str, float | None], dict[str, int]]:
    """Average each metric over rows where that metric applies."""
    summary: dict[str, float | None] = {}
    sample_counts: dict[str, int] = {}

    for metric in metric_names:
        values = [
            row[metric]
            for row in results
            if row[metric] is not None
        ]
        summary[metric] = mean(values) if values else None
        sample_counts[metric] = len(values)

    return summary, sample_counts


def summarize_by_answerability(
    results: list[dict],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Keep answer-quality metrics separate from refusal correctness."""
    answerable_results = [
        row for row in results if row["answerable"]
    ]
    unanswerable_results = [
        row for row in results if not row["answerable"]
    ]

    answerable_summary, answerable_counts = summarize_metrics(
        answerable_results,
        ANSWERABLE_METRICS,
    )
    unanswerable_summary, unanswerable_counts = summarize_metrics(
        unanswerable_results,
        UNANSWERABLE_METRICS,
    )

    return (
        {
            "answerable": answerable_summary,
            "unanswerable": unanswerable_summary,
        },
        {
            "answerable": answerable_counts,
            "unanswerable": unanswerable_counts,
        },
    )
