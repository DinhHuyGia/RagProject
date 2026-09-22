import pytest
from langchain_core.documents import Document

from rag_tutorial.langgraph_rag import (
    AvailableSource,
    ContextGrade,
    EvidenceGrade,
    EvidenceTask,
    LangGraphRAG,
    RetrievalPlan,
    RetrievalScope,
)


def make_plan(
    question: str,
    *,
    source_hint: str | None = None,
) -> RetrievalPlan:
    return RetrievalPlan(
        tasks=[
            EvidenceTask(
                subquestion=question,
                search_query=question,
                source_hint=source_hint,
            )
        ],
        requires_multiple_sources=False,
        reason="The question requires one evidence task.",
    )


class FakeRetrievalPlanner:
    def __init__(self) -> None:
        self.result: RetrievalPlan | None = None
        self.last_input: dict[str, str] | None = None

    def invoke(self, input_data: dict[str, str]) -> RetrievalPlan:
        self.last_input = input_data
        return self.result or make_plan(input_data["question"])


class FakeContextGrader:
    def __init__(self) -> None:
        self.result = ContextGrade(
            sufficient=True,
            reason="The context contains the required fact.",
        )
        self.last_input: dict[str, str] | None = None

    def invoke(self, input_data: dict[str, str]) -> ContextGrade:
        self.last_input = input_data
        return self.result


class FakeEvidenceTaskGrader:
    def __init__(self) -> None:
        self.result = EvidenceGrade(
            sufficient=True,
            reason="The evidence directly supports the subquestion.",
        )
        self.calls: list[dict[str, str]] = []

    def invoke(self, input_data: dict[str, str]) -> EvidenceGrade:
        self.calls.append(input_data)
        return self.result


class FakeQueryRewriter:
    def __init__(self) -> None:
        self.result = (
            "Apple homepage product sections image headline links"
        )
        self.last_input: dict[str, str] | None = None

    def invoke(self, input_data: dict[str, str]) -> str:
        self.last_input = input_data
        return self.result


class FakeTextChain:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[dict[str, str]] = []

    def invoke(self, input_data: dict[str, str]) -> str:
        self.calls.append(input_data)
        return self.result


class FakeRAGComponents:
    def __init__(self) -> None:
        self.retrieval_call: tuple[list[str], list[str]] | None = None
        self.retrieval_calls: list[
            tuple[list[str], list[str], str | None]
        ] = []
        self.answer_chain = FakeTextChain(
            "The Macintosh made computers easier to use."
        )

    def build_queries(self, question: str, strategy: str) -> list[str]:
        return [question, f"{strategy}: transformed query"]

    def normalize_indexes(
        self,
        indexes: list[str] | None,
    ) -> list[str]:
        return indexes or ["chunks"]

    def retrieve(
        self,
        queries: list[str],
        indexes: list[str],
        document_id: str | None = None,
    ) -> list[Document]:
        self.retrieval_call = (queries, indexes)
        self.retrieval_calls.append((queries, indexes, document_id))
        resolved_document_id = document_id or "website-document"
        return [
            Document(
                page_content="The Macintosh popularized the GUI.",
                metadata={
                    "page_number": 3,
                    "document_id": resolved_document_id,
                    "chunk_id": f"{resolved_document_id}:0",
                },
            )
        ]


def create_graph_rag_without_model() -> LangGraphRAG:
    graph_rag = object.__new__(LangGraphRAG)
    graph_rag.components = FakeRAGComponents()
    graph_rag.source_catalog_override = []
    graph_rag.retrieval_planner = FakeRetrievalPlanner()
    graph_rag.evidence_task_grader = FakeEvidenceTaskGrader()
    graph_rag.context_grader = FakeContextGrader()
    graph_rag.query_rewriter = FakeQueryRewriter()
    graph_rag.refusal_chain = FakeTextChain(
        "The documents do not provide the requested information."
    )
    return graph_rag


def test_plan_retrieval_creates_source_aware_tasks() -> None:
    graph_rag = create_graph_rag_without_model()
    question = "Compare the website analysis with the 2025 10-K."
    available_sources = [
        AvailableSource(
            document_id="website-document",
            filename="apple_website_analysis.pdf",
        ),
        AvailableSource(
            document_id="10k-document",
            filename="Apple 10K.pdf",
        ),
    ]
    graph_rag.source_catalog_override = available_sources
    plan = RetrievalPlan(
        tasks=[
            EvidenceTask(
                subquestion="How does the website describe the ecosystem?",
                search_query="website ecosystem devices work together",
                source_hint="Apple website analysis",
                source_document_id="website-document",
            ),
            EvidenceTask(
                subquestion="How does the 10-K describe cloud services?",
                search_query="cloud services content across devices",
                source_hint="Apple 2025 Form 10-K",
                source_document_id="10k-document",
            ),
        ],
        requires_multiple_sources=True,
        reason="The comparison explicitly names two sources.",
    )
    graph_rag.retrieval_planner.result = plan

    update = graph_rag.plan_retrieval({"question": question})

    assert update["available_sources"] == available_sources
    planned_tasks = update["retrieval_plan"].tasks
    assert [task.task_id for task in planned_tasks] == [
        "evidence-1",
        "evidence-2",
    ]
    assert all(task.status == "pending" for task in planned_tasks)
    assert all(task.attempts == 0 for task in planned_tasks)
    assert graph_rag.retrieval_planner.last_input == {
        "question": question,
        "available_sources": (
            "- document_id=website-document; "
            "filename=apple_website_analysis.pdf\n"
            "- document_id=10k-document; filename=Apple 10K.pdf"
        ),
    }


def test_plan_retrieval_rejects_unknown_document_id() -> None:
    graph_rag = create_graph_rag_without_model()
    graph_rag.source_catalog_override = [
        AvailableSource(
            document_id="website-document",
            filename="apple_website_analysis.pdf",
        )
    ]
    graph_rag.retrieval_planner.result = RetrievalPlan(
        tasks=[
            EvidenceTask(
                subquestion="Find the website claim.",
                search_query="website claim",
                source_hint="Imaginary source",
                source_document_id="invented-document",
            )
        ],
        requires_multiple_sources=False,
        reason="One source is required.",
    )

    with pytest.raises(ValueError, match="unknown document ID"):
        graph_rag.plan_retrieval(
            {"question": "What does the website say?"}
        )


def test_build_queries_node_updates_retrieval_queries() -> None:
    graph_rag = create_graph_rag_without_model()
    question = "Why was the Macintosh important?"

    update = graph_rag.build_queries(
        {
            "question": question,
            "strategy": "multi_query",
            "retrieval_plan": make_plan(question),
        }
    )

    assert update["retrieval_queries"] == [
        "Why was the Macintosh important?",
        "multi_query: transformed query",
    ]
    assert update["retrieval_scopes"] == [
        RetrievalScope(query="Why was the Macintosh important?"),
        RetrievalScope(query="multi_query: transformed query"),
    ]


def test_build_queries_node_requires_route_strategy() -> None:
    graph_rag = create_graph_rag_without_model()
    question = "Why was the Macintosh important?"

    with pytest.raises(
        ValueError,
        match="route_question node must set strategy",
    ):
        graph_rag.build_queries(
            {
                "question": question,
                "retrieval_plan": make_plan(question),
            }
        )


def test_build_queries_uses_planned_source_hints() -> None:
    graph_rag = create_graph_rag_without_model()
    question = "How do the two documents describe the ecosystem?"
    plan = RetrievalPlan(
        tasks=[
            EvidenceTask(
                subquestion="What does the website analysis say?",
                search_query="devices work seamlessly together",
                source_hint="Apple website analysis",
                source_document_id="website-document",
            ),
            EvidenceTask(
                subquestion="What does the 10-K say?",
                search_query="cloud services content across devices",
                source_hint="Apple 2025 Form 10-K",
                source_document_id="10k-document",
            ),
        ],
        requires_multiple_sources=True,
        reason="Two sources are required.",
    )

    update = graph_rag.build_queries(
        {
            "question": question,
            "strategy": "decomposition",
            "retrieval_plan": plan,
        }
    )

    assert update["retrieval_queries"] == [
        question,
        "devices work seamlessly together",
        "cloud services content across devices",
    ]
    assert update["retrieval_scopes"] == [
        RetrievalScope(query=question),
        RetrievalScope(
            query="devices work seamlessly together",
            source_document_id="website-document",
        ),
        RetrievalScope(
            query="cloud services content across devices",
            source_document_id="10k-document",
        ),
    ]


def test_route_question_uses_decomposition_for_multi_task_plan() -> None:
    graph_rag = create_graph_rag_without_model()
    plan = RetrievalPlan(
        tasks=[
            EvidenceTask(
                subquestion="Find the website claim.",
                search_query="website ecosystem claim",
                source_hint="Apple website analysis",
            ),
            EvidenceTask(
                subquestion="Find the 10-K claim.",
                search_query="10-K cloud services claim",
                source_hint="Apple 2025 Form 10-K",
            ),
        ],
        requires_multiple_sources=True,
        reason="The question compares two sources.",
    )

    update = graph_rag.route_question(
        {
            "question": "Compare the website analysis and 10-K.",
            "requested_strategy": "auto",
            "retrieval_plan": plan,
        }
    )

    assert update == {
        "strategy": "decomposition",
        "routing_reason": (
            "The retrieval plan contains multiple evidence tasks, "
            "so the graph selected decomposition."
        ),
    }


def test_retrieve_node_updates_documents_and_default_indexes() -> None:
    graph_rag = create_graph_rag_without_model()
    question = "Why was the Macintosh important?"
    queries = [
        question,
        "Macintosh impact on everyday users",
    ]
    plan = make_plan(question)
    plan = plan.model_copy(
        update={
            "tasks": [
                plan.tasks[0].model_copy(
                    update={"task_id": "evidence-1"}
                )
            ]
        }
    )

    update = graph_rag.retrieve(
        {
            "question": question,
            "retrieval_plan": plan,
            "retrieval_queries": queries,
            "retrieval_scopes": [
                RetrievalScope(query=queries[0]),
                RetrievalScope(
                        query=queries[1],
                        source_document_id="website-document",
                        task_id="evidence-1",
                    ),
            ],
        }
    )

    assert update["indexes"] == ["chunks"]
    assert len(update["documents"]) == 1
    assert update["documents"][0].metadata["page_number"] == 3
    assert graph_rag.components.retrieval_calls == [
        ([queries[0]], ["chunks"], None),
        ([queries[1]], ["chunks"], "website-document"),
    ]
    evidence_task = update["retrieval_plan"].tasks[0]
    assert evidence_task.status == "retrieved"
    assert evidence_task.attempts == 1
    assert evidence_task.evidence_chunk_ids == ["website-document:0"]
    assert list(update["task_evidence"]) == ["evidence-1"]


def test_retrieve_node_requires_queries() -> None:
    graph_rag = create_graph_rag_without_model()

    with pytest.raises(
        ValueError,
        match="build_queries node must set retrieval queries and scopes",
    ):
        graph_rag.retrieve(
            {"question": "Why was the Macintosh important?"}
        )


def test_grade_evidence_task_marks_supported_task_satisfied() -> None:
    graph_rag = create_graph_rag_without_model()
    question = "Why was the Macintosh important?"
    plan = make_plan(question)
    plan = plan.model_copy(
        update={
            "tasks": [
                plan.tasks[0].model_copy(
                    update={
                        "task_id": "evidence-1",
                        "status": "retrieved",
                        "attempts": 1,
                    }
                )
            ]
        }
    )
    document = Document(
        page_content="The Macintosh popularized graphical interfaces."
    )

    update = graph_rag.grade_evidence_task(
        {
            "question": question,
            "retrieval_plan": plan,
            "task_evidence": {"evidence-1": [document]},
        }
    )

    graded_task = update["retrieval_plan"].tasks[0]
    assert graded_task.status == "satisfied"
    assert graded_task.attempts == 1
    assert graded_task.grade_reason == (
        "The evidence directly supports the subquestion."
    )
    assert update["evidence_tasks_graded"] is True
    assert update["all_evidence_tasks_satisfied"] is True
    assert update["evidence_grade_reason"] == (
        "Every evidence task is sufficiently supported."
    )
    assert graph_rag.evidence_task_grader.calls == [
        {
            "subquestion": question,
            "source": "any relevant source",
            "evidence": document.page_content,
        }
    ]


def test_grade_evidence_task_marks_missing_evidence_insufficient() -> None:
    graph_rag = create_graph_rag_without_model()
    question = "What were Apple's 2025 Services net sales?"
    plan = make_plan(question)
    plan = plan.model_copy(
        update={
            "tasks": [
                plan.tasks[0].model_copy(
                    update={"task_id": "evidence-1"}
                )
            ]
        }
    )

    update = graph_rag.grade_evidence_task(
        {
            "question": question,
            "retrieval_plan": plan,
            "task_evidence": {"evidence-1": []},
        }
    )

    graded_task = update["retrieval_plan"].tasks[0]
    assert graded_task.status == "insufficient"
    assert graded_task.grade_reason == (
        "No evidence was retrieved for this task."
    )
    assert update["all_evidence_tasks_satisfied"] is False
    assert update["evidence_grade_reason"] == (
        "evidence-1: No evidence was retrieved for this task."
    )
    assert graph_rag.evidence_task_grader.calls == []


def test_route_after_evidence_grading_selects_targeted_retry() -> None:
    plan = RetrievalPlan(
        tasks=[
            EvidenceTask(
                task_id="evidence-1",
                subquestion="Find the missing sales amount.",
                search_query="Services sales",
                status="insufficient",
                attempts=1,
                grade_reason="The amount is missing.",
            )
        ],
        requires_multiple_sources=False,
        reason="One fact is required.",
    )

    next_node = LangGraphRAG.route_after_evidence_grading(
        {
            "question": "What were Services sales?",
            "retrieval_plan": plan,
            "evidence_tasks_graded": True,
            "all_evidence_tasks_satisfied": False,
        }
    )

    assert next_node == "rewrite_task_queries"


def test_route_after_evidence_grading_continues_when_satisfied() -> None:
    next_node = LangGraphRAG.route_after_evidence_grading(
        {
            "question": "What were Services sales?",
            "evidence_tasks_graded": True,
            "all_evidence_tasks_satisfied": True,
        }
    )

    assert next_node == "grade_context"


def test_rewrite_task_queries_only_schedules_insufficient_tasks() -> None:
    graph_rag = create_graph_rag_without_model()
    plan = RetrievalPlan(
        tasks=[
            EvidenceTask(
                task_id="evidence-1",
                subquestion="Find the website ecosystem description.",
                search_query="devices work together",
                source_document_id="website-document",
                status="satisfied",
                attempts=1,
                grade_reason="The website evidence is sufficient.",
            ),
            EvidenceTask(
                task_id="evidence-2",
                subquestion="Find the 10-K cloud-services description.",
                search_query="cloud services",
                source_document_id="10k-document",
                status="insufficient",
                attempts=1,
                grade_reason="Cross-device availability is missing.",
            ),
        ],
        requires_multiple_sources=True,
        reason="The question requires two sources.",
    )

    update = graph_rag.rewrite_task_queries(
        {
            "question": "Compare the ecosystem descriptions.",
            "retrieval_plan": plan,
            "retry_count": 0,
        }
    )

    updated_tasks = update["retrieval_plan"].tasks
    assert updated_tasks[0].status == "satisfied"
    assert updated_tasks[0].search_query == "devices work together"
    assert updated_tasks[1].status == "pending"
    assert updated_tasks[1].search_query == (
        "Apple homepage product sections image headline links"
    )
    assert update["retrieval_scopes"] == [
        RetrievalScope(
            query="Apple homepage product sections image headline links",
            source_document_id="10k-document",
            task_id="evidence-2",
        )
    ]
    assert update["retry_count"] == 1
    assert graph_rag.query_rewriter.last_input == {
        "question": "Find the 10-K cloud-services description.",
        "previous_queries": "cloud services",
        "context_grade_reason": (
            "Cross-device availability is missing."
        ),
    }


def test_targeted_retrieval_preserves_satisfied_task_evidence() -> None:
    graph_rag = create_graph_rag_without_model()
    website_document = Document(
        page_content="Apple devices work seamlessly together.",
        metadata={
            "document_id": "website-document",
            "chunk_id": "website-document:0",
        },
    )
    plan = RetrievalPlan(
        tasks=[
            EvidenceTask(
                task_id="evidence-1",
                subquestion="Find the website ecosystem description.",
                search_query="devices work together",
                source_document_id="website-document",
                status="satisfied",
                attempts=1,
                evidence_chunk_ids=["website-document:0"],
            ),
            EvidenceTask(
                task_id="evidence-2",
                subquestion="Find the 10-K cloud-services description.",
                search_query="cloud services across devices",
                source_document_id="10k-document",
                status="pending",
                attempts=1,
            ),
        ],
        requires_multiple_sources=True,
        reason="Two sources are required.",
    )
    scope = RetrievalScope(
        query="cloud services across devices",
        source_document_id="10k-document",
        task_id="evidence-2",
    )

    update = graph_rag.retrieve(
        {
            "question": "Compare the ecosystem descriptions.",
            "retrieval_plan": plan,
            "retrieval_queries": [scope.query],
            "retrieval_scopes": [scope],
            "task_evidence": {"evidence-1": [website_document]},
        }
    )

    assert graph_rag.components.retrieval_calls == [
        ([scope.query], ["chunks"], "10k-document")
    ]
    assert {
        document.metadata["document_id"]
        for document in update["documents"]
    } == {"website-document", "10k-document"}
    updated_tasks = update["retrieval_plan"].tasks
    assert updated_tasks[0].status == "satisfied"
    assert updated_tasks[0].attempts == 1
    assert updated_tasks[1].status == "retrieved"
    assert updated_tasks[1].attempts == 2


def test_exhausted_evidence_task_routes_to_failure() -> None:
    plan = RetrievalPlan(
        tasks=[
            EvidenceTask(
                task_id="evidence-1",
                subquestion="Find the missing sales amount.",
                search_query="Services sales",
                status="insufficient",
                attempts=3,
                grade_reason="The amount was not found.",
            )
        ],
        requires_multiple_sources=False,
        reason="One fact is required.",
    )

    next_node = LangGraphRAG.route_after_evidence_grading(
        {
            "question": "What were Services sales?",
            "retrieval_plan": plan,
            "evidence_tasks_graded": True,
            "all_evidence_tasks_satisfied": False,
        }
    )

    assert next_node == "fail_evidence_tasks"


def test_grade_context_stops_when_an_evidence_task_failed() -> None:
    graph_rag = create_graph_rag_without_model()
    question = "What were Apple's 2025 Services net sales?"
    plan = make_plan(question)
    plan = plan.model_copy(
        update={
            "tasks": [
                plan.tasks[0].model_copy(
                    update={
                        "task_id": "evidence-1",
                        "status": "insufficient",
                        "grade_reason": "The sales amount is missing.",
                    }
                )
            ]
        }
    )

    update = graph_rag.grade_context(
        {
            "question": question,
            "retrieval_plan": plan,
            "documents": [Document(page_content="Related evidence")],
            "evidence_tasks_graded": True,
            "all_evidence_tasks_satisfied": False,
            "evidence_grade_reason": (
                "evidence-1: The sales amount is missing."
            ),
        }
    )

    assert update["context_sufficient"] is False
    assert update["context_grade_reason"] == (
        "evidence-1: The sales amount is missing."
    )
    assert graph_rag.context_grader.last_input is None


def test_grade_context_node_records_structured_decision() -> None:
    graph_rag = create_graph_rag_without_model()
    question = "Why was the Macintosh important?"
    document = Document(
        page_content=(
            "The Macintosh popularized graphical user interfaces."
        )
    )

    update = graph_rag.grade_context(
        {
            "question": question,
            "retrieval_plan": make_plan(question),
            "documents": [document],
        }
    )

    assert update["context_sufficient"] is True
    assert update["context_grade_reason"] == (
        "The context contains the required fact."
    )
    graded_task = update["retrieval_plan"].tasks[0]
    assert graded_task.status == "satisfied"
    assert graded_task.grade_reason == (
        "The context contains the required fact."
    )
    assert graph_rag.context_grader.last_input == {
        "question": question,
        "retrieval_plan": (
            "1. Why was the Macintosh important? "
            "(source: any relevant source)"
        ),
        "context": document.page_content,
    }


def test_grade_context_node_marks_empty_results_insufficient() -> None:
    graph_rag = create_graph_rag_without_model()
    question = "Why was the Macintosh important?"

    update = graph_rag.grade_context(
        {
            "question": question,
            "retrieval_plan": make_plan(question),
            "documents": [],
        }
    )

    assert update == {
        "context_sufficient": False,
        "context_grade_reason": "No documents were retrieved.",
    }
    assert graph_rag.context_grader.last_input is None


def test_grade_context_node_requires_retrieved_documents() -> None:
    graph_rag = create_graph_rag_without_model()

    with pytest.raises(
        ValueError,
        match="retrieve node must set documents",
    ):
        graph_rag.grade_context(
            {"question": "Why was the Macintosh important?"}
        )


def test_route_after_grading_generates_for_sufficient_context() -> None:
    next_node = LangGraphRAG.route_after_grading(
        {
            "question": "Why was the Macintosh important?",
            "context_sufficient": True,
            "retry_count": 0,
        }
    )

    assert next_node == "generate_answer"


def test_route_after_grading_rewrites_insufficient_context() -> None:
    next_node = LangGraphRAG.route_after_grading(
        {
            "question": "Why was the Macintosh important?",
            "context_sufficient": False,
            "retry_count": 1,
        }
    )

    assert next_node == "rewrite_query"


def test_route_after_grading_refuses_after_retry_limit() -> None:
    next_node = LangGraphRAG.route_after_grading(
        {
            "question": "Why was the Macintosh important?",
            "context_sufficient": False,
            "retry_count": 2,
        }
    )

    assert next_node == "refuse_answer"


def test_route_after_grading_requires_grade_decision() -> None:
    with pytest.raises(
        ValueError,
        match="grade_context node must set context_sufficient",
    ):
        LangGraphRAG.route_after_grading(
            {"question": "Why was the Macintosh important?"}
        )


def test_rewrite_query_targets_missing_information() -> None:
    graph_rag = create_graph_rag_without_model()
    question = (
        "What elements typically appear in each homepage section?"
    )
    previous_queries = [question, "Apple homepage layout"]
    grade_reason = (
        "The context does not identify the links in each section."
    )

    update = graph_rag.rewrite_query(
        {
            "question": question,
            "retrieval_plan": make_plan(question),
            "retrieval_queries": previous_queries,
            "context_grade_reason": grade_reason,
            "retry_count": 0,
        }
    )

    assert update["retrieval_queries"] == [
        question,
        "Apple homepage product sections image headline links",
    ]
    assert update["retrieval_scopes"] == [
        RetrievalScope(query=question),
        RetrievalScope(
            query="Apple homepage product sections image headline links"
        ),
    ]
    assert update["retry_count"] == 1
    assert graph_rag.query_rewriter.last_input == {
        "question": question,
        "previous_queries": "\n".join(previous_queries),
        "context_grade_reason": grade_reason,
    }


def test_rewrite_query_requires_previous_queries() -> None:
    graph_rag = create_graph_rag_without_model()

    with pytest.raises(
        ValueError,
        match="build_queries node must set retrieval_queries",
    ):
        graph_rag.rewrite_query(
            {
                "question": "What appears in each homepage section?",
                "context_grade_reason": "The links are missing.",
            }
        )


def test_rewrite_query_requires_grader_reason() -> None:
    graph_rag = create_graph_rag_without_model()

    with pytest.raises(
        ValueError,
        match="grade_context node must set context_grade_reason",
    ):
        graph_rag.rewrite_query(
            {
                "question": "What appears in each homepage section?",
                "retrieval_queries": ["Apple homepage sections"],
            }
        )


def test_rewrite_query_rejects_empty_model_output() -> None:
    graph_rag = create_graph_rag_without_model()
    graph_rag.query_rewriter.result = "   "
    question = "What appears in each homepage section?"

    with pytest.raises(
        ValueError,
        match="query rewriter returned an empty query",
    ):
        graph_rag.rewrite_query(
            {
                "question": question,
                "retrieval_plan": make_plan(question),
                "retrieval_queries": ["Apple homepage sections"],
                "context_grade_reason": "The links are missing.",
            }
        )


def test_generate_answer_uses_retrieved_documents() -> None:
    graph_rag = create_graph_rag_without_model()
    document = Document(
        page_content="The Macintosh made computers easier to use."
    )

    update = graph_rag.generate_answer(
        {
            "question": "Why was the Macintosh important?",
            "documents": [document],
            "context_sufficient": True,
        }
    )

    assert update == {
        "answer": "The Macintosh made computers easier to use.",
        "llm_call_counts": {"answer_generator": 1},
    }
    assert graph_rag.components.answer_chain.calls == [
        {
            "question": "Why was the Macintosh important?",
            "context": document.page_content,
        }
    ]


def test_generate_answer_requires_sufficient_context() -> None:
    graph_rag = create_graph_rag_without_model()

    with pytest.raises(
        ValueError,
        match="context_sufficient=True",
    ):
        graph_rag.generate_answer(
            {
                "question": "Why was the Macintosh important?",
                "documents": [Document(page_content="Related text")],
                "context_sufficient": False,
            }
        )


def test_refuse_answer_after_retries() -> None:
    graph_rag = create_graph_rag_without_model()

    update = graph_rag.refuse_answer(
        {
            "question": "What is the average page-load time?",
            "context_sufficient": False,
            "context_grade_reason": "No load-time figure was found.",
            "retry_count": 2,
        }
    )

    assert update == {
        "answer": (
            "The documents do not provide the requested information."
        ),
        "llm_call_counts": {"refusal_generator": 1},
    }


def test_compiled_graph_runs_successful_answer_path() -> None:
    graph_rag = create_graph_rag_without_model()
    graph_rag.graph = graph_rag._build_graph()

    result = graph_rag.invoke(
        "Why was the Macintosh important?",
        strategy="simple",
    )

    assert result["context_sufficient"] is True
    assert result["retry_count"] == 0
    assert result["answer"] == (
        "The Macintosh made computers easier to use."
    )


def test_compiled_graph_rewrites_twice_then_refuses() -> None:
    graph_rag = create_graph_rag_without_model()
    graph_rag.context_grader.result = ContextGrade(
        sufficient=False,
        reason="The requested information is missing.",
    )
    graph_rag.graph = graph_rag._build_graph()

    result = graph_rag.invoke(
        "What is the average page-load time?",
        strategy="simple",
    )

    assert result["context_sufficient"] is False
    assert result["retry_count"] == 2
    assert result["answer"] == (
        "The documents do not provide the requested information."
    )
