from typing import Any, Literal, NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from grounded.adaptive_rag import (
    AdaptiveRAG,
    IndexName,
    RequestedStrategy,
    RouteDecision,
    StrategyName,
)
from grounded.section_05_rag_fusion import reciprocal_rank_fusion
from grounded.shared import format_docs

RETRIEVAL_PLANNER_PROMPT = """
You are planning document retrieval for a RAG system whose corpus may contain
multiple documents.

Create the smallest set of independently searchable tasks needed to answer
the user's complete question.

Requirements:
- Use one task for a simple, single-fact question.
- Split multi-part or comparison questions into one task per required fact.
- Write each search_query using concise terms likely to occur in a document.
- When a task belongs to one available source, copy that source's exact
  document_id into source_document_id and its filename into source_hint.
- Never invent a document ID. Use null when no single source is implied.
- Set requires_multiple_sources=true only when answering the question requires
  evidence from more than one source.
- Leave task_id, status, attempts, evidence_chunk_ids, and grade_reason at
  their defaults; the graph manages those runtime fields.
- Do not answer the question and do not invent facts.

Available sources:
{available_sources}

Question:
{question}
"""

CONTEXT_GRADER_PROMPT = """
You are grading whether retrieved document context is sufficient to answer
the user's question.

Return sufficient=true only when the context contains enough evidence to
directly and completely answer every part of the question without outside
knowledge. A passage that is merely related to the topic is not sufficient.
Return sufficient=false when any requested fact is missing or when the
context is empty.

Question:
{question}

Required evidence from the retrieval plan:
{retrieval_plan}

Retrieved context:
{context}
"""

EVIDENCE_TASK_GRADER_PROMPT = """
You are grading the retrieved evidence for one evidence task in a RAG system.

Return sufficient=true only when the evidence directly and completely
supports an answer to this task's subquestion without outside knowledge.
Return sufficient=false when a required fact is missing, the evidence is only
topically related, or the evidence is empty. Explain exactly what is present
or missing in one concise reason.

Evidence task:
{subquestion}

Expected source:
{source}

Retrieved evidence:
{evidence}
"""

QUERY_REWRITE_PROMPT = """
Rewrite the search query so document retrieval can find the information that
was missing from the previous results.

Requirements:
- Preserve the intent of the original question.
- Target the specific information identified as missing by the grader.
- Use concise terms likely to appear in the source document.
- Return exactly one search query and no explanation.
- Do not answer the question.

Original question:
{question}

Previous retrieval queries:
{previous_queries}

Why the retrieved context was insufficient:
{context_grade_reason}

Rewritten search query:
"""

REFUSAL_PROMPT = """
Write one concise sentence explaining that the retrieved documents do not
provide enough information to answer the user's question.

Requirements:
- Mention the exact topic or information the user requested.
- Do not attempt to answer the question.
- Do not use outside knowledge.
- Return only the refusal sentence.

Question:
{question}

Why the context was insufficient:
{context_grade_reason}

Refusal:
"""

MAX_RETRIEVAL_RETRIES = 2
NextAfterGrading = Literal[
    "generate_answer",
    "rewrite_query",
    "refuse_answer",
]
NextAfterEvidenceGrading = Literal[
    "grade_context",
    "rewrite_task_queries",
    "fail_evidence_tasks",
]
EvidenceStatus = Literal[
    "pending",
    "retrieved",
    "satisfied",
    "insufficient",
    "failed",
]


class ContextGrade(BaseModel):
    """Structured decision produced by the context-grading model."""

    sufficient: bool
    reason: str


class EvidenceGrade(BaseModel):
    """Structured sufficiency grade for one evidence task."""

    sufficient: bool
    reason: str


class EvidenceTask(BaseModel):
    """One required fact and its evidence-gathering state."""

    task_id: str | None = None
    subquestion: str
    search_query: str
    source_hint: str | None = None
    source_document_id: str | None = None
    status: EvidenceStatus = "pending"
    attempts: int = 0
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    grade_reason: str | None = None


class RetrievalPlan(BaseModel):
    """Structured source-aware plan produced before retrieval."""

    tasks: list[EvidenceTask]
    requires_multiple_sources: bool
    reason: str


class AvailableSource(BaseModel):
    """One source currently available to the retrieval planner."""

    document_id: str
    filename: str


class RetrievalScope(BaseModel):
    """One query and its optional strict document filter."""

    query: str
    source_document_id: str | None = None
    task_id: str | None = None


class TaskRetryEvent(BaseModel):
    """Audit record for one evidence task scheduled for retrieval again."""

    task_id: str
    retry_type: Literal["targeted", "global"]
    status_before: EvidenceStatus
    query_before: str
    query_after: str
    source_document_id: str | None = None


class RAGState(TypedDict):
    # Initial request
    question: str
    requested_strategy: NotRequired[RequestedStrategy]
    indexes: NotRequired[list[IndexName]]

    # Router output
    strategy: NotRequired[StrategyName]
    routing_reason: NotRequired[str]

    # Retrieval planning output
    retrieval_plan: NotRequired[RetrievalPlan]
    available_sources: NotRequired[list[AvailableSource]]

    # Retrieval state
    retrieval_queries: NotRequired[list[str]]
    retrieval_scopes: NotRequired[list[RetrievalScope]]
    documents: NotRequired[list[Document]]
    task_evidence: NotRequired[dict[str, list[Document]]]

    # Per-task evidence-grading state
    evidence_tasks_graded: NotRequired[bool]
    all_evidence_tasks_satisfied: NotRequired[bool]
    evidence_grade_reason: NotRequired[str]
    task_retry_events: NotRequired[list[TaskRetryEvent]]
    llm_call_counts: NotRequired[dict[str, int]]

    # Context-grading and retry state
    context_sufficient: NotRequired[bool]
    context_grade_reason: NotRequired[str]
    retry_count: NotRequired[int]

    # Final output
    answer: NotRequired[str]


class LangGraphRAG:
    """LangGraph orchestration using the existing RAG components."""

    def __init__(self, chunk_retriever: Any):
        self.components = AdaptiveRAG(chunk_retriever)
        self.retrieval_planner = (
            ChatPromptTemplate.from_template(RETRIEVAL_PLANNER_PROMPT)
            | self.components.llm.with_structured_output(RetrievalPlan)
        )
        self.context_grader = (
            ChatPromptTemplate.from_template(CONTEXT_GRADER_PROMPT)
            | self.components.llm.with_structured_output(ContextGrade)
        )
        self.evidence_task_grader = (
            ChatPromptTemplate.from_template(
                EVIDENCE_TASK_GRADER_PROMPT
            )
            | self.components.llm.with_structured_output(EvidenceGrade)
        )
        self.query_rewriter = (
            ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
            | self.components.llm
            | StrOutputParser()
        )
        self.refusal_chain = (
            ChatPromptTemplate.from_template(REFUSAL_PROMPT)
            | self.components.llm
            | StrOutputParser()
        )
        self.graph = self._build_graph()

    @staticmethod
    def _increment_llm_calls(
        state: RAGState,
        call_name: str,
        count: int = 1,
    ) -> dict[str, int]:
        calls = dict(state.get("llm_call_counts", {}))
        calls[call_name] = calls.get(call_name, 0) + count
        return calls

    def _list_available_sources(self) -> list[AvailableSource]:
        """Read the current document catalog from retriever metadata."""
        override = getattr(self, "source_catalog_override", None)
        if override is not None:
            return override

        retriever = getattr(self.components, "chunk_retriever", None)
        vectorstore = getattr(retriever, "vectorstore", None)

        if vectorstore is None or not hasattr(vectorstore, "get"):
            return []

        stored = vectorstore.get(include=["metadatas"])
        source_map: dict[str, AvailableSource] = {}

        for metadata in stored.get("metadatas") or []:
            if not metadata:
                continue

            document_id = metadata.get("document_id")
            filename = metadata.get("filename")

            if document_id and filename:
                source_map[str(document_id)] = AvailableSource(
                    document_id=str(document_id),
                    filename=str(filename),
                )

        return sorted(
            source_map.values(),
            key=lambda source: source.filename.casefold(),
        )

    @staticmethod
    def _format_available_sources(
        sources: list[AvailableSource],
    ) -> str:
        if not sources:
            return "No source catalog is available. Use null document IDs."

        return "\n".join(
            (
                f"- document_id={source.document_id}; "
                f"filename={source.filename}"
            )
            for source in sources
        )

    def plan_retrieval(self, state: RAGState) -> dict:
        """Break the request into source-aware retrieval tasks."""
        available_sources = self._list_available_sources()
        plan = self.retrieval_planner.invoke(
            {
                "question": state["question"],
                "available_sources": self._format_available_sources(
                    available_sources
                ),
            }
        )

        if not plan.tasks:
            raise ValueError(
                "The retrieval planner returned no retrieval tasks."
            )

        # Runtime evidence state is controlled by the graph, not the LLM.
        # Stable IDs let later nodes update the correct task independently.
        plan = plan.model_copy(
            update={
                "tasks": [
                    task.model_copy(
                        update={
                            "task_id": f"evidence-{position}",
                            "status": "pending",
                            "attempts": 0,
                            "evidence_chunk_ids": [],
                            "grade_reason": None,
                        }
                    )
                    for position, task in enumerate(
                        plan.tasks,
                        start=1,
                    )
                ]
            }
        )

        for task in plan.tasks:
            if not task.subquestion.strip():
                raise ValueError(
                    "The retrieval planner returned an empty subquestion."
                )
            if not task.search_query.strip():
                raise ValueError(
                    "The retrieval planner returned an empty search query."
                )

        available_ids = {
            source.document_id for source in available_sources
        }
        invalid_ids = {
            task.source_document_id
            for task in plan.tasks
            if task.source_document_id is not None
            and task.source_document_id not in available_ids
        }
        if invalid_ids:
            invalid = ", ".join(sorted(invalid_ids))
            raise ValueError(
                "The retrieval planner selected unknown document ID(s): "
                f"{invalid}."
            )

        return {
            "available_sources": available_sources,
            "retrieval_plan": plan,
            "llm_call_counts": self._increment_llm_calls(
                state,
                "retrieval_planner",
            ),
        }

    def route_question(self, state: RAGState) -> dict:
        """Choose a retrieval strategy and update the graph state."""
        requested = state.get("requested_strategy", "auto")
        plan = state.get("retrieval_plan")

        if plan is None:
            raise ValueError(
                "The plan_retrieval node must set retrieval_plan before "
                "route_question runs."
            )

        if requested == "auto" and (
            plan.requires_multiple_sources or len(plan.tasks) > 1
        ):
            decision = RouteDecision(
                strategy="decomposition",
                reason=(
                    "The retrieval plan contains multiple evidence tasks, "
                    "so the graph selected decomposition."
                ),
            )
        elif requested == "auto":
            decision = self.components.router.invoke(
                {"question": state["question"]}
            )
            llm_call_counts = self._increment_llm_calls(
                state,
                "strategy_router",
            )
        else:
            decision = RouteDecision(
                strategy=requested,
                reason="The API caller selected this strategy.",
            )

        update = {
            "strategy": decision.strategy,
            "routing_reason": decision.reason,
        }
        if requested == "auto" and not (
            plan.requires_multiple_sources or len(plan.tasks) > 1
        ):
            update["llm_call_counts"] = llm_call_counts
        return update

    def build_queries(self, state: RAGState) -> dict:
        """Create retrieval queries from the selected strategy."""
        strategy = state.get("strategy")
        plan = state.get("retrieval_plan")

        if strategy is None:
            raise ValueError(
                "The route_question node must set strategy before "
                "build_queries runs."
            )

        if plan is None:
            raise ValueError(
                "The plan_retrieval node must set retrieval_plan before "
                "build_queries runs."
            )

        planned_scopes = self._scopes_from_plan(plan)

        # The plan already performs decomposition. Running the older
        # decomposition transformer again would create redundant tasks.
        if strategy == "decomposition":
            transformed_queries: list[str] = []
        else:
            transformed_queries = self.components.build_queries(
                question=state["question"],
                strategy=strategy,
            )

        candidate_scopes = [
            RetrievalScope(query=state["question"]),
            *planned_scopes,
            *(
                RetrievalScope(query=query)
                for query in transformed_queries
            ),
        ]
        scopes = self._deduplicate_scopes(candidate_scopes)

        update = {
            "retrieval_queries": [scope.query for scope in scopes],
            "retrieval_scopes": scopes,
        }
        if strategy not in {"simple", "decomposition"}:
            update["llm_call_counts"] = self._increment_llm_calls(
                state,
                "query_transformer",
            )
        return update

    @staticmethod
    def _deduplicate_scopes(
        scopes: list[RetrievalScope],
    ) -> list[RetrievalScope]:
        unique_scopes = []
        seen = set()

        for scope in scopes:
            key = (
                scope.query,
                scope.source_document_id,
                scope.task_id,
            )
            if key not in seen:
                seen.add(key)
                unique_scopes.append(scope)

        return unique_scopes

    @staticmethod
    def _scopes_from_plan(plan: RetrievalPlan) -> list[RetrievalScope]:
        """Turn plan tasks into queries with optional metadata filters."""
        scopes = []

        for task in plan.tasks:
            query = task.search_query.strip()
            source_hint = (task.source_hint or "").strip()

            # A hint helps semantic retrieval only when an exact document
            # ID was not available for strict metadata filtering.
            if source_hint and task.source_document_id is None:
                query = f"{query} {source_hint}"

            scopes.append(
                RetrievalScope(
                    query=query,
                    source_document_id=task.source_document_id,
                    task_id=task.task_id,
                )
            )

        return LangGraphRAG._deduplicate_scopes(scopes)

    @staticmethod
    def _format_retrieval_plan(plan: RetrievalPlan) -> str:
        """Format planned evidence tasks for the context grader."""
        lines = []

        for position, task in enumerate(plan.tasks, start=1):
            source = task.source_hint or "any relevant source"
            lines.append(
                f"{position}. {task.subquestion} (source: {source})"
            )

        return "\n".join(lines)

    def retrieve(self, state: RAGState) -> dict:
        """Retrieve and fuse documents for the generated queries."""
        queries = state.get("retrieval_queries")
        scopes = state.get("retrieval_scopes")
        plan = state.get("retrieval_plan")

        if not queries or not scopes:
            raise ValueError(
                "The build_queries node must set retrieval queries and "
                "scopes before retrieve runs."
            )

        if plan is None:
            raise ValueError(
                "The plan_retrieval node must set retrieval_plan before "
                "retrieve runs."
            )

        indexes = self.components.normalize_indexes(
            state.get("indexes")
        )
        ranked_results = []
        results_by_task: dict[str, list[Document]] = {}

        for scope in scopes:
            results = self.components.retrieve(
                queries=[scope.query],
                indexes=indexes,
                document_id=scope.source_document_id,
            )
            ranked_results.append(results)

            if scope.task_id is not None:
                results_by_task.setdefault(scope.task_id, []).extend(
                    results
                )

        current_documents = reciprocal_rank_fusion(ranked_results)[:8]
        task_evidence = dict(state.get("task_evidence", {}))

        for task_id, results in results_by_task.items():
            task_evidence[task_id] = self._deduplicate_documents(
                [*task_evidence.get(task_id, []), *results]
            )

        updated_tasks = []
        for task in plan.tasks:
            task_id = task.task_id

            if task_id is None or task_id not in results_by_task:
                updated_tasks.append(task)
                continue

            evidence = task_evidence.get(task_id, [])
            chunk_ids = [
                str(document.metadata["chunk_id"])
                for document in evidence
                if document.metadata.get("chunk_id") is not None
            ]
            updated_tasks.append(
                task.model_copy(
                    update={
                        "status": (
                            "retrieved" if evidence else "insufficient"
                        ),
                        "attempts": task.attempts + 1,
                        "evidence_chunk_ids": chunk_ids,
                        "grade_reason": (
                            None
                            if evidence
                            else "No documents were retrieved for this task."
                        ),
                    }
                )
            )

        updated_plan = plan.model_copy(
            update={"tasks": updated_tasks}
        )
        documents = self._combine_task_evidence(
            plan=updated_plan,
            task_evidence=task_evidence,
            fallback_documents=current_documents,
        )

        return {
            "indexes": indexes,
            "documents": documents,
            "retrieval_plan": updated_plan,
            "task_evidence": task_evidence,
        }

    @staticmethod
    def _deduplicate_documents(
        documents: list[Document],
    ) -> list[Document]:
        unique_documents = []
        seen = set()

        for document in documents:
            metadata = document.metadata
            key = (
                metadata.get("document_id"),
                metadata.get("chunk_id"),
                metadata.get("page_number"),
                document.page_content,
            )

            if key not in seen:
                seen.add(key)
                unique_documents.append(document)

        return unique_documents

    @classmethod
    def _combine_task_evidence(
        cls,
        *,
        plan: RetrievalPlan,
        task_evidence: dict[str, list[Document]],
        fallback_documents: list[Document],
        limit: int = 8,
    ) -> list[Document]:
        """Combine task evidence while giving each task representation."""
        evidence_lists = [
            task_evidence.get(task.task_id, [])
            for task in plan.tasks
            if task.task_id is not None
        ]
        balanced_documents = []
        largest_list = max(
            (len(documents) for documents in evidence_lists),
            default=0,
        )

        for position in range(largest_list):
            for documents in evidence_lists:
                if position < len(documents):
                    balanced_documents.append(documents[position])

        combined = cls._deduplicate_documents(
            [*balanced_documents, *fallback_documents]
        )
        return combined[:limit]

    def grade_evidence_task(self, state: RAGState) -> dict:
        """Grade each task using only the evidence retrieved for it."""
        plan = state.get("retrieval_plan")
        task_evidence = state.get("task_evidence")

        if plan is None:
            raise ValueError(
                "The plan_retrieval node must set retrieval_plan before "
                "grade_evidence_task runs."
            )

        if task_evidence is None:
            raise ValueError(
                "The retrieve node must set task_evidence before "
                "grade_evidence_task runs."
            )

        updated_tasks = []
        grader_call_count = 0

        for task in plan.tasks:
            if task.task_id is None:
                raise ValueError(
                    "Every evidence task must have a task_id before "
                    "grading."
                )

            if task.status in {"satisfied", "failed"}:
                updated_tasks.append(task)
                continue

            evidence = task_evidence.get(task.task_id, [])

            if not evidence:
                updated_tasks.append(
                    task.model_copy(
                        update={
                            "status": "insufficient",
                            "grade_reason": (
                                "No evidence was retrieved for this task."
                            ),
                        }
                    )
                )
                continue

            grade = self.evidence_task_grader.invoke(
                {
                    "subquestion": task.subquestion,
                    "source": (
                        task.source_hint or "any relevant source"
                    ),
                    "evidence": format_docs(evidence),
                }
            )
            grader_call_count += 1
            updated_tasks.append(
                task.model_copy(
                    update={
                        "status": (
                            "satisfied"
                            if grade.sufficient
                            else "insufficient"
                        ),
                        "grade_reason": grade.reason,
                    }
                )
            )

        updated_plan = plan.model_copy(
            update={"tasks": updated_tasks}
        )
        insufficient_tasks = [
            task for task in updated_tasks if task.status != "satisfied"
        ]

        if insufficient_tasks:
            grade_reason = "; ".join(
                (
                    f"{task.task_id}: "
                    f"{task.grade_reason or 'Evidence is insufficient.'}"
                )
                for task in insufficient_tasks
            )
        else:
            grade_reason = "Every evidence task is sufficiently supported."

        update = {
            "retrieval_plan": updated_plan,
            "evidence_tasks_graded": True,
            "all_evidence_tasks_satisfied": not insufficient_tasks,
            "evidence_grade_reason": grade_reason,
        }
        if grader_call_count:
            update["llm_call_counts"] = self._increment_llm_calls(
                state,
                "evidence_task_grader",
                grader_call_count,
            )
        return update

    @staticmethod
    def _evidence_task_can_retry(task: EvidenceTask) -> bool:
        """Return whether an insufficient task has retries remaining."""
        return (
            task.status == "insufficient"
            and task.attempts <= MAX_RETRIEVAL_RETRIES
        )

    @classmethod
    def route_after_evidence_grading(
        cls,
        state: RAGState,
    ) -> NextAfterEvidenceGrading:
        """Choose whether to continue, retry missing tasks, or fail."""
        if state.get("evidence_tasks_graded") is not True:
            raise ValueError(
                "grade_evidence_task must finish before conditional "
                "routing runs."
            )

        if state.get("all_evidence_tasks_satisfied") is True:
            return "grade_context"

        plan = state.get("retrieval_plan")
        if plan is None:
            raise ValueError(
                "grade_evidence_task must preserve retrieval_plan before "
                "conditional routing runs."
            )

        if any(cls._evidence_task_can_retry(task) for task in plan.tasks):
            return "rewrite_task_queries"

        return "fail_evidence_tasks"

    def rewrite_task_queries(self, state: RAGState) -> dict:
        """Rewrite and schedule only insufficient, retryable tasks."""
        plan = state.get("retrieval_plan")

        if plan is None:
            raise ValueError(
                "grade_evidence_task must set retrieval_plan before "
                "rewrite_task_queries runs."
            )

        updated_tasks = []
        retry_scopes = []
        retry_events = list(state.get("task_retry_events", []))

        for task in plan.tasks:
            if not self._evidence_task_can_retry(task):
                if task.status == "insufficient":
                    task = task.model_copy(update={"status": "failed"})
                updated_tasks.append(task)
                continue

            rewritten_query = self.query_rewriter.invoke(
                {
                    "question": task.subquestion,
                    "previous_queries": task.search_query,
                    "context_grade_reason": (
                        task.grade_reason
                        or "The retrieved evidence was insufficient."
                    ),
                }
            ).strip()

            if not rewritten_query:
                raise ValueError(
                    "The task query rewriter returned an empty query."
                )

            updated_task = task.model_copy(
                update={
                    "search_query": rewritten_query,
                    "status": "pending",
                    "grade_reason": None,
                }
            )
            updated_tasks.append(updated_task)
            retry_scopes.append(
                RetrievalScope(
                    query=rewritten_query,
                    source_document_id=task.source_document_id,
                    task_id=task.task_id,
                )
            )
            if task.task_id is not None:
                retry_events.append(
                    TaskRetryEvent(
                        task_id=task.task_id,
                        retry_type="targeted",
                        status_before=task.status,
                        query_before=task.search_query,
                        query_after=rewritten_query,
                        source_document_id=task.source_document_id,
                    )
                )

        if not retry_scopes:
            raise ValueError(
                "No insufficient evidence tasks have retries remaining."
            )

        retry_scopes = self._deduplicate_scopes(retry_scopes)
        updated_plan = plan.model_copy(
            update={"tasks": updated_tasks}
        )

        return {
            "retrieval_plan": updated_plan,
            "retrieval_queries": [
                scope.query for scope in retry_scopes
            ],
            "retrieval_scopes": retry_scopes,
            "task_retry_events": retry_events,
            "llm_call_counts": self._increment_llm_calls(
                state,
                "task_query_rewriter",
                len(retry_scopes),
            ),
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def fail_evidence_tasks(self, state: RAGState) -> dict:
        """Mark unresolved tasks failed after their retries are exhausted."""
        plan = state.get("retrieval_plan")

        if plan is None:
            raise ValueError(
                "grade_evidence_task must set retrieval_plan before "
                "fail_evidence_tasks runs."
            )

        updated_tasks = [
            (
                task.model_copy(update={"status": "failed"})
                if task.status != "satisfied"
                else task
            )
            for task in plan.tasks
        ]
        failed_tasks = [
            task for task in updated_tasks if task.status == "failed"
        ]
        reason = "; ".join(
            (
                f"{task.task_id}: "
                f"{task.grade_reason or 'Required evidence was not found.'}"
            )
            for task in failed_tasks
        )

        return {
            "retrieval_plan": plan.model_copy(
                update={"tasks": updated_tasks}
            ),
            "context_sufficient": False,
            "context_grade_reason": reason,
            "retry_count": max(
                state.get("retry_count", 0),
                MAX_RETRIEVAL_RETRIES,
            ),
        }

    def grade_context(self, state: RAGState) -> dict:
        """Decide whether the retrieved documents can answer the question."""
        documents = state.get("documents")
        plan = state.get("retrieval_plan")

        if documents is None:
            raise ValueError(
                "The retrieve node must set documents before "
                "grade_context runs."
            )

        if plan is None:
            raise ValueError(
                "The plan_retrieval node must set retrieval_plan before "
                "grade_context runs."
            )

        if (
            state.get("evidence_tasks_graded") is True
            and state.get("all_evidence_tasks_satisfied") is not True
        ):
            return {
                "context_sufficient": False,
                "context_grade_reason": state.get(
                    "evidence_grade_reason",
                    "At least one evidence task is insufficient.",
                ),
                "retrieval_plan": plan,
            }

        if not documents:
            return {
                "context_sufficient": False,
                "context_grade_reason": "No documents were retrieved.",
            }

        grade = self.context_grader.invoke(
            {
                "question": state["question"],
                "retrieval_plan": self._format_retrieval_plan(plan),
                "context": format_docs(documents),
            }
        )
        llm_call_counts = self._increment_llm_calls(
            state,
            "context_grader",
        )

        if grade.sufficient and not state.get("evidence_tasks_graded"):
            plan = plan.model_copy(
                update={
                    "tasks": [
                        task.model_copy(
                            update={
                                "status": "satisfied",
                                "grade_reason": grade.reason,
                            }
                        )
                        for task in plan.tasks
                    ]
                }
            )

        return {
            "context_sufficient": grade.sufficient,
            "context_grade_reason": grade.reason,
            "retrieval_plan": plan,
            "llm_call_counts": llm_call_counts,
        }

    @staticmethod
    def route_after_grading(state: RAGState) -> NextAfterGrading:
        """Choose the next node from context quality and retry count."""
        context_sufficient = state.get("context_sufficient")

        if context_sufficient is None:
            raise ValueError(
                "The grade_context node must set context_sufficient "
                "before conditional routing runs."
            )

        if context_sufficient:
            return "generate_answer"

        retry_count = state.get("retry_count", 0)

        if retry_count < MAX_RETRIEVAL_RETRIES:
            return "rewrite_query"

        return "refuse_answer"

    def rewrite_query(self, state: RAGState) -> dict:
        """Rewrite an insufficient query and increment the retry count."""
        previous_queries = state.get("retrieval_queries")

        if not previous_queries:
            raise ValueError(
                "The build_queries node must set retrieval_queries "
                "before rewrite_query runs."
            )

        context_grade_reason = state.get("context_grade_reason")
        plan = state.get("retrieval_plan")

        if not context_grade_reason:
            raise ValueError(
                "The grade_context node must set context_grade_reason "
                "before rewrite_query runs."
            )

        if plan is None:
            raise ValueError(
                "The plan_retrieval node must set retrieval_plan before "
                "rewrite_query runs."
            )

        rewritten_query = self.query_rewriter.invoke(
            {
                "question": state["question"],
                "previous_queries": "\n".join(previous_queries),
                "context_grade_reason": context_grade_reason,
            }
        ).strip()

        if not rewritten_query:
            raise ValueError("The query rewriter returned an empty query.")

        scopes = self._deduplicate_scopes(
            [
                RetrievalScope(query=state["question"]),
                *self._scopes_from_plan(plan),
                RetrievalScope(query=rewritten_query),
            ]
        )
        retry_events = list(state.get("task_retry_events", []))
        retry_events.extend(
            TaskRetryEvent(
                task_id=task.task_id,
                retry_type="global",
                status_before=task.status,
                query_before=task.search_query,
                query_after=task.search_query,
                source_document_id=task.source_document_id,
            )
            for task in plan.tasks
            if task.task_id is not None
        )

        return {
            "retrieval_queries": [scope.query for scope in scopes],
            "retrieval_scopes": scopes,
            "task_retry_events": retry_events,
            "llm_call_counts": self._increment_llm_calls(
                state,
                "global_query_rewriter",
            ),
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def generate_answer(self, state: RAGState) -> dict:
        """Generate a grounded answer from sufficient context."""
        if state.get("context_sufficient") is not True:
            raise ValueError(
                "generate_answer requires context_sufficient=True."
            )

        documents = state.get("documents")

        if not documents:
            raise ValueError(
                "The retrieve node must provide documents before "
                "generate_answer runs."
            )

        answer = self.components.answer_chain.invoke(
            {
                "question": state["question"],
                "context": format_docs(documents),
            }
        ).strip()

        if not answer:
            raise ValueError("The answer generator returned an empty answer.")

        return {
            "answer": answer,
            "llm_call_counts": self._increment_llm_calls(
                state,
                "answer_generator",
            ),
        }

    def refuse_answer(self, state: RAGState) -> dict:
        """Return a grounded refusal after retrieval retries are exhausted."""
        if state.get("context_sufficient") is not False:
            raise ValueError(
                "refuse_answer requires context_sufficient=False."
            )

        retry_count = state.get("retry_count", 0)
        if retry_count < MAX_RETRIEVAL_RETRIES:
            raise ValueError(
                "refuse_answer requires all retrieval retries to be used."
            )

        refusal = self.refusal_chain.invoke(
            {
                "question": state["question"],
                "context_grade_reason": state.get(
                    "context_grade_reason",
                    "The requested information was not found.",
                ),
            }
        ).strip()

        if not refusal:
            raise ValueError("The refusal generator returned an empty answer.")

        return {
            "answer": refusal,
            "llm_call_counts": self._increment_llm_calls(
                state,
                "refusal_generator",
            ),
        }

    def _build_graph(self):
        """Build and compile the complete adaptive RAG state graph."""
        builder = StateGraph(RAGState)

        builder.add_node("plan_retrieval", self.plan_retrieval)
        builder.add_node("route_question", self.route_question)
        builder.add_node("build_queries", self.build_queries)
        builder.add_node("retrieve", self.retrieve)
        builder.add_node(
            "grade_evidence_task",
            self.grade_evidence_task,
        )
        builder.add_node(
            "rewrite_task_queries",
            self.rewrite_task_queries,
        )
        builder.add_node(
            "fail_evidence_tasks",
            self.fail_evidence_tasks,
        )
        builder.add_node("grade_context", self.grade_context)
        builder.add_node("rewrite_query", self.rewrite_query)
        builder.add_node("generate_answer", self.generate_answer)
        builder.add_node("refuse_answer", self.refuse_answer)

        builder.add_edge(START, "plan_retrieval")
        builder.add_edge("plan_retrieval", "route_question")
        builder.add_edge("route_question", "build_queries")
        builder.add_edge("build_queries", "retrieve")
        builder.add_edge("retrieve", "grade_evidence_task")
        builder.add_conditional_edges(
            "grade_evidence_task",
            self.route_after_evidence_grading,
            {
                "grade_context": "grade_context",
                "rewrite_task_queries": "rewrite_task_queries",
                "fail_evidence_tasks": "fail_evidence_tasks",
            },
        )
        builder.add_edge("rewrite_task_queries", "retrieve")
        builder.add_edge("fail_evidence_tasks", "refuse_answer")
        builder.add_conditional_edges(
            "grade_context",
            self.route_after_grading,
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
                "refuse_answer": "refuse_answer",
            },
        )
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("generate_answer", END)
        builder.add_edge("refuse_answer", END)

        return builder.compile()

    def invoke(
        self,
        question: str,
        strategy: RequestedStrategy = "auto",
        indexes: list[IndexName] | None = None,
    ) -> RAGState:
        """Run the compiled LangGraph workflow for one question."""
        if not question.strip():
            raise ValueError("question must not be empty.")

        initial_state: RAGState = {
            "question": question,
            "requested_strategy": strategy,
            "retry_count": 0,
        }

        if indexes is not None:
            initial_state["indexes"] = indexes

        return self.graph.invoke(initial_state)
