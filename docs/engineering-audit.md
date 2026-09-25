# Engineering and security audit

Audit date: 2026-09-25. Findings describe the checked-in implementation.

## Retrieval implementation

- `grounded/api.py:build_services` instantiates the LangChain `AdaptiveRAG` with
  an eight-result Chroma retriever. The HTTP API does not invoke LangGraph.
- `grounded/langgraph_rag.py` implements a compiled LangGraph `StateGraph`.
  Nodes: `plan_retrieval`, `route_question`, `build_queries`, `retrieve`,
  `grade_evidence_task`, `rewrite_task_queries`, `fail_evidence_tasks`,
  `grade_context`, `rewrite_query`, `generate_answer`, and `refuse_answer`.
- Evidence grading routes fully satisfied plans to context grading, insufficient
  retryable tasks to targeted query rewriting, and exhausted plans to failure
  and refusal. Satisfied evidence is retained during targeted retries.
- Context grading routes sufficient evidence to generation; insufficient
  evidence triggers a global rewrite while its retry counter is below two,
  otherwise refusal. `MAX_RETRIEVAL_RETRIES = 2`; task attempts and the global
  retry counter are separate state fields.
- Strategies are direct search (`simple`), four alternative phrasings
  (`multi_query`), three subquestions (`decomposition` in LangChain), one broader
  question (`step_back`), and a hypothetical answer used for search (`hyde`).
  LangGraph uses planned evidence tasks for decomposition and chooses it
  automatically for multi-task/multi-source plans. Reciprocal rank fusion merges
  results. Summary/proposition indexes are standalone examples, not API indexes;
  the API accepts only `chunks`.
- Upload and evaluation chunking: 600 tokens / 100 overlap in
  `grounded/ingestion.py` and `evaluation/evaluate_retrieval.py`. Shared web
  examples: 300/50; summary/proposition parent chunks: 2000/200 in
  `grounded/shared.py`. These use a tiktoken-based recursive splitter.

## Evaluation inventory

- Two PDF fixtures: an Apple website analysis and Apple's 2025 Form 10-K.
- Active questions: `apple_website_analysis_questions.jsonl`.
  `dataset.jsonl` is empty and is not the configured dataset.
- `evaluate_retrieval.py`: evidence Precision@8, Recall@8, and source recall.
- `compare_rag.py`: direct LangChain/LangGraph comparison with console output.
- `evaluate_ragas.py`: simple-strategy baseline.
- `evaluate_ragas_comparison.py`: configurable strategy comparison, task metrics,
  source routing, retries, LLM-call counts, and timing.
- RAGAS metrics: Context Precision, Context Recall, Faithfulness, Answer
  Relevancy, Factual Correctness (F1, high atomicity/coverage).
  `scoring.py` separately scores refusals for unanswerable questions.
- `ragas_compat.py`: optional Vertex AI import compatibility shim.
- `report/ragas_results.json`: 20 evaluated questions (16 answerable and four
  unanswerable), responses, contexts, scores, and aggregates.
- `report/ragas_comparison_results.json`: one answerable cross-document question,
  `auto` strategy, both pipelines' results and graph traces. The graph refused
  after exhausting one evidence task. No separate console log is committed.

The root README cites both reports and preserves the unfavorable graph results.
Their sample sizes and strategies differ. Evaluation was not rerun during this
presentation work; saved model/deployment/date/revision provenance is absent.

## Configuration and compatibility

The package is now `grounded`; the working standalone launcher is retained as
`main.py`, installed as the `grounded` command. No functional modules were deleted.
Imports, package discovery, entry points, tests, and nested documentation were
updated. Vite's proxy already uses an HTTP address and needed no package rename.

The root `.env.example` lists all application settings, including the legacy key
alias, evaluation overrides, web user agent, and the frontend-only base URL.
`python-dotenv` loads the root `.env` without overriding existing environment
variables. Frontend settings still belong in `frontend/.env`. `.gitignore`
already ignores `.env` and `.env.*` while allowing `.env.example`.

## Security findings

No real API keys, resource-specific Azure endpoints, connection strings, access
tokens, or private keys were found in the scanned project files or reachable
Git text history. Matches were configuration names, placeholder endpoints,
test fixtures, or incidental text (for example, CSS `ask-button`).

Scope and method:

- Scanned tracked and non-ignored project files, including extracted text from
  both committed PDFs. Dependency caches and ignored runtime data were excluded.
- Applied patterns for Azure endpoints, literal credentials, OpenAI/GitHub/Slack/
  AWS token formats, connection-string keys, SAS signatures, and private keys.
- Searched `git log --all -p --no-textconv` using the requested case-insensitive
  expression `azure_openai_api_key|api[_-]?key|sk-|https://.*openai\.azure\.com`,
  then the broader credential patterns. The checkout is not shallow.
- Disabled Git text conversion because its configured PDF converter failed on
  this machine. The requested history scan covers text diffs; it is not a
  forensic scan of deleted binary content or unreachable objects.

No history was rewritten, and no remote changes were made. Pattern scanning
cannot establish that arbitrary encoded or unrecognized secrets never existed.

## Validation

- Baseline: 60 tests passed and Ruff passed after the initial tokenizer download.
- After renaming: all 60 tests and Ruff passed.
- With dotenv regression coverage: 62 tests and Ruff passed. Tests cover loading
  from the project root while running elsewhere and preserving environment values.
- Editable package installation and `grounded --help` succeeded.
- Frontend TypeScript check and Vite production build passed.
- No old naming remains in versioned project content. Local checkout paths,
  ignored caches, and historical commits retain their original names.

## Remaining owner tasks

1. Add `docs/images/ui-upload.png` and `docs/images/ui-answer.png`.
2. On the next evaluation run, record the date, code/dataset revisions, and
   actual chat, embedding, and judge deployments as requested in the README.
