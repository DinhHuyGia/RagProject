# Backend and RAG example

The complete fresh-machine setup is documented in the
[root README](../README.md). This guide contains backend-specific commands and
example-module details.

## Backend setup

Python 3.11 through 3.13 is supported. From the repository root, create an
isolated environment and install the project with its development tools:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Set `AZURE_OPENAI_API_KEY` (or `AZURE_OPENAI_KEY`) and
`AZURE_OPENAI_ENDPOINT` in the environment. Optional overrides are
`AZURE_OPENAI_CHAT_DEPLOYMENT` and `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`.
The defaults are `gpt-5-mini` and `text-embedding-3-small`, respectively.
Deployment values must match the deployment names in the configured Azure
resource.

The endpoint should contain only the Azure resource address:

```text
https://resource-name.openai.azure.com/
```

Do not add `/openai/v1/`; the backend adds it automatically.

Start the document API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn grounded.api:app --reload
```

Start the React frontend in a second terminal:

```powershell
cd frontend
pnpm.cmd install
pnpm.cmd dev
```

Then open `http://localhost:5173`. The local frontend forwards its API requests
to FastAPI at `http://127.0.0.1:8000`. More frontend details are in
`frontend/README.md`.

Run the tests and lint checks:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

Using the explicit `.venv` interpreter keeps the tests from accidentally
running with a separately installed Windows Store Python.

The API integration tests inject local test doubles for embeddings, retrieval,
and model responses, so they run without an Azure API key or network access.

The ingestion integration tests use the production TXT/PDF loaders, text
splitter, and an ephemeral Chroma collection. They use deterministic local
embeddings and generate their PDF fixture at runtime, so they also remain
offline:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ingestion_integration.py -v
```

Uploaded files, Chroma indexes, and the SQLite document registry are runtime
data under `data/` and are intentionally excluded from version control.

## example modules

Each section is an independent module. Run one from the repository root:

```powershell
.\.venv\Scripts\python.exe -m grounded.section_01_embedding_similarity
.\.venv\Scripts\python.exe -m grounded.section_03_simple_rag
.\.venv\Scripts\python.exe -m grounded.section_11_summary_indexing
.\.venv\Scripts\python.exe -m grounded.section_11b_proposition_indexing
```

Shared model, embedding, loading, splitting, and formatting helpers live in
`shared.py`. Importing a module does not make network or model calls; those
operations run only from its functions or `main()`.

If no key is set, a runnable example section asks for it without echoing the
value. `AZURE_OPENAI_ENDPOINT` must still be configured. The API should be
started with both required values already present in its environment.
