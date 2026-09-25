# Grounded frontend

This folder contains the React and TypeScript interface for the Adaptive RAG
API in `grounded/api.py`.

See the [root README](../README.md) for complete Python, Azure OpenAI, and
fresh-machine setup instructions.

## Prerequisites

- Node.js 24 LTS
- pnpm 11
- The FastAPI backend installed and configured

Install pnpm on Windows:

```powershell
npm.cmd install --global pnpm@latest-11
pnpm.cmd --version
```

The `.cmd` form avoids PowerShell execution-policy errors.

## Run locally

Start the FastAPI backend from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn grounded.api:app --reload
```

In a second terminal, start the frontend:

```powershell
cd frontend
pnpm.cmd install
pnpm.cmd dev
```

Open `http://localhost:5173`. During development, Vite forwards `/api`
requests to FastAPI at `http://127.0.0.1:8000`.

Create a production build with:

```powershell
pnpm.cmd build
```

## Configuration

For a separately hosted API, copy `.env.example` to `.env` and set:

```text
VITE_API_BASE_URL=https://your-api.example.com
```

The backend must allow the deployed frontend origin when the two applications
are hosted on different domains.

Do not put `AZURE_OPENAI_API_KEY` or any other secret in this folder. Variables
beginning with `VITE_` are included in browser-facing code.
