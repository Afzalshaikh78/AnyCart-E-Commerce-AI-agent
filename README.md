# AnyCart AI Commerce Platform

Resume-ready full-stack application using React, Tailwind CSS, FastAPI,
LangGraph, LangChain, Hugging Face, PostgreSQL, and pgvector.

## Architecture

```text
React + Tailwind
        |
        v
FastAPI REST API
        |
        +--> LangGraph router and tools
        +--> LangChain PDF RAG + pgvector
        +--> PostgreSQL products, orders, and embeddings
        +--> Hugging Face or Mistral generation
```

The uploaded PDF is the product source of truth. PostgreSQL stores extracted
products, orders, and the semantic catalog embeddings through pgvector.

## Run locally

Start PostgreSQL with Docker Desktop running:

```powershell
docker compose up -d postgres
```

Install backend dependencies:

```powershell
uv venv Backend\.venv --python 3.11
uv pip install --python Backend\.venv\Scripts\python.exe -r Backend\api\requirements.txt
Copy-Item Backend\api\.env.example Backend\api\.env
```

Start FastAPI:

```powershell
Backend\.venv\Scripts\python.exe -m uvicorn Backend.api.main:app --reload --port 8000
```

Start React in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, upload a PDF, and use the chat panel.

## Verification

```powershell
cd frontend
npm run build

cd ..
Backend\.venv\Scripts\python.exe -m pytest Backend\api\tests -q -p no:cacheprovider
```

## Resume bullet

Built a full-stack agentic commerce platform with React/Tailwind, FastAPI,
LangGraph conditional workflows, LangChain tools, Hugging Face embeddings,
pgvector semantic retrieval, PostgreSQL persistence, PDF ingestion, and tested
order-tracking workflows.
