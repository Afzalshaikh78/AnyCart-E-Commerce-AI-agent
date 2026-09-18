# AnyCart AI E-Commerce Platform

AnyCart is a full-stack AI E-commerce assistant that turns an uploaded product
catalog PDF into a searchable shopping experience. It extracts products from
the PDF, stores product data and orders in PostgreSQL, stores semantic
embeddings with pgvector, and uses a LangGraph agent to recommend products,
place orders, and track order status.

This project is structured for AI engineer portfolio and resume use.

## Tech Stack

- Frontend: React, Vite, Tailwind CSS
- Backend: FastAPI, Pydantic, Uvicorn
- Agent workflow: LangGraph
- RAG pipeline: LangChain text splitting, PDF parsing, Hugging Face embeddings
- Database: PostgreSQL with pgvector
- LLM providers: Mistral AI or Hugging Face Inference API
- Deployment: Vercel frontend, Render backend, Neon PostgreSQL

## Architecture

```text
React + Tailwind frontend
        |
        v
FastAPI REST API
        |
        +--> PDF upload and product extraction
        +--> LangGraph agent router
        +--> RAG retrieval over catalog chunks
        +--> Hugging Face embedding API
        +--> Mistral or Hugging Face generation
        +--> PostgreSQL + pgvector
```

The uploaded PDF is the source of truth. Products shown in the UI and products
used by the assistant are extracted from the uploaded catalog, not hardcoded.

## Features

- Upload a product catalog PDF
- Extract structured product data from the PDF
- Show featured product cards after upload
- Ask product questions using semantic RAG
- Store embeddings inside PostgreSQL using pgvector
- Place orders from catalog product IDs
- Track orders by order ID
- Use LangGraph to route between recommendation, ordering, and tracking flows
- Deployable frontend and backend with production environment variables

## Project Structure

```text
.
+-- Backend/
|   +-- api/
|   |   +-- agent.py          # LangGraph workflow and shopping tools
|   |   +-- database.py       # PostgreSQL and pgvector setup
|   |   +-- local_rag.py      # PDF parsing, chunking, embeddings, retrieval
|   |   +-- llm_provider.py   # Mistral and Hugging Face generation
|   |   +-- main.py           # FastAPI routes
|   |   +-- schemas.py        # API request schemas
|   |   +-- requirements.txt
|   |   +-- tests/
+-- frontend/
|   +-- src/
|   |   +-- App.jsx           # Main React application
|   |   +-- styles.css
|   +-- public/
|   +-- package.json
+-- docker-compose.yml        # Local PostgreSQL with pgvector
+-- README.md
```

## Prerequisites

- Python 3.11
- uv
- Node.js 20 or newer
- Docker Desktop
- Hugging Face token for embeddings
- Optional: Mistral API key for better LLM answers

## Environment Variables

Create the backend env file:

```powershell
Copy-Item Backend\api\.env.example Backend\api\.env
```

For local Docker PostgreSQL, use:

```env
DATABASE_URL=postgresql://anycart:anycart@localhost:5433/anycart
CORS_ORIGINS=http://localhost:5173
HF_TOKEN=your_huggingface_token
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_PROVIDER=mistral
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_MODEL=mistral-small-latest
```

If you want to use Hugging Face for generation instead of Mistral:

```env
LLM_PROVIDER=huggingface
HF_GENERATION_MODEL=google/flan-t5-small
```

## Run Locally

Start PostgreSQL with pgvector:

```powershell
docker compose up -d postgres
```

Create the backend virtual environment:

```powershell
uv venv Backend\.venv --python 3.11
uv pip install --python Backend\.venv\Scripts\python.exe -r Backend\api\requirements.txt
Copy-Item Backend\api\.env.example Backend\api\.env
```

Update `Backend\api\.env` with your API keys.

Start FastAPI:

```powershell
Backend\.venv\Scripts\python.exe -m uvicorn Backend.api.main:app --reload --port 8000
```

Start React in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

Upload a catalog PDF, then ask questions in the chat.

## API Endpoints

```text
GET  /api/health
POST /api/catalog/upload
GET  /api/catalog/products
POST /api/chat
GET  /api/orders/{order_id}
```

## Database Tables

- `products`: product rows extracted from the uploaded PDF
- `orders`: customer orders created through the assistant
- `catalog_chunks`: PDF text chunks and pgvector embeddings

pgvector allows PostgreSQL to search by semantic similarity, so the assistant
can find relevant catalog chunks even when the user does not use exact product
keywords.

## Verification

Frontend build:

```powershell
cd frontend
npm run build
```

Backend tests:

```powershell
cd ..
Backend\.venv\Scripts\python.exe -m pytest Backend\api\tests -q -p no:cacheprovider
```

## Deployment

Frontend on Vercel:

```env
VITE_API_URL=https://your-render-service.onrender.com/api
```

Backend on Render:

```text
Build Command:
uv pip install --system -r Backend/api/requirements.txt

Start Command:
uvicorn Backend.api.main:app --host 0.0.0.0 --port $PORT
```

Render environment variables:

```env
DATABASE_URL=your_neon_postgresql_url
CORS_ORIGINS=https://your-vercel-app.vercel.app
HF_TOKEN=your_huggingface_token
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_PROVIDER=mistral
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_MODEL=mistral-small-latest
```

Neon PostgreSQL must support the `vector` extension. The app creates the
extension and tables during backend startup.


