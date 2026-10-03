import os
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .agent import agent_graph  # noqa: E402
from .database import connection, init_db  # noqa: E402
from .local_rag import active_catalog, set_catalog  # noqa: E402
from .metrics import elapsed_ms, metrics_summary, record_metric  # noqa: E402
from .schemas import ChatRequest  # noqa: E402

app = FastAPI(title="AnyCart API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict[str, str]:
    with connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok", "service": "anycart-api"}


@app.post("/api/catalog/upload")
async def upload_catalog(file: UploadFile = File(...)) -> dict:
    started_at = time.perf_counter()
    metadata = {"filename": file.filename or "catalog.pdf"}
    success = False
    if file.content_type != "application/pdf":
        record_metric("catalog_upload", elapsed_ms(started_at), False, metadata)
        raise HTTPException(status_code=415, detail="Only PDF catalogs are supported.")
    try:
        content = await file.read()
        metadata["file_size_bytes"] = len(content)
        catalog = set_catalog(content, file.filename or "catalog.pdf")
        with connection() as conn:
            conn.execute("DELETE FROM orders")
            conn.execute("DELETE FROM products")
            for product in catalog.products:
                conn.execute(
                    """INSERT INTO products(product_id, name, category, price, description, source_file)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (product["product_id"], product["name"], product["category"], product["price"], product["description"], file.filename),
                )
        metadata.update(products=len(catalog.products), chunks=catalog.chunk_count)
        success = True
        return {"filename": file.filename, "products": len(catalog.products), "chunks": catalog.chunk_count}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        record_metric("catalog_upload", elapsed_ms(started_at), success, metadata)


@app.get("/api/catalog/products")
def products() -> list[dict]:
    with connection() as conn:
        return list(conn.execute("SELECT product_id, name, category, price, description FROM products ORDER BY product_id"))


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, str]:
    started_at = time.perf_counter()
    success = False
    try:
        result = agent_graph.invoke({"message": request.message})
        success = True
        return {"response": result["response"]}
    finally:
        # Never store a user's question; only its execution outcome is measured.
        record_metric("chat", elapsed_ms(started_at), success)


@app.get("/api/orders/{order_id}")
def order_status(order_id: str) -> dict:
    started_at = time.perf_counter()
    success = False
    with connection() as conn:
        row = conn.execute(
            "SELECT order_id::text, product_id, quantity, status, created_at FROM orders WHERE order_id = %s",
            (order_id,),
        ).fetchone()
    if not row:
        record_metric("order_lookup", elapsed_ms(started_at), False)
        raise HTTPException(status_code=404, detail="Order not found")
    success = True
    record_metric("order_lookup", elapsed_ms(started_at), success)
    return dict(row)


@app.get("/api/metrics/summary")
def metric_summary() -> dict:
    return metrics_summary()



