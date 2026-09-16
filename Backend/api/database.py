import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

load_dotenv(Path(__file__).resolve().parent / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://anycart:anycart@localhost:5432/anycart",
)


@contextmanager
def connection():
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        try:
            register_vector(conn)
        except psycopg.errors.UndefinedObject:
            # init_db enables the extension before vector values are used.
            pass
        yield conn


def init_db() -> None:
    with connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price NUMERIC(10, 2) NOT NULL,
                description TEXT NOT NULL,
                source_file TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS orders (
                order_id UUID PRIMARY KEY,
                product_id TEXT NOT NULL REFERENCES products(product_id),
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                status TEXT NOT NULL DEFAULT 'Pending',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS catalog_chunks (
                document_hash TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                source_file TEXT NOT NULL,
                page INTEGER,
                content TEXT NOT NULL,
                embedding vector(384) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (document_hash, chunk_id)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS catalog_chunks_embedding_idx
            ON catalog_chunks USING hnsw (embedding vector_cosine_ops)
            """
        )
