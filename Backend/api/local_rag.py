"""LangChain-based semantic RAG for product catalog PDFs."""

from __future__ import annotations

import hashlib
import io
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import requests

from .database import connection
from .llm_provider import configured_model, generate_answer, provider_name

load_dotenv()

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
MAX_PDF_BYTES = 25 * 1024 * 1024

# pgvector cosine distance is 1 - cosine similarity. A distance of 0.55
# corresponds to roughly cosine similarity >= 0.45.
RELEVANCE_DISTANCE = float(os.getenv("ANYCART_RELEVANCE_DISTANCE", "0.55"))


class HuggingFaceApiEmbeddings:
    """Remote embeddings keep the API service within free-host memory limits."""

    def __init__(self, model_name: str, token: str) -> None:
        self.token = token
        self.url = os.getenv(
            "HF_EMBEDDING_API_URL",
            f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self.token:
            raise RuntimeError("HF_TOKEN is required for Hugging Face embeddings.")
        response = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.token}"},
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) != len(texts):
            raise RuntimeError("Unexpected Hugging Face embedding response.")
        return [self._normalize(vector) for vector in payload]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @staticmethod
    def _normalize(vector: Any) -> list[float]:
        if not isinstance(vector, list) or not vector or not all(isinstance(value, (int, float)) for value in vector):
            raise RuntimeError("Hugging Face did not return a sentence embedding.")
        magnitude = sum(float(value) ** 2 for value in vector) ** 0.5
        if magnitude == 0:
            raise RuntimeError("Hugging Face returned an empty embedding.")
        return [float(value) / magnitude for value in vector]


class CatalogRAG:
    """Persistent LangChain embedding retriever backed by PostgreSQL pgvector."""

    _embedding_model: Any = None

    def __init__(self, pdf_bytes: bytes, filename: str = "catalog.pdf") -> None:
        if not pdf_bytes:
            raise ValueError("The uploaded PDF is empty.")
        self.filename = filename
        self.document_hash = hashlib.sha256(pdf_bytes).hexdigest()
        self.embedding_model_name = DEFAULT_EMBEDDING_MODEL
        self.generation_model_name = configured_model()
        self.chunk_count = 0
        self.products: list[dict[str, Any]] = []
        self._load_or_build(pdf_bytes)

    @classmethod
    def _get_embedding_model(cls) -> Any:
        if cls._embedding_model is None:
            cls._embedding_model = HuggingFaceApiEmbeddings(
                DEFAULT_EMBEDDING_MODEL,
                os.getenv("HF_TOKEN", "").strip(),
            )
        return cls._embedding_model

    @staticmethod
    def _extract_documents(pdf_bytes: bytes) -> list[Document]:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        documents: list[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = " ".join((page.extract_text() or "").split())
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"page": page_number},
                    )
                )
        if not documents:
            raise ValueError(
                "The PDF did not contain extractable text. Use a text-based PDF "
                "or add OCR before uploading it."
            )
        return documents

    @staticmethod
    def _extract_products(documents: list[Document]) -> list[dict[str, Any]]:
        """Extract structured product rows from the catalog's text layout."""
        context = " ".join(document.page_content for document in documents)
        rows = re.findall(
            r"\b(\d{2})\s+(.+?)\s+(Clothing|Shoes|Accessories)\s+"
            r"\$(\d+\.\d{2})\s+(.*?)(?=\s+\d{2}\s+[A-Z].+?\s+"
            r"(?:Clothing|Shoes|Accessories)\s+\$|$)",
            context,
        )
        products = []
        for product_id, name, category, price, description in rows:
            description = re.sub(r"\s+\d+$", "", description).strip(" .")
            products.append(
                {
                    "product_id": product_id,
                    "name": name.strip(),
                    "category": category,
                    "price": float(price),
                    "description": f"{description}.",
                }
            )
        return products

    def _load_or_build(self, pdf_bytes: bytes) -> None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        documents = self._extract_documents(pdf_bytes)
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise ValueError("No text chunks were created from the uploaded PDF.")
        for chunk_number, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = chunk_number
            chunk.metadata["source"] = self.filename

        self.products = self._extract_products(documents)
        with connection() as conn:
            existing = conn.execute(
                "SELECT COUNT(*) AS count FROM catalog_chunks WHERE document_hash = %s",
                (self.document_hash,),
            ).fetchone()
            if existing and int(existing["count"]) == len(chunks):
                self.chunk_count = len(chunks)
                return

            conn.execute(
                "DELETE FROM catalog_chunks WHERE document_hash = %s",
                (self.document_hash,),
            )
            embeddings = self._get_embedding_model().embed_documents(
                [chunk.page_content for chunk in chunks]
            )
            conn.executemany(
                """
                INSERT INTO catalog_chunks
                    (document_hash, chunk_id, source_file, page, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        self.document_hash,
                        chunk.metadata["chunk_id"],
                        self.filename,
                        chunk.metadata.get("page"),
                        chunk.page_content,
                        embedding,
                    )
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ],
            )
            self.chunk_count = len(chunks)

    def search(
        self,
        query: str,
        limit: int = 4,
        max_distance: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return relevant LangChain documents and similarity distances.

        When max_distance is given, matches whose distance exceeds it are
        dropped. Without this, similarity_search_with_score always
        returns its top-k regardless of how poor the match is, which is
        what let unrelated questions surface catalog chunks.
        """
        if not query.strip():
            return []
        query_embedding = self._get_embedding_model().embed_query(query)
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT content, page, embedding <=> %s AS distance
                FROM catalog_chunks
                WHERE document_hash = %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, self.document_hash, query_embedding, limit),
            ).fetchall()
        results = [
            {
                "text": row["content"],
                "page": row["page"] or "?",
                "score": round(float(row["distance"]), 4),
            }
            for row in rows
        ]
        if max_distance is not None:
            results = [item for item in results if item["score"] <= max_distance]
        return results

    def product_by_id(self, product_id: str) -> dict[str, Any] | None:
        return next(
            (product for product in self.products if product["product_id"].lower() == product_id.lower()),
            None,
        )

    def answer(self, query: str) -> str:
        """Generate an answer from retrieved catalog context.

        Structured keyword matches (via _matching_products) are tried
        first since they're precise. Otherwise, vector search results
        are filtered by RELEVANCE_DISTANCE: if nothing survives the
        filter, the catalog genuinely has nothing relevant, and we say
        so instead of formatting whatever the top-k happened to return.
        """
        structured_matches = self._matching_products(query)
        if structured_matches:
            return self._format_products(structured_matches)

        results = self.search(query, max_distance=RELEVANCE_DISTANCE)
        if not results:
            return "I could not find a matching product in the uploaded catalog."

        context = "\n\n".join(
            f"[Page {item['page']}] {item['text']}" for item in results
        )
        use_generation = os.getenv("ANYCART_GENERATE_ANSWERS", "1").lower() not in {
            "0", "false", "no"
        }
        if use_generation:
            try:
                prompt = (
                    "Answer the question using only the catalog context. "
                    "Return only a concise answer with a bullet list when listing products. "
                    "For each product include product name, product ID, price, and one short description. "
                    "Do not mention pages, similarity scores, context, or these instructions. "
                    "If the answer is not present, say you cannot find it. "
                    f"\nQuestion: {query}\nContext:\n{context}"
                )
                generated = generate_answer(prompt)
                return f"{generated.strip()}\n\nSources: " + ", ".join(
                    f"page {item['page']}" for item in results
                )
            except (ImportError, RuntimeError, OSError, ValueError):
                pass
        concise = self._concise_catalog_answer(query, results)
        if concise:
            return concise
        return "Based on the uploaded catalog:\n\n" + "\n\n".join(
            f"- {item['text']}" for item in results
        )

    def _matching_products(self, query: str) -> list[dict[str, Any]]:
        """Rank structured PDF products for precise catalog questions."""
        stop_words = {
            "what", "which", "kind", "type", "do", "you", "have", "show",
            "me", "some", "find", "available", "product", "products", "the",
            "a", "an", "for", "please", "can", "i", "get", "want",
        }
        query_terms = {
            self._normalise_word(word)
            for word in re.findall(r"[a-zA-Z0-9-]+", query.lower())
            if len(word) > 2 and word not in stop_words
        }
        if not query_terms:
            return []
        ranked: list[tuple[int, dict[str, Any]]] = []
        for product in self.products:
            product_terms = {
                self._normalise_word(word)
                for word in re.findall(
                    r"[a-zA-Z0-9-]+",
                    " ".join(str(product[field]) for field in ("name", "category", "description")).lower(),
                )
            }
            score = sum(term in product_terms for term in query_terms)
            if score:
                ranked.append((score, product))
        if not ranked:
            return []
        best_score = max(score for score, _ in ranked)
        return [product for score, product in ranked if score == best_score][:6]

    @staticmethod
    def _normalise_word(word: str) -> str:
        word = word.lower().strip("-")
        return word[:-1] if word.endswith("s") and len(word) > 3 else word

    @staticmethod
    def _format_products(products: list[dict[str, Any]]) -> str:
        lines = [
            f"- **{product['name']}** (product_id: {product['product_id']}) - ${product['price']:.2f}\n"
            f"  {product['description']}"
            for product in products
        ]
        return "Here are the matching products:\n\n" + "\n".join(lines)

    @staticmethod
    def _concise_catalog_answer(query: str, results: list[dict[str, Any]]) -> str:
        """Format simple product rows when an LLM is unavailable or verbose.

        `results` has already passed the RELEVANCE_DISTANCE filter in
        answer(), so this only ever formats chunks judged relevant --
        it no longer needs its own guard against dumping the whole
        catalog.
        """
        context = " ".join(item["text"] for item in results)
        rows = re.findall(
            r"\b(\d{2})\s+(.+?)\s+(Clothing|Shoes|Accessories)\s+"
            r"\$(\d+\.\d{2})\s+(.*?)(?=\s+\d{2}\s+[A-Z].+?\s+"
            r"(?:Clothing|Shoes|Accessories)\s+\$|$)",
            context,
        )
        if not rows:
            return ""
        query_lower = query.lower()
        requested_category = next(
            (category for category in ("shoes", "clothing", "accessories") if category in query_lower),
            None,
        )
        selected = [row for row in rows if not requested_category or row[2].lower() == requested_category]
        if not selected:
            return ""
        lines = []
        for product_id, name, category, price, description in selected[:6]:
            description = re.sub(r"\s+\d+$", "", description).strip(" .")
            lines.append(
                f"- **{name.strip()}** (product_id: {product_id}) - ${price}\n"
                f"  {description}."
            )
        return "Here are the available products:\n\n" + "\n".join(lines)

    def status(self) -> dict[str, Any]:
        return {
            "document": self.filename,
            "document_hash": self.document_hash,
            "embedding_model": self.embedding_model_name,
            "chunk_count": self.chunk_count,
            "products": len(self.products),
            "generation_model": self.generation_model_name,
            "llm_provider": provider_name(),
            "vector_store": "postgresql+pgvector",
        }


_active_catalog: CatalogRAG | None = None


def set_catalog(pdf_bytes: bytes, filename: str = "catalog.pdf") -> CatalogRAG:
    """Build and activate a PostgreSQL pgvector index from uploaded PDF bytes."""
    global _active_catalog
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("The catalog PDF is larger than the 25 MB upload limit.")
    _active_catalog = CatalogRAG(pdf_bytes, filename)
    return _active_catalog


def active_catalog() -> CatalogRAG | None:
    return _active_catalog
