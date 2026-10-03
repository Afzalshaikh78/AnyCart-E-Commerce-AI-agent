"""Privacy-safe operational metrics stored alongside application data."""

from __future__ import annotations

import json
import time
from typing import Any

from .database import connection


def elapsed_ms(started_at: float) -> float:
    """Return a monotonic elapsed duration suitable for request measurements."""
    return round((time.perf_counter() - started_at) * 1000, 2)


def record_metric(
    operation: str,
    duration_ms: float,
    success: bool,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist aggregate-ready telemetry without storing prompts or PDF content."""
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO api_metrics (operation, duration_ms, success, metadata)
            VALUES (%s, %s, %s, %s::jsonb)
            """,
            (operation, duration_ms, success, json.dumps(metadata or {})),
        )


def metrics_summary(window_days: int = 30) -> dict[str, Any]:
    """Return real service metrics collected during the selected time window."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT
                operation,
                COUNT(*) AS requests,
                COUNT(*) FILTER (WHERE success) AS successful_requests,
                ROUND(AVG(duration_ms)::numeric, 2) AS average_latency_ms,
                ROUND(
                    (percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms))::numeric,
                    2
                ) AS p95_latency_ms
            FROM api_metrics
            WHERE created_at >= NOW() - (%s * INTERVAL '1 day')
            GROUP BY operation
            ORDER BY operation
            """,
            (window_days,),
        ).fetchall()
        catalog = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM products) AS products,
                (SELECT COUNT(*) FROM catalog_chunks) AS chunks,
                (SELECT COUNT(*) FROM orders) AS orders
            """
        ).fetchone()

    operations = {}
    for row in rows:
        requests = int(row["requests"])
        successful_requests = int(row["successful_requests"])
        operations[row["operation"]] = {
            "requests": requests,
            "successful_requests": successful_requests,
            "success_rate": round((successful_requests / requests) * 100, 1) if requests else 0,
            "average_latency_ms": float(row["average_latency_ms"]),
            "p95_latency_ms": float(row["p95_latency_ms"]),
        }

    return {
        "window_days": window_days,
        "catalog": {
            "products": int(catalog["products"]),
            "chunks": int(catalog["chunks"]),
            "orders": int(catalog["orders"]),
        },
        "operations": operations,
    }
