import re
import uuid
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from .database import connection

load_dotenv(Path(__file__).resolve().parent / ".env")
from .local_rag import active_catalog


class AgentState(TypedDict, total=False):
    message: str
    intent: str
    product_id: str
    quantity: int
    order_id: str
    response: str


def classify(message: str) -> dict[str, Any]:
    lower = message.lower()
    if _is_conversational(lower):
        return {"intent": "conversation"}
    product_id = _product_id(message)
    order_id = _order_id(message)
    if order_id and any(word in lower for word in ("track", "trace", "status", "delivery", "where", "check")):
        return {"intent": "track", "order_id": order_id}
    if product_id and any(word in lower for word in ("buy", "order", "purchase")):
        return {"intent": "order", "product_id": product_id, "quantity": _quantity(message)}
    return {"intent": "recommend"}


def _is_conversational(message: str) -> bool:
    """Keep greetings and assistant-capability questions out of product RAG."""
    normalized = re.sub(r"[^a-z\s]", " ", message.lower())
    words = set(normalized.split())
    conversational_phrases = (
        {"hi"}, {"hello"}, {"hey"}, {"hey", "hello"}, {"thanks"},
        {"thank", "you"}, {"good", "morning"}, {"good", "afternoon"},
        {"good", "evening"}, {"who", "are", "you"},
        {"what", "can", "you", "do"}, {"how", "can", "you", "help"},
    )
    return any(words == phrase for phrase in conversational_phrases)


def _product_id(message: str) -> str:
    match = re.search(r"\b(?:product[_ -]?id|product)\s*[:=]?\s*([A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?)\b", message, re.I)
    return match.group(1).upper().replace("_", "-") if match else ""


def _order_id(message: str) -> str:
    match = re.search(r"\b(?:order[_ -]?id|order)\s*[:=]?\s*([a-f0-9-]{36})\b", message, re.I)
    if not match:
        match = re.search(r"\b([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\b", message, re.I)
    return match.group(1) if match else ""


def _quantity(message: str) -> int:
    match = re.search(r"\b(?:quantity|qty|x|of)\s*[:=]?\s*(\d+)\b", message, re.I)
    return int(match.group(1)) if match else 1


@tool
def recommend_catalog(prompt: str) -> str:
    """Answer a product question from the uploaded PDF catalog."""
    catalog = active_catalog()
    return catalog.answer(prompt) if catalog else "Upload a catalog PDF first."


@tool
def create_order(product_id: str, quantity: int) -> dict[str, Any]:
    """Create an order in PostgreSQL after validating the uploaded catalog product."""
    catalog = active_catalog()
    if not catalog:
        return {"error": "Upload a catalog PDF first."}
    product = catalog.product_by_id(product_id)
    if not product:
        return {"error": f"Product {product_id} is not in the uploaded catalog."}
    order_id = str(uuid.uuid4())
    with connection() as conn:
        conn.execute(
            "INSERT INTO orders(order_id, product_id, quantity) VALUES (%s, %s, %s)",
            (order_id, product["product_id"], quantity),
        )
    return {"order_id": order_id, "product_id": product["product_id"], "quantity": quantity, "status": "Pending"}


@tool
def lookup_order(order_id: str) -> dict[str, Any]:
    """Look up an order in PostgreSQL by order ID."""
    with connection() as conn:
        row = conn.execute(
            "SELECT order_id::text, product_id, quantity, status, created_at FROM orders WHERE order_id = %s",
            (order_id,),
        ).fetchone()
    return dict(row) if row else {"error": f"Order {order_id} was not found."}


def route(state: AgentState) -> str:
    return {"conversation": "conversation", "recommend": "recommendation", "order": "order", "track": "tracking"}.get(
        state.get("intent", "recommend"), "recommendation"
    )


def router(state: AgentState) -> dict[str, Any]:
    return classify(state["message"])


def recommendation_node(state: AgentState) -> dict[str, str]:
    return {"response": recommend_catalog.invoke({"prompt": state["message"]})}


def conversation_node(state: AgentState) -> dict[str, str]:
    lower = state["message"].lower()
    if "thank" in lower:
        response = "You're welcome! I can help you find products, place orders, or track orders."
    elif "who" in lower:
        response = "I'm AnyCart, your shopping assistant. I use the uploaded catalog to recommend products and manage orders."
    elif "what can" in lower or "how can" in lower:
        response = "I can recommend products from your uploaded PDF catalog, place orders, and track existing orders."
    else:
        response = "Hi! I'm AnyCart, your shopping assistant. Ask me about products in your uploaded catalog."
    return {"response": response}


def order_node(state: AgentState) -> dict[str, str]:
    result = create_order.invoke({"product_id": state["product_id"], "quantity": state["quantity"]})
    return {"response": result.get("error", f"Order placed! Details: {result}")}


def tracking_node(state: AgentState) -> dict[str, str]:
    result = lookup_order.invoke({"order_id": state["order_id"]})
    if "error" in result:
        return {"response": result["error"]}
    return {
        "response": (
            f"Order {result['order_id']} is {result['status']}. "
            f"Product: {result['product_id']}, quantity: {result['quantity']}."
        )
    }


builder = StateGraph(AgentState)
builder.add_node("router", router)
builder.add_node("conversation", conversation_node)
builder.add_node("recommendation", recommendation_node)
builder.add_node("order", order_node)
builder.add_node("tracking", tracking_node)
builder.add_edge(START, "router")
# builder.add_edge(START, "router")
builder.add_conditional_edges("router", route)
builder.add_edge("conversation", END)
builder.add_edge("recommendation", END)
builder.add_edge("order", END)
builder.add_edge("tracking", END)
agent_graph = builder.compile()
