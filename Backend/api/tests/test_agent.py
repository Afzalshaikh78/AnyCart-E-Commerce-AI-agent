from Backend.api.agent import classify, agent_graph


def test_recommendation_is_default_intent() -> None:
    assert classify("show me running shoes") == {"intent": "recommend"}


def test_order_intent_extracts_product_and_quantity() -> None:
    result = classify("buy product 05 quantity 2")
    assert result["intent"] == "order"
    assert result["product_id"] == "05"
    assert result["quantity"] == 2


def test_greeting_uses_conversation_intent() -> None:
    assert classify("hey hello") == {"intent": "conversation"}


def test_greeting_does_not_query_the_catalog() -> None:
    result = agent_graph.invoke({"message": "hello"})
    assert result["response"].startswith("Hi!")
