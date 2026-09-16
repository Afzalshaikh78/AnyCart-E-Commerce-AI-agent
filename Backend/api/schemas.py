from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class OrderRequest(BaseModel):
    product_id: str
    quantity: int = Field(default=1, ge=1, le=100)

