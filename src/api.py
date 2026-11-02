"""Minimal scoring API. Run: uvicorn src.api:app --reload  (docs at http://127.0.0.1:8000/docs)"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import score_order

app = FastAPI(title="Olist order risk")


class OrderFeatures(BaseModel):
    # known at delivery time; same definitions as sql/features.sql
    estimated_days: float = Field(description="promised delivery time, days since purchase")
    shipping_days: float = Field(description="actual delivery time, days since purchase")
    delivery_delay_days: float = Field(description="actual - promised; > 0 means late")
    n_items: int = Field(1, ge=1)
    total_price: float = Field(ge=0)
    total_freight: float = Field(ge=0)
    max_installments: float | None = None
    customer_prev_orders: int = Field(0, ge=0)
    seller_prev_reviews: int = Field(0, ge=0)
    seller_prev_bad_rate: float | None = Field(None, ge=0, le=1)  # null = seller has no history
    main_payment_type: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(order: OrderFeatures):
    return score_order(order.model_dump())
