"""Minimal scoring API. Run: uvicorn src.api:app --reload  (docs at http://127.0.0.1:8000/docs)"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import score_order

app = FastAPI(title="Olist order risk")


class OrderFeatures(BaseModel):
    # known at delivery time; same definitions as sql/features.sql
    estimated_days: float            # promised delivery time, days since purchase
    shipping_days: float             # actual delivery time, days since purchase
    delivery_delay_days: float       # actual - promised; > 0 means late
    n_items: int = 1
    freight_ratio: float             # freight cost / item price
    customer_prev_orders: int = 0
    seller_prev_reviews: int = 0
    seller_prev_bad_rate: float | None = None  # None = seller has no history


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(order: OrderFeatures):
    return score_order(order.model_dump())
