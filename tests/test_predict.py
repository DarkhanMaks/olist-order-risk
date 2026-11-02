import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.predict import MODEL_PATH, build_row, score_order

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m src.train` first")

ORDER = {"estimated_days": 25.0, "shipping_days": 10.0, "delivery_delay_days": -15.0,
         "n_items": 1, "total_price": 100.0, "total_freight": 20.0,
         "max_installments": 1, "customer_prev_orders": 0, "seller_prev_reviews": 50,
         "seller_prev_bad_rate": 0.1, "main_payment_type": "credit_card"}


@needs_model
def test_build_row_derives_features():
    row = build_row(ORDER)
    assert row.loc[0, "is_late"] == 0  # delay < 0 -> not late
    assert row.loc[0, "freight_ratio"] == pytest.approx(0.2)


@needs_model
def test_late_order_is_riskier_than_early_order():
    late = {**ORDER, "shipping_days": 40.0, "delivery_delay_days": 15.0}
    early = score_order(ORDER)["risk_score"]
    assert 0 <= early <= 1
    assert score_order(late)["risk_score"] > early


@needs_model
def test_score_endpoint():
    client = TestClient(app)
    ok = client.post("/score", json=ORDER)
    assert ok.status_code == 200
    assert set(ok.json()) == {"risk_score", "flagged", "threshold"}
    bad = client.post("/score", json={"n_items": 1})  # required fields missing
    assert bad.status_code == 422
