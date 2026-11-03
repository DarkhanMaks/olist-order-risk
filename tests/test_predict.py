import pytest

from src.config import MODEL_PATH

if not MODEL_PATH.exists():
    pytest.skip("run `python -m src.train` first", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from src.api import app  # noqa: E402
from src.predict import build_row, score_order  # noqa: E402

ORDER = {"estimated_days": 25.0, "shipping_days": 10.0, "delivery_delay_days": -15.0,
         "n_items": 1, "freight_ratio": 0.2, "customer_prev_orders": 0,
         "seller_prev_reviews": 50, "seller_prev_bad_rate": 0.1}


def test_build_row_keeps_column_order_and_missing_values():
    row = build_row({"estimated_days": 25.0})  # all other features missing
    assert row.shape == (1, 8)
    assert row.loc[0, "estimated_days"] == 25.0
    assert row["seller_prev_bad_rate"].isna().all()


def test_late_order_is_riskier_than_early_order():
    late = {**ORDER, "shipping_days": 40.0, "delivery_delay_days": 15.0}
    early = score_order(ORDER)["risk_score"]
    assert 0 <= early <= 1
    assert score_order(late)["risk_score"] > early


def test_score_endpoint():
    client = TestClient(app)
    ok = client.post("/score", json=ORDER)
    assert ok.status_code == 200
    assert set(ok.json()) == {"risk_score", "flagged", "threshold"}
    bad = client.post("/score", json={"n_items": 1})  # required fields missing
    assert bad.status_code == 422
