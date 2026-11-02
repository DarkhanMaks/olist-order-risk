"""Score one order with the saved model."""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import ROOT

MODEL_PATH = ROOT / "models" / "model.joblib"


@lru_cache(maxsize=1)
def load_artifact():
    # loaded once, then reused for every request
    return joblib.load(MODEL_PATH)


def build_row(order):
    """Turn a dict of order features into the one-row DataFrame the model expects."""
    order = dict(order)
    if order.get("is_late") is None:
        order["is_late"] = int(order["delivery_delay_days"] > 0)
    if order.get("freight_ratio") is None and order.get("total_price"):
        order["freight_ratio"] = order["total_freight"] / order["total_price"]
    art = load_artifact()
    cols = art["num"] + art["cat"]
    row = pd.DataFrame([order]).reindex(columns=cols)  # absent columns become NaN
    row[art["num"]] = row[art["num"]].astype("float64")
    return row


def score_order(order):
    art = load_artifact()
    p = float(art["model"].predict_proba(build_row(order))[0, 1])
    return {"risk_score": round(p, 4), "flagged": bool(p >= art["threshold"]),
            "threshold": round(art["threshold"], 4)}
