"""Score one order with the saved model."""
import joblib
import pandas as pd

from src.config import MODEL_PATH

artifact = joblib.load(MODEL_PATH)  # {"model", "threshold", "features"}


def build_row(order):
    """One-row DataFrame with the columns in the order the model was trained on."""
    return pd.DataFrame([order]).reindex(columns=artifact["features"]).astype("float64")


def score_order(order):
    p = float(artifact["model"].predict_proba(build_row(order))[0, 1])
    return {"risk_score": round(p, 4), "flagged": bool(p >= artifact["threshold"]),
            "threshold": round(artifact["threshold"], 4)}
