"""Train a logistic-regression baseline and a HistGradientBoosting model on a time-based split."""
import json

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DB_PATH, MODEL_PATH, ROOT

FEATURES = ["estimated_days", "shipping_days", "delivery_delay_days", "n_items", "freight_ratio",
            "customer_prev_orders", "seller_prev_reviews", "seller_prev_bad_rate"]
TARGET = "bad_review"
FLAG_SHARE = 0.10  # support can contact about 10% of delivered orders


def ks_stat(y, p):
    """Max gap between the cumulative score distributions of bad and good orders."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def scores(y, p):
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "ks": ks_stat(y, p)}


def main():
    df = duckdb.connect(str(DB_PATH), read_only=True).execute("SELECT * FROM features").df()
    df[FEATURES] = df[FEATURES].astype("float64")  # DuckDB gives nullable Int64 (pd.NA); sklearn wants NaN

    # --- time-based split: train (oldest 60%) -> validation (next 20%) -> test (newest 20%) ---
    cut_val, cut_test = df["purchase_at"].quantile([0.6, 0.8])
    train = df[df["purchase_at"] <= cut_val]
    val = df[(df["purchase_at"] > cut_val) & (df["purchase_at"] <= cut_test)]
    test = df[df["purchase_at"] > cut_test]
    for name, part in [("train", train), ("val  ", val), ("test ", test)]:
        print(f"{name}: {len(part):,} orders, bad rate {part[TARGET].mean():.3f}")
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]

    # --- models -----------------------------------------------------------------------
    # logistic regression needs complete, scaled numbers; add_indicator marks "was missing"
    logreg = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                           LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    # gradient boosting handles missing values (NaN) by itself
    hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                                         l2_regularization=1.0, random_state=42).fit(X_tr, y_tr)
    p_log = logreg.predict_proba(X_te)[:, 1]
    p_hgb = hgb.predict_proba(X_te)[:, 1]

    results = pd.DataFrame({
        "rule: delivered late": scores(y_te, (test["delivery_delay_days"] > 0).astype(int)),
        "logistic regression": scores(y_te, p_log),
        "hist gradient boosting": scores(y_te, p_hgb),
    }).T
    print("\nTEST metrics\n", results.round(3).to_string())

    # --- threshold: the score that flags the top 10% of VALIDATION orders, applied once to test ---
    thr = float(np.quantile(hgb.predict_proba(val[FEATURES])[:, 1], 1 - FLAG_SHARE))
    pred = (p_hgb >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    cm = pd.DataFrame([[tn, fp], [fn, tp]], index=["actual good", "actual bad"],
                      columns=["flagged: no", "flagged: yes"])
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    print(f"\nthreshold {thr:.3f} = score of the top {FLAG_SHARE:.0%} of validation orders")
    print("confusion matrix on TEST\n", cm.to_string())
    print(f"flagged {pred.mean():.1%} of orders | precision {precision:.2f} | recall {recall:.2f}")

    # --- permutation importance on test: how much does shuffling a column hurt PR-AUC? ---
    imp = permutation_importance(hgb, X_te, y_te, scoring="average_precision",
                                 n_repeats=5, random_state=42, n_jobs=-1)
    imp = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
    print("\npermutation importance (drop in PR-AUC)\n", imp.iloc[::-1].round(4).to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color="#2a6f97", height=0.6)
    ax.set_title("What the model relies on (drop in PR-AUC when shuffled)", loc="left", fontsize=11)
    ax.grid(axis="x", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "permutation_importance.png", dpi=150)

    # --- save the model and threshold so predict.py needs nothing else ----------------
    joblib.dump({"model": hgb, "threshold": thr, "features": FEATURES}, MODEL_PATH)
    metrics = {"val_starts": str(cut_val.date()), "test_starts": str(cut_test.date()),
               "flag_share": FLAG_SHARE, "threshold": thr,
               "precision": precision, "recall": recall, "test": results.to_dict("index")}
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nsaved models/model.joblib, reports/metrics.json, reports/permutation_importance.png")


if __name__ == "__main__":
    main()
