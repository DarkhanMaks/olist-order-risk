"""Train a logistic-regression baseline and a HistGradientBoosting model on a time-based split."""
import json

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import DB_PATH, ROOT

NUM = ["estimated_days", "shipping_days", "delivery_delay_days", "is_late", "n_items",
       "total_price", "total_freight", "freight_ratio", "max_installments",
       "customer_prev_orders", "seller_prev_reviews", "seller_prev_bad_rate"]
CAT = ["main_payment_type"]
TARGET = "bad_review"
PRECISION_TARGET = 0.5  # business rule for the threshold, see pick_threshold()


def make_logreg():
    pre = ColumnTransformer([
        # median-fill gaps; add_indicator adds a "was missing" column (e.g. seller with no history)
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUM),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), CAT),
    ])
    return Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000))])


def make_hgb():
    pre = ColumnTransformer([
        ("num", "passthrough", NUM),  # HistGradientBoosting handles NaN natively
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CAT),
    ])
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, l2_regularization=1.0,
        categorical_features=[False] * len(NUM) + [True], random_state=42)
    return Pipeline([("prep", pre), ("clf", clf)])


def ks_stat(y, p):
    """Max gap between the cumulative score distributions of bad and good orders."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def scores(y, p):
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "ks": ks_stat(y, p)}


def pick_threshold(y, p, precision_target=PRECISION_TARGET):
    """Lowest threshold (= highest recall) whose precision is still >= the target."""
    precision, _, thresholds = precision_recall_curve(y, p)
    ok = precision[:-1] >= precision_target
    return float(thresholds[ok].min()) if ok.any() else 0.5


def main():
    df = duckdb.connect(str(DB_PATH), read_only=True).execute("SELECT * FROM features").df()
    df[NUM] = df[NUM].astype("float64")  # DuckDB gives nullable Int64 (pd.NA); sklearn wants plain NaN

    # --- time-based split: train on the past, test on the future ---------------------
    cut = df["purchase_at"].quantile(0.8)
    train, test = df[df["purchase_at"] <= cut], df[df["purchase_at"] > cut]
    print(f"train: {len(train):,} orders up to {cut:%Y-%m-%d}, bad rate {train[TARGET].mean():.3f}")
    print(f"test : {len(test):,} orders after that,    bad rate {test[TARGET].mean():.3f}")
    X_tr, y_tr, X_te, y_te = train[NUM + CAT], train[TARGET], test[NUM + CAT], test[TARGET]

    # --- models -----------------------------------------------------------------------
    logreg = make_logreg().fit(X_tr, y_tr)
    hgb = make_hgb().fit(X_tr, y_tr)
    p_log = logreg.predict_proba(X_te)[:, 1]
    p_hgb = hgb.predict_proba(X_te)[:, 1]

    results = pd.DataFrame({
        "rule: is_late": scores(y_te, test["is_late"]),
        "logistic regression": scores(y_te, p_log),
        "hist gradient boosting": scores(y_te, p_hgb),
    }).T
    print("\nTEST metrics\n", results.round(3).to_string())

    # --- threshold: chosen on a validation slice of the TRAIN period, never on test ---
    inner_cut = train["purchase_at"].quantile(0.75)
    fit, val = train[train["purchase_at"] <= inner_cut], train[train["purchase_at"] > inner_cut]
    p_val = make_hgb().fit(fit[NUM + CAT], fit[TARGET]).predict_proba(val[NUM + CAT])[:, 1]
    thr = pick_threshold(val[TARGET], p_val)
    print(f"\nthreshold {thr:.3f} (max recall on validation with precision >= {PRECISION_TARGET})")

    pred = (p_hgb >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    cm = pd.DataFrame([[tn, fp], [fn, tp]], index=["actual good", "actual bad"],
                      columns=["flagged: no", "flagged: yes"])
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    print("\nconfusion matrix on TEST at that threshold\n", cm.to_string())
    print(f"flagged {pred.mean():.1%} of orders | precision {precision:.2f} | recall {recall:.2f}")

    # --- permutation importance on test: how much does shuffling a column hurt PR-AUC? ---
    imp = permutation_importance(hgb, X_te, y_te, scoring="average_precision",
                                 n_repeats=5, random_state=42, n_jobs=-1)
    imp = pd.Series(imp.importances_mean, index=NUM + CAT).sort_values()
    print("\npermutation importance (drop in PR-AUC)\n", imp.iloc[::-1].round(4).to_string())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color="#2a6f97", height=0.6)
    ax.set_title("What the model relies on (drop in PR-AUC when shuffled)", loc="left", fontsize=11)
    ax.grid(axis="x", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "permutation_importance.png", dpi=150)

    # --- save: model + threshold + column lists, so predict.py needs nothing else -----
    joblib.dump({"model": hgb, "threshold": thr, "num": NUM, "cat": CAT}, ROOT / "models" / "model.joblib")
    metrics = {"split_date": str(cut.date()), "threshold": thr, "precision": precision, "recall": recall,
               "flagged_share": float(pred.mean()), "test": results.to_dict("index")}
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nsaved models/model.joblib, reports/metrics.json, reports/permutation_importance.png")


if __name__ == "__main__":
    main()
