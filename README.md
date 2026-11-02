# Olist Order Risk: predict bad reviews

Flag delivered orders that are likely to get a bad review (score <= 2) so customer support can reach out early.

## Business framing

- **Hypothesis:** delivery delays, long shipping times and high freight cost make a bad review more likely.
- **Target:** `bad_review = 1` if `review_score <= 2` (12.8% of 95,824 delivered, reviewed orders).
- **Use:** when an order is delivered, score it. Support contacts the flagged orders (offer help, a refund or a coupon) before the customer leaves a bad review.
- **Data:** [Olist Brazilian E-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), 9 CSV tables, orders from 2016-09 to 2018-08.

## Pipeline

| Step | File | What it does |
|---|---|---|
| Load | `src/load_data.py` | CSVs into DuckDB (`data/olist.duckdb`) |
| Check | `sql/sanity_checks.sql` | Row counts, nulls, join keys, duplicate reviews |
| Features | `sql/features.sql` | One row per order, point-in-time features, no target leakage |
| EDA | `src/eda.py` | Bad-review rate by delay, shipping time, freight, seller history |
| Model | `src/train.py` | Time-based split, logistic baseline, gradient boosting, threshold |
| Serve | `src/predict.py`, `src/api.py` | Score one order, FastAPI `/score` |

## How to run

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# put the Kaggle CSVs into data/raw/, then:
python -m src.load_data
python -m src.run_sanity
python -m src.build_features
python -m src.eda
python -m src.train
pytest
uvicorn src.api:app --reload      # docs and a test form: http://127.0.0.1:8000/docs
```

Example request to `POST /score`:

```json
{"estimated_days": 25, "shipping_days": 40, "delivery_delay_days": 15,
 "total_price": 100, "total_freight": 20, "seller_prev_bad_rate": 0.1,
 "main_payment_type": "credit_card"}
```

Response: `{"risk_score": 0.837, "flagged": true, "threshold": 0.2274}`. With `shipping_days: 10` and `delivery_delay_days: -15` (15 days early) the same order is not flagged.

## Results

Time-based split: train on orders up to 2018-05-27 (76,659 orders, 13.6% bad), test on later orders (19,165 orders, 9.7% bad).

| Model (test set) | ROC-AUC | PR-AUC | KS |
|---|---|---|---|
| Rule: "is late" | 0.571 | 0.139 | 0.141 |
| Logistic regression | 0.700 | 0.253 | 0.305 |
| HistGradientBoosting | 0.715 | 0.316 | 0.323 |

**Threshold 0.227.** Chosen on a validation slice of the train period: the highest recall that keeps precision at 0.5 or more, so that at least half of the contacted customers really are at risk. On the test set it flags 8.3% of orders with precision 0.38 and recall 0.33 (603 bad reviews caught, 1,251 missed, 995 false alarms).

**What drives the score** (permutation importance, `reports/permutation_importance.png`): delivery delay by far, then number of items and the seller's earlier bad-review rate. Freight ratio and price add nothing.

**Hypothesis check** (`reports/eda_conclusion.md`): delay is confirmed (3+ days late gives 61-79% bad reviews against 9-11% for early orders). Shipping time matters only in the slowest 10%. Freight cost does not matter.

## Limitations

- **Moderate accuracy.** ROC-AUC 0.715. Many bad reviews have causes not in this data (product quality, wrong item), so recall is 0.33 at the chosen threshold.
- **Threshold drift.** Precision was 0.5 on validation but 0.38 on test, because the bad-review rate fell from 13.6% to 9.7%. In production the threshold needs to be re-checked regularly.
- **Scored at delivery, not earlier.** The strongest features (delay, shipping days) are only known when the order arrives. Scoring while the order is still in transit would need a separate model with only early signals.
- **The API takes features, not an order id.** The caller must compute the features the way `sql/features.sql` does. There is no feature store, monitoring or retraining.
- **Associations, not causes.** Late orders are also slow orders, and the model does not separate the effects.
- **Coverage.** Only delivered orders that have a review are in the data. Cancelled orders and customers who never review are not covered.
