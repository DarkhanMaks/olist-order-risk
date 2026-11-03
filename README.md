# Olist Order Risk

Predict whether a delivered order will get a bad review (score 1-2), so support can contact the customer first.

Data: [Olist Brazilian E-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle. Tools: DuckDB, pandas, scikit-learn, FastAPI.

## Idea

Orders that arrive late, take long to ship or have high freight cost probably get more bad reviews. I tested this and built a model that scores each order when it is delivered.

- Target: `bad_review = 1` if `review_score <= 2` (12.8% of 95,824 orders).
- Eight features, built in SQL ([sql/features.sql](sql/features.sql)): promised delivery time, actual shipping time, delay, number of items, freight ratio, the customer's earlier orders, and the seller's earlier number of reviews and bad-review rate.
- The seller features use only reviews the customer had already answered before the order was delivered, so they do not leak the future.

## What I found

![Bad-review rate by delivery delay](reports/eda_delay.png)

- Delay matters a lot: 3+ days late gives 61-79% bad reviews, early orders about 9-11%.
- Long shipping time matters only for the slowest 10% of orders.
- Freight cost does not matter (12-14% in every group).

## Model

I split the orders by time, because the model has to predict the future in real use: train on the oldest 60%, validation on the next 20% (from 2018-03-03), test on the newest 20% (from 2018-05-27). The bad-review rate is 13.3% in train, 14.5% in validation and 9.7% in test.

| Model (test) | ROC-AUC | PR-AUC |
|---|---|---|
| Rule: delivered late | 0.571 | 0.139 |
| Logistic regression | 0.669 | 0.228 |
| Gradient boosting | 0.710 | 0.311 |

Support can contact about 10% of orders, so I picked the score threshold (0.343) that flags the top 10% of validation orders, and applied it once to the test set. On test it flags only 3.8% of orders, because scores were lower in the newer period. Of those 733 flagged orders, 365 really got a bad review (precision 0.50, recall 0.20). Contacting random orders would hit only 9.7%, so the model is about 5 times better, but it finds only a fifth of the bad reviews.

The most important feature is the delivery delay, then the number of items and the seller's earlier bad-review rate (see `reports/permutation_importance.png`).

## Limits

- The model is only moderately accurate. It catches about 1 in 5 bad reviews, and many bad reviews have reasons that are not in the data.
- The score distribution drifts over time, so a fixed threshold flags 3.8% of test orders instead of 10%. In real use the threshold would need to be recalculated on recent orders.
- It scores at delivery time, so it cannot warn earlier.
- I did not tune the hyperparameters.

## Run it

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put the Kaggle CSV files into `data/raw/`, then:

```bash
python -m src.load_data
python -m src.build_features
python -m src.eda
python -m src.train
pytest
uvicorn src.api:app --reload
```

The API is at http://127.0.0.1:8000/docs. Send an order's features to `POST /score` and get back a risk score and a flag. For example, an order delivered 15 days late:

```json
{"estimated_days": 25, "shipping_days": 40, "delivery_delay_days": 15,
 "freight_ratio": 0.2, "seller_prev_bad_rate": 0.1}
```

returns a high risk score and `"flagged": true`. The same order delivered 15 days early gets a low score and is not flagged.
