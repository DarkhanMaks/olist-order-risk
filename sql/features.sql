-- One row per delivered order that has a review. Features are what we know when the order
-- is delivered ("as-of" time = delivered_at). Target: bad_review = review_score <= 2.
CREATE OR REPLACE TABLE features AS
WITH base AS (  -- delivered orders with a valid delivery date
    SELECT order_id,
           order_purchase_timestamp       AS purchase_at,
           order_delivered_customer_date  AS delivered_at,
           order_estimated_delivery_date  AS estimated_at
    FROM orders
    WHERE order_status = 'delivered'
      AND order_delivered_customer_date IS NOT NULL
      AND order_delivered_customer_date >= order_purchase_timestamp
),
reviews AS (  -- one review per order: keep the latest one
    SELECT order_id, review_score, review_creation_date
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY order_id
                                  ORDER BY review_creation_date DESC, review_answer_timestamp DESC) AS rn
        FROM order_reviews
    )
    WHERE rn = 1
),
items AS (  -- order_items has one row per item -> aggregate to one row per order
    SELECT order_id,
           count(*) AS n_items,
           sum(freight_value) / NULLIF(sum(price), 0) AS freight_ratio
    FROM order_items
    GROUP BY order_id
),
cust AS (  -- how many orders this person placed BEFORE this one (current row excluded)
    SELECT o.order_id,
           COUNT(*) OVER (PARTITION BY c.customer_unique_id
                          ORDER BY o.order_purchase_timestamp, o.order_id
                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS customer_prev_orders
    FROM orders o
    JOIN customers c USING (customer_id)
),
cur AS (  -- (order, seller) pairs we need a history for
    SELECT DISTINCT i.order_id, i.seller_id, b.delivered_at
    FROM order_items i
    JOIN base b USING (order_id)
),
past AS (  -- every labelled (seller, order) with the time its review appeared
    SELECT DISTINCT i.seller_id, i.order_id, r.review_score, r.review_creation_date
    FROM order_items i
    JOIN reviews r USING (order_id)
),
seller_hist AS (  -- no leakage: only reviews created BEFORE this order was delivered
    SELECT cur.order_id, cur.seller_id,
           count(past.order_id)                              AS prev_reviews,
           count(*) FILTER (WHERE past.review_score <= 2)    AS prev_bad
    FROM cur
    LEFT JOIN past
           ON past.seller_id = cur.seller_id
          AND past.order_id <> cur.order_id
          AND past.review_creation_date < cur.delivered_at
    GROUP BY cur.order_id, cur.seller_id
),
seller_agg AS (  -- an order can have several sellers -> pool their histories
    SELECT order_id,
           sum(prev_reviews)                                   AS seller_prev_reviews,
           sum(prev_bad)::DOUBLE / NULLIF(sum(prev_reviews), 0) AS seller_prev_bad_rate
    FROM seller_hist
    GROUP BY order_id
)
SELECT
    b.order_id,
    b.purchase_at,
    -- delivery timing, in days (delay > 0 means late)
    date_diff('second', b.purchase_at,  b.estimated_at) / 86400.0 AS estimated_days,
    date_diff('second', b.purchase_at,  b.delivered_at) / 86400.0 AS shipping_days,
    date_diff('second', b.estimated_at, b.delivered_at) / 86400.0 AS delivery_delay_days,
    -- order content and cost
    i.n_items,
    i.freight_ratio,
    -- history (point-in-time)
    cu.customer_prev_orders,
    sa.seller_prev_reviews,
    sa.seller_prev_bad_rate,
    -- target
    (r.review_score <= 2)::INT AS bad_review
FROM base b
JOIN reviews r  ON r.order_id  = b.order_id
JOIN items i    ON i.order_id  = b.order_id
JOIN cust cu    ON cu.order_id = b.order_id
LEFT JOIN seller_agg sa ON sa.order_id = b.order_id;
