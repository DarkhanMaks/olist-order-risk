-- row counts per table
SELECT 'orders' AS tbl, count(*) AS n FROM orders
UNION ALL SELECT 'customers', count(*) FROM customers
UNION ALL SELECT 'order_items', count(*) FROM order_items
UNION ALL SELECT 'order_payments', count(*) FROM order_payments
UNION ALL SELECT 'order_reviews', count(*) FROM order_reviews
UNION ALL SELECT 'products', count(*) FROM products
UNION ALL SELECT 'sellers', count(*) FROM sellers;

-- orders: status distribution
SELECT order_status, count(*) AS n,
       round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
FROM orders
GROUP BY 1
ORDER BY n DESC;

-- orders: nulls in key columns, and date range
SELECT count(*) AS total,
       count(*) - count(order_purchase_timestamp)        AS null_purchase,
       count(*) - count(order_delivered_customer_date)   AS null_delivered,
       count(*) - count(order_estimated_delivery_date)   AS null_estimated,
       min(order_purchase_timestamp)                     AS first_order,
       max(order_purchase_timestamp)                     AS last_order
FROM orders;

-- delivered orders that still have no delivery date (data quality)
SELECT count(*) AS delivered_without_date
FROM orders
WHERE order_status = 'delivered' AND order_delivered_customer_date IS NULL;

-- join keys: is orders.order_id unique, and who is missing from each child table?
SELECT count(*) AS orders,
       count(DISTINCT order_id) AS distinct_orders,
       count(*) FILTER (WHERE order_id NOT IN (SELECT order_id FROM order_items))    AS no_items,
       count(*) FILTER (WHERE order_id NOT IN (SELECT order_id FROM order_payments)) AS no_payments,
       count(*) FILTER (WHERE order_id NOT IN (SELECT order_id FROM order_reviews))  AS no_reviews,
       count(*) FILTER (WHERE customer_id NOT IN (SELECT customer_id FROM customers)) AS no_customer
FROM orders;

-- reviews: some orders have more than one review (we will need one row per order)
SELECT n_reviews, count(*) AS n_orders
FROM (SELECT order_id, count(*) AS n_reviews FROM order_reviews GROUP BY 1)
GROUP BY 1
ORDER BY 1;

-- reviews: score distribution and bad-review rate (score <= 2)
SELECT review_score, count(*) AS n,
       round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
FROM order_reviews
GROUP BY 1
ORDER BY 1;

-- order_items: any items pointing to unknown orders / sellers / products?
SELECT count(*) FILTER (WHERE order_id   NOT IN (SELECT order_id   FROM orders))   AS orphan_orders,
       count(*) FILTER (WHERE seller_id  NOT IN (SELECT seller_id  FROM sellers))  AS orphan_sellers,
       count(*) FILTER (WHERE product_id NOT IN (SELECT product_id FROM products)) AS orphan_products
FROM order_items;
