# Olist Delivery Time Prediction

Predicting how many days an order will take to reach the customer, using **only information known at the moment of purchase**. The goal is to show customers an accurate delivery estimate at checkout.

## Results

Evaluated on 19,257 orders the models never saw (20% test split).

| Model | Average error (MAE, days) |
|---|---|
| Olist's current estimate (baseline) | 12.7 |
| Linear Regression | 5.1 |
| **Gradient Boosting, tuned (final)** | **4.3** |

## Repository

| Folder | Contents |
|---|---|
| `notebook/` | Full analysis: cleaning, EDA, features, models, evaluation |
| `app/` | Streamlit demo: a mock store that shows an estimated delivery date after "Order Now" |
| `data/` | Where to place the Olist CSV files (see `data/README.md`) |

## Approach and why

**1. One row per order.**
The raw data has one row per item, but delivery is a property of the order. I merged orders, items, customers, sellers, products and translations, then aggregated to one row per order. Of 99,441 orders, 96,281 (97%) were delivered and usable.

**2. Only features available at purchase.**
Anything known only after the order ships (carrier dates, delivery date) would leak the answer and make the model useless at checkout. Features include:
- distance between seller and customer (haversine, from zip-code coordinates)
- product weight and volume, price and freight
- seller's shipping time limit
- purchase month, weekday and hour
- whether seller and customer are in the same state
- the first item's category and the seller's state

**3. Metric: MAE (mean absolute error).**
It is in days, so it reads directly as "on average the estimate is off by N days", and it is not dominated by a few extreme orders the way RMSE is. RMSE and R² are reported as secondary metrics.

**4. Baseline: Olist's own estimate.**
The estimate the platform already shows customers is the natural baseline: any model has to beat it to be worth deploying.

**5. Split and outliers.**
Train / validation / test = 64 / 16 / 20 %. Extreme orders (over 100 days; 64 orders) were removed from **training only**, so the test score still reflects real conditions. Final models were refit on train + validation.

**6. Two models, compared fairly.**
- *Linear Regression*: simple, interpretable baseline (imputation, one-hot encoding of categories, scaling).
- *Histogram Gradient Boosting*: captures non-linear effects such as distance and season, handles categories and missing values natively, and uses an absolute-error loss that matches the MAE metric. Hyperparameters were tuned with randomized search (3-fold CV, MAE scoring).

**7. Total time vs. two legs.**
I also predicted purchase→carrier and carrier→customer separately and added them. The result (4.35 days) was no better than predicting the total directly (4.3), so the final model is the simpler direct one.

**8. Error analysis.**
Permutation importance shows which features drive predictions. The model is least accurate for remote states and during the Nov 2017 – Mar 2018 peak. The remaining error mostly comes from events not visible at checkout (carrier delays, holidays), which is why R² is about 0.33.

## Key data insights

- About 74% of total delivery time is shipping (carrier to customer); the rest is the seller's handling time.
- Orders within the same state arrive in about 6.6 days on average, vs about 12.8 days otherwise.
- Average delivery time rises with distance, from about 5 days for the nearest orders to about 18 for the farthest.
- Olist's estimate is conservative: on average 23.7 days vs 12.6 days actual.

## Limitations and next steps

- Add seller and carrier history (for example a seller's past handling time), computed only from earlier orders.
- Add external factors such as weather and holiday calendars to reduce the "random" part of the error.

## Run the demo locally

```
pip install -r requirements.txt
streamlit run app/app.py
```

The demo's product list and delivery cities are a small sample, and shipping fees are fixed per product.

## Run the notebook

1. Download the dataset into `data/` (see `data/README.md`).
2. Open the notebook in `notebook/` and run all cells.

## Notes

- Dataset: [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle).
