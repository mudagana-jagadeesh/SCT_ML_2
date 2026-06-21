# Retail Customer Segmentation (K-Means)

Groups mall/retail customers into segments based on purchase behavior,
using K-Means clustering on the `Mall_Customers.csv` dataset.

## Files

| File | Description |
|---|---|
| `mall_customer_segmentation.py` | Main script — run this |
| `Mall_Customers.csv` | Input dataset (200 customers) |
| `customer_segments.csv` | Output: every customer with their assigned cluster + label |
| `segment_profile_summary.csv` | Output: average Age/Income/Spending per cluster |
| `k_selection.png` | Elbow + silhouette plots used to pick the number of clusters |
| `cluster_scatter.png` | Visual: Income vs. Spending Score, colored by cluster |
| `segment_profiles.png` | Bar charts comparing Age/Income/Spending across clusters |

## Dataset columns

```
CustomerID, Genre, Age, Annual Income (k$), Spending Score (1-100)
```

`Spending Score` is a store-assigned score (1–100) based on customer
behavior and spending nature — it's not raw transaction data, but it's
the closest proxy for purchase history available in this dataset.

## How it works

1. **Features used for clustering:** `Annual Income (k$)` and
   `Spending Score (1-100)` (scaled with `StandardScaler`).
2. **Choosing k:** the script tests k = 2 through 8, plotting inertia
   (elbow method) and silhouette score, then automatically picks the k
   with the highest silhouette score.
3. **Fitting:** runs `KMeans` with the chosen k and assigns every
   customer to a cluster.
4. **Labeling:** each cluster gets a human-readable label
   (e.g. "High Income, High Spend — Target customers") based on how its
   average income/spending compares to the overall median.
5. **Outputs:** segment CSVs and three plots, saved to the output
   directory.

## Usage

```bash
python mall_customer_segmentation.py Mall_Customers.csv
```

If you omit the argument, it looks for `Mall_Customers.csv` in the
current directory.

## Requirements

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Notes / extending this

- To cluster on **Age** too (3D clustering), add `"Age"` to
  `feature_cols` in `main()`.
- The cluster labeling logic in `label_segment()` is a simple
  income/spending quadrant heuristic — adjust the thresholds or labels
  to match your own business logic.
- This dataset has no real transaction history (dates, order values),
  so true RFM (Recency/Frequency/Monetary) segmentation isn't possible
  here — if you get access to actual transaction-level data, the
  original RFM-based script structure can be swapped back in.
