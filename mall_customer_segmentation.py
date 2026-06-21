"""
Retail Customer Segmentation using K-Means Clustering
========================================================
Groups mall/retail customers into segments based on their purchase
behavior using Annual Income and Spending Score — the two features
available in the classic "Mall_Customers" dataset.

REQUIRED COLUMNS (as found in Mall_Customers.csv):
  - CustomerID
  - Genre                    (Male/Female)
  - Age
  - Annual Income (k$)
  - Spending Score (1-100)   (store-assigned score based on purchase behavior)

Usage:
  python mall_customer_segmentation.py Mall_Customers.csv
  python mall_customer_segmentation.py                     # looks for Mall_Customers.csv in cwd
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ----------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    required = {"CustomerID", "Annual Income (k$)", "Spending Score (1-100)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    return df


# ----------------------------------------------------------------------
# 2. CHOOSE K: elbow method + silhouette score
# ----------------------------------------------------------------------
def evaluate_k_range(X_scaled, k_range=range(2, 9), random_state=42):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return list(k_range), inertias, silhouettes


def plot_k_selection(k_values, inertias, silhouettes, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(k_values, inertias, marker="o", color="#4C72B0")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia (within-cluster SSE)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(k_values, silhouettes, marker="o", color="#55A868")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Average silhouette score")
    axes[1].grid(alpha=0.3)

    best_k = k_values[int(np.argmax(silhouettes))]
    axes[1].axvline(best_k, color="red", linestyle="--", alpha=0.6,
                     label=f"best k = {best_k}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return best_k


# ----------------------------------------------------------------------
# 3. FIT FINAL MODEL + LABEL SEGMENTS
# ----------------------------------------------------------------------
def label_segment(row, df):
    """Heuristic, human-readable label based on how a cluster's average
    income/spending compares to the overall customer base."""
    income_med = df["Annual Income (k$)"].median()
    spend_med = df["Spending Score (1-100)"].median()

    high_income = row["Annual Income (k$)"] >= income_med
    high_spend = row["Spending Score (1-100)"] >= spend_med

    if high_income and high_spend:
        return "High Income, High Spend — Target customers"
    if high_income and not high_spend:
        return "High Income, Low Spend — Careful spenders"
    if not high_income and high_spend:
        return "Low Income, High Spend — Impulsive buyers"
    return "Low Income, Low Spend — Price-sensitive"


def fit_kmeans(df, k, feature_cols):
    features = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    df["cluster"] = model.fit_predict(X_scaled)
    return df, model, scaler, X_scaled


def plot_clusters(df, out_path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, df["cluster"].nunique()))
    for cluster_id, color in zip(sorted(df["cluster"].unique()), colors):
        subset = df[df["cluster"] == cluster_id]
        ax.scatter(subset["Annual Income (k$)"], subset["Spending Score (1-100)"],
                   s=60, alpha=0.7, color=color, label=f"Cluster {cluster_id}")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Customer Segments")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_segment_profiles(profile, out_path):
    metrics = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric in zip(axes, metrics):
        ax.bar(profile.index.astype(str), profile[metric], color="#4C72B0")
        ax.set_title(metric)
        ax.set_xlabel("Cluster")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Average Profile per Segment")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------
def main(input_path="Mall_Customers.csv", output_dir="."):
    df = load_data(input_path)
    print(f"Loaded {len(df)} customers.\n")

    feature_cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    X_scaled = StandardScaler().fit_transform(df[feature_cols])

    k_values, inertias, silhouettes = evaluate_k_range(X_scaled)
    best_k = plot_k_selection(k_values, inertias, silhouettes,
                               f"{output_dir}/k_selection.png")
    print(f"Suggested optimal number of clusters (by silhouette score): k = {best_k}\n")

    df, model, scaler, X_scaled = fit_kmeans(df, best_k, feature_cols)

    profile = df.groupby("cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1)
    profile["customer_count"] = df.groupby("cluster").size()
    profile["segment_label"] = profile.apply(lambda row: label_segment(row, df), axis=1)

    print("=== Segment Profiles ===")
    print(profile.to_string())
    print()

    plot_clusters(df, f"{output_dir}/cluster_scatter.png")
    plot_segment_profiles(profile, f"{output_dir}/segment_profiles.png")

    df_out = df.merge(profile["segment_label"], left_on="cluster", right_index=True)
    df_out.to_csv(f"{output_dir}/customer_segments.csv", index=False)
    profile.to_csv(f"{output_dir}/segment_profile_summary.csv")

    print("Saved: customer_segments.csv, segment_profile_summary.csv")
    print("Saved plots: k_selection.png, cluster_scatter.png, segment_profiles.png")
    return df_out, profile


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "Mall_Customers.csv"
    main(input_path, output_dir=".")
