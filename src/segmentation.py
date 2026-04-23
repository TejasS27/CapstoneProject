"""
clustering.py — Instacart Customer Segmentation via K-Means
============================================================
Builds customer-level behavioural features, selects the optimal
number of clusters automatically (elbow method + silhouette
cross-check), fits a final K-Means model, maps clusters to named
business segments, and serialises every artefact needed for
deployment.

Saved outputs  (outputs/models/):
    kmeans_model.pkl        – fitted KMeans estimator
    cluster_scaler.pkl      – fitted StandardScaler (customer features)
    cluster_features.pkl    – list of feature column names
    segment_map.pkl         – dict {cluster_id → segment_label}
    cluster_summary.pkl     – DataFrame: mean features per segment
    cluster_counts.pkl      – Series: customer count per segment
    kmeans_meta.pkl         – dict: optimal_k, inertia list, sil list,
                                     K_range, pca_explained_variance

Run from anywhere:
    python src/clustering.py
"""

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = BASE_DIR / "data"    / "ml_ready_switching_data.parquet"
MODEL_DIR   = BASE_DIR / "outputs" / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "clustering_figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Business-segment definitions
# Segments are assigned based on variety_score rank of cluster centroids.
# Change labels here without touching any other logic.
# ---------------------------------------------------------------------------

SEGMENT_LABELS = {
    "low":    "Loyal Customers",      # lowest  variety_score centroid
    "mid":    "Moderate Customers",   # middle  variety_score centroid
    "high":   "Variety Seekers",      # highest variety_score centroid
}

SEGMENT_COLORS = {
    "Loyal Customers":    "#4fc3f7",
    "Moderate Customers": "#ffa726",
    "Variety Seekers":    "#66bb6a",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_pkl(obj, filename: str) -> None:
    path = MODEL_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log.info("Saved  %-40s  (%s)", filename, type(obj).__name__)


def save_fig(fig: plt.Figure, filename: str) -> None:
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info("Figure → %s", path.name)


# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------

log.info("Loading data from %s …", DATA_PATH)
df = pd.read_parquet(DATA_PATH)
log.info("Raw shape: %s", df.shape)

# ---------------------------------------------------------------------------
# 2. Build customer-level feature matrix
# ---------------------------------------------------------------------------

log.info("Building customer feature matrix …")

customer_df = pd.DataFrame({
    "total_purchases": df.groupby("user_id")["product_name"].count(),
    "unique_products":  df.groupby("user_id")["product_name"].nunique(),
    "unique_aisles":    df.groupby("user_id")["aisle"].nunique(),
    "avg_order_gap":    df.groupby("user_id")["order_gap"].mean(),
    "switch_rate":      df.groupby("user_id")["is_switch"].mean(),
})

customer_df["variety_score"] = (
    customer_df["unique_products"] / customer_df["total_purchases"]
)
customer_df = customer_df.fillna(0)

CLUSTER_FEATURES = list(customer_df.columns)   # all six features
log.info("Customer matrix shape: %s | features: %s", customer_df.shape, CLUSTER_FEATURES)

# ---------------------------------------------------------------------------
# 3. Scale
# ---------------------------------------------------------------------------

cluster_scaler = StandardScaler()
X_scaled = cluster_scaler.fit_transform(customer_df[CLUSTER_FEATURES])

# ---------------------------------------------------------------------------
# 4. Determine optimal K  (elbow + silhouette, range 2–9)
# ---------------------------------------------------------------------------

K_RANGE   = range(2, 10)
inertia   = []
sil_scores = []

log.info("Scanning K = %d … %d …", K_RANGE.start, K_RANGE.stop - 1)

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    log.info("  K=%d  inertia=%.1f  silhouette=%.4f", k, km.inertia_, sil_scores[-1])

# Elbow: largest second-derivative of inertia curve
diffs2      = np.diff(np.diff(inertia))
optimal_k   = list(K_RANGE)[int(np.argmax(diffs2)) + 1]

# Silhouette cross-check (logged but not used to override)
optimal_k_sil = list(K_RANGE)[int(np.argmax(sil_scores))]

log.info("Optimal K (elbow)      → %d", optimal_k)
log.info("Optimal K (silhouette) → %d  [cross-check]", optimal_k_sil)

# ---------------------------------------------------------------------------
# 5. Plot elbow & silhouette curves
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(list(K_RANGE), inertia, marker="o", color="#4fc3f7")
axes[0].axvline(optimal_k, color="#ff6b6b", linestyle="--", label=f"Optimal K={optimal_k}")
axes[0].set_title("Elbow Method", fontweight="bold")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].legend()

axes[1].plot(list(K_RANGE), sil_scores, marker="o", color="#66bb6a")
axes[1].axvline(optimal_k_sil, color="#ffa726", linestyle="--",
                label=f"Best sil K={optimal_k_sil}")
axes[1].set_title("Silhouette Scores", fontweight="bold")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].legend()

plt.tight_layout()
save_fig(fig, "elbow_silhouette.png")

# ---------------------------------------------------------------------------
# 6. Fit final model with optimal K
# ---------------------------------------------------------------------------

log.info("Fitting final KMeans with K=%d …", optimal_k)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_df["cluster"] = kmeans_final.fit_predict(X_scaled)

cluster_summary_raw = customer_df.groupby("cluster")[CLUSTER_FEATURES].mean()
log.info("Cluster summary:\n%s", cluster_summary_raw.to_string())

# ---------------------------------------------------------------------------
# 7. Map clusters → business segments
#    Works for any K; extra clusters beyond 3 are labelled "Segment N"
# ---------------------------------------------------------------------------

variety_rank = cluster_summary_raw["variety_score"].rank().astype(int)
n_clusters   = optimal_k

# Assign low / mid / high based on rank position
def rank_to_label(rank: int, n: int) -> str:
    if rank == 1:
        return SEGMENT_LABELS["low"]
    if rank == n:
        return SEGMENT_LABELS["high"]
    return SEGMENT_LABELS["mid"]

segment_map: dict[int, str] = {
    cluster_id: rank_to_label(rank, n_clusters)
    for cluster_id, rank in variety_rank.items()
}

customer_df["segment"] = customer_df["cluster"].map(segment_map)

cluster_summary = customer_df.groupby("segment")[CLUSTER_FEATURES].mean()
cluster_counts  = customer_df["segment"].value_counts()

log.info("Segment distribution:\n%s", cluster_counts.to_string())

# ---------------------------------------------------------------------------
# 8. PCA visualisation (2-D)
# ---------------------------------------------------------------------------

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
customer_df["pca1"] = X_pca[:, 0]
customer_df["pca2"] = X_pca[:, 1]

explained = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(8, 6))
for seg, grp in customer_df.groupby("segment"):
    color = SEGMENT_COLORS.get(seg, "#aaaaaa")
    ax.scatter(grp["pca1"], grp["pca2"], s=10, alpha=0.4, color=color, label=seg)

ax.set_title("Customer Segments — PCA Projection", fontsize=13, fontweight="bold")
ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
ax.legend(markerscale=2)
ax.spines[["top","right"]].set_visible(False)
save_fig(fig, "pca_segments.png")

# ---------------------------------------------------------------------------
# 9. Segment radar / bar profile chart
# ---------------------------------------------------------------------------

profile_cols = ["variety_score", "switch_rate", "avg_order_gap",
                "unique_aisles", "total_purchases"]
profile = customer_df.groupby("segment")[profile_cols].mean()
# normalise 0-1 for radar readability
profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(profile_cols))
w = 0.25
for i, (seg, row) in enumerate(profile_norm.iterrows()):
    color = SEGMENT_COLORS.get(seg, "#aaaaaa")
    ax.bar(x + i * w, row.values, w, label=seg, color=color, alpha=0.85)

ax.set_xticks(x + w)
ax.set_xticklabels(profile_cols, fontsize=9)
ax.set_ylabel("Normalised Score")
ax.set_title("Segment Behaviour Profile", fontsize=13, fontweight="bold")
ax.legend(frameon=False)
ax.spines[["top","right"]].set_visible(False)
save_fig(fig, "segment_profile.png")

# ---------------------------------------------------------------------------
# 10. Serialise all artefacts
# ---------------------------------------------------------------------------

log.info("Saving artefacts …")

save_pkl(kmeans_final,    "kmeans_model.pkl")
save_pkl(cluster_scaler,  "cluster_scaler.pkl")
save_pkl(CLUSTER_FEATURES,"cluster_features.pkl")
save_pkl(segment_map,     "segment_map.pkl")
save_pkl(cluster_summary, "cluster_summary.pkl")
save_pkl(cluster_counts,  "cluster_counts.pkl")
save_pkl(
    {
        "optimal_k":           optimal_k,
        "optimal_k_sil":       optimal_k_sil,
        "K_range":             list(K_RANGE),
        "inertia":             inertia,
        "sil_scores":          sil_scores,
        "pca_explained_variance": explained.tolist(),
    },
    "kmeans_meta.pkl",
)

log.info("✓ Done. All artefacts saved to: %s", MODEL_DIR)

# ---------------------------------------------------------------------------
# 11. Console summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("CLUSTERING SUMMARY")
print("=" * 55)
print(f"  Customers   : {len(customer_df):,}")
print(f"  Optimal K   : {optimal_k}  (elbow)")
print(f"  Sil-check K : {optimal_k_sil}")
print("\nSegment Counts:")
print(cluster_counts.to_string())
print("\nCluster Summary (mean features):")
print(cluster_summary.to_string())
print("=" * 55)