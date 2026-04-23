"""
eda.py — Instacart EDA + Results Serialization
================================================
Runs exploratory analysis on the ML-ready switching dataset and saves
all results to .pkl files under outputs/eda/ for use in deployment.

Run from anywhere:
    python src/eda.py
"""

import pickle
import logging
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # No pop-up windows — saves figures to disk
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths  (all resolved relative to THIS file, not the terminal location)
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = BASE_DIR / "data"    / "ml_ready_switching_data.parquet"
OUTPUT_DIR  = BASE_DIR / "outputs" / "eda"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
# Helpers
# ---------------------------------------------------------------------------

def save_pkl(obj, filename: str) -> None:
    """Save any object to outputs/eda/<filename>.pkl"""
    path = OUTPUT_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log.info("Saved %-45s  (%s)", filename, type(obj).__name__)


def save_fig(fig: plt.Figure, filename: str) -> None:
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info("Figure  → %s", path.name)


def segment(score: float) -> str:
    if score < 0.3:
        return "Loyal Customers"
    elif score < 0.7:
        return "Moderate Customers"
    return "Variety Seekers"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

log.info("Loading data from %s …", DATA_PATH)
final_data = pd.read_parquet(DATA_PATH)

dataset_info = {
    "n_rows":       len(final_data),
    "n_columns":    final_data.shape[1],
    "n_customers":  final_data["user_id"].nunique(),
    "n_products":   final_data["product_name"].nunique(),
    "n_aisles":     final_data["aisle"].nunique(),
}

log.info(
    "Loaded | rows: %d | customers: %d | products: %d | aisles: %d",
    dataset_info["n_rows"],
    dataset_info["n_customers"],
    dataset_info["n_products"],
    dataset_info["n_aisles"],
)

# ---------------------------------------------------------------------------
# 2. Orders per customer
# ---------------------------------------------------------------------------

orders_per_user = final_data.groupby("user_id")["order_number"].nunique()
orders_per_user_stats = orders_per_user.describe()
log.info("Orders per user:\n%s", orders_per_user_stats.to_string())

# ---------------------------------------------------------------------------
# 3. Repeat / switching rates
# ---------------------------------------------------------------------------

final_data["is_repeat"] = (
    final_data["product_name"].astype(str)
    == final_data["previous_product"].astype(str)
)
repeat_rate    = final_data["is_repeat"].mean()
switching_rate = 1 - repeat_rate

log.info(
    "Repeat rate: %.2f%%  |  Switching rate: %.2f%%",
    repeat_rate * 100, switching_rate * 100,
)

# ---------------------------------------------------------------------------
# 4. Top products
# ---------------------------------------------------------------------------

top_products = final_data["product_name"].value_counts().head(10)

# ---------------------------------------------------------------------------
# 5. Variety per customer
# ---------------------------------------------------------------------------

variety = final_data.groupby("user_id")["product_name"].nunique()
variety_stats = variety.describe()

# ---------------------------------------------------------------------------
# 6. Variety-score segmentation
# ---------------------------------------------------------------------------

total_purchases = final_data.groupby("user_id")["product_name"].count()
unique_products  = final_data.groupby("user_id")["product_name"].nunique()

variety_df = pd.DataFrame({
    "total_purchases": total_purchases,
    "unique_products":  unique_products,
})
variety_df["variety_score"] = variety_df["unique_products"] / variety_df["total_purchases"]
variety_df["segment"]       = variety_df["variety_score"].apply(segment)

segment_counts = variety_df["segment"].value_counts()
log.info("Segment distribution:\n%s", segment_counts.to_string())

# ---------------------------------------------------------------------------
# 7. Visualizations  (saved as PNG, not shown interactively)
# ---------------------------------------------------------------------------

# 7a. Orders per customer histogram
fig, ax = plt.subplots(figsize=(8, 4))
orders_per_user.hist(bins=30, ax=ax, color="#4C72B0", edgecolor="white")
ax.set_title("Orders per Customer", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Orders")
ax.set_ylabel("Number of Customers")
save_fig(fig, "orders_per_customer.png")

# 7b. Repeat vs Switch pie chart
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(
    [repeat_rate, switching_rate],
    labels=["Repeat", "Switch"],
    autopct="%1.1f%%",
    colors=["#4C72B0", "#DD8452"],
    startangle=140,
)
ax.set_title("Customer Behavior — Product Level", fontsize=13, fontweight="bold")
save_fig(fig, "repeat_vs_switch.png")

# 7c. Top 10 product
top_products = final_data["product_name"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 4))
top_products.plot(kind="bar", ax=ax, color="#55A868", edgecolor="white")
ax.set_title("Top 10 Products", fontsize=14, fontweight="bold")
ax.set_ylabel("Purchase Count")
plt.xticks(rotation=45, ha="right")
save_fig(fig, "top_products.png")

# 7c. Top 10 aisles
top_aisles = final_data["aisle"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 4))
top_aisles.plot(kind="bar", ax=ax, color="#55A868", edgecolor="white")
ax.set_title("Top 10 Aisles", fontsize=14, fontweight="bold")
ax.set_ylabel("Purchase Count")
plt.xticks(rotation=45, ha="right")
save_fig(fig, "top_aisles.png")

# 7d. Customer segments bar chart
fig, ax = plt.subplots(figsize=(6, 4))
segment_counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="white")
ax.set_title("Customer Segments", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Customers")
plt.xticks(rotation=20, ha="right")
save_fig(fig, "customer_segments.png")

# ---------------------------------------------------------------------------
# 8. Save all results to .pkl
# ---------------------------------------------------------------------------

log.info("Saving results …")

save_pkl(dataset_info,           "dataset_info.pkl")          # dict  — basic dataset stats
save_pkl(orders_per_user,        "orders_per_user.pkl")        # Series — orders per user_id
save_pkl(orders_per_user_stats,  "orders_per_user_stats.pkl")  # Series — describe() output
save_pkl(repeat_rate,            "repeat_rate.pkl")            # float
save_pkl(switching_rate,         "switching_rate.pkl")         # float
save_pkl(top_products,           "top_products.pkl")           # Series — top 10 products
save_pkl(variety_stats,          "variety_stats.pkl")          # Series — describe() output
save_pkl(variety_df,             "variety_df.pkl")             # DataFrame — score + segment per user
save_pkl(segment_counts,         "segment_counts.pkl")         # Series — count per segment

log.info("✓ Done. All outputs saved to: %s", OUTPUT_DIR)

# ---------------------------------------------------------------------------
# 9. Print summary (mirrors original Colab output)
# ---------------------------------------------------------------------------

print("\n" + "="*55)
print("DATASET SUMMARY")
print("="*55)
print(f"  Customers : {dataset_info['n_customers']:,}")
print(f"  Products  : {dataset_info['n_products']:,}")
print(f"  Aisles    : {dataset_info['n_aisles']:,}")
print(f"\n  Repeat Rate   : {repeat_rate:.2%}")
print(f"  Switching Rate: {switching_rate:.2%}")
print("\nOrders per User:")
print(orders_per_user_stats.to_string())
print("\nTop 10 Products:")
print(top_products.to_string())
print("\nCustomer Segments:")
print(segment_counts.to_string())
print("\nvariety_df sample:")
print(variety_df.head(5).to_string())
print("="*55)