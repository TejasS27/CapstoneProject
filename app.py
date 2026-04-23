"""
app.py — Instacart Analytics Suite  (Streamlit)
================================================
Three modules in one app:

  SWITCHING PREDICTOR
    🔍 Single Prediction   – form-based inference
    📂 Batch Prediction    – CSV upload → scored CSV download
    📊 Model Performance   – accuracy, confusion matrix, per-class metrics

  CUSTOMER SEGMENTATION
    👥 Segment Explorer    – cluster summary, radar profiles, PCA scatter
    🔮 Assign Segment      – enter a customer's stats → get segment + traits

Run:
    streamlit run app.py

Requires (run once first):
    python src/prediction.py
    python src/clustering.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import ConfusionMatrixDisplay

# ---------------------------------------------------------------------------
# Config & paths
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "outputs" / "models"

st.set_page_config(
    page_title="Instacart Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * { color: #c9d1e0 !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.88rem; }
section[data-testid="stSidebar"] h3 { color: #7b8ba8 !important; font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: .1em; }

.header-strip {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%);
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.4rem;
}
.header-strip h1 { margin: 0; font-size: 1.6rem; font-weight: 700; color: #f0f4ff; }
.header-strip p  { margin: .25rem 0 0; color: #7b8ba8; font-size: 0.88rem; }

.metric-row { display: flex; gap: .9rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px;
    background: #0f1117;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
.metric-card .label { font-size: 0.72rem; color: #7b8ba8;
    text-transform: uppercase; letter-spacing: .08em; }
.metric-card .value { font-size: 1.75rem; font-weight: 700; color: #4fc3f7;
    font-family: 'DM Mono', monospace; }

.pred-badge {
    display: inline-block; padding: .3rem .9rem;
    border-radius: 999px; font-size: 0.83rem; font-weight: 600; letter-spacing: .04em;
}
.pred-switch  { background:#ff6b6b22; color:#ff6b6b; border:1px solid #ff6b6b55; }
.pred-repeat  { background:#4caf7d22; color:#4caf7d; border:1px solid #4caf7d55; }

.seg-badge {
    display: inline-block; padding: .3rem .9rem;
    border-radius: 999px; font-size: 0.83rem; font-weight: 600; letter-spacing: .04em;
}
.seg-loyal    { background:#4fc3f722; color:#4fc3f7; border:1px solid #4fc3f755; }
.seg-moderate { background:#ffa72622; color:#ffa726; border:1px solid #ffa72655; }
.seg-variety  { background:#66bb6a22; color:#66bb6a; border:1px solid #66bb6a55; }

.section-title {
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: .12em; color: #7b8ba8; margin-bottom: .5rem;
}
.divider { border-top: 1px solid #1e2130; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading artefacts …")
def load_artefacts():
    def load(name):
        with open(MODEL_DIR / name, "rb") as f:
            return pickle.load(f)
    return {
        # switching
        "models":           load("models.pkl"),
        "label_encoders":   load("label_encoders.pkl"),
        "scaler":           load("scaler.pkl"),
        "feature_names":    load("feature_names.pkl"),
        "rf_feature_names": load("rf_feature_names.pkl"),
        "metrics":          load("model_metrics.pkl"),
        # clustering
        "kmeans":           load("kmeans_model.pkl"),
        "cluster_scaler":   load("cluster_scaler.pkl"),
        "cluster_features": load("cluster_features.pkl"),
        "segment_map":      load("segment_map.pkl"),
        "cluster_summary":  load("cluster_summary.pkl"),
        "cluster_counts":   load("cluster_counts.pkl"),
        "kmeans_meta":      load("kmeans_meta.pkl"),
    }

try:
    art = load_artefacts()
except FileNotFoundError as exc:
    st.error(
        f"⚠️  Missing artefact: {exc}\n\n"
        "Run `python src/prediction.py` **and** `python src/clustering.py` first."
    )
    st.stop()

# unpack switching
models      = art["models"]
le          = art["label_encoders"]
pred_scaler = art["scaler"]
lr_feats    = art["feature_names"]
rf_feats    = art["rf_feature_names"]
metrics     = art["metrics"]

# unpack clustering
kmeans           = art["kmeans"]
cluster_scaler   = art["cluster_scaler"]
cluster_features = art["cluster_features"]
segment_map      = art["segment_map"]
cluster_summary  = art["cluster_summary"]
cluster_counts   = art["cluster_counts"]
kmeans_meta      = art["kmeans_meta"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

SEG_COLORS = {
    "Loyal Customers":    "#4fc3f7",
    "Moderate Customers": "#ffa726",
    "Variety Seekers":    "#66bb6a",
}

with st.sidebar:
    st.markdown("## 🛒 Instacart Analytics")
    st.markdown("---")
    st.markdown("### Switching Predictor")
    pred_pages = ["🔍 Single Prediction", "📂 Batch Prediction", "📊 Model Performance"]
    st.markdown("### Customer Segmentation")
    seg_pages  = ["👥 Segment Explorer", "🔮 Assign Segment"]

    page = st.radio("Navigate", pred_pages + seg_pages, label_visibility="collapsed")
    st.markdown("---")

    if page in pred_pages:
        model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"])

    st.markdown(
        "<p style='font-size:.75rem;color:#3a4a5e;margin-top:1.5rem;'>"
        "Artefacts: <code>outputs/models/</code></p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Switching helpers
# ---------------------------------------------------------------------------

def safe_encode(encoder, value, fallback=0):
    try:
        return int(encoder.transform([value])[0])
    except ValueError:
        return fallback


def build_lr_row(inp):
    row = {
        "aisle":             safe_encode(le["aisle"],            inp["aisle"]),
        "product_name":      safe_encode(le["product_name"],     inp["product_name"]),
        "previous_product":  safe_encode(le["previous_product"], inp["previous_product"]),
        "order_gap":         inp["order_gap"],
        "user_total_orders": inp["user_total_orders"],
        "user_aisle_count":  inp["user_aisle_count"],
        "product_popularity":   inp["product_popularity"],
        "product_repeat_rate":  inp["product_repeat_rate"],
        "user_switch_rate":     inp["user_switch_rate"],
    }
    return pred_scaler.transform(pd.DataFrame([row])[lr_feats])


def build_rf_row(inp):
    row = {
        "aisle":             safe_encode(le["aisle"],            inp["aisle"]),
        "product_name":      safe_encode(le["product_name"],     inp["product_name"]),
        "previous_product":  safe_encode(le["previous_product"], inp["previous_product"]),
        "order_gap":         inp["order_gap"],
        "user_total_orders": inp["user_total_orders"],
        "user_aisle_count":  inp["user_aisle_count"],
        "product_popularity":   inp["product_popularity"],
    }
    return pd.DataFrame([row])[rf_feats]


def predict(inp, model_name):
    if model_name == "Logistic Regression":
        X, model = build_lr_row(inp), models["logistic_regression"]
    else:
        X, model = build_rf_row(inp), models["random_forest"]
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return int(pred), proba


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

SEG_TRAITS = {
    "Loyal Customers": {
        "icon": "🏅",
        "description": "Buys the same products repeatedly. Low variety score and switch rate.",
        "strategy": "Reward loyalty with subscription discounts and first-look offers on restocks.",
    },
    "Moderate Customers": {
        "icon": "⚖️",
        "description": "Mix of repeat purchases and occasional exploration.",
        "strategy": "Introduce complementary products via cross-sell; curated 'you might also like' nudges.",
    },
    "Variety Seekers": {
        "icon": "🌟",
        "description": "High variety score — regularly explores new products and aisles.",
        "strategy": "Drive discovery with new-arrival highlights, bundles, and seasonal specials.",
    },
}

SEG_CSS = {
    "Loyal Customers":    "seg-loyal",
    "Moderate Customers": "seg-moderate",
    "Variety Seekers":    "seg-variety",
}


def assign_segment(customer_inputs: dict) -> str:
    row        = pd.DataFrame([customer_inputs])[cluster_features]
    X          = cluster_scaler.transform(row)
    cluster_id = int(kmeans.predict(X)[0])
    return segment_map.get(cluster_id, f"Cluster {cluster_id}")


# ===========================================================================
# HEADER
# ===========================================================================

PAGE_META = {
    "🔍 Single Prediction":  ("🛒 Product Switching Predictor",  "Will this customer switch products or repeat their last purchase?"),
    "📂 Batch Prediction":   ("🛒 Batch Scoring",                 "Score an entire CSV of purchase records at once."),
    "📊 Model Performance":  ("🛒 Model Performance Dashboard",   "Held-out test-set metrics for both classifiers."),
    "👥 Segment Explorer":   ("👥 Customer Segment Explorer",     "Understand each segment's behaviour profile and size."),
    "🔮 Assign Segment":     ("🔮 Segment Assignment",            "Enter a customer's stats to instantly assign them to a segment."),
}

title, subtitle = PAGE_META.get(page, ("Instacart Analytics", ""))
st.markdown(f"""
<div class="header-strip">
  <div><h1>{title}</h1><p>{subtitle}</p></div>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# PAGE: Single Prediction
# ===========================================================================

if page == "🔍 Single Prediction":

    st.markdown('<div class="section-title">Enter purchase details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        aisle            = st.text_input("Aisle",            value="fresh vegetables")
        product_name     = st.text_input("Product Name",     value="Organic Baby Spinach")
        previous_product = st.text_input("Previous Product", value="Organic Baby Spinach")

    with c2:
        order_gap         = st.number_input("Order Gap (days)",    min_value=0,   max_value=365, value=7)
        user_total_orders = st.number_input("User Total Orders",   min_value=1,   max_value=500, value=20)
        user_aisle_count  = st.number_input("User Aisle Count",    min_value=1,   max_value=100, value=8)

    with c3:
        product_popularity  = st.number_input("Product Popularity",  min_value=1, max_value=100_000, value=500)
        product_repeat_rate = st.slider("Product Repeat Rate", 0.0, 1.0, 0.6, 0.01)
        user_switch_rate    = st.slider("User Switch Rate",    0.0, 1.0, 0.4, 0.01)

    if st.button("Predict", type="primary", use_container_width=True):
        inp = dict(
            aisle=aisle, product_name=product_name, previous_product=previous_product,
            order_gap=order_gap, user_total_orders=user_total_orders,
            user_aisle_count=user_aisle_count, product_popularity=product_popularity,
            product_repeat_rate=product_repeat_rate, user_switch_rate=user_switch_rate,
        )
        pred, proba = predict(inp, model_choice)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)

        with r1:
            label = "🔄 Will Switch" if pred == 1 else "✅ Will Repeat"
            css   = "pred-switch"   if pred == 1 else "pred-repeat"
            st.markdown("**Prediction**")
            st.markdown(f'<span class="pred-badge {css}">{label}</span>', unsafe_allow_html=True)

        with r2:
            st.metric("Switch Probability", f"{proba[1]:.1%}")

        with r3:
            st.metric("Repeat Probability", f"{proba[0]:.1%}")

        fig, ax = plt.subplots(figsize=(6, 1.1))
        fig.patch.set_alpha(0)
        ax.barh([""], [proba[0]], color="#4caf7d", height=.5, label="Repeat")
        ax.barh([""], [proba[1]], left=[proba[0]], color="#ff6b6b", height=.5, label="Switch")
        ax.set_xlim(0, 1); ax.axis("off")
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ===========================================================================
# PAGE: Batch Prediction
# ===========================================================================

elif page == "📂 Batch Prediction":

    st.info(
        "CSV must contain: `aisle`, `product_name`, `previous_product`, "
        "`order_gap`, `user_total_orders`, `user_aisle_count`, `product_popularity`, "
        "`product_repeat_rate`, `user_switch_rate`"
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        raw = pd.read_csv(uploaded)
        st.markdown(f"**{len(raw):,} rows loaded**")
        st.dataframe(raw.head(5), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            results = []
            for _, row in raw.iterrows():
                inp = row.to_dict()
                try:
                    pred, proba = predict(inp, model_choice)
                    results.append({"prediction": pred,
                                    "switch_prob": round(proba[1], 4),
                                    "repeat_prob": round(proba[0], 4)})
                except Exception:
                    results.append({"prediction": -1, "switch_prob": None, "repeat_prob": None})

            out = raw.copy()
            out[["prediction", "switch_prob", "repeat_prob"]] = pd.DataFrame(results)
            out["prediction_label"] = out["prediction"].map({1: "Switch", 0: "Repeat", -1: "Error"})

            st.success(f"✅ Scored {len(out):,} rows.")
            st.dataframe(out, use_container_width=True)
            st.download_button("⬇ Download Results CSV",
                               out.to_csv(index=False).encode(),
                               "switching_predictions.csv", "text/csv")

            counts = out["prediction_label"].value_counts()
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_alpha(0)
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
                   colors=["#4caf7d", "#ff6b6b"][:len(counts)],
                   startangle=90, wedgeprops=dict(width=0.55))
            ax.set_title("Prediction Split", fontsize=12, fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)


# ===========================================================================
# PAGE: Model Performance
# ===========================================================================

elif page == "📊 Model Performance":

    mk  = "lr" if model_choice == "Logistic Regression" else "rf"
    m   = metrics[mk]
    rep = m["report"]
    acc    = m["accuracy"]
    f1     = rep.get("weighted avg", {}).get("f1-score", 0)
    prec   = rep.get("weighted avg", {}).get("precision", 0)
    recall = rep.get("weighted avg", {}).get("recall", 0)

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="label">Accuracy</div>
        <div class="value">{acc:.2%}</div></div>
      <div class="metric-card"><div class="label">F1 (weighted)</div>
        <div class="value">{f1:.3f}</div></div>
      <div class="metric-card"><div class="label">Precision</div>
        <div class="value">{prec:.3f}</div></div>
      <div class="metric-card"><div class="label">Recall</div>
        <div class="value">{recall:.3f}</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        cm  = np.array(m["confusion"])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        fig.patch.set_alpha(0)
        ConfusionMatrixDisplay(cm, display_labels=["Repeat","Switch"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown("**Per-class Metrics**")
        classes = [k for k in rep if k not in ("accuracy","macro avg","weighted avg")]
        x, w    = np.arange(len(classes)), 0.25
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_alpha(0)
        ax.bar(x-w, [rep[c]["f1-score"]  for c in classes], w, label="F1",       color="#4fc3f7")
        ax.bar(x,   [rep[c]["precision"] for c in classes], w, label="Precision", color="#4caf7d")
        ax.bar(x+w, [rep[c]["recall"]    for c in classes], w, label="Recall",    color="#ff6b6b")
        ax.set_xticks(x); ax.set_xticklabels(classes)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("**Full Classification Report**")
    report_rows = {k: v for k, v in rep.items() if isinstance(v, dict)}
    st.dataframe(pd.DataFrame(report_rows).T.style.format("{:.3f}"),
                 use_container_width=True)


# ===========================================================================
# PAGE: Segment Explorer
# ===========================================================================

elif page == "👥 Segment Explorer":

    meta            = kmeans_meta
    total_customers = int(cluster_counts.sum())
    optimal_k       = meta["optimal_k"]
    best_sil        = round(max(meta["sil_scores"]), 4)
    pca_var         = sum(meta["pca_explained_variance"][:2])

    # ── KPI strip ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="label">Total Customers</div>
        <div class="value">{total_customers:,}</div></div>
      <div class="metric-card"><div class="label">Optimal K (elbow)</div>
        <div class="value">{optimal_k}</div></div>
      <div class="metric-card"><div class="label">Best Silhouette</div>
        <div class="value">{best_sil:.3f}</div></div>
      <div class="metric-card"><div class="label">PCA Variance (2 PC)</div>
        <div class="value">{pca_var:.1%}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── segment count cards ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">Segment Breakdown</div>', unsafe_allow_html=True)
    cols = st.columns(len(cluster_counts))
    for col, (seg, cnt) in zip(cols, cluster_counts.items()):
        traits = SEG_TRAITS.get(seg, {})
        color  = SEG_COLORS.get(seg, "#888")
        pct    = cnt / total_customers
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}44;">
              <div class="label">{traits.get('icon','')}&nbsp;{seg}</div>
              <div class="value" style="color:{color};">{cnt:,}</div>
              <div style="font-size:.78rem;color:#7b8ba8;">{pct:.1%} of customers</div>
              <div style="font-size:.76rem;color:#5a6a82;margin-top:.4rem;">
                {traits.get('description','')}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── elbow + silhouette curves ────────────────────────────────────────────
    st.markdown('<div class="section-title">K Selection Curves</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.patch.set_alpha(0)
    K = meta["K_range"]

    axes[0].plot(K, meta["inertia"],   marker="o", color="#4fc3f7", lw=2)
    axes[0].axvline(optimal_k, color="#ff6b6b", ls="--", lw=1.5, label=f"K={optimal_k}")
    axes[0].set_title("Elbow — Inertia", fontweight="bold")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
    axes[0].legend(frameon=False); axes[0].spines[["top","right"]].set_visible(False)

    axes[1].plot(K, meta["sil_scores"], marker="o", color="#66bb6a", lw=2)
    axes[1].axvline(meta["optimal_k_sil"], color="#ffa726", ls="--", lw=1.5,
                    label=f"K={meta['optimal_k_sil']}")
    axes[1].set_title("Silhouette Score", fontweight="bold")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Score")
    axes[1].legend(frameon=False); axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── behaviour profile bar chart ──────────────────────────────────────────
    st.markdown('<div class="section-title">Behaviour Profile (mean per segment)</div>',
                unsafe_allow_html=True)

    profile_cols = [c for c in cluster_summary.columns if c in
                    ["variety_score","switch_rate","avg_order_gap","unique_aisles","total_purchases"]]
    profile      = cluster_summary[profile_cols]
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_alpha(0)
    x = np.arange(len(profile_cols))
    w = 0.8 / max(len(profile_norm), 1)

    for i, (seg, row) in enumerate(profile_norm.iterrows()):
        color = SEG_COLORS.get(seg, "#aaa")
        ax.bar(x + i * w - 0.4 + w / 2, row.values, w, label=seg, color=color, alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(profile_cols, fontsize=9)
    ax.set_ylabel("Normalised Score (0–1)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── raw summary table ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Mean Feature Values per Segment</div>',
                unsafe_allow_html=True)
    st.dataframe(cluster_summary.style.format("{:.3f}"), use_container_width=True)

    # ── strategy cards ───────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recommended Strategies</div>', unsafe_allow_html=True)
    scols = st.columns(len(SEG_TRAITS))
    for col, (seg, info) in zip(scols, SEG_TRAITS.items()):
        color = SEG_COLORS.get(seg, "#888")
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}44;">
              <div class="label">{info['icon']}&nbsp;{seg}</div>
              <div style="font-size:.82rem;color:#c9d1e0;margin-top:.5rem;">{info['strategy']}</div>
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# PAGE: Assign Segment
# ===========================================================================

elif page == "🔮 Assign Segment":

    st.markdown('<div class="section-title">Enter customer behavioural stats</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        total_purchases = st.number_input("Total Purchases",     min_value=1,   max_value=5000, value=50)
        unique_products = st.number_input("Unique Products",     min_value=1,   max_value=2000, value=20)

    with c2:
        unique_aisles   = st.number_input("Unique Aisles",       min_value=1,   max_value=100,  value=8)
        avg_order_gap   = st.number_input("Avg Order Gap (days)",min_value=0.0, max_value=365.0,value=10.0, step=0.5)

    with c3:
        switch_rate   = st.slider("Switch Rate",   0.0, 1.0, 0.35, 0.01)
        variety_score = st.slider(
            "Variety Score", 0.0, 1.0,
            float(round(min(unique_products / max(total_purchases, 1), 1.0), 2)),
            0.01,
            help="unique_products / total_purchases — auto-filled but editable",
        )

    if st.button("Assign Segment", type="primary", use_container_width=True):
        customer_inp = {
            "total_purchases": total_purchases,
            "unique_products":  unique_products,
            "unique_aisles":    unique_aisles,
            "avg_order_gap":    avg_order_gap,
            "switch_rate":      switch_rate,
            "variety_score":    variety_score,
        }
        segment = assign_segment(customer_inp)
        traits  = SEG_TRAITS.get(segment, {"icon": "❓", "description": "", "strategy": ""})
        css     = SEG_CSS.get(segment, "seg-loyal")
        color   = SEG_COLORS.get(segment, "#888")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        r1, r2 = st.columns([1, 2])

        with r1:
            st.markdown("**Assigned Segment**")
            st.markdown(
                f'<span class="seg-badge {css}">{traits["icon"]} {segment}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='color:#7b8ba8;font-size:.85rem;margin-top:.6rem;'>"
                f"{traits['description']}</p>",
                unsafe_allow_html=True,
            )

        with r2:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}44;">
              <div class="label">📣 Recommended Strategy</div>
              <div style="font-size:.88rem;color:#c9d1e0;margin-top:.5rem;">{traits['strategy']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Customer vs segment average bar chart
        if segment in cluster_summary.index:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("**Your customer vs segment average**")
            seg_avg      = cluster_summary.loc[segment]
            compare_cols = [c for c in cluster_features if c in seg_avg.index]
            compare_df   = pd.DataFrame({
                "Your Customer": [customer_inp.get(c, 0) for c in compare_cols],
                "Segment Avg":   [seg_avg[c]             for c in compare_cols],
            }, index=compare_cols)

            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_alpha(0)
            x, w = np.arange(len(compare_cols)), 0.35
            ax.bar(x - w/2, compare_df["Your Customer"], w, color=color,     label="Your Customer", alpha=0.85)
            ax.bar(x + w/2, compare_df["Segment Avg"],   w, color="#7b8ba8", label="Segment Avg",   alpha=0.65)
            ax.set_xticks(x); ax.set_xticklabels(compare_cols, fontsize=9)
            ax.legend(frameon=False, fontsize=9)
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)