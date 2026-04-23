"""
prediction.py — Instacart Product Switching Prediction
=======================================================
Trains Logistic Regression and Random Forest classifiers on the
ML-ready switching dataset, evaluates them with SHAP, and serialises
all artefacts needed for deployment.

Saved outputs  (under outputs/models/):
    models.pkl           – dict with both fitted estimators
    label_encoders.pkl   – dict of LabelEncoders  (aisle / product / prev)
    scaler.pkl           – fitted StandardScaler
    feature_names.pkl    – list: LR feature set
    rf_feature_names.pkl – list: RF feature set
    model_metrics.pkl    – dict with accuracy + classification reports
"""

import pickle
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "ml_ready_switching_data.parquet"
MODEL_DIR  = BASE_DIR / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
# Helper
# ---------------------------------------------------------------------------

def save_pkl(obj, filename: str) -> None:
    path = MODEL_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log.info("Saved %-40s (%s)", filename, type(obj).__name__)

# ---------------------------------------------------------------------------
# 1. Load & feature-engineer
# ---------------------------------------------------------------------------

log.info("Loading data …")
df = pd.read_parquet(DATA_PATH)
log.info("Initial shape: %s", df.shape)

df["product_popularity"]  = df["product_name"].map(df["product_name"].value_counts())
df["product_repeat_rate"] = df.groupby("product_name")["is_switch"].transform("mean")
df["user_switch_rate"]    = df.groupby("user_id")["is_switch"].transform("mean")

# ---------------------------------------------------------------------------
# 2. Define feature sets
# ---------------------------------------------------------------------------

LR_FEATURES = [
    "aisle", "product_name", "previous_product",
    "order_gap", "user_total_orders", "user_aisle_count",
    "product_popularity", "product_repeat_rate", "user_switch_rate",
]

RF_FEATURES = [
    "aisle", "product_name", "previous_product",
    "order_gap", "user_total_orders", "user_aisle_count",
    "product_popularity",
]

TARGET = "is_switch"

# ---------------------------------------------------------------------------
# 3. Encode categoricals
# ---------------------------------------------------------------------------

X_full = df[LR_FEATURES].copy()
y      = df[TARGET]

le_aisle   = LabelEncoder()
le_product = LabelEncoder()
le_prev    = LabelEncoder()

X_full["aisle"]            = le_aisle.fit_transform(X_full["aisle"])
X_full["product_name"]     = le_product.fit_transform(X_full["product_name"])
X_full["previous_product"] = le_prev.fit_transform(X_full["previous_product"])

label_encoders = {
    "aisle":            le_aisle,
    "product_name":     le_product,
    "previous_product": le_prev,
}

# ---------------------------------------------------------------------------
# 4. Train / test split
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 5. Scale
# ---------------------------------------------------------------------------

scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)
X_test_scaled   = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 6. SMOTE
# ---------------------------------------------------------------------------

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

# ---------------------------------------------------------------------------
# 7. Logistic Regression  (GridSearchCV)
# ---------------------------------------------------------------------------

log.info("Training Logistic Regression …")

grid_lr = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid={"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"], "solver": ["liblinear"]},
    cv=3, scoring="f1", n_jobs=-1,
)
grid_lr.fit(X_train_sm, y_train_sm)
best_lr   = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test_scaled)

print("\n========== LOGISTIC REGRESSION ==========")
print("Best params :", grid_lr.best_params_)
print("Accuracy    :", accuracy_score(y_test, y_pred_lr))
print("\nReport:\n",    classification_report(y_test, y_pred_lr))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred_lr))

lr_report = classification_report(y_test, y_pred_lr, output_dict=True)

# ---------------------------------------------------------------------------
# 8. SHAP — Logistic Regression
# ---------------------------------------------------------------------------

log.info("Computing SHAP values for LR …")
explainer_lr      = shap.LinearExplainer(best_lr, X_train_scaled)
shap_values_lr    = explainer_lr.shap_values(X_test_scaled)
shap.summary_plot(shap_values_lr, X_test, feature_names=X_train.columns)
shap.summary_plot(shap_values_lr, X_test, feature_names=X_train.columns, plot_type="bar")

# ---------------------------------------------------------------------------
# 9. Random Forest
# ---------------------------------------------------------------------------

log.info("Training Random Forest …")

X_train_rf = X_train[RF_FEATURES]
X_test_rf  = X_test[RF_FEATURES]

rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=10,
    random_state=42, n_jobs=-1,
)
rf.fit(X_train_rf, y_train)
y_pred_rf = rf.predict(X_test_rf)

print("\n========== RANDOM FOREST ==========")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("\nReport:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred_rf))

rf_report = classification_report(y_test, y_pred_rf, output_dict=True)

# ---------------------------------------------------------------------------
# 10. SHAP — Random Forest
# ---------------------------------------------------------------------------

# log.info("Computing SHAP values for RF …")
# explainer_rf   = shap.TreeExplainer(rf)
# shap_values_rf = explainer_rf.shap_values(X_test)

# ---------------------------------------------------------------------------
# 11. Persist all artefacts
# ---------------------------------------------------------------------------

log.info("Saving model artefacts to %s …", MODEL_DIR)

save_pkl(
    {"logistic_regression": best_lr, "random_forest": rf},
    "models.pkl",
)
save_pkl(label_encoders,  "label_encoders.pkl")
save_pkl(scaler,          "scaler.pkl")
save_pkl(LR_FEATURES,     "feature_names.pkl")
save_pkl(RF_FEATURES,     "rf_feature_names.pkl")
save_pkl(
    {
        "lr": {
            "accuracy":  accuracy_score(y_test, y_pred_lr),
            "report":    lr_report,
            "confusion": confusion_matrix(y_test, y_pred_lr).tolist(),
        },
        "rf": {
            "accuracy":  accuracy_score(y_test, y_pred_rf),
            "report":    rf_report,
            "confusion": confusion_matrix(y_test, y_pred_rf).tolist(),
        },
    },
    "model_metrics.pkl",
)

log.info("✓ All artefacts saved.")