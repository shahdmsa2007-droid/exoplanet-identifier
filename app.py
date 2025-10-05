# Exoplanet Auto-Classifier (Kepler/K2/TESS)
# -------------------------------------------------------------
# A single-file Streamlit app that:
# 1) Pulls public exoplanet tables from NASA Exoplanet Archive via TAP (CSV)
# 2) Lets you pick features / model / hyperparameters
# 3) Trains a 3-class classifier: CONFIRMED vs CANDIDATE vs FALSE POSITIVE
# 4) Evaluates with train/validation split & cross-val metrics
# 5) Predicts on user-uploaded CSVs and allows result download
# -------------------------------------------------------------
# How to run locally:
#   pip install streamlit pandas numpy scikit-learn shap xgboost matplotlib requests
#   streamlit run app.py
# -------------------------------------------------------------

import io
import json
import textwrap
import zipfile
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier  # optional
    HAS_XGB = True
except Exception:
    HAS_XGB = False

st.set_page_config(page_title="Exoplanet Auto-Classifier", layout="wide")

st.title("🔭 Exoplanet Auto-Classifier — Kepler / K2 / TESS")
st.caption(
    "Train a machine-learning model on NASA Exoplanet Archive tables to classify entries as CONFIRMED, CANDIDATE, or FALSE POSITIVE."
)

# -----------------------------
# Data sources (NASA Exoplanet Archive TAP CSV endpoints)
# -----------------------------
TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Table-specific sensible default columns (mix of transit + stellar params where available)
KOI_COLS = [
    "kepid", "koi_disposition",  # label
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_srad", "koi_steff", "koi_slogg", "koi_smet",
]

K2_COLS = [
    "disposition",  # label per k2pandc docs
    # Transit & planet params (some may be NaN for candidates):
    "pl_orbper", "pl_trandurh", "pl_trandep", "pl_rade",
    # Stellar:
    "st_teff", "st_logg", "st_rad", "st_met",
]

TOI_COLS = [
    "tfopwg_disp",  # label enum: CP/KP/PC/APC/FP/FA
    # Transit/planet
    "pl_orbper", "pl_trandurh", "pl_trandep", "pl_rade",
    # Stellar
    "st_teff", "st_logg", "st_rad", "st_tmag",
]

TABLE_META = {
    "Kepler KOI (cumulative)": {
        "table": "cumulative",
        "label_col": "koi_disposition",
        "feature_cols": KOI_COLS[2:],
        "default_cols": KOI_COLS,
    },
    "K2 Planets & Candidates (k2pandc)": {
        "table": "k2pandc",
        "label_col": "disposition",
        "feature_cols": K2_COLS[1:],
        "default_cols": K2_COLS,
    },
    "TESS TOI (toi)": {
        "table": "toi",
        "label_col": "tfopwg_disp",
        "feature_cols": TOI_COLS[1:],
        "default_cols": TOI_COLS,
    },
}

@st.cache_data(show_spinner=False)
def fetch_table(table: str, cols: List[str]) -> pd.DataFrame:
    # Build TAP SQL: select only the columns we need, wherever possible
    # If '*' fails for restricted tables, columns list will be used.
    cols_q = ",".join(cols)
    sql = f"select {cols_q} from {table}"
    params = {"query": sql, "format": "csv"}
    r = requests.get(TAP_SYNC, params=params, timeout=90)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Dataset & Features")
    dataset_name = st.selectbox(
        "Choose a dataset",
        list(TABLE_META.keys()),
        help="All pulled live from NASA Exoplanet Archive TAP."
    )
    meta = TABLE_META[dataset_name]

    # Allow user to tweak the feature set
    all_feats = meta["default_cols"][1:]  # drop ID if present
    label_col = meta["label_col"]
    # Filter out label_col from features list just in case
    feat_choices = [c for c in all_feats if c != label_col]
    selected_features = st.multiselect(
        "Feature columns",
        feat_choices,
        default=meta["feature_cols"],
    )

    st.divider()
    st.header("Model")
    model_type = st.selectbox("Algorithm", [
        "RandomForest",
        "LogisticRegression",
        "XGBoost" if HAS_XGB else "XGBoost (install xgboost to enable)",
    ])

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random seed", value=42, step=1)
    cv_folds = st.slider("Cross-val folds", 3, 10, 5)

    st.subheader("Hyperparameters")
    if model_type.startswith("RandomForest"):
        n_estimators = st.slider("n_estimators", 50, 600, 300, 50)
        max_depth = st.slider("max_depth (0 = None)", 0, 40, 0)
        rf_params = dict(n_estimators=n_estimators,
                         max_depth=None if max_depth == 0 else max_depth,
                         class_weight="balanced",
                         n_jobs=-1,
                         random_state=random_state)
    elif model_type.startswith("LogisticRegression"):
        C = st.slider("C (inverse regularization)", 0.01, 10.0, 1.0)
        lr_params = dict(C=C, max_iter=2000, n_jobs=-1, multi_class="ovr")
    else:  # XGB
        xgb_lr = st.slider("learning_rate", 0.01, 0.5, 0.1)
        xgb_estimators = st.slider("n_estimators", 50, 1000, 500, 50)
        xgb_depth = st.slider("max_depth", 2, 12, 6)
        xgb_params = dict(
            learning_rate=xgb_lr,
            n_estimators=xgb_estimators,
            max_depth=xgb_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="multi:softprob",
            n_jobs=-1,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=random_state,
        )

    st.divider()
    st.header("Preprocessing")
    scale_features = st.checkbox("Standardize numeric features", value=True)
    drop_na_rows = st.checkbox("Drop rows where label is missing", value=True)

# -----------------------------
# Load data
# -----------------------------
st.info(f"Fetching **{dataset_name}** from NASA Exoplanet Archive TAP…")
try:
    df = fetch_table(meta["table"], [meta["label_col"], *selected_features])
except Exception as e:
    st.error(f"Failed to fetch `{meta['table']}` table from archive: {e}")
    st.stop()

# Clean labels into a common 3-class scheme
LABELS_3 = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]


def normalize_labels(series: pd.Series, dataset: str) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    if dataset.startswith("Kepler"):
        # koi_disposition: CONFIRMED / CANDIDATE / FALSE POSITIVE / NOT DISPOSITIONED
        s = s.replace({
            "NOT DISPOSITIONED": np.nan,
        })
        return s
    elif dataset.startswith("K2"):
        # disposition: CANDIDATE, FALSE POSITIVE [CANDIDATE], CONFIRMED, REFUTED [PLANET]
        s = s.replace({
            "FALSE POSITIVE [CANDIDATE]": "FALSE POSITIVE",
            "REFUTED [PLANET]": "FALSE POSITIVE",
        })
        return s
    else:  # TESS TOI tfopwg_disp: CP, KP, PC, APC, FP, FA
        s = s.replace({
            "CP": "CONFIRMED",
            "KP": "CONFIRMED",
            "PC": "CANDIDATE",
            "APC": "CANDIDATE",
            "FP": "FALSE POSITIVE",
            "FA": "FALSE POSITIVE",
        })
        return s

raw_n = len(df)
df["label"] = normalize_labels(df[meta["label_col"]], dataset_name)
if drop_na_rows:
    df = df.dropna(subset=["label"])  # remove NOT DISPOSITIONED etc.

st.success(f"Loaded {len(df):,} rows (from {raw_n:,}; after filtering missing labels).")

# Show a preview
with st.expander("Preview data (first 20 rows)"):
    st.dataframe(df.head(20))

# -----------------------------
# Build ML dataset
# -----------------------------
X = df[selected_features].copy()
# Keep only numeric columns (auto-coerce)
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors='coerce')

y = df["label"].copy()

num_features = list(X.columns)

preprocess_steps = []
preprocess_steps.append(("impute", SimpleImputer(strategy="median")))
if scale_features:
    preprocess_steps.append(("scale", StandardScaler(with_mean=True)))

preprocess = Pipeline(preprocess_steps)

# Choose model
if model_type.startswith("RandomForest"):
    model = RandomForestClassifier(**rf_params)
elif model_type.startswith("LogisticRegression"):
    model = LogisticRegression(**lr_params)
else:
    if not HAS_XGB:
        st.error("xgboost is not installed. Please `pip install xgboost`.")
        st.stop()
    model = XGBClassifier(**xgb_params)

clf = Pipeline([
    ("prep", preprocess),
    ("model", model),
])

# Train/Validation split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# Fit
with st.spinner("Training model…"):
    clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
try:
    # Probabilities are needed for ROC AUC; if unavailable, fall back gracefully
    y_proba = clf.predict_proba(X_test)
    auc = roc_auc_score(pd.get_dummies(y_test, columns=LABELS_3), y_proba, multi_class="ovr")
except Exception:
    auc = np.nan

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Validation metrics")
    st.code(classification_report(y_test, y_pred, digits=3))
    if not np.isnan(auc):
        st.metric("ROC AUC (OvR)", f"{auc:.3f}")
with col2:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred, labels=LABELS_3)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS_3], columns=[f"pred_{l}" for l in LABELS_3])
    st.dataframe(cm_df)

# Cross-validation (macro F1)
skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
try:
    cv_scores = cross_val_score(clf, X, y, scoring="f1_macro", cv=skf, n_jobs=-1)
    st.caption(f"Cross-val macro F1 ({cv_folds}-fold): mean {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
except Exception as e:
    st.caption(f"Cross-val skipped: {e}")

st.divider()
# -----------------------------
# Feature importance / coefficients (if supported)
# -----------------------------
st.subheader("Model insights (feature influence)")
try:
    # Pull importance from tree models or coefficients from linear models
    if model_type.startswith("RandomForest") and hasattr(clf.named_steps["model"], "feature_importances_"):
        importances = clf.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({"feature": num_features, "importance": importances}).sort_values("importance", ascending=False)
        st.dataframe(imp_df)
    elif model_type.startswith("LogisticRegression") and hasattr(clf.named_steps["model"], "coef_"):
        coefs = clf.named_steps["model"].coef_
        coef_df = pd.DataFrame(coefs.T, index=num_features, columns=[f"class_{c}" for c in clf.named_steps["model"].classes_]).sort_index()
        st.dataframe(coef_df)
    elif model_type.startswith("XGBoost") and hasattr(clf.named_steps["model"], "feature_importances_"):
        importances = clf.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({"feature": num_features, "importance": importances}).sort_values("importance", ascending=False)
        st.dataframe(imp_df)
    else:
        st.write("Model does not expose feature importances.")
except Exception as e:
    st.write(f"Could not compute importances: {e}")

# -----------------------------
# Inference on user-uploaded data
# -----------------------------
st.divider()
st.header("Predict on your own data")

data_help = f"Your CSV should include these numeric columns: {', '.join(num_features)}"
up = st.file_uploader("Upload CSV", type=["csv"], help=data_help)

if up:
    try:
        new_df = pd.read_csv(up)
        disp_cols = [c for c in new_df.columns if c in num_features]
        missing = [c for c in num_features if c not in new_df.columns]
        if missing:
            st.warning(f"Missing expected columns: {missing}. We'll fill them with NaNs (imputed).")
            for m in missing:
                new_df[m] = np.nan
        X_new = new_df[num_features].apply(pd.to_numeric, errors='coerce')
        preds = clf.predict(X_new)
        try:
            proba = clf.predict_proba(X_new)
            proba_df = pd.DataFrame(proba, columns=[f"P({c})" for c in clf.classes_])
            out_df = pd.concat([new_df.reset_index(drop=True), pd.Series(preds, name="prediction"), proba_df], axis=1)
        except Exception:
            out_df = pd.concat([new_df.reset_index(drop=True), pd.Series(preds, name="prediction")], axis=1)
        st.subheader("Predictions preview")
        st.dataframe(out_df.head(50))

        # Download
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions as CSV", data=csv_bytes, file_name="exoplanet_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

# -----------------------------
# Tips and provenance
# -----------------------------
st.divider()
st.subheader("Notes & Tips")
st.markdown(
    """
- **Class imbalance** is real. Try tree-based models with `class_weight='balanced'` (already enabled for RandomForest) or resampling.
- Feature sets vary by table. For example, Kepler KOI uses `koi_*` fields (period, duration, depth, radius), while K2/TOI use `pl_*`/`st_*` columns.
- You can broaden the feature set in the sidebar, but keep it mostly numeric to avoid heavy preprocessing.
- For rigorous science, prefer **cross-validation**, model calibration, and external validation against held-out sectors/campaigns.
- Record data **access date/time** and table DOIs when you publish results.
    """
)

st.caption("Data courtesy of NASA Exoplanet Archive — Caltech/IPAC. This tool pulls public TAP CSVs for KOI cumulative, K2 Planets & Candidates, and TESS TOI tables.")
