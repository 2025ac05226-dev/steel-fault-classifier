"""
Steel Plate Surface-Fault Inspector
===================================

Streamlit front-end for the six classifiers trained in model/train_models.py.

Workflow the app supports:
  1. upload a held-out CSV of plate measurements (test_data.csv in this repo),
  2. pick one of the six trained classifiers,
  3. read the six evaluation metrics scored on that upload,
  4. inspect the confusion matrix and per-class classification report.

Launch locally with:  streamlit run app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = Path(__file__).parent / "model"

# Display name -> artifact filename, in the order they appear in the dropdown.
MODEL_CATALOGUE = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
    "Gradient Boosting (Ensemble)": "gradient_boosting_ensemble.joblib",
}

FAULT_GLOSSARY = {
    "Pastry": "Pastry-type surface defect",
    "Z_Scratch": "Z-shaped scratch",
    "K_Scratch": "K-shaped scratch",
    "Stains": "Staining / discolouration",
    "Dirtiness": "Embedded dirt particles",
    "Bumps": "Raised bumps on the surface",
    "Other_Faults": "Unclassified / mixed defect",
}

st.set_page_config(
    page_title="Steel Plate Fault Inspector",
    page_icon="🔩",
    layout="wide",
)


# --------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_label_space():
    return joblib.load(MODEL_DIR / "label_space.joblib")


@st.cache_resource(show_spinner=False)
def load_classifier(filename: str):
    return joblib.load(MODEL_DIR / filename)


@st.cache_data(show_spinner=False)
def load_training_scorecard():
    """Metrics recorded at training time, used for the leaderboard tab."""
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        return None
    return pd.read_json(path).T


def read_upload(upload) -> pd.DataFrame:
    return pd.read_csv(upload)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def evaluate(pipeline, X: pd.DataFrame, y_true: pd.Series, class_order: list) -> dict:
    """Score one fitted pipeline on the uploaded rows."""
    y_pred = pipeline.predict(X)

    proba_raw = pipeline.predict_proba(X)
    col_index = [list(pipeline.classes_).index(c) for c in class_order]
    y_proba = proba_raw[:, col_index]

    # AUC is only defined when the upload actually contains more than one
    # class; a single-class upload is legal but leaves AUC undefined.
    present = sorted(pd.unique(y_true))
    if len(present) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted", labels=class_order
        )

    return {
        "y_pred": y_pred,
        "scores": {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": auc,
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        },
    }


def draw_confusion(y_true, y_pred, class_order):
    matrix = confusion_matrix(y_true, y_pred, labels=class_order)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="rocket_r",
        cbar=False,
        xticklabels=class_order,
        yticklabels=class_order,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted fault", fontsize=11)
    ax.set_ylabel("Actual fault", fontsize=11)
    ax.set_title("Confusion matrix", fontsize=13, pad=12)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
space = load_label_space()
FEATURES = space["features"]
CLASS_ORDER = space["classes"]
TARGET = space["target"]

st.title("🔩 Steel Plate Surface-Fault Inspector")
st.caption(
    "Six classifiers trained on the UCI *Steel Plates Faults* dataset "
    "(1,941 plates · 27 geometric and luminosity measurements · 7 fault types). "
    "Upload a held-out CSV to score any model on unseen plates."
)

with st.sidebar:
    st.header("1 · Upload test data")
    upload = st.file_uploader(
        "CSV of plate measurements",
        type="csv",
        help=f"Needs the 27 feature columns plus a '{TARGET}' column of true labels.",
    )
    st.caption("Use `test_data.csv` from this repository (486 held-out plates).")

    st.divider()
    st.header("2 · Choose a model")
    chosen = st.selectbox("Classifier", list(MODEL_CATALOGUE.keys()), index=4)

    st.divider()
    with st.expander("Fault types in this dataset"):
        for label, description in FAULT_GLOSSARY.items():
            st.markdown(f"**{label}** — {description}")

if upload is None:
    st.info(
        "⬅️ Upload `test_data.csv` in the sidebar to score a model. "
        "The leaderboard below shows results recorded during training."
    )
    scorecard = load_training_scorecard()
    if scorecard is not None:
        st.subheader("Training-time leaderboard (486 held-out plates)")
        st.dataframe(
            scorecard.drop(columns=["FitSeconds"], errors="ignore")
            .style.format("{:.4f}")
            .background_gradient(cmap="Greens", axis=0),
            width="stretch",
        )
    st.stop()

# ---- validate the upload -------------------------------------------------
plates = read_upload(upload)

missing_features = [c for c in FEATURES if c not in plates.columns]
if missing_features:
    st.error(
        f"The upload is missing {len(missing_features)} required feature column(s): "
        f"`{'`, `'.join(missing_features[:8])}`"
        + (" …" if len(missing_features) > 8 else "")
    )
    st.stop()

if TARGET not in plates.columns:
    st.error(
        f"The upload has no `{TARGET}` column, so it cannot be scored. "
        "Add the true fault label for each row."
    )
    st.stop()

unknown = sorted(set(plates[TARGET]) - set(CLASS_ORDER))
if unknown:
    st.error(f"Unrecognised fault label(s) in `{TARGET}`: {unknown}")
    st.stop()

X = plates[FEATURES]
y_true = plates[TARGET]

top = st.container()
with top:
    a, b, c = st.columns(3)
    a.metric("Plates uploaded", f"{len(plates):,}")
    b.metric("Feature columns", len(FEATURES))
    c.metric("Fault classes present", y_true.nunique())

pipeline = load_classifier(MODEL_CATALOGUE[chosen])
outcome = evaluate(pipeline, X, y_true, CLASS_ORDER)
scores = outcome["scores"]
y_pred = outcome["y_pred"]

st.subheader(f"Evaluation metrics — {chosen}")
metric_cols = st.columns(6)
for col, (label, value) in zip(metric_cols, scores.items()):
    col.metric(label, "n/a" if np.isnan(value) else f"{value:.4f}")

tab_matrix, tab_report, tab_compare, tab_preds = st.tabs(
    ["Confusion matrix", "Classification report", "Compare all models", "Predictions"]
)

with tab_matrix:
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(draw_confusion(y_true, y_pred, CLASS_ORDER))
    with right:
        st.markdown("**Where this model struggles**")
        matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
        confusions = []
        for i, actual in enumerate(CLASS_ORDER):
            for j, predicted in enumerate(CLASS_ORDER):
                if i != j and matrix[i][j] > 0:
                    confusions.append((matrix[i][j], actual, predicted))
        confusions.sort(reverse=True)
        if not confusions:
            st.success("No misclassifications on this upload.")
        for count, actual, predicted in confusions[:6]:
            st.markdown(f"- **{count}** × `{actual}` → predicted `{predicted}`")

with tab_report:
    report = classification_report(
        y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).T
    report_df["support"] = report_df["support"].astype(int)
    st.dataframe(
        report_df.style.format(
            {"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:d}"}
        ),
        width="stretch",
    )
    st.caption(
        "Per-class precision / recall / F1. The rare classes (Dirtiness, Stains) "
        "carry the least support and are the hardest to recall."
    )

with tab_compare:
    st.markdown("Scoring **every** model on this upload:")
    rows = {}
    progress = st.progress(0.0)
    for i, (name, artifact) in enumerate(MODEL_CATALOGUE.items(), start=1):
        rows[name] = evaluate(load_classifier(artifact), X, y_true, CLASS_ORDER)["scores"]
        progress.progress(i / len(MODEL_CATALOGUE))
    progress.empty()

    comparison = pd.DataFrame(rows).T
    st.dataframe(
        comparison.style.format("{:.4f}").background_gradient(cmap="Greens", axis=0),
        width="stretch",
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    comparison[["Accuracy", "F1", "MCC"]].plot(kind="bar", ax=ax, width=0.78)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy / F1 / MCC by model")
    ax.legend(loc="lower right", frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    st.pyplot(fig)

    best = comparison["MCC"].idxmax()
    st.success(f"Best model on this upload by MCC: **{best}** ({comparison.loc[best, 'MCC']:.4f})")

with tab_preds:
    preview = plates.copy()
    preview["Predicted"] = y_pred
    preview["Correct"] = np.where(preview[TARGET] == preview["Predicted"], "✅", "❌")
    only_wrong = st.checkbox("Show misclassified plates only", value=False)
    if only_wrong:
        preview = preview[preview["Correct"] == "❌"]
    st.dataframe(
        preview[[TARGET, "Predicted", "Correct"] + FEATURES[:6]],
        width="stretch",
        height=420,
    )
    st.download_button(
        "Download predictions as CSV",
        preview.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{chosen.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
