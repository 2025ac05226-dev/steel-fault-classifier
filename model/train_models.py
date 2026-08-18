"""
Surface-fault classification for steel plates - model training stage.

Reads the raw UCI "Steel Plates Faults" export (27 numeric measurements per
plate, seven mutually exclusive fault labels stored one-hot), folds the seven
label columns back into a single categorical target, fits six classifiers and
writes everything the Streamlit front-end needs to disk:

    model/*.joblib        one fitted pipeline per algorithm
    model/label_space.joblib
    model/metrics.json    scores measured on the held-out split
    test_data.csv         the held-out split itself, for app upload

Run from the repository root:  python model/train_models.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = REPO_ROOT / "data" / "steel_plates_faults_raw.csv"
MODEL_DIR = REPO_ROOT / "model"
TEST_CSV = REPO_ROOT / "test_data.csv"

# The seven one-hot columns that make up the target in the raw UCI export.
FAULT_COLUMNS = [
    "Pastry",
    "Z_Scratch",
    "K_Scratch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]

TARGET = "Fault_Type"
HOLDOUT_FRACTION = 0.25
SEED = 20250818


def load_plate_measurements(csv_path: Path) -> pd.DataFrame:
    """Load the raw export and collapse the one-hot target into one column."""
    frame = pd.read_csv(csv_path)

    missing = [c for c in FAULT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"raw export is missing fault columns: {missing}")

    # Every row in this dataset carries exactly one fault label; assert it
    # rather than silently letting idxmax pick an arbitrary winner.
    labels_per_row = frame[FAULT_COLUMNS].sum(axis=1)
    if not (labels_per_row == 1).all():
        raise ValueError("expected exactly one active fault label per row")

    frame[TARGET] = frame[FAULT_COLUMNS].idxmax(axis=1)
    return frame.drop(columns=FAULT_COLUMNS)


def build_model_zoo(n_classes: int) -> dict:
    """The six algorithms compared in this study.

    Distance- and gradient-based learners sit behind a StandardScaler because
    the raw measurements span wildly different ranges (Pixels_Areas reaches
    ~150k while Luminosity_Index lives in [-1, 1]). The tree-based learners are
    scale-invariant, so they consume the raw columns directly.
    """
    balanced_tree_depth = 12

    return {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        C=1.0,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=balanced_tree_depth,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                )
            ]
        ),
        "kNN": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=7,
                        weights="distance",
                        metric="minkowski",
                    ),
                ),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", GaussianNB(var_smoothing=1e-8)),
            ]
        ),
        "Random Forest (Ensemble)": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_features="sqrt",
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                )
            ]
        ),
        "Gradient Boosting (Ensemble)": Pipeline(
            [
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.1,
                        max_depth=3,
                        subsample=0.9,
                        random_state=SEED,
                    ),
                )
            ]
        ),
    }


def score_predictions(y_true, y_pred, y_proba, class_order) -> dict:
    """The six metrics the assignment asks for.

    This is a seven-class problem, so precision / recall / F1 are averaged with
    `weighted` (each class contributes in proportion to its support) and AUC is
    computed one-vs-rest over the full probability matrix.
    """
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted", labels=class_order
        ),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return {k: round(float(v), 4) for k, v in scores.items()}


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    plates = load_plate_measurements(RAW_CSV)
    features = [c for c in plates.columns if c != TARGET]

    print(f"plates       : {len(plates)} rows x {len(features)} features")
    print(f"fault classes: {plates[TARGET].nunique()}")
    print(plates[TARGET].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        plates[features],
        plates[TARGET],
        test_size=HOLDOUT_FRACTION,
        random_state=SEED,
        stratify=plates[TARGET],
    )
    print(f"\ntrain/test   : {len(X_train)} / {len(X_test)}")

    # The app scores uploads against this exact column order.
    class_order = sorted(plates[TARGET].unique())
    joblib.dump(
        {"features": features, "classes": class_order, "target": TARGET},
        MODEL_DIR / "label_space.joblib",
    )

    zoo = build_model_zoo(n_classes=len(class_order))
    results = {}

    for name, pipeline in zoo.items():
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        y_pred = pipeline.predict(X_test)
        # Reorder the probability matrix to `class_order` so AUC lines up with
        # the labels we pass in, regardless of the estimator's internal order.
        proba_raw = pipeline.predict_proba(X_test)
        col_index = [list(pipeline.classes_).index(c) for c in class_order]
        y_proba = proba_raw[:, col_index]

        results[name] = score_predictions(y_test, y_pred, y_proba, class_order)
        results[name]["FitSeconds"] = round(fit_seconds, 2)

        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        # compress=3 keeps the 500-tree forest around 8 MB instead of 31 MB,
        # which matters for repo size and Streamlit Cloud cold-start time.
        joblib.dump(pipeline, MODEL_DIR / f"{slug}.joblib", compress=3)
        print(f"  fitted {name:<30} acc={results[name]['Accuracy']:.4f} "
              f"mcc={results[name]['MCC']:.4f} ({fit_seconds:.1f}s)")

    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    holdout = X_test.copy()
    holdout[TARGET] = y_test.values
    holdout.to_csv(TEST_CSV, index=False)
    print(f"\nwrote {TEST_CSV.name} ({len(holdout)} rows) and model/metrics.json")

    table = pd.DataFrame(results).T
    print("\n" + table.to_string())


if __name__ == "__main__":
    main()
