# Steel Plate Surface-Fault Inspector

Machine Learning Assignment 2 — M.Tech (AIML/DSE), BITS Pilani WILP
Six classification models compared on the UCI *Steel Plates Faults* dataset, served through an interactive Streamlit application.

---

## a. Problem Statement

In a steel rolling mill, finished plates are inspected for surface defects before dispatch. Manual visual inspection is slow, inconsistent between inspectors, and cannot keep pace with line speed. Automated optical inspection systems can measure the geometry and luminosity of a detected surface anomaly, but a measurement alone does not tell the operator *what kind* of defect it is — and the corrective action differs by defect type (a rolling-mill adjustment for `Bumps`, a cleaning-stage fix for `Dirtiness`, a coolant issue for `Stains`).

**The task is a supervised multi-class classification problem:** given 27 geometric and luminosity measurements extracted from a detected anomaly on a steel plate, predict which of **7 mutually exclusive fault types** it is.

Two properties make this a genuinely difficult problem rather than a toy one:

1. **Severe class imbalance** — the rarest class (`Dirtiness`, 55 plates) is outnumbered roughly 12:1 by the most common (`Other_Faults`, 673 plates). Plain accuracy is therefore a misleading headline metric, which is why **MCC** is treated as the primary decision metric in this study.
2. **A heterogeneous catch-all class** — `Other_Faults` is not a coherent defect type but a bucket of everything the original annotators could not assign elsewhere. It has no consistent signature, and as shown in the observations below, **it is the single biggest driver of the accuracy differences between the six models.**

---

## b. Dataset Description

| Property | Value |
|---|---|
| **Name** | Steel Plates Faults |
| **Source** | UCI Machine Learning Repository — [dataset #198](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) |
| **Donor** | Semeion, Research Center of Sciences of Communication, Rome, Italy |
| **Instances** | **1,941** (assignment minimum: 500 ✅) |
| **Features** | **27**, all numeric (assignment minimum: 12 ✅) |
| **Target** | `Fault_Type` — 7 classes |
| **Missing values** | None |
| **Duplicate rows** | None |
| **Task type** | Multi-class classification |

### Target distribution

| Fault type | Count | Share | Description |
|---|---:|---:|---|
| `Other_Faults` | 673 | 34.7% | Unclassified / mixed defect (catch-all bucket) |
| `Bumps` | 402 | 20.7% | Raised bumps on the plate surface |
| `K_Scratch` | 391 | 20.1% | K-shaped scratch |
| `Z_Scratch` | 190 | 9.8% | Z-shaped scratch |
| `Pastry` | 158 | 8.1% | Pastry-type surface defect |
| `Stains` | 72 | 3.7% | Staining / discolouration |
| `Dirtiness` | 55 | 2.8% | Embedded dirt particles |

**Imbalance ratio ≈ 12.2 : 1** (majority to minority).

### Feature groups

The 27 predictors fall into four natural groups:

- **Bounding-box geometry (4)** — `X_Minimum`, `X_Maximum`, `Y_Minimum`, `Y_Maximum`
- **Size & shape (10)** — `Pixels_Areas`, `X_Perimeter`, `Y_Perimeter`, `Edges_Index`, `Empty_Index`, `Square_Index`, `Outside_X_Index`, `Edges_X_Index`, `Edges_Y_Index`, `Outside_Global_Index`
- **Luminosity (4)** — `Sum_of_Luminosity`, `Minimum_of_Luminosity`, `Maximum_of_Luminosity`, `Luminosity_Index`
- **Plate & derived (9)** — `Length_of_Conveyer`, `TypeOfSteel_A300`, `TypeOfSteel_A400`, `Steel_Plate_Thickness`, `LogOfAreas`, `Log_X_Index`, `Log_Y_Index`, `Orientation_Index`, `SigmoidOfAreas`

### Preprocessing

In the raw UCI export the target is stored as **seven one-hot columns**. Every row carries exactly one active label (verified programmatically in `model/train_models.py`), so these are folded back into a single categorical `Fault_Type` column via `idxmax`.

Feature scales differ by seven orders of magnitude — `Y_Maximum` spans ≈ 1.3 × 10⁷ while `Luminosity_Index` lives in [-1, 1]. Distance- and gradient-based learners (Logistic Regression, kNN, Naive Bayes) are therefore wrapped in a `StandardScaler` inside an sklearn `Pipeline`; the tree-based learners are scale-invariant and consume raw columns. Putting the scaler *inside* the pipeline ensures it is fitted on training data only, so no test-set statistics leak into training.

**Split:** stratified 75 / 25 → **1,455 training** / **486 test** plates, `random_state=20250818`. The 486-row test split is saved as `test_data.csv` and is the file uploaded to the Streamlit app.

---

## c. GitHub Repository Link

**https://github.com/2025ac05226-dev/steel-fault-classifier**

**Live Streamlit App:** _(deploying — link added after Streamlit Cloud deploy)_

### Repository structure

```
steel-fault-classifier/
├── app.py                          Streamlit application
├── requirements.txt                pinned dependencies
├── README.md                       this file
├── test_data.csv                   486 held-out plates for app upload
├── data/
│   └── steel_plates_faults_raw.csv full 1,941-row UCI export
└── model/
    ├── train_models.py             training + evaluation pipeline
    ├── metrics.json                scores recorded at training time
    ├── label_space.joblib          feature order + class order
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest_ensemble.joblib
    └── gradient_boosting_ensemble.joblib
```

### Reproducing the results

```bash
pip install -r requirements.txt
python model/train_models.py     # retrains all six models, rewrites metrics.json + test_data.csv
streamlit run app.py             # launches the app at http://localhost:8501
```

---

## d. Models Used

All six classifiers are trained on the **same** 1,455-plate training split and evaluated on the **same** 486-plate held-out test split.

| # | Model | Key hyperparameters |
|---|---|---|
| 1 | Logistic Regression | `max_iter=3000`, `C=1.0`, `class_weight='balanced'`, multinomial |
| 2 | Decision Tree | `max_depth=12`, `min_samples_leaf=4`, `class_weight='balanced'` |
| 3 | kNN | `n_neighbors=7`, `weights='distance'`, Minkowski metric, scaled |
| 4 | Naive Bayes | Gaussian, `var_smoothing=1e-8`, scaled |
| 5 | Random Forest (Ensemble) | `n_estimators=500`, `max_features='sqrt'`, `class_weight='balanced_subsample'` |
| 6 | Gradient Boosting (Ensemble) | `n_estimators=200`, `learning_rate=0.1`, `max_depth=3`, `subsample=0.9` |

### Metric definitions

Because this is a 7-class problem, **Precision, Recall and F1 are weighted averages** (each class contributes in proportion to its support) and **AUC is one-vs-rest**, weighted, computed over the full 7-column predicted-probability matrix. **MCC** is computed on the multi-class confusion matrix directly and needs no averaging — it is the metric least fooled by the class imbalance, and is used here to pick the winner.

### Comparison Table — evaluation metrics on the 486-plate held-out test set

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6584 | 0.8870 | 0.6956 | 0.6584 | 0.6492 | 0.5956 |
| Decision Tree | 0.6502 | 0.8398 | 0.6777 | 0.6502 | 0.6480 | 0.5763 |
| kNN | 0.7305 | 0.9032 | 0.7293 | 0.7305 | 0.7233 | 0.6539 |
| Naive Bayes | 0.6337 | 0.8745 | 0.6806 | 0.6337 | 0.6268 | 0.5673 |
| **Random Forest (Ensemble)** | **0.7737** | **0.9354** | **0.7833** | **0.7737** | **0.7729** | **0.7061** |
| Gradient Boosting (Ensemble) | 0.7654 | 0.9310 | 0.7677 | 0.7654 | 0.7647 | 0.6988 |

*(Bold = best score in that column. All figures are reproducible by running `python model/train_models.py`.)*

### Supporting diagnostics

Two additional measurements were taken to explain *why* the models rank the way they do.

**Per-class recall** (the metric that exposes what the headline numbers hide):

| Model | Bumps | Dirtiness | K_Scratch | **Other_Faults** | Pastry | Stains | Z_Scratch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.554 | 0.786 | 0.918 | **0.417** | 0.821 | 0.944 | 0.917 |
| Decision Tree | 0.713 | 0.571 | 0.939 | **0.423** | 0.590 | 0.833 | 0.729 |
| kNN | 0.653 | 0.857 | 0.939 | **0.685** | 0.308 | 0.889 | 0.875 |
| Naive Bayes | 0.733 | 0.571 | 0.898 | **0.327** | 0.667 | 0.944 | 0.833 |
| Random Forest | 0.644 | 0.714 | 0.949 | **0.792** | 0.513 | 0.833 | 0.833 |
| Gradient Boosting | 0.743 | 0.786 | 0.959 | **0.690** | 0.462 | 0.833 | 0.896 |

**Train vs. test accuracy** (generalisation gap):

| Model | Train acc | Test acc | Gap |
|---|---:|---:|---:|
| Logistic Regression | 0.6818 | 0.6584 | +0.023 |
| Decision Tree | 0.8330 | 0.6502 | +0.183 |
| kNN | 1.0000 | 0.7305 | +0.270 |
| Naive Bayes | 0.6337 | 0.6337 | −0.000 |
| Random Forest | 1.0000 | 0.7737 | +0.226 |
| Gradient Boosting | 1.0000 | 0.7654 | +0.235 |

---

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Accuracy 0.6584 but AUC 0.8870 — a 0.23 spread that is the most informative result in the table. The model *ranks* classes well but places its decision boundaries badly, because a single linear boundary per class cannot carve out `Other_Faults`. Its per-class recall is bimodal: excellent on the geometrically distinctive faults (`Stains` 0.944, `Z_Scratch` 0.917, `K_Scratch` 0.918) and near-random on the catch-all (`Other_Faults` 0.417). Its generalisation gap is the second-smallest at +0.023, confirming it is **underfitting, not overfitting** — the linear hypothesis class is simply too rigid for this decision surface. `class_weight='balanced'` is what rescues its recall on the 55-sample `Dirtiness` class (0.786, beating both ensembles). |
| **Decision Tree** | Weakest of the non-Bayesian models (Accuracy 0.6502, MCC 0.5763) and the weakest AUC overall at 0.8398. The low AUC is structural: a single tree emits piecewise-constant leaf probabilities, so it produces coarse, poorly-calibrated confidence scores that rank badly even when the top-1 prediction is right. Its +0.183 train/test gap shows meaningful overfitting *despite* `max_depth=12` and `min_samples_leaf=4` — the axis-aligned splits chase noise in the 27 correlated features. Recall collapses on `Pastry` (0.590) and `Z_Scratch` (0.729) relative to Logistic Regression, indicating high variance in which splits it happens to pick. It is best read as the single-estimator baseline that motivates the ensembles. |
| **kNN** | The strongest non-ensemble model by a clear margin (Accuracy 0.7305, MCC 0.6539) and the biggest surprise in the study. Its advantage is entirely concentrated in `Other_Faults` recall (0.685 vs. 0.417 for Logistic Regression) — because that class is a *union of several small local clusters* rather than one convex region, and a local method models unions of clusters naturally where a global parametric model cannot. It pays for this with the worst `Pastry` recall in the table (0.308): `Pastry` occupies a sparse region adjacent to denser classes, so its 7 nearest neighbours are usually dominated by the majority class. Train accuracy is exactly 1.0000 (expected — with `weights='distance'` each training point is its own nearest neighbour at distance 0, so this is memorisation, not a red flag). **`StandardScaler` is load-bearing here**, not cosmetic: unscaled, `Y_Maximum`'s ~1.3 × 10⁷ span would dominate the Minkowski distance and reduce kNN to a 1-D nearest-neighbour on that column. |
| **Naive Bayes** | Lowest accuracy (0.6337), F1 (0.6268) and MCC (0.5673) of the six — and the reason is directly measurable in the data. GaussianNB assumes conditional independence between features, but this feature set contains **7 pairs correlated above \|r\| > 0.9 and 21 pairs above \|r\| > 0.7** (`Pixels_Areas` / `LogOfAreas` / `SigmoidOfAreas` are transforms of one another; `X_Minimum` / `X_Maximum` bound the same box). Duplicated evidence is therefore counted several times over, producing overconfident posteriors, which is exactly why it is worst on the diffuse `Other_Faults` class (recall 0.327, lowest in the table). Notably its generalisation gap is **−0.0001 — the model does not overfit at all**; its error is pure bias from a violated assumption, not variance. It is not useless: it ties for the best `Stains` recall (0.944) and trains in well under a second, so it remains a legitimate fast baseline. |
| **Random Forest (Ensemble)** | **Best model on every one of the six metrics** — Accuracy 0.7737, AUC 0.9354, Precision 0.7833, Recall 0.7737, F1 0.7729, MCC 0.7061. Bagging 500 decorrelated trees cancels the variance that cripples the single Decision Tree, lifting accuracy by +12.3 points and MCC by +0.130 over it, while `max_features='sqrt'` prevents the correlated feature groups from being selected in lockstep. Its win is not spread evenly: it comes almost entirely from `Other_Faults` recall (**0.792, best in the table by a wide margin**), which is precisely the class every other model fails on. The trade-off is visible on `Pastry` (0.513) — averaged voting smooths away the small, sharp region that class occupies. The +0.226 train/test gap looks large but is the normal signature of a fully-grown forest, and is not harmful here since test performance is still the highest recorded. |
| **Gradient Boosting (Ensemble)** | A very close second (Accuracy 0.7654, MCC 0.6988) — within 0.008 accuracy and 0.007 MCC of Random Forest, a margin far smaller than the sampling noise of a 486-row test set, so the two ensembles should be treated as **statistically indistinguishable in quality**. It is the best model on three individual classes (`Bumps` 0.743, `K_Scratch` 0.959, `Z_Scratch` 0.896) but loses ground on `Other_Faults` (0.690 vs. 0.792), and that one class is enough to decide the ranking. The decisive practical difference is cost: **6.76 s to train versus 0.67 s for Random Forest — 10× slower for slightly worse results**, because its depth-3 stages are built sequentially and cannot be parallelised, whereas the forest's 500 trees fit in parallel across cores. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — it takes the top score on all six required metrics simultaneously (Accuracy 0.7737, AUC 0.9354, Precision 0.7833, Recall 0.7737, F1 0.7729, MCC 0.7061), trains in 0.67 s, and wins for a reason specific to this dataset rather than by generic ensemble advantage: it is the only model that handles the heterogeneous `Other_Faults` class competently (recall 0.792, next-best 0.690), and since that class is 34.7% of all plates, it dominates every aggregate metric. Because MCC is the metric least distorted by the 12:1 imbalance, its 0.7061 — a full +0.139 above the best non-ensemble linear model — is the most trustworthy summary of the gap. Gradient Boosting is a defensible alternative on quality alone, but is 10× more expensive to train for no measurable gain. |

**Practical caveat:** the ceiling for every model here is set by `Other_Faults`. Since that label is an annotation artefact (a bucket for defects the original annotators could not classify) rather than a real physical defect category, the most promising next step is not further hyperparameter tuning but **re-labelling or excluding that class** — a 6-class version of this problem would likely exceed 0.90 accuracy for all six models.

---

## Streamlit Application

**Live app:** _(deploying — link added after Streamlit Cloud deploy)_

The app implements all four required features:

| # | Required feature | Where it lives |
|---|---|---|
| a | **Dataset upload option (CSV)** | Sidebar → *1 · Upload test data*. Accepts `test_data.csv` (486 rows). Validates that all 27 feature columns and the `Fault_Type` column are present, and rejects unknown class labels with a clear error rather than a stack trace. |
| b | **Model selection dropdown** | Sidebar → *2 · Choose a model*. All six trained classifiers. |
| c | **Display of evaluation metrics** | Six metric tiles — Accuracy, AUC, Precision, Recall, F1, MCC — recomputed live on the uploaded rows. |
| d | **Confusion matrix / classification report** | *Confusion matrix* tab (7×7 annotated heatmap plus a ranked "where this model struggles" list of the most frequent misclassifications) and *Classification report* tab (per-class precision / recall / F1 / support). |

Two extras beyond the requirement:

- **Compare all models** tab — scores all six classifiers on your upload at once and renders a ranked table and grouped bar chart, so the comparison table above can be reproduced live in the browser.
- **Predictions** tab — per-plate predicted vs. actual with a misclassified-only filter and a CSV download of the predictions.

Only the 486-row test split is shipped for upload, in line with the assignment's guidance to keep within the Streamlit free-tier capacity.

---

## Tech Stack

Python · scikit-learn 1.6.1 · pandas · NumPy · Matplotlib · seaborn · Streamlit · joblib

## Academic Integrity

This work was carried out independently for BITS Pilani WILP Machine Learning Assignment 2. The dataset is publicly available from the UCI Machine Learning Repository; all modelling code, evaluation logic, application code and written analysis in this repository are original. AI tools were used for learning support only.
