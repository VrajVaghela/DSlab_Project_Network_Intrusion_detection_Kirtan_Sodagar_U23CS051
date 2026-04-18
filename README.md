<div align="center">

# 🔐 Network Intrusion Detection System
### Research-Grade NIDS on NSL-KDD | CS361 Data Science Lab | SVNIT Surat

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=for-the-badge)](https://shap.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-4ECDC4?style=for-the-badge)](https://optuna.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Detecting malicious network traffic with supervised classification, unsupervised anomaly detection, SHAP explainability, and Optuna hyperparameter tuning — all on the NSL-KDD benchmark.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Dataset](#-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Techniques & Upgrades](#-techniques--upgrades)
- [Results](#-results)
- [Generated Figures](#-generated-figures)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## 🧠 Overview

This notebook implements a **research-grade Network Intrusion Detection System (NIDS)** on the widely-used **NSL-KDD benchmark dataset**. It goes well beyond a basic ML pipeline — incorporating production-quality practices such as:

- ✅ **One-Hot Encoding** instead of LabelEncoder for nominal categorical features  
- ✅ **SMOTE oversampling** to handle critically rare U2R and R2L attack classes  
- ✅ **Optuna Bayesian hyperparameter search** for XGBoost  
- ✅ **5-Fold Stratified Cross-Validation** for statistically robust evaluation  
- ✅ **Threshold Optimization** via Youden's J statistic and F1-maximisation  
- ✅ **SHAP Explainability** with beeswarm, bar, and waterfall plots  
- ✅ **Isolation Forest** with training-data-derived contamination (no leakage)  
- ✅ **PCA Scree Plot** and **attack-category boxplots** for deep EDA  

---

## 🚀 Key Contributions

| Contribution | Impact |
|---|---|
| **OHE for `service` (70+ values), `flag`, `protocol_type`** | Eliminates false ordinal relationships that bias tree splits |
| **SMOTE on minority classes** | Critical for U2R/R2L which are <1% of traffic |
| **Optuna TPE search (30 trials, 6 params)** | Outperforms default XGBoost configuration |
| **Threshold optimization** | Security-aware maximisation of Recall over Precision |
| **SHAP TreeExplainer** | Analyst-grade local + global explainability for audit compliance |
| **Training-only contamination** | Fixes Isolation Forest data leakage (was hardcoded `0.46` from full dataset) |
| **Stratified 5-Fold CV** | Proves model stability; reports mean ± std F1 across folds |

---

## 📦 Dataset

**NSL-KDD** — The standard benchmark for Network Intrusion Detection research. Corrects the class-duplication issues of the original KDD'99 dataset.

| Property | Value |
|---|---|
| **Source** | [defcom17/NSL_KDD on GitHub](https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt) |
| **Rows** | 125,973 |
| **Features** | 41 (3 categorical + 38 numeric) |
| **Labels** | Binary (normal / attack) + 5-class (Normal, DoS, Probe, R2L, U2R) |
| **Attack ratio** | ~46% |

### Attack Category Breakdown

```
Normal  ████████████████████░░░░░  53.5%
DoS     ████████████░░░░░░░░░░░░░  36.5%
Probe   ████░░░░░░░░░░░░░░░░░░░░░   8.5%
R2L     █░░░░░░░░░░░░░░░░░░░░░░░░   1.2%
U2R     ░░░░░░░░░░░░░░░░░░░░░░░░░   0.1%
```

> ⚠️ U2R and R2L attacks are critically rare — this is why **SMOTE is non-negotiable**.

---

## 🏗️ Pipeline Architecture

```
Raw NSL-KDD Dataset (125,973 rows)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Section 1 — Data Loading & Label Creation        │
│  Binary label (0/1) + 5-class attack_category     │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  Section 2 — EDA                                  │
│  • Label distributions  • Correlation heatmap     │
│  • Feature histograms   • Scree plot (PCA)        │
│  • Categorical charts   • Boxplots by category    │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  Section 3 — Preprocessing                        │
│  • OHE (protocol_type, service, flag)             │
│  • Feature engineering (4 derived features)       │
│  • Train/Test split (80/20, stratified)           │
│  • StandardScaler (fit on train only)             │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  Section 3b — SMOTE Oversampling                  │
│  Synthesises minority class samples (U2R, R2L)    │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  Section 4b — 5-Fold Stratified Cross-Validation  │
│  F1 mean ± std for RF and XGBoost                 │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  Section 4c — Optuna Hyperparameter Tuning        │
│  30 TPE trials over 6 XGBoost parameters          │
└───────────────────────┬───────────────────────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
┌─────────────────────┐ ┌─────────────────────────┐
│ Section 5           │ │ Section 7                │
│ Random Forest       │ │ Isolation Forest         │
│ (SMOTE + balanced)  │ │ (train-derived contam.)  │
└──────────┬──────────┘ └──────────┬──────────────┘
           │                       │
           ▼                       │
┌─────────────────────┐            │
│ Section 6           │            │
│ XGBoost             │            │
│ (Optuna params)     │            │
└──────────┬──────────┘            │
           └──────────┬────────────┘
                      ▼
┌───────────────────────────────────────────────────┐
│  Section 8 — Evaluation & Comparison              │
│  Confusion matrices | ROC | Precision-Recall      │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  Section 7b — Threshold Optimization              │
│  Youden's J + F1-max threshold search             │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  Section 8b — SHAP Explainability                 │
│  Beeswarm | Bar | Waterfall (500 test samples)    │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  Section 10 — Multi-Class Classification          │
│  5-class RF (Normal/DoS/Probe/R2L/U2R)            │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  Section 12 — Final Summary Table                 │
│  CV_F1_Mean/Std | Threshold | Training_Samples    │
└───────────────────────────────────────────────────┘
```

---

## 🔬 Techniques & Upgrades

### 1. One-Hot Encoding (OHE)
```python
# OHE avoids spurious ordinal assumptions on nominal features
df_model = pd.get_dummies(df_model, columns=['protocol_type', 'service', 'flag'], drop_first=True)
```
`service` has 70+ unique values. LabelEncoder would assign integers (0–69), implying `ftp < http < smtp` — a completely meaningless ordering that biases tree splits. OHE encodes each category as an independent binary dimension.

---

### 2. SMOTE Oversampling
```python
smote = SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```
SMOTE synthesises new minority-class samples by interpolating between existing real samples in feature space — not by duplicating them. Applied **only on training data** to prevent evaluation leakage.

---

### 3. Optuna Hyperparameter Tuning
```
Search Space:
  n_estimators    : [100, 400]
  max_depth       : [4, 10]
  learning_rate   : [0.01, 0.3]  (log scale)
  subsample       : [0.6, 1.0]
  colsample_bytree: [0.6, 1.0]
  min_child_weight: [1, 10]

Method: Tree-of-Parzen-Estimators (TPE) | Trials: 30
Objective: Maximise F1 on 10% internal validation split
```

---

### 4. Threshold Optimization
```python
# Youden's J — maximises balance of sensitivity and specificity
youden_j = tpr - fpr
best_thresh_youden = thresholds[np.argmax(youden_j)]

# F1-max — directly targets the primary evaluation metric
f1_scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
best_thresh_f1 = thresholds[np.argmax(f1_scores)]
```
In security, a missed attack (False Negative) is far more costly than a false alarm (False Positive). Threshold tuning lets us shift the decision boundary for maximum Recall without catastrophic Precision loss.

---

### 5. SHAP Explainability

Three plot types generated:
- **Beeswarm** — global importance + direction of effect for each feature
- **Bar** — mean |SHAP| ranked importance
- **Waterfall** — local explanation for a single attack prediction

**Top 3 SHAP features (network security context):**

| Feature | Meaning | SHAP Insight |
|---|---|---|
| `serror_rate` | SYN error rate | High → SYN-flood DoS attack |
| `src_bytes` | Bytes sent from source | Abnormally large → potential data exfiltration |
| `count` | Connections to same host in 2s | Elevated → port scanning (Probe) or DoS amplification |

---

## 📊 Results

### Binary Classification Performance

| Model | Accuracy | Precision | Recall | **F1-Score** | ROC-AUC |
|---|---|---|---|---|---|
| 🥇 **XGBoost (Optuna)** | ~0.9985 | ~0.9980 | ~0.9991 | **~0.9985** | ~0.9999 |
| 🥈 **Random Forest** | ~0.9975 | ~0.9972 | ~0.9981 | **~0.9976** | ~0.9998 |
| 🥉 **Isolation Forest** | ~0.8200 | ~0.8100 | ~0.8700 | **~0.8400** | ~0.8200 |

> *Exact values will vary slightly based on Optuna trial outcomes and random seeds.*

### Cross-Validation (5-Fold Stratified)

| Model | Mean F1 | Std F1 |
|---|---|---|
| Random Forest | ~0.997 | ±0.001 |
| XGBoost | ~0.998 | ±0.001 |

> Low std confirms models generalise well — not sensitive to the particular random split.

### Multi-Class Classification (5 Categories)

| Category | Notes |
|---|---|
| Normal | Near-perfect classification |
| DoS | Excellent — large representative sample |
| Probe | Very good — distinct traffic patterns |
| R2L | Moderate — low sample count, SMOTE helps |
| U2R | Challenging — extremely rare, <100 samples |

---

## 🖼️ Generated Figures

All figures are saved as `.png` files in the working directory when the notebook is executed:

| File | Description |
|---|---|
| `fig1_label_distribution.png` | Binary & multi-label class distribution |
| `fig2_eda_categorical.png` | Attack categories, protocol types, top services |
| `fig3_feature_distributions.png` | Numeric feature histograms by label |
| `fig4_correlation_heatmap.png` | Full feature correlation heatmap |
| `fig4b_scree_plot.png` | PCA Scree plot — cumulative explained variance |
| `fig4c_boxplots_by_category.png` | Feature distributions by attack category |
| `fig5_confusion_matrices.png` | Confusion matrices for all 3 models |
| `fig6_roc_comparison.png` | ROC curves & performance bar chart |
| `fig7_precision_recall.png` | Precision-Recall curves |
| `fig7b_threshold_optimization.png` | Threshold vs F1 — Youden's J & F1-max marked |
| `fig8_feature_importance.png` | MDI feature importances — RF & XGBoost |
| `fig_shap_beeswarm.png` | SHAP beeswarm — global feature impact |
| `fig_shap_bar.png` | SHAP bar — mean absolute importance |
| `fig_shap_waterfall.png` | SHAP waterfall — single attack explanation |
| `fig9_pca_visualization.png` | PCA 2D projection — ground truth vs predictions |
| `fig10_multiclass_cm.png` | Multi-class confusion matrix |

**Total: 16 figures**

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Clone the Repository

```bash
git clone https://github.com/<your-username>/network-intrusion-detection.git
cd network-intrusion-detection
```

### Install Dependencies

**Option A — pip (recommended):**
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn imbalanced-learn shap optuna jupyter
```

**Option B — from requirements file (if provided):**
```bash
pip install -r requirements.txt
```

> 💡 The notebook's first cell also contains `%pip install ...` which installs everything automatically when you run it.

---

## ▶️ Usage

### Run in Jupyter
```bash
jupyter notebook network_intrusion_detection.ipynb
```

### Run in VS Code
1. Open `network_intrusion_detection.ipynb` in VS Code
2. Select your Python interpreter/kernel (top-right corner)
3. Click **Run All** (`Ctrl+Shift+P` → *Notebook: Run All Cells*)

### Run All Cells Headlessly
```bash
jupyter nbconvert --to notebook --execute network_intrusion_detection.ipynb --output network_intrusion_detection_executed.ipynb
```

> ⏱️ **Expected runtime:** ~5–10 minutes on a modern CPU (Optuna 30 trials + SHAP + RF training dominate the time). Internet required for dataset download on first run.

---

## 📁 Project Structure

```
network-intrusion-detection/
│
├── network_intrusion_detection.ipynb   # Main notebook (all 13 sections)
├── README.md                           # This file
│
└── figures/                            # Auto-generated when notebook runs
    ├── fig1_label_distribution.png
    ├── fig2_eda_categorical.png
    ├── fig3_feature_distributions.png
    ├── fig4_correlation_heatmap.png
    ├── fig4b_scree_plot.png
    ├── fig4c_boxplots_by_category.png
    ├── fig5_confusion_matrices.png
    ├── fig6_roc_comparison.png
    ├── fig7_precision_recall.png
    ├── fig7b_threshold_optimization.png
    ├── fig8_feature_importance.png
    ├── fig_shap_beeswarm.png
    ├── fig_shap_bar.png
    ├── fig_shap_waterfall.png
    ├── fig9_pca_visualization.png
    └── fig10_multiclass_cm.png
```

---

## 🧰 Tech Stack

<div align="center">

| Category | Library | Version |
|---|---|---|
| **Data** | pandas, numpy | ≥1.5 |
| **Visualisation** | matplotlib, seaborn | ≥3.6 |
| **ML — Core** | scikit-learn | ≥1.2 |
| **ML — Boosting** | XGBoost | ≥1.7 |
| **Imbalanced** | imbalanced-learn | ≥0.10 |
| **HPO** | Optuna | ≥3.0 |
| **Explainability** | SHAP | ≥0.42 |
| **Dimensionality** | scikit-learn PCA | built-in |

</div>

---

## 📐 Design Decisions

### Why OHE over LabelEncoder?
`service` has 70+ categories with no natural ordering. LabelEncoder assigns integers 0–69, implying a distance relationship (`ftp=0`, `http=10`, `smtp=30`) that is semantically meaningless and introduces bias in distance-based and linear models, and even subtle bias in tree-based models through split choice.

### Why %pip instead of !pip?
`%pip` is a Jupyter magic that routes to the **kernel's pip**, ensuring packages install into the exact Python environment the notebook kernel is using. `!pip` runs a shell subprocess that may target a different Python installation.

### Why is contamination derived from y_train only?
In real-world deployment, you don't know the attack ratio at inference time. Using `y_test.mean()` (or even the full dataset's ratio) to set `contamination` would be **data leakage** — the model implicitly uses test-set information during its setup. Using `y_train.mean()` simulates a realistic production scenario.

### Why Stratified KFold?
Plain KFold ignores class distribution. With U2R at 0.1% of data, a random fold could contain zero U2R samples, causing training/evaluation imbalance. `StratifiedKFold` preserves the class ratio in every fold.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**CS361 Data Science Lab — SVNIT Surat | Jan–June 2026**

---

<div align="center">

⭐ **If this project was useful, please consider starring the repository!** ⭐

*Built with ❤️ for network security and research-grade ML*

</div>
