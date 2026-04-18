
# ═══════════════════════════════════════════════════════════════════
# MASTERPROMPT: Network Intrusion Detection — Elite Level Upgrade
# Target file: network_intrusion_detection.ipynb
# Dataset: NSL-KDD (KDDTrain+.txt)
# ═══════════════════════════════════════════════════════════════════

You are an expert data scientist and ML engineer. Rewrite the attached Jupyter notebook `network_intrusion_detection.ipynb` to elite/research-grade quality. Keep all existing correct logic — do NOT start from scratch. Apply every upgrade listed below, in order, without skipping any.

## CONTEXT
- Dataset: NSL-KDD (125,973 rows, 41 features + label)
- Task: Binary classification (normal vs attack) + multi-class (DoS/Probe/R2L/U2R) + anomaly detection
- Current issues: wrong label encoding, dead imports, hardcoded contamination, no CV, no SHAP, no threshold optimization

## UPGRADE 1 — Fix Categorical Encoding
PROBLEM: Current code uses LabelEncoder on `service` (70+ values) and `flag` (11 values), implying false ordinal relationship.

FIX:
- Use pd.get_dummies() with drop_first=True for `protocol_type`, `service`, `flag`
- After get_dummies, update feature_cols list to include all OHE columns
- Print shape before and after: "Shape before OHE: (X, Y) → after: (X, Z)"
- Add a comment: "# OHE avoids spurious ordinal assumptions on nominal features"

## UPGRADE 2 — Fix LabelEncoder Loop
PROBLEM: `le` is overwritten each iteration → cannot inverse_transform later.

FIX:
- Create a dict: encoders = {}
- Store each encoder: encoders[col] = LabelEncoder().fit(df[col])
- Then transform: df_model[col] = encoders[col].transform(df_model[col])
- Note: This applies only to any remaining ordinal encoding (if kept); primary encoding is OHE per Upgrade 1.

## UPGRADE 3 — Remove Dead Imports, Add SHAP + Optuna
PROBLEM: cross_val_score, StratifiedKFold, MinMaxScaler, LogisticRegression imported but never used.

FIX:
- Remove all dead imports
- Add: import shap
- Add: import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
- Add pip install cell at top: !pip install shap optuna -q

## UPGRADE 4 — Stratified K-Fold Cross Validation
Add a new section "4b. Cross-Validation" AFTER the train-test split and BEFORE model training.

CODE TO ADD:
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("5-Fold Stratified CV — Random Forest:")
rf_cv = RandomForestClassifier(n_estimators=100, max_depth=20,
        min_samples_split=5, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1)
cv_scores = cross_val_score(rf_cv, X, y, cv=skf, scoring='f1', n_jobs=-1)
print(f"  F1 per fold: {cv_scores.round(4)}")
print(f"  Mean F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```
Print mean ± std for both RF and XGBoost. Add comment explaining why CV > single split.

## UPGRADE 5 — Hyperparameter Tuning with Optuna
Add new section "4c. Hyperparameter Tuning (Optuna)" after CV.

- Run Optuna for XGBoost with n_trials=30 (fast but meaningful)
- Optimize for F1 score on a 10% validation split from training data
- Search space:
  - n_estimators: [100, 400]
  - max_depth: [4, 10]
  - learning_rate: [0.01, 0.3] (log)
  - subsample: [0.6, 1.0]
  - colsample_bytree: [0.6, 1.0]
  - min_child_weight: [1, 10]
- Print best params and best F1
- Use best params to re-instantiate xgb_model before training in Section 5

## UPGRADE 6 — Fix Isolation Forest Contamination
PROBLEM: contamination=0.46 leaks test knowledge (you shouldn't know the attack ratio at inference).

FIX:
- Compute contamination ONLY from training labels:
  contamination = y_train.mean()
- Pass contamination=round(contamination, 3) to IsolationForest
- Print: "Contamination derived from training data: {contamination:.3f}"
- Add comment: "# Never derive contamination from test set — data leakage"

## UPGRADE 7 — SMOTE for Minority Classes (U2R, R2L)
Add new section "3b. Class Imbalance Handling" after preprocessing.

CODE:
```python
from imblearn.over_sampling import SMOTE
from collections import Counter

print("Before SMOTE:", Counter(y_train))
smote = SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("After SMOTE :", Counter(y_train_sm))
```
- Train RF and XGB on X_train_sm, y_train_sm (rename variables clearly)
- Compare F1 on test set before vs after SMOTE in a printed table
- Add comment: "# SMOTE synthesizes minority samples — critical for U2R/R2L which are <1% of data"

## UPGRADE 8 — Threshold Optimization
Add new section "7b. Threshold Optimization" after confusion matrices.

RATIONALE (add as markdown cell): In security, missing an attack (False Negative) is far more costly than a false alarm (False Positive). Therefore we optimize threshold for maximum Recall without catastrophic Precision loss, using Youden's J statistic and F1-maximization.

CODE:
```python
from sklearn.metrics import roc_curve, f1_score
import numpy as np

fpr, tpr, thresholds = roc_curve(y_test, y_prob_xgb)
youden_j = tpr - fpr
best_thresh_youden = thresholds[np.argmax(youden_j)]

f1_scores = [f1_score(y_test, (y_prob_xgb >= t).astype(int)) for t in thresholds]
best_thresh_f1 = thresholds[np.argmax(f1_scores)]

print(f"Default threshold (0.5) F1   : {f1_score(y_test, y_pred_xgb):.4f}")
print(f"Youden's J optimal threshold : {best_thresh_youden:.3f}")
print(f"F1-max optimal threshold     : {best_thresh_f1:.3f}")

y_pred_xgb_opt = (y_prob_xgb >= best_thresh_f1).astype(int)
print(f"Optimized threshold F1       : {f1_score(y_test, y_pred_xgb_opt):.4f}")
print(classification_report(y_test, y_pred_xgb_opt, target_names=['Normal', 'Attack']))
```

Plot threshold vs F1 curve. Mark both optimal thresholds on the plot.

## UPGRADE 9 — SHAP Analysis
Add new section "8b. SHAP Explainability" after feature importance.

CODE:
```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_sample = X_test.iloc[:500]  # sample for speed
shap_values = explainer.shap_values(shap_sample)

# Plot 1: Beeswarm (global importance + direction)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, shap_sample, plot_type='dot',
                  max_display=20, show=False)
plt.title('SHAP Beeswarm — XGBoost Feature Impact', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_shap_beeswarm.png', bbox_inches='tight')
plt.show()

# Plot 2: Bar (mean |SHAP| importance)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, shap_sample, plot_type='bar',
                  max_display=15, show=False)
plt.title('SHAP Bar — Mean Absolute Feature Importance', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_shap_bar.png', bbox_inches='tight')
plt.show()

# Plot 3: Waterfall for a single attack prediction
attack_idx = np.where(y_test.values == 1)[0][0]
shap.waterfall_plot(shap.Explanation(
    values=shap_values[attack_idx],
    base_values=explainer.expected_value,
    data=shap_sample.iloc[0],
    feature_names=feature_cols
), show=False)
plt.title('SHAP Waterfall — Single Attack Prediction', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_shap_waterfall.png', bbox_inches='tight')
plt.show()
```

Add markdown cell interpreting top 3 SHAP features: what they mean in network security context.

## UPGRADE 10 — Enhanced EDA
Add to Section 2 (EDA):

a) Scree plot (PCA explained variance):
- Fit PCA on X_train_scaled with n_components=20
- Plot cumulative explained variance vs number of components
- Mark the elbow point and the 95% variance threshold
- Title: "Scree Plot — Cumulative Explained Variance"
- Add note: "First N components explain 95% of variance"

b) Boxplots by attack category:
- For features: src_bytes, dst_bytes, serror_rate, rerror_rate, count
- One subplot per feature, colored by attack_category
- Use log scale for byte features
- Title: "Feature Distribution by Attack Category"

## UPGRADE 11 — Professional Markdown Documentation
For EVERY section, ensure there is a markdown cell at the top containing:
- What the section does (1 sentence)
- Why it matters for intrusion detection (1-2 sentences)
- Any key design decisions with justification

At the notebook top, add a professional abstract cell:
```
## Abstract
This notebook implements a research-grade Network Intrusion Detection System (NIDS) 
on the NSL-KDD benchmark dataset. We apply supervised classification (Random Forest, 
XGBoost with Optuna tuning) and unsupervised anomaly detection (Isolation Forest) 
to detect malicious network traffic. Key contributions: OHE for categorical features, 
SMOTE for minority class oversampling, SHAP-based explainability, threshold 
optimization for security-aware recall maximization, and 5-fold stratified 
cross-validation for robust evaluation.
```

## UPGRADE 12 — Final Summary Table (Enhanced)
Replace existing summary table with:
- Add columns: CV_F1_Mean, CV_F1_Std, Threshold_Used, Training_Samples
- Include a "Notes" column with 1-line interpretation per model
- Sort by F1-Score descending
- Print: "Recommended for production: {best_model} — Reason: {reason}"

## OUTPUT REQUIREMENTS
- All cells must be runnable top-to-bottom with zero errors
- All figures saved as PNG with descriptive names
- No dead imports anywhere
- Every new technique has a markdown explanation cell
- Code comments on every non-obvious line
- No hardcoded test-set statistics anywhere in model setup
- Final cell prints a complete list of all saved figures
- Total target: ~15 sections, ~12-15 figures

## DO NOT CHANGE
- Dataset URL and column names
- Binary label creation logic (normal=0, attack=1)
- Attack category mapping (DoS/Probe/R2L/U2R)
- Figure saving pattern (fig*.png)
- Overall pipeline order
  
