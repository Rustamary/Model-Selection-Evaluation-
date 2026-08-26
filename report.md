# Model Selection & Evaluation Report

## 1. Dataset & setup

Input: `data/customer_churn_processed.csv` — 1,000 rows, 8 engineered
features + binary `churn` target (produced by the companion
feature-engineering task).

- **Split:** 80% train (800 rows) / 20% test (200 rows), stratified on
  `churn` so both sets keep the ~40% churn rate.
- **Cross-validation:** 5-fold stratified CV, run on the full dataset for
  each model.

## 2. Models evaluated

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `max_depth=5` |
| Random Forest | `n_estimators=200`, `max_depth=8` |

**Note on "linear regression":** the task brief lists linear regression as
an example algorithm, but `churn` is a binary target, and the required
metrics (accuracy, precision, recall, F1) are classification metrics —
linear regression doesn't produce those directly. **Logistic Regression**
is used instead, as the linear-model equivalent for classification.

## 3. Held-out test set results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.730 | 0.697 | 0.575 | 0.630 |
| Decision Tree | 0.660 | 0.594 | 0.475 | 0.528 |
| Random Forest | 0.710 | 0.672 | 0.538 | 0.597 |

Confusion matrices (rows = actual, columns = predicted; class order
`[no-churn, churn]`):

- **Logistic Regression:** `[[100, 20], [34, 46]]`
- **Decision Tree:** `[[94, 26], [42, 38]]`
- **Random Forest:** `[[99, 21], [37, 43]]`

## 4. 5-fold cross-validation results (mean ± std)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.725 ± 0.023 | 0.684 ± 0.022 | 0.578 ± 0.060 | 0.625 ± 0.043 |
| Decision Tree | 0.664 ± 0.030 | 0.591 ± 0.054 | 0.548 ± 0.074 | 0.564 ± 0.041 |
| Random Forest | 0.725 ± 0.014 | 0.680 ± 0.029 | 0.595 ± 0.038 | **0.633 ± 0.021** |

Full per-fold and per-metric scores are in
`reports/model_comparison_summary.json`; a visual comparison is in
`reports/model_comparison_chart.png`.

## 5. Model selection & justification

**Selected model: Random Forest**

Reasoning:

- **Highest mean cross-validated F1-score** (0.633) — F1 balances
  precision and recall, which matters here since only ~40% of customers
  churn (a moderately imbalanced target where accuracy alone can be
  misleading).
- **Lowest variance across folds** (F1 std of 0.021 vs. 0.043 for Logistic
  Regression and 0.041 for the Decision Tree) — the Random Forest's
  performance is the most consistent and least sensitive to which rows
  land in each fold, a sign it generalizes better rather than overfitting
  to one particular split.
- **Holdout vs. CV agreement:** Random Forest's holdout F1 (0.597) and CV
  F1 (0.633) are close, and its holdout accuracy (0.710) closely tracks
  its CV accuracy (0.725) — no large train/test gap, so it isn't
  overfitting.
- Logistic Regression is a close second and is simpler/more interpretable
  — a reasonable choice if interpretability matters more than the small
  F1 gain, but Random Forest is the more reliable performer overall.
- The Decision Tree underperformed both other models on every metric and
  showed the widest recall variance across folds (± 0.074), consistent
  with a single tree being prone to overfitting/instability — this is
  exactly the kind of issue cross-validation is meant to catch.

## 6. Reproducing these results

```bash
pip install -r requirements.txt
python src/model_selection.py
```

Outputs: `reports/model_comparison_summary.json` (full metrics) and
`reports/model_comparison_chart.png` (comparison chart).
