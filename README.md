# Model Selection & Evaluation — Customer Churn

Internship task: train and compare multiple machine learning models,
evaluate them with standard classification metrics, use cross-validation to
guard against overfitting, and justify the final model choice.

This repo builds directly on the output of the **feature-engineering task**
(`data/customer_churn_processed.csv` — already cleaned, encoded, scaled,
and reduced to the top 8 predictive features).

## Project structure

```
.
├── data/
│   └── customer_churn_processed.csv       # input: feature-engineered dataset
├── notebooks/
│   └── model_selection.ipynb              # exploratory walkthrough with charts
├── reports/
│   ├── report.md                          # write-up comparing model performance
│   ├── model_comparison_summary.json      # raw metrics for every model
│   └── model_comparison_chart.png         # bar chart: accuracy/precision/recall/F1
├── src/
│   └── model_selection.py                 # production pipeline script
├── requirements.txt
└── .gitignore
```

## Models compared

| Model | Type | Why included |
|---|---|---|
| Logistic Regression | linear | fast, interpretable baseline (linear-model equivalent of linear regression, for a binary classification target) |
| Decision Tree | non-linear, single tree | captures non-linear splits, easy to interpret, prone to overfitting |
| Random Forest | non-linear, ensemble | usually more robust/accurate than a single tree |

> The task brief mentions "linear regression" as an example algorithm, but
> the target (`churn`) is binary and the required metrics (accuracy,
> precision, recall, F1) are classification metrics — so **Logistic
> Regression** is used as the linear model. See `reports/report.md` for
> details.

## Evaluation approach

1. **80/20 stratified train/test split** — preserves the churn rate in
   both sets.
2. **Holdout metrics**: accuracy, precision, recall, F1-score, and a
   confusion matrix for each model.
3. **5-fold stratified cross-validation**: mean + standard deviation of
   each metric across folds, to check the holdout result isn't a lucky
   split and to catch overfitting (large train/test gaps).
4. **Model selection**: the model with the highest **mean cross-validated
   F1-score** is chosen as the best model — F1 balances precision and
   recall, which matters here since churners are the minority class.

## Usage

```bash
pip install -r requirements.txt
python src/model_selection.py
```

This trains all three models, prints metrics to the console, and writes:
- `reports/model_comparison_summary.json` (full metrics + best model)
- `reports/model_comparison_chart.png` (comparison bar chart)

To explore interactively with confusion matrices and charts, open
`notebooks/model_selection.ipynb`.

## Results

See [`reports/report.md`](reports/report.md) for the full write-up. In
short: **Random Forest** was selected — it had the highest mean
cross-validated F1-score (0.633) and the lowest variance across folds,
making it the most reliable performer even though Logistic Regression
scored similarly on raw accuracy.

## Requirements

- Python 3.9+
- pandas, numpy, scikit-learn, matplotlib (see `requirements.txt`)

## Related

This repo assumes `data/customer_churn_processed.csv` was produced by the
companion **feature-engineering-task** repo. To regenerate it from raw
data, see that repo's `src/feature_engineering.py`.
