"""
model_selection.py

Model Selection & Evaluation pipeline for the customer churn dataset.

This script:
    1. Loads the processed, feature-selected dataset
       (output of the feature-engineering task)
    2. Splits into train/test sets
    3. Trains three candidate models:
         - Logistic Regression   (linear baseline for classification)
         - Decision Tree
         - Random Forest
    4. Evaluates each with accuracy, precision, recall, F1-score
    5. Runs k-fold cross-validation on each to check for overfitting
    6. Picks and justifies a "best" model
    7. Saves a JSON report + a comparison chart

Run:
    python src/model_selection.py

Note on "linear regression": the task lists linear regression as an example
algorithm, but the target (`churn`) is binary/classification, and the
required metrics (accuracy, precision, recall, F1) are classification
metrics. Logistic Regression is used as the linear model for this problem —
it is the linear-model equivalent for classification tasks.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "customer_churn_processed.csv"
REPORT_JSON_PATH = Path(__file__).resolve().parents[1] / "reports" / "model_comparison_summary.json"
CHART_PATH = Path(__file__).resolve().parents[1] / "reports" / "model_comparison_chart.png"

TARGET = "churn"
RANDOM_STATE = 42
CV_FOLDS = 5

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
}


def load_data(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Loaded processed data: {X.shape[0]} rows, {X.shape[1]} features")
    return X, y


def evaluate_holdout(model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def evaluate_cross_val(model, X, y, folds: int) -> dict:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1"]
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    return {
        f"cv_{metric}_mean": round(float(np.mean(scores[f"test_{metric}"])), 4)
        for metric in scoring
    } | {
        f"cv_{metric}_std": round(float(np.std(scores[f"test_{metric}"])), 4)
        for metric in scoring
    }


def plot_comparison(results: dict, path: Path):
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    models = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model_name in enumerate(models):
        values = [results[model_name]["holdout"][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=model_name)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on held-out test set")
    ax.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    print(f"Saved comparison chart -> {path}")


def main():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    results = {}
    for name, model in MODELS.items():
        print(f"\n--- {name} ---")
        holdout = evaluate_holdout(model, X_train, X_test, y_train, y_test)
        cv = evaluate_cross_val(model, X, y, CV_FOLDS)
        results[name] = {"holdout": holdout, "cross_validation": cv}
        print(f"Holdout: acc={holdout['accuracy']} prec={holdout['precision']} "
              f"rec={holdout['recall']} f1={holdout['f1_score']}")
        print(f"{CV_FOLDS}-fold CV f1: {cv['cv_f1_mean']} (+/- {cv['cv_f1_std']})")

    best_model = max(results.items(), key=lambda kv: kv[1]["cross_validation"]["cv_f1_mean"])
    best_name = best_model[0]
    print(f"\nBest model by mean CV F1-score: {best_name}")

    summary = {
        "cv_folds": CV_FOLDS,
        "test_size": 0.2,
        "results": results,
        "best_model": best_name,
        "selection_criterion": "highest mean cross-validated F1-score "
                                "(balances precision/recall, robust to the "
                                "class imbalance in churn vs. no-churn)",
    }

    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved model comparison summary -> {REPORT_JSON_PATH}")

    plot_comparison(results, CHART_PATH)


if __name__ == "__main__":
    main()
