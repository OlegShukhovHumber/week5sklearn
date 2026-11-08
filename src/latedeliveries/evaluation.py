"""Metrics, the cost rule and the threshold choice."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate

from latedeliveries.pipeline import build_pipeline

VOUCHER_COST = 4  # paid for every flagged order
LATE_COST = 15  # paid for every late order that was not flagged
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def order_cost(y_true, flagged) -> int:
    """Total cost: $4 per flagged order plus $15 per late order not flagged."""
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    vouchers = flagged.sum()
    missed_late = (y_true & ~flagged).sum()
    return int(VOUCHER_COST * vouchers + LATE_COST * missed_late)


def cost_by_threshold(y_true, proba, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Cost of flagging every order whose probability is >= each threshold."""
    proba = np.asarray(proba)
    rows = [
        {"threshold": float(t), "cost": order_cost(y_true, proba >= t)}
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def choose_threshold(y_true, proba, thresholds=THRESHOLDS) -> float:
    """Threshold with the lowest total cost. Ties go to the lower threshold."""
    table = cost_by_threshold(y_true, proba, thresholds)
    return float(table.loc[table["cost"].idxmin(), "threshold"])


def policy_costs(y_true, proba, threshold: float) -> dict[str, int]:
    """Cost of the three policies: flag nobody, flag everybody, use the model."""
    y_true = np.asarray(y_true)
    return {
        "flag_nobody": order_cost(y_true, np.zeros(len(y_true), dtype=bool)),
        "flag_everybody": order_cost(y_true, np.ones(len(y_true), dtype=bool)),
        "model": order_cost(y_true, np.asarray(proba) >= threshold),
    }


def score_metrics(y_true, proba, threshold: float) -> dict:
    """Confusion matrix and metrics at the given threshold."""
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def compare_models(models: dict, X, y, cv) -> pd.DataFrame:
    """Cross-validate each model inside the pipeline. Mean and std per metric."""
    rows = []
    for name, model in models.items():
        result = cross_validate(build_pipeline(model), X, y, cv=cv, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            scores = result[f"test_{metric}"]
            row[f"{metric}_mean"] = float(scores.mean())
            row[f"{metric}_std"] = float(scores.std())
        rows.append(row)
    return pd.DataFrame(rows)
