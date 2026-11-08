"""Run the whole project: python -m latedeliveries.train."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

from latedeliveries.data import make_orders
from latedeliveries.evaluation import (
    choose_threshold,
    compare_models,
    cost_by_threshold,
    order_cost,
    policy_costs,
    score_metrics,
)
from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    build_pipeline,
    split_data,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
N_SPLITS = 5


def candidate_models() -> dict:
    """The baseline and the real candidates."""
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(max_iter=1000),
        "logreg_balanced": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }


def param_grid(name: str) -> dict:
    """Search space for the chosen model. Names follow the pipeline step."""
    if name.startswith("logreg"):
        return {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        }
    if name == "random_forest":
        return {
            "model__max_depth": [4, 8, None],
            "model__min_samples_leaf": [1, 5, 10],
        }
    return {
        "model__learning_rate": [0.03, 0.1, 0.2],
        "model__max_depth": [2, 3, None],
    }


def out_of_fold_proba(model, X, y, cv) -> np.ndarray:
    """Predicted probability of 'late' for each training row, from the fold
    in which that row was held out."""
    pipe = build_pipeline(clone(model)) if not hasattr(model, "steps") else model
    return cross_val_predict(clone(pipe), X, y, cv=cv, method="predict_proba")[:, 1]


def add_cost_columns(results: pd.DataFrame, models: dict, X, y, cv) -> pd.DataFrame:
    """Add the cost at each model's own best out-of-fold threshold."""
    thresholds, costs = [], []
    for name in results["model"]:
        proba = out_of_fold_proba(models[name], X, y, cv)
        threshold = choose_threshold(y, proba)
        thresholds.append(threshold)
        costs.append(order_cost(y, proba >= threshold))
    results = results.copy()
    results["cv_best_threshold"] = thresholds
    results["cv_cost"] = costs
    return results


def pick_model(results: pd.DataFrame) -> str:
    """Cheapest real candidate on the training data. Ties go to higher AUC."""
    real = results[results["model"] != "dummy_prior"]
    ordered = real.sort_values(["cv_cost", "roc_auc_mean"], ascending=[True, False])
    return str(ordered.iloc[0]["model"])


def leakage_check(df, train_index, model, cv) -> dict:
    """ROC-AUC on the training data with and without customer_rating."""
    X_train = df.loc[train_index]
    y_train = X_train[TARGET]
    without = build_pipeline(clone(model))
    with_rating = build_pipeline(
        clone(model), numeric=[*NUMERIC_FEATURES, "customer_rating"]
    )
    auc_without = cross_val_score(
        without, X_train[FEATURES], y_train, cv=cv, scoring="roc_auc"
    )
    cols = [*FEATURES, "customer_rating"]
    auc_with = cross_val_score(
        with_rating, X_train[cols], y_train, cv=cv, scoring="roc_auc"
    )
    return {
        "cv_roc_auc_without_customer_rating": float(auc_without.mean()),
        "cv_roc_auc_with_customer_rating": float(auc_with.mean()),
    }


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. split (the test set is not touched until step 6)
    df = make_orders()
    X_train, X_test, y_train, y_test = split_data(df)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # 2. compare the baseline and the candidates, training data only
    models = candidate_models()
    results = compare_models(models, X_train, y_train, cv)
    results = add_cost_columns(results, models, X_train, y_train, cv)
    results.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    chosen = pick_model(results)
    print(results.round(3).to_string(index=False))
    print("chosen model:", chosen)

    # 3. tune the chosen model
    search = GridSearchCV(
        build_pipeline(clone(models[chosen])),
        param_grid(chosen),
        scoring="roc_auc",
        cv=cv,
    )
    search.fit(X_train, y_train)
    row = results.set_index("model").loc[chosen]
    gain = float(search.best_score_ - row["roc_auc_mean"])
    tuning = {
        "model": chosen,
        "scoring": "roc_auc",
        "n_combinations": len(search.cv_results_["params"]),
        "best_params": search.best_params_,
        "untuned_cv_roc_auc_mean": float(row["roc_auc_mean"]),
        "untuned_cv_roc_auc_std": float(row["roc_auc_std"]),
        "tuned_cv_roc_auc_mean": float(search.best_score_),
        "gain": gain,
        "gain_larger_than_fold_spread": bool(gain > row["roc_auc_std"]),
    }
    write_json(REPORTS_DIR / "tuning.json", tuning)
    print("tuning:", tuning)

    # 4. threshold from out-of-fold probabilities on the training data
    oof = out_of_fold_proba(search.best_estimator_, X_train, y_train, cv)
    threshold = choose_threshold(y_train, oof)
    cost_by_threshold(y_train, oof).to_csv(
        REPORTS_DIR / "threshold_costs.csv", index=False
    )
    print("threshold:", threshold)

    # 5. leakage experiment (training data only)
    leak = leakage_check(
        df, X_train.index, search.best_estimator_.named_steps["model"], cv
    )
    write_json(REPORTS_DIR / "leakage_check.json", leak)

    # 6. fit on the whole training set and score the test set once
    final = clone(search.best_estimator_).fit(X_train, y_train)
    proba = final.predict_proba(X_test)[:, 1]
    test_metrics = score_metrics(y_test, proba, threshold)
    metrics = {
        "model": chosen,
        "threshold": threshold,
        "n_test": int(len(y_test)),
        "test_late_rate": float(y_test.mean()),
        "test_metrics": test_metrics,
        "policy_costs": policy_costs(y_test, proba, threshold),
    }
    write_json(REPORTS_DIR / "metrics.json", metrics)
    print("test:", metrics)

    # 7. save the pipeline and its description
    joblib.dump(final, MODELS_DIR / "late_orders.joblib")
    info = {
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "model": chosen,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_params": search.best_params_,
        "test_metrics": test_metrics,
        "policy_costs": metrics["policy_costs"],
    }
    write_json(MODELS_DIR / "model_info.json", info)


if __name__ == "__main__":
    main()
