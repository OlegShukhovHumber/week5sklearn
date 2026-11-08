import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_pipeline,
    split_data,
)


def test_feature_list_excludes_leaky_columns():
    for column in ["customer_rating", "order_id", TARGET]:
        assert column not in FEATURES


def test_features_exist_in_data(orders):
    assert set(FEATURES) <= set(orders.columns)
    assert FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_split_is_stratified(orders, split):
    _, _, y_train, y_test = split
    assert len(y_test) == pytest.approx(0.25 * len(orders), abs=1)
    assert y_train.mean() == pytest.approx(orders[TARGET].mean(), abs=0.002)
    assert y_test.mean() == pytest.approx(orders[TARGET].mean(), abs=0.005)


def test_split_is_repeatable(orders, split):
    again = split_data(orders)
    assert split[0].index.equals(again[0].index)
    assert split[1].index.equals(again[1].index)


def _one_order(**changes):
    row = {
        "distance_km": 4.0,
        "items": 3,
        "hour": 19,
        "prep_minutes": 14.0,
        "courier_months": 12,
        "weather": "clear",
        "day_type": "weekday",
        "vehicle": "bike",
    }
    row.update(changes)
    return pd.DataFrame([row])[FEATURES]


def test_pipeline_handles_missing_prep_minutes(fitted_pipeline):
    proba = fitted_pipeline.predict_proba(_one_order(prep_minutes=np.nan))
    assert np.isfinite(proba).all()


def test_pipeline_handles_unseen_weather(fitted_pipeline):
    proba = fitted_pipeline.predict_proba(_one_order(weather="hail"))
    assert np.isfinite(proba).all()


@pytest.mark.parametrize(
    "model",
    [
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(n_estimators=20, random_state=0),
    ],
)
def test_predicted_probabilities_are_valid(split, model):
    X_train, X_test, y_train, _ = split
    proba = build_pipeline(model).fit(X_train, y_train).predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1)
