"""Feature lists, the train/test split and the model pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.25
TARGET = "late"

# Columns known at the moment the order is placed.
# Left out: order_id (only a label), customer_rating (given after delivery,
# so it leaks the outcome) and late (the target).
NUMERIC_FEATURES = [
    "distance_km",
    "items",
    "hour",
    "prep_minutes",
    "courier_months",
]
CATEGORICAL_FEATURES = ["weather", "day_type", "vehicle"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 75/25 split. Returns X_train, X_test, y_train, y_test."""
    return train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )


def build_pipeline(
    model,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> Pipeline:
    """Preprocessing followed by any scikit-learn model.

    numeric and categorical default to the feature lists above. They are
    only overridden for the leakage experiment.
    """
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric_steps, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])
