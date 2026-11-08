import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.data import make_orders
from latedeliveries.pipeline import build_pipeline, split_data


@pytest.fixture(scope="session")
def orders():
    return make_orders()


@pytest.fixture(scope="session")
def split(orders):
    return split_data(orders)


@pytest.fixture(scope="session")
def fitted_pipeline(split):
    X_train, _, y_train, _ = split
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)


@pytest.fixture
def small_example():
    """Six orders, worked out on paper.

    Late orders have probabilities 0.9 and 0.6; the four on-time orders have
    0.3, 0.2, 0.1 and 0.1. Flagging at threshold t costs:
      t <= 0.10 : 6 flagged                 = 24
      t <= 0.20 : 4 flagged                 = 16
      t <= 0.30 : 3 flagged                 = 12
      t <= 0.60 : 2 flagged                 = 8   <- cheapest
      t <= 0.90 : 1 flagged + 1 missed late = 4 + 15 = 19
      t >  0.90 : 0 flagged + 2 missed late = 30
    On the 0.05 grid the lowest cheapest threshold is 0.35.
    """
    y_true = [1, 1, 0, 0, 0, 0]
    proba = [0.9, 0.6, 0.3, 0.2, 0.1, 0.1]
    return y_true, proba
