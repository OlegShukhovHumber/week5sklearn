import pytest

from latedeliveries.evaluation import (
    choose_threshold,
    cost_by_threshold,
    order_cost,
    policy_costs,
    score_metrics,
)


def test_order_cost_small_example():
    y_true = [1, 0, 0, 1, 0]
    flagged = [True, True, False, False, False]
    # two vouchers (2 x 4) and one missed late order (1 x 15)
    assert order_cost(y_true, flagged) == 2 * 4 + 1 * 15


def test_order_cost_on_time_flagged_orders_still_cost_a_voucher():
    assert order_cost([0, 0, 0], [True, True, True]) == 3 * 4


def test_threshold_flags_nobody_when_that_is_cheapest():
    # nothing is ever late, so every voucher is wasted
    y_true = [0, 0, 0, 0]
    proba = [0.5, 0.5, 0.5, 0.5]
    threshold = choose_threshold(y_true, proba)
    assert order_cost(y_true, [p >= threshold for p in proba]) == 0
    assert threshold == 0.55


def test_threshold_flags_everybody_when_that_is_cheapest():
    # everything is late, so flagging all costs 4 each instead of 15 each
    y_true = [1, 1, 1, 1]
    proba = [0.9, 0.9, 0.9, 0.9]
    threshold = choose_threshold(y_true, proba)
    assert threshold == 0.05
    assert order_cost(y_true, [p >= threshold for p in proba]) == 4 * 4


def test_threshold_on_hand_worked_example(small_example):
    y_true, proba = small_example
    table = cost_by_threshold(y_true, proba).set_index("threshold")["cost"]
    assert table[0.05] == 24
    assert table[0.15] == 16
    assert table[0.25] == 12
    assert table[0.35] == 8
    assert table[0.65] == 19
    assert table[0.95] == 30
    assert choose_threshold(y_true, proba) == 0.35


def test_policy_costs_on_hand_worked_example(small_example):
    y_true, proba = small_example
    assert policy_costs(y_true, proba, 0.35) == {
        "flag_nobody": 2 * 15,
        "flag_everybody": 6 * 4,
        "model": 8,
    }


def test_score_metrics_on_hand_worked_example(small_example):
    y_true, proba = small_example
    metrics = score_metrics(y_true, proba, 0.35)
    assert metrics["confusion_matrix"] == {"tn": 4, "fp": 0, "fn": 0, "tp": 2}
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
