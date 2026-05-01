import pandas as pd
import pytest

from portfolio.rebalance import compute_weights, generate_rebalance_orders


def test_compute_weights_respects_top_n_and_cap():
    scores = pd.Series(
        {
            "XLB": 0.9,
            "XLE": 0.8,
            "XLF": 0.7,
            "XLI": 0.1,
            "XLK": -0.2,
        }
    )

    weights = compute_weights(scores, max_weight=0.4, top_n=3)

    assert weights.sum() == pytest.approx(1.0)
    assert weights.max() <= 0.4 + 1e-12
    assert (weights > 0).sum() == 3
    assert weights["XLK"] == 0


def test_compute_weights_rejects_impossible_cap():
    scores = pd.Series({"A": 1.0, "B": 0.5, "C": 0.1})

    with pytest.raises(ValueError, match="max_weight is too low"):
        compute_weights(scores, max_weight=0.2)


def test_generate_rebalance_orders_aligns_missing_tickers():
    current = pd.Series({"A": 0.5, "B": 0.5})
    target = pd.Series({"B": 0.25, "C": 0.75})

    orders = generate_rebalance_orders(current, target, min_trade_size=0.01)

    assert set(orders["ticker"]) == {"A", "B", "C"}
    assert orders.set_index("ticker").loc["A", "direction"] == "SELL"
    assert orders.set_index("ticker").loc["C", "direction"] == "BUY"
