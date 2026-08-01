import pandas as pd
import numpy as np
import pytest

from backtest.engine import CASH_ASSET
from portfolio.rebalance import (
    build_rebalance_ticket,
    compute_weights,
    generate_rebalance_orders,
    reconcile_current_weights,
    validate_portfolio_weights,
)


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


def test_selection_change_moves_removed_weight_to_cash_and_new_etf_starts_zero():
    current = pd.Series({"SPY": 0.4, "IEF": 0.3, "GLD": 0.2, CASH_ASSET: 0.1})

    removed = reconcile_current_weights(
        ["SPY", "IEF", "GLD"], ["SPY", "IEF"], current
    )
    assert removed.to_dict() == pytest.approx(
        {"SPY": 0.4, "IEF": 0.3, CASH_ASSET: 0.3}
    )
    readded = reconcile_current_weights(
        ["SPY", "IEF"], ["SPY", "IEF", "GLD"], removed
    )
    assert readded["GLD"] == 0.0
    assert readded[CASH_ASSET] == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"SPY": 1.0}, "explicit cash"),
        ({"SPY": 0.8, CASH_ASSET: 0.1}, "sum to 100%"),
        ({"SPY": -0.1, CASH_ASSET: 1.1}, "between 0%"),
        ({"SPY": np.nan, CASH_ASSET: np.nan}, "finite"),
        ({"SPY": 0.5, "IEF": 0.0, CASH_ASSET: 0.5}, "exactly"),
    ],
)
def test_current_weight_validation_rejects_invalid_explicit_cash_inputs(
    weights, message
):
    with pytest.raises(ValueError, match=message):
        validate_portfolio_weights(weights, ["SPY"])


def test_ticket_reconciles_percentage_dollars_turnover_cost_and_separate_cash():
    current = pd.Series({"SPY": 0.2, "IEF": 0.3, CASH_ASSET: 0.5})
    target = pd.Series({"SPY": 0.5, "IEF": 0.4, CASH_ASSET: 0.1})

    ticket = build_rebalance_ticket(
        current,
        target,
        ["SPY", "IEF"],
        transaction_cost_bps=10,
        portfolio_value=100_000,
    )

    assert CASH_ASSET not in set(ticket.security_orders["asset"])
    assert ticket.cash_balance["asset"].tolist() == [CASH_ASSET]
    assert ticket.turnover == pytest.approx(0.4)
    assert ticket.estimated_cost_rate == pytest.approx(0.0004)
    assert ticket.estimated_cost_amount == pytest.approx(40.0)
    assert (
        ticket.security_orders["trade_amount"].sum()
        + ticket.cash_balance["trade_amount"].sum()
    ) == pytest.approx(0.0)
    assert ticket.download_frame().query("row_type == 'security_order'")["asset"].tolist() == [
        "SPY",
        "IEF",
    ]


def test_ticket_rejects_invalid_current_even_when_target_is_valid():
    target = pd.Series({"SPY": 0.8, CASH_ASSET: 0.2})
    with pytest.raises(ValueError, match="sum to 100%"):
        build_rebalance_ticket(
            {"SPY": 0.7, CASH_ASSET: 0.2}, target, ["SPY"]
        )
