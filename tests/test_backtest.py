import numpy as np
import pandas as pd

from quant_ml.backtest import allocate, simulate


def make_preds():
    """3 dates x 4 assets; pred_return is 0..11 so asset D (last) is highest."""
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-31", periods=3, freq="ME"), ["A", "B", "C", "D"]],
        names=["date", "asset"],
    )
    return pd.DataFrame({"pred_return": np.arange(12.0)}, index=idx)


def test_allocate_sums_to_one_per_date():
    w = allocate(make_preds(), top_n=2)
    assert set(w.columns) == {"A", "B", "C", "D"}
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.allclose(w.max(axis=1), 0.5)


def test_allocate_selects_highest_pred():
    w = allocate(make_preds(), top_n=1)
    assert (w["D"] == 1.0).all()


def test_simulate_zero_weights_gives_zero_pnl():
    # All-cash portfolio: PnL is zero every period regardless of returns.
    w = allocate(make_preds(), top_n=1)
    zero_w = pd.DataFrame(0.0, index=w.index, columns=w.columns)
    returns = pd.DataFrame(0.01 * np.ones((3, 4)), index=w.index, columns=w.columns)
    bt = simulate(zero_w, returns)
    assert np.allclose(bt["pnl"], 0.0)
    assert np.allclose(bt["equity"], 1.0)


def test_simulate_constant_weights_compounds():
    # 3 dates, all rebalances; weights at date t earn the return at t+1.
    # Period 1 is the initial build (no prior basket -> PnL=0).
    # Periods 2 and 3 each earn 1% in D.  equity = 1.01^2.
    w = allocate(make_preds(), top_n=1)  # always 100% in D
    returns = pd.DataFrame(0.0, index=w.index, columns=w.columns)
    returns["D"] = 0.01
    bt = simulate(w, returns, cost_bps=0.0)
    assert abs(bt["equity"].iloc[-1] - 1.01 ** 2) < 1e-9
