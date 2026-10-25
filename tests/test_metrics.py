import numpy as np
import pandas as pd

from quant_ml.metrics import annualised_return, sharpe, max_drawdown, win_rate, volatility


def test_annualised_return_level_pnl():
    pnl = pd.Series([0.01] * 12)
    # 1.01**12 - 1 total -> annualised is just that compounded over 1 year
    assert abs(annualised_return(pnl) - (1.01 ** 12 - 1)) < 1e-9


def test_win_rate():
    pnl = pd.Series([0.01, -0.02, 0.03, 0.0])
    assert win_rate(pnl) == 0.5


def test_max_drawdown_positive():
    pnl = pd.Series([0.10, -0.30, 0.05])
    # equity: 1.10, 0.77, 0.8085 -> dd = 0.77/1.10 - 1
    dd = max_drawdown(pnl)
    assert abs(dd - (1.0 - 0.77 / 1.10)) < 1e-9


def test_volatility_positive():
    pnl = pd.Series(np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.01] * 2))
    assert volatility(pnl) > 0
    assert sharpe(pnl) == sharpe(pnl)  # deterministic
