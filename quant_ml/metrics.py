"""Performance metrics for portfolio returns."""

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12.0  # monthly


def annualised_return(pnl: pd.Series) -> float:
    """Compound monthly PnL into an annualised return."""
    equity = (1.0 + pnl).cumprod()
    n_periods = len(pnl)
    return float(equity.iloc[-1] ** (PERIODS_PER_YEAR / n_periods) - 1.0)


def volatility(pnl: pd.Series) -> float:
    """Annualised standard deviation of monthly PnL."""
    return float(pnl.std() * np.sqrt(PERIODS_PER_YEAR))


def sharpe(pnl: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio (risk-free passed as an annual rate)."""
    monthly_rf = risk_free / PERIODS_PER_YEAR
    excess = pnl - monthly_rf
    vol = excess.std() * np.sqrt(PERIODS_PER_YEAR)
    if vol == 0:
        return 0.0
    return float((excess.mean() * PERIODS_PER_YEAR) / vol)


def max_drawdown(pnl: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
    equity = (1.0 + pnl).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(-dd.min())


def win_rate(pnl: pd.Series) -> float:
    """Fraction of months with positive PnL."""
    return float((pnl > 0).mean())


def benchmark_relative(pnl: pd.Series, bench: pd.Series) -> dict:
    """Information ratio, tracking error, and beta vs the benchmark."""
    aligned = pd.concat({"p": pnl, "b": bench}, axis=1).dropna()
    active = aligned["p"] - aligned["b"]
    te = active.std() * np.sqrt(PERIODS_PER_YEAR)
    ir = float((active.mean() * PERIODS_PER_YEAR) / te) if te > 0 else 0.0
    cov = np.cov(aligned["p"], aligned["b"])
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else np.nan
    return {
        "ir": ir,
        "tracking_error": float(te),
        "beta": beta,
        "active_annualised": float((active.mean()) * PERIODS_PER_YEAR),
    }
