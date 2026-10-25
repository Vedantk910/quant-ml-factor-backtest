"""Cross-sectional long-only backtest with periodic rebalancing.

Timing convention
-----------------
The model predicts a one-month forward return for every asset at each
rebalance date ``t``. The weights chosen at ``t`` earn the return realised
over the period ``(t, t+1]``. The simulation below walks forward one month
at a time:

  * on a rebalance date, the new basket is selected from that date's
    predictions, turnover is charged, and the new basket earns the *next*
    period's return;
  * between rebalances the basket drifts with the market and keeps earning
    the next period's return.

So a prediction made at ``t`` is never scored against a return that was
already known at ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_SECTOR = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "AMZN": "tech",
    "NVDA": "tech", "META": "tech", "TSLA": "consumer", "AVGO": "tech",
    "JPM": "fin", "GS": "fin", "BAC": "fin", "WFC": "fin",
    "BLK": "fin", "SCHW": "fin", "MS": "fin", "AXP": "fin",
    "JNJ": "health", "PFE": "health", "MRK": "health",
    "ABBV": "health", "LLY": "health", "UNH": "health",
    "CAT": "ind", "GE": "ind",
}


def _pick_topn(
    ranked_assets: list,
    sector_of: dict,
    sector_neutral: bool,
    max_per_sector: int,
) -> list:
    """Take assets in rank order; with sector_neutral, cap each sector."""
    if not sector_neutral:
        return ranked_assets
    counts: dict = {}
    picked = []
    for a in ranked_assets:
        s = sector_of.get(a, "other")
        if counts.get(s, 0) >= max_per_sector:
            continue
        picked.append(a)
        counts[s] = counts.get(s, 0) + 1
    return picked


def allocate(
    preds: pd.DataFrame,
    top_n: int = 10,
    sector_neutral: bool = False,
    max_per_sector: int = 3,
    sectors: dict | None = None,
) -> pd.DataFrame:
    """For each date, allocate equal weight across the top-N predicted assets.

    `preds` has one row per (date, asset) with a `pred_return` column.
    Returns a wide DataFrame: rows = dates, columns = assets, values = weight
    (0 if not held, 1/top_n if held). Unheld assets are cash and earn 0.

    With `sector_neutral=True`, the top-N is filled in rank order but capped
    at `max_per_sector` per sector (`sectors` maps asset -> sector, defaulting
    to DEFAULT_SECTOR). Slots that would exceed the cap fall through to the
    next-ranked asset, so the basket stays top-N-weighted overall.
    """
    sector_map = {**DEFAULT_SECTOR, **(sectors or {})}
    rows = {}
    all_assets = preds.index.get_level_values(1).unique()
    for date, grp in preds.groupby(level="date"):
        ranked = grp["pred_return"].sort_values(ascending=False)
        ranked_assets = ranked.index.get_level_values(1).tolist()
        chosen = _pick_topn(ranked_assets, sector_map, sector_neutral, max_per_sector)[:top_n]
        w = pd.Series(0.0, index=all_assets)
        if chosen:
            w.loc[chosen] = 1.0 / top_n
        rows[date] = w
    return pd.DataFrame(rows).T.sort_index().fillna(0.0)


def _rebalance_dates(dates: pd.DatetimeIndex, every_n_months: int) -> list:
    """The last observed date in each calendar month, keeping every Nth one.

    Uses the actual data dates (month-end for a monthly panel) rather than
    synthetic month-start timestamps, so rebalance dates always line up
    with rows that exist in the returns panel.
    """
    month_last: dict = {}
    for d in dates:
        month_last[d.to_period("M")] = d
    ends = sorted(month_last.values())
    step = max(1, int(every_n_months))
    return [d for i, d in enumerate(ends) if i % step == 0]


def simulate(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    every_n_months: int = 1,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Simulate the long-only strategy and return a monthly PnL/equity table.

    Parameters
    ----------
    weights : wide DataFrame of portfolio weights by date (from `allocate`).
        A row's weights take effect from that date onward.
    returns : wide DataFrame of monthly log returns by date and asset.
        The value at date ``t`` is the return realised over ``(t-1, t]``.
    every_n_months : rebalance frequency in months.
    cost_bps : transaction cost in basis points per unit of turnover.

    Returns
    -------
    DataFrame indexed by date with columns:
      pnl       net monthly return of the portfolio
      equity    compounded equity curve
      turnover  weight change on the date (0 on non-rebalance dates)
      invested  fraction of capital deployed
    """
    returns = returns.sort_index()
    dates = returns.index

    # Only simulate from the first date we actually have a signal for.
    if weights.empty:
        raise ValueError("No weights supplied.")
    start = weights.index[0]
    dates = dates[dates >= start]
    if len(dates) < 2:
        raise ValueError("Not enough return periods after the first signal date.")

    w_all = weights.reindex(columns=returns.columns).fillna(0.0)
    rebal = set(_rebalance_dates(dates, every_n_months))

    out = []
    equity = 1.0
    current_w = None  # basket in force, set at the most recent completed rebalance

    for i, date in enumerate(dates):
        # --- PnL for period (t-1, t] --------------------------------------
        # The return for period (t-1, t] was earned by the basket that was in
        # force *before* date t, i.e. the weights set at the most recent
        # rebalance strictly before t.  A decision made at the close of t
        # only takes effect from period (t, t+1] onward.
        if current_w is not None:
            port_ret = float((current_w * returns.loc[date]).sum())
            invested = float(current_w.sum())
        else:
            # Before the first rebalance we hold cash.
            port_ret = 0.0
            invested = 0.0

        # --- rebalance at the close of t (trade cost charged in period t) ---
        turnover = 0.0
        is_rebal = date in rebal
        if is_rebal and date in w_all.index:
            new_w = w_all.loc[date]
            if current_w is None:
                turnover = float(new_w.abs().sum())
            else:
                turnover = float((new_w - current_w).abs().sum())
            current_w = new_w.copy()  # takes effect from period (t, t+1]

        cost = turnover * cost_bps / 10_000.0
        net_ret = port_ret - cost
        equity *= 1.0 + net_ret

        out.append({
            "date": date,
            "pnl": net_ret,
            "equity": equity,
            "turnover": turnover,
            "invested": invested,
        })

    df = pd.DataFrame(out).set_index("date")
    df.index.name = "date"
    return df
