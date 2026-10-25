"""Feature engineering and walk-forward split helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


ASSET_FEATURES = ["f_lag1", "f_mom12", "f_vol12", "f_volrank", "f_maclen"]


def build_features(returns: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the signal features used by the model.

    `returns` is a monthly log-return panel (dates x assets).
    Features (per asset, all computed strictly with lagged information):
      f_lag1     last month's log return
      f_mom12    12-month momentum
      f_vol12    12-month rolling volatility of log returns
      f_volrank  cross-sectional percentile rank of f_vol12 (0..1)
      f_maclen   short-minus-long momentum spread (2-mo minus 6-mo mean
                 log return, the monthly analogue of a 20/60-day macd-lag
                 style signal)
    The label `y` is the next month's log return, the same period that a
    portfolio position held from date t earns, so features and label line up
    with the backtest timing.

    If `macro` (dates x series) is given, each series is lagged one period
    and broadcast to every asset as a `m_<SERIES>` column, giving the model
    a time-varying macro layer.

    Returns a DataFrame indexed by date with a two-level column index
    (feature, asset), so that `.stack()` in `make_training_panel` collapses
    the asset level into the row index.
    """
    feat = {
        "f_lag1": returns.shift(1),
        "f_mom12": np.log(returns / returns.shift(12)),
        "f_vol12": returns.rolling(12).std(),
        "f_maclen": returns.rolling(2).mean() - returns.rolling(6).mean(),
        "y": returns.shift(-1),
    }
    feat["f_volrank"] = feat["f_vol12"].rank(axis=1, pct=True)

    if macro is not None and not macro.empty:
        m = macro.reindex(returns.index, method="ffill")
        for col in m.columns:
            # shift(1): the macro value known at month t-1 is usable at t (no look-ahead)
            shifted = m[col].shift(1)
            broadcasted = pd.DataFrame(
                {a: shifted for a in returns.columns}, index=returns.index
            )
            feat[f"m_{col}"] = broadcasted

    # MultiIndex columns: (feature, asset)
    out = pd.concat(feat, axis=1, keys=feat.keys(), names=["feature", "asset"])
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def feature_columns(macro: pd.DataFrame | None = None) -> list[str]:
    """The scalar feature/label columns `make_training_panel` must emit."""
    cols = ASSET_FEATURES + ["y"]
    if macro is not None and not macro.empty:
        cols += [f"m_{c}" for c in macro.columns]
    return cols


def make_training_panel(returns: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Flatten features + label into a (date, asset) training panel.

    `build_features` returns a wide frame with MultiIndex columns
    (feature, asset).  We melt each feature block separately and concat
    along the row axis, so the result is indexed by (date, asset) with one
    scalar column per feature/label.
    """
    feat = build_features(returns, macro)
    cols = feature_columns(macro)
    parts = []
    for c in cols:
        block = feat[c]  # dates x assets
        long_block = (
            block.stack()
            .to_frame(c)
            .reset_index()
            .rename(columns={"level_0": "date", "level_1": "asset"})
        )
        parts.append(long_block)
    panel = parts[0]
    for p in parts[1:]:
        panel = panel.merge(p, on=["date", "asset"])
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna()
    panel = panel.set_index(["date", "asset"])
    return panel


def walk_forward_splits(
    data: pd.DataFrame,
    target_years: list[int],
    train_years: int = 4,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Walk-forward splits over a (date, asset) panel.

    For each target year:
      train on the preceding `train_years` years
      hold out the target year (no overlap, no peeking)
    """
    splits = []
    dates = data.index.get_level_values(0)
    for year in target_years:
        train_mask = (dates.year < year) & (dates.year >= year - train_years)
        test_mask = dates.year == year
        train, test = data[train_mask], data[test_mask]
        if not train.empty and not test.empty:
            splits.append((train, test))
    return splits
