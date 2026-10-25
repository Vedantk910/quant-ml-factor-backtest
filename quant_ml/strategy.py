"""Walk-forward model training and out-of-sample prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_model(name: str, seed: int = 42):
    """Return an unfitted model for the given name.

    Supported: "rf" (RandomForestRegressor), "gbm" (LightGBM), "lr" (LinearRegression).
    All are regressors on forward returns — the strategy ranks by predicted value.
    """
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=20, random_state=seed)
    if name == "gbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            random_state=seed,
            verbose=-1,
        )
    if name == "lr":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    raise ValueError(f"Unknown model: {name!r} (use 'rf', 'gbm' or 'lr')")


def walk_forward_predict(
    data: pd.DataFrame,
    target_years: list[int],
    model_name: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Train on all years before the target year, predict the target year.

    For each year in `target_years`:
      train on [start .. year-1]
      predict  [year .. year]
    Predictions are out-of-sample by construction (no peeking into the
    future), which is what makes the backtest honest.

    Returns a DataFrame indexed by (date, asset) with one column: pred_return.
    """
    all_preds = []
    for year in target_years:
        train = data[data.index.get_level_values(0).year < year]
        test = data[data.index.get_level_values(0).year == year]
        if train.empty or test.empty:
            print(f"  skip year {year}: missing train or test data")
            continue
        X_train = train.drop(columns="y", errors="ignore").values
        y_train = train["y"].values
        X_test = test.drop(columns="y", errors="ignore").values

        model = get_model(model_name, seed)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rows = []
        for (date, asset), p in zip(test.index, pred, strict=False):
            rows.append({"date": date, "asset": asset, "pred_return": p})
        block = pd.DataFrame(rows).set_index(["date", "asset"])
        all_preds.append(block)
        print(f"  trained {model_name} on {len(train):,} rows -> predicted {len(test):,} rows for {year}")

    if not all_preds:
        raise ValueError("No predictions produced — check target_years and data range.")
    out = pd.concat(all_preds)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()
