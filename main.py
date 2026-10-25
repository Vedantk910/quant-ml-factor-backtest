"""End-to-end ML factor backtest pipeline.

Usage:
    python main.py                     # default: yfinance panel, rf model
    python main.py --source synthetic  # offline run with synthetic data
    python main.py --model gbm --years 2022 2023 2024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_ml.data import load_yfinance_panel, synthetic_panel, fred_macro_panel
from quant_ml.features import build_features, make_training_panel, walk_forward_splits
from quant_ml.strategy import get_model, walk_forward_predict
from quant_ml.backtest import allocate, simulate
from quant_ml.metrics import annualised_return, volatility, sharpe, max_drawdown, win_rate

HERE = Path(__file__).parent
DEFAULT_CFG = json.loads((HERE / "config" / "default.json").read_text())


def pick_data(cfg: dict, args) -> pd.DataFrame:
    source = args.source or cfg["data"]["source"]
    if source == "yfinance":
        panel = load_yfinance_panel(
            tickers=cfg["data"]["tickers"],
            start=cfg["data"]["start"],
            end=cfg["data"]["end"],
        )
        print(f"Loaded yfinance panel: {panel.shape[0]} months x {panel.shape[1]} assets")
        return panel
    if source == "synthetic":
        panel = synthetic_panel(
            n_years=cfg["data"].get("synthetic_years", 6),
            n_assets=cfg["data"].get("synthetic_assets", 24),
            seed=cfg["data"].get("seed", 123),
        )
        print(f"Generated synthetic panel: {panel.shape[0]} months x {panel.shape[1]} assets")
        return panel
    raise ValueError(f"Unknown data source: {source!r}")


def run(cfg: dict, args) -> None:
    panel = pick_data(cfg, args)

    model_name = args.model or cfg["model"]["name"]
    target_years = args.years or cfg["years"]

    # Optional FRED macro overlay (yfinance source only; synthetic has no live dates)
    macro_cfg = cfg["data"].get("macro", {})
    macro = None
    if macro_cfg.get("enabled") and cfg["data"].get("source") == "yfinance" and args.source != "synthetic":
        print(f"Fetching FRED macro panel: {macro_cfg.get('series_ids')} ...")
        macro = fred_macro_panel(
            macro_cfg.get("series_ids", []),
            cfg["data"]["start"],
            cfg["data"].get("end"),
        )
        if macro.empty:
            print("  FRED returned no data; continuing without macro features")
            macro = None
        else:
            print(f"  macro panel: {macro.shape[0]} months x {macro.shape[1]} series")

    # Build features and training panel
    print("Building features ...")
    data = make_training_panel(panel, macro)

    print(f"Walk-forward predicting years {target_years} with model '{model_name}' ...")
    preds = walk_forward_predict(data, target_years, model_name, seed=cfg.get("seed", 42))

    print("Allocating and simulating backtest ...")
    weights = allocate(
        preds,
        top_n=cfg["strategy"]["top_n"],
        sector_neutral=cfg["strategy"].get("sector_neutral", False),
        max_per_sector=cfg["strategy"].get("max_per_sector", 3),
    )
    returns = panel  # monthly returns panel
    bt = simulate(weights, returns,
                  every_n_months=cfg["strategy"]["rebalance_every_n_months"],
                  cost_bps=cfg["strategy"]["cost_bps"])

    # Benchmark: equal-weight across the whole universe over the same period
    bench = returns[returns.index.isin(weights.index)].mean(axis=1)

    print("\n=== Strategy metrics ===")
    for name, val in [
        ("annualised_return", annualised_return(bt["pnl"])),
        ("volatility", volatility(bt["pnl"])),
        ("sharpe", sharpe(bt["pnl"])),
        ("max_drawdown", max_drawdown(bt["pnl"])),
        ("win_rate", win_rate(bt["pnl"])),
    ]:
        print(f"  {name:20s} {val:.4f}")

    print("\n=== Benchmark-relative ===")
    from quant_ml.metrics import benchmark_relative
    rel = benchmark_relative(bt["pnl"], bench)
    for name, val in rel.items():
        print(f"  {name:20s} {val:.4f}")

    # Save outputs
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    bt.to_csv(out_dir / "backtest_monthly.csv")
    weights.to_csv(out_dir / "weights.csv")
    preds.to_csv(out_dir / "preds.csv")
    print(f"\nSaved outputs to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["yfinance", "synthetic"], default=None)
    ap.add_argument("--model", choices=["rf", "gbm", "lr"], default=None)
    ap.add_argument("--years", type=int, nargs="*", default=None)
    args = ap.parse_args()
    cfg = {**DEFAULT_CFG}
    run(cfg, args)


if __name__ == "__main__":
    main()
