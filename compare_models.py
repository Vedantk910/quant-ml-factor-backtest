"""P1 Day 2: model + allocation comparison on yfinance + FRED macro data."""
import json
import argparse
import pandas as pd

from quant_ml.data import load_yfinance_panel, fred_macro_panel
from quant_ml.features import make_training_panel
from quant_ml.strategy import walk_forward_predict
from quant_ml.backtest import allocate
from quant_ml.metrics import (annualised_return, volatility, sharpe,
                             max_drawdown, win_rate, benchmark_relative)


def run(cfg: dict, source: str, model: str, sector_neutral: bool) -> dict:
    if source == "yfinance":
        panel = load_yfinance_panel(cfg["data"]["tickers"], cfg["data"]["start"], cfg["data"].get("end"))
        macro = fred_macro_panel(cfg["data"].get("macro", {}).get("series_ids", []),
                                 cfg["data"]["start"], cfg["data"].get("end"))
        if macro is not None and macro.empty:
            macro = None
    else:
        from quant_ml.data import synthetic_panel
        panel = synthetic_panel(n_years=9, n_assets=24, start_year=2017, seed=cfg.get("seed", 42))
        macro = None
    years = cfg.get("years", [2023, 2024])
    data = make_training_panel(panel, macro)
    preds = walk_forward_predict(data, years, model, seed=cfg.get("seed", 42))
    w = allocate(preds, top_n=cfg["strategy"]["top_n"],
                 sector_neutral=sector_neutral,
                 max_per_sector=cfg["strategy"].get("max_per_sector", 3))
    returns = panel
    out = simulate(w, returns,
                   every_n_months=cfg["strategy"]["rebalance_every_n_months"],
                   cost_bps=cfg["strategy"]["cost_bps"])
    bench = returns[returns.index.isin(w.index)].mean(axis=1)
    row = {"model": model, "sector_neutral": sector_neutral,
           "annualised_return": annualised_return(out["pnl"]),
           "volatility": volatility(out["pnl"]),
           "sharpe": sharpe(out["pnl"]),
           "max_drawdown": max_drawdown(out["pnl"]),
           "win_rate": win_rate(out["pnl"])}
    rel = benchmark_relative(out["pnl"], bench)
    row.update(rel)
    return row


from quant_ml.backtest import simulate

if __name__ == "__main__":
    import json
    from pathlib import Path
    from main import HERE
    cfg = json.loads((Path(__file__).parent / "config" / "default.json").read_text())
    source = "yfinance"
    rows = []
    for model in ["rf", "gbm", "lr"]:
        for sn in [False, True]:
            print(f"running {model} sector_neutral={sn} ...")
            rows.append(run(cfg, source, model, sn))
    comp = pd.DataFrame(rows)
    comp.to_csv(Path(__file__).parent / "out" / "compare_models.csv", index=False)
    print(comp.to_string(index=False))
