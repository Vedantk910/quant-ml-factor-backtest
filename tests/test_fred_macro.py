import numpy as np
import pandas as pd

from quant_ml.data import _parse_fred_csv
from quant_ml.backtest import allocate, _pick_topn


def test_parse_fred_csv_observation_date_layout():
    text = "observation_date,FEDFUNDS\n2024-01-01,4.90\n2024-02-01,4.90\n"
    out = _parse_fred_csv(text, "FEDFUNDS")
    assert list(out) == [4.90, 4.90]
    assert out.name == "FEDFUNDS"
    assert out.index.is_monotonic_increasing


def test_parse_fred_csv_legacy_date_layout():
    text = "DATE,UNRATE\n2024-01-01,3.4\nnotadate,.\n"
    out = _parse_fred_csv(text, "UNRATE")
    assert len(out) == 1  # bad date row dropped
    assert out.iloc[0] == 3.4


def test_pick_topn_respects_sector_cap():
    ranked = ["AAPL", "MSFT", "GOOGL", "NVDA", "AVGO", "JPM", "GS"]
    sectors = {a: "tech" for a in ["AAPL", "MSFT", "GOOGL", "NVDA", "AVGO"]}
    sectors["JPM"] = sectors["GS"] = "fin"
    picked = _pick_topn(ranked, sectors, sector_neutral=True, max_per_sector=2)
    # two tech, then two fin; slots that would exceed the cap fall through
    assert picked == ["AAPL", "MSFT", "JPM", "GS"], picked


def test_pick_topn_non_neutral_ignores_cap():
    ranked = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "GS"]
    sectors = {a: "tech" for a in ["AAPL", "MSFT", "GOOGL", "NVDA"]}
    sectors["JPM"] = sectors["GS"] = "fin"
    picked = _pick_topn(ranked, sectors, sector_neutral=False, max_per_sector=2)
    assert picked == ranked


def test_allocate_sector_neutral_caps_basket():
    assets = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "GS"]
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-31", periods=1, freq="ME"), assets],
        names=["date", "asset"],
    )
    preds = pd.DataFrame({"pred_return": np.arange(len(assets), dtype=float)}, index=idx)
    sectors = {"AAPL": "tech", "MSFT": "tech", "GOOGL": "tech",
               "NVDA": "tech", "JPM": "fin", "GS": "fin"}
    w = allocate(preds, top_n=4, sector_neutral=True, max_per_sector=2, sectors=sectors)
    held = w.columns[w.iloc[0] > 0].tolist()
    tech_in = [a for a in held if a in ("AAPL", "MSFT", "GOOGL", "NVDA")]
    assert len(tech_in) <= 2, held
    assert len(held) == 4
    assert np.isclose(w.iloc[0].sum(), 1.0)
