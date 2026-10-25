"""Data loading: real data via yfinance, or a synthetic panel for offline runs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_panel(n_years: int = 6, n_assets: int = 24,
                    start_year: int = 2019, seed: int = 123) -> pd.DataFrame:
    """Monthly-close returns panel with a few cross-sectional factors built in.

    Generates a panel of monthly returns with realistic correlation structure
    and idiosyncratic variance. Used in tests and for a smoke run when
    yfinance is unavailable (e.g. offline Codespaces dev).
    """
    rng = np.random.default_rng(seed)
    # Monthly close dates
    dates = pd.period_range(
        start=f"{start_year}-01-31",
        end=f"{start_year + n_years - 1}-12-31",
        freq="M",
    ).to_timestamp()
    assets = [f"A{i:02d}" for i in range(n_assets)]

    # Market factor: common shock to all assets
    market = rng.normal(0.0, 0.08, len(dates))
    # Per-asset alphas (slowly varying)
    alpha = rng.normal(0.0, 0.001, (n_assets, len(dates)))
    alpha = np.cumsum(alpha, axis=1) * 0.05 + rng.normal(0.0, 0.0005, (n_assets, len(dates)))
    # Betas
    betas = rng.uniform(0.6, 1.4, n_assets)
    # Idiosyncratic
    idio = rng.normal(0.0, 0.04, (len(dates), n_assets))

    rets = market[None, :] * betas[:, None] + alpha + idio.T
    panel = pd.DataFrame(rets.T, index=dates, columns=assets)
    return panel


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """Parse a FRED CSV into a date-indexed Series (native frequency kept).

    Handles both the keyless fredgraph.csv endpoint (header
    ``observation_date,<ID>``) and the legacy ``DATE,<ID>`` layout.  The
    native frequency of the series (monthly, quarterly, ...) is preserved —
    callers resample as needed.  Kept as a pure function so it can be
    unit-tested offline.
    """
    import io
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [c.strip().lower() for c in raw.columns]
    date_col = "observation_date" if "observation_date" in raw.columns else raw.columns[0]
    val_col = raw.columns[1] if len(raw.columns) > 1 else series_id
    idx = pd.to_datetime(raw[date_col], errors="coerce")
    # .to_numpy() matters: pd.Series(series, index=new) would align by label
    # and turn everything into NaN
    vals = pd.to_numeric(raw[val_col], errors="coerce").to_numpy()
    out = pd.Series(vals, index=idx).rename(series_id)
    out = out[~out.index.isna() & ~out.isna()]
    return out.sort_index()


def fred_series(series_id: str, start: str, end: str | None = None) -> pd.Series:
    """Load a FRED series (daily) via the keyless CSV endpoint."""
    import urllib.request

    url = FRED_CSV_URL.format(series_id=series_id)
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    out = _parse_fred_csv(text, series_id)
    mask = out.index >= pd.Timestamp(start)
    if end:
        mask &= out.index <= pd.Timestamp(end)
    return out[mask]


def fred_macro_panel(
    series_ids: list[str], start: str, end: str | None = None
) -> pd.DataFrame:
    """Fetch FRED series, resample to month-end, one column per series.

    Used for time-varying macro features.  A series that fails to download
    is skipped (printed), never fatal.
    """
    cols = {}
    for s in series_ids:
        try:
            d = fred_series(s, start, end)
            cols[s] = d.resample("ME").last()
        except Exception as e:
            print(f"  FRED series {s} unavailable ({type(e).__name__}); skipping")
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).ffill()


def load_yfinance_panel(
    tickers: list[str], start: str, end: str | None = None
) -> pd.DataFrame:
    """Download monthly-close prices -> log returns for the given tickers."""
    import yfinance as yf  # optional dependency

    idx = pd.date_range(start=start, end=end or None, freq="MS")
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  yfinance returned no data for {t}; skipping")
            continue
        # yfinance may return multi-column MultiIndex; flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        data[t] = close
    px = pd.DataFrame(data)
    px = px.reindex(idx).ffill()
    px = px.dropna()
    rets = np.log(px / px.shift(1)).dropna()
    # guard: a zero/negative close artifact can yield -inf/NaN; drop it
    rets = rets.replace([np.inf, -np.inf], np.nan)
    return rets
