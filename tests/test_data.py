import pandas as pd

from quant_ml.data import synthetic_panel


def test_synthetic_panel_shape():
    panel = synthetic_panel(n_years=4, n_assets=10, start_year=2019, seed=7)
    assert panel.shape == (48, 10)
    assert isinstance(panel.index, pd.DatetimeIndex)
    assert panel.index.is_monotonic_increasing


def test_synthetic_panel_is_deterministic():
    a = synthetic_panel(n_years=3, n_assets=5, seed=42)
    b = synthetic_panel(n_years=3, n_assets=5, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_panel_has_variation():
    panel = synthetic_panel(n_years=3, n_assets=5, seed=1)
    assert panel.values.std() > 0
    assert not panel.isna().any().any()
