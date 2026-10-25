# quant-ml-factor-backtest

Walk-forward machine-learning factor backtest on a cross-section of stocks. The model is retrained every year only on data from before that year, and predictions are scored against out-of-sample returns. No future data leaks into training.

## What it does

1. **Data** — monthly log returns for a ticker panel (yfinance, or a synthetic offline panel)
2. **Features** — lagged returns, 12-month momentum, rolling volatility, cross-sectional volatility rank
3. **Model** — Random Forest, LightGBM, or Logistic Regression, trained walk-forward (train on years before T, predict year T)
4. **Strategy** — long an equal-weight basket of the top-N predicted assets, rebalance monthly (or quarterly), with a turnover cost in basis points
5. **Metrics** — annualised return, Sharpe, max drawdown, win rate, IR/tracking error vs an equal-weight benchmark

The result is a backtest you can defend in an interview: honest walk-forward splits, cost-aware rebalancing, and a clear story about what the features signal.

## Project layout

```
quant_ml/
  data.py       # yfinance / synthetic data panels
  features.py   # feature engineering + walk-forward splits
  strategy.py   # model factory + walk_forward_predict
  backtest.py   # allocation + simulation
  metrics.py    # performance statistics
main.py         # end-to-end CLI pipeline
config/default.json
tests/          # pytest suite (offline, synthetic data)
.devcontainer/  # GitHub Codespaces / dev container setup
.github/workflows/ci.yml
```

## Quick start (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py --source synthetic   # offline smoke run, no network
python main.py                      # real data via yfinance
python -m pytest tests/ -q
```

## GitHub Codespaces (zero-install cloud)

1. Push this folder to a GitHub repo
2. Open the repo → **Code** → **Codespaces** → **Create codespace on main**
3. The devcontainer installs all dependencies automatically
4. Run: `python main.py --source synthetic` then `python main.py`

No local Python required; everything runs in the cloud.

## Data notes

- **yfinance** is free and rate-limited; the default 25-ticker monthly panel is fine on the free tier.
- `--source synthetic` generates a seeded synthetic panel so the pipeline and test suite run fully offline — useful for CI and for understanding the code before touching real data.

## Results (yfinance 2017-2024 + FRED macro overlay, walk-forward 2023/2024)

Out-of-sample, quarterly rebalance, 10 bps turnover cost, top-8 long vs equal-weight:

| Model | Sector-neutral | Sharpe | IR | Max DD |
|---|---|---|---|---|
| lr | no | **1.335** | 0.086 | **0.060** |
| gbm | yes | 1.274 | **0.563** | 0.121 |
| rf | no | 1.272 | 0.503 | 0.100 |
| rf | yes | 1.155 | 0.145 | 0.124 |
| gbm | no | 1.143 | 0.454 | 0.150 |
| lr | yes | 1.165 | -0.437 | 0.089 |

Two honest findings:
- The **linear baseline posts the highest Sharpe (1.34)** — on a 24-ticker
  monthly panel the non-linear models do not beat it out-of-sample. That is
  exactly the overfitting story walk-forward validation exists to expose.
- **Sector-neutrality lifts GBM's IR from 0.45 to 0.56** — the constraint
  concentrates exposure and improves the risk-adjusted active return.

Reproduce with `python compare_models.py` (writes `out/compare_models.csv`).

## What a recruiter sees

- **CV line** — "Walk-forward ML equity factor backtest: RF/LightGBM/Linear on 5 engineered features plus a FRED macro overlay (Fed funds, VIX, unemployment, yield curve, dollar index); long top-8 with optional sector-neutral cap, quarterly rebalance, 10 bps costs. Out-of-sample Sharpe 1.34 (linear), IR 0.56 (GBM, sector-neutral); the honest finding is that non-linear models do not beat the linear baseline on a 24-ticker panel."
- **Interview story (30 s)** — "I built a walk-forward ML backtest on a 25-stock US panel. I engineered four features — lagged return, 12-month momentum, rolling volatility, and a cross-sectional volatility rank — then trained a Random Forest and LightGBM model each year only on prior data and scored it on that year's out-of-sample returns. The strategy goes long the top eight predicted assets, rebalances quarterly with a 10 bps cost per unit of turnover, and I report Sharpe, max drawdown, and information ratio versus an equal-weight benchmark. The key design choice — walk-forward splits — is what keeps the results honest."

## Limitations to be honest about

- yfinance monthly data is a proxy, not a licensed feed; fine for a portfolio piece, not production.
- Top-N long-only strategy is deliberately simple and transparent; a real quant would add a continuous-weight layer, a risk model, and a walk-forward hyperparameter search.
- The 25-ticker US panel is a demo universe; a serious quant piece would use a broader, more representative universe (e.g. MSCI USA sectors or a full S&P 500 subset).
