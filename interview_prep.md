# Interview prep — quant ML factor backtest

The questions you will actually get asked on a walk-forward ML project, and how to answer them well.

## 1. "Walk me through the project in 30 seconds."

See the CV line + story in the README. Keep it to: what, how, what it shows. Do not lead with code details.

## 2. "Why walk-forward and not a random train/test split?"

Random splits on time-series data leak the future into the training set. Walk-forward respects the temporal order: the model can only ever be trained on data before the period it predicts. For a strategy you could actually trade, that is the only honest evaluation scheme.

## 3. "How do you make sure there is no look-ahead bias?"

Three things:
- Features are computed strictly with lagged data (`shift(1)`, 12-month momentum uses the past 12 months, not including the current month).
- The label is the forward 1-month return, aligned so it is only known at prediction time.
- Walk-forward splits: train on years before T, predict year T.

## 4. "What is your strategy and why long-only top-N?"

Equal-weight long of the top-N highest-predicted assets, quarterly rebalance. It is deliberately transparent: a recruiter can follow the logic in one sentence. A continuous-weight or risk-model layer is a natural next step, and saying so shows you know where the project stops.

## 5. "How do you handle transaction costs?"

Turnover cost in bps: on each rebalance, cost = |new weight - old weight| summed across assets, times cost_bps. It is deducted from that month's PnL. I chose a quarterly rebalance partly to keep turnover (and cost) manageable.

## 6. "What benchmark do you use and why?"

Equal-weight across the whole universe over the same out-of-sample period. It is the baseline a simple, broad-market strategy would produce; beating it on Sharpe and IR is the first hurdle before claiming alpha.

## 7. "What are the limitations of this project?"

- 25-ticker US panel: too narrow for production, fine for a portfolio piece
- yfinance data: free, not licensed
- No walk-forward hyperparameter search
- No risk model layer — the top-N selection is a hard threshold

Being specific and unambiguous about the limits is what separates a strong candidate from a weak one.

## 8. "What would you do next?"

- Walk-forward grid search over hyperparameters (n_estimators, depth) with the same temporal split discipline
- A continuous-weight layer (e.g. optimise weights proportional to predicted return, capped at max position)
- A broader universe: full S&P 500 or a sector-neutral selection
- Add a volatility-scaled position sizing layer so the strategy is risk-parity across months

## 9. "Why Python and not C++ for a quant project?"

Python is the right tool at this scope: fast to iterate, the ML ecosystem is first-class, and the backtest engine itself is trivially fast on a monthly panel. If this were a production strategy with live data, the order-execution layer would be in C++ or a language with a real-time event loop, but the research layer stays Python. Saying that out loud shows you know where the line is.

## 10. "What did you learn from running it?"

An honest answer: the random forest on this feature set does not reliably beat the equal-weight benchmark on Sharpe. That is a finding, not a failure — it is exactly what walk-forward validation is for. A model that looked great on an in-sample fit can fail completely out-of-sample. That is the core lesson of the whole project, and stating it is a strong interview answer.
