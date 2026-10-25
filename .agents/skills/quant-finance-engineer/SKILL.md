---
name: quant-finance-engineer
description: Quantitative financial modeling, analytical Black-Scholes options Greeks, Markowitz and Black-Litterman portfolio optimization, vectorbt backtesting, GARCH volatility, and VaR/CVaR risk analytics. Use when developing quantitative strategies, evaluating portfolio risk, or backtesting returns.
allowed-tools: [Bash, Read, Edit]
compatibility: Python 3.10+, numpy, scipy, pandas, vectorbt
---
# Quantitative Finance & Derivatives Engine

## Non-Negotiable Quantitative Invariants
1. Zero Look-Ahead Bias: Signals at timestamp t must strictly rely on market information <= t-1.
2. Friction Modeling: Backtests must include slippage and exchange fees; never report zero-cost returns.
3. Greeks Sign Conventions: Long calls/puts have positive Vega (V > 0); long options approaching expiration experience negative Theta decay (Theta < 0).

## Core Execution Implementations

### A. Analytical Black-Scholes-Merton & Greeks

```python
import numpy as np
from scipy.stats import norm

def bsm_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1, cdf_d1, cdf_d2 = norm.pdf(d1), norm.cdf(d1), norm.cdf(d2)
    if option_type.lower() == "call":
        price = S * cdf_d1 - K * np.exp(-r * T) * cdf_d2
        delta = cdf_d1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2)
        rho = K * T * np.exp(-r * T) * cdf_d2
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = cdf_d1 - 1.0
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * pdf_d1
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100.0, "theta": theta / 365.0, "rho": rho / 100.0}
```

### B. Risk Metrics (Cornish-Fisher VaR & Expected Shortfall / CVaR)

```python
def compute_risk_metrics(returns: np.ndarray, alpha: float = 0.05):
    hist_var = -np.percentile(returns, alpha * 100)
    hist_cvar = -returns[returns <= -hist_var].mean()
    mu, sigma = np.mean(returns), np.std(returns)
    z = norm.ppf(1 - alpha)
    s = float(np.mean(((returns - mu) / sigma)**3))
    k = float(np.mean(((returns - mu) / sigma)**4) - 3)
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * (s**2) / 36
    cf_var = -(mu - z_cf * sigma)
    return {"Hist_VaR": hist_var, "Hist_CVaR": hist_cvar, "Cornish_Fisher_VaR": cf_var}
```
