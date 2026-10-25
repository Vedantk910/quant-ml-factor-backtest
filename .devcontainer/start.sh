#!/bin/bash
set -e

cd /workspaces/quant-ml-factor-backtest || cd /home/vscode/quant-ml-factor-backtest

echo "=== quant-ml-factor-backtest dev container ready ==="
echo ""
echo "Quick start:"
echo "  python main.py --source synthetic   # offline smoke test (no network)"
echo "  python main.py                      # real yfinance data"
echo "  python -m pytest tests/ -q          # test suite"
echo ""
echo "Opening in terminal..."
bash
