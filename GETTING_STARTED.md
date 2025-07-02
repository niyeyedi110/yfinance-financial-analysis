# Getting Started with yfinance Financial Analysis Toolkit

## Installation

1. Clone or download this repository:
```bash
cd /mnt/f/PYTHON/yfinance-financial-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the quick start example to see all modules in action:
```bash
python examples/quick_start.py
```

## Module Overview

### 1. Portfolio Optimization (`portfolio_optimization/`)
- **Modern Portfolio Theory**: Mean-variance optimization
- **Efficient Frontier**: Visualize risk-return tradeoffs
- **Risk Parity**: Alternative allocation methods

Example:
```python
from portfolio_optimization.modern_portfolio_theory import optimize_portfolio
from utils.data_fetcher import fetch_stock_data

# Get data
prices = fetch_stock_data(['AAPL', 'MSFT', 'GOOGL'], period='1y')
returns = prices.pct_change().dropna()

# Optimize
result = optimize_portfolio(returns, method='max_sharpe')
print("Optimal weights:", result['weights'])
```

### 2. Risk Analysis (`risk_analysis/`)
- **Value at Risk**: Historical, parametric, and Monte Carlo VaR
- **Volatility Analysis**: GARCH, rolling volatility
- **Risk Metrics**: Comprehensive risk measurement

Example:
```python
from risk_analysis.value_at_risk import calculate_var, calculate_cvar

# Calculate 95% VaR
var_95 = calculate_var(returns, confidence=0.95)
cvar_95 = calculate_cvar(returns, confidence=0.95)
```

### 3. Technical Analysis (`technical_analysis/`)
- **Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Patterns**: Candlestick pattern detection
- **Signals**: Trading signal generation

Example:
```python
from technical_analysis.indicators import add_all_indicators
from technical_analysis.trading_signals import TradingSignals

# Add indicators
data_with_indicators = add_all_indicators(price_data)

# Generate signals
signals = TradingSignals(price_data)
combined = signals.combined_signals()
```

### 4. Market Analysis (`market_analysis/`)
- **Regime Detection**: Identify market states
- **Correlation Analysis**: Sector and asset correlations
- **Sector Rotation**: Track sector performance

Example:
```python
from market_analysis.regime_detection import detect_regimes

regimes = detect_regimes(market_data, method='hmm')
```

### 5. Quantitative Strategies (`quantitative_strategies/`)
- **Momentum**: Trend-following strategies
- **Mean Reversion**: Contrarian strategies
- **Pairs Trading**: Statistical arbitrage

Example:
```python
from quantitative_strategies.momentum_strategy import MomentumStrategy

strategy = MomentumStrategy(lookback_period=20, n_stocks=5)
backtest = strategy.backtest(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
```

### 6. Options Pricing (`options_pricing/`)
- **Black-Scholes**: Analytical option pricing
- **Monte Carlo**: Simulation-based pricing
- **Implied Volatility**: IV calculation and analysis

Example:
```python
from options_pricing.black_scholes import BlackScholes

bs = BlackScholes()
call_price = bs.call_price(S=100, K=105, T=0.25, r=0.05, sigma=0.2)
greeks = bs.calculate_greeks(S=100, K=105, T=0.25, r=0.05, sigma=0.2)
```

### 7. Utilities (`utils/`)
- **Data Fetcher**: Enhanced yfinance wrapper with caching
- **Performance Metrics**: Sharpe, Sortino, Calmar ratios
- **Visualization**: Common plotting functions
- **Backtesting**: Simple strategy testing framework

## Example Notebooks

Each module has Jupyter notebooks in its `examples/` folder:

1. **Portfolio Optimization**: `portfolio_optimization/examples/optimize_portfolio.ipynb`
2. **Risk Analysis**: `risk_analysis/examples/risk_metrics.ipynb`
3. **Technical Analysis**: `technical_analysis/examples/technical_analysis.ipynb`

## Common Use Cases

### 1. Build an Optimal Portfolio
```python
# See portfolio_optimization/examples/optimize_portfolio.ipynb
```

### 2. Analyze Risk
```python
# See risk_analysis/examples/risk_metrics.ipynb
```

### 3. Generate Trading Signals
```python
# See technical_analysis/examples/technical_analysis.ipynb
```

### 4. Backtest a Strategy
```python
from utils.backtesting import Backtester, SimpleMovingAverageCrossover

strategy = SimpleMovingAverageCrossover(fast_period=20, slow_period=50)
backtester = Backtester(initial_capital=10000)
result = backtester.run(data, strategy)
```

## Tips

1. **Data Caching**: Use `DataFetcher` to avoid repeated API calls
2. **Parallel Processing**: Many functions support multiple tickers
3. **Visualization**: Most modules include plotting functions
4. **Customization**: All strategies and models are extensible

## Next Steps

1. Run the quick start example
2. Explore the Jupyter notebooks
3. Read individual module documentation
4. Customize strategies for your needs

## Support

For issues or questions:
1. Check module docstrings
2. Review example notebooks
3. See function documentation

Happy analyzing! 📈