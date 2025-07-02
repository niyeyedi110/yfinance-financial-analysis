"""
Quick Start Guide for yfinance Financial Analysis Toolkit
=========================================================

This script demonstrates the main features of each module.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Import our modules
from utils.data_fetcher import DataFetcher
from portfolio_optimization.modern_portfolio_theory import optimize_portfolio
from portfolio_optimization.efficient_frontier import plot_efficient_frontier
from risk_analysis.value_at_risk import calculate_var
from technical_analysis.indicators import add_all_indicators
from technical_analysis.trading_signals import TradingSignals
from quantitative_strategies.momentum_strategy import MomentumStrategy
from options_pricing.black_scholes import BlackScholes

def main():
    print("=== yfinance Financial Analysis Toolkit - Quick Start ===\n")
    
    # 1. Data Fetching
    print("1. Fetching Data...")
    fetcher = DataFetcher()
    
    # Define our portfolio
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Fetch data for multiple stocks
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    stock_data = fetcher.fetch_multiple_stocks(
        tickers, 
        start=start_date, 
        end=end_date
    )
    
    print(f"✓ Data fetched for {len(tickers)} stocks\n")
    
    # 2. Portfolio Optimization
    print("2. Portfolio Optimization...")
    
    # Calculate returns
    returns = stock_data.pct_change().dropna()
    
    # Optimize portfolio
    result = optimize_portfolio(returns, method='max_sharpe')
    
    print("Optimal Portfolio Weights (Max Sharpe):")
    for ticker, weight in zip(tickers, result['weights']):
        print(f"  {ticker}: {weight:.2%}")
    print(f"Expected Return: {result['expected_return']:.2%}")
    print(f"Expected Volatility: {result['expected_volatility']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}\n")
    
    # 3. Risk Analysis
    print("3. Risk Analysis...")
    
    # Calculate portfolio returns
    portfolio_returns = (returns * result['weights']).sum(axis=1)
    
    # Calculate VaR
    var_95 = calculate_var(portfolio_returns, confidence=0.95, method='historical')
    print(f"95% VaR (1-day): {var_95:.2%}")
    
    # 4. Technical Analysis
    print("\n4. Technical Analysis (AAPL)...")
    
    # Get single stock data
    aapl_data = fetcher.fetch_stock_data('AAPL', period='3mo')
    
    # Add technical indicators
    aapl_with_indicators = add_all_indicators(aapl_data)
    
    # Generate trading signals
    signals = TradingSignals(aapl_data)
    ma_signals = signals.moving_average_signals()
    
    buy_signals = ma_signals[ma_signals['signal'] == 1]
    sell_signals = ma_signals[ma_signals['signal'] == -1]
    
    print(f"Buy signals generated: {len(buy_signals)}")
    print(f"Sell signals generated: {len(sell_signals)}\n")
    
    # 5. Quantitative Strategy
    print("5. Running Momentum Strategy...")
    
    # Run momentum strategy
    momentum = MomentumStrategy(
        lookback_period=20,
        holding_period=5,
        n_stocks=3
    )
    
    backtest_result = momentum.backtest(
        tickers[:3],  # Use fewer stocks for demo
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"Strategy Return: {backtest_result['total_return']:.2%}")
    print(f"Strategy Sharpe: {backtest_result['sharpe_ratio']:.2f}\n")
    
    # 6. Options Pricing
    print("6. Options Pricing (AAPL)...")
    
    bs = BlackScholes()
    
    # Current AAPL price
    current_price = aapl_data['Close'][-1]
    
    # Price a call option
    call_price = bs.call_price(
        S=current_price,
        K=current_price * 1.05,  # 5% OTM
        T=0.25,  # 3 months
        r=0.05,  # Risk-free rate
        sigma=0.25  # Implied volatility
    )
    
    print(f"AAPL Current Price: ${current_price:.2f}")
    print(f"Call Option Price (K=${current_price*1.05:.2f}, 3mo): ${call_price:.2f}")
    
    # Calculate Greeks
    greeks = bs.calculate_greeks(
        S=current_price,
        K=current_price * 1.05,
        T=0.25,
        r=0.05,
        sigma=0.25,
        option_type='call'
    )
    
    print("\nOption Greeks:")
    for greek, value in greeks.items():
        print(f"  {greek}: {value:.4f}")
    
    print("\n=== Quick Start Complete ===")
    print("\nFor more examples, check the examples folder in each module.")
    print("For detailed documentation, see the README.md file.")

if __name__ == "__main__":
    main()