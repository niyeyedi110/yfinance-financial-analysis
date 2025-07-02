"""
Example script demonstrating all quantitative trading strategies.

This script runs backtests for momentum, mean reversion, and pairs trading strategies
and compares their performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from quantitative_strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    PairsTradingStrategy,
    find_cointegrated_pairs
)
import pandas as pd
import matplotlib.pyplot as plt


def compare_strategies():
    """
    Run and compare all three trading strategies.
    """
    print("=" * 80)
    print("QUANTITATIVE TRADING STRATEGIES COMPARISON")
    print("=" * 80)
    
    results = {}
    
    # 1. Momentum Strategy
    print("\n1. MOMENTUM STRATEGY")
    print("-" * 40)
    
    # Tech stocks for momentum
    momentum_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'CRM']
    
    momentum_strategy = MomentumStrategy(
        symbols=momentum_symbols,
        lookback_period=252,  # 1 year
        holding_period=21,    # 1 month
        top_n=3,             # Top 3 performers
        short_n=0,           # Long-only
        rebalance_frequency='monthly'
    )
    
    print("Fetching data for momentum strategy...")
    momentum_strategy.fetch_data('2020-01-01', '2024-01-01')
    
    print("Running momentum backtest...")
    momentum_results = momentum_strategy.backtest(initial_capital=100000)
    results['Momentum'] = momentum_results['metrics']
    
    print("\nMomentum Strategy Results:")
    for metric, value in momentum_results['metrics'].items():
        if isinstance(value, float):
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    # 2. Mean Reversion Strategy
    print("\n\n2. MEAN REVERSION STRATEGY")
    print("-" * 40)
    
    mean_reversion_strategy = MeanReversionStrategy(
        symbol='SPY',
        lookback_period=20,
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        method='zscore'
    )
    
    print("Fetching data for mean reversion strategy...")
    mean_reversion_strategy.fetch_data('2020-01-01', '2024-01-01')
    
    print("Running mean reversion backtest...")
    mean_reversion_results = mean_reversion_strategy.backtest(initial_capital=100000)
    results['Mean Reversion'] = mean_reversion_results['metrics']
    
    print("\nMean Reversion Strategy Results:")
    for metric, value in mean_reversion_results['metrics'].items():
        if isinstance(value, float):
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    # 3. Pairs Trading Strategy
    print("\n\n3. PAIRS TRADING STRATEGY")
    print("-" * 40)
    
    # First, find cointegrated pairs
    print("Searching for cointegrated pairs...")
    pairs_symbols = ['XOM', 'CVX', 'PEP', 'KO', 'WMT', 'TGT', 'HD', 'LOW']
    cointegrated_pairs = find_cointegrated_pairs(pairs_symbols, '2020-01-01', '2024-01-01')
    
    if len(cointegrated_pairs) > 0:
        # Use the best cointegrated pair
        best_pair = cointegrated_pairs[0]
        print(f"\nUsing best cointegrated pair: {best_pair[0]}-{best_pair[1]} (p-value: {best_pair[2]:.4f})")
        
        pairs_strategy = PairsTradingStrategy(
            symbol1=best_pair[0],
            symbol2=best_pair[1],
            lookback_period=60,
            entry_zscore=2.0,
            exit_zscore=0.5,
            stop_loss_zscore=3.0,
            hedge_ratio_lookback=20
        )
        
        print("Fetching data for pairs trading strategy...")
        pairs_strategy.fetch_data('2020-01-01', '2024-01-01')
        
        print("Running pairs trading backtest...")
        pairs_results = pairs_strategy.backtest(initial_capital=100000)
        results['Pairs Trading'] = pairs_results['metrics']
        
        print("\nPairs Trading Strategy Results:")
        for metric, value in pairs_results['metrics'].items():
            if isinstance(value, float):
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    else:
        print("No cointegrated pairs found!")
        
    # Create comparison visualization
    create_comparison_chart(results)
    
    return {
        'momentum': (momentum_strategy, momentum_results),
        'mean_reversion': (mean_reversion_strategy, mean_reversion_results),
        'pairs': (pairs_strategy, pairs_results) if len(cointegrated_pairs) > 0 else None
    }


def create_comparison_chart(results: dict):
    """
    Create a comparison chart of strategy performance metrics.
    """
    # Select key metrics for comparison
    metrics_to_compare = [
        'total_return',
        'annualized_return',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown',
        'calmar_ratio'
    ]
    
    # Prepare data for plotting
    strategies = list(results.keys())
    metric_values = {metric: [] for metric in metrics_to_compare}
    
    for strategy in strategies:
        for metric in metrics_to_compare:
            value = results[strategy].get(metric, 0)
            metric_values[metric].append(value)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_compare):
        ax = axes[i]
        
        # Create bar chart
        bars = ax.bar(strategies, metric_values[metric])
        
        # Color bars based on positive/negative values
        for j, bar in enumerate(bars):
            if metric_values[metric][j] < 0:
                bar.set_color('red')
            else:
                bar.set_color('green')
                
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{metric_values[metric][j]:.3f}',
                   ha='center', va='bottom' if height >= 0 else 'top')
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Value')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
    plt.suptitle('Quantitative Trading Strategies Performance Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()


def create_equity_curves_comparison(strategies_data: dict):
    """
    Create a comparison of equity curves for all strategies.
    """
    plt.figure(figsize=(12, 8))
    
    # Normalize all portfolios to start at 100
    for name, (strategy, results) in strategies_data.items():
        if results is not None:
            portfolio = results['portfolio']
            normalized_value = (portfolio['total'] / portfolio['total'].iloc[0]) * 100
            plt.plot(portfolio.index, normalized_value, label=name.replace('_', ' ').title(), linewidth=2)
    
    plt.title('Normalized Equity Curves Comparison', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (Base = 100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_strategy_correlations(strategies_data: dict):
    """
    Analyze correlations between strategy returns.
    """
    returns_data = {}
    
    for name, (strategy, results) in strategies_data.items():
        if results is not None:
            returns = results['portfolio']['returns'].dropna()
            returns_data[name] = returns
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(returns_data)
    
    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()
    
    print("\n\nSTRATEGY RETURNS CORRELATION MATRIX")
    print("-" * 40)
    print(correlation_matrix.round(3))
    
    # Create correlation heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    
    # Add labels
    strategies = list(correlation_matrix.columns)
    plt.xticks(range(len(strategies)), strategies, rotation=45)
    plt.yticks(range(len(strategies)), strategies)
    
    # Add correlation values
    for i in range(len(strategies)):
        for j in range(len(strategies)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center', color='black' if abs(correlation_matrix.iloc[i, j]) < 0.5 else 'white')
    
    plt.title('Strategy Returns Correlation Matrix')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run strategy comparison
    strategies_data = compare_strategies()
    
    # Create equity curves comparison
    print("\n\nCreating equity curves comparison...")
    create_equity_curves_comparison(strategies_data)
    
    # Analyze correlations
    analyze_strategy_correlations(strategies_data)
    
    print("\n\nStrategy comparison complete!")
    print("\nKey Insights:")
    print("- Momentum strategies work well in trending markets")
    print("- Mean reversion strategies excel in range-bound markets")
    print("- Pairs trading provides market-neutral returns with lower volatility")
    print("- Diversifying across strategies can improve risk-adjusted returns")