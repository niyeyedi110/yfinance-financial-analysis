"""
Quantitative Trading Strategies Module

This module provides implementations of various quantitative trading strategies
with comprehensive backtesting functionality and performance metrics.
"""

from .momentum_strategy import MomentumStrategy, example_momentum_backtest
from .mean_reversion import MeanReversionStrategy, example_mean_reversion_backtest
from .pairs_trading import (
    PairsTradingStrategy, 
    example_pairs_trading_backtest,
    find_cointegrated_pairs
)

__all__ = [
    'MomentumStrategy',
    'MeanReversionStrategy',
    'PairsTradingStrategy',
    'example_momentum_backtest',
    'example_mean_reversion_backtest',
    'example_pairs_trading_backtest',
    'find_cointegrated_pairs'
]

__version__ = '1.0.0'