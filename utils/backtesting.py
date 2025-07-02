"""
Simple backtesting framework for trading strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    trade_type: str  # 'long' or 'short'
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    
    def close_trade(self, exit_date: pd.Timestamp, exit_price: float):
        """Close the trade and calculate P&L."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        
        if self.trade_type == 'long':
            self.pnl = (self.exit_price - self.entry_price) * self.position_size
            self.return_pct = (self.exit_price - self.entry_price) / self.entry_price
        else:  # short
            self.pnl = (self.entry_price - self.exit_price) * self.position_size
            self.return_pct = (self.entry_price - self.exit_price) / self.entry_price


@dataclass
class BacktestResult:
    """Container for backtest results."""
    equity_curve: pd.Series
    trades: List[Trade]
    returns: pd.Series
    positions: pd.Series
    metrics: Dict[str, float]
    signals: pd.DataFrame


class Strategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str = "Strategy"):
        self.name = name
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.
        Should return a Series with values:
        1 for long, -1 for short, 0 for no position
        """
        raise NotImplementedError("Subclasses must implement generate_signals")


class SimpleMovingAverageCrossover(Strategy):
    """Simple moving average crossover strategy."""
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__("SMA Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on SMA crossover."""
        fast_sma = data['Close'].rolling(window=self.fast_period).mean()
        slow_sma = data['Close'].rolling(window=self.slow_period).mean()
        
        signals = pd.Series(index=data.index, data=0)
        signals[fast_sma > slow_sma] = 1
        signals[fast_sma < slow_sma] = -1
        
        return signals


class RSIMeanReversion(Strategy):
    """RSI-based mean reversion strategy."""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSI Mean Reversion")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on RSI levels."""
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(index=data.index, data=0)
        signals[rsi < self.oversold] = 1  # Buy when oversold
        signals[rsi > self.overbought] = -1  # Sell when overbought
        
        return signals


class BollingerBandStrategy(Strategy):
    """Bollinger Band breakout/mean reversion strategy."""
    
    def __init__(self, period: int = 20, num_std: float = 2, mode: str = 'mean_reversion'):
        super().__init__(f"Bollinger Band {mode}")
        self.period = period
        self.num_std = num_std
        self.mode = mode  # 'mean_reversion' or 'breakout'
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on Bollinger Bands."""
        sma = data['Close'].rolling(window=self.period).mean()
        std = data['Close'].rolling(window=self.period).std()
        upper_band = sma + (std * self.num_std)
        lower_band = sma - (std * self.num_std)
        
        signals = pd.Series(index=data.index, data=0)
        
        if self.mode == 'mean_reversion':
            # Buy when price touches lower band, sell when touches upper band
            signals[data['Close'] <= lower_band] = 1
            signals[data['Close'] >= upper_band] = -1
        else:  # breakout
            # Buy when price breaks above upper band, sell when breaks below lower band
            signals[data['Close'] > upper_band] = 1
            signals[data['Close'] < lower_band] = -1
        
        return signals


class Backtester:
    """Simple backtesting engine."""
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 commission: float = 0.001,
                 slippage: float = 0.0,
                 position_size: Union[float, str] = 1.0):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (as fraction)
            slippage: Slippage per trade (as fraction)
            position_size: Fixed position size or 'full' for full capital
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
    
    def run(self, 
            data: pd.DataFrame,
            strategy: Strategy,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None) -> BacktestResult:
        """
        Run backtest on given data and strategy.
        
        Args:
            data: DataFrame with OHLC data
            strategy: Strategy instance
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResult object
        """
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0
        trades = []
        current_trade = None
        
        equity_curve = pd.Series(index=data.index, dtype=float)
        positions = pd.Series(index=data.index, dtype=float)
        
        # Simulate trading
        for i, (date, row) in enumerate(data.iterrows()):
            if i == 0 or pd.isna(signals.iloc[i]):
                equity_curve.iloc[i] = capital
                positions.iloc[i] = position
                continue
            
            signal = signals.iloc[i]
            price = row['Close']
            
            # Handle position changes
            if signal != 0 and position != signal:
                # Close existing position
                if position != 0 and current_trade:
                    exit_price = price * (1 - self.slippage if position > 0 else 1 + self.slippage)
                    current_trade.close_trade(date, exit_price)
                    
                    # Update capital
                    capital += current_trade.pnl
                    capital -= abs(current_trade.pnl) * self.commission
                    
                    trades.append(current_trade)
                    current_trade = None
                    position = 0
                
                # Open new position
                if signal != 0:
                    entry_price = price * (1 + self.slippage if signal > 0 else 1 - self.slippage)
                    
                    if self.position_size == 'full':
                        trade_capital = capital * (1 - self.commission)
                        size = trade_capital / entry_price
                    else:
                        size = self.position_size
                    
                    current_trade = Trade(
                        entry_date=date,
                        exit_date=None,
                        entry_price=entry_price,
                        exit_price=None,
                        position_size=size,
                        trade_type='long' if signal > 0 else 'short'
                    )
                    
                    # Update capital for commission
                    capital -= abs(size * entry_price) * self.commission
                    position = signal
            
            # Update equity based on current position
            if position != 0 and current_trade:
                if position > 0:
                    unrealized_pnl = (price - current_trade.entry_price) * current_trade.position_size
                else:
                    unrealized_pnl = (current_trade.entry_price - price) * current_trade.position_size
                
                equity_curve.iloc[i] = capital + unrealized_pnl
            else:
                equity_curve.iloc[i] = capital
            
            positions.iloc[i] = position
        
        # Close any remaining position
        if current_trade:
            current_trade.close_trade(data.index[-1], data.iloc[-1]['Close'])
            trades.append(current_trade)
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, returns, trades)
        
        # Create signals DataFrame for analysis
        signals_df = pd.DataFrame({
            'signal': signals,
            'position': positions,
            'price': data['Close'],
            'equity': equity_curve
        })
        
        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            returns=returns,
            positions=positions,
            metrics=metrics,
            signals=signals_df
        )
    
    def _calculate_metrics(self, 
                          equity_curve: pd.Series,
                          returns: pd.Series,
                          trades: List[Trade]) -> Dict[str, float]:
        """Calculate performance metrics."""
        from performance_metrics import (
            sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
            annualized_return, annualized_volatility
        )
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Trade statistics
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else np.inf
        
        # Risk metrics
        try:
            sharpe = sharpe_ratio(returns) if len(returns) > 1 else 0
            sortino = sortino_ratio(returns) if len(returns) > 1 else 0
            calmar = calmar_ratio(returns) if len(returns) > 1 else 0
            max_dd, _, _ = max_drawdown(returns) if len(returns) > 1 else (0, None, None)
            annual_return = annualized_return(returns) if len(returns) > 1 else 0
            annual_vol = annualized_volatility(returns) if len(returns) > 1 else 0
        except:
            sharpe = sortino = calmar = max_dd = annual_return = annual_vol = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_equity': equity_curve.iloc[-1]
        }


def optimize_strategy(data: pd.DataFrame,
                     strategy_class: type,
                     param_grid: Dict[str, List],
                     metric: str = 'sharpe_ratio',
                     **backtest_kwargs) -> Tuple[Dict, BacktestResult]:
    """
    Optimize strategy parameters.
    
    Args:
        data: DataFrame with OHLC data
        strategy_class: Strategy class to optimize
        param_grid: Dictionary of parameter names to lists of values
        metric: Metric to optimize
        **backtest_kwargs: Additional arguments for Backtester
        
    Returns:
        Tuple of (best_params, best_result)
    """
    from itertools import product
    
    backtester = Backtester(**backtest_kwargs)
    
    best_metric = -np.inf
    best_params = None
    best_result = None
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for values in product(*param_values):
        params = dict(zip(param_names, values))
        
        # Create strategy with current parameters
        strategy = strategy_class(**params)
        
        # Run backtest
        try:
            result = backtester.run(data, strategy)
            
            # Check if this is the best result
            current_metric = result.metrics.get(metric, -np.inf)
            if current_metric > best_metric:
                best_metric = current_metric
                best_params = params
                best_result = result
        except:
            continue
    
    return best_params, best_result


def walk_forward_analysis(data: pd.DataFrame,
                         strategy: Strategy,
                         window_size: int,
                         step_size: int,
                         **backtest_kwargs) -> List[BacktestResult]:
    """
    Perform walk-forward analysis.
    
    Args:
        data: DataFrame with OHLC data
        strategy: Strategy instance
        window_size: Size of each training window
        step_size: Step size for moving window
        **backtest_kwargs: Additional arguments for Backtester
        
    Returns:
        List of BacktestResult objects
    """
    backtester = Backtester(**backtest_kwargs)
    results = []
    
    for i in range(window_size, len(data), step_size):
        # Define train and test periods
        train_end = i
        test_start = i
        test_end = min(i + step_size, len(data))
        
        # Skip if test period is too small
        if test_end - test_start < 10:
            continue
        
        # Run backtest on test period
        test_data = data.iloc[test_start:test_end]
        result = backtester.run(test_data, strategy)
        results.append(result)
    
    return results