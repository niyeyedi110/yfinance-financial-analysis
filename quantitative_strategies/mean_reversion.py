"""
Mean Reversion Trading Strategy Module

This module implements a mean reversion strategy with backtesting functionality.
Mean reversion strategies assume that asset prices tend to revert to their historical
average over time, allowing traders to profit from temporary deviations.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MeanReversionStrategy:
    """
    Implements a mean reversion trading strategy using various statistical methods
    including z-score, Bollinger Bands, and RSI-based approaches.
    """
    
    def __init__(self,
                 symbol: str,
                 lookback_period: int = 20,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.0,
                 stop_loss: float = 3.0,
                 method: str = 'zscore'):
        """
        Initialize the Mean Reversion Strategy.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol to trade
        lookback_period : int
            Number of days for calculating moving average (default: 20)
        entry_threshold : float
            Number of standard deviations for entry signal (default: 2.0)
        exit_threshold : float
            Number of standard deviations for exit signal (default: 0.0)
        stop_loss : float
            Number of standard deviations for stop loss (default: 3.0)
        method : str
            Method to use ('zscore', 'bollinger', 'rsi')
        """
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.method = method
        self.data = None
        self.signals = None
        self.portfolio = None
        
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical price data for the symbol.
        
        Parameters:
        -----------
        start_date : str
            Start date for data fetch (YYYY-MM-DD)
        end_date : str
            End date for data fetch (YYYY-MM-DD)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with OHLCV data
        """
        # Adjust start date to account for lookback period
        start_dt = pd.to_datetime(start_date) - pd.DateOffset(days=self.lookback_period * 2)
        
        # Fetch data
        self.data = yf.download(self.symbol, start=start_dt, end=end_date, progress=False)
        return self.data
    
    def calculate_zscore(self, prices: pd.Series) -> pd.Series:
        """
        Calculate z-score of prices.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
            
        Returns:
        --------
        pd.Series
            Z-score series
        """
        rolling_mean = prices.rolling(window=self.lookback_period).mean()
        rolling_std = prices.rolling(window=self.lookback_period).std()
        zscore = (prices - rolling_mean) / rolling_std
        return zscore
    
    def calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series]
            Middle band (SMA), upper band, lower band
        """
        middle_band = prices.rolling(window=self.lookback_period).mean()
        std = prices.rolling(window=self.lookback_period).std()
        upper_band = middle_band + (std * self.entry_threshold)
        lower_band = middle_band - (std * self.entry_threshold)
        return middle_band, upper_band, lower_band
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        period : int
            RSI period (default: 14)
            
        Returns:
        --------
        pd.Series
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on the selected method.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with trading signals
        """
        if self.data is None:
            raise ValueError("No data available. Please run fetch_data() first.")
            
        prices = self.data['Adj Close']
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = prices
        signals['signal'] = 0
        signals['position'] = 0
        
        if self.method == 'zscore':
            # Z-score method
            zscore = self.calculate_zscore(prices)
            signals['zscore'] = zscore
            
            # Entry signals
            signals.loc[zscore < -self.entry_threshold, 'signal'] = 1  # Buy when oversold
            signals.loc[zscore > self.entry_threshold, 'signal'] = -1  # Short when overbought
            
            # Exit signals
            signals.loc[abs(zscore) < self.exit_threshold, 'signal'] = 0
            
            # Stop loss
            signals.loc[zscore < -self.stop_loss, 'signal'] = 0  # Exit long on extreme move
            signals.loc[zscore > self.stop_loss, 'signal'] = 0   # Exit short on extreme move
            
        elif self.method == 'bollinger':
            # Bollinger Bands method
            middle, upper, lower = self.calculate_bollinger_bands(prices)
            signals['middle_band'] = middle
            signals['upper_band'] = upper
            signals['lower_band'] = lower
            
            # Entry signals
            signals.loc[prices < lower, 'signal'] = 1   # Buy at lower band
            signals.loc[prices > upper, 'signal'] = -1  # Short at upper band
            
            # Exit signals (price crosses middle band)
            for i in range(1, len(signals)):
                if signals['position'].iloc[i-1] == 1 and prices.iloc[i] > middle.iloc[i]:
                    signals.loc[signals.index[i], 'signal'] = 0
                elif signals['position'].iloc[i-1] == -1 and prices.iloc[i] < middle.iloc[i]:
                    signals.loc[signals.index[i], 'signal'] = 0
                    
        elif self.method == 'rsi':
            # RSI method
            rsi = self.calculate_rsi(prices)
            signals['rsi'] = rsi
            
            # Entry signals
            signals.loc[rsi < 30, 'signal'] = 1   # Buy when oversold (RSI < 30)
            signals.loc[rsi > 70, 'signal'] = -1  # Short when overbought (RSI > 70)
            
            # Exit signals
            signals.loc[(rsi > 40) & (rsi < 60), 'signal'] = 0  # Exit in neutral zone
            
        # Calculate positions
        signals['position'] = signals['signal'].replace(to_replace=0, method='ffill').fillna(0)
        
        self.signals = signals
        return signals
    
    def backtest(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.001) -> Dict:
        """
        Backtest the mean reversion strategy.
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital (default: $100,000)
        commission : float
            Commission rate per trade (default: 0.1%)
        slippage : float
            Slippage rate per trade (default: 0.1%)
            
        Returns:
        --------
        Dict
            Dictionary containing backtest results and performance metrics
        """
        if self.signals is None:
            self.generate_signals()
            
        signals = self.signals
        
        # Initialize portfolio
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['holdings'] = 0.0
        portfolio['cash'] = initial_capital
        portfolio['total'] = initial_capital
        portfolio['returns'] = 0.0
        
        # Track trades
        trades = []
        position = 0
        entry_price = 0
        
        for i in range(1, len(signals)):
            date = signals.index[i]
            price = signals['price'].iloc[i]
            signal = signals['signal'].iloc[i]
            prev_position = signals['position'].iloc[i-1]
            current_position = signals['position'].iloc[i]
            
            # Check for position changes
            if current_position != prev_position:
                # Close previous position
                if prev_position != 0:
                    # Calculate P&L
                    exit_price = price * (1 - slippage * np.sign(prev_position))
                    pnl = (exit_price - entry_price) * position
                    trade_cost = abs(position * exit_price) * commission
                    
                    portfolio.loc[date, 'cash'] = portfolio['cash'].iloc[i-1] + (position * exit_price) - trade_cost
                    position = 0
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': prev_position,
                        'pnl': pnl - trade_cost,
                        'return': (exit_price - entry_price) / entry_price * np.sign(prev_position)
                    })
                else:
                    portfolio.loc[date, 'cash'] = portfolio['cash'].iloc[i-1]
                    
                # Open new position
                if current_position != 0:
                    entry_price = price * (1 + slippage * np.sign(current_position))
                    position_value = portfolio.loc[date, 'cash'] * 0.95  # Use 95% of capital
                    position = position_value / entry_price * np.sign(current_position)
                    trade_cost = abs(position * entry_price) * commission
                    
                    portfolio.loc[date, 'cash'] = portfolio.loc[date, 'cash'] - (position * entry_price) - trade_cost
                    entry_date = date
            else:
                # Maintain position
                portfolio.loc[date, 'cash'] = portfolio['cash'].iloc[i-1]
                
            # Calculate holdings value
            portfolio.loc[date, 'holdings'] = position * price
            portfolio.loc[date, 'total'] = portfolio.loc[date, 'cash'] + portfolio.loc[date, 'holdings']
            
            if i > 1:
                portfolio.loc[date, 'returns'] = (portfolio.loc[date, 'total'] / portfolio['total'].iloc[i-1]) - 1
                
        self.portfolio = portfolio
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(portfolio, trades)
        
        return {
            'portfolio': portfolio,
            'trades': pd.DataFrame(trades),
            'metrics': metrics,
            'signals': signals
        }
    
    def calculate_performance_metrics(self, portfolio: pd.DataFrame, trades: List[Dict]) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Parameters:
        -----------
        portfolio : pd.DataFrame
            Portfolio value and returns over time
        trades : List[Dict]
            List of completed trades
            
        Returns:
        --------
        Dict
            Dictionary containing various performance metrics
        """
        # Basic return metrics
        returns = portfolio['returns'].dropna()
        total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1
        
        # Annualized metrics
        days = (portfolio.index[-1] - portfolio.index[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown analysis
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        trade_df = pd.DataFrame(trades)
        if len(trade_df) > 0:
            winning_trades = trade_df[trade_df['pnl'] > 0]
            losing_trades = trade_df[trade_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trade_df)
            avg_win = winning_trades['return'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['return'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
            
            # Average trade duration
            trade_df['duration'] = (pd.to_datetime(trade_df['exit_date']) - pd.to_datetime(trade_df['entry_date'])).dt.days
            avg_trade_duration = trade_df['duration'].mean()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade_duration = 0
            
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(trade_df) if len(trades) > 0 else 0,
            'avg_trade_duration': avg_trade_duration
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot backtest results including signals, equity curve, and indicators.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        if self.portfolio is None or self.signals is None:
            raise ValueError("No backtest results available. Please run backtest() first.")
            
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot price and signals
        ax1 = axes[0]
        ax1.plot(self.signals.index, self.signals['price'], label='Price', color='black', linewidth=1)
        
        # Plot method-specific indicators
        if self.method == 'bollinger':
            ax1.plot(self.signals.index, self.signals['middle_band'], label='Middle Band', linestyle='--', alpha=0.7)
            ax1.plot(self.signals.index, self.signals['upper_band'], label='Upper Band', linestyle='--', alpha=0.7)
            ax1.plot(self.signals.index, self.signals['lower_band'], label='Lower Band', linestyle='--', alpha=0.7)
            
        # Mark buy and sell signals
        buy_signals = self.signals[self.signals['signal'] == 1]
        sell_signals = self.signals[self.signals['signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['price'], color='green', marker='^', s=100, label='Buy')
        ax1.scatter(sell_signals.index, sell_signals['price'], color='red', marker='v', s=100, label='Sell')
        
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'Mean Reversion Strategy ({self.method.upper()}) - {self.symbol}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot indicator
        ax2 = axes[1]
        if self.method == 'zscore':
            ax2.plot(self.signals.index, self.signals['zscore'], label='Z-Score', color='blue')
            ax2.axhline(y=self.entry_threshold, color='r', linestyle='--', alpha=0.7, label='Entry Threshold')
            ax2.axhline(y=-self.entry_threshold, color='r', linestyle='--', alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_ylabel('Z-Score')
        elif self.method == 'rsi':
            ax2.plot(self.signals.index, self.signals['rsi'], label='RSI', color='purple')
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
            ax2.set_ylabel('RSI')
            
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot portfolio value
        ax3 = axes[2]
        ax3.plot(self.portfolio.index, self.portfolio['total'], label='Portfolio Value', color='green', linewidth=2)
        ax3.set_ylabel('Portfolio Value ($)')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage function
def example_mean_reversion_backtest():
    """
    Example of running a mean reversion strategy backtest.
    """
    # Initialize strategy
    strategy = MeanReversionStrategy(
        symbol='SPY',
        lookback_period=20,
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        method='zscore'  # Can use 'zscore', 'bollinger', or 'rsi'
    )
    
    # Fetch data
    print("Fetching historical data...")
    strategy.fetch_data('2020-01-01', '2024-01-01')
    
    # Run backtest
    print("Running backtest...")
    results = strategy.backtest(initial_capital=100000)
    
    # Display results
    print("\nBacktest Results:")
    print("-" * 50)
    for metric, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"{metric.replace('_', ' ').title()}: {value}")
            
    # Display trade summary
    if len(results['trades']) > 0:
        print("\nTrade Summary:")
        print("-" * 50)
        print(f"Total Trades: {len(results['trades'])}")
        print(f"Average Trade Duration: {results['trades']['duration'].mean():.1f} days")
        print(f"Best Trade: {results['trades']['return'].max():.2%}")
        print(f"Worst Trade: {results['trades']['return'].min():.2%}")
        
    return strategy, results


if __name__ == "__main__":
    strategy, results = example_mean_reversion_backtest()
    strategy.plot_results()