"""
Momentum Trading Strategy Module

This module implements a momentum-based trading strategy with backtesting functionality.
Momentum strategies capitalize on the tendency of winning stocks to continue performing well
and losing stocks to continue performing poorly.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MomentumStrategy:
    """
    Implements a momentum trading strategy with various lookback periods and rebalancing frequencies.
    
    The strategy ranks stocks based on their past returns and goes long on top performers
    while optionally shorting poor performers.
    """
    
    def __init__(self, 
                 symbols: List[str],
                 lookback_period: int = 252,
                 holding_period: int = 21,
                 top_n: int = 5,
                 short_n: int = 0,
                 rebalance_frequency: str = 'monthly'):
        """
        Initialize the Momentum Strategy.
        
        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols to trade
        lookback_period : int
            Number of days to calculate momentum (default: 252 days = 1 year)
        holding_period : int
            Number of days to hold positions (default: 21 days = 1 month)
        top_n : int
            Number of top performers to go long (default: 5)
        short_n : int
            Number of bottom performers to short (default: 0, long-only)
        rebalance_frequency : str
            How often to rebalance ('daily', 'weekly', 'monthly')
        """
        self.symbols = symbols
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self.top_n = top_n
        self.short_n = short_n
        self.rebalance_frequency = rebalance_frequency
        self.data = None
        self.signals = None
        self.portfolio = None
        
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical price data for all symbols.
        
        Parameters:
        -----------
        start_date : str
            Start date for data fetch (YYYY-MM-DD)
        end_date : str
            End date for data fetch (YYYY-MM-DD)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with adjusted close prices for all symbols
        """
        # Adjust start date to account for lookback period
        start_dt = pd.to_datetime(start_date) - pd.DateOffset(days=self.lookback_period + 50)
        
        # Fetch data for all symbols
        data = pd.DataFrame()
        for symbol in self.symbols:
            try:
                stock_data = yf.download(symbol, start=start_dt, end=end_date, progress=False)
                data[symbol] = stock_data['Adj Close']
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                
        # Remove any symbols with insufficient data
        data = data.dropna(axis=1, thresh=len(data) * 0.8)
        self.data = data
        return data
    
    def calculate_momentum(self) -> pd.DataFrame:
        """
        Calculate momentum scores for all stocks.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with momentum scores for each stock
        """
        if self.data is None:
            raise ValueError("No data available. Please run fetch_data() first.")
            
        # Calculate returns over lookback period
        returns = self.data.pct_change(self.lookback_period)
        
        # Calculate momentum score (simple return over lookback period)
        momentum_scores = returns
        
        # Alternative momentum calculations can be added:
        # - Risk-adjusted momentum (Sharpe ratio)
        # - Time-series momentum
        # - Cross-sectional momentum
        
        return momentum_scores
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on momentum scores.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with trading signals (1 for long, -1 for short, 0 for no position)
        """
        momentum_scores = self.calculate_momentum()
        signals = pd.DataFrame(index=momentum_scores.index, columns=momentum_scores.columns)
        signals[:] = 0
        
        # Determine rebalancing dates
        if self.rebalance_frequency == 'monthly':
            rebalance_dates = momentum_scores.resample('M').last().index
        elif self.rebalance_frequency == 'weekly':
            rebalance_dates = momentum_scores.resample('W').last().index
        else:  # daily
            rebalance_dates = momentum_scores.index
            
        # Generate signals for each rebalancing date
        for date in rebalance_dates:
            if date not in momentum_scores.index:
                continue
                
            # Get momentum scores for this date
            scores = momentum_scores.loc[date].dropna()
            
            if len(scores) < self.top_n + self.short_n:
                continue
                
            # Rank stocks by momentum
            ranked = scores.rank(ascending=False)
            
            # Long positions (top performers)
            long_stocks = ranked[ranked <= self.top_n].index
            
            # Short positions (bottom performers)
            if self.short_n > 0:
                short_stocks = ranked[ranked > len(ranked) - self.short_n].index
            else:
                short_stocks = []
                
            # Set signals for holding period
            end_date = date + pd.DateOffset(days=self.holding_period)
            mask = (signals.index >= date) & (signals.index < end_date)
            
            for stock in long_stocks:
                if stock in signals.columns:
                    signals.loc[mask, stock] = 1
                    
            for stock in short_stocks:
                if stock in signals.columns:
                    signals.loc[mask, stock] = -1
                    
        self.signals = signals
        return signals
    
    def backtest(self, 
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.001) -> Dict:
        """
        Backtest the momentum strategy.
        
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
        prices = self.data
        
        # Initialize portfolio
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['holdings'] = 0.0
        portfolio['cash'] = initial_capital
        portfolio['total'] = initial_capital
        portfolio['returns'] = 0.0
        
        # Track positions
        positions = pd.DataFrame(index=signals.index, columns=signals.columns)
        positions[:] = 0
        
        # Calculate daily positions and returns
        for i in range(1, len(signals)):
            date = signals.index[i]
            prev_date = signals.index[i-1]
            
            # Calculate position sizes (equal weight)
            long_positions = (signals.loc[date] == 1).sum()
            short_positions = (signals.loc[date] == -1).sum()
            total_positions = long_positions + short_positions
            
            if total_positions > 0:
                position_size = portfolio.loc[prev_date, 'total'] / total_positions
            else:
                position_size = 0
                
            # Update positions
            for symbol in signals.columns:
                if pd.isna(prices.loc[date, symbol]) or pd.isna(prices.loc[prev_date, symbol]):
                    continue
                    
                # Check for signal changes
                if signals.loc[date, symbol] != signals.loc[prev_date, symbol]:
                    # Close previous position
                    if signals.loc[prev_date, symbol] != 0:
                        shares = positions.loc[prev_date, symbol]
                        trade_value = shares * prices.loc[date, symbol]
                        trade_cost = abs(trade_value) * (commission + slippage)
                        portfolio.loc[date, 'cash'] += trade_value - trade_cost
                        positions.loc[date, symbol] = 0
                        
                    # Open new position
                    if signals.loc[date, symbol] != 0:
                        shares = (position_size / prices.loc[date, symbol]) * signals.loc[date, symbol]
                        trade_value = shares * prices.loc[date, symbol]
                        trade_cost = abs(trade_value) * (commission + slippage)
                        portfolio.loc[date, 'cash'] = portfolio.loc[prev_date, 'cash'] - trade_value - trade_cost
                        positions.loc[date, symbol] = shares
                else:
                    # Maintain position
                    positions.loc[date, symbol] = positions.loc[prev_date, symbol]
                    if i == 1:
                        portfolio.loc[date, 'cash'] = portfolio.loc[prev_date, 'cash']
                        
            # Calculate holdings value
            holdings_value = 0
            for symbol in signals.columns:
                if pd.notna(prices.loc[date, symbol]):
                    holdings_value += positions.loc[date, symbol] * prices.loc[date, symbol]
                    
            portfolio.loc[date, 'holdings'] = holdings_value
            portfolio.loc[date, 'total'] = portfolio.loc[date, 'cash'] + holdings_value
            portfolio.loc[date, 'returns'] = (portfolio.loc[date, 'total'] / portfolio.loc[prev_date, 'total']) - 1
            
        self.portfolio = portfolio
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(portfolio)
        
        return {
            'portfolio': portfolio,
            'positions': positions,
            'metrics': metrics
        }
    
    def calculate_performance_metrics(self, portfolio: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Parameters:
        -----------
        portfolio : pd.DataFrame
            Portfolio value and returns over time
            
        Returns:
        --------
        Dict
            Dictionary containing various performance metrics
        """
        # Remove any initial NaN values
        returns = portfolio['returns'].dropna()
        
        # Basic metrics
        total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
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
        
        # Calculate drawdown duration
        drawdown_start = drawdown[drawdown == max_drawdown].index[0]
        if (drawdown[drawdown.index > drawdown_start] == 0).any():
            drawdown_end = drawdown[drawdown.index > drawdown_start][drawdown[drawdown.index > drawdown_start] == 0].index[0]
            max_drawdown_duration = (drawdown_end - drawdown_start).days
        else:
            max_drawdown_duration = (drawdown.index[-1] - drawdown_start).days
            
        # Win rate and profit factor
        winning_days = (returns > 0).sum()
        losing_days = (returns < 0).sum()
        win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0
        
        avg_win = returns[returns > 0].mean() if winning_days > 0 else 0
        avg_loss = abs(returns[returns < 0].mean()) if losing_days > 0 else 0
        profit_factor = (winning_days * avg_win) / (losing_days * avg_loss) if losing_days > 0 and avg_loss > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot backtest results including equity curve and drawdown.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        if self.portfolio is None:
            raise ValueError("No backtest results available. Please run backtest() first.")
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot equity curve
        ax1.plot(self.portfolio.index, self.portfolio['total'], label='Portfolio Value', linewidth=2)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Momentum Strategy - Equity Curve')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot drawdown
        returns = self.portfolio['returns'].fillna(0)
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        ax2.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.3, color='red')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.set_title('Drawdown')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage function
def example_momentum_backtest():
    """
    Example of running a momentum strategy backtest.
    """
    # Define universe of stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'V']
    
    # Initialize strategy
    strategy = MomentumStrategy(
        symbols=symbols,
        lookback_period=252,  # 1 year momentum
        holding_period=21,    # Hold for 1 month
        top_n=3,             # Long top 3 performers
        short_n=0,           # Long-only strategy
        rebalance_frequency='monthly'
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
            
    return strategy, results


if __name__ == "__main__":
    strategy, results = example_momentum_backtest()
    strategy.plot_results()