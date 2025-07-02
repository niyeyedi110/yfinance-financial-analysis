"""
Pairs Trading Strategy Module

This module implements a statistical arbitrage pairs trading strategy with backtesting functionality.
Pairs trading involves identifying two stocks that historically move together and trading
the spread when they diverge from their typical relationship.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class PairsTradingStrategy:
    """
    Implements a pairs trading strategy using cointegration and statistical arbitrage methods.
    The strategy identifies pairs of stocks with strong historical correlation and trades
    the spread when it deviates from its mean.
    """
    
    def __init__(self,
                 symbol1: str,
                 symbol2: str,
                 lookback_period: int = 60,
                 entry_zscore: float = 2.0,
                 exit_zscore: float = 0.0,
                 stop_loss_zscore: float = 3.0,
                 hedge_ratio_lookback: int = 20):
        """
        Initialize the Pairs Trading Strategy.
        
        Parameters:
        -----------
        symbol1 : str
            First stock symbol
        symbol2 : str
            Second stock symbol
        lookback_period : int
            Number of days for calculating spread statistics (default: 60)
        entry_zscore : float
            Z-score threshold for entering trades (default: 2.0)
        exit_zscore : float
            Z-score threshold for exiting trades (default: 0.0)
        stop_loss_zscore : float
            Z-score threshold for stop loss (default: 3.0)
        hedge_ratio_lookback : int
            Period for calculating dynamic hedge ratio (default: 20)
        """
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.lookback_period = lookback_period
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.hedge_ratio_lookback = hedge_ratio_lookback
        self.data = None
        self.spread_data = None
        self.signals = None
        self.portfolio = None
        
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical price data for both symbols.
        
        Parameters:
        -----------
        start_date : str
            Start date for data fetch (YYYY-MM-DD)
        end_date : str
            End date for data fetch (YYYY-MM-DD)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with adjusted close prices for both symbols
        """
        # Adjust start date to account for lookback period
        start_dt = pd.to_datetime(start_date) - pd.DateOffset(days=self.lookback_period * 2)
        
        # Fetch data for both symbols
        data = pd.DataFrame()
        for symbol in [self.symbol1, self.symbol2]:
            stock_data = yf.download(symbol, start=start_dt, end=end_date, progress=False)
            data[symbol] = stock_data['Adj Close']
            
        # Remove any NaN values
        data = data.dropna()
        self.data = data
        return data
    
    def test_cointegration(self, series1: pd.Series, series2: pd.Series) -> Tuple[float, float, bool]:
        """
        Test for cointegration between two time series.
        
        Parameters:
        -----------
        series1 : pd.Series
            First price series
        series2 : pd.Series
            Second price series
            
        Returns:
        --------
        Tuple[float, float, bool]
            p-value, test statistic, and whether series are cointegrated
        """
        # Engle-Granger two-step cointegration test
        score, pvalue, _ = coint(series1, series2)
        
        # Consider cointegrated if p-value < 0.05
        is_cointegrated = pvalue < 0.05
        
        return pvalue, score, is_cointegrated
    
    def calculate_hedge_ratio(self, series1: pd.Series, series2: pd.Series) -> float:
        """
        Calculate the hedge ratio using OLS regression.
        
        Parameters:
        -----------
        series1 : pd.Series
            First price series (dependent variable)
        series2 : pd.Series
            Second price series (independent variable)
            
        Returns:
        --------
        float
            Hedge ratio (beta coefficient)
        """
        # Prepare data for regression
        X = series2.values.reshape(-1, 1)
        y = series1.values
        
        # Add constant term
        X = np.column_stack([np.ones(len(X)), X])
        
        # Run OLS regression
        model = OLS(y, X).fit()
        
        # Return the slope coefficient (hedge ratio)
        return model.params[1]
    
    def calculate_spread(self, rolling: bool = True) -> pd.DataFrame:
        """
        Calculate the spread between the two assets.
        
        Parameters:
        -----------
        rolling : bool
            Whether to use rolling hedge ratio (default: True)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing spread data and statistics
        """
        if self.data is None:
            raise ValueError("No data available. Please run fetch_data() first.")
            
        spread_data = pd.DataFrame(index=self.data.index)
        spread_data['price1'] = self.data[self.symbol1]
        spread_data['price2'] = self.data[self.symbol2]
        
        if rolling:
            # Calculate rolling hedge ratio
            spread_data['hedge_ratio'] = np.nan
            for i in range(self.hedge_ratio_lookback, len(spread_data)):
                window_data1 = spread_data['price1'].iloc[i-self.hedge_ratio_lookback:i]
                window_data2 = spread_data['price2'].iloc[i-self.hedge_ratio_lookback:i]
                spread_data.loc[spread_data.index[i], 'hedge_ratio'] = self.calculate_hedge_ratio(window_data1, window_data2)
        else:
            # Use static hedge ratio
            hedge_ratio = self.calculate_hedge_ratio(spread_data['price1'], spread_data['price2'])
            spread_data['hedge_ratio'] = hedge_ratio
            
        # Calculate spread
        spread_data['spread'] = spread_data['price1'] - spread_data['hedge_ratio'] * spread_data['price2']
        
        # Calculate spread statistics
        spread_data['spread_mean'] = spread_data['spread'].rolling(window=self.lookback_period).mean()
        spread_data['spread_std'] = spread_data['spread'].rolling(window=self.lookback_period).std()
        spread_data['zscore'] = (spread_data['spread'] - spread_data['spread_mean']) / spread_data['spread_std']
        
        # Test for stationarity of the spread
        spread_clean = spread_data['spread'].dropna()
        if len(spread_clean) > 0:
            adf_result = adfuller(spread_clean)
            spread_data['is_stationary'] = adf_result[1] < 0.05  # p-value < 0.05
        
        self.spread_data = spread_data
        return spread_data
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on spread z-score.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with trading signals
        """
        if self.spread_data is None:
            self.calculate_spread()
            
        signals = self.spread_data.copy()
        signals['signal'] = 0
        signals['position'] = 0
        
        # Generate entry and exit signals
        # When z-score is high, spread is too wide -> short the spread (short stock1, long stock2)
        # When z-score is low, spread is too narrow -> long the spread (long stock1, short stock2)
        
        for i in range(1, len(signals)):
            zscore = signals['zscore'].iloc[i]
            prev_position = signals['position'].iloc[i-1] if i > 0 else 0
            
            if pd.isna(zscore):
                signals.loc[signals.index[i], 'position'] = prev_position
                continue
                
            # Entry signals
            if prev_position == 0:
                if zscore > self.entry_zscore:
                    signals.loc[signals.index[i], 'signal'] = -1  # Short the spread
                    signals.loc[signals.index[i], 'position'] = -1
                elif zscore < -self.entry_zscore:
                    signals.loc[signals.index[i], 'signal'] = 1   # Long the spread
                    signals.loc[signals.index[i], 'position'] = 1
                else:
                    signals.loc[signals.index[i], 'position'] = 0
                    
            # Exit signals
            else:
                # Check for exit conditions
                if prev_position == 1:  # Long position
                    if zscore > -self.exit_zscore or zscore < -self.stop_loss_zscore:
                        signals.loc[signals.index[i], 'signal'] = 0
                        signals.loc[signals.index[i], 'position'] = 0
                    else:
                        signals.loc[signals.index[i], 'position'] = prev_position
                        
                elif prev_position == -1:  # Short position
                    if zscore < self.exit_zscore or zscore > self.stop_loss_zscore:
                        signals.loc[signals.index[i], 'signal'] = 0
                        signals.loc[signals.index[i], 'position'] = 0
                    else:
                        signals.loc[signals.index[i], 'position'] = prev_position
                        
        self.signals = signals
        return signals
    
    def backtest(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.001) -> Dict:
        """
        Backtest the pairs trading strategy.
        
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
        portfolio['stock1_position'] = 0.0
        portfolio['stock2_position'] = 0.0
        portfolio['cash'] = initial_capital
        portfolio['holdings'] = 0.0
        portfolio['total'] = initial_capital
        portfolio['returns'] = 0.0
        
        # Track trades
        trades = []
        stock1_shares = 0
        stock2_shares = 0
        
        for i in range(1, len(signals)):
            date = signals.index[i]
            signal = signals['signal'].iloc[i]
            position = signals['position'].iloc[i]
            prev_position = signals['position'].iloc[i-1]
            
            price1 = signals['price1'].iloc[i]
            price2 = signals['price2'].iloc[i]
            hedge_ratio = signals['hedge_ratio'].iloc[i]
            
            # Initialize cash from previous day
            portfolio.loc[date, 'cash'] = portfolio['cash'].iloc[i-1]
            
            # Check for position changes
            if position != prev_position:
                # Close previous position
                if prev_position != 0:
                    # Calculate closing values
                    close_value1 = stock1_shares * price1 * (1 - slippage * np.sign(-stock1_shares))
                    close_value2 = stock2_shares * price2 * (1 - slippage * np.sign(-stock2_shares))
                    close_cost = (abs(close_value1) + abs(close_value2)) * commission
                    
                    portfolio.loc[date, 'cash'] += close_value1 + close_value2 - close_cost
                    
                    # Record trade
                    trades.append({
                        'exit_date': date,
                        'exit_price1': price1,
                        'exit_price2': price2,
                        'position': prev_position,
                        'pnl': close_value1 + close_value2 - entry_value1 - entry_value2 - close_cost - entry_cost
                    })
                    
                    stock1_shares = 0
                    stock2_shares = 0
                    
                # Open new position
                if position != 0:
                    # Calculate position sizes
                    total_capital = portfolio.loc[date, 'cash'] * 0.95  # Use 95% of capital
                    
                    # For pairs trading, allocate capital based on hedge ratio
                    capital_stock1 = total_capital / (1 + hedge_ratio)
                    capital_stock2 = total_capital - capital_stock1
                    
                    if position == 1:  # Long spread: long stock1, short stock2
                        stock1_shares = capital_stock1 / (price1 * (1 + slippage))
                        stock2_shares = -capital_stock2 / (price2 * (1 - slippage))
                    else:  # Short spread: short stock1, long stock2
                        stock1_shares = -capital_stock1 / (price1 * (1 - slippage))
                        stock2_shares = capital_stock2 / (price2 * (1 + slippage))
                        
                    # Calculate entry values
                    entry_value1 = stock1_shares * price1 * (1 + slippage * np.sign(stock1_shares))
                    entry_value2 = stock2_shares * price2 * (1 + slippage * np.sign(stock2_shares))
                    entry_cost = (abs(entry_value1) + abs(entry_value2)) * commission
                    
                    portfolio.loc[date, 'cash'] -= entry_value1 + entry_value2 + entry_cost
                    
                    # Start new trade record
                    trades.append({
                        'entry_date': date,
                        'entry_price1': price1,
                        'entry_price2': price2,
                        'hedge_ratio': hedge_ratio,
                        'stock1_shares': stock1_shares,
                        'stock2_shares': stock2_shares
                    })
                    
            # Update positions
            portfolio.loc[date, 'stock1_position'] = stock1_shares
            portfolio.loc[date, 'stock2_position'] = stock2_shares
            
            # Calculate holdings value
            holdings_value = stock1_shares * price1 + stock2_shares * price2
            portfolio.loc[date, 'holdings'] = holdings_value
            portfolio.loc[date, 'total'] = portfolio.loc[date, 'cash'] + holdings_value
            portfolio.loc[date, 'returns'] = (portfolio.loc[date, 'total'] / portfolio['total'].iloc[i-1]) - 1
            
        self.portfolio = portfolio
        
        # Process trades
        completed_trades = []
        i = 0
        while i < len(trades) - 1:
            if 'entry_date' in trades[i] and 'exit_date' in trades[i+1]:
                trade = {**trades[i], **trades[i+1]}
                trade['duration'] = (pd.to_datetime(trade['exit_date']) - pd.to_datetime(trade['entry_date'])).days
                completed_trades.append(trade)
                i += 2
            else:
                i += 1
                
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(portfolio, completed_trades)
        
        return {
            'portfolio': portfolio,
            'trades': pd.DataFrame(completed_trades),
            'metrics': metrics,
            'signals': signals
        }
    
    def calculate_performance_metrics(self, portfolio: pd.DataFrame, trades: List[Dict]) -> Dict:
        """
        Calculate comprehensive performance metrics for pairs trading.
        
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
        if len(trades) > 0:
            trade_df = pd.DataFrame(trades)
            winning_trades = trade_df[trade_df['pnl'] > 0]
            losing_trades = trade_df[trade_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trade_df)
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
            
            avg_trade_duration = trade_df['duration'].mean()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade_duration = 0
            
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Pairs-specific metrics
        if self.spread_data is not None:
            spread_mean_reversion_time = self.calculate_mean_reversion_time()
            correlation = self.data[self.symbol1].corr(self.data[self.symbol2])
        else:
            spread_mean_reversion_time = 0
            correlation = 0
            
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
            'avg_trade_duration': avg_trade_duration,
            'correlation': correlation,
            'mean_reversion_time': spread_mean_reversion_time
        }
    
    def calculate_mean_reversion_time(self) -> float:
        """
        Calculate the average time for the spread to revert to its mean.
        
        Returns:
        --------
        float
            Average number of days for mean reversion
        """
        if self.spread_data is None:
            return 0
            
        zscore = self.spread_data['zscore'].dropna()
        
        # Find crossings of zero
        crossings = []
        for i in range(1, len(zscore)):
            if zscore.iloc[i] * zscore.iloc[i-1] < 0:  # Sign change
                crossings.append(i)
                
        # Calculate average time between crossings
        if len(crossings) > 1:
            times = []
            for i in range(1, len(crossings)):
                days = (zscore.index[crossings[i]] - zscore.index[crossings[i-1]]).days
                times.append(days)
            return np.mean(times)
        else:
            return 0
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot backtest results including spread, z-score, and portfolio performance.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        if self.portfolio is None or self.signals is None:
            raise ValueError("No backtest results available. Please run backtest() first.")
            
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        
        # Plot normalized prices
        ax1 = axes[0]
        normalized_price1 = self.signals['price1'] / self.signals['price1'].iloc[0]
        normalized_price2 = self.signals['price2'] / self.signals['price2'].iloc[0]
        ax1.plot(self.signals.index, normalized_price1, label=self.symbol1, alpha=0.8)
        ax1.plot(self.signals.index, normalized_price2, label=self.symbol2, alpha=0.8)
        ax1.set_ylabel('Normalized Price')
        ax1.set_title(f'Pairs Trading: {self.symbol1} vs {self.symbol2}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot spread
        ax2 = axes[1]
        ax2.plot(self.signals.index, self.signals['spread'], label='Spread', color='purple')
        ax2.plot(self.signals.index, self.signals['spread_mean'], label='Mean', linestyle='--', alpha=0.7)
        ax2.fill_between(self.signals.index, 
                        self.signals['spread_mean'] + self.signals['spread_std'],
                        self.signals['spread_mean'] - self.signals['spread_std'],
                        alpha=0.2, label='±1 Std Dev')
        ax2.set_ylabel('Spread')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot z-score with signals
        ax3 = axes[2]
        ax3.plot(self.signals.index, self.signals['zscore'], label='Z-Score', color='blue')
        ax3.axhline(y=self.entry_zscore, color='r', linestyle='--', alpha=0.7, label='Entry Threshold')
        ax3.axhline(y=-self.entry_zscore, color='r', linestyle='--', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Mark positions
        long_positions = self.signals[self.signals['position'] == 1]
        short_positions = self.signals[self.signals['position'] == -1]
        
        if len(long_positions) > 0:
            ax3.scatter(long_positions.index, long_positions['zscore'], 
                       color='green', marker='^', s=50, alpha=0.7, label='Long Spread')
        if len(short_positions) > 0:
            ax3.scatter(short_positions.index, short_positions['zscore'], 
                       color='red', marker='v', s=50, alpha=0.7, label='Short Spread')
            
        ax3.set_ylabel('Z-Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot portfolio value
        ax4 = axes[3]
        ax4.plot(self.portfolio.index, self.portfolio['total'], label='Portfolio Value', color='green', linewidth=2)
        ax4.set_ylabel('Portfolio Value ($)')
        ax4.set_xlabel('Date')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage function
def example_pairs_trading_backtest():
    """
    Example of running a pairs trading strategy backtest.
    """
    # Initialize strategy with a known pair
    strategy = PairsTradingStrategy(
        symbol1='PEP',    # Pepsi
        symbol2='KO',     # Coca-Cola
        lookback_period=60,
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=3.0,
        hedge_ratio_lookback=20
    )
    
    # Fetch data
    print("Fetching historical data...")
    strategy.fetch_data('2020-01-01', '2024-01-01')
    
    # Test cointegration
    print("\nTesting cointegration...")
    pvalue, score, is_cointegrated = strategy.test_cointegration(
        strategy.data[strategy.symbol1], 
        strategy.data[strategy.symbol2]
    )
    print(f"Cointegration p-value: {pvalue:.4f}")
    print(f"Stocks are {'cointegrated' if is_cointegrated else 'not cointegrated'}")
    
    # Calculate spread
    print("\nCalculating spread...")
    spread_data = strategy.calculate_spread()
    
    # Run backtest
    print("\nRunning backtest...")
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
        print(f"Best Trade P&L: ${results['trades']['pnl'].max():.2f}")
        print(f"Worst Trade P&L: ${results['trades']['pnl'].min():.2f}")
        
    return strategy, results


def find_cointegrated_pairs(symbols: List[str], 
                           start_date: str, 
                           end_date: str,
                           p_value_threshold: float = 0.05) -> List[Tuple[str, str, float]]:
    """
    Find cointegrated pairs from a list of symbols.
    
    Parameters:
    -----------
    symbols : List[str]
        List of stock symbols to test
    start_date : str
        Start date for historical data
    end_date : str
        End date for historical data
    p_value_threshold : float
        P-value threshold for cointegration (default: 0.05)
        
    Returns:
    --------
    List[Tuple[str, str, float]]
        List of cointegrated pairs with their p-values
    """
    # Fetch data for all symbols
    data = pd.DataFrame()
    for symbol in symbols:
        try:
            stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            data[symbol] = stock_data['Adj Close']
        except:
            print(f"Error fetching data for {symbol}")
            
    data = data.dropna(axis=1)
    
    # Test all pairs for cointegration
    cointegrated_pairs = []
    
    for i in range(len(data.columns)):
        for j in range(i+1, len(data.columns)):
            symbol1 = data.columns[i]
            symbol2 = data.columns[j]
            
            try:
                score, pvalue, _ = coint(data[symbol1], data[symbol2])
                if pvalue < p_value_threshold:
                    cointegrated_pairs.append((symbol1, symbol2, pvalue))
                    print(f"Found cointegrated pair: {symbol1}-{symbol2} (p-value: {pvalue:.4f})")
            except:
                pass
                
    # Sort by p-value
    cointegrated_pairs.sort(key=lambda x: x[2])
    
    return cointegrated_pairs


if __name__ == "__main__":
    # Run example backtest
    strategy, results = example_pairs_trading_backtest()
    strategy.plot_results()
    
    # Example: Find cointegrated pairs
    print("\n\nSearching for cointegrated pairs...")
    print("-" * 50)
    
    # Consumer staples stocks (often cointegrated)
    symbols = ['PEP', 'KO', 'PG', 'CL', 'KMB', 'GIS', 'K', 'CPB']
    
    pairs = find_cointegrated_pairs(symbols, '2020-01-01', '2024-01-01')
    
    print(f"\nFound {len(pairs)} cointegrated pairs:")
    for symbol1, symbol2, pvalue in pairs[:5]:  # Show top 5
        print(f"{symbol1}-{symbol2}: p-value = {pvalue:.4f}")