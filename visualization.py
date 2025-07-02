"""
Common plotting functions for financial data visualization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Union
import warnings
warnings.filterwarnings('ignore')


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9


def plot_price_series(data: Union[pd.Series, pd.DataFrame, Dict[str, pd.DataFrame]], 
                     title: str = "Stock Price History",
                     figsize: Tuple[int, int] = (12, 6),
                     show_volume: bool = True) -> plt.Figure:
    """
    Plot price series with optional volume.
    
    Args:
        data: Price data (Series, DataFrame, or dict of DataFrames)
        title: Plot title
        figsize: Figure size
        show_volume: Whether to show volume subplot
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    if show_volume and isinstance(data, (pd.DataFrame, dict)):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None
    
    # Handle different data types
    if isinstance(data, pd.Series):
        ax1.plot(data.index, data.values, label=data.name or 'Price')
    elif isinstance(data, pd.DataFrame):
        if 'Close' in data.columns:
            ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2)
        else:
            ax1.plot(data.index, data.iloc[:, 0], label=data.columns[0])
        
        if show_volume and 'Volume' in data.columns and ax2:
            ax2.bar(data.index, data['Volume'], alpha=0.3)
            ax2.set_ylabel('Volume')
    elif isinstance(data, dict):
        for ticker, df in data.items():
            if isinstance(df, pd.DataFrame) and 'Close' in df.columns:
                ax1.plot(df.index, df['Close'], label=ticker)
            elif isinstance(df, pd.Series):
                ax1.plot(df.index, df.values, label=ticker)
    
    ax1.set_title(title)
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if ax2:
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Date')
    else:
        ax1.set_xlabel('Date')
    
    plt.tight_layout()
    return fig


def plot_candlestick(data: pd.DataFrame,
                    title: str = "Candlestick Chart",
                    figsize: Tuple[int, int] = (12, 6),
                    show_volume: bool = True) -> plt.Figure:
    """
    Plot candlestick chart.
    
    Args:
        data: DataFrame with OHLC data
        title: Plot title
        figsize: Figure size
        show_volume: Whether to show volume
        
    Returns:
        Matplotlib figure
    """
    from matplotlib.patches import Rectangle
    setup_plot_style()
    
    if show_volume and 'Volume' in data.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None
    
    # Prepare data
    df = data.copy()
    df['Date_idx'] = range(len(df))
    
    # Plot candlesticks
    for idx, row in df.iterrows():
        color = 'green' if row['Close'] >= row['Open'] else 'red'
        
        # High-Low line
        ax1.plot([row['Date_idx'], row['Date_idx']], 
                [row['Low'], row['High']], 
                color='black', linewidth=0.5)
        
        # Open-Close rectangle
        height = abs(row['Close'] - row['Open'])
        bottom = min(row['Open'], row['Close'])
        
        rect = Rectangle((row['Date_idx'] - 0.3, bottom), 0.6, height,
                        facecolor=color, edgecolor='black', alpha=0.8)
        ax1.add_patch(rect)
    
    # Set x-axis labels
    num_labels = min(10, len(df))
    label_indices = np.linspace(0, len(df) - 1, num_labels, dtype=int)
    ax1.set_xticks(label_indices)
    ax1.set_xticklabels([df.index[i].strftime('%Y-%m-%d') for i in label_indices], rotation=45)
    
    ax1.set_title(title)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    
    if ax2:
        colors = ['green' if df.iloc[i]['Close'] >= df.iloc[i]['Open'] else 'red' 
                 for i in range(len(df))]
        ax2.bar(df['Date_idx'], df['Volume'], color=colors, alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_returns_distribution(returns: pd.Series,
                            title: str = "Returns Distribution",
                            figsize: Tuple[int, int] = (12, 6),
                            bins: int = 50) -> plt.Figure:
    """
    Plot returns distribution with normal distribution overlay.
    
    Args:
        returns: Series of returns
        title: Plot title
        figsize: Figure size
        bins: Number of histogram bins
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(returns, bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Fit normal distribution
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax1.plot(x, (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-0.5*((x-mu)/sigma)**2), 
             'r-', linewidth=2, label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
    
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{title} - Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title(f'{title} - Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(data: pd.DataFrame,
                          title: str = "Correlation Matrix",
                          figsize: Tuple[int, int] = (10, 8),
                          annot: bool = True) -> plt.Figure:
    """
    Plot correlation matrix heatmap.
    
    Args:
        data: DataFrame with multiple series
        title: Plot title
        figsize: Figure size
        annot: Whether to annotate cells with values
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate correlation matrix
    corr = data.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=annot, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                fmt='.2f' if annot else None, ax=ax)
    
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_rolling_metrics(returns: pd.Series,
                        window: int = 252,
                        title: str = "Rolling Performance Metrics",
                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot rolling performance metrics.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    
    # Rolling returns
    rolling_returns = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
    axes[0].plot(rolling_returns.index, rolling_returns.values)
    axes[0].set_title(f'Rolling {window}-Period Returns')
    axes[0].set_ylabel('Returns')
    axes[0].grid(True, alpha=0.3)
    
    # Rolling volatility
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    axes[1].plot(rolling_vol.index, rolling_vol.values)
    axes[1].set_title(f'Rolling {window}-Period Volatility (Annualized)')
    axes[1].set_ylabel('Volatility')
    axes[1].grid(True, alpha=0.3)
    
    # Rolling Sharpe ratio
    rolling_sharpe = returns.rolling(window).apply(
        lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else 0
    )
    axes[2].plot(rolling_sharpe.index, rolling_sharpe.values)
    axes[2].set_title(f'Rolling {window}-Period Sharpe Ratio')
    axes[2].set_ylabel('Sharpe Ratio')
    axes[2].grid(True, alpha=0.3)
    
    # Rolling drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    axes[3].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    axes[3].set_title('Drawdown')
    axes[3].set_ylabel('Drawdown %')
    axes[3].grid(True, alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_efficient_frontier(returns_df: pd.DataFrame,
                          n_portfolios: int = 10000,
                          risk_free_rate: float = 0.02,
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot efficient frontier for portfolio optimization.
    
    Args:
        returns_df: DataFrame with returns for multiple assets
        n_portfolios: Number of random portfolios to generate
        risk_free_rate: Annual risk-free rate
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate returns and covariance
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    
    # Generate random portfolios
    n_assets = len(returns_df.columns)
    results = np.zeros((3, n_portfolios))
    
    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(weights * mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_vol
        results[2, i] = sharpe_ratio
    
    # Create scatter plot
    scatter = ax.scatter(results[1], results[0], c=results[2], cmap='viridis', alpha=0.5)
    ax.set_xlabel('Volatility (Annual)')
    ax.set_ylabel('Return (Annual)')
    ax.set_title('Efficient Frontier')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio')
    
    # Mark maximum Sharpe ratio portfolio
    max_sharpe_idx = np.argmax(results[2])
    ax.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], 
              c='red', s=100, edgecolors='black', marker='*',
              label=f'Max Sharpe ({results[2, max_sharpe_idx]:.2f})')
    
    # Mark minimum variance portfolio
    min_vol_idx = np.argmin(results[1])
    ax.scatter(results[1, min_vol_idx], results[0, min_vol_idx],
              c='blue', s=100, edgecolors='black', marker='*',
              label='Min Variance')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_performance_comparison(returns_dict: Dict[str, pd.Series],
                              title: str = "Performance Comparison",
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Compare performance of multiple strategies/assets.
    
    Args:
        returns_dict: Dictionary mapping names to return series
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    
    # Cumulative returns
    for name, returns in returns_dict.items():
        cumulative = (1 + returns).cumprod()
        axes[0].plot(cumulative.index, cumulative.values, label=name)
    axes[0].set_title('Cumulative Returns')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Underwater plot (drawdown)
    for name, returns in returns_dict.items():
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, label=name)
    axes[1].set_title('Drawdown')
    axes[1].set_ylabel('Drawdown %')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Rolling volatility comparison
    for name, returns in returns_dict.items():
        rolling_vol = returns.rolling(63).std() * np.sqrt(252)  # 63 days = ~3 months
        axes[2].plot(rolling_vol.index, rolling_vol.values, label=name)
    axes[2].set_title('Rolling 3-Month Volatility')
    axes[2].set_ylabel('Annualized Volatility')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Return distribution comparison
    positions = list(range(1, len(returns_dict) + 1))
    data_to_plot = [returns.values for returns in returns_dict.values()]
    
    bp = axes[3].boxplot(data_to_plot, positions=positions, labels=list(returns_dict.keys()))
    axes[3].set_title('Return Distribution')
    axes[3].set_ylabel('Daily Returns')
    axes[3].grid(True, alpha=0.3)
    
    # Rotate x-axis labels if needed
    if len(returns_dict) > 3:
        axes[3].set_xticklabels(list(returns_dict.keys()), rotation=45, ha='right')
    
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_technical_indicators(data: pd.DataFrame,
                            indicators: List[str] = ['SMA', 'EMA', 'BB', 'RSI'],
                            sma_periods: List[int] = [20, 50],
                            ema_periods: List[int] = [12, 26],
                            bb_period: int = 20,
                            rsi_period: int = 14,
                            figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Plot price with technical indicators.
    
    Args:
        data: DataFrame with OHLC data
        indicators: List of indicators to plot
        sma_periods: Simple moving average periods
        ema_periods: Exponential moving average periods
        bb_period: Bollinger bands period
        rsi_period: RSI period
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    # Determine number of subplots
    n_subplots = 1  # Price plot
    if 'RSI' in indicators:
        n_subplots += 1
    if 'Volume' in data.columns:
        n_subplots += 1
    
    # Create subplots
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    if n_subplots == 1:
        axes = [axes]
    
    # Price plot
    ax_idx = 0
    axes[ax_idx].plot(data.index, data['Close'], label='Close', linewidth=2, color='black')
    
    # Simple Moving Averages
    if 'SMA' in indicators:
        for period in sma_periods:
            sma = data['Close'].rolling(window=period).mean()
            axes[ax_idx].plot(data.index, sma, label=f'SMA {period}', alpha=0.7)
    
    # Exponential Moving Averages
    if 'EMA' in indicators:
        for period in ema_periods:
            ema = data['Close'].ewm(span=period, adjust=False).mean()
            axes[ax_idx].plot(data.index, ema, label=f'EMA {period}', alpha=0.7)
    
    # Bollinger Bands
    if 'BB' in indicators:
        sma = data['Close'].rolling(window=bb_period).mean()
        std = data['Close'].rolling(window=bb_period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        axes[ax_idx].fill_between(data.index, upper_band, lower_band, alpha=0.1, color='gray')
        axes[ax_idx].plot(data.index, upper_band, '--', color='gray', alpha=0.5, label='BB Upper')
        axes[ax_idx].plot(data.index, lower_band, '--', color='gray', alpha=0.5, label='BB Lower')
    
    axes[ax_idx].set_ylabel('Price')
    axes[ax_idx].legend(loc='upper left')
    axes[ax_idx].grid(True, alpha=0.3)
    axes[ax_idx].set_title('Price and Technical Indicators')
    
    # RSI plot
    if 'RSI' in indicators:
        ax_idx += 1
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        axes[ax_idx].plot(data.index, rsi, label='RSI', color='purple')
        axes[ax_idx].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        axes[ax_idx].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        axes[ax_idx].set_ylabel('RSI')
        axes[ax_idx].set_ylim(0, 100)
        axes[ax_idx].legend(loc='upper left')
        axes[ax_idx].grid(True, alpha=0.3)
    
    # Volume plot
    if 'Volume' in data.columns:
        ax_idx += 1
        axes[ax_idx].bar(data.index, data['Volume'], alpha=0.3)
        axes[ax_idx].set_ylabel('Volume')
        axes[ax_idx].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    return fig