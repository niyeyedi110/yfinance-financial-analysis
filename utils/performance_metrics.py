"""
Financial performance metrics calculations.
Includes Sharpe ratio, Sortino ratio, Calmar ratio, and more.
"""
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from scipy import stats


def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Series of prices
        method: 'simple' or 'log' returns
        
    Returns:
        Series of returns
    """
    if method == 'simple':
        return prices.pct_change().dropna()
    elif method == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Method must be 'simple' or 'log'")


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year (252 for daily, 52 for weekly, 12 for monthly)
        
    Returns:
        Annualized return
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    
    if years == 0:
        return 0
    
    return (1 + total_return) ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility (standard deviation).
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, 
                risk_free_rate: float = 0.0,
                periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if returns.std() == 0:
        return 0
    
    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()


def sortino_ratio(returns: pd.Series,
                 target_return: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).
    
    Args:
        returns: Series of returns
        target_return: Target return (annual)
        periods_per_year: Number of periods in a year
        
    Returns:
        Sortino ratio
    """
    target_return_per_period = target_return / periods_per_year
    excess_returns = returns - target_return_per_period
    
    # Downside returns only
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_deviation = np.sqrt((downside_returns ** 2).mean())
    
    if downside_deviation == 0:
        return 0
    
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_deviation


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Calmar ratio
    """
    annual_return = annualized_return(returns, periods_per_year)
    max_dd = max_drawdown(returns)[0]
    
    if max_dd == 0:
        return np.inf
    
    return annual_return / abs(max_dd)


def max_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and dates.
    
    Args:
        returns: Series of returns
        
    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    
    if max_dd == 0:
        return 0, returns.index[0], returns.index[0]
    
    # Find the peak and trough dates
    end_idx = drawdown.idxmin()
    start_idx = cumulative[:end_idx].idxmax()
    
    return max_dd, start_idx, end_idx


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Calculate Omega ratio.
    
    Args:
        returns: Series of returns
        threshold: Threshold return
        
    Returns:
        Omega ratio
    """
    excess_returns = returns - threshold
    positive_sum = excess_returns[excess_returns > 0].sum()
    negative_sum = abs(excess_returns[excess_returns < 0].sum())
    
    if negative_sum == 0:
        return np.inf
    
    return positive_sum / negative_sum


def information_ratio(returns: pd.Series, 
                     benchmark_returns: pd.Series,
                     periods_per_year: int = 252) -> float:
    """
    Calculate Information ratio.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Information ratio
    """
    active_returns = returns - benchmark_returns
    
    if active_returns.std() == 0:
        return 0
    
    return np.sqrt(periods_per_year) * active_returns.mean() / active_returns.std()


def treynor_ratio(returns: pd.Series,
                 market_returns: pd.Series,
                 risk_free_rate: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """
    Calculate Treynor ratio.
    
    Args:
        returns: Portfolio returns
        market_returns: Market returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Treynor ratio
    """
    # Calculate beta
    covariance = np.cov(returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    
    if market_variance == 0:
        return 0
    
    beta = covariance / market_variance
    
    if beta == 0:
        return 0
    
    # Calculate excess return
    portfolio_return = annualized_return(returns, periods_per_year)
    
    return (portfolio_return - risk_free_rate) / beta


def var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        VaR (as a positive number)
    """
    return -np.percentile(returns, (1 - confidence_level) * 100)


def cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level
        
    Returns:
        CVaR (as a positive number)
    """
    var_threshold = var(returns, confidence_level)
    conditional_returns = returns[returns <= -var_threshold]
    
    if len(conditional_returns) == 0:
        return var_threshold
    
    return -conditional_returns.mean()


def skewness(returns: pd.Series) -> float:
    """Calculate skewness of returns."""
    return stats.skew(returns)


def kurtosis(returns: pd.Series) -> float:
    """Calculate kurtosis of returns."""
    return stats.kurtosis(returns)


def downside_deviation(returns: pd.Series, 
                      target_return: float = 0.0,
                      periods_per_year: int = 252) -> float:
    """
    Calculate downside deviation.
    
    Args:
        returns: Series of returns
        target_return: Target return (annual)
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized downside deviation
    """
    target_return_per_period = target_return / periods_per_year
    downside_returns = returns[returns < target_return_per_period]
    
    if len(downside_returns) == 0:
        return 0
    
    return np.sqrt((downside_returns ** 2).mean()) * np.sqrt(periods_per_year)


def ulcer_index(returns: pd.Series) -> float:
    """
    Calculate Ulcer Index (measures downside volatility).
    
    Args:
        returns: Series of returns
        
    Returns:
        Ulcer Index
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return np.sqrt((drawdown ** 2).mean())


def calculate_all_metrics(returns: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None,
                         risk_free_rate: float = 0.02,
                         periods_per_year: int = 252) -> dict:
    """
    Calculate all available performance metrics.
    
    Args:
        returns: Series of returns
        benchmark_returns: Optional benchmark returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {
        'total_return': (1 + returns).prod() - 1,
        'annualized_return': annualized_return(returns, periods_per_year),
        'annualized_volatility': annualized_volatility(returns, periods_per_year),
        'sharpe_ratio': sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': sortino_ratio(returns, 0, periods_per_year),
        'calmar_ratio': calmar_ratio(returns, periods_per_year),
        'max_drawdown': max_drawdown(returns)[0],
        'omega_ratio': omega_ratio(returns),
        'var_95': var(returns, 0.95),
        'cvar_95': cvar(returns, 0.95),
        'skewness': skewness(returns),
        'kurtosis': kurtosis(returns),
        'downside_deviation': downside_deviation(returns, 0, periods_per_year),
        'ulcer_index': ulcer_index(returns)
    }
    
    if benchmark_returns is not None:
        metrics['information_ratio'] = information_ratio(returns, benchmark_returns, periods_per_year)
        metrics['treynor_ratio'] = treynor_ratio(returns, benchmark_returns, risk_free_rate, periods_per_year)
    
    return metrics


def performance_summary(returns: pd.Series,
                       benchmark_returns: Optional[pd.Series] = None,
                       risk_free_rate: float = 0.02,
                       periods_per_year: int = 252) -> pd.DataFrame:
    """
    Create a formatted performance summary.
    
    Args:
        returns: Series of returns
        benchmark_returns: Optional benchmark returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        DataFrame with formatted metrics
    """
    metrics = calculate_all_metrics(returns, benchmark_returns, risk_free_rate, periods_per_year)
    
    # Format metrics for display
    df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    
    # Add formatting
    percentage_metrics = [
        'total_return', 'annualized_return', 'annualized_volatility',
        'max_drawdown', 'var_95', 'cvar_95', 'downside_deviation'
    ]
    
    for metric in percentage_metrics:
        if metric in df.index:
            df.loc[metric, 'Value'] = f"{df.loc[metric, 'Value']:.2%}"
    
    # Format ratios to 2 decimal places
    ratio_metrics = [
        'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'omega_ratio',
        'information_ratio', 'treynor_ratio', 'skewness', 'kurtosis', 'ulcer_index'
    ]
    
    for metric in ratio_metrics:
        if metric in df.index:
            df.loc[metric, 'Value'] = f"{df.loc[metric, 'Value']:.2f}"
    
    return df