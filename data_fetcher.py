"""
Enhanced yfinance data fetching with caching functionality.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
from typing import Union, List, Optional, Dict
import hashlib


class DataFetcher:
    """Enhanced data fetcher with caching capabilities."""
    
    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize DataFetcher with cache directory.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_key(self, ticker: str, start: str, end: str, interval: str) -> str:
        """Generate unique cache key for the request."""
        key_string = f"{ticker}_{start}_{end}_{interval}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get full path for cache file."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _is_cache_valid(self, cache_path: str, max_age_hours: int = 24) -> bool:
        """Check if cache file exists and is not too old."""
        if not os.path.exists(cache_path):
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
        return file_age < timedelta(hours=max_age_hours)
    
    def fetch_stock_data(self, 
                        ticker: str, 
                        start: Optional[str] = None, 
                        end: Optional[str] = None,
                        period: Optional[str] = None,
                        interval: str = "1d",
                        use_cache: bool = True,
                        cache_max_age: int = 24) -> pd.DataFrame:
        """
        Fetch stock data with caching.
        
        Args:
            ticker: Stock ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            period: Period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Whether to use cached data
            cache_max_age: Maximum age of cache in hours
            
        Returns:
            DataFrame with OHLCV data
        """
        # Set default dates if not provided
        if not end:
            end = datetime.now().strftime('%Y-%m-%d')
        if not start and not period:
            start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Check cache if enabled
        if use_cache and start and end:
            cache_key = self._get_cache_key(ticker, start, end, interval)
            cache_path = self._get_cache_path(cache_key)
            
            if self._is_cache_valid(cache_path, cache_max_age):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        # Fetch data
        try:
            stock = yf.Ticker(ticker)
            if period:
                df = stock.history(period=period, interval=interval)
            else:
                df = stock.history(start=start, end=end, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Save to cache
            if use_cache and start and end:
                with open(cache_path, 'wb') as f:
                    pickle.dump(df, f)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")
    
    def fetch_multiple_stocks(self,
                            tickers: List[str],
                            start: Optional[str] = None,
                            end: Optional[str] = None,
                            period: Optional[str] = None,
                            interval: str = "1d",
                            use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            tickers: List of ticker symbols
            start: Start date
            end: End date
            period: Period to download
            interval: Data interval
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        data = {}
        for ticker in tickers:
            try:
                data[ticker] = self.fetch_stock_data(
                    ticker, start, end, period, interval, use_cache
                )
            except Exception as e:
                print(f"Warning: Failed to fetch {ticker}: {str(e)}")
        
        return data
    
    def get_stock_info(self, ticker: str) -> dict:
        """
        Get stock information and metadata.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(ticker)
            return stock.info
        except Exception as e:
            raise Exception(f"Error fetching info for {ticker}: {str(e)}")
    
    def get_dividends(self, ticker: str) -> pd.Series:
        """Get dividend history for a stock."""
        stock = yf.Ticker(ticker)
        return stock.dividends
    
    def get_splits(self, ticker: str) -> pd.Series:
        """Get stock split history."""
        stock = yf.Ticker(ticker)
        return stock.splits
    
    def get_options_chain(self, ticker: str, date: Optional[str] = None) -> tuple:
        """
        Get options chain for a stock.
        
        Args:
            ticker: Stock ticker symbol
            date: Expiration date (if None, gets next expiration)
            
        Returns:
            Tuple of (calls DataFrame, puts DataFrame)
        """
        stock = yf.Ticker(ticker)
        if date:
            opt = stock.option_chain(date)
        else:
            # Get next expiration date
            expirations = stock.options
            if expirations:
                opt = stock.option_chain(expirations[0])
            else:
                raise ValueError(f"No options available for {ticker}")
        
        return opt.calls, opt.puts
    
    def clear_cache(self, older_than_hours: Optional[int] = None):
        """
        Clear cache files.
        
        Args:
            older_than_hours: Only clear files older than this many hours
        """
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.isfile(filepath):
                if older_than_hours:
                    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_age > timedelta(hours=older_than_hours):
                        os.remove(filepath)
                else:
                    os.remove(filepath)


# Convenience functions
def fetch_stock_data(ticker: str, 
                    start: Optional[str] = None,
                    end: Optional[str] = None,
                    period: Optional[str] = None,
                    interval: str = "1d") -> pd.DataFrame:
    """Convenience function to fetch stock data."""
    fetcher = DataFetcher()
    return fetcher.fetch_stock_data(ticker, start, end, period, interval)


def fetch_multiple_stocks(tickers: List[str],
                         start: Optional[str] = None,
                         end: Optional[str] = None,
                         period: Optional[str] = None,
                         interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """Convenience function to fetch multiple stocks."""
    fetcher = DataFetcher()
    return fetcher.fetch_multiple_stocks(tickers, start, end, period, interval)