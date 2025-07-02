"""
Earnings Day Volatility Analysis - Verification of Goldman Sachs Research
========================================================================

This script analyzes earnings day moves for S&P 500 stocks and compares them
with the findings from the Goldman Sachs Weekly Options Watch report.

Key findings from the report to verify:
- Average stock moved +/-4.4% on earnings day
- Options-implied moves were +/-7.1%
- Average stock was up +0.4% on earnings day
- Non-earnings day moves averaged +/-1.7%
- Earnings to non-earnings day move ratio below 2.6x
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class EarningsVolatilityAnalyzer:
    def __init__(self):
        self.earnings_cache = {}
        
    def get_sp500_tickers(self):
        """Get S&P 500 tickers"""
        # Using a subset for demonstration - in production, use full S&P 500 list
        # Full list can be obtained from Wikipedia or other sources
        sp500_sample = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'JPM', 'JNJ', 'V', 'PG', 'XOM', 'UNH', 'HD', 'DIS', 'MA', 'AVGO',
            'PFE', 'CVX', 'LLY', 'PEP', 'ABBV', 'KO', 'BAC', 'MRK', 'WMT', 'TMO',
            'COST', 'ORCL', 'CSCO', 'ACN', 'ADBE', 'CRM', 'ABT', 'NKE', 'WFC',
            'TXN', 'MCD', 'NFLX', 'VZ', 'CMCSA', 'PM', 'INTC', 'T', 'RTX',
            'NEE', 'HON', 'IBM', 'QCOM', 'MS', 'GS', 'CAT', 'BA', 'MMM', 'GE'
        ]
        return sp500_sample
    
    def get_earnings_dates(self, ticker, start_date, end_date):
        """Get earnings dates for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            # Get earnings dates from calendar
            earnings_dates = stock.earnings_dates
            
            if earnings_dates is not None and len(earnings_dates) > 0:
                # Filter for date range
                mask = (earnings_dates.index >= start_date) & (earnings_dates.index <= end_date)
                return earnings_dates[mask].index.tolist()
            else:
                # Fallback: estimate quarterly earnings (every 3 months)
                return self.estimate_earnings_dates(ticker, start_date, end_date)
        except:
            return self.estimate_earnings_dates(ticker, start_date, end_date)
    
    def estimate_earnings_dates(self, ticker, start_date, end_date):
        """Estimate earnings dates based on quarterly pattern"""
        # Most companies report quarterly
        dates = []
        current = start_date
        while current <= end_date:
            # Assume earnings in months 1, 4, 7, 10
            if current.month in [1, 4, 7, 10]:
                dates.append(current + timedelta(days=20))  # Around 20th of month
            current = current + timedelta(days=30)
        return dates
    
    def calculate_earnings_day_move(self, ticker, earnings_date, price_data):
        """Calculate the stock move on earnings day"""
        try:
            # Find the earnings date in price data
            if earnings_date in price_data.index:
                idx = price_data.index.get_loc(earnings_date)
                if idx > 0:
                    prev_close = price_data['Close'].iloc[idx-1]
                    curr_close = price_data['Close'].iloc[idx]
                    return ((curr_close - prev_close) / prev_close) * 100
            return None
        except:
            return None
    
    def calculate_non_earnings_moves(self, ticker, earnings_dates, price_data):
        """Calculate average daily moves on non-earnings days"""
        # Create mask for non-earnings days
        is_earnings_day = price_data.index.isin(earnings_dates)
        non_earnings_data = price_data[~is_earnings_day]
        
        # Calculate daily returns
        returns = non_earnings_data['Close'].pct_change().dropna()
        return returns.abs().mean() * 100
    
    def analyze_single_stock(self, ticker, start_date, end_date):
        """Analyze earnings volatility for a single stock"""
        # Get price data
        stock = yf.Ticker(ticker)
        price_data = stock.history(start=start_date, end=end_date)
        
        if len(price_data) == 0:
            return None
        
        # Get earnings dates
        earnings_dates = self.get_earnings_dates(ticker, start_date, end_date)
        
        if len(earnings_dates) == 0:
            return None
        
        # Calculate earnings day moves
        earnings_moves = []
        for earnings_date in earnings_dates:
            move = self.calculate_earnings_day_move(ticker, earnings_date, price_data)
            if move is not None:
                earnings_moves.append(move)
        
        if len(earnings_moves) == 0:
            return None
        
        # Calculate non-earnings day moves
        non_earnings_avg = self.calculate_non_earnings_moves(ticker, earnings_dates, price_data)
        
        return {
            'ticker': ticker,
            'earnings_moves': earnings_moves,
            'avg_earnings_move': np.mean(np.abs(earnings_moves)),
            'avg_earnings_direction': np.mean(earnings_moves),
            'non_earnings_avg_move': non_earnings_avg,
            'earnings_to_non_earnings_ratio': np.mean(np.abs(earnings_moves)) / non_earnings_avg if non_earnings_avg > 0 else 0
        }
    
    def analyze_sp500_earnings(self, start_date=None, end_date=None):
        """Analyze earnings volatility for S&P 500 stocks"""
        if start_date is None:
            # Default to last quarter
            end_date = datetime.now()
            start_date = end_date - timedelta(days=120)
        
        tickers = self.get_sp500_tickers()
        results = []
        
        print(f"Analyzing {len(tickers)} stocks from {start_date.date()} to {end_date.date()}")
        
        for ticker in tqdm(tickers, desc="Analyzing stocks"):
            result = self.analyze_single_stock(ticker, start_date, end_date)
            if result is not None:
                results.append(result)
        
        return pd.DataFrame(results)
    
    def generate_report(self, results_df):
        """Generate analysis report comparing with Goldman Sachs findings"""
        print("\n" + "="*80)
        print("EARNINGS DAY VOLATILITY ANALYSIS REPORT")
        print("="*80)
        
        # Calculate aggregate statistics
        all_earnings_moves = []
        for moves in results_df['earnings_moves']:
            all_earnings_moves.extend(moves)
        
        avg_abs_move = np.mean(np.abs(all_earnings_moves))
        avg_direction = np.mean(all_earnings_moves)
        avg_non_earnings = results_df['non_earnings_avg_move'].mean()
        avg_ratio = results_df['earnings_to_non_earnings_ratio'].mean()
        
        print(f"\nSample Size: {len(results_df)} stocks analyzed")
        print(f"Total Earnings Events: {len(all_earnings_moves)}")
        
        print("\n1. EARNINGS DAY MOVES:")
        print(f"   Average Absolute Move: {avg_abs_move:.2f}% (GS Report: 4.4%)")
        print(f"   Average Directional Move: {avg_direction:+.2f}% (GS Report: +0.4%)")
        print(f"   Positive Moves: {sum(1 for m in all_earnings_moves if m > 0)} ({sum(1 for m in all_earnings_moves if m > 0)/len(all_earnings_moves)*100:.1f}%)")
        print(f"   Negative Moves: {sum(1 for m in all_earnings_moves if m < 0)} ({sum(1 for m in all_earnings_moves if m < 0)/len(all_earnings_moves)*100:.1f}%)")
        
        print("\n2. NON-EARNINGS DAY MOVES:")
        print(f"   Average Daily Move: {avg_non_earnings:.2f}% (GS Report: 1.7%)")
        
        print("\n3. EARNINGS TO NON-EARNINGS RATIO:")
        print(f"   Average Ratio: {avg_ratio:.2f}x (GS Report: Below 2.6x)")
        
        print("\n4. SECTOR/STOCK HIGHLIGHTS:")
        top_movers = results_df.nlargest(5, 'avg_earnings_move')
        print("   Highest Earnings Volatility:")
        for _, row in top_movers.iterrows():
            print(f"     {row['ticker']}: {row['avg_earnings_move']:.2f}%")
        
        return {
            'avg_abs_move': avg_abs_move,
            'avg_direction': avg_direction,
            'avg_non_earnings': avg_non_earnings,
            'avg_ratio': avg_ratio,
            'all_moves': all_earnings_moves
        }
    
    def plot_analysis(self, results_df, report_stats):
        """Create visualizations of the analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribution of earnings day moves
        ax1 = axes[0, 0]
        all_moves = report_stats['all_moves']
        ax1.hist(all_moves, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax1.axvline(x=np.mean(all_moves), color='green', linestyle='-', label=f'Mean: {np.mean(all_moves):.2f}%')
        ax1.set_xlabel('Earnings Day Move (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Earnings Day Moves')
        ax1.legend()
        
        # 2. Earnings vs Non-Earnings Moves
        ax2 = axes[0, 1]
        comparison_data = pd.DataFrame({
            'Earnings Day': [report_stats['avg_abs_move']],
            'Non-Earnings Day': [report_stats['avg_non_earnings']],
            'GS Earnings': [4.4],
            'GS Non-Earnings': [1.7]
        })
        comparison_data.plot(kind='bar', ax=ax2)
        ax2.set_ylabel('Average Move (%)')
        ax2.set_title('Earnings vs Non-Earnings Day Moves')
        ax2.set_xticklabels(['Our Analysis vs GS Report'], rotation=0)
        
        # 3. Top movers
        ax3 = axes[1, 0]
        top_10 = results_df.nlargest(10, 'avg_earnings_move')
        ax3.barh(top_10['ticker'], top_10['avg_earnings_move'])
        ax3.set_xlabel('Average Earnings Day Move (%)')
        ax3.set_title('Top 10 Stocks by Earnings Volatility')
        
        # 4. Earnings to Non-Earnings Ratio Distribution
        ax4 = axes[1, 1]
        ratios = results_df['earnings_to_non_earnings_ratio']
        ax4.hist(ratios, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(x=ratios.mean(), color='red', linestyle='-', label=f'Mean: {ratios.mean():.2f}x')
        ax4.axvline(x=2.6, color='green', linestyle='--', label='GS Long-term Avg: 2.6x')
        ax4.set_xlabel('Earnings/Non-Earnings Ratio')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Earnings to Non-Earnings Ratios')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('earnings_volatility_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def analyze_specific_stocks(analyzer, tickers, start_date, end_date):
    """Analyze specific stocks mentioned in the GS report"""
    print("\n" + "="*80)
    print("ANALYSIS OF SPECIFIC STOCKS FROM GS REPORT")
    print("="*80)
    
    for ticker in tickers:
        print(f"\n{ticker}:")
        result = analyzer.analyze_single_stock(ticker, start_date, end_date)
        if result:
            print(f"  Average Earnings Move: {result['avg_earnings_move']:.2f}%")
            print(f"  Average Direction: {result['avg_earnings_direction']:+.2f}%")
            print(f"  Non-Earnings Avg: {result['non_earnings_avg_move']:.2f}%")
            print(f"  Ratio: {result['earnings_to_non_earnings_ratio']:.2f}x")
            
            # Get current implied volatility if available
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                if 'impliedVolatility' in info:
                    print(f"  Current IV: {info['impliedVolatility']*100:.1f}%")
            except:
                pass

def main():
    print("Starting Earnings Day Volatility Analysis...")
    print("Comparing with Goldman Sachs Research findings...")
    
    analyzer = EarningsVolatilityAnalyzer()
    
    # Analyze last quarter (approximately)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)
    
    # Analyze S&P 500 sample
    results_df = analyzer.analyze_sp500_earnings(start_date, end_date)
    
    if len(results_df) > 0:
        # Generate report
        report_stats = analyzer.generate_report(results_df)
        
        # Create visualizations
        analyzer.plot_analysis(results_df, report_stats)
        
        # Analyze specific stocks from GS report
        gs_stocks = ['NKE', 'STZ', 'SAM']
        analyze_specific_stocks(analyzer, gs_stocks, start_date, end_date)
        
        # Save results
        results_df.to_csv('earnings_volatility_results.csv', index=False)
        print("\nResults saved to 'earnings_volatility_results.csv'")
        
        print("\n" + "="*80)
        print("ANALYSIS LIMITATIONS:")
        print("="*80)
        print("1. Sample size limited due to data availability")
        print("2. Earnings dates are estimated for some stocks")
        print("3. Options-implied volatility data limited in yfinance")
        print("4. Analysis period may differ from GS report period")
        
    else:
        print("No data available for analysis")

if __name__ == "__main__":
    main()