"""
Goldman Sachs Options Trade Analysis
====================================

This script analyzes the specific options trades recommended in the GS report:
1. NKE straddle (26-Jun earnings)
2. STZ calls (1-Jul earnings)
3. SAM puts (24-Jul earnings)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm

class GSOptionsAnalyzer:
    def __init__(self):
        self.trades = {
            'NKE': {
                'type': 'straddle',
                'strike': 61,
                'expiry': '2025-07-03',
                'earnings': '2025-06-26',
                'premium': 5.2,
                'stock_ref': 61.42
            },
            'STZ': {
                'type': 'call',
                'strike': 167.5,
                'expiry': '2025-07-03',
                'earnings': '2025-07-01',
                'premium': 3.92,
                'stock_ref': 164.49
            },
            'SAM': {
                'type': 'put',
                'strike': 195,
                'expiry': '2025-08-25',
                'earnings': '2025-07-24',
                'premium': 13.06,
                'stock_ref': 196.79
            }
        }
    
    def fetch_current_data(self, ticker):
        """Fetch current stock data and calculate metrics"""
        stock = yf.Ticker(ticker)
        
        # Get current price
        hist = stock.history(period="1d")
        current_price = hist['Close'].iloc[-1]
        
        # Get historical data for volatility
        hist_90d = stock.history(period="3mo")
        returns = hist_90d['Close'].pct_change().dropna()
        historical_vol = returns.std() * np.sqrt(252)
        
        # Get options chain if available
        try:
            options_dates = stock.options
            if len(options_dates) > 0:
                # Get nearest expiry
                nearest_expiry = min(options_dates, key=lambda x: abs(pd.to_datetime(x) - datetime.now()))
                opt_chain = stock.option_chain(nearest_expiry)
                return {
                    'current_price': current_price,
                    'historical_vol': historical_vol,
                    'options_chain': opt_chain,
                    'has_options': True
                }
        except:
            pass
        
        return {
            'current_price': current_price,
            'historical_vol': historical_vol,
            'options_chain': None,
            'has_options': False
        }
    
    def calculate_implied_volatility(self, option_price, S, K, T, r, option_type='call'):
        """Calculate implied volatility using Newton-Raphson method"""
        def black_scholes(S, K, T, r, sigma, option_type='call'):
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == 'call':
                return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        def vega(S, K, T, r, sigma):
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            return S * norm.pdf(d1) * np.sqrt(T)
        
        # Newton-Raphson iteration
        sigma = 0.3  # Initial guess
        for _ in range(100):
            price = black_scholes(S, K, T, r, sigma, option_type)
            v = vega(S, K, T, r, sigma)
            if v < 1e-10:
                break
            sigma = sigma - (price - option_price) / v
            if sigma <= 0:
                sigma = 0.01
        
        return sigma
    
    def analyze_trade(self, ticker):
        """Analyze a specific options trade"""
        trade = self.trades[ticker]
        data = self.fetch_current_data(ticker)
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS: {ticker} {trade['type'].upper()}")
        print(f"{'='*60}")
        
        # Price movement since recommendation
        price_change = ((data['current_price'] - trade['stock_ref']) / trade['stock_ref']) * 100
        print(f"Stock Price Movement:")
        print(f"  Reference Price: ${trade['stock_ref']:.2f}")
        print(f"  Current Price: ${data['current_price']:.2f}")
        print(f"  Change: {price_change:+.2f}%")
        
        # Volatility analysis
        print(f"\nVolatility Analysis:")
        print(f"  Historical Volatility (90d): {data['historical_vol']*100:.1f}%")
        
        # Calculate days to earnings and expiry
        today = datetime.now()
        earnings_date = pd.to_datetime(trade['earnings'])
        expiry_date = pd.to_datetime(trade['expiry'])
        days_to_earnings = max(0, (earnings_date - today).days)
        days_to_expiry = max(0, (expiry_date - today).days)
        
        print(f"\nTime Analysis:")
        print(f"  Days to Earnings: {days_to_earnings}")
        print(f"  Days to Expiry: {days_to_expiry}")
        
        # Calculate implied volatility from premium
        T = days_to_expiry / 365
        if T > 0:
            if trade['type'] == 'straddle':
                # Approximate IV for ATM straddle
                approx_iv = (trade['premium'] / trade['stock_ref']) / (0.8 * np.sqrt(T))
                print(f"  Approximate Implied Vol: {approx_iv*100:.1f}%")
            else:
                # Calculate IV for single option
                iv = self.calculate_implied_volatility(
                    trade['premium'], 
                    trade['stock_ref'], 
                    trade['strike'], 
                    T, 
                    0.05,  # Risk-free rate
                    'call' if trade['type'] == 'call' else 'put'
                )
                print(f"  Implied Volatility: {iv*100:.1f}%")
        
        # P&L Analysis
        print(f"\nP&L Analysis:")
        if trade['type'] == 'straddle':
            breakeven_up = trade['strike'] + trade['premium']
            breakeven_down = trade['strike'] - trade['premium']
            print(f"  Upper Breakeven: ${breakeven_up:.2f} (+{(breakeven_up/trade['stock_ref']-1)*100:.1f}%)")
            print(f"  Lower Breakeven: ${breakeven_down:.2f} ({(breakeven_down/trade['stock_ref']-1)*100:.1f}%)")
            
            # Current P&L if exercised today
            intrinsic = abs(data['current_price'] - trade['strike'])
            current_pl = intrinsic - trade['premium']
            print(f"  Current Intrinsic Value: ${intrinsic:.2f}")
            print(f"  Current P&L (if exercised): ${current_pl:.2f}")
            
        elif trade['type'] == 'call':
            breakeven = trade['strike'] + trade['premium']
            print(f"  Breakeven: ${breakeven:.2f} (+{(breakeven/trade['stock_ref']-1)*100:.1f}%)")
            
            intrinsic = max(0, data['current_price'] - trade['strike'])
            current_pl = intrinsic - trade['premium']
            print(f"  Current Intrinsic Value: ${intrinsic:.2f}")
            print(f"  Current P&L (if exercised): ${current_pl:.2f}")
            
        else:  # put
            breakeven = trade['strike'] - trade['premium']
            print(f"  Breakeven: ${breakeven:.2f} ({(breakeven/trade['stock_ref']-1)*100:.1f}%)")
            
            intrinsic = max(0, trade['strike'] - data['current_price'])
            current_pl = intrinsic - trade['premium']
            print(f"  Current Intrinsic Value: ${intrinsic:.2f}")
            print(f"  Current P&L (if exercised): ${current_pl:.2f}")
        
        return {
            'ticker': ticker,
            'current_price': data['current_price'],
            'price_change': price_change,
            'historical_vol': data['historical_vol'],
            'days_to_earnings': days_to_earnings,
            'days_to_expiry': days_to_expiry
        }
    
    def plot_payoff_diagrams(self):
        """Plot payoff diagrams for all trades"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (ticker, trade) in enumerate(self.trades.items()):
            ax = axes[idx]
            
            # Generate price range
            price_range = np.linspace(
                trade['stock_ref'] * 0.7,
                trade['stock_ref'] * 1.3,
                100
            )
            
            # Calculate payoffs
            if trade['type'] == 'straddle':
                payoffs = np.maximum(price_range - trade['strike'], 0) + \
                         np.maximum(trade['strike'] - price_range, 0) - trade['premium']
                title = f"{ticker} Straddle"
            elif trade['type'] == 'call':
                payoffs = np.maximum(price_range - trade['strike'], 0) - trade['premium']
                title = f"{ticker} Call"
            else:  # put
                payoffs = np.maximum(trade['strike'] - price_range, 0) - trade['premium']
                title = f"{ticker} Put"
            
            # Plot
            ax.plot(price_range, payoffs, 'b-', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=trade['stock_ref'], color='r', linestyle='--', alpha=0.5, label='Reference Price')
            ax.axvline(x=trade['strike'], color='g', linestyle='--', alpha=0.5, label='Strike')
            
            # Add current price if available
            current_data = self.fetch_current_data(ticker)
            ax.axvline(x=current_data['current_price'], color='orange', linestyle='-', alpha=0.7, label='Current Price')
            
            ax.set_xlabel('Stock Price ($)')
            ax.set_ylabel('P&L ($)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gs_options_payoffs.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate summary report for all trades"""
        print("\n" + "="*80)
        print("GOLDMAN SACHS OPTIONS TRADES SUMMARY")
        print("="*80)
        print(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        results = []
        for ticker in self.trades.keys():
            result = self.analyze_trade(ticker)
            results.append(result)
        
        # Summary table
        df = pd.DataFrame(results)
        print("\nSummary Table:")
        print(df.to_string(index=False))
        
        # Risk assessment
        print("\n" + "="*80)
        print("RISK ASSESSMENT")
        print("="*80)
        
        for ticker in self.trades.keys():
            trade = self.trades[ticker]
            current_data = self.fetch_current_data(ticker)
            
            print(f"\n{ticker}:")
            
            # Volatility risk
            if current_data['historical_vol'] > 0.5:
                print("  ⚠️  High volatility stock (>50% annualized)")
            elif current_data['historical_vol'] > 0.3:
                print("  ⚡ Moderate volatility stock (30-50%)")
            else:
                print("  ✓ Low volatility stock (<30%)")
            
            # Time decay risk
            days_to_expiry = (pd.to_datetime(trade['expiry']) - datetime.now()).days
            if days_to_expiry < 7:
                print("  ⚠️  High theta risk (expires in <7 days)")
            elif days_to_expiry < 30:
                print("  ⚡ Moderate theta risk (expires in <30 days)")
            else:
                print("  ✓ Low theta risk (>30 days to expiry)")
            
            # Earnings risk
            days_to_earnings = (pd.to_datetime(trade['earnings']) - datetime.now()).days
            if days_to_earnings <= 0:
                print("  ✓ Earnings already passed")
            elif days_to_earnings < 3:
                print("  ⚠️  Earnings imminent (<3 days)")
            else:
                print(f"  📅 Earnings in {days_to_earnings} days")

def main():
    print("Goldman Sachs Options Trade Analysis")
    print("Based on Weekly Options Watch Report")
    
    analyzer = GSOptionsAnalyzer()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    # Plot payoff diagrams
    print("\nGenerating payoff diagrams...")
    analyzer.plot_payoff_diagrams()
    
    print("\nAnalysis complete!")
    print("\nDISCLAIMER: This analysis is for educational purposes only.")
    print("Options trading involves substantial risk. Past performance")
    print("does not guarantee future results.")

if __name__ == "__main__":
    main()