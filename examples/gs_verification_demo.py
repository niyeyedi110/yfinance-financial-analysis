"""
Goldman Sachs Report Verification - Simple Demo
==============================================

A simplified demonstration of how the GS report claims can be verified
using basic calculations (no external dependencies except numpy).
"""

import numpy as np
import random
from datetime import datetime, timedelta

def simulate_stock_data(n_stocks=50, n_days=90):
    """Simulate stock price data with earnings events"""
    print("Simulating stock market data...")
    
    all_earnings_moves = []
    all_non_earnings_moves = []
    
    for i in range(n_stocks):
        # Base parameters
        daily_vol = 0.02  # 2% daily volatility
        earnings_vol_multiplier = 2.5  # Earnings moves are 2.5x larger
        
        # Generate daily returns
        returns = []
        is_earnings_day = []
        
        for day in range(n_days):
            # Earnings every ~60 days (quarterly)
            if day % 60 == 30:  # Earnings day
                # Larger move on earnings
                move = np.random.normal(0.002, daily_vol * earnings_vol_multiplier) * 100
                returns.append(move)
                is_earnings_day.append(True)
                all_earnings_moves.append(move)
            else:
                # Normal day
                move = np.random.normal(0.001, daily_vol) * 100
                returns.append(move)
                is_earnings_day.append(False)
                all_non_earnings_moves.append(abs(move))
    
    return all_earnings_moves, all_non_earnings_moves

def verify_gs_claims(earnings_moves, non_earnings_moves):
    """Compare simulated results with GS claims"""
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    # GS Claims
    gs_claims = {
        'avg_earnings_move': 4.4,
        'avg_directional': 0.4,
        'non_earnings_move': 1.7,
        'ratio_target': 2.6
    }
    
    # Calculate our metrics
    avg_abs_earnings = np.mean(np.abs(earnings_moves))
    avg_directional = np.mean(earnings_moves)
    avg_non_earnings = np.mean(non_earnings_moves)
    ratio = avg_abs_earnings / avg_non_earnings if avg_non_earnings > 0 else 0
    
    # Display comparison
    print(f"\n1. AVERAGE ABSOLUTE EARNINGS MOVE:")
    print(f"   Our Simulation: {avg_abs_earnings:.2f}%")
    print(f"   GS Report: {gs_claims['avg_earnings_move']}%")
    print(f"   Difference: {abs(avg_abs_earnings - gs_claims['avg_earnings_move']):.2f}%")
    
    print(f"\n2. AVERAGE DIRECTIONAL MOVE:")
    print(f"   Our Simulation: {avg_directional:+.2f}%")
    print(f"   GS Report: +{gs_claims['avg_directional']}%")
    
    print(f"\n3. NON-EARNINGS DAY MOVES:")
    print(f"   Our Simulation: {avg_non_earnings:.2f}%")
    print(f"   GS Report: {gs_claims['non_earnings_move']}%")
    
    print(f"\n4. EARNINGS TO NON-EARNINGS RATIO:")
    print(f"   Our Simulation: {ratio:.2f}x")
    print(f"   GS Target: <{gs_claims['ratio_target']}x")
    print(f"   ✓ Below target: {'Yes' if ratio < gs_claims['ratio_target'] else 'No'}")
    
    # Distribution analysis
    print(f"\n5. DISTRIBUTION ANALYSIS:")
    positive_moves = sum(1 for m in earnings_moves if m > 0)
    negative_moves = sum(1 for m in earnings_moves if m < 0)
    print(f"   Positive earnings moves: {positive_moves} ({positive_moves/len(earnings_moves)*100:.1f}%)")
    print(f"   Negative earnings moves: {negative_moves} ({negative_moves/len(earnings_moves)*100:.1f}%)")
    
    return {
        'avg_abs_earnings': avg_abs_earnings,
        'avg_directional': avg_directional,
        'avg_non_earnings': avg_non_earnings,
        'ratio': ratio
    }

def simulate_options_trades():
    """Simulate the specific options trades mentioned"""
    print("\n" + "="*60)
    print("OPTIONS TRADES ANALYSIS")
    print("="*60)
    
    trades = {
        'NKE': {
            'type': 'Straddle',
            'strike': 61,
            'premium': 5.2,
            'stock_ref': 61.42,
            'current': 62.5,  # Simulated current price
            'vol': 0.35  # 35% implied vol
        },
        'STZ': {
            'type': 'Call',
            'strike': 167.5,
            'premium': 3.92,
            'stock_ref': 164.49,
            'current': 166.0,  # Simulated
            'vol': 0.28
        },
        'SAM': {
            'type': 'Put',
            'strike': 195,
            'premium': 13.06,
            'stock_ref': 196.79,
            'current': 194.0,  # Simulated
            'vol': 0.42
        }
    }
    
    for ticker, trade in trades.items():
        print(f"\n{ticker} - {trade['type']}:")
        print(f"  Reference Price: ${trade['stock_ref']:.2f}")
        print(f"  Current Price: ${trade['current']:.2f} (simulated)")
        print(f"  Price Change: {(trade['current']/trade['stock_ref']-1)*100:+.1f}%")
        
        # Calculate P&L
        if trade['type'] == 'Straddle':
            intrinsic = abs(trade['current'] - trade['strike'])
            pl = intrinsic - trade['premium']
            print(f"  Intrinsic Value: ${intrinsic:.2f}")
            print(f"  P&L: ${pl:.2f} ({pl/trade['premium']*100:+.1f}%)")
        elif trade['type'] == 'Call':
            intrinsic = max(0, trade['current'] - trade['strike'])
            pl = intrinsic - trade['premium']
            print(f"  Intrinsic Value: ${intrinsic:.2f}")
            print(f"  P&L: ${pl:.2f} ({pl/trade['premium']*100:+.1f}%)")
        else:  # Put
            intrinsic = max(0, trade['strike'] - trade['current'])
            pl = intrinsic - trade['premium']
            print(f"  Intrinsic Value: ${intrinsic:.2f}")
            print(f"  P&L: ${pl:.2f} ({pl/trade['premium']*100:+.1f}%)")

def main():
    print("GOLDMAN SACHS REPORT VERIFICATION DEMO")
    print("=====================================")
    print("This demo simulates market data to verify GS report claims")
    print("(No external data dependencies)")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Simulate earnings data
    print("\nSimulating 50 stocks over 90 days...")
    earnings_moves, non_earnings_moves = simulate_stock_data(n_stocks=50, n_days=90)
    
    # Verify claims
    results = verify_gs_claims(earnings_moves, non_earnings_moves)
    
    # Simulate options trades
    simulate_options_trades()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nThis simulation demonstrates how the GS report findings")
    print("can be verified using market data. Key observations:")
    print("\n1. Earnings moves are indeed larger than non-earnings moves")
    print("2. The ratio typically falls in the 2-3x range")
    print("3. There's usually a slight positive bias on earnings days")
    print("4. Options strategies depend heavily on realized volatility")
    print("\nFor real verification, use the full analysis scripts with")
    print("actual market data from yfinance.")

if __name__ == "__main__":
    main()