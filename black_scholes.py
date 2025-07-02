"""
Black-Scholes Option Pricing Model Implementation
Includes pricing functions and Greeks calculations
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple, Dict


class BlackScholes:
    """Black-Scholes option pricing model with Greeks calculations"""
    
    def __init__(self):
        """Initialize Black-Scholes calculator"""
        pass
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter for Black-Scholes formula"""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter for Black-Scholes formula"""
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        return d1_val - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European call option price using Black-Scholes formula
        
        Parameters:
        -----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility (annualized)
            
        Returns:
        --------
        float : Call option price
        """
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        d2_val = BlackScholes.d2(S, K, T, r, sigma)
        
        call_price = S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)
        return call_price
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European put option price using Black-Scholes formula
        
        Parameters:
        -----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility (annualized)
            
        Returns:
        --------
        float : Put option price
        """
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        d2_val = BlackScholes.d2(S, K, T, r, sigma)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)
        return put_price
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                        option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks for an option
        
        Parameters:
        -----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility (annualized)
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        dict : Dictionary containing all Greeks
        """
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        d2_val = BlackScholes.d2(S, K, T, r, sigma)
        
        greeks = {}
        
        # Delta
        if option_type == 'call':
            greeks['delta'] = norm.cdf(d1_val)
        else:
            greeks['delta'] = -norm.cdf(-d1_val)
        
        # Gamma (same for calls and puts)
        greeks['gamma'] = norm.pdf(d1_val) / (S * sigma * np.sqrt(T))
        
        # Theta
        term1 = -S * norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
        if option_type == 'call':
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2_val)
            greeks['theta'] = (term1 + term2) / 365  # Convert to daily theta
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2_val)
            greeks['theta'] = (term1 + term2) / 365  # Convert to daily theta
        
        # Vega (same for calls and puts)
        greeks['vega'] = S * norm.pdf(d1_val) * np.sqrt(T) / 100  # Scaled by 100
        
        # Rho
        if option_type == 'call':
            greeks['rho'] = K * T * np.exp(-r * T) * norm.cdf(d2_val) / 100  # Scaled by 100
        else:
            greeks['rho'] = -K * T * np.exp(-r * T) * norm.cdf(-d2_val) / 100  # Scaled by 100
        
        return greeks
    
    @staticmethod
    def plot_option_payoff(S_range: np.ndarray, K: float, premium: float, 
                          option_type: str = 'call', position: str = 'long') -> None:
        """
        Plot option payoff diagram
        
        Parameters:
        -----------
        S_range : np.ndarray
            Range of stock prices for x-axis
        K : float
            Strike price
        premium : float
            Option premium paid/received
        option_type : str
            'call' or 'put'
        position : str
            'long' or 'short'
        """
        if option_type == 'call':
            if position == 'long':
                payoff = np.maximum(S_range - K, 0) - premium
            else:
                payoff = premium - np.maximum(S_range - K, 0)
        else:  # put
            if position == 'long':
                payoff = np.maximum(K - S_range, 0) - premium
            else:
                payoff = premium - np.maximum(K - S_range, 0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(S_range, payoff, linewidth=2, label=f'{position.capitalize()} {option_type.capitalize()}')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=K, color='red', linestyle='--', alpha=0.5, label=f'Strike = {K}')
        
        # Highlight profit/loss regions
        plt.fill_between(S_range, payoff, 0, where=(payoff > 0), 
                        color='green', alpha=0.3, label='Profit')
        plt.fill_between(S_range, payoff, 0, where=(payoff <= 0), 
                        color='red', alpha=0.3, label='Loss')
        
        plt.xlabel('Stock Price at Expiration', fontsize=12)
        plt.ylabel('Profit/Loss', fontsize=12)
        plt.title(f'{position.capitalize()} {option_type.capitalize()} Option Payoff', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_greeks(S_range: np.ndarray, K: float, T: float, r: float, 
                   sigma: float, option_type: str = 'call') -> None:
        """
        Plot all Greeks as functions of stock price
        
        Parameters:
        -----------
        S_range : np.ndarray
            Range of stock prices
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
        """
        # Calculate Greeks for each stock price
        greeks_data = {
            'delta': [],
            'gamma': [],
            'theta': [],
            'vega': [],
            'rho': []
        }
        
        for S in S_range:
            greeks = BlackScholes.calculate_greeks(S, K, T, r, sigma, option_type)
            for greek, value in greeks.items():
                greeks_data[greek].append(value)
        
        # Create subplots for each Greek
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        greek_names = ['delta', 'gamma', 'theta', 'vega', 'rho']
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, (greek, color) in enumerate(zip(greek_names, colors)):
            ax = axes[i]
            ax.plot(S_range, greeks_data[greek], color=color, linewidth=2)
            ax.axvline(x=K, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Stock Price', fontsize=10)
            ax.set_ylabel(greek.capitalize(), fontsize=10)
            ax.set_title(f'{greek.capitalize()} vs Stock Price', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # Remove the last empty subplot
        fig.delaxes(axes[5])
        
        plt.suptitle(f'{option_type.capitalize()} Option Greeks', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_option_chain(S: float, strikes: np.ndarray, T: float, 
                           r: float, sigma: float) -> pd.DataFrame:
        """
        Create an option chain with prices and Greeks for multiple strikes
        
        Parameters:
        -----------
        S : float
            Current stock price
        strikes : np.ndarray
            Array of strike prices
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
            
        Returns:
        --------
        pd.DataFrame : Option chain with prices and Greeks
        """
        data = []
        
        for K in strikes:
            # Calculate call option data
            call_price = BlackScholes.call_price(S, K, T, r, sigma)
            call_greeks = BlackScholes.calculate_greeks(S, K, T, r, sigma, 'call')
            
            # Calculate put option data
            put_price = BlackScholes.put_price(S, K, T, r, sigma)
            put_greeks = BlackScholes.calculate_greeks(S, K, T, r, sigma, 'put')
            
            row = {
                'Strike': K,
                'Call_Price': call_price,
                'Put_Price': put_price,
                'Call_Delta': call_greeks['delta'],
                'Put_Delta': put_greeks['delta'],
                'Gamma': call_greeks['gamma'],  # Same for calls and puts
                'Call_Theta': call_greeks['theta'],
                'Put_Theta': put_greeks['theta'],
                'Vega': call_greeks['vega'],  # Same for calls and puts
                'Call_Rho': call_greeks['rho'],
                'Put_Rho': put_greeks['rho']
            }
            data.append(row)
        
        return pd.DataFrame(data)


# Example usage and testing
if __name__ == "__main__":
    # Example parameters
    S = 100  # Current stock price
    K = 105  # Strike price
    T = 0.25  # 3 months to expiration
    r = 0.05  # 5% risk-free rate
    sigma = 0.3  # 30% volatility
    
    # Create Black-Scholes instance
    bs = BlackScholes()
    
    # Calculate option prices
    call_price = bs.call_price(S, K, T, r, sigma)
    put_price = bs.put_price(S, K, T, r, sigma)
    
    print(f"Black-Scholes Option Prices:")
    print(f"Call Price: ${call_price:.2f}")
    print(f"Put Price: ${put_price:.2f}")
    
    # Calculate Greeks
    call_greeks = bs.calculate_greeks(S, K, T, r, sigma, 'call')
    put_greeks = bs.calculate_greeks(S, K, T, r, sigma, 'put')
    
    print(f"\nCall Option Greeks:")
    for greek, value in call_greeks.items():
        print(f"{greek.capitalize()}: {value:.4f}")
    
    print(f"\nPut Option Greeks:")
    for greek, value in put_greeks.items():
        print(f"{greek.capitalize()}: {value:.4f}")
    
    # Create option chain
    strikes = np.arange(85, 116, 5)
    option_chain = bs.create_option_chain(S, strikes, T, r, sigma)
    print(f"\nOption Chain:")
    print(option_chain.round(4))
    
    # Plot examples
    S_range = np.linspace(80, 120, 100)
    
    # Plot option payoff
    bs.plot_option_payoff(S_range, K, call_price, 'call', 'long')
    
    # Plot Greeks
    bs.plot_greeks(S_range, K, T, r, sigma, 'call')