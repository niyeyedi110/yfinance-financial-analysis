"""
Monte Carlo Option Pricing Implementation
Supports European and American options, path-dependent options, and exotic options
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List
from tqdm import tqdm
import seaborn as sns


class MonteCarloOptions:
    """Monte Carlo simulation for option pricing"""
    
    def __init__(self, n_simulations: int = 10000, n_steps: int = 252, seed: Optional[int] = None):
        """
        Initialize Monte Carlo option pricer
        
        Parameters:
        -----------
        n_simulations : int
            Number of Monte Carlo simulations
        n_steps : int
            Number of time steps (default 252 for daily steps in a year)
        seed : int, optional
            Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        if seed is not None:
            np.random.seed(seed)
    
    def generate_price_paths(self, S0: float, T: float, r: float, sigma: float, 
                           dividend_yield: float = 0) -> np.ndarray:
        """
        Generate stock price paths using Geometric Brownian Motion
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility (annualized)
        dividend_yield : float
            Continuous dividend yield
            
        Returns:
        --------
        np.ndarray : Array of price paths (shape: n_simulations x n_steps+1)
        """
        dt = T / self.n_steps
        
        # Generate random shocks
        Z = np.random.standard_normal((self.n_simulations, self.n_steps))
        
        # Initialize price paths
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = S0
        
        # Generate paths using GBM
        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (r - dividend_yield - 0.5 * sigma**2) * dt + 
                sigma * np.sqrt(dt) * Z[:, t-1]
            )
        
        return paths
    
    def european_option_price(self, S0: float, K: float, T: float, r: float, 
                            sigma: float, option_type: str = 'call',
                            dividend_yield: float = 0) -> Tuple[float, float]:
        """
        Price European option using Monte Carlo simulation
        
        Parameters:
        -----------
        S0 : float
            Current stock price
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
        dividend_yield : float
            Continuous dividend yield
            
        Returns:
        --------
        tuple : (option_price, standard_error)
        """
        # Generate price paths
        paths = self.generate_price_paths(S0, T, r, sigma, dividend_yield)
        
        # Get final prices
        ST = paths[:, -1]
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        # Calculate standard error
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)
        
        return option_price, std_error
    
    def american_option_price(self, S0: float, K: float, T: float, r: float, 
                            sigma: float, option_type: str = 'put',
                            dividend_yield: float = 0) -> Tuple[float, float]:
        """
        Price American option using Longstaff-Schwartz method
        
        Parameters:
        -----------
        S0 : float
            Current stock price
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
        dividend_yield : float
            Continuous dividend yield
            
        Returns:
        --------
        tuple : (option_price, standard_error)
        """
        dt = T / self.n_steps
        
        # Generate price paths
        paths = self.generate_price_paths(S0, T, r, sigma, dividend_yield)
        
        # Initialize cash flow matrix
        cash_flows = np.zeros_like(paths)
        
        # Calculate payoff at maturity
        if option_type == 'call':
            cash_flows[:, -1] = np.maximum(paths[:, -1] - K, 0)
        else:
            cash_flows[:, -1] = np.maximum(K - paths[:, -1], 0)
        
        # Backward induction
        for t in range(self.n_steps - 1, 0, -1):
            # In-the-money paths
            if option_type == 'call':
                itm = paths[:, t] > K
                exercise_value = paths[:, t] - K
            else:
                itm = paths[:, t] < K
                exercise_value = K - paths[:, t]
            
            # Only consider ITM paths for regression
            if np.sum(itm) > 0:
                # Features for regression (stock price and its square)
                X = paths[itm, t]
                X_features = np.column_stack([np.ones_like(X), X, X**2])
                
                # Continuation value (discounted future cash flows)
                Y = np.zeros(np.sum(itm))
                for i in range(t + 1, self.n_steps + 1):
                    Y += np.exp(-r * dt * (i - t)) * cash_flows[itm, i]
                
                # Regression to estimate continuation value
                if len(Y) > 3:  # Need at least 3 points for quadratic regression
                    coeffs = np.linalg.lstsq(X_features, Y, rcond=None)[0]
                    continuation_value = X_features @ coeffs
                    
                    # Exercise decision
                    exercise = exercise_value[itm] > continuation_value
                    
                    # Update cash flows
                    exercise_indices = np.where(itm)[0][exercise]
                    cash_flows[exercise_indices, t] = exercise_value[itm][exercise]
                    cash_flows[exercise_indices, t+1:] = 0
        
        # Calculate option value
        option_values = np.zeros(self.n_simulations)
        for i in range(self.n_simulations):
            # Find first exercise time
            exercise_times = np.where(cash_flows[i, :] > 0)[0]
            if len(exercise_times) > 0:
                t_exercise = exercise_times[0]
                option_values[i] = np.exp(-r * t_exercise * dt) * cash_flows[i, t_exercise]
        
        option_price = np.mean(option_values)
        std_error = np.std(option_values) / np.sqrt(self.n_simulations)
        
        return option_price, std_error
    
    def asian_option_price(self, S0: float, K: float, T: float, r: float, 
                          sigma: float, option_type: str = 'call',
                          averaging_type: str = 'arithmetic') -> Tuple[float, float]:
        """
        Price Asian (average price) option
        
        Parameters:
        -----------
        S0 : float
            Current stock price
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
        averaging_type : str
            'arithmetic' or 'geometric'
            
        Returns:
        --------
        tuple : (option_price, standard_error)
        """
        # Generate price paths
        paths = self.generate_price_paths(S0, T, r, sigma)
        
        # Calculate average prices
        if averaging_type == 'arithmetic':
            average_prices = np.mean(paths[:, 1:], axis=1)  # Exclude initial price
        else:  # geometric
            average_prices = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(average_prices - K, 0)
        else:
            payoffs = np.maximum(K - average_prices, 0)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)
        
        return option_price, std_error
    
    def barrier_option_price(self, S0: float, K: float, T: float, r: float, 
                           sigma: float, barrier: float, option_type: str = 'call',
                           barrier_type: str = 'up-and-out') -> Tuple[float, float]:
        """
        Price barrier option
        
        Parameters:
        -----------
        S0 : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        barrier : float
            Barrier level
        option_type : str
            'call' or 'put'
        barrier_type : str
            'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
            
        Returns:
        --------
        tuple : (option_price, standard_error)
        """
        # Generate price paths
        paths = self.generate_price_paths(S0, T, r, sigma)
        
        # Check barrier conditions
        if 'up' in barrier_type:
            barrier_hit = np.any(paths > barrier, axis=1)
        else:  # down
            barrier_hit = np.any(paths < barrier, axis=1)
        
        # Calculate final payoffs
        ST = paths[:, -1]
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Apply barrier conditions
        if 'out' in barrier_type:
            payoffs[barrier_hit] = 0  # Knock-out
        else:  # in
            payoffs[~barrier_hit] = 0  # Knock-in
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)
        
        return option_price, std_error
    
    def plot_price_paths(self, S0: float, T: float, r: float, sigma: float, 
                        n_paths: int = 10, highlight_mean: bool = True) -> None:
        """
        Plot sample price paths
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        n_paths : int
            Number of paths to plot
        highlight_mean : bool
            Whether to highlight the mean path
        """
        # Generate paths
        paths = self.generate_price_paths(S0, T, r, sigma)
        
        # Time grid
        time_grid = np.linspace(0, T, self.n_steps + 1)
        
        plt.figure(figsize=(12, 6))
        
        # Plot sample paths
        for i in range(min(n_paths, self.n_simulations)):
            plt.plot(time_grid, paths[i, :], alpha=0.5, linewidth=0.8)
        
        # Plot mean path
        if highlight_mean:
            mean_path = np.mean(paths, axis=0)
            plt.plot(time_grid, mean_path, 'k-', linewidth=2, label='Mean Path')
        
        # Plot initial price
        plt.axhline(y=S0, color='red', linestyle='--', alpha=0.5, label=f'S0 = {S0}')
        
        plt.xlabel('Time (years)', fontsize=12)
        plt.ylabel('Stock Price', fontsize=12)
        plt.title(f'Monte Carlo Price Paths (σ = {sigma:.1%})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_payoff_distribution(self, S0: float, K: float, T: float, r: float, 
                               sigma: float, option_type: str = 'call') -> None:
        """
        Plot distribution of option payoffs
        
        Parameters:
        -----------
        S0 : float
            Current stock price
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
        # Generate price paths
        paths = self.generate_price_paths(S0, T, r, sigma)
        ST = paths[:, -1]
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Final price distribution
        plt.subplot(1, 2, 1)
        plt.hist(ST, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=K, color='red', linestyle='--', linewidth=2, label=f'Strike = {K}')
        plt.xlabel('Stock Price at Maturity', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of Stock Prices', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Payoff distribution
        plt.subplot(1, 2, 2)
        plt.hist(payoffs, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(x=np.mean(payoffs), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean = {np.mean(payoffs):.2f}')
        plt.xlabel(f'{option_type.capitalize()} Option Payoff', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of Option Payoffs', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def convergence_analysis(self, S0: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'call',
                           max_simulations: int = 50000, 
                           step_size: int = 1000) -> pd.DataFrame:
        """
        Analyze convergence of Monte Carlo estimates
        
        Parameters:
        -----------
        S0 : float
            Current stock price
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
        max_simulations : int
            Maximum number of simulations
        step_size : int
            Step size for convergence analysis
            
        Returns:
        --------
        pd.DataFrame : Convergence analysis results
        """
        n_sims_range = range(step_size, max_simulations + 1, step_size)
        results = []
        
        for n_sims in tqdm(n_sims_range, desc="Convergence Analysis"):
            # Temporarily change number of simulations
            original_n_sims = self.n_simulations
            self.n_simulations = n_sims
            
            # Price option
            price, std_error = self.european_option_price(S0, K, T, r, sigma, option_type)
            
            results.append({
                'n_simulations': n_sims,
                'option_price': price,
                'std_error': std_error,
                'confidence_interval_lower': price - 1.96 * std_error,
                'confidence_interval_upper': price + 1.96 * std_error
            })
            
            # Restore original setting
            self.n_simulations = original_n_sims
        
        return pd.DataFrame(results)
    
    def plot_convergence(self, convergence_df: pd.DataFrame) -> None:
        """
        Plot convergence analysis results
        
        Parameters:
        -----------
        convergence_df : pd.DataFrame
            Results from convergence_analysis method
        """
        plt.figure(figsize=(12, 6))
        
        # Plot option price with confidence intervals
        plt.fill_between(convergence_df['n_simulations'],
                        convergence_df['confidence_interval_lower'],
                        convergence_df['confidence_interval_upper'],
                        alpha=0.3, color='blue', label='95% Confidence Interval')
        
        plt.plot(convergence_df['n_simulations'], 
                convergence_df['option_price'], 
                'b-', linewidth=2, label='Option Price')
        
        # Add final converged value
        final_price = convergence_df['option_price'].iloc[-1]
        plt.axhline(y=final_price, color='red', linestyle='--', alpha=0.7,
                   label=f'Converged Price = {final_price:.3f}')
        
        plt.xlabel('Number of Simulations', fontsize=12)
        plt.ylabel('Option Price', fontsize=12)
        plt.title('Monte Carlo Convergence Analysis', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Example parameters
    S0 = 100   # Current stock price
    K = 105    # Strike price
    T = 0.25   # 3 months to expiration
    r = 0.05   # 5% risk-free rate
    sigma = 0.3  # 30% volatility
    
    # Create Monte Carlo pricer
    mc = MonteCarloOptions(n_simulations=10000, n_steps=63, seed=42)  # 63 trading days in 3 months
    
    # Price European options
    print("Monte Carlo Option Pricing Results:")
    print("=" * 50)
    
    call_price, call_se = mc.european_option_price(S0, K, T, r, sigma, 'call')
    put_price, put_se = mc.european_option_price(S0, K, T, r, sigma, 'put')
    
    print(f"European Call: ${call_price:.3f} (SE: {call_se:.4f})")
    print(f"European Put: ${put_price:.3f} (SE: {put_se:.4f})")
    
    # Price American put
    american_put_price, american_put_se = mc.american_option_price(S0, K, T, r, sigma, 'put')
    print(f"\nAmerican Put: ${american_put_price:.3f} (SE: {american_put_se:.4f})")
    print(f"Early Exercise Premium: ${american_put_price - put_price:.3f}")
    
    # Price exotic options
    print("\nExotic Options:")
    
    # Asian option
    asian_call_price, asian_call_se = mc.asian_option_price(S0, K, T, r, sigma, 'call')
    print(f"Asian Call (Arithmetic): ${asian_call_price:.3f} (SE: {asian_call_se:.4f})")
    
    # Barrier option
    barrier = 110
    barrier_call_price, barrier_call_se = mc.barrier_option_price(
        S0, K, T, r, sigma, barrier, 'call', 'up-and-out'
    )
    print(f"Barrier Call (Up-and-Out, B={barrier}): ${barrier_call_price:.3f} (SE: {barrier_call_se:.4f})")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Plot price paths
    mc.plot_price_paths(S0, T, r, sigma, n_paths=20)
    
    # Plot payoff distribution
    mc.plot_payoff_distribution(S0, K, T, r, sigma, 'call')
    
    # Convergence analysis
    print("\nPerforming convergence analysis...")
    convergence_df = mc.convergence_analysis(S0, K, T, r, sigma, 'call', 
                                           max_simulations=20000, step_size=500)
    mc.plot_convergence(convergence_df)