"""
Implied Volatility Calculation and Volatility Smile Analysis
Includes various IV calculation methods and volatility surface visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, brentq
from scipy.interpolate import griddata
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import warnings
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from black_scholes import BlackScholes


class ImpliedVolatility:
    """Implied volatility calculations and analysis"""
    
    def __init__(self):
        """Initialize IV calculator"""
        self.bs = BlackScholes()
    
    def calculate_iv_newton_raphson(self, option_price: float, S: float, K: float, 
                                   T: float, r: float, option_type: str = 'call',
                                   initial_guess: float = 0.3, max_iterations: int = 100,
                                   tolerance: float = 1e-6) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Parameters:
        -----------
        option_price : float
            Market price of the option
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        option_type : str
            'call' or 'put'
        initial_guess : float
            Initial volatility guess
        max_iterations : int
            Maximum iterations
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        float or None : Implied volatility if converged, None otherwise
        """
        sigma = initial_guess
        
        for i in range(max_iterations):
            # Calculate option price and vega
            if option_type == 'call':
                price = self.bs.call_price(S, K, T, r, sigma)
            else:
                price = self.bs.put_price(S, K, T, r, sigma)
            
            # Calculate vega
            d1 = self.bs.d1(S, K, T, r, sigma)
            vega = S * norm.pdf(d1) * np.sqrt(T)
            
            # Check for convergence
            price_diff = option_price - price
            if abs(price_diff) < tolerance:
                return sigma
            
            # Avoid division by zero
            if vega < 1e-10:
                return None
            
            # Newton-Raphson update
            sigma = sigma + price_diff / vega
            
            # Ensure sigma stays positive
            if sigma <= 0:
                return None
        
        # Did not converge
        warnings.warn(f"Newton-Raphson did not converge after {max_iterations} iterations")
        return None
    
    def calculate_iv_bisection(self, option_price: float, S: float, K: float, 
                              T: float, r: float, option_type: str = 'call',
                              vol_min: float = 0.001, vol_max: float = 5.0,
                              tolerance: float = 1e-6) -> Optional[float]:
        """
        Calculate implied volatility using bisection method
        
        Parameters:
        -----------
        option_price : float
            Market price of the option
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        option_type : str
            'call' or 'put'
        vol_min : float
            Minimum volatility bound
        vol_max : float
            Maximum volatility bound
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        float or None : Implied volatility if found, None otherwise
        """
        def objective(sigma):
            if option_type == 'call':
                return self.bs.call_price(S, K, T, r, sigma) - option_price
            else:
                return self.bs.put_price(S, K, T, r, sigma) - option_price
        
        try:
            # Check if root exists in the interval
            if objective(vol_min) * objective(vol_max) > 0:
                # Try to expand the search range
                if abs(objective(vol_min)) < abs(objective(vol_max)):
                    vol_min = vol_min / 2
                else:
                    vol_max = vol_max * 2
                
                # Check again
                if objective(vol_min) * objective(vol_max) > 0:
                    warnings.warn("No root found in the given interval")
                    return None
            
            # Use Brent's method (improvement over bisection)
            iv = brentq(objective, vol_min, vol_max, xtol=tolerance)
            return iv
            
        except ValueError:
            warnings.warn("Bisection method failed to find implied volatility")
            return None
    
    def calculate_iv_from_prices(self, option_prices: pd.DataFrame, S: float, 
                               r: float, method: str = 'newton') -> pd.DataFrame:
        """
        Calculate implied volatilities for a DataFrame of option prices
        
        Parameters:
        -----------
        option_prices : pd.DataFrame
            DataFrame with columns: 'Strike', 'Expiry', 'Type', 'Price'
        S : float
            Current stock price
        r : float
            Risk-free rate
        method : str
            'newton' or 'bisection'
            
        Returns:
        --------
        pd.DataFrame : Original DataFrame with added 'IV' column
        """
        ivs = []
        
        for idx, row in option_prices.iterrows():
            K = row['Strike']
            T = row['Expiry']
            option_type = row['Type'].lower()
            price = row['Price']
            
            if method == 'newton':
                iv = self.calculate_iv_newton_raphson(price, S, K, T, r, option_type)
            else:
                iv = self.calculate_iv_bisection(price, S, K, T, r, option_type)
            
            ivs.append(iv)
        
        option_prices['IV'] = ivs
        return option_prices
    
    def plot_volatility_smile(self, strikes: np.ndarray, ivs: np.ndarray, 
                            S: float, expiry_label: str = '') -> None:
        """
        Plot volatility smile
        
        Parameters:
        -----------
        strikes : np.ndarray
            Strike prices
        ivs : np.ndarray
            Implied volatilities
        S : float
            Current stock price
        expiry_label : str
            Label for the expiry
        """
        # Calculate moneyness
        moneyness = strikes / S
        
        plt.figure(figsize=(10, 6))
        
        # Plot IV smile
        plt.plot(moneyness, ivs * 100, 'b-o', linewidth=2, markersize=6)
        
        # Add vertical line at ATM
        plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='ATM')
        
        # Formatting
        plt.xlabel('Moneyness (K/S)', fontsize=12)
        plt.ylabel('Implied Volatility (%)', fontsize=12)
        title = 'Volatility Smile'
        if expiry_label:
            title += f' - {expiry_label}'
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_volatility_term_structure(self, expiries: np.ndarray, 
                                     atm_ivs: np.ndarray) -> None:
        """
        Plot volatility term structure
        
        Parameters:
        -----------
        expiries : np.ndarray
            Time to expiry (in years)
        atm_ivs : np.ndarray
            ATM implied volatilities
        """
        plt.figure(figsize=(10, 6))
        
        # Plot term structure
        plt.plot(expiries * 365, atm_ivs * 100, 'b-o', linewidth=2, markersize=6)
        
        # Formatting
        plt.xlabel('Days to Expiry', fontsize=12)
        plt.ylabel('ATM Implied Volatility (%)', fontsize=12)
        plt.title('Volatility Term Structure', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_volatility_surface(self, strikes: np.ndarray, expiries: np.ndarray, 
                              iv_matrix: np.ndarray, S: float) -> None:
        """
        Plot 3D volatility surface
        
        Parameters:
        -----------
        strikes : np.ndarray
            Strike prices
        expiries : np.ndarray
            Time to expiry (in years)
        iv_matrix : np.ndarray
            Matrix of implied volatilities (expiries x strikes)
        S : float
            Current stock price
        """
        # Create mesh
        K_mesh, T_mesh = np.meshgrid(strikes, expiries)
        moneyness_mesh = K_mesh / S
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(moneyness_mesh, T_mesh * 365, iv_matrix * 100,
                             cmap=cm.viridis, linewidth=0, antialiased=True,
                             alpha=0.8)
        
        # Add contours
        ax.contour(moneyness_mesh, T_mesh * 365, iv_matrix * 100,
                  zdir='z', offset=np.min(iv_matrix * 100), cmap=cm.viridis,
                  alpha=0.5)
        
        # Labels
        ax.set_xlabel('Moneyness (K/S)', fontsize=12)
        ax.set_ylabel('Days to Expiry', fontsize=12)
        ax.set_zlabel('Implied Volatility (%)', fontsize=12)
        ax.set_title('Volatility Surface', fontsize=14)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_volatility_skew(self, strikes: np.ndarray, ivs: np.ndarray, 
                               S: float) -> Dict[str, float]:
        """
        Analyze volatility skew characteristics
        
        Parameters:
        -----------
        strikes : np.ndarray
            Strike prices
        ivs : np.ndarray
            Implied volatilities
        S : float
            Current stock price
            
        Returns:
        --------
        dict : Skew metrics
        """
        # Find ATM strike (closest to S)
        atm_idx = np.argmin(np.abs(strikes - S))
        atm_iv = ivs[atm_idx]
        
        # Calculate 25-delta put and call approximations
        # Approximate 25-delta strikes
        put_25d_strike = S * np.exp(-1.0 * atm_iv * np.sqrt(0.25))  # Rough approximation
        call_25d_strike = S * np.exp(1.0 * atm_iv * np.sqrt(0.25))
        
        # Find closest strikes
        put_25d_idx = np.argmin(np.abs(strikes - put_25d_strike))
        call_25d_idx = np.argmin(np.abs(strikes - call_25d_strike))
        
        # Calculate skew metrics
        metrics = {
            'atm_iv': atm_iv,
            'atm_strike': strikes[atm_idx],
            '25d_risk_reversal': ivs[call_25d_idx] - ivs[put_25d_idx],
            '25d_butterfly': 0.5 * (ivs[call_25d_idx] + ivs[put_25d_idx]) - atm_iv,
            'skew_slope': (ivs[put_25d_idx] - ivs[call_25d_idx]) / (strikes[put_25d_idx] - strikes[call_25d_idx])
        }
        
        # Calculate smile steepness
        if len(strikes) > 2:
            # Fit quadratic to capture smile
            moneyness = strikes / S
            poly_coeffs = np.polyfit(moneyness, ivs, 2)
            metrics['smile_curvature'] = 2 * poly_coeffs[0]  # Second derivative
        
        return metrics
    
    def fit_svi_smile(self, strikes: np.ndarray, ivs: np.ndarray, S: float, 
                     T: float) -> Dict[str, float]:
        """
        Fit SVI (Stochastic Volatility Inspired) model to volatility smile
        
        Parameters:
        -----------
        strikes : np.ndarray
            Strike prices
        ivs : np.ndarray
            Implied volatilities
        S : float
            Current stock price
        T : float
            Time to expiry
            
        Returns:
        --------
        dict : SVI parameters
        """
        # Convert to total variance and log-moneyness
        total_var = ivs**2 * T
        log_moneyness = np.log(strikes / S)
        
        # SVI parameterization: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        def svi_variance(params, k):
            a, b, rho, m, sigma = params
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        
        def objective(params):
            pred_var = svi_variance(params, log_moneyness)
            return np.sum((pred_var - total_var)**2)
        
        # Initial guess
        a0 = np.mean(total_var)
        b0 = 0.1
        rho0 = -0.5
        m0 = 0.0
        sigma0 = 0.1
        
        # Optimization with bounds
        from scipy.optimize import minimize
        bounds = [(0, None), (0, None), (-1, 1), (None, None), (0.01, None)]
        result = minimize(objective, [a0, b0, rho0, m0, sigma0], 
                         bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            a, b, rho, m, sigma = result.x
            return {
                'a': a,
                'b': b,
                'rho': rho,
                'm': m,
                'sigma': sigma,
                'fit_error': result.fun
            }
        else:
            warnings.warn("SVI fitting failed")
            return None
    
    def create_iv_report(self, option_data: pd.DataFrame, S: float, 
                        r: float) -> Dict[str, pd.DataFrame]:
        """
        Create comprehensive implied volatility analysis report
        
        Parameters:
        -----------
        option_data : pd.DataFrame
            Options data with columns: Strike, Expiry, Type, Price
        S : float
            Current stock price
        r : float
            Risk-free rate
            
        Returns:
        --------
        dict : Dictionary of analysis DataFrames
        """
        # Calculate IVs
        iv_data = self.calculate_iv_from_prices(option_data.copy(), S, r)
        
        # Group by expiry
        expiry_groups = iv_data.groupby('Expiry')
        
        # Analyze each expiry
        smile_analysis = []
        for expiry, group in expiry_groups:
            # Separate calls and puts
            calls = group[group['Type'] == 'call'].sort_values('Strike')
            puts = group[group['Type'] == 'put'].sort_values('Strike')
            
            # Combine IVs (use OTM options)
            otm_ivs = []
            otm_strikes = []
            
            # OTM puts (K < S)
            otm_puts = puts[puts['Strike'] < S]
            otm_ivs.extend(otm_puts['IV'].values)
            otm_strikes.extend(otm_puts['Strike'].values)
            
            # OTM calls (K > S)
            otm_calls = calls[calls['Strike'] > S]
            otm_ivs.extend(otm_calls['IV'].values)
            otm_strikes.extend(otm_calls['Strike'].values)
            
            if len(otm_strikes) > 3:  # Need at least 3 points
                # Sort by strike
                sort_idx = np.argsort(otm_strikes)
                otm_strikes = np.array(otm_strikes)[sort_idx]
                otm_ivs = np.array(otm_ivs)[sort_idx]
                
                # Remove NaN values
                valid_idx = ~np.isnan(otm_ivs)
                otm_strikes = otm_strikes[valid_idx]
                otm_ivs = otm_ivs[valid_idx]
                
                if len(otm_strikes) > 3:
                    # Analyze skew
                    skew_metrics = self.analyze_volatility_skew(otm_strikes, otm_ivs, S)
                    skew_metrics['expiry'] = expiry
                    smile_analysis.append(skew_metrics)
        
        # Create summary DataFrames
        results = {
            'iv_data': iv_data,
            'smile_metrics': pd.DataFrame(smile_analysis) if smile_analysis else pd.DataFrame()
        }
        
        # Add term structure analysis
        if not results['smile_metrics'].empty:
            results['term_structure'] = results['smile_metrics'][['expiry', 'atm_iv']].copy()
            results['term_structure'].sort_values('expiry', inplace=True)
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Create IV calculator
    iv_calc = ImpliedVolatility()
    
    # Example 1: Calculate single implied volatility
    print("Example 1: Single IV Calculation")
    print("=" * 50)
    
    S = 100  # Current stock price
    K = 105  # Strike price
    T = 0.25  # 3 months
    r = 0.05  # Risk-free rate
    option_price = 3.50  # Market price
    
    # Calculate IV using Newton-Raphson
    iv_newton = iv_calc.calculate_iv_newton_raphson(option_price, S, K, T, r, 'call')
    print(f"Implied Volatility (Newton-Raphson): {iv_newton:.1%}")
    
    # Calculate IV using Bisection
    iv_bisection = iv_calc.calculate_iv_bisection(option_price, S, K, T, r, 'call')
    print(f"Implied Volatility (Bisection): {iv_bisection:.1%}")
    
    # Example 2: Volatility smile analysis
    print("\nExample 2: Volatility Smile Analysis")
    print("=" * 50)
    
    # Create sample option chain
    strikes = np.arange(80, 121, 5)
    ivs = np.array([0.45, 0.38, 0.32, 0.28, 0.25, 0.24, 0.25, 0.28, 0.33])  # Smile shape
    
    # Plot volatility smile
    iv_calc.plot_volatility_smile(strikes, ivs, S, 'T = 0.25')
    
    # Analyze skew
    skew_metrics = iv_calc.analyze_volatility_skew(strikes, ivs, S)
    print("\nSkew Metrics:")
    for metric, value in skew_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Example 3: Volatility surface
    print("\nExample 3: Volatility Surface")
    print("=" * 50)
    
    # Create sample data for surface
    strikes = np.arange(80, 121, 5)
    expiries = np.array([0.08, 0.17, 0.25, 0.5, 1.0])  # 1m, 2m, 3m, 6m, 1y
    
    # Generate IV matrix with term structure and smile
    iv_matrix = np.zeros((len(expiries), len(strikes)))
    for i, T in enumerate(expiries):
        # Base IV with term structure
        base_iv = 0.20 + 0.05 * np.sqrt(T)
        
        # Add smile effect
        moneyness = strikes / S
        smile = 0.1 * (moneyness - 1)**2
        iv_matrix[i, :] = base_iv + smile
    
    # Plot surface
    iv_calc.plot_volatility_surface(strikes, expiries, iv_matrix, S)
    
    # Example 4: Complete IV analysis
    print("\nExample 4: Complete IV Analysis")
    print("=" * 50)
    
    # Create sample option data
    option_data = []
    for expiry in [0.08, 0.25, 0.5]:
        for strike in np.arange(85, 116, 5):
            # Generate synthetic option prices
            sigma = 0.25 + 0.1 * ((strike/S - 1)**2)  # Add smile
            bs = BlackScholes()
            
            call_price = bs.call_price(S, strike, expiry, r, sigma)
            put_price = bs.put_price(S, strike, expiry, r, sigma)
            
            option_data.append({
                'Strike': strike,
                'Expiry': expiry,
                'Type': 'call',
                'Price': call_price
            })
            option_data.append({
                'Strike': strike,
                'Expiry': expiry,
                'Type': 'put',
                'Price': put_price
            })
    
    option_df = pd.DataFrame(option_data)
    
    # Analyze
    analysis_results = iv_calc.create_iv_report(option_df, S, r)
    
    print("\nImplied Volatility Summary:")
    print(analysis_results['iv_data'].groupby(['Expiry', 'Type'])['IV'].describe())
    
    if not analysis_results['smile_metrics'].empty:
        print("\nSmile Metrics by Expiry:")
        print(analysis_results['smile_metrics'])