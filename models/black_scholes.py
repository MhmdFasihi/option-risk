"""Black-Scholes option pricing model and Greeks calculations."""

import numpy as np
from scipy.stats import norm
from datetime import datetime
from typing import Dict, Tuple


class BlackScholes:
    """Black-Scholes option pricing model."""
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
        """
        if T <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        if T <= 0:
            return 0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European call option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            
        Returns:
            Call option price
        """
        if T <= 0:
            return max(0, S - K)
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European put option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            
        Returns:
            Put option price
        """
        if T <= 0:
            return max(0, K - S)
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate option delta.
        
        Delta measures the rate of change of option price with respect to the underlying.
        
        Returns:
            Delta value (between -1 and 1 for puts, 0 and 1 for calls)
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma.
        
        Gamma measures the rate of change of delta with respect to the underlying.
        Gamma is the same for calls and puts.
        
        Returns:
            Gamma value
        """
        if T <= 0:
            return 0.0
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega.
        
        Vega measures sensitivity to volatility (per 1% change).
        Vega is the same for calls and puts.
        
        Returns:
            Vega value (divided by 100 to represent 1% change)
        """
        if T <= 0:
            return 0.0
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate option theta.
        
        Theta measures time decay (per day, so divide by 365).
        
        Returns:
            Theta value (per day)
        """
        if T <= 0:
            return 0.0
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2))
        
        return theta / 365  # Per day
    
    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate option rho.
        
        Rho measures sensitivity to interest rate (per 1% change).
        
        Returns:
            Rho value (divided by 100 to represent 1% change)
        """
        if T <= 0:
            return 0.0
        
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str
) -> Dict[str, float]:
    """Calculate all Greeks for an option.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
        
    Returns:
        Dictionary with all Greeks
    """
    bs = BlackScholes()
    
    return {
        'delta': bs.delta(S, K, T, r, sigma, option_type),
        'gamma': bs.gamma(S, K, T, r, sigma),
        'vega': bs.vega(S, K, T, r, sigma),
        'theta': bs.theta(S, K, T, r, sigma, option_type),
        'rho': bs.rho(S, K, T, r, sigma, option_type)
    }


def time_to_maturity(expiry: datetime) -> float:
    """Calculate time to maturity in years.
    
    Args:
        expiry: Expiration datetime
        
    Returns:
        Time to maturity in years
    """
    now = datetime.now()
    if expiry <= now:
        return 0.0
    
    days_to_expiry = (expiry - now).days
    return days_to_expiry / 365.25


if __name__ == '__main__':
    # Test Black-Scholes
    S = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    
    bs = BlackScholes()
    call = bs.call_price(S, K, T, r, sigma)
    put = bs.put_price(S, K, T, r, sigma)
    
    print(f"Call price: ${call:.2f}")
    print(f"Put price: ${put:.2f}")
    print("\nCall Greeks:")
    greeks = calculate_greeks(S, K, T, r, sigma, 'call')
    for greek, value in greeks.items():
        print(f"  {greek}: {value:.4f}")
