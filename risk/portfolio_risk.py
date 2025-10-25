"""Portfolio risk wrapper class."""

from typing import Dict, Tuple
import numpy as np
from data.sample_portfolio import SamplePortfolio
from .metrics import (
    historical_var_cvar,
    monte_carlo_var_cvar,
    calculate_portfolio_greeks,
    calculate_gamma_exposure
)


class PortfolioRisk:
    """Wrapper class for portfolio risk calculations."""
    
    def __init__(self, portfolio: SamplePortfolio, risk_free_rate: float = 0.05):
        """Initialize portfolio risk calculator.
        
        Args:
            portfolio: SamplePortfolio object
            risk_free_rate: Risk-free rate (default 5%)
        """
        self.portfolio = portfolio
        self.risk_free_rate = risk_free_rate
        self._greeks = None
        
    def get_greeks(self) -> Dict[str, float]:
        """Get portfolio Greeks (cached)."""
        if self._greeks is None:
            self._greeks = calculate_portfolio_greeks(self.portfolio, self.risk_free_rate)
        return self._greeks
    
    def get_gamma_exposure(self) -> float:
        """Get dollar gamma exposure."""
        return calculate_gamma_exposure(self.portfolio, self.risk_free_rate)
    
    def calculate_var_cvar(
        self,
        method: str = 'historical',
        confidence_level: float = 0.95,
        **kwargs
    ) -> Tuple[float, float]:
        """Calculate VaR and CVaR.
        
        Args:
            method: 'historical' or 'monte_carlo'
            confidence_level: Confidence level (default 95%)
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Tuple of (VaR, CVaR) in dollars
        """
        if method == 'historical':
            # Fetch historical data
            historical_data = {}
            symbols = set(p.symbol for p in self.portfolio.positions)
            
            for symbol in symbols:
                hist = self.portfolio.get_historical_data(symbol, period='1y')
                historical_data[symbol] = hist
            
            return historical_var_cvar(
                self.portfolio,
                historical_data,
                confidence_level,
                kwargs.get('window', 252)
            )
        
        elif method == 'monte_carlo':
            var, cvar, _ = monte_carlo_var_cvar(
                self.portfolio,
                num_simulations=kwargs.get('num_simulations', 10000),
                time_horizon=kwargs.get('time_horizon', 1),
                confidence_level=confidence_level
            )
            return var, cvar
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def summary(self) -> Dict:
        """Get a summary of portfolio risk metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        greeks = self.get_greeks()
        gamma_exp = self.get_gamma_exposure()
        
        try:
            hist_var, hist_cvar = self.calculate_var_cvar(method='historical')
        except:
            hist_var, hist_cvar = 0.0, 0.0
        
        try:
            mc_var, mc_cvar = self.calculate_var_cvar(method='monte_carlo')
        except:
            mc_var, mc_cvar = 0.0, 0.0
        
        return {
            'portfolio_value': self.portfolio.total_value,
            'greeks': greeks,
            'gamma_exposure': gamma_exp,
            'historical_var_95': hist_var,
            'historical_cvar_95': hist_cvar,
            'monte_carlo_var_95': mc_var,
            'monte_carlo_cvar_95': mc_cvar
        }
