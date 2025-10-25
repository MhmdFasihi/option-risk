"""Sample portfolio loader using yfinance for testing and development."""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Position:
    """Represents a single position (stock or option)."""
    symbol: str
    position_type: str  # 'stock' or 'option'
    quantity: int
    current_price: float
    
    # Option-specific fields
    option_type: Optional[str] = None  # 'call' or 'put'
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    implied_volatility: Optional[float] = None
    
    # Greeks (calculated later)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        """Calculate position market value."""
        if self.position_type == 'option':
            # Options are quoted per share, contracts are 100 shares
            return self.quantity * self.current_price * 100
        return self.quantity * self.current_price
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'type': self.position_type,
            'quantity': self.quantity,
            'price': self.current_price,
            'market_value': self.market_value,
            'option_type': self.option_type,
            'strike': self.strike,
            'expiry': self.expiry,
            'iv': self.implied_volatility,
            'delta': self.delta,
            'gamma': self.gamma,
            'vega': self.vega,
            'theta': self.theta,
            'rho': self.rho
        }


class SamplePortfolio:
    """Sample portfolio with stocks and options."""
    
    def __init__(self):
        self.positions: List[Position] = []
        self.market_data: Dict = {}
        
    def add_stock_position(self, symbol: str, quantity: int) -> None:
        """Add a stock position to the portfolio.
        
        Args:
            symbol: Stock ticker symbol
            quantity: Number of shares (can be negative for short)
        """
        ticker = yf.Ticker(symbol)
        current_price = ticker.info.get('currentPrice', ticker.info.get('regularMarketPrice', 0))
        
        if current_price == 0:
            # Fallback to history if info doesn't have price
            hist = ticker.history(period='1d')
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
        
        position = Position(
            symbol=symbol,
            position_type='stock',
            quantity=quantity,
            current_price=current_price
        )
        
        self.positions.append(position)
        self.market_data[symbol] = ticker
        
    def add_option_position(
        self, 
        underlying: str,
        option_type: str,
        strike: float,
        expiry: datetime,
        quantity: int,
        implied_vol: Optional[float] = None
    ) -> None:
        """Add an option position to the portfolio.
        
        Args:
            underlying: Underlying stock ticker
            option_type: 'call' or 'put'
            strike: Strike price
            expiry: Expiration date
            quantity: Number of contracts (can be negative for short)
            implied_vol: Implied volatility (if None, will try to fetch from yfinance)
        """
        ticker = yf.Ticker(underlying)
        
        # Get current underlying price
        current_underlying = ticker.info.get('currentPrice', ticker.info.get('regularMarketPrice', 0))
        if current_underlying == 0:
            hist = ticker.history(period='1d')
            if not hist.empty:
                current_underlying = hist['Close'].iloc[-1]
        
        # Try to get option price from yfinance
        option_price = 0
        iv = implied_vol
        
        try:
            expiry_str = expiry.strftime('%Y-%m-%d')
            options = ticker.option_chain(expiry_str)
            
            if option_type.lower() == 'call':
                chain = options.calls
            else:
                chain = options.puts
            
            # Find the closest strike
            chain['strike_diff'] = abs(chain['strike'] - strike)
            closest = chain.loc[chain['strike_diff'].idxmin()]
            option_price = closest['lastPrice']
            
            if iv is None and 'impliedVolatility' in closest:
                iv = closest['impliedVolatility']
                
        except Exception as e:
            print(f"Warning: Could not fetch option data for {underlying} {strike} {option_type}: {e}")
            # Use simple Black-Scholes estimate if yfinance fails
            option_price = max(0.01, abs(current_underlying - strike) * 0.1)
            if iv is None:
                iv = 0.3  # Default 30% volatility
        
        position = Position(
            symbol=underlying,
            position_type='option',
            quantity=quantity,
            current_price=option_price,
            option_type=option_type.lower(),
            strike=strike,
            expiry=expiry,
            implied_volatility=iv
        )
        
        self.positions.append(position)
        
        if underlying not in self.market_data:
            self.market_data[underlying] = ticker
    
    def get_historical_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Get historical price data for a symbol.
        
        Args:
            symbol: Ticker symbol
            period: Period string (e.g., '1y', '6mo', '3mo')
            
        Returns:
            DataFrame with historical OHLCV data
        """
        if symbol not in self.market_data:
            ticker = yf.Ticker(symbol)
            self.market_data[symbol] = ticker
        else:
            ticker = self.market_data[symbol]
        
        return ticker.history(period=period)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert portfolio to DataFrame."""
        return pd.DataFrame([pos.to_dict() for pos in self.positions])
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio market value."""
        return sum(pos.market_value for pos in self.positions)
    
    def __repr__(self) -> str:
        return f"<SamplePortfolio: {len(self.positions)} positions, ${self.total_value:,.2f}>"


def load_sample_portfolio() -> SamplePortfolio:
    """Load a predefined sample portfolio for testing.
    
    Returns:
        SamplePortfolio with sample positions
    """
    portfolio = SamplePortfolio()
    
    # Add stock positions
    portfolio.add_stock_position('AAPL', 100)  # Long 100 shares Apple
    portfolio.add_stock_position('MSFT', 50)   # Long 50 shares Microsoft
    portfolio.add_stock_position('GOOGL', -25) # Short 25 shares Google
    
    # Add option positions
    # Near-term ATM calls
    expiry_1m = datetime.now() + timedelta(days=30)
    portfolio.add_option_position('AAPL', 'call', 180, expiry_1m, 5)  # Long 5 calls
    portfolio.add_option_position('MSFT', 'put', 370, expiry_1m, -3)  # Short 3 puts
    
    # Further dated options
    expiry_3m = datetime.now() + timedelta(days=90)
    portfolio.add_option_position('AAPL', 'call', 190, expiry_3m, 10)
    portfolio.add_option_position('GOOGL', 'put', 140, expiry_3m, 8)
    
    return portfolio


if __name__ == '__main__':
    # Test the sample portfolio loader
    print("Loading sample portfolio...")
    portfolio = load_sample_portfolio()
    print(portfolio)
    print("\nPositions:")
    print(portfolio.to_dataframe())
