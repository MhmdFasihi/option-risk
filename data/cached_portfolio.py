"""Cached sample portfolio data to avoid yfinance rate limits."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.sample_portfolio import Position, SamplePortfolio


def get_cached_historical_data(symbol: str, period: str = '1y') -> pd.DataFrame:
    """Generate synthetic historical data for cached portfolio.
    
    This creates realistic-looking historical prices based on GBM simulation.
    
    Args:
        symbol: Stock ticker
        period: Period (currently only supports '1y')
        
    Returns:
        DataFrame with OHLCV data
    """
    # Base prices for each symbol
    base_prices = {
        'AAPL': 178.50,
        'MSFT': 372.85,
        'GOOGL': 142.30
    }
    
    if symbol not in base_prices:
        raise ValueError(f"Unknown symbol: {symbol}")
    
    # Generate 252 trading days (1 year)
    days = 252
    current_price = base_prices[symbol]
    
    # Simulate price path backwards in time
    mu = 0.10 / 252  # 10% annual return
    sigma = 0.25 / np.sqrt(252)  # 25% annual volatility
    
    np.random.seed(hash(symbol) % 2**32)  # Consistent data for same symbol
    
    prices = [current_price]
    for _ in range(days - 1):
        prev_price = prices[-1]
        # Go backwards in time
        change = np.random.normal(-mu, sigma)
        new_price = prev_price * np.exp(-change)
        prices.append(new_price)
    
    prices = list(reversed(prices))
    
    # Create OHLC data
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    
    df = pd.DataFrame({
        'Open': [p * np.random.uniform(0.995, 1.005) for p in prices],
        'High': [p * np.random.uniform(1.005, 1.015) for p in prices],
        'Low': [p * np.random.uniform(0.985, 0.995) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(50_000_000, 150_000_000) for _ in prices]
    }, index=dates)
    
    return df


def load_cached_sample_portfolio() -> SamplePortfolio:
    """Load a sample portfolio with hardcoded prices (no API calls).
    
    This avoids yfinance rate limits and loads instantly.
    Prices are representative as of October 2025.
    
    Returns:
        SamplePortfolio with hardcoded positions
    """
    portfolio = SamplePortfolio()
    
    # Manually create stock positions with approximate prices
    # AAPL - Long 100 shares
    aapl_position = Position(
        symbol='AAPL',
        position_type='stock',
        quantity=100,
        current_price=178.50
    )
    portfolio.positions.append(aapl_position)
    
    # MSFT - Long 50 shares
    msft_position = Position(
        symbol='MSFT',
        position_type='stock',
        quantity=50,
        current_price=372.85
    )
    portfolio.positions.append(msft_position)
    
    # GOOGL - Short 25 shares
    googl_position = Position(
        symbol='GOOGL',
        position_type='stock',
        quantity=-25,
        current_price=142.30
    )
    portfolio.positions.append(googl_position)
    
    # Option positions
    expiry_1m = datetime.now() + timedelta(days=30)
    expiry_3m = datetime.now() + timedelta(days=90)
    
    # AAPL Call - Long 5 contracts, 30 days out, near ATM
    aapl_call_1m = Position(
        symbol='AAPL',
        position_type='option',
        quantity=5,
        current_price=4.25,
        option_type='call',
        strike=180.0,
        expiry=expiry_1m,
        implied_volatility=0.28
    )
    portfolio.positions.append(aapl_call_1m)
    
    # MSFT Put - Short 3 contracts, 30 days out
    msft_put_1m = Position(
        symbol='MSFT',
        position_type='option',
        quantity=-3,
        current_price=6.50,
        option_type='put',
        strike=370.0,
        expiry=expiry_1m,
        implied_volatility=0.26
    )
    portfolio.positions.append(msft_put_1m)
    
    # AAPL Call - Long 10 contracts, 90 days out, slightly OTM
    aapl_call_3m = Position(
        symbol='AAPL',
        position_type='option',
        quantity=10,
        current_price=7.80,
        option_type='call',
        strike=190.0,
        expiry=expiry_3m,
        implied_volatility=0.30
    )
    portfolio.positions.append(aapl_call_3m)
    
    # GOOGL Put - Long 8 contracts, 90 days out
    googl_put_3m = Position(
        symbol='GOOGL',
        position_type='option',
        quantity=8,
        current_price=5.20,
        option_type='put',
        strike=140.0,
        expiry=expiry_3m,
        implied_volatility=0.29
    )
    portfolio.positions.append(googl_put_3m)
    
    # Store historical data generator
    portfolio.get_historical_data = get_cached_historical_data
    
    return portfolio


if __name__ == '__main__':
    # Test the cached portfolio
    portfolio = load_cached_sample_portfolio()
    print(f"Portfolio loaded: {len(portfolio.positions)} positions")
    print(f"Total value: ${portfolio.total_value:,.2f}")
    print("\nPositions:")
    for pos in portfolio.positions:
        print(f"  {pos.symbol} {pos.position_type}: {pos.quantity} @ ${pos.current_price:.2f}")
