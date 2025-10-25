"""Simple test to verify the setup works."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from data import load_sample_portfolio
        print("✓ data module imported")
    except ImportError as e:
        print(f"✗ data module failed: {e}")
        return False
    
    try:
        from models import BlackScholes
        print("✓ models module imported")
    except ImportError as e:
        print(f"✗ models module failed: {e}")
        return False
    
    try:
        from risk import PortfolioRisk
        print("✓ risk module imported")
    except ImportError as e:
        print(f"✗ risk module failed: {e}")
        return False
    
    try:
        from surfaces import create_sample_volatility_surface
        print("✓ surfaces module imported")
    except ImportError as e:
        print(f"✗ surfaces module failed: {e}")
        return False
    
    try:
        from utils import format_currency
        print("✓ utils module imported")
    except ImportError as e:
        print(f"✗ utils module failed: {e}")
        return False
    
    return True


def test_black_scholes():
    """Test Black-Scholes calculations."""
    print("\nTesting Black-Scholes model...")
    
    from models import BlackScholes
    
    bs = BlackScholes()
    call_price = bs.call_price(S=100, K=100, T=1, r=0.05, sigma=0.2)
    put_price = bs.put_price(S=100, K=100, T=1, r=0.05, sigma=0.2)
    
    print(f"  ATM Call price: ${call_price:.2f}")
    print(f"  ATM Put price: ${put_price:.2f}")
    
    # Sanity check
    assert call_price > 0, "Call price should be positive"
    assert put_price > 0, "Put price should be positive"
    
    print("✓ Black-Scholes test passed")
    return True


def test_sample_portfolio():
    """Test sample portfolio loading."""
    print("\nTesting sample portfolio...")
    
    from data import load_sample_portfolio
    
    portfolio = load_sample_portfolio()
    
    print(f"  Loaded {len(portfolio.positions)} positions")
    print(f"  Total value: ${portfolio.total_value:,.2f}")
    
    assert len(portfolio.positions) > 0, "Portfolio should have positions"
    assert portfolio.total_value > 0, "Portfolio should have positive value"
    
    print("✓ Sample portfolio test passed")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Option Risk Management - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_black_scholes,
        test_sample_portfolio
    ]
    
    failed = []
    
    for test in tests:
        try:
            if not test():
                failed.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} failed with error: {e}")
            failed.append(test.__name__)
    
    print("\n" + "=" * 50)
    if failed:
        print(f"❌ {len(failed)} test(s) failed: {', '.join(failed)}")
        return 1
    else:
        print("✅ All tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
