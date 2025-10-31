# Quick Start Guide

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MhmdFasihi/option-risk.git
   cd option-risk
   ```

2. **Create and activate conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate option-risk
   ```

3. **Create environment**
   
   **Option A: Using conda (recommended for local dev):**
   ```bash
   conda env create -f environment.yml.local
   conda activate option-risk
   ```
   
   **Option B: Using pip:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Configure environment (optional)**
   ```bash
   cp .env.example .env
   # Edit .env if you want to customize settings
   ```

## Running the Dashboard

**Important**: Make sure the conda environment is activated first!

```bash
conda activate option-risk
streamlit run dashboard/app.py
```

The dashboard will open in your browser at <http://localhost:8501>

## Features Overview

### 📈 Portfolio Overview
- Total portfolio value
- Position breakdown (stocks and options)
- Portfolio-level Greeks (Delta, Gamma, Vega, Theta, Rho)
- Gamma exposure

### ⚠️ Risk Metrics
- **Historical VaR/CVaR**: Based on historical price data (1-year window)
- **Monte Carlo VaR/CVaR**: Based on 10,000 simulations using GBM
- Adjustable confidence levels (90%, 95%, 99%)

### 🏔️ 3D Surfaces
- **Volatility Surface**: Shows implied volatility across strikes and expiries
- **Greek Surfaces**: Visualize Delta, Gamma, Vega, Theta, or Rho in 3D

### 📊 Position Details
- Detailed position information
- Filter by position type and option type
- Summary statistics

## Testing

Run the setup test to verify everything works:

```bash
python tests/test_setup.py
```

Note: yfinance may occasionally hit rate limits. If this happens, wait a few minutes and try again.

## Sample Portfolio

The default sample portfolio includes:
- **Stocks**: AAPL (long), MSFT (long), GOOGL (short)
- **Options**: Various calls and puts on AAPL, MSFT, and GOOGL

Data is fetched in real-time from yfinance.

## Project Structure

```
option-risk/
├── dashboard/          # Streamlit dashboard
│   └── app.py         # Main application
├── data/              # Data loading
│   └── sample_portfolio.py  # Sample portfolio with yfinance
├── models/            # Pricing models
│   └── black_scholes.py     # Black-Scholes + Greeks
├── risk/              # Risk calculations
│   ├── metrics.py     # VaR, CVaR, Greeks
│   └── portfolio_risk.py    # Risk wrapper
├── surfaces/          # 3D visualizations
│   ├── volatility_surface.py
│   └── greek_surface.py
├── utils/             # Utilities
│   └── formatting.py  # Display helpers
├── tests/             # Tests
└── config.py          # Configuration
```

## Next Steps

1. **Customize the portfolio**: Edit `data/sample_portfolio.py` to test different positions
2. **Adjust risk parameters**: Modify `config.py` or create a `.env` file
3. **API Integration**: When ready, implement the custom portfolio API client in `data/`

## Troubleshooting

### yfinance Rate Limits
If you see "Too Many Requests" errors:
- Wait a few minutes before retrying
- Reduce the number of positions in the sample portfolio
- Use cached data if available

### Missing Dependencies
If you get import errors:
```bash
conda activate option-risk
conda env update -f environment.yml --prune
```

### Dashboard Won't Start
Make sure you're in the correct directory and environment:
```bash
conda activate option-risk
streamlit run dashboard/app.py
```

## Documentation

See `.github/copilot-instructions.md` for detailed developer guidance and architecture notes.
