# NVDA Portfolio Manager

A comprehensive local trading dashboard for managing stock and options portfolios with live data, technical indicators, and transparent buy/sell signals.

## Features

- **Portfolio Overview**: Track all holdings with brokerage-like columns (last price, today's gain/loss, total gain/loss, current value, % of account, quantity, cost basis, 52-week range)
- **Interactive Charts**: Candlestick charts with EMA 20/50/100/200, VWAP, RSI(14), MACD(12,26,9), ATR(14), Volume
- **Signal Engine**: Rule-based trading signals with transparent reasoning
- **Options Data**: Live options chain data (bid/ask/last, IV, Greeks) with fallback to Black-Scholes estimates
- **Price Alerts**: Configurable price and signal alerts
- **Risk Controls**: Stop-loss suggestions, concentration risk warnings, scenario modeling

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys (Optional)

The dashboard works without API keys but will use estimated values for options. To get live options data:

#### Tradier (Default Provider)
1. Create a [Tradier](https://tradier.com) account
2. Get your API token from Settings → API Access
3. Edit `config.json` and add your token:

```json
{
    "apis": {
        "tradier_token": "YOUR_TRADIER_TOKEN_HERE"
    }
}
```

#### Polygon.io (Optional)
1. Create a [Polygon.io](https://polygon.io) account
2. Get your API key from the dashboard
3. Edit `config.json`:

```json
{
    "apis": {
        "polygon_api_key": "YOUR_POLYGON_API_KEY_HERE"
    }
}
```

### 4. Run the Application

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Configuration

### config.json Structure

```json
{
    "portfolio": {
        "cash": 32.61,
        "pending_activity": 137.53,
        "positions": [
            {
                "symbol": "NVDA",
                "type": "stock",
                "quantity": 37.787,
                "cost_basis_total": 4056.23,
                "cost_basis_per_share": 107.34
            }
        ]
    },
    "settings": {
        "refresh_interval": 15,
        "default_timeframe": "1d",
        "iv_fallback": 0.55,
        "stop_loss_atr_multiplier": 2.0,
        "concentration_threshold": 0.25
    },
    "apis": {
        "tradier_token": "",
        "polygon_api_key": "",
        "marketdata_token": ""
    },
    "alerts": []
}
```

### Portfolio Seed Data

The config is pre-filled with your current holdings:

| Position | Type | Quantity | Cost Basis |
|----------|------|----------|------------|
| NVDA | Stock | 37.787 | $4,056.23 |
| NVDA 2027-01-15 C 300 | Option | 1 contract | $640.67 |
| PLTR | Stock | 2.384 | $330.14 |
| XLF 2026-09-18 P 48 | Option | 2 contracts | $193.35 |
| AMC 2026-05-08 C 2 | Option | 2 contracts | $39.35 |

Cash: $32.61 | Pending: $137.53

## Pages

### 1. Portfolio Overview
- Table of all holdings with brokerage-like columns
- Portfolio totals: total value, today P/L, total P/L, cash, pending activity
- % of account per position
- Auto-refresh every 15 seconds

### 2. Charts
- Select any symbol from your positions
- Timeframes: 1m, 5m, 30m, 1h, 1d
- Technical overlays:
  - EMA 20/50/100/200
  - VWAP (intraday)
  - RSI(14)
  - MACD(12,26,9)
  - ATR(14)
  - Volume + volume MA
- Buy/sell signals displayed on chart

### 3. Options Details
- Live bid/ask/last, IV, Greeks (delta/gamma/theta/vega/rho)
- Intrinsic vs extrinsic value
- Underlying price + today change
- Position P/L (today and total)
- Mini option chain view

### 4. Alerts
- Price alerts per underlying
- Signal alerts when new signals trigger
- On-screen notifications

### 5. Risk Management
- Stop loss suggestions using ATR + EMA
- Scenario table with Black-Scholes estimates
- Concentration risk warnings (>25%)

## Signal Engine

The signal engine uses rule-based logic (no ML):

1. **Trend Filter**: Bullish when price > EMA200 AND EMA50 > EMA100
2. **Momentum Trigger**: EMA20 crosses above EMA50 AND RSI > 50
3. **Risk-off Trigger**: Close below EMA50 OR RSI < 40
4. **Breakout**: Close above prior day high with volume confirmation

Each signal stores a "reason string" shown in UI and logged.

## Data Sources

### Stocks / OHLCV
- **yfinance** (default): Free, no API key needed
- **Polygon** (optional): For intraday data if `POLYGON_API_KEY` is set

### Options Chain
1. **Tradier** (default): Requires `TRADIER_TOKEN`
2. **Polygon** (optional): Requires `POLYGON_API_KEY`
3. **Fallback**: Black-Scholes estimate with 55% IV

## Rate Limits & Known Limitations

### yfinance
- May have rate limits on repeated requests
- Intraday data limited to 7 days
- Some tickers may not be available

### Tradier
- Free tier: 50 API calls/minute
- Real-time data requires funded account

### Polygon
- Free tier: 5 API calls/minute
- Options data requires paid plan for full access

### General
- No auto-trading capability
- No connection to brokerage accounts
- Options Greeks are estimates unless using live data
- 52-week range may not be available for all securities

## Security

- All data stored locally in `config.json`
- No external connections except to data providers
- API keys stored locally (not sent anywhere)
- No trading capability - display only

## Troubleshooting

### No data showing
- Check internet connection
- Verify symbol is correct
- Try different timeframe

### Options showing "Estimated"
- No live options API configured
- Add Tradier or Polygon API key to config.json

### Chart not loading
- yfinance may be rate-limited
- Wait a few seconds and refresh

### Error messages
- Check API keys are valid
- Verify config.json is valid JSON

## License

MIT License - Use at your own risk for educational purposes.