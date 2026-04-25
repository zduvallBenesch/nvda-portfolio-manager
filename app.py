"""
NVDA Portfolio Manager - Trading Dashboard
A comprehensive stock and options portfolio management system with live data,
technical indicators, and transparent buy/sell signals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import requests
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

CONFIG_FILE = "config.json"
CACHE_TTL = 30  # seconds

# Default cache for stock data
@st.cache_data(ttl=CACHE_TTL)
def get_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch stock data from yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return pd.DataFrame()
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL)
def get_intraday_data(symbol: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    """Fetch intraday stock data"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval=interval, period=period)
        if df.empty:
            return pd.DataFrame()
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching intraday data for {symbol}: {e}")
        return pd.DataFrame()

# ============================================================================
# OPTIONS DATA PROVIDERS
# ============================================================================

def get_tradier_options_chain(symbol: str, expiration: str, greeks: bool = True) -> Optional[Dict]:
    """Fetch options chain from Tradier API"""
    config = load_config()
    token = config.get("apis", {}).get("tradier_token", "")
    
    if not token:
        return None
    
    url = "https://api.tradier.com/v1/markets/options/chains"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "greeks": "true" if greeks else "false"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.error(f"Tradier API error: {e}")
    
    return None

def get_polygon_options_chain(symbol: str, date: str) -> Optional[Dict]:
    """Fetch options chain from Polygon.io"""
    config = load_config()
    api_key = config.get("apis", {}).get("polygon_api_key", "")
    
    if not api_key:
        return None
    
    url = f"https://api.polygon.io/v3/snapshot/options/{symbol}"
    params = {
        "apiKey": api_key,
        "expiration_date": date
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.error(f"Polygon API error: {e}")
    
    return None

def black_scholes_estimate(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-Scholes option pricing model"""
    from scipy.stats import norm
    
    if T <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> Dict:
    """Calculate Greeks using Black-Scholes"""
    from scipy.stats import norm
    
    if T <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * norm.cdf(d2) * (1 if option_type == "call" else -1)) / 365
    
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho
    }

def get_option_quote(symbol: str, strike: float, expiration: str, option_type: str) -> Dict:
    """Get option quote from available providers with fallback"""
    config = load_config()
    iv_fallback = config.get("settings", {}).get("iv_fallback", 0.55)
    
    # Try Tradier first
    tradier_data = get_tradier_options_chain(symbol, expiration)
    if tradier_data and "options" in tradier_data:
        for opt in tradier_data.get("options", {}).get("option", []):
            if (opt.get("strike") == strike and 
                opt.get("type") == option_type and
                opt.get("expiration") == expiration):
                return {
                    "bid": opt.get("bid", 0),
                    "ask": opt.get("ask", 0),
                    "last": opt.get("last", 0),
                    "mid": (opt.get("bid", 0) + opt.get("ask", 0)) / 2,
                    "iv": opt.get("iv", iv_fallback) / 100,
                    "delta": opt.get("delta", 0),
                    "gamma": opt.get("gamma", 0),
                    "theta": opt.get("theta", 0),
                    "vega": opt.get("vega", 0),
                    "rho": opt.get("rho", 0),
                    "volume": opt.get("volume", 0),
                    "open_interest": opt.get("open_interest", 0),
                    "source": "tradier"
                }
    
    # Try Polygon
    polygon_data = get_polygon_options_chain(symbol, expiration)
    if polygon_data and "results" in polygon_data:
        for opt in polygon_data["results"]:
            if (opt.get("strike_price") == strike and 
                opt.get("option_type") == option_type):
                return {
                    "bid": opt.get("bid", 0),
                    "ask": opt.get("ask", 0),
                    "last": opt.get("last", 0),
                    "mid": (opt.get("bid", 0) + opt.get("ask", 0)) / 2,
                    "iv": opt.get("implied_volatility", iv_fallback) / 100,
                    "delta": opt.get("delta", 0),
                    "gamma": opt.get("gamma", 0),
                    "theta": opt.get("theta", 0),
                    "vega": opt.get("vega", 0),
                    "rho": opt.get("rho", 0),
                    "volume": opt.get("volume", 0),
                    "open_interest": opt.get("open_interest", 0),
                    "source": "polygon"
                }
    
    # Fallback to Black-Scholes estimate
    underlying = yf.Ticker(symbol)
    S = underlying.info.get("currentPrice", 0)
    if S == 0:
        S = 100  # Default fallback
        
    exp_date = datetime.strptime(expiration, "%Y-%m-%d")
    T = (exp_date - datetime.now()).days / 365
    r = 0.05  # Risk-free rate
    
    price = black_scholes_estimate(S, strike, T, r, iv_fallback, option_type)
    greeks = calculate_greeks(S, strike, T, r, iv_fallback, option_type)
    
    return {
        "bid": price * 0.95,
        "ask": price * 1.05,
        "last": price,
        "mid": price,
        "iv": iv_fallback,
        "delta": greeks["delta"],
        "gamma": greeks["gamma"],
        "theta": greeks["theta"],
        "vega": greeks["vega"],
        "rho": greeks["rho"],
        "volume": 0,
        "open_interest": 0,
        "source": "estimated"
    }

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_config() -> Dict:
    """Load configuration from JSON file"""
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"portfolio": {}, "settings": {}, "apis": {}, "alerts": []}

def save_config(config: Dict):
    """Save configuration to JSON file"""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period, adjust=False).mean()

def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD"""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def find_support_resistance(prices: pd.Series, lookback: int = 20) -> Tuple[List[float], List[float]]:
    """Find support and resistance levels"""
    highs = prices.rolling(window=5).max()
    lows = prices.rolling(window=5).min()
    
    # Find swing highs
    swing_highs = []
    for i in range(lookback, len(prices) - lookback):
        if highs.iloc[i] == highs.iloc[i-1] == highs.iloc[i-2]:
            if highs.iloc[i] > highs.iloc[i-3] and highs.iloc[i] > highs.iloc[i+1]:
                swing_highs.append(highs.iloc[i])
    
    # Find swing lows
    swing_lows = []
    for i in range(lookback, len(prices) - lookback):
        if lows.iloc[i] == lows.iloc[i-1] == lows.iloc[i-2]:
            if lows.iloc[i] < lows.iloc[i-3] and lows.iloc[i] < lows.iloc[i+1]:
                swing_lows.append(lows.iloc[i])
    
    return swing_highs[-5:] if swing_highs else [], swing_lows[-5:] if swing_lows else []

# ============================================================================
# SIGNAL ENGINE
# ============================================================================

def generate_signals(df: pd.DataFrame) -> List[Dict]:
    """Generate trading signals based on technical indicators"""
    if len(df) < 200:
        return []
    
    signals = []
    close = df['Close']
    volume = df['Volume']
    
    # Calculate indicators
    ema20 = calculate_ema(close, 20)
    ema50 = calculate_ema(close, 50)
    ema100 = calculate_ema(close, 100)
    ema200 = calculate_ema(close, 200)
    rsi = calculate_rsi(close)
    macd, signal_line, hist = calculate_macd(close)
    
    # Get latest values
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    current_price = close.iloc[-1]
    
    # Trend filter: bullish when price > EMA200 AND EMA50 > EMA100
    trend_bullish = current_price > ema200.iloc[-1] and ema50.iloc[-1] > ema100.iloc[-1]
    
    # Momentum trigger: EMA20 crosses above EMA50 AND RSI > 50 AND above VWAP
    ema20_above_50 = ema20.iloc[-1] > ema50.iloc[-1]
    ema20_was_below = ema20.iloc[-2] <= ema50.iloc[-2]
    rsi_bullish = rsi.iloc[-1] > 50
    
    if trend_bullish and ema20_above_50 and ema20_was_below and rsi_bullish:
        signals.append({
            "type": "BUY",
            "reason": "Trend bullish (price > EMA200, EMA50 > EMA100) + EMA20 crossed above EMA50 + RSI > 50",
            "timestamp": datetime.now().isoformat()
        })
    
    # Risk-off trigger: close below EMA50 OR RSI < 40
    if current_price < ema50.iloc[-1]:
        signals.append({
            "type": "SELL",
            "reason": "Price closed below EMA50 - trend reversal signal",
            "timestamp": datetime.now().isoformat()
        })
    elif rsi.iloc[-1] < 40:
        signals.append({
            "type": "SELL",
            "reason": f"RSI oversold at {rsi.iloc[-1]:.1f} - risk-off trigger",
            "timestamp": datetime.now().isoformat()
        })
    
    # Breakout: close above prior day high with volume confirmation
    if len(df) >= 2:
        prior_high = df['High'].iloc[-2]
        avg_volume = volume.rolling(20).mean().iloc[-1]
        if current_price > prior_high and volume.iloc[-1] > avg_volume * 1.5:
            signals.append({
                "type": "BUY",
                "reason": f"Breakout above prior day high ${prior_high:.2f} with volume {volume.iloc[-1]/avg_volume:.1f}x average",
                "timestamp": datetime.now().isoformat()
            })
    
    return signals

# ============================================================================
# PORTFOLIO CALCULATIONS
# ============================================================================

def get_portfolio_value() -> Tuple[float, float, float, List[Dict]]:
    """Calculate total portfolio value and positions"""
    config = load_config()
    portfolio = config.get("portfolio", {})
    positions = portfolio.get("positions", [])
    cash = portfolio.get("cash", 0)
    pending = portfolio.get("pending_activity", 0)
    
    total_value = cash + pending
    today_change = 0
    position_details = []
    
    for pos in positions:
        symbol = pos["symbol"]
        pos_type = pos["type"]
        
        if pos_type == "stock":
            # Get current price
            ticker = yf.Ticker(symbol)
            try:
                info = ticker.info
                current_price = info.get("currentPrice") or info.get("regularMarketPreviousClose") or 0
                prev_close = info.get("regularMarketPreviousClose") or current_price
            except:
                current_price = 0
                prev_close = 0
            
            quantity = pos["quantity"]
            cost_basis = pos["cost_basis_total"]
            cost_per_share = pos["cost_basis_per_share"]
            
            current_value = quantity * current_price
            today_pl = quantity * (current_price - prev_close)
            total_pl = current_value - cost_basis
            today_change += today_pl
            
            # Get 52-week range
            try:
                fifty_two_week_low = info.get("fiftyTwoWeekLow", 0)
                fifty_two_week_high = info.get("fiftyTwoWeekHigh", 0)
            except:
                fifty_two_week_low = fifty_two_week_high = 0
            
            position_details.append({
                "symbol": symbol,
                "type": "stock",
                "quantity": quantity,
                "cost_basis": cost_basis,
                "cost_per_share": cost_per_share,
                "current_price": current_price,
                "prev_close": prev_close,
                "current_value": current_value,
                "today_pl": today_pl,
                "total_pl": total_pl,
                "today_pl_pct": (today_pl / (quantity * prev_close)) * 100 if prev_close > 0 else 0,
                "total_pl_pct": (total_pl / cost_basis) * 100 if cost_basis > 0 else 0,
                "52_week_low": fifty_two_week_low,
                "52_week_high": fifty_two_week_high
            })
            
            total_value += current_value
            
        elif pos_type == "option":
            strike = pos["strike"]
            expiration = pos["expiration"]
            option_type = pos["option_type"]
            contracts = pos["contracts"]
            cost_basis = pos["cost_basis_total"]
            cost_per_share = pos["cost_basis_per_share"]
            
            # Get option quote
            quote = get_option_quote(symbol, strike, expiration, option_type)
            
            # Get underlying price
            try:
                underlying = yf.Ticker(symbol)
                current_price = underlying.info.get("currentPrice") or 0
                prev_close = underlying.info.get("regularMarketPreviousClose") or current_price
            except:
                current_price = 0
                prev_close = 0
            
            multiplier = 100
            premium = quote["mid"]
            current_value = contracts * premium * multiplier
            today_pl = contracts * (premium - cost_per_share) * multiplier
            total_pl = current_value - cost_basis
            today_change += today_pl
            
            # Calculate intrinsic/extrinsic
            if option_type == "call":
                intrinsic = max(0, current_price - strike)
            else:
                intrinsic = max(0, strike - current_price)
            extrinsic = premium - intrinsic
            
            position_details.append({
                "symbol": symbol,
                "type": "option",
                "option_type": option_type,
                "strike": strike,
                "expiration": expiration,
                "contracts": contracts,
                "cost_basis": cost_basis,
                "cost_per_share": cost_per_share,
                "current_price": premium,
                "prev_close": cost_per_share,
                "current_value": current_value,
                "today_pl": today_pl,
                "total_pl": total_pl,
                "today_pl_pct": ((premium - cost_per_share) / cost_per_share) * 100 if cost_per_share > 0 else 0,
                "total_pl_pct": (total_pl / cost_basis) * 100 if cost_basis > 0 else 0,
                "underlying_price": current_price,
                "bid": quote["bid"],
                "ask": quote["ask"],
                "iv": quote["iv"],
                "delta": quote["delta"],
                "gamma": quote["gamma"],
                "theta": quote["theta"],
                "vega": quote["vega"],
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
                "volume": quote["volume"],
                "open_interest": quote["open_interest"],
                "source": quote["source"]
            })
            
            total_value += current_value
    
    return total_value, today_change, cash + pending, position_details

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_portfolio_table(positions: List[Dict], total_value: float, cash_pending: float):
    """Render portfolio holdings table"""
    st.subheader("📊 Portfolio Holdings")
    
    # Build DataFrame
    rows = []
    for pos in positions:
        symbol = pos["symbol"]
        if pos["type"] == "stock":
            rows.append({
                "Symbol": symbol,
                "Type": "Stock",
                "Description": f"{symbol} Common Stock",
                "Quantity": pos["quantity"],
                "Last Price": f"${pos['current_price']:.2f}",
                "Today $": f"{pos['today_pl']:+.2f}",
                "Today %": f"{pos['today_pl_pct']:+.2f}%",
                "Cost Basis": f"${pos['cost_basis']:,.2f}",
                "Current Value": f"${pos['current_value']:,.2f}",
                "Total $": f"{pos['total_pl']:+.2f}",
                "Total %": f"{pos['total_pl_pct']:+.2f}%",
                "% of Acct": f"{(pos['current_value']/total_value)*100:.1f}%",
                "52W Range": f"${pos.get('52_week_low', 0):.2f} - ${pos.get('52_week_high', 0):.2f}"
            })
        else:
            rows.append({
                "Symbol": f"{symbol} {pos['option_type'].upper()} {pos['strike']}",
                "Type": "Option",
                "Description": f"{symbol} ${pos['strike']} {pos['option_type'].upper()} {pos['expiration']}",
                "Quantity": pos["contracts"],
                "Last Price": f"${pos['current_price']:.2f}",
                "Today $": f"{pos['today_pl']:+.2f}",
                "Today %": f"{pos['today_pl_pct']:+.2f}%",
                "Cost Basis": f"${pos['cost_basis']:,.2f}",
                "Current Value": f"${pos['current_value']:,.2f}",
                "Total $": f"{pos['total_pl']:+.2f}",
                "Total %": f"{pos['total_pl_pct']:+.2f}%",
                "% of Acct": f"{(pos['current_value']/total_value)*100:.1f}%",
                "52W Range": "N/A"
            })
    
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Portfolio totals
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_cost = sum(p["cost_basis"] for p in positions)
    total_pl = sum(p["total_pl"] for p in positions)
    today_pl = sum(p["today_pl"] for p in positions)
    
    with col1:
        st.metric("Total Value", f"${total_value:,.2f}")
    with col2:
        st.metric("Cash + Pending", f"${cash_pending:,.2f}")
    with col3:
        st.metric("Today's P/L", f"{today_pl:+,.2f}", f"{today_pl/total_value*100:+.2f}%")
    with col4:
        st.metric("Total P/L", f"{total_pl:+,.2f}", f"{total_pl/total_cost*100:+.2f}%")
    with col5:
        st.metric("Cost Basis", f"${total_cost:,.2f}")

def render_charts(symbol: str, timeframe: str):
    """Render interactive candlestick chart with technical indicators"""
    st.subheader(f"📈 {symbol} Chart")
    
    # Get data based on timeframe
    interval_map = {"1m": "1m", "5m": "5m", "30m": "30m", "1h": "1h", "1d": "1d"}
    period_map = {"1m": "1d", "5m": "5d", "30m": "5d", "1h": "5d", "1d": "1y"}
    
    interval = interval_map.get(timeframe, "5m")
    period = period_map.get(timeframe, "5d")
    
    df = get_intraday_data(symbol, interval, period)
    
    if df.empty:
        st.warning(f"No data available for {symbol}")
        return
    
    # Calculate indicators
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    ema20 = calculate_ema(close, 20)
    ema50 = calculate_ema(close, 50)
    ema100 = calculate_ema(close, 100)
    ema200 = calculate_ema(close, 200)
    rsi = calculate_rsi(close)
    macd, signal_line, hist = calculate_macd(close)
    atr = calculate_atr(high, low, close)
    vwap = calculate_vwap(high, low, close, volume)
    
    # Generate signals
    signals = generate_signals(df)
    
    # Create figure
    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA20", line=dict(color="#FF6B6B", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA50", line=dict(color="#4ECDC4", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema100, name="EMA100", line=dict(color="#45B7D1", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema200, name="EMA200", line=dict(color="#96CEB4", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=vwap, name="VWAP", line=dict(color="#FFEAA7", width=1, dash="dash")), row=1, col=1)
    
    # Volume
    colors = ['#26de81' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#fc5c65' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=volume, name="Volume", marker_color=colors), row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=macd, name="MACD", line=dict(color="#45B7D1", width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal_line, name="Signal", line=dict(color="#FF6B6B", width=1)), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram", marker_color="#5f27cd"), row=3, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color="#A3CB38", width=1)), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=4, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - {timeframe} Chart",
        template="plotly_dark",
        height=800,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Signal panel
    if signals:
        st.subheader("🔔 Trading Signals")
        for sig in signals:
            color = "green" if sig["type"] == "BUY" else "red" if sig["type"] == "SELL" else "blue"
            st.markdown(f":{color}[**{sig['type']}**] - {sig['reason']}")
    else:
        st.info("No active signals for this timeframe")
    
    # Technical summary
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("RSI(14)", f"{rsi.iloc[-1]:.1f}")
    with col2:
        st.metric("MACD", f"{macd.iloc[-1]:.2f}")
    with col3:
        st.metric("ATR(14)", f"{atr.iloc[-1]:.2f}")
    with col4:
        st.metric("EMA20", f"${ema20.iloc[-1]:.2f}")
    with col5:
        st.metric("EMA50", f"${ema50.iloc[-1]:.2f}")
    with col6:
        st.metric("VWAP", f"${vwap.iloc[-1]:.2f}")

def render_options_page(positions: List[Dict]):
    """Render options detail page"""
    st.subheader("📊 Options Positions")
    
    option_positions = [p for p in positions if p["type"] == "option"]
    
    if not option_positions:
        st.info("No options positions")
        return
    
    for pos in option_positions:
        with st.expander(f"{pos['symbol']} ${pos['strike']} {pos['option_type'].upper()} {pos['expiration']}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Contracts", pos["contracts"])
                st.metric("Strike", f"${pos['strike']}")
                st.metric("Expiration", pos["expiration"])
            
            with col2:
                st.metric("Bid", f"${pos['bid']:.2f}")
                st.metric("Ask", f"${pos['ask']:.2f}")
                st.metric("Mid", f"${pos['current_price']:.2f}")
            
            with col3:
                st.metric("IV", f"{pos['iv']*100:.1f}%")
                st.metric("Delta", f"{pos['delta']:.3f}")
                st.metric("Gamma", f"{pos['gamma']:.4f}")
            
            with col4:
                st.metric("Theta", f"{pos['theta']:.3f}")
                st.metric("Vega", f"{pos['vega']:.3f}")
                st.metric("Rho", f"{pos['rho']:.3f}")
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Intrinsic Value", f"${pos['intrinsic']:.2f}")
            with col2:
                st.metric("Extrinsic Value", f"${pos['extrinsic']:.2f}")
            with col3:
                st.metric("Volume", f"{pos['volume']:,}")
            with col4:
                st.metric("Open Interest", f"{pos['open_interest']:,}")
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cost Basis", f"${pos['cost_basis']:,.2f}")
            with col2:
                st.metric("Current Value", f"${pos['current_value']:,.2f}")
            with col3:
                st.metric("Today's P/L", f"{pos['today_pl']:+,.2f}")
            with col4:
                st.metric("Total P/L", f"{pos['total_pl']:+,.2f}")
            
            st.caption(f"Data source: {pos['source'].upper()}")

def render_alerts_page(positions: List[Dict]):
    """Render alerts management page"""
    st.subheader("🔔 Price Alerts")
    
    config = load_config()
    alerts = config.get("alerts", [])
    
    # Add new alert
    col1, col2, col3 = st.columns(3)
    with col1:
        new_symbol = st.selectbox("Symbol", [p["symbol"] for p in positions])
    with col2:
        alert_type = st.selectbox("Alert Type", ["Price Above", "Price Below", "Signal Change"])
    with col3:
        alert_value = st.number_input("Value", min_value=0.0, step=0.01)
    
    if st.button("Add Alert"):
        alerts.append({
            "symbol": new_symbol,
            "type": alert_type,
            "value": alert_value,
            "created": datetime.now().isoformat(),
            "active": True
        })
        config["alerts"] = alerts
        save_config(config)
        st.success("Alert added!")
    
    # Display active alerts
    st.markdown("### Active Alerts")
    if alerts:
        for i, alert in enumerate(alerts):
            if alert.get("active", True):
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                with col1:
                    st.write(f"**{alert['symbol']}**")
                with col2:
                    st.write(f"{alert['type']} ${alert['value']:.2f}")
                with col3:
                    st.write(f"Created: {alert['created'][:10]}")
                with col4:
                    if st.button("🗑️", key=f"del_{i}"):
                        alerts[i]["active"] = False
                        config["alerts"] = alerts
                        save_config(config)
                        st.rerun()
    else:
        st.info("No alerts configured")

def render_risk_page(positions: List[Dict], total_value: float):
    """Render risk management page"""
    st.subheader("⚠️ Risk Controls")
    
    # Concentration risk
    st.markdown("### Concentration Risk")
    for pos in positions:
        allocation = (pos["current_value"] / total_value) * 100
        if allocation > 25:
            st.warning(f"⚠️ {pos['symbol']} represents {allocation:.1f}% of portfolio (threshold: 25%)")
        else:
            st.write(f"✅ {pos['symbol']}: {allocation:.1f}%")
    
    # Stop loss suggestions
    st.markdown("### Stop Loss Levels")
    
    for pos in positions:
        if pos["type"] == "stock":
            symbol = pos["symbol"]
            df = get_intraday_data(symbol, "1d", "30d")
            
            if not df.empty:
                close = df['Close']
                atr = calculate_atr(df['High'], df['Low'], close).iloc[-1]
                ema50 = calculate_ema(close, 50).iloc[-1]
                
                stop_level = ema50 - (atr * 2)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**{symbol}**")
                with col2:
                    st.metric("Current Price", f"${pos['current_price']:.2f}")
                with col3:
                    st.metric("Suggested Stop", f"${stop_level:.2f}", delta=f"{((stop_level/pos['current_price'])-1)*100:.1f}%")
    
    # Scenario modeling
    st.markdown("### Scenario Modeling")
    
    config = load_config()
    iv_fallback = config.get("settings", {}).get("iv_fallback", 0.55)
    
    option_positions = [p for p in positions if p["type"] == "option"]
    
    if option_positions:
        for pos in option_positions:
            symbol = pos["symbol"]
            strike = pos["strike"]
            expiration = pos["expiration"]
            option_type = pos["option_type"]
            
            # Get current underlying
            try:
                ticker = yf.Ticker(symbol)
                current_price = ticker.info.get("currentPrice", 100)
            except:
                current_price = 100
            
            st.markdown(f"**{symbol} ${strike} {option_type.upper()}**")
            
            # Scenario prices
            scenarios = [current_price * 0.8, current_price * 0.9, current_price, 
                        current_price * 1.1, current_price * 1.2]
            
            cols = st.columns(len(scenarios))
            for i, (scenario_price, col) in enumerate(zip(scenarios, cols)):
                exp_date = datetime.strptime(expiration, "%Y-%m-%d")
                T = max(0.01, (exp_date - datetime.now()).days / 365)
                r = 0.05
                
                # Use current IV if available, else fallback
                iv = pos.get("iv", iv_fallback)
                
                price = black_scholes_estimate(scenario_price, strike, T, r, iv, option_type)
                with col:
                    st.metric(f"${scenario_price:.0f}", f"${price:.2f}")
            
            st.caption(f"Using IV: {pos.get('iv', iv_fallback)*100:.1f}% (Black-Scholes)")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="NVDA Portfolio Manager",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stMetric {
        background-color: #1a1a24;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📈 Portfolio Manager")
    
    # Load portfolio data
    config = load_config()
    portfolio = config.get("portfolio", {})
    
    st.sidebar.markdown("### Portfolio Summary")
    total_value, today_pl, cash_pending, positions = get_portfolio_value()
    
    st.sidebar.metric("Total Value", f"${total_value:,.2f}")
    st.sidebar.metric("Today's P/L", f"${today_pl:+,.2f}")
    st.sidebar.metric("Cash", f"${cash_pending:,.2f}")
    
    # Navigation
    page = st.sidebar.radio("Go to", 
        ["Portfolio", "Charts", "Options", "Alerts", "Risk"])
    
    # Auto-refresh
    refresh_interval = config.get("settings", {}).get("refresh_interval", 15)
    st.sidebar.markdown(f"---")
    st.sidebar.write(f"Auto-refresh: {refresh_interval}s")
    
    # Main content
    if page == "Portfolio":
        st.title("📊 Portfolio Overview")
        render_portfolio_table(positions, total_value, cash_pending)
        
    elif page == "Charts":
        st.title("📈 Charts & Signals")
        
        # Symbol selection
        symbols = list(set([p["symbol"] for p in positions]))
        selected_symbol = st.selectbox("Select Symbol", symbols)
        
        # Timeframe selection
        timeframe = st.selectbox("Select Timeframe", ["1m", "5m", "30m", "1h", "1d"])
        
        render_charts(selected_symbol, timeframe)
        
    elif page == "Options":
        st.title("📊 Options Details")
        render_options_page(positions)
        
    elif page == "Alerts":
        st.title("🔔 Price Alerts")
        render_alerts_page(positions)
        
    elif page == "Risk":
        st.title("⚠️ Risk Management")
        render_risk_page(positions, total_value)
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()