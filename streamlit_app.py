import streamlit as st
import pandas as pd
import json
import re
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import ta  # Technical Analysis library
import base64
import fitz # PyMuPDF for reading PDFs

# --- AI Imports ---
try:
    import google.generativeai as genai
    from google.generativeai import types
except ImportError:
    st.error("Google Generative AI library not found. Please install it using `pip install google-generativeai`.")
    st.stop()
    
# --- KiteConnect Imports ---
try:
    from kiteconnect import KiteConnect
except ImportError:
    st.error("KiteConnect library not found. Please install it using `pip install kiteconnect`.")
    st.stop()


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Invsion Connect - Portfolio Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("Invsion Connect")
st.markdown("A comprehensive platform for fetching market data, validating investment compliance, and AI-powered analysis.")

# --- Global Constants & Session State Initialization ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"
BENCHMARK_SYMBOL = "NIFTY 50"

# Initialize session state variables
if "kite_access_token" not in st.session_state: st.session_state["kite_access_token"] = None
if "kite_login_response" not in st.session_state: st.session_state["kite_login_response"] = None
if "instruments_df" not in st.session_state: st.session_state["instruments_df"] = pd.DataFrame()
if "historical_data" not in st.session_state: st.session_state["historical_data"] = pd.DataFrame()
if "last_fetched_symbol" not in st.session_state: st.session_state["last_fetched_symbol"] = None
if "current_market_data" not in st.session_state: st.session_state["current_market_data"] = None
if "holdings_data" not in st.session_state: st.session_state["holdings_data"] = None
if "compliance_results_df" not in st.session_state: st.session_state["compliance_results_df"] = pd.DataFrame()
if "advanced_metrics" not in st.session_state: st.session_state["advanced_metrics"] = None
if "ai_analysis_response" not in st.session_state: st.session_state["ai_analysis_response"] = None
if "security_level_compliance" not in st.session_state: st.session_state["security_level_compliance"] = pd.DataFrame()
if "breach_alerts" not in st.session_state: st.session_state["breach_alerts"] = []


# --- Load Credentials from Streamlit Secrets ---
def load_secrets():
    secrets = st.secrets
    kite_conf = secrets.get("kite", {})
    gemini_conf = secrets.get("google_gemini", {})
    
    errors = []
    if not kite_conf.get("api_key") or not kite_conf.get("api_secret") or not kite_conf.get("redirect_uri"):
        errors.append("Kite credentials (api_key, api_secret, redirect_uri)")
    if not gemini_conf.get("api_key"):
        errors.append("Google Gemini API key")

    if errors:
        st.error(f"Missing required credentials in `.streamlit/secrets.toml`: {', '.join(errors)}.")
        st.info("Ensure your `secrets.toml` includes both [kite] and [google_gemini] sections.")
        st.stop()
    return kite_conf, gemini_conf

KITE_CREDENTIALS, GEMINI_CREDENTIALS = load_secrets()
genai.configure(api_key=GEMINI_CREDENTIALS["api_key"])


# --- KiteConnect Client Initialization ---
@st.cache_resource(ttl=3600)
def init_kite_unauth_client(api_key: str) -> KiteConnect:
    return KiteConnect(api_key=api_key)

kite_unauth_client = init_kite_unauth_client(KITE_CREDENTIALS["api_key"])
login_url = kite_unauth_client.login_url()

# --- Utility Functions ---

def get_authenticated_kite_client(api_key: str | None, access_token: str | None) -> KiteConnect | None:
    if api_key and access_token:
        k_instance = KiteConnect(api_key=api_key)
        k_instance.set_access_token(access_token)
        return k_instance
    return None

@st.cache_data(ttl=86400, show_spinner="Loading instruments...")
def load_instruments_cached(api_key: str, access_token: str, exchange: str = None) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance: return pd.DataFrame({"_error": ["Kite not authenticated."]})
    try:
        instruments = kite_instance.instruments(exchange) if exchange else kite_instance.instruments()
        df = pd.DataFrame(instruments)
        if "instrument_token" in df.columns: df["instrument_token"] = df["instrument_token"].astype("int64")
        if 'tradingsymbol' in df.columns and 'name' in df.columns: df = df[['instrument_token', 'tradingsymbol', 'name', 'exchange']]
        return df
    except Exception as e: return pd.DataFrame({"_error": [f"Failed to load instruments: {e}"]})

@st.cache_data(ttl=60)
def get_ltp_price_cached(api_key: str, access_token: str, symbol: str, exchange: str = DEFAULT_EXCHANGE):
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance: return {"_error": "Kite not authenticated."}
    try: return kite_instance.ltp([f"{exchange.upper()}:{symbol.upper()}"]).get(f"{exchange.upper()}:{symbol.upper()}")
    except Exception as e: return {"_error": str(e)}

@st.cache_data(ttl=3600)
def get_historical_data_cached(api_key: str, access_token: str, symbol: str, from_date: datetime.date, to_date: datetime.date, interval: str, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance: return pd.DataFrame({"_error": ["Kite not authenticated."]})
    instruments_df = load_instruments_cached(api_key, access_token)
    token = find_instrument_token(instruments_df, symbol, exchange)
    if not token and symbol in ["NIFTY BANK", "NIFTYBANK", "BANKNIFTY", BENCHMARK_SYMBOL, "SENSEX"]:
        index_exchange = "NSE" if symbol not in ["SENSEX"] else "BSE"
        instruments_secondary = load_instruments_cached(api_key, access_token, index_exchange)
        token = find_instrument_token(instruments_secondary, symbol, index_exchange)
    if not token: return pd.DataFrame({"_error": [f"Instrument token not found for {symbol}."]})
    try:
        data = kite_instance.historical_data(token, from_date=datetime.combine(from_date, datetime.min.time()), to_date=datetime.combine(to_date, datetime.max.time()), interval=interval)
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]); df.set_index("date", inplace=True); df.sort_index(inplace=True)
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce'); df.dropna(subset=['close'], inplace=True)
        return df
    except Exception as e: return pd.DataFrame({"_error": [str(e)]})

def find_instrument_token(df: pd.DataFrame, tradingsymbol: str, exchange: str = DEFAULT_EXCHANGE) -> int | None:
    if df.empty: return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    hits = df[mask]
    return int(hits.iloc[0]["instrument_token"]) if not hits.empty else None

def add_technical_indicators(df: pd.DataFrame, sma_periods, ema_periods, rsi_window, macd_fast, macd_slow, macd_signal, bb_window, bb_std_dev) -> pd.DataFrame:
    if df.empty or 'close' not in df.columns: return df.copy()
    df_copy = df.copy()
    for period in sma_periods:
        if period > 0: df_copy[f'SMA_{period}'] = ta.trend.sma_indicator(df_copy['close'], window=period)
    for period in ema_periods:
        if period > 0: df_copy[f'EMA_{period}'] = ta.trend.ema_indicator(df_copy['close'], window=period)
    df_copy['RSI'] = ta.momentum.rsi(df_copy['close'], window=rsi_window)
    macd_obj = ta.trend.MACD(df_copy['close'], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    df_copy['MACD'], df_copy['MACD_signal'], df_copy['MACD_hist'] = macd_obj.macd(), macd_obj.macd_signal(), macd_obj.macd_diff()
    bollinger = ta.volatility.BollingerBands(df_copy['close'], window=bb_window, window_dev=bb_std_dev)
    df_copy['Bollinger_High'], df_copy['Bollinger_Low'], df_copy['Bollinger_Mid'], df_copy['Bollinger_Width'] = bollinger.bollinger_hband(), bollinger.bollinger_lband(), bollinger.bollinger_mavg(), bollinger.bollinger_wband()
    df_copy['Daily_Return'] = df_copy['close'].pct_change() * 100
    df_copy.fillna(method='bfill', inplace=True); df_copy.fillna(method='ffill', inplace=True)
    return df_copy.dropna()

def calculate_performance_metrics(returns_series: pd.Series, risk_free_rate: float = 0.0) -> dict:
    if returns_series.empty or len(returns_series) < 2: return {}
    daily_returns_decimal = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
    if daily_returns_decimal.empty: return {}
    cumulative_returns = (1 + daily_returns_decimal).cumprod() - 1
    total_return = cumulative_returns.iloc[-1] * 100 if not cumulative_returns.empty else 0
    annualized_return = ((1 + daily_returns_decimal.mean()) ** TRADING_DAYS_PER_YEAR - 1) * 100
    annualized_volatility = daily_returns_decimal.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    risk_free_rate_decimal = risk_free_rate / 100.0
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else np.nan
    if not cumulative_returns.empty: max_drawdown = (((1 + cumulative_returns).cummax() - (1 + cumulative_returns)) / (1 + cumulative_returns).cummax()).max() * 100
    else: max_drawdown = np.nan
    def round_if_float(x): return round(x, 4) if isinstance(x, (int, float)) and not np.isnan(x) else np.nan
    return {"Total Return (%)": round_if_float(total_return), "Annualized Return (%)": round_if_float(annualized_return), "Annualized Volatility (%)": round_if_float(annualized_volatility), "Sharpe Ratio": round_if_float(sharpe_ratio), "Max Drawdown (%)": round_if_float(max_drawdown)}


# --- ENHANCED COMPLIANCE FUNCTIONS ---

def parse_and_validate_rules_enhanced(rules_text: str, portfolio_df: pd.DataFrame):
    """Enhanced rule parser with comprehensive validation capabilities"""
    results = []
    if not rules_text.strip() or portfolio_df.empty: return results
    
    # Prepare aggregations
    sector_weights = portfolio_df.groupby('Industry')['Weight %'].sum()
    stock_weights = portfolio_df.set_index('Symbol')['Weight %']
    rating_weights = portfolio_df.groupby('Rating')['Weight %'].sum() if 'Rating' in portfolio_df.columns else pd.Series()
    asset_class_weights = portfolio_df.groupby('Asset Class')['Weight %'].sum() if 'Asset Class' in portfolio_df.columns else pd.Series()
    market_cap_weights = portfolio_df.groupby('Market Cap')['Weight %'].sum() if 'Market Cap' in portfolio_df.columns else pd.Series()
    
    def check_pass(actual, op, threshold):
        if op == '>': return actual > threshold
        if op == '<': return actual < threshold
        if op == '>=': return actual >= threshold
        if op == '<=': return actual <= threshold
        if op == '=': return actual == threshold
        return False
    
    for rule in rules_text.strip().split('\n'):
        rule = rule.strip()
        if not rule or rule.startswith('#'): continue
        parts = re.split(r'\s+', rule)
        rule_type = parts[0].upper()
        
        try:
            actual_value = None
            details = ""
            
            if len(parts) < 3:
                results.append({'rule': rule, 'status': 'Error', 'details': 'Invalid format.', 'severity': 'N/A'})
                continue
            
            op = parts[-2]
            if op not in ['>', '<', '>=', '<=', '=']:
                results.append({'rule': rule, 'status': 'Error', 'details': f"Invalid operator '{op}'.", 'severity': 'N/A'})
                continue
            
            threshold = float(parts[-1].replace('%', ''))
            
            # STOCK level rules
            if rule_type == 'STOCK' and len(parts) == 4:
                symbol = parts[1].upper()
                if symbol in stock_weights.index:
                    actual_value = stock_weights.get(symbol, 0.0)
                    details = f"Actual for {symbol}: {actual_value:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': f"Symbol '{symbol}' not found.", 'severity': 'N/A'})
                    continue
            
            # SECTOR level rules
            elif rule_type == 'SECTOR':
                sector_name = ' '.join(parts[1:-2]).upper()
                matching_sector = next((s for s in sector_weights.index if s.upper() == sector_name), None)
                if matching_sector:
                    actual_value = sector_weights.get(matching_sector, 0.0)
                    details = f"Actual for {matching_sector}: {actual_value:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': f"Sector '{sector_name}' not found.", 'severity': 'N/A'})
                    continue
            
            # RATING level rules
            elif rule_type == 'RATING':
                rating_name = ' '.join(parts[1:-2]).upper()
                actual_value = rating_weights.get(rating_name, 0.0)
                details = f"Actual for {rating_name}: {actual_value:.2f}%"
            
            # ASSET_CLASS rules
            elif rule_type == 'ASSET_CLASS':
                class_name = ' '.join(parts[1:-2]).upper()
                actual_value = asset_class_weights.get(class_name, 0.0)
                details = f"Actual for {class_name}: {actual_value:.2f}%"
            
            # MARKET_CAP rules
            elif rule_type == 'MARKET_CAP':
                cap_name = ' '.join(parts[1:-2]).upper()
                actual_value = market_cap_weights.get(cap_name, 0.0)
                details = f"Actual for {cap_name}: {actual_value:.2f}%"
            
            # TOP_N_STOCKS rules
            elif rule_type == 'TOP_N_STOCKS' and len(parts) == 4:
                n = int(parts[1])
                actual_value = portfolio_df.nlargest(n, 'Weight %')['Weight %'].sum()
                details = f"Actual weight of top {n} stocks: {actual_value:.2f}%"
            
            # TOP_N_SECTORS rules
            elif rule_type == 'TOP_N_SECTORS' and len(parts) == 4:
                n = int(parts[1])
                actual_value = sector_weights.nlargest(n).sum()
                details = f"Actual weight of top {n} sectors: {actual_value:.2f}%"
            
            # COUNT_STOCKS rules
            elif rule_type == 'COUNT_STOCKS' and len(parts) == 3:
                actual_value = len(portfolio_df)
                details = f"Actual count: {actual_value}"
            
            # COUNT_SECTORS rules
            elif rule_type == 'COUNT_SECTORS' and len(parts) == 3:
                actual_value = portfolio_df['Industry'].nunique()
                details = f"Actual count: {actual_value}"
            
            # SINGLE_ISSUER_GROUP rules (group exposure)
            elif rule_type == 'ISSUER_GROUP':
                group_name = ' '.join(parts[1:-2]).upper()
                if 'Issuer Group' in portfolio_df.columns:
                    actual_value = portfolio_df[portfolio_df['Issuer Group'].str.upper() == group_name]['Weight %'].sum()
                    details = f"Actual for {group_name}: {actual_value:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Issuer Group' not found.", 'severity': 'N/A'})
                    continue
            
            # LIQUIDITY rules (based on avg volume)
            elif rule_type == 'MIN_LIQUIDITY' and len(parts) == 4:
                symbol = parts[1].upper()
                if 'Avg Volume (90d)' in portfolio_df.columns:
                    stock_row = portfolio_df[portfolio_df['Symbol'] == symbol]
                    if not stock_row.empty:
                        actual_value = stock_row['Avg Volume (90d)'].values[0]
                        details = f"Actual volume for {symbol}: {actual_value:,.0f}"
                    else:
                        results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': f"Symbol '{symbol}' not found.", 'severity': 'N/A'})
                        continue
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Avg Volume (90d)' not found.", 'severity': 'N/A'})
                    continue
            
            # UNRATED_EXPOSURE rules
            elif rule_type == 'UNRATED_EXPOSURE' and len(parts) == 3:
                if 'Rating' in portfolio_df.columns:
                    unrated_mask = portfolio_df['Rating'].isin(['UNRATED', 'NR', 'NOT RATED', ''])
                    actual_value = portfolio_df[unrated_mask]['Weight %'].sum()
                    details = f"Actual unrated exposure: {actual_value:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Rating' not found.", 'severity': 'N/A'})
                    continue
            
            # FOREIGN_EXPOSURE rules
            elif rule_type == 'FOREIGN_EXPOSURE' and len(parts) == 3:
                if 'Country' in portfolio_df.columns:
                    foreign_mask = portfolio_df['Country'].str.upper() != 'INDIA'
                    actual_value = portfolio_df[foreign_mask]['Weight %'].sum()
                    details = f"Actual foreign exposure: {actual_value:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Country' not found.", 'severity': 'N/A'})
                    continue
            
            # DERIVATIVES_EXPOSURE rules
            elif rule_type == 'DERIVATIVES_EXPOSURE' and len(parts) == 3:
                if 'Instrument Type' in portfolio_df.columns:
                    deriv_mask = portfolio_df['Instrument Type'].str.upper().isin(['FUTURES', 'OPTIONS', 'SWAPS'])
                    actual_value = portfolio_df[deriv_mask]['Weight %'].sum()
                    details = f"Actual derivatives exposure: {actual_value:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Instrument Type' not found.", 'severity': 'N/A'})
                    continue
            
            else:
                results.append({'rule': rule, 'status': 'Error', 'details': 'Unrecognized rule format.', 'severity': 'N/A'})
                continue
            
            if actual_value is not None:
                passed = check_pass(actual_value, op, threshold)
                status = "✅ PASS" if passed else "❌ FAIL"
                
                # Determine severity
                if not passed:
                    breach_magnitude = abs(actual_value - threshold)
                    if breach_magnitude > threshold * 0.2:  # >20% breach
                        severity = "🔴 Critical"
                    elif breach_magnitude > threshold * 0.1:  # >10% breach
                        severity = "🟠 High"
                    else:
                        severity = "🟡 Medium"
                else:
                    severity = "✅ Compliant"
                
                results.append({
                    'rule': rule,
                    'status': status,
                    'details': f"{details} | Rule: {op} {threshold}",
                    'severity': severity,
                    'actual_value': actual_value,
                    'threshold': threshold,
                    'breach_amount': actual_value - threshold if not passed else 0
                })
        
        except (ValueError, IndexError) as e:
            results.append({'rule': rule, 'status': 'Error', 'details': f"Parse error: {e}", 'severity': 'N/A'})
    
    return results


def calculate_security_level_compliance(portfolio_df: pd.DataFrame, rules_config: dict):
    """Calculate compliance metrics at individual security level"""
    if portfolio_df.empty:
        return pd.DataFrame()
    
    security_compliance = portfolio_df.copy()
    
    # Single stock limit check
    single_stock_limit = rules_config.get('single_stock_limit', 10.0)
    security_compliance['Stock Limit Breach'] = security_compliance['Weight %'].apply(
        lambda x: '❌ Breach' if x > single_stock_limit else '✅ Compliant'
    )
    security_compliance['Stock Limit Gap (%)'] = single_stock_limit - security_compliance['Weight %']
    
    # Liquidity check (if data available)
    if 'Avg Volume (90d)' in security_compliance.columns:
        min_liquidity = rules_config.get('min_liquidity', 100000)
        security_compliance['Liquidity Status'] = security_compliance['Avg Volume (90d)'].apply(
            lambda x: '✅ Adequate' if x >= min_liquidity else '⚠️ Low'
        )
    
    # Rating check (if data available)
    if 'Rating' in security_compliance.columns:
        min_rating = rules_config.get('min_rating', ['AAA', 'AA+', 'AA', 'AA-', 'A+'])
        security_compliance['Rating Compliance'] = security_compliance['Rating'].apply(
            lambda x: '✅ Compliant' if x in min_rating else '⚠️ Below Threshold'
        )
    
    # Concentration risk flag
    security_compliance['Concentration Risk'] = security_compliance['Weight %'].apply(
        lambda x: '🔴 High' if x > 8 else '🟡 Medium' if x > 5 else '🟢 Low'
    )
    
    return security_compliance


def calculate_advanced_metrics(portfolio_df, api_key, access_token):
    """Enhanced advanced metrics calculation"""
    symbols = portfolio_df['Symbol'].tolist()
    weights = (portfolio_df['Real-time Value (Rs)'] / portfolio_df['Real-time Value (Rs)'].sum()).values
    from_date = datetime.now().date() - timedelta(days=366)
    to_date = datetime.now().date()
    
    returns_df = pd.DataFrame()
    failed_symbols = []
    
    progress_bar = st.progress(0, "Fetching historical data for metrics...")
    
    for i, symbol in enumerate(symbols):
        hist_data = get_historical_data_cached(api_key, access_token, symbol, from_date, to_date, 'day')
        if not hist_data.empty and '_error' not in hist_data.columns:
            returns_df[symbol] = hist_data['close'].pct_change()
        else:
            failed_symbols.append(symbol)
        progress_bar.progress((i + 1) / len(symbols), f"Fetching data for {symbol}...")
    
    if failed_symbols:
        st.warning(f"Could not fetch historical data for: {', '.join(failed_symbols)}. They will be excluded.")
    
    returns_df.dropna(how='all', inplace=True)
    returns_df.fillna(0, inplace=True)
    
    if returns_df.empty:
        st.error("Not enough historical data to calculate advanced metrics.")
        return None
    
    # Portfolio returns
    portfolio_returns = returns_df.dot(weights)
    
    # VaR calculations
    var_95 = portfolio_returns.quantile(0.05)
    var_99 = portfolio_returns.quantile(0.01)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    
    # Benchmark data
    benchmark_data = get_historical_data_cached(api_key, access_token, BENCHMARK_SYMBOL, from_date, to_date, 'day')
    
    if benchmark_data.empty or '_error' in benchmark_data.columns:
        st.error(f"Could not fetch benchmark data. Beta cannot be calculated.")
        portfolio_beta = None
        alpha = None
        tracking_error = None
        information_ratio = None
    else:
        benchmark_returns = benchmark_data['close'].pct_change()
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
        aligned_returns.columns = ['portfolio', 'benchmark']
        
        # Beta
        covariance = aligned_returns.cov().iloc[0, 1]
        benchmark_variance = aligned_returns['benchmark'].var()
        portfolio_beta = covariance / benchmark_variance if benchmark_variance > 0 else None
        
        # Alpha (annualized)
        portfolio_annual_return = ((1 + aligned_returns['portfolio'].mean()) ** 252 - 1)
        benchmark_annual_return = ((1 + aligned_returns['benchmark'].mean()) ** 252 - 1)
        risk_free_rate = 0.06  # 6% assumed
        
        if portfolio_beta:
            alpha = portfolio_annual_return - (risk_free_rate + portfolio_beta * (benchmark_annual_return - risk_free_rate))
        else:
            alpha = None
        
        # Tracking Error
        tracking_diff = aligned_returns['portfolio'] - aligned_returns['benchmark']
        tracking_error = tracking_diff.std() * np.sqrt(252)
        
        # Information Ratio
        if tracking_error and tracking_error > 0:
            information_ratio = (portfolio_annual_return - benchmark_annual_return) / tracking_error
        else:
            information_ratio = None
    
    # Sortino Ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    portfolio_annual_return = ((1 + portfolio_returns.mean()) ** 252 - 1)
    sortino_ratio = (portfolio_annual_return - 0.06) / downside_std if downside_std > 0 else None
    
    # Correlation matrix
    correlation_matrix = returns_df.corr()
    avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
    
    # Diversification ratio
    portfolio_vol = portfolio_returns.std() * np.sqrt(252)
    weighted_vol = np.sum(weights * returns_df.std() * np.sqrt(252))
    diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else None
    
    progress_bar.empty()
    
    return {
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "beta": portfolio_beta,
        "alpha": alpha,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "sortino_ratio": sortino_ratio,
        "avg_correlation": avg_correlation,
        "diversification_ratio": diversification_ratio,
        "portfolio_volatility": portfolio_vol
    }


# --- Sidebar: Kite Login ---
with st.sidebar:
    st.markdown("### 1. Login to Kite Connect")
    if not st.session_state["kite_access_token"]:
        st.markdown(f"Click the link, authorize, and you'll be redirected back.")
        st.link_button("🔗 Open Kite login", login_url, use_container_width=True)
    request_token_param = st.query_params.get("request_token")
    if request_token_param and not st.session_state["kite_access_token"]:
        with st.spinner("Authenticating..."):
            try:
                data = kite_unauth_client.generate_session(request_token_param, api_secret=KITE_CREDENTIALS["api_secret"])
                st.session_state["kite_access_token"] = data.get("access_token")
                st.session_state["kite_login_response"] = data
                st.sidebar.success("Kite authentication successful.")
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Authentication failed: {e}")
    if st.session_state["kite_access_token"]:
        st.success("Kite Authenticated ✅")
        if st.sidebar.button("Logout from Kite", use_container_width=True):
            st.session_state.clear()
            st.success("Logged out from Kite.")
            st.rerun()
    else:
        st.info("Not authenticated with Kite yet.")
    st.markdown("---")
    st.markdown("### 2. Quick Data Access")
    if st.session_state["kite_access_token"]:
        if st.button("Fetch Current Holdings", key="sidebar_fetch_holdings_btn", use_container_width=True):
            current_k_client_for_sidebar = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])
            try:
                holdings = current_k_client_for_sidebar.holdings()
                st.session_state["holdings_data"] = pd.DataFrame(holdings)
                st.success(f"Fetched {len(holdings)} holdings.")
            except Exception as e:
                st.error(f"Error fetching holdings: {e}")
        if st.session_state.get("holdings_data") is not None and not st.session_state["holdings_data"].empty:
            with st.expander("Show Holdings"):
                st.dataframe(st.session_state["holdings_data"])
                st.download_button("Download Holdings (CSV)", st.session_state["holdings_data"].to_csv(index=False).encode('utf-8'), "kite_holdings.csv", "text/csv", key="download_holdings_sidebar_csv", use_container_width=True)
    else:
        st.info("Login to Kite to access quick data.")

# --- Authenticated KiteConnect client (used by main tabs) ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

# --- Main UI - Tabs for modules ---
tabs = st.tabs(["📈 Market & Historical", "💼 Investment Compliance", "🤖 AI-Powered Analysis"])
tab_market, tab_compliance, tab_ai = tabs


# --- Tab Logic Functions ---

def render_market_historical_tab(kite_client, api_key, access_token):
    st.header("📈 Market Data & Historical Candles with TA")
    if not kite_client:
        st.info("Login first to fetch market data.")
        return
    st.subheader("Current Market Data Snapshot")
    col_market_quote1, col_market_quote2 = st.columns([1, 2])
    with col_market_quote1:
        q_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="market_exchange_tab")
        q_symbol = st.text_input("Tradingsymbol", value="RELIANCE", key="market_symbol_tab")
        if st.button("Get Market Data", key="get_market_data_btn"):
            ltp_data = get_ltp_price_cached(api_key, access_token, q_symbol, q_exchange)
            if ltp_data and "_error" not in ltp_data:
                st.session_state["current_market_data"] = ltp_data
            else:
                st.error(f"Failed: {ltp_data.get('_error', 'Unknown error')}")
    with col_market_quote2:
        if st.session_state.get("current_market_data"):
            st.json(st.session_state["current_market_data"])
        else:
            st.info("Market data will appear here.")
    st.markdown("---")
    st.subheader("Historical Price Data & Technical Analysis")
    hist_symbol = st.text_input("Tradingsymbol", value="INFY", key="hist_sym_tab_input_ta")
    col_fetch, col_interval, col_dates = st.columns(3)
    with col_dates:
        from_date = st.date_input("From Date", value=datetime.now().date() - timedelta(days=365), key="from_dt_tab_input")
        to_date = st.date_input("To Date", value=datetime.now().date(), key="to_dt_tab_input")
    with col_interval:
        interval = st.selectbox("Interval", ["day", "week", "minute", "5minute", "30minute", "month"], index=0, key="hist_interval_selector_ta")
    with col_fetch:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Fetch & Prepare Data", key="fetch_historical_data_ta_btn", type="primary"):
            with st.spinner(f"Fetching data..."):
                df_hist = get_historical_data_cached(api_key, access_token, hist_symbol, from_date, to_date, interval, DEFAULT_EXCHANGE)
                if isinstance(df_hist, pd.DataFrame) and "_error" not in df_hist.columns:
                    st.session_state["historical_data"] = df_hist
                    st.session_state["last_fetched_symbol"] = hist_symbol
                else:
                    st.error(f"Fetch failed: {df_hist.get('_error', 'Unknown error')}")
    if not st.session_state.get("historical_data", pd.DataFrame()).empty:
        df = st.session_state["historical_data"]
        with st.expander("Technical Indicator & Plotting Options"):
            ta_c1, ta_c2, ta_c3 = st.columns(3)
            with ta_c1:
                sma_periods_str = st.text_input("SMA Periods", "20,50")
                ema_periods_str = st.text_input("EMA Periods", "12,26")
                rsi_window = st.number_input("RSI Window", 5, 50, 14)
            with ta_c2:
                macd_fast = st.number_input("MACD Fast", 5, 50, 12)
                macd_slow = st.number_input("MACD Slow", 10, 100, 26)
                macd_signal = st.number_input("MACD Signal", 5, 50, 9)
            with ta_c3:
                bb_window = st.number_input("Bollinger Window", 5, 50, 20)
                bb_std_dev = st.number_input("Bollinger Std Dev", 1.0, 4.0, 2.0, 0.5)
                chart_type = st.selectbox("Chart Style", ["Candlestick", "Line"])
                indicators_to_plot = st.multiselect("Plot on Price Chart", ["SMA", "EMA", "Bollinger Bands"])
            sma_periods = [int(p.strip()) for p in sma_periods_str.split(',') if p.strip().isdigit()]
            ema_periods = [int(p.strip()) for p in ema_periods_str.split(',') if p.strip().isdigit()]
            df_with_ta = add_technical_indicators(df, sma_periods, ema_periods, rsi_window, macd_fast, macd_slow, macd_signal, bb_window, bb_std_dev)
        st.subheader(f"Technical Analysis for {st.session_state['last_fetched_symbol']} ({interval})")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.1, 0.2, 0.2])
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Price'), row=1, col=1)
        if "SMA" in indicators_to_plot:
            for p in sma_periods:
                fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta.get(f'SMA_{p}'), mode='lines', name=f'SMA {p}'), row=1, col=1)
        if "EMA" in indicators_to_plot:
            for p in ema_periods:
                fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta.get(f'EMA_{p}'), mode='lines', name=f'EMA {p}'), row=1, col=1)
        if "Bollinger Bands" in indicators_to_plot:
            fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['Bollinger_High'], mode='lines', line=dict(width=0.5, color='gray'), name='BB High'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['Bollinger_Low'], mode='lines', line=dict(width=0.5, color='gray'), fill='tonexty', fillcolor='rgba(128,128,128,0.2)', name='BB Low'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['RSI'], mode='lines', name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)
        fig.add_trace(go.Bar(x=df_with_ta.index, y=df_with_ta['MACD_hist'], name='MACD Hist', marker_color='orange'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['MACD'], mode='lines', name='MACD Line', line=dict(color='blue')), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['MACD_signal'], mode='lines', name='MACD Signal', line=dict(color='red')), row=4, col=1)
        fig.update_layout(height=1000, xaxis_rangeslider_visible=False, title_text=f"{st.session_state['last_fetched_symbol']} Technical Analysis", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Performance Snapshot")
        metrics = calculate_performance_metrics(df['close'].pct_change(), risk_free_rate=6.0)
        st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}).style.format("{:.4f}"))


def render_investment_compliance_tab(kite_client, api_key, access_token):
    st.header("💼 Enhanced Investment Compliance & Portfolio Analysis")
    st.markdown("Comprehensive compliance validation at portfolio and security level with regulatory oversight.")
    
    if not kite_client:
        st.info("Please login to Kite Connect to fetch live prices.")
        return
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("1. Upload Portfolio")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Required: 'Symbol', 'Industry', 'Quantity', 'Market/Fair Value(Rs. in Lacs)'.")
        
        st.markdown("##### Compliance Configuration")
        with st.expander("⚙️ Set Compliance Thresholds"):
            single_stock_limit = st.number_input("Single Stock Limit (%)", 1.0, 25.0, 10.0, 0.5, help="Maximum weight for any single stock")
            single_sector_limit = st.number_input("Single Sector Limit (%)", 5.0, 50.0, 25.0, 1.0, help="Maximum weight for any single sector")
            top_10_limit = st.number_input("Top 10 Holdings Limit (%)", 20.0, 80.0, 50.0, 5.0, help="Maximum combined weight of top 10 holdings")
            min_holdings = st.number_input("Minimum Holdings Count", 10, 200, 30, 5, help="Minimum number of securities")
            unrated_limit = st.number_input("Unrated Securities Limit (%)", 0.0, 30.0, 10.0, 1.0, help="Maximum weight in unrated securities")
    
    with col2:
        st.subheader("2. Define Custom Compliance Rules")
        rules_text = st.text_area("Enter one rule per line.", height=200, key="compliance_rules_input", 
                                   help="Define custom rules for validation", 
                                   value="""# Example Rules (remove/modify as needed)
# STOCK RELIANCE < 10
# SECTOR BANKING < 25
# TOP_N_STOCKS 10 <= 50
# RATING AAA >= 30
# UNRATED_EXPOSURE <= 10""")
        
        with st.expander("📖 Comprehensive Rule Syntax Guide"):
            st.markdown("""
            **Available Rule Types:**
            
            **1. Stock Level:**
            - `STOCK [Symbol] <op> [Value]%` - Single stock weight limit
            - Example: `STOCK RELIANCE < 10`
            
            **2. Sector Level:**
            - `SECTOR [Name] <op> [Value]%` - Sector exposure limit
            - Example: `SECTOR BANKING < 25`
            
            **3. Concentration Rules:**
            - `TOP_N_STOCKS [N] <op> [Value]%` - Top N stocks combined weight
            - `TOP_N_SECTORS [N] <op> [Value]%` - Top N sectors combined weight
            - Example: `TOP_N_STOCKS 5 <= 35`
            
            **4. Count Rules:**
            - `COUNT_STOCKS <op> [Value]` - Total number of holdings
            - `COUNT_SECTORS <op> [Value]` - Number of sectors
            - Example: `COUNT_STOCKS >= 30`
            
            **5. Rating & Quality:**
            - `RATING [Rating] <op> [Value]%` - Specific rating exposure
            - `UNRATED_EXPOSURE <op> [Value]%` - Unrated securities limit
            - Example: `RATING AAA >= 40`
            
            **6. Asset Class:**
            - `ASSET_CLASS [Class] <op> [Value]%` - Asset class allocation
            - Example: `ASSET_CLASS EQUITY >= 80`
            
            **7. Market Cap:**
            - `MARKET_CAP [Cap] <op> [Value]%` - Market cap exposure
            - Example: `MARKET_CAP LARGE >= 70`
            
            **8. Issuer Group:**
            - `ISSUER_GROUP [Group] <op> [Value]%` - Group company exposure
            - Example: `ISSUER_GROUP TATA < 15`
            
            **9. Liquidity:**
            - `MIN_LIQUIDITY [Symbol] >= [Volume]` - Minimum trading volume
            - Example: `MIN_LIQUIDITY INFY >= 100000`
            
            **10. Foreign & Derivatives:**
            - `FOREIGN_EXPOSURE <op> [Value]%` - International holdings
            - `DERIVATIVES_EXPOSURE <op> [Value]%` - Derivatives position
            
            **Operators:** `>`, `<`, `>=`, `<=`, `=`
            """)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = [str(col).strip().lower().replace(' ', '_').replace('.', '').replace('/', '_') for col in df.columns]
            
            header_map = {
                'isin': 'ISIN',
                'name_of_the_instrument': 'Name',
                'symbol': 'Symbol',
                'industry': 'Industry',
                'quantity': 'Quantity',
                'rating': 'Rating',
                'asset_class': 'Asset Class',
                'market_cap': 'Market Cap',
                'issuer_group': 'Issuer Group',
                'country': 'Country',
                'instrument_type': 'Instrument Type',
                'avg_volume_(90d)': 'Avg Volume (90d)',
                'market_fair_value(rs_in_lacs)': 'Uploaded Value (Lacs)'
            }
            df = df.rename(columns=header_map)
            
            # Normalize categorical columns
            for col in ['Rating', 'Asset Class', 'Industry', 'Market Cap', 'Issuer Group', 'Country', 'Instrument Type']:
                if col in df.columns:
                    df[col] = df[col].fillna('UNKNOWN').str.strip().str.upper()
            
            if st.button("🔍 Analyze & Validate Portfolio", type="primary", use_container_width=True):
                with st.spinner("Fetching live prices and performing comprehensive analysis..."):
                    symbols = df['Symbol'].unique().tolist()
                    ltp_data = kite_client.ltp([f"{DEFAULT_EXCHANGE}:{s}" for s in symbols])
                    prices = {sym: ltp_data.get(f"{DEFAULT_EXCHANGE}:{sym}", {}).get('last_price') for sym in symbols}
                    
                    df_results = df.copy()
                    df_results['LTP'] = df_results['Symbol'].map(prices)
                    df_results['Real-time Value (Rs)'] = (df_results['LTP'] * pd.to_numeric(df_results['Quantity'], errors='coerce')).fillna(0)
                    total_value = df_results['Real-time Value (Rs)'].sum()
                    df_results['Weight %'] = (df_results['Real-time Value (Rs)'] / total_value * 100) if total_value > 0 else 0
                    
                    # Calculate security-level compliance
                    rules_config = {
                        'single_stock_limit': single_stock_limit,
                        'single_sector_limit': single_sector_limit,
                        'min_liquidity': 100000  # Default
                    }
                    
                    security_compliance = calculate_security_level_compliance(df_results, rules_config)
                    
                    st.session_state.compliance_results_df = df_results
                    st.session_state.security_level_compliance = security_compliance
                    st.session_state.advanced_metrics = None
                    
                    # Generate breach alerts
                    breaches = []
                    if (df_results['Weight %'] > single_stock_limit).any():
                        breach_stocks = df_results[df_results['Weight %'] > single_stock_limit]
                        for _, stock in breach_stocks.iterrows():
                            breaches.append({
                                'type': 'Single Stock Limit',
                                'severity': '🔴 Critical',
                                'details': f"{stock['Symbol']} at {stock['Weight %']:.2f}% (Limit: {single_stock_limit}%)"
                            })
                    
                    sector_weights = df_results.groupby('Industry')['Weight %'].sum()
                    if (sector_weights > single_sector_limit).any():
                        breach_sectors = sector_weights[sector_weights > single_sector_limit]
                        for sector, weight in breach_sectors.items():
                            breaches.append({
                                'type': 'Sector Limit',
                                'severity': '🟠 High',
                                'details': f"{sector} at {weight:.2f}% (Limit: {single_sector_limit}%)"
                            })
                    
                    st.session_state.breach_alerts = breaches
                    
                    st.success("✅ Analysis Complete!")
                    if breaches:
                        st.warning(f"⚠️ {len(breaches)} compliance breach(es) detected!")
        
        except Exception as e:
            st.error(f"Failed to process CSV file. Error: {e}")
            st.exception(e)
    
    results_df = st.session_state.get("compliance_results_df", pd.DataFrame())
    
    if not results_df.empty and 'Weight %' in results_df.columns:
        st.markdown("---")
        
        # Display breach alerts
        if st.session_state.get("breach_alerts"):
            st.error("🚨 **Compliance Breach Alert**")
            breach_df = pd.DataFrame(st.session_state["breach_alerts"])
            st.dataframe(breach_df, use_container_width=True, hide_index=True)
        
        analysis_tabs = st.tabs([
            "📊 Executive Dashboard", 
            "🔍 Detailed Breakdowns", 
            "📈 Advanced Risk Analytics",
            "⚖️ Rule Validation",
            "🔐 Security-Level Compliance",
            "📊 Concentration Analysis",
            "📄 Full Report"
        ])
        
        # TAB 1: Executive Dashboard
        with analysis_tabs[0]:
            st.subheader("Portfolio Executive Dashboard")
            total_value = results_df['Real-time Value (Rs)'].sum()
            
            kpi_cols = st.columns(5)
            kpi_cols[0].metric("Portfolio Value", f"₹ {total_value:,.2f}")
            kpi_cols[1].metric("Holdings Count", f"{len(results_df)}")
            kpi_cols[2].metric("Unique Sectors", f"{results_df['Industry'].nunique()}")
            if 'Rating' in results_df.columns:
                kpi_cols[3].metric("Unique Ratings", f"{results_df['Rating'].nunique()}")
            kpi_cols[4].metric("Compliance Status", "✅ Pass" if not st.session_state.get("breach_alerts") else f"❌ {len(st.session_state['breach_alerts'])} Breaches")
            
            st.markdown("#### Key Concentration Metrics")
            conc_cols = st.columns(4)
            
            with conc_cols[0]:
                st.metric("Top Stock Weight", f"{results_df['Weight %'].max():.2f}%")
                st.metric("Top 5 Stocks", f"{results_df.nlargest(5, 'Weight %')['Weight %'].sum():.2f}%")
            
            with conc_cols[1]:
                st.metric("Top 10 Stocks", f"{results_df.nlargest(10, 'Weight %')['Weight %'].sum():.2f}%")
                st.metric("Top 3 Sectors", f"{results_df.groupby('Industry')['Weight %'].sum().nlargest(3).sum():.2f}%")
            
            with conc_cols[2]:
                stock_hhi = (results_df['Weight %'] ** 2).sum()
                def get_hhi_category(score):
                    return "🟢 Low" if score < 1500 else "🟡 Moderate" if score <= 2500 else "🔴 High"
                st.metric("Stock HHI", f"{stock_hhi:,.0f}", help=get_hhi_category(stock_hhi))
                sector_hhi = (results_df.groupby('Industry')['Weight %'].sum() ** 2).sum()
                st.metric("Sector HHI", f"{sector_hhi:,.0f}", help=get_hhi_category(sector_hhi))
            
            with conc_cols[3]:
                # Effective N
                effective_n_stocks = 1 / ((results_df['Weight %'] / 100) ** 2).sum()
                st.metric("Effective N (Stocks)", f"{effective_n_stocks:.1f}", help="Portfolio acts like this many equal-weighted stocks")
                sector_weights_pct = results_df.groupby('Industry')['Weight %'].sum() / 100
                effective_n_sectors = 1 / (sector_weights_pct ** 2).sum()
                st.metric("Effective N (Sectors)", f"{effective_n_sectors:.1f}")
            
            # Visual dashboard
            st.markdown("#### Portfolio Composition Overview")
            dash_cols = st.columns(2)
            
            with dash_cols[0]:
                # Top 15 holdings pie chart
                top_15 = results_df.nlargest(15, 'Weight %')
                others_weight = results_df.nsmallest(len(results_df) - 15, 'Weight %')['Weight %'].sum()
                
                plot_data = pd.concat([
                    top_15[['Name', 'Weight %']],
                    pd.DataFrame([{'Name': 'Others', 'Weight %': others_weight}])
                ])
                
                fig_pie = px.pie(plot_data, values='Weight %', names='Name', 
                                title='Portfolio Composition (Top 15 + Others)',
                                hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with dash_cols[1]:
                # Sector allocation
                sector_data = results_df.groupby('Industry')['Weight %'].sum().reset_index().sort_values('Weight %', ascending=False)
                fig_sector = px.bar(sector_data.head(10), x='Weight %', y='Industry', orientation='h',
                                   title='Top 10 Sector Allocations',
                                   color='Weight %', color_continuous_scale='Blues')
                fig_sector.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_sector, use_container_width=True)
        
        # TAB 2: Detailed Breakdowns
        with analysis_tabs[1]:
            st.subheader("Comprehensive Portfolio Breakdowns")
            
            breakdown_subtabs = st.tabs(["Holdings", "Sectors", "Ratings", "Market Cap", "Asset Class"])
            
            with breakdown_subtabs[0]:
                st.markdown("##### Top 20 Holdings Analysis")
                top_20 = results_df.nlargest(20, 'Weight %')[['Name', 'Symbol', 'Industry', 'Weight %', 'Real-time Value (Rs)', 'LTP']]
                
                fig_holdings = px.bar(top_20, x='Weight %', y='Name', orientation='h',
                                    title='Top 20 Holdings by Weight',
                                    color='Industry',
                                    hover_data=['Symbol', 'Real-time Value (Rs)'])
                fig_holdings.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
                st.plotly_chart(fig_holdings, use_container_width=True)
                
                st.dataframe(top_20.style.format({
                    'Weight %': '{:.2f}%',
                    'Real-time Value (Rs)': '₹{:,.2f}',
                    'LTP': '₹{:,.2f}'
                }), use_container_width=True)
            
            with breakdown_subtabs[1]:
                st.markdown("##### Sector-wise Analysis")
                sector_analysis = results_df.groupby('Industry').agg({
                    'Weight %': 'sum',
                    'Real-time Value (Rs)': 'sum',
                    'Symbol': 'count'
                }).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_sector_bar = px.bar(sector_analysis.reset_index().head(15), 
                                          x='Weight %', y='Industry', orientation='h',
                                          title='Top 15 Sectors by Weight')
                    fig_sector_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_sector_bar, use_container_width=True)
                
                with col2:
                    fig_sector_count = px.bar(sector_analysis.reset_index().head(15),
                                            x='Count', y='Industry', orientation='h',
                                            title='Top 15 Sectors by Holdings Count',
                                            color='Count', color_continuous_scale='Greens')
                    fig_sector_count.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_sector_count, use_container_width=True)
                
                st.dataframe(sector_analysis.style.format({
                    'Weight %': '{:.2f}%',
                    'Real-time Value (Rs)': '₹{:,.2f}'
                }), use_container_width=True)
            
            with breakdown_subtabs[2]:
                if 'Rating' in results_df.columns:
                    st.markdown("##### Credit Rating Distribution")
                    rating_analysis = results_df.groupby('Rating').agg({
                        'Weight %': 'sum',
                        'Symbol': 'count'
                    }).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_rating = px.pie(rating_analysis.reset_index(), values='Weight %', names='Rating',
                                          title='Rating Distribution by Weight', hole=0.3)
                        st.plotly_chart(fig_rating, use_container_width=True)
                    
                    with col2:
                        fig_rating_bar = px.bar(rating_analysis.reset_index(), x='Weight %', y='Rating',
                                              orientation='h', title='Rating Exposure by Weight',
                                              color='Weight %', color_continuous_scale='RdYlGn_r')
                        fig_rating_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_rating_bar, use_container_width=True)
                    
                    st.dataframe(rating_analysis.style.format({'Weight %': '{:.2f}%'}), use_container_width=True)
                else:
                    st.info("Rating information not available in portfolio data.")
            
            with breakdown_subtabs[3]:
                if 'Market Cap' in results_df.columns:
                    st.markdown("##### Market Capitalization Analysis")
                    mcap_analysis = results_df.groupby('Market Cap').agg({
                        'Weight %': 'sum',
                        'Symbol': 'count'
                    }).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_mcap = px.pie(mcap_analysis.reset_index(), values='Weight %', names='Market Cap',
                                        title='Market Cap Distribution', hole=0.3)
                        st.plotly_chart(fig_mcap, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Market Cap Statistics**")
                        st.dataframe(mcap_analysis.style.format({'Weight %': '{:.2f}%'}), use_container_width=True)
                else:
                    st.info("Market Cap information not available in portfolio data.")
            
            with breakdown_subtabs[4]:
                if 'Asset Class' in results_df.columns:
                    st.markdown("##### Asset Class Allocation")
                    asset_analysis = results_df.groupby('Asset Class').agg({
                        'Weight %': 'sum',
                        'Symbol': 'count'
                    }).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_asset = px.pie(asset_analysis.reset_index(), values='Weight %', names='Asset Class',
                                         title='Asset Class Distribution', hole=0.3)
                        st.plotly_chart(fig_asset, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Asset Class Statistics**")
                        st.dataframe(asset_analysis.style.format({'Weight %': '{:.2f}%'}), use_container_width=True)
                else:
                    st.info("Asset Class information not available in portfolio data.")
            
            # Interactive Treemap
            st.markdown("---")
            st.markdown("##### 🗺️ Interactive Portfolio Treemap")
            treemap_cols = st.columns([3, 1])
            
            with treemap_cols[1]:
                treemap_depth = st.radio("Hierarchy Depth", ["Industry → Stock", "Industry → Rating → Stock"], key="treemap_depth")
            
            with treemap_cols[0]:
                if treemap_depth == "Industry → Stock":
                    fig_treemap = px.treemap(results_df, path=[px.Constant("Portfolio"), 'Industry', 'Name'],
                                           values='Real-time Value (Rs)',
                                           hover_data={'Weight %': ':.2f'},
                                           title='Portfolio Treemap: Industry → Stock')
                else:
                    if 'Rating' in results_df.columns:
                        fig_treemap = px.treemap(results_df, path=[px.Constant("Portfolio"), 'Industry', 'Rating', 'Name'],
                                               values='Real-time Value (Rs)',
                                               hover_data={'Weight %': ':.2f'},
                                               title='Portfolio Treemap: Industry → Rating → Stock')
                    else:
                        st.warning("Rating column not found. Showing default hierarchy.")
                        fig_treemap = px.treemap(results_df, path=[px.Constant("Portfolio"), 'Industry', 'Name'],
                                               values='Real-time Value (Rs)',
                                               hover_data={'Weight %': ':.2f'},
                                               title='Portfolio Treemap: Industry → Stock')
                
                fig_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25), height=600)
                st.plotly_chart(fig_treemap, use_container_width=True)
        
        # TAB 3: Advanced Risk Analytics
        with analysis_tabs[2]:
            st.subheader("Advanced Risk & Return Analytics")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                if st.button("🔄 Calculate Advanced Metrics", key="calc_adv_metrics", use_container_width=True, type="primary"):
                    with st.spinner("Calculating advanced metrics... This may take 1-2 minutes."):
                        st.session_state.advanced_metrics = calculate_advanced_metrics(results_df, api_key, access_token)
            
            with col1:
                st.info("💡 Advanced metrics require historical data for all holdings. Click button to calculate.")
            
            if st.session_state.advanced_metrics:
                metrics = st.session_state.advanced_metrics
                
                st.markdown("#### Risk Metrics Dashboard")
                risk_cols = st.columns(4)
                
                risk_cols[0].metric("Daily VaR (95%)", 
                                   f"{metrics['var_95'] * 100:.2f}%",
                                   help="Maximum expected daily loss with 95% confidence")
                risk_cols[1].metric("Daily VaR (99%)", 
                                   f"{metrics['var_99'] * 100:.2f}%",
                                   help="Maximum expected daily loss with 99% confidence")
                risk_cols[2].metric("CVaR (95%)", 
                                   f"{metrics['cvar_95'] * 100:.2f}%",
                                   help="Expected loss beyond VaR threshold")
                risk_cols[3].metric("Portfolio Volatility", 
                                   f"{metrics['portfolio_volatility']:.2f}%",
                                   help="Annualized standard deviation")
                
                st.markdown("#### Performance Metrics")
                perf_cols = st.columns(4)
                
                if metrics['beta'] is not None:
                    perf_cols[0].metric(f"Beta (vs {BENCHMARK_SYMBOL})", 
                                       f"{metrics['beta']:.3f}",
                                       help="Systematic risk relative to market")
                else:
                    perf_cols[0].metric("Beta", "N/A", help="Benchmark data unavailable")
                
                if metrics['alpha'] is not None:
                    perf_cols[1].metric("Alpha (Annualized)", 
                                       f"{metrics['alpha'] * 100:.2f}%",
                                       help="Excess return vs expected return")
                else:
                    perf_cols[1].metric("Alpha", "N/A")
                
                if metrics['tracking_error'] is not None:
                    perf_cols[2].metric("Tracking Error", 
                                       f"{metrics['tracking_error']:.2f}%",
                                       help="Standard deviation of active returns")
                else:
                    perf_cols[2].metric("Tracking Error", "N/A")
                
                if metrics['information_ratio'] is not None:
                    perf_cols[3].metric("Information Ratio", 
                                       f"{metrics['information_ratio']:.3f}",
                                       help="Active return per unit of tracking error")
                else:
                    perf_cols[3].metric("Information Ratio", "N/A")
                
                st.markdown("#### Diversification Metrics")
                div_cols = st.columns(3)
                
                if metrics['sortino_ratio'] is not None:
                    div_cols[0].metric("Sortino Ratio", 
                                      f"{metrics['sortino_ratio']:.3f}",
                                      help="Risk-adjusted return using downside deviation")
                else:
                    div_cols[0].metric("Sortino Ratio", "N/A")
                
                div_cols[1].metric("Avg Correlation", 
                                  f"{metrics['avg_correlation']:.3f}",
                                  help="Average pairwise correlation between holdings")
                
                if metrics['diversification_ratio'] is not None:
                    div_cols[2].metric("Diversification Ratio", 
                                      f"{metrics['diversification_ratio']:.3f}",
                                      help="Weighted volatility / portfolio volatility (>1 is better)")
                else:
                    div_cols[2].metric("Diversification Ratio", "N/A")
                
                # Risk visualization
                st.markdown("---")
                st.markdown("#### Risk Analysis Interpretation")
                
                interp_cols = st.columns(2)
                
                with interp_cols[0]:
                    st.markdown("**🎯 Risk Assessment:**")
                    
                    if abs(metrics['var_95'] * 100) < 2:
                        st.success("✅ Low daily risk exposure")
                    elif abs(metrics['var_95'] * 100) < 4:
                        st.warning("⚠️ Moderate daily risk exposure")
                    else:
                        st.error("🔴 High daily risk exposure")
                    
                    if metrics['beta'] is not None:
                        if metrics['beta'] < 0.8:
                            st.info("📉 Portfolio is less volatile than market (defensive)")
                        elif metrics['beta'] < 1.2:
                            st.info("📊 Portfolio volatility aligned with market")
                        else:
                            st.warning("📈 Portfolio is more volatile than market (aggressive)")
                
                with interp_cols[1]:
                    st.markdown("**📊 Performance Assessment:**")
                    
                    if metrics['alpha'] is not None:
                        if metrics['alpha'] > 0.02:
                            st.success("✅ Strong positive alpha - outperforming expectations")
                        elif metrics['alpha'] > -0.02:
                            st.info("➡️ Neutral alpha - performing as expected")
                        else:
                            st.error("🔴 Negative alpha - underperforming expectations")
                    
                    if metrics['information_ratio'] is not None:
                        if metrics['information_ratio'] > 0.5:
                            st.success("✅ Good active management efficiency")
                        elif metrics['information_ratio'] > 0:
                            st.info("➡️ Moderate active management efficiency")
                        else:
                            st.warning("⚠️ Poor active management efficiency")
            
            else:
                st.info("Click 'Calculate Advanced Metrics' to view risk analytics based on 1-year historical data.")
        
        # TAB 4: Rule Validation
        with analysis_tabs[3]:
            st.subheader("⚖️ Compliance Rule Validation")
            
            st.markdown("**Custom Rules Validation Results:**")
            
            validation_results = parse_and_validate_rules_enhanced(rules_text, results_df)
            
            if not validation_results:
                st.info("Define rules in the sidebar to see validation results.")
            else:
                # Summary metrics
                total_rules = len(validation_results)
                passed = sum(1 for r in validation_results if r['status'] == "✅ PASS")
                failed = sum(1 for r in validation_results if r['status'] == "❌ FAIL")
                errors = sum(1 for r in validation_results if 'Error' in r['status'] or 'Invalid' in r['status'])
                
                summary_cols = st.columns(4)
                summary_cols[0].metric("Total Rules", total_rules)
                summary_cols[1].metric("✅ Passed", passed)
                summary_cols[2].metric("❌ Failed", failed)
                summary_cols[3].metric("⚠️ Errors", errors)
                
                # Filter options
                filter_cols = st.columns([2, 1])
                with filter_cols[0]:
                    status_filter = st.multiselect(
                        "Filter by Status",
                        ["✅ PASS", "❌ FAIL", "⚠️ Invalid", "Error"],
                        default=["✅ PASS", "❌ FAIL", "⚠️ Invalid", "Error"]
                    )
                
                with filter_cols[1]:
                    severity_filter = st.multiselect(
                        "Filter by Severity",
                        ["🔴 Critical", "🟠 High", "🟡 Medium", "✅ Compliant"],
                        default=["🔴 Critical", "🟠 High", "🟡 Medium", "✅ Compliant"]
                    )
                
                # Display results
                st.markdown("---")
                
                for res in validation_results:
                    if res['status'] not in status_filter:
                        continue
                    if 'severity' in res and res['severity'] not in severity_filter and res['severity'] != 'N/A':
                        continue
                    
                    severity_label = res.get('severity', 'N/A')
                    
                    if res['status'] == "✅ PASS":
                        with st.expander(f"{res['status']} {severity_label} | `{res['rule']}`", expanded=False):
                            st.success(f"**Status:** {res['status']}")
                            st.write(f"**Details:** {res['details']}")
                    
                    elif res['status'] == "❌ FAIL":
                        with st.expander(f"{res['status']} {severity_label} | `{res['rule']}`", expanded=True):
                            st.error(f"**Status:** {res['status']} - {severity_label}")
                            st.write(f"**Details:** {res['details']}")
                            
                            if 'actual_value' in res and 'threshold' in res:
                                st.metric("Actual Value", f"{res['actual_value']:.2f}%")
                                st.metric("Threshold", f"{res['threshold']:.2f}%")
                                st.metric("Breach Amount", f"{res['breach_amount']:.2f}%", 
                                         delta=f"{res['breach_amount']:.2f}%", delta_color="inverse")
                    
                    else:
                        with st.expander(f"{res['status']} | `{res['rule']}`", expanded=False):
                            st.warning(f"**Status:** {res['status']}")
                            st.write(f"**Details:** {res['details']}")
                
                # Export validation report
                st.markdown("---")
                if st.button("📥 Export Validation Report", use_container_width=True):
                    validation_df = pd.DataFrame(validation_results)
                    csv = validation_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Validation Report (CSV)",
                        csv,
                        f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
        
        # TAB 5: Security-Level Compliance
        with analysis_tabs[4]:
            st.subheader("🔐 Security-Level Compliance Analysis")
            
            security_df = st.session_state.get("security_level_compliance", pd.DataFrame())
            
            if security_df.empty:
                st.info("Security-level compliance data not available. Please analyze portfolio first.")
            else:
                st.markdown("**Individual Security Compliance Status:**")
                
                # Summary statistics
                breach_count = (security_df['Stock Limit Breach'] == '❌ Breach').sum()
                compliant_count = (security_df['Stock Limit Breach'] == '✅ Compliant').sum()
                
                summary_cols = st.columns(4)
                summary_cols[0].metric("Total Securities", len(security_df))
                summary_cols[1].metric("✅ Compliant", compliant_count)
                summary_cols[2].metric("❌ Breaches", breach_count)
                summary_cols[3].metric("Breach Rate", f"{(breach_count/len(security_df)*100):.1f}%")
                
                # Filters
                st.markdown("---")
                filter_cols = st.columns(3)
                
                with filter_cols[0]:
                    compliance_filter = st.multiselect(
                        "Stock Limit Status",
                        security_df['Stock Limit Breach'].unique(),
                        default=security_df['Stock Limit Breach'].unique()
                    )
                
                with filter_cols[1]:
                    if 'Concentration Risk' in security_df.columns:
                        risk_filter = st.multiselect(
                            "Concentration Risk",
                            security_df['Concentration Risk'].unique(),
                            default=security_df['Concentration Risk'].unique()
                        )
                
                with filter_cols[2]:
                    if 'Rating Compliance' in security_df.columns:
                        rating_filter = st.multiselect(
                            "Rating Compliance",
                            security_df['Rating Compliance'].unique(),
                            default=security_df['Rating Compliance'].unique()
                        )
                
                # Apply filters
                filtered_df = security_df[security_df['Stock Limit Breach'].isin(compliance_filter)]
                
                if 'Concentration Risk' in security_df.columns and 'risk_filter' in locals():
                    filtered_df = filtered_df[filtered_df['Concentration Risk'].isin(risk_filter)]
                
                if 'Rating Compliance' in security_df.columns and 'rating_filter' in locals():
                    filtered_df = filtered_df[filtered_df['Rating Compliance'].isin(rating_filter)]
                
                # Display table
                st.markdown(f"**Showing {len(filtered_df)} of {len(security_df)} securities**")
                
                display_columns = ['Name', 'Symbol', 'Industry', 'Weight %', 'Stock Limit Breach', 
                                  'Stock Limit Gap (%)', 'Concentration Risk']
                
                if 'Liquidity Status' in filtered_df.columns:
                    display_columns.append('Liquidity Status')
                if 'Rating Compliance' in filtered_df.columns:
                    display_columns.append('Rating Compliance')
                if 'Rating' in filtered_df.columns:
                    display_columns.append('Rating')
                
                available_columns = [col for col in display_columns if col in filtered_df.columns]
                
                # Color code the dataframe
                def highlight_compliance(row):
                    if row['Stock Limit Breach'] == '❌ Breach':
                        return ['background-color: #ffcccc'] * len(row)
                    return [''] * len(row)
                
                styled_df = filtered_df[available_columns].style.apply(highlight_compliance, axis=1).format({
                    'Weight %': '{:.2f}%',
                    'Stock Limit Gap (%)': '{:.2f}%'
                })
                
                st.dataframe(styled_df, use_container_width=True, height=500)
                
                # Breach details
                if breach_count > 0:
                    st.markdown("---")
                    st.markdown("#### 🚨 Detailed Breach Analysis")
                    
                    breach_df = security_df[security_df['Stock Limit Breach'] == '❌ Breach'].sort_values('Weight %', ascending=False)
                    
                    for idx, row in breach_df.iterrows():
                        with st.expander(f"🔴 {row['Symbol']} - {row['Name']} ({row['Weight %']:.2f}%)", expanded=False):
                            cols = st.columns(3)
                            cols[0].metric("Current Weight", f"{row['Weight %']:.2f}%")
                            cols[1].metric("Limit", f"{single_stock_limit:.2f}%")
                            cols[2].metric("Excess", f"{row['Weight %'] - single_stock_limit:.2f}%",
                                         delta=f"{row['Weight %'] - single_stock_limit:.2f}%", delta_color="inverse")
                            
                            st.write(f"**Industry:** {row['Industry']}")
                            st.write(f"**Value:** ₹{row['Real-time Value (Rs)']:,.2f}")
                            
                            if 'Rating' in row:
                                st.write(f"**Rating:** {row['Rating']}")
                
                # Export security compliance
                st.markdown("---")
                csv = security_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Export Security-Level Compliance Report",
                    csv,
                    f"security_compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        # TAB 6: Concentration Analysis
        with analysis_tabs[5]:
            st.subheader("📊 Deep Dive Concentration Analysis")
            
            st.markdown("#### Cumulative Weight Distribution")
            
            # Calculate cumulative weights
            sorted_df = results_df.sort_values('Weight %', ascending=False).reset_index(drop=True)
            sorted_df['Cumulative Weight %'] = sorted_df['Weight %'].cumsum()
            sorted_df['Rank'] = range(1, len(sorted_df) + 1)
            
            # Lorenz curve
            fig_lorenz = go.Figure()
            
            fig_lorenz.add_trace(go.Scatter(
                x=sorted_df['Rank'],
                y=sorted_df['Cumulative Weight %'],
                mode='lines+markers',
                name='Actual Portfolio',
                line=dict(color='blue', width=2)
            ))
            
            # Perfect equality line
            fig_lorenz.add_trace(go.Scatter(
                x=[0, len(sorted_df)],
                y=[0, 100],
                mode='lines',
                name='Perfect Equality',
                line=dict(color='red', dash='dash')
            ))
            
            fig_lorenz.update_layout(
                title='Portfolio Concentration Curve (Lorenz)',
                xaxis_title='Number of Holdings (Ranked by Weight)',
                yaxis_title='Cumulative Weight %',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_lorenz, use_container_width=True)
            
            # Concentration benchmarks
            st.markdown("#### Concentration Benchmarks")
            
            bench_cols = st.columns(5)
            
            top_1_weight = sorted_df.iloc[0]['Weight %']
            top_3_weight = sorted_df.head(3)['Weight %'].sum()
            top_5_weight = sorted_df.head(5)['Weight %'].sum()
            top_10_weight = sorted_df.head(10)['Weight %'].sum()
            top_20_weight = sorted_df.head(20)['Weight %'].sum() if len(sorted_df) >= 20 else sorted_df['Weight %'].sum()
            
            bench_cols[0].metric("Top 1", f"{top_1_weight:.2f}%")
            bench_cols[1].metric("Top 3", f"{top_3_weight:.2f}%")
            bench_cols[2].metric("Top 5", f"{top_5_weight:.2f}%")
            bench_cols[3].metric("Top 10", f"{top_10_weight:.2f}%")
            bench_cols[4].metric("Top 20", f"{top_20_weight:.2f}%")
            
            # Sector concentration
            st.markdown("---")
            st.markdown("#### Sector Concentration Matrix")
            
            sector_concentration = results_df.groupby('Industry').agg({
                'Weight %': ['sum', 'count', 'max'],
                'Symbol': lambda x: ', '.join(x.head(3))
            }).round(2)
            
            sector_concentration.columns = ['Total Weight %', 'Count', 'Max Single Stock %', 'Top 3 Stocks']
            sector_concentration = sector_concentration.sort_values('Total Weight %', ascending=False)
            sector_concentration['Concentration Level'] = sector_concentration['Total Weight %'].apply(
                lambda x: '🔴 High' if x > 25 else '🟡 Medium' if x > 15 else '🟢 Low'
            )
            
            st.dataframe(sector_concentration.style.format({
                'Total Weight %': '{:.2f}%',
                'Max Single Stock %': '{:.2f}%'
            }), use_container_width=True)
            
            # Heat map
            st.markdown("---")
            st.markdown("#### Sector vs Market Cap Heatmap")
            
            if 'Market Cap' in results_df.columns:
                pivot_data = results_df.pivot_table(
                    values='Weight %',
                    index='Industry',
                    columns='Market Cap',
                    aggfunc='sum',
                    fill_value=0
                )
                
                fig_heatmap = px.imshow(
                    pivot_data,
                    labels=dict(x="Market Cap", y="Industry", color="Weight %"),
                    title="Portfolio Allocation: Sector vs Market Cap",
                    color_continuous_scale='RdYlGn_r',
                    aspect="auto"
                )
                
                fig_heatmap.update_layout(height=max(400, len(pivot_data) * 30))
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Market Cap data not available for heatmap visualization.")
        
        # TAB 7: Full Report
        with analysis_tabs[6]:
            st.subheader("📄 Comprehensive Portfolio Report")
            
            # Report generation options
            report_cols = st.columns([2, 1])
            
            with report_cols[0]:
                st.markdown("**Generate a comprehensive report with all analysis**")
                include_sections = st.multiselect(
                    "Select Sections to Include",
                    ["Executive Summary", "Holdings Detail", "Sector Analysis", "Risk Metrics", 
                     "Compliance Validation", "Security-Level Compliance"],
                    default=["Executive Summary", "Holdings Detail", "Sector Analysis", "Compliance Validation"]
                )
            
            with report_cols[1]:
                report_format = st.radio("Format", ["Excel", "CSV"], key="report_format")
                
                if st.button("📊 Generate Full Report", type="primary", use_container_width=True):
                    with st.spinner("Generating comprehensive report..."):
                        # Prepare data
                        report_data = {}
                        
                        if "Executive Summary" in include_sections:
                            summary_data = {
                                'Metric': ['Total Value', 'Holdings Count', 'Unique Sectors', 
                                          'Top Stock Weight', 'Top 10 Weight', 'Stock HHI', 'Sector HHI'],
                                'Value': [
                                    f"₹{total_value:,.2f}",
                                    len(results_df),
                                    results_df['Industry'].nunique(),
                                    f"{results_df['Weight %'].max():.2f}%",
                                    f"{results_df.nlargest(10, 'Weight %')['Weight %'].sum():.2f}%",
                                    f"{(results_df['Weight %'] ** 2).sum():.0f}",
                                    f"{(results_df.groupby('Industry')['Weight %'].sum() ** 2).sum():.0f}"
                                ]
                            }
                            report_data['Executive Summary'] = pd.DataFrame(summary_data)
                        
                        if "Holdings Detail" in include_sections:
                            report_data['Holdings Detail'] = results_df[['Name', 'Symbol', 'Industry', 'Quantity', 
                                                                         'LTP', 'Real-time Value (Rs)', 'Weight %']]
                        
                        if "Sector Analysis" in include_sections:
                            sector_analysis = results_df.groupby('Industry').agg({
                                'Weight %': 'sum',
                                'Real-time Value (Rs)': 'sum',
                                'Symbol': 'count'
                            }).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                            report_data['Sector Analysis'] = sector_analysis
                        
                        if "Risk Metrics" in include_sections and st.session_state.advanced_metrics:
                            metrics = st.session_state.advanced_metrics
                            risk_data = pd.DataFrame([{
                                'Metric': k.replace('_', ' ').title(),
                                'Value': f"{v:.4f}" if v is not None else "N/A"
                            } for k, v in metrics.items()])
                            report_data['Risk Metrics'] = risk_data
                        
                        if "Compliance Validation" in include_sections:
                            validation_results = parse_and_validate_rules_enhanced(rules_text, results_df)
                            if validation_results:
                                report_data['Compliance Validation'] = pd.DataFrame(validation_results)
                        
                        if "Security-Level Compliance" in include_sections and not st.session_state.get("security_level_compliance", pd.DataFrame()).empty:
                            report_data['Security Compliance'] = st.session_state["security_level_compliance"]
                        
                        # Create download
                        if report_format == "Excel":
                            from io import BytesIO
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                for sheet_name, df in report_data.items():
                                    df.to_excel(writer, sheet_name=sheet_name[:31], index=True)
                            
                            output.seek(0)
                            st.download_button(
                                "📥 Download Excel Report",
                                output,
                                f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        else:
                            # Combine all dataframes for CSV
                            combined_df = pd.concat([df.assign(Section=name) for name, df in report_data.items()], 
                                                    ignore_index=True)
                            csv = combined_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download CSV Report",
                                csv,
                                f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        
                        st.success("✅ Report generated successfully!")
            
            # Display current holdings table
            st.markdown("---")
            st.markdown("**Current Holdings Summary**")
            
            display_df = results_df.copy()
            format_dict = {
                'Real-time Value (Rs)': '₹{:,.2f}',
                'LTP': '₹{:,.2f}',
                'Weight %': '{:.2f}%'
            }
            
            column_order = ['Name', 'Symbol', 'Industry', 'Real-time Value (Rs)', 'Weight %', 'Quantity', 'LTP']
            
            if 'Asset Class' in display_df.columns:
                column_order.insert(3, 'Asset Class')
            if 'Rating' in display_df.columns:
                column_order.insert(3, 'Rating')
            if 'Market Cap' in display_df.columns:
                column_order.insert(3, 'Market Cap')
            
            display_columns = [col for col in column_order if col in display_df.columns]
            
            st.dataframe(
                display_df[display_columns].style.format(format_dict),
                use_container_width=True,
                height=500
            )


# --- AI ANALYSIS TAB FUNCTIONS ---
def extract_text_from_files(uploaded_files):
    full_text = ""
    for file in uploaded_files:
        full_text += f"\n\n--- DOCUMENT: {file.name} ---\n\n"
        if file.type == "application/pdf":
            with fitz.open(stream=file.getvalue(), filetype="pdf") as doc:
                for page in doc:
                    full_text += page.get_text()
        else:
            full_text += file.getvalue().decode("utf-8")
    return full_text


def get_portfolio_summary(df):
    if df.empty:
        return "No portfolio data available."
    
    total_value = df['Real-time Value (Rs)'].sum()
    top_10_stocks = df.nlargest(10, 'Weight %')[['Name', 'Weight %']]
    sector_weights = df.groupby('Industry')['Weight %'].sum().nlargest(10)
    
    summary = f"""**Portfolio Snapshot (as of {datetime.now().strftime('%Y-%m-%d')})**
    
- **Total Value:** ₹ {total_value:,.2f}
- **Number of Holdings:** {len(df)}
- **Top Stock Weight:** {df['Weight %'].max():.2f}%
- **Top 10 Combined Weight:** {df.nlargest(10, 'Weight %')['Weight %'].sum():.2f}%

**Top 10 Holdings:**
"""
    for _, row in top_10_stocks.iterrows():
        summary += f"- {row['Name']}: {row['Weight %']:.2f}%\n"
    
    summary += "\n**Top 10 Sector Exposures:**\n"
    for sector, weight in sector_weights.items():
        summary += f"- {sector}: {weight:.2f}%\n"
    
    return summary


def render_ai_analysis_tab(kite_client):
    st.header("🤖 AI-Powered Compliance Analysis (with Google Gemini)")
    st.markdown("Analyze your portfolio against scheme documents (SID/KIM) and general regulatory guidelines using advanced AI.")
    
    portfolio_df = st.session_state.get("compliance_results_df")
    
    if portfolio_df is None or portfolio_df.empty:
        st.warning("⚠️ Please upload and analyze a portfolio in the 'Investment Compliance' tab first.")
        return
    
    st.info("💡 This tool uses AI for analysis. The output is for informational purposes and is not financial or legal advice. Always verify all findings independently.", icon="ℹ️")
    
    # Document upload
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "📄 Upload Scheme Documents (SID, KIM, Investment Policy, etc.)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload regulatory documents for AI-powered compliance analysis"
        )
    
    with col2:
        st.markdown("**Analysis Options**")
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Standard", "Comprehensive"],
            value="Standard"
        )
        
        include_recommendations = st.checkbox("Include Recommendations", value=True)
        include_risk_assessment = st.checkbox("Include Risk Assessment", value=True)
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} document(s) uploaded successfully")
        
        with st.expander("📋 Uploaded Documents"):
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size / 1024:.2f} KB)")
        
        if st.button("🚀 Run AI Compliance Analysis", type="primary", use_container_width=True):
            with st.spinner("🔄 Reading documents and preparing for AI analysis..."):
                try:
                    # Extract document text
                    docs_text = extract_text_from_files(uploaded_files)
                    portfolio_summary = get_portfolio_summary(portfolio_df)
                    
                    # Calculate additional context
                    breach_alerts = st.session_state.get("breach_alerts", [])
                    breach_summary = "\n".join([f"- {b['type']}: {b['details']}" for b in breach_alerts]) if breach_alerts else "No immediate breaches detected."
                    
                    # Build comprehensive prompt
                    if analysis_depth == "Quick":
                        depth_instruction = "Provide a concise analysis focusing on critical compliance issues only."
                    elif analysis_depth == "Standard":
                        depth_instruction = "Provide a balanced analysis covering key compliance areas and major risks."
                    else:
                        depth_instruction = "Provide an exhaustive, detailed analysis covering all aspects of compliance, risk, and regulatory requirements."
                    
                    prompt = f"""You are an expert investment compliance analyst for an Indian Asset Management Company with deep knowledge of SEBI regulations, mutual fund guidelines, and portfolio management best practices.

**YOUR TASK:**
Perform a comprehensive compliance analysis of the given investment portfolio against the provided scheme documents and SEBI/AMFI regulations.

{depth_instruction}

**PORTFOLIO DATA:**
```
{portfolio_summary}
```

**DETECTED ISSUES (Automated Checks):**
```
{breach_summary}
```

**SCHEME DOCUMENT(S) TEXT:**
```
{docs_text[:120000]}
```
(Note: Document text may be truncated due to length constraints)

**ANALYSIS FRAMEWORK:**

Please structure your response in clear markdown format with the following sections:

## 1. Executive Summary
- Overall compliance status (Compliant / Partially Compliant / Non-Compliant)
- Critical findings count
- Key action items (if any)

## 2. Investment Objective & Strategy Alignment
- Compare portfolio composition with stated investment philosophy
- Analyze if top holdings align with fund's stated objectives
- Identify any style drift or strategy deviation
- Specific holdings or sectors that enhance or detract from stated goals

## 3. Regulatory Compliance Assessment

### 3.1 SEBI Mutual Fund Regulations
Check against standard SEBI norms:
- **Single Issuer Limit:** Maximum 10% in any single stock (relaxed to 12% under certain conditions)
- **Sectoral Concentration:** Maximum 25% in any single sector (excluding financial services)
- **Group Company Exposure:** Maximum 25% in any single group
- **Cash & Cash Equivalents:** Minimum liquid assets requirement
- **Derivatives Usage:** Within prescribed limits (if applicable)

### 3.2 Scheme-Specific Restrictions
Based on the uploaded documents, verify:
- Minimum/maximum equity exposure limits
- Debt quality requirements
- Market cap allocation mandates (large/mid/small cap)
- Foreign securities limits
- Unrated securities exposure
- Any other fund-specific constraints

## 4. Portfolio Quality & Risk Assessment
{("Include this section:" if include_risk_assessment else "Skip this section.")}

### 4.1 Concentration Risk
- Top 10 holdings analysis
- Sector concentration assessment
- Single stock risks
- HHI (Herfindahl-Hirschman Index) interpretation

### 4.2 Credit Risk (if applicable)
- Rating distribution analysis
- Unrated exposure evaluation
- Credit quality assessment

### 4.3 Liquidity Risk
- Assessment of liquid vs illiquid holdings
- Average daily volume considerations
- Market impact risk

### 4.4 Market Risk
- Volatility considerations
- Beta analysis (if mentioned in documents)
- Market cap exposure risks

## 5. Specific Violations & Concerns
List any identified violations with:
- **Severity Level** (Critical / High / Medium / Low)
- **Description** of the violation
- **Regulatory Reference** (quote relevant clause)
- **Current Status** vs **Required Status**
- **Potential Impact**

## 6. Best Practices & Industry Benchmarks
- How does this portfolio compare to industry standards?
- Are there any emerging concerns?
- Compliance with ESG considerations (if mentioned)

{("## 7. Recommendations & Action Items" if include_recommendations else "")}
{("Provide specific, actionable recommendations:" if include_recommendations else "")}
{("- Immediate actions to address critical violations" if include_recommendations else "")}
{("- Portfolio rebalancing suggestions" if include_recommendations else "")}
{("- Risk mitigation strategies" if include_recommendations else "")}
{("- Process improvements for ongoing compliance" if include_recommendations else "")}

## 8. Disclaimers & Limitations
- Note any missing information that limits the analysis
- Areas requiring further investigation
- Assumptions made during analysis

**IMPORTANT GUIDELINES:**
1. Be specific - cite actual holdings, percentages, and document clauses
2. Use bullet points and tables where appropriate
3. Highlight critical issues with ⚠️ or 🔴 emoji
4. Be objective and professional
5. If documents don't specify certain limits, state "Not specified in documents" and use SEBI default norms
6. For any ambiguous situations, provide multiple interpretations

Begin your analysis now:
"""

                    # Call Gemini API
                    with st.spinner("🤖 AI is analyzing your portfolio... This may take 30-60 seconds."):
                        model = genai.GenerativeModel('gemini-2.0-flash-exp')
                        
                        safety_settings = [
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        ]
                        
                        response = model.generate_content(
                            prompt,
                            safety_settings=safety_settings,
                            generation_config={
                                'temperature': 0.3,
                                'top_p': 0.8,
                                'top_k': 40,
                                'max_output_tokens': 8192,
                            }
                        )
                        
                        st.session_state.ai_analysis_response = response.text
                        st.success("✅ AI Analysis Complete!")
                
                except Exception as e:
                    st.error(f"❌ An error occurred during AI analysis: {e}")
                    st.exception(e)
                    st.session_state.ai_analysis_response = None
    
    # Display AI analysis results
    if st.session_state.get("ai_analysis_response"):
        st.markdown("---")
        st.markdown("## 📊 AI Compliance Analysis Report")
        st.markdown("---")
        
        # Display the analysis
        st.markdown(st.session_state.ai_analysis_response)
        
        # Export options
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download as text
            txt_data = st.session_state.ai_analysis_response.encode('utf-8')
            st.download_button(
                "📄 Download as Text",
                txt_data,
                f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain",
                use_container_width=True
            )
        
        with col2:
            # Download as markdown
            md_data = st.session_state.ai_analysis_response.encode('utf-8')
            st.download_button(
                "📝 Download as Markdown",
                md_data,
                f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                "text/markdown",
                use_container_width=True
            )
        
        with col3:
            # Clear analysis
            if st.button("🗑️ Clear Analysis", use_container_width=True):
                st.session_state.ai_analysis_response = None
                st.rerun()
        
        # Feedback section
        st.markdown("---")
        st.markdown("### 📝 Analysis Feedback")
        
        feedback_cols = st.columns([3, 1])
        
        with feedback_cols[0]:
            feedback = st.text_area(
                "How useful was this analysis? Any suggestions for improvement?",
                height=100,
                placeholder="Your feedback helps us improve the AI analysis..."
            )
        
        with feedback_cols[1]:
            st.markdown("<br>", unsafe_allow_html=True)
            usefulness = st.radio(
                "Usefulness",
                ["⭐ Poor", "⭐⭐ Fair", "⭐⭐⭐ Good", "⭐⭐⭐⭐ Very Good", "⭐⭐⭐⭐⭐ Excellent"],
                index=2
            )
        
        if st.button("Submit Feedback"):
            st.success("✅ Thank you for your feedback!")
    
    else:
        # Show example/guide when no analysis is available
        st.markdown("---")
        st.markdown("### 📚 How to Use AI Compliance Analysis")
        
        with st.expander("📖 Step-by-Step Guide", expanded=True):
            st.markdown("""
            **Step 1: Prepare Your Documents**
            - Gather scheme information documents (SID)
            - Collect Key Information Memorandums (KIM)
            - Include investment policy statements
            - Add any fund-specific compliance documents
            
            **Step 2: Upload Documents**
            - Use the file uploader above
            - Supports PDF and TXT formats
            - Multiple files can be uploaded simultaneously
            
            **Step 3: Configure Analysis**
            - Choose analysis depth based on your needs
            - Enable recommendations for actionable insights
            - Enable risk assessment for comprehensive coverage
            
            **Step 4: Run Analysis**
            - Click "Run AI Compliance Analysis"
            - Wait 30-60 seconds for processing
            - Review comprehensive report
            
            **Step 5: Act on Findings**
            - Address critical violations immediately
            - Plan for medium/low priority items
            - Export report for documentation
            """)
        
        with st.expander("✨ Features of AI Analysis"):
            st.markdown("""
            - **Regulatory Compliance:** Checks against SEBI and AMFI guidelines
            - **Scheme-Specific Rules:** Validates fund-specific restrictions
            - **Risk Assessment:** Comprehensive risk analysis across multiple dimensions
            - **Actionable Insights:** Specific recommendations for compliance
            - **Document Citations:** References actual clauses from uploaded documents
            - **Severity Classification:** Prioritizes issues by criticality
            """)
        
        with st.expander("⚠️ Important Notes"):
            st.markdown("""
            - AI analysis is for informational purposes only
            - Always verify findings with compliance experts
            - Not a substitute for professional legal advice
            - Document quality affects analysis accuracy
            - Regular human oversight is essential
            - Keep documents updated for best results
            """)


# --- Main Application Logic (Tab Rendering) ---
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

with tab_market:
    render_market_historical_tab(k, api_key, access_token)

with tab_compliance:
    render_investment_compliance_tab(k, api_key, access_token)

with tab_ai:
    render_ai_analysis_tab(k)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Invsion Connect</strong> - Professional Portfolio Compliance & Analysis Platform</p>
    <p style='font-size: 0.9em;'>⚠️ This tool is for informational purposes only. Always consult with qualified professionals for investment decisions.</p>
    <p style='font-size: 0.8em;'>Powered by KiteConnect API & Google Gemini AI | © 2025</p>
</div>
""", unsafe_allow_html=True)
