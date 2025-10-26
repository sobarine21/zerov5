import streamlit as st
import pandas as pd
import json
import threading 
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import ta  # Technical Analysis library
import base64 # For encoding HTML for download

# Supabase imports
from supabase import create_client, Client
from kiteconnect import KiteConnect

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Kite Connect - Advanced Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("Invsion Connect")
st.markdown("A comprehensive platform for fetching market data, performing ML-driven analysis, risk assessment, and live data streaming.")

# --- Global Constants & Session State Initialization ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"
BENCHMARK_SYMBOL = "NIFTY 50" # Primary benchmark for advanced ratios

# Initialize session state variables
if "kite_access_token" not in st.session_state: st.session_state["kite_access_token"] = None
if "kite_login_response" not in st.session_state: st.session_state["kite_login_response"] = None
if "instruments_df" not in st.session_state: st.session_state["instruments_df"] = pd.DataFrame()
if "historical_data" not in st.session_state: st.session_state["historical_data"] = pd.DataFrame()
if "last_fetched_symbol" not in st.session_state: st.session_state["last_fetched_symbol"] = None
if "user_session" not in st.session_state: st.session_state["user_session"] = None
if "user_id" not in st.session_state: st.session_state["user_id"] = None
if "saved_indexes" not in st.session_state: st.session_state["saved_indexes"] = []
if "current_calculated_index_data" not in st.session_state: st.session_state["current_calculated_index_data"] = pd.DataFrame()
if "current_calculated_index_history" not in st.session_state: st.session_state["current_calculated_index_history"] = pd.DataFrame()
if "last_comparison_df" not in st.session_state: st.session_state["last_comparison_df"] = pd.DataFrame()
if "last_comparison_metrics" not in st.session_state: st.session_state["last_comparison_metrics"] = {}
if "last_facts_data" not in st.session_state: st.session_state["last_facts_data"] = None
if "last_factsheet_html_data" not in st.session_state: st.session_state["last_factsheet_html_data"] = None
if "current_market_data" not in st.session_state: st.session_state["current_market_data"] = None
if "holdings_data" not in st.session_state: st.session_state["holdings_data"] = None
if "benchmark_historical_data" not in st.session_state: st.session_state["benchmark_historical_data"] = pd.DataFrame() # Store benchmark data for ratios
if "factsheet_selected_constituents_index_names" not in st.session_state: st.session_state["factsheet_selected_constituents_index_names"] = []


# --- Load Credentials from Streamlit Secrets ---
def load_secrets():
    secrets = st.secrets
    kite_conf = secrets.get("kite", {})
    supabase_conf = secrets.get("supabase", {})

    errors = []
    if not kite_conf.get("api_key") or not kite_conf.get("api_secret") or not kite_conf.get("redirect_uri"):
        errors.append("Kite credentials (api_key, api_secret, redirect_uri)")
    if not supabase_conf.get("url") or not supabase_conf.get("anon_key"):
        errors.append("Supabase credentials (url, anon_key)")

    if errors:
        st.error(f"Missing required credentials in `.streamlit/secrets.toml`: {', '.join(errors)}.")
        st.info("Example `secrets.toml`:\n```toml\n[kite]\napi_key=\"YOUR_KITE_API_KEY\"\napi_secret=\"YOUR_KITE_SECRET\"\nredirect_uri=\"http://localhost:8501\"\n\n[supabase]\nurl=\"YOUR_SUPABASE_URL\"\nanon_key=\"YOUR_SUPABASE_ANON_KEY\"\n```")
        st.stop()
    return kite_conf, supabase_conf

KITE_CREDENTIALS, SUPABASE_CREDENTIALS = load_secrets()

# --- Supabase Client Initialization ---
@st.cache_resource(ttl=3600)
def init_supabase_client(url: str, key: str) -> Client:
    return create_client(url, key)

supabase: Client = init_supabase_client(SUPABASE_CREDENTIALS["url"], SUPABASE_CREDENTIALS["anon_key"])

# --- KiteConnect Client Initialization (Unauthenticated for login URL) ---
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

def _refresh_supabase_session():
    try:
        session_data = supabase.auth.get_session()
        if session_data and session_data.user:
            st.session_state["user_session"] = session_data
            st.session_state["user_id"] = session_data.user.id
        else:
            st.session_state["user_session"] = None
            st.session_state["user_id"] = None
    except Exception:
        st.session_state["user_session"] = None
        st.session_state["user_id"] = None

@st.cache_data(ttl=86400, show_spinner="Loading instruments...")
def load_instruments_cached(api_key: str, access_token: str, exchange: str = None) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame({"_error": ["Kite not authenticated to load instruments."]})
    try:
        instruments = kite_instance.instruments(exchange) if exchange else kite_instance.instruments()
        df = pd.DataFrame(instruments)
        if "instrument_token" in df.columns:
            df["instrument_token"] = df["instrument_token"].astype("int64")
        if 'tradingsymbol' in df.columns and 'name' in df.columns:
            df = df[['instrument_token', 'tradingsymbol', 'name', 'exchange']]
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [f"Failed to load instruments for {exchange or 'all exchanges'}: {e}"]})


@st.cache_data(ttl=60)
def get_ltp_price_cached(api_key: str, access_token: str, symbol: str, exchange: str = DEFAULT_EXCHANGE):
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return {"_error": "Kite not authenticated to fetch LTP."}
    
    exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
    try:
        ltp_data = kite_instance.ltp([exchange_symbol])
        return ltp_data.get(exchange_symbol)
    except Exception as e:
        return {"_error": str(e)}

@st.cache_data(ttl=3600)
def get_historical_data_cached(api_key: str, access_token: str, symbol: str, from_date: datetime.date, to_date: datetime.date, interval: str, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    """Fetches historical data for a symbol, robustly checking NSE for common indices."""
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame({"_error": ["Kite not authenticated to fetch historical data."]})

    # Load instruments for lookup
    instruments_df = load_instruments_cached(api_key, access_token, exchange)
    if "_error" in instruments_df.columns:
        # Fallback: Try loading all instruments if exchange specific fails (sometimes needed for indices)
        instruments_df = load_instruments_cached(api_key, access_token)
        if "_error" in instruments_df.columns:
            return pd.DataFrame({"_error": [instruments_df.loc[0, '_error']]})

    token = find_instrument_token(instruments_df, symbol, exchange)
    
    # ENHANCEMENT: Robust Benchmark Token Lookup
    if not token and symbol in ["NIFTY BANK", "NIFTYBANK", "BANKNIFTY", BENCHMARK_SYMBOL, "SENSEX"]:
        # Try fetching instruments specifically from NSE/BSE if it's a common index and the initial lookup failed
        index_exchange = "NSE" if symbol not in ["SENSEX"] else "BSE"
        instruments_secondary = load_instruments_cached(api_key, access_token, index_exchange)
        token = find_instrument_token(instruments_secondary, symbol, index_exchange)
        
        if not token:
            return pd.DataFrame({"_error": [f"Instrument token not found for {symbol} on {exchange} or {index_exchange}."]})

    if not token:
        return pd.DataFrame({"_error": [f"Instrument token not found for {symbol} on {exchange}."]})

    from_datetime = datetime.combine(from_date, datetime.min.time())
    to_datetime = datetime.combine(to_date, datetime.max.time())
    try:
        data = kite_instance.historical_data(token, from_date=from_datetime, to_date=to_datetime, interval=interval)
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=['close'], inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [str(e)]})


def find_instrument_token(df: pd.DataFrame, tradingsymbol: str, exchange: str = DEFAULT_EXCHANGE) -> int | None:
    if df.empty: return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & \
           (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    hits = df[mask]
    return int(hits.iloc[0]["instrument_token"]) if not hits.empty else None


def add_technical_indicators(df: pd.DataFrame, sma_periods, ema_periods, rsi_window, macd_fast, macd_slow, macd_signal, bb_window, bb_std_dev) -> pd.DataFrame:
    if df.empty or 'close' not in df.columns:
        return df.copy()

    df_copy = df.copy()
    
    for period in sma_periods:
        if period > 0: df_copy[f'SMA_{period}'] = ta.trend.sma_indicator(df_copy['close'], window=period)
    for period in ema_periods:
        if period > 0: df_copy[f'EMA_{period}'] = ta.trend.ema_indicator(df_copy['close'], window=period)
        
    df_copy['RSI'] = ta.momentum.rsi(df_copy['close'], window=rsi_window)
    
    macd_obj = ta.trend.MACD(df_copy['close'], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    df_copy['MACD'] = macd_obj.macd()
    df_copy['MACD_signal'] = macd_obj.macd_signal()
    df_copy['MACD_hist'] = macd_obj.macd_diff() 
    
    bollinger = ta.volatility.BollingerBands(df_copy['close'], window=bb_window, window_dev=bb_std_dev)
    df_copy['Bollinger_High'] = bollinger.bollinger_hband()
    df_copy['Bollinger_Low'] = bollinger.bollinger_lband()
    df_copy['Bollinger_Mid'] = bollinger.bollinger_mavg()
    df_copy['Bollinger_Width'] = bollinger.bollinger_wband()
    
    df_copy['Daily_Return'] = df_copy['close'].pct_change() * 100
    
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(method='ffill', inplace=True)
    return df_copy.dropna()

# ENHANCEMENT: Advanced Ratio Calculation with Alpha/Beta/VaR/CVaR/IR
def calculate_performance_metrics(returns_series: pd.Series, risk_free_rate: float = 0.0, benchmark_returns: pd.Series = None) -> dict:
    """
    Calculates comprehensive performance metrics including risk-adjusted ratios and CAPM metrics.
    
    :param returns_series: Daily percentage returns (e.g., 1.5 for 1.5%).
    :param risk_free_rate: Annual risk-free rate percentage (e.g., 6.0).
    :param benchmark_returns: Daily returns of the benchmark (e.g., NIFTY 50) as a decimal.
    """
    if returns_series.empty or len(returns_series) < 2: return {}
    
    # Ensure returns are in decimal form for calculation
    daily_returns_decimal = returns_series / 100.0 if returns_series.abs().mean() > 0.1 else returns_series
    daily_returns_decimal = daily_returns_decimal.replace([np.inf, -np.inf], np.nan).dropna()
    if daily_returns_decimal.empty: return {}

    cumulative_returns = (1 + daily_returns_decimal).cumprod() - 1
    total_return = cumulative_returns.iloc[-1] * 100 if not cumulative_returns.empty else 0
    num_periods = len(daily_returns_decimal)
    
    # Annualized Return (Geometric)
    if num_periods > 0 and (1 + daily_returns_decimal > 0).all():
        geometric_mean_daily_return = np.expm1(np.log1p(daily_returns_decimal).mean())
        annualized_return = ((1 + geometric_mean_daily_return) ** TRADING_DAYS_PER_YEAR - 1) * 100
    else: annualized_return = np.nan

    daily_volatility = daily_returns_decimal.std()
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR) * 100 if daily_volatility is not None else np.nan

    risk_free_rate_decimal = risk_free_rate / 100.0
    
    # Convert annual risk-free rate to daily equivalent
    daily_rf_rate = (1 + risk_free_rate_decimal)**(1/TRADING_DAYS_PER_YEAR) - 1

    # Sharpe Ratio
    sharpe_ratio = (annualized_return / 100 - risk_free_rate_decimal) / (annualized_volatility / 100) if annualized_volatility > 0 else np.nan

    # Max Drawdown
    if not cumulative_returns.empty:
        peak = (1 + cumulative_returns).cummax()
        drawdown = ((1 + cumulative_returns) - peak) / peak
        max_drawdown = drawdown.min() * 100 
    else: max_drawdown = np.nan

    # Sortino Ratio
    downside_returns = daily_returns_decimal[daily_returns_decimal < daily_rf_rate]
    downside_std_dev_daily = downside_returns.std() if not downside_returns.empty else np.nan
    annualized_downside_std_dev = downside_std_dev_daily * np.sqrt(TRADING_DAYS_PER_YEAR) if not np.isnan(downside_std_dev_daily) else np.nan
    sortino_ratio = (annualized_return / 100 - risk_free_rate_decimal) / (annualized_downside_std_dev) if annualized_downside_std_dev > 0 else np.nan

    # Calmar Ratio
    calmar_ratio = (annualized_return / 100) / abs(max_drawdown / 100) if max_drawdown != 0 and not np.isnan(max_drawdown) else np.nan

    # VaR and CVaR (95% confidence level)
    confidence_level = 0.05
    # VaR is the negative of the quantile, representing the maximum expected loss
    var_daily = -daily_returns_decimal.quantile(confidence_level)
    var_annualized = var_daily * np.sqrt(TRADING_DAYS_PER_YEAR) * 100 # Approx. annualization
    
    cvar_daily_losses = daily_returns_decimal[daily_returns_decimal < daily_returns_decimal.quantile(confidence_level)].mean()
    cvar_annualized = -cvar_daily_losses * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    # Beta, Alpha, Treynor, Information Ratio (Requires Benchmark)
    beta, alpha, treynor_ratio, information_ratio = np.nan, np.nan, np.nan, np.nan
    
    if benchmark_returns is not None and not benchmark_returns.empty:
        # Align indexes for comparison
        common_index = daily_returns_decimal.index.intersection(benchmark_returns.index)
        aligned_asset_returns = daily_returns_decimal.loc[common_index]
        # Benchmark returns might be in % or decimal based on how they were generated. Ensure decimal.
        aligned_benchmark_returns_decimal = benchmark_returns.loc[common_index]
        if aligned_benchmark_returns_decimal.abs().mean() > 0.1: # Check if it looks like %
             aligned_benchmark_returns_decimal /= 100.0

        if len(common_index) > 1:
            covariance_matrix = np.cov(aligned_asset_returns, aligned_benchmark_returns_decimal)
            covariance = covariance_matrix[0, 1]
            benchmark_variance = aligned_benchmark_returns_decimal.var()
            
            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
                
                # Annualized expected returns
                expected_asset_return_ann = annualized_return / 100
                
                # Benchmark annualized return (geometric mean)
                if (1 + aligned_benchmark_returns_decimal > 0).all():
                    bench_geom_mean_daily_return = np.expm1(np.log1p(aligned_benchmark_returns_decimal).mean())
                    benchmark_annualized_return = ((1 + bench_geom_mean_daily_return) ** TRADING_DAYS_PER_YEAR - 1)
                else:
                    benchmark_annualized_return = ((aligned_benchmark_returns_decimal.mean() + 1) ** TRADING_DAYS_PER_YEAR - 1)


                # Jensen's Alpha: Actual Return - Required Return (CAPM)
                alpha = (expected_asset_return_ann - (risk_free_rate_decimal + beta * (benchmark_annualized_return - risk_free_rate_decimal))) * 100
                
                # Treynor Ratio: (Rp - Rf) / Beta
                treynor_ratio = (expected_asset_return_ann - risk_free_rate_decimal) / beta if beta != 0 else np.nan
                
                # Information Ratio: (Rp - Rb) / Tracking Error
                tracking_error_daily = (aligned_asset_returns - aligned_benchmark_returns_decimal).std()
                tracking_error = tracking_error_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
                
                if tracking_error > 0:
                    information_ratio = (expected_asset_return_ann - benchmark_annualized_return) / tracking_error

    # Helper function for consistent rounding
    def round_if_float(x):
        return round(x, 4) if isinstance(x, (int, float)) and not np.isnan(x) else np.nan
    
    return {
        "Total Return (%)": round_if_float(total_return),
        "Annualized Return (%)": round_if_float(annualized_return),
        "Annualized Volatility (%)": round_if_float(annualized_volatility),
        "Sharpe Ratio": round_if_float(sharpe_ratio),
        "Sortino Ratio": round_if_float(sortino_ratio),
        "Max Drawdown (%)": round_if_float(max_drawdown),
        "Calmar Ratio": round_if_float(calmar_ratio),
        "VaR (95%, Ann.) (%)": round_if_float(var_annualized),
        "CVaR (95%, Ann.) (%)": round_if_float(cvar_annualized),
        f"Beta (vs {BENCHMARK_SYMBOL})": round_if_float(beta),
        f"Alpha (%) (vs {BENCHMARK_SYMBOL})": round_if_float(alpha),
        "Treynor Ratio": round_if_float(treynor_ratio),
        "Information Ratio": round_if_float(information_ratio)
    }

@st.cache_data(ttl=3600, show_spinner="Calculating historical index values...")
def _calculate_historical_index_value(api_key: str, access_token: str, constituents_df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    """
    Calculates the historical value of a custom index based on its constituents and weights.
    """
    if constituents_df.empty: return pd.DataFrame({"_error": ["No constituents provided for historical index calculation."]})

    all_historical_closes = {}
    
    progress_bar_placeholder = st.empty()
    progress_text_placeholder = st.empty()
    
    if st.session_state["instruments_df"].empty:
        st.session_state["instruments_df"] = load_instruments_cached(api_key, access_token, exchange)
        if "_error" in st.session_state["instruments_df"].columns:
            return pd.DataFrame({"_error": [st.session_state["instruments_df"].loc[0, '_error']]})

    for i, row in constituents_df.iterrows():
        symbol = row['symbol']
        progress_text_placeholder.text(f"Fetching historical data for {symbol} ({i+1}/{len(constituents_df)})...")
        
        hist_df = get_historical_data_cached(api_key, access_token, symbol, start_date, end_date, "day", exchange)
        
        if isinstance(hist_df, pd.DataFrame) and "_error" not in hist_df.columns and not hist_df.empty:
            all_historical_closes[symbol] = hist_df['close']
        else:
            error_msg = hist_df.get('_error', ['Unknown error'])[0] if isinstance(hist_df, pd.DataFrame) else 'Unknown error'
            st.warning(f"Could not fetch historical data for {symbol}. Skipping for historical calculation. Error: {error_msg}")
        progress_bar_placeholder.progress((i + 1) / len(constituents_df))

    progress_text_placeholder.empty()
    progress_bar_placeholder.empty()

    if not all_historical_closes:
        return pd.DataFrame({"_error": ["No historical data available for any constituent to build index."]})

    combined_closes = pd.DataFrame(all_historical_closes)
    
    combined_closes = combined_closes.ffill().bfill()
    combined_closes.dropna(how='all', inplace=True)

    if combined_closes.empty: return pd.DataFrame({"_error": ["Insufficient common historical data for index calculation after cleaning."]})

    weights_series = constituents_df.set_index('symbol')['Weights']
    common_symbols = weights_series.index.intersection(combined_closes.columns)
    if common_symbols.empty: return pd.DataFrame({"_error": ["No common symbols between historical data and constituent weights."]})

    aligned_combined_closes = combined_closes[common_symbols]
    aligned_weights = weights_series[common_symbols]

    weighted_closes = aligned_combined_closes.mul(aligned_weights, axis=1)
    index_history_series = weighted_closes.sum(axis=1)

    if not index_history_series.empty:
        first_valid_index = index_history_series.first_valid_index()
        if first_valid_index is not None:
            base_value = index_history_series[first_valid_index]
            if base_value != 0:
                index_history_df = pd.DataFrame({"index_value": (index_history_series / base_value) * 100})
                index_history_df.index.name = 'date'
                return index_history_df.dropna()
            else:
                return pd.DataFrame({"_error": ["First day's index value is zero, cannot normalize."]})
    return pd.DataFrame({"_error": ["Error in calculating or normalizing historical index values."]})

# ENHANCEMENT: Drawdown Chart function
def plot_drawdown_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty: return fig
    
    for col in df.columns:
        daily_returns = df[col].pct_change().dropna()
        cumulative_performance = (1 + daily_returns).cumprod()
        peak = cumulative_performance.expanding(min_periods=1).max()
        drawdown = ((cumulative_performance / peak) - 1) * 100
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode='lines', name=f'{col} Drawdown'))
        
    fig.update_layout(
        title_text="Drawdown Comparison (Percentage Loss from Peak)", 
        yaxis_title="Drawdown (%)", 
        template="plotly_dark", 
        height=400, 
        hovermode="x unified",
        yaxis_tickformat=".2f"
    )
    fig.update_yaxes(rangemode="tozero")
    return fig

# ENHANCEMENT: Rolling Volatility Chart function
def plot_rolling_volatility_chart(df: pd.DataFrame, window=30) -> go.Figure:
    fig = go.Figure()
    if df.empty: return fig
    
    for col in df.columns:
        daily_returns = df[col].pct_change()
        # Annualized rolling volatility
        rolling_vol = daily_returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, mode='lines', name=f'{col} {window}-Day Rolling Volatility'))
    fig.update_layout(
        title_text=f"{window}-Day Rolling Volatility Comparison (Annualized)", 
        yaxis_title="Annualized Volatility (%)", 
        template="plotly_dark", 
        height=400, 
        hovermode="x unified"
    )
    return fig

# ENHANCEMENT: Rolling Beta & Correlation Chart function
def plot_rolling_risk_charts(comparison_df: pd.DataFrame, benchmark_returns: pd.Series, window=60) -> tuple[go.Figure, go.Figure]:
    if comparison_df.empty or benchmark_returns is None or benchmark_returns.empty:
        return go.Figure(), go.Figure()

    daily_returns_df = comparison_df.pct_change().dropna()
    
    fig_beta = go.Figure()
    fig_corr = go.Figure()

    common_index = daily_returns_df.index.intersection(benchmark_returns.index)
    aligned_benchmark_returns = benchmark_returns.loc[common_index]
    
    # Ensure benchmark returns are in decimal form
    if aligned_benchmark_returns.abs().mean() > 0.1:
         aligned_benchmark_returns /= 100.0

    aligned_returns_df = daily_returns_df.loc[common_index]

    if aligned_returns_df.empty or len(aligned_returns_df) < window:
        return go.Figure(), go.Figure()

    for col in aligned_returns_df.columns:
        # Calculate rolling beta and correlation
        
        def calculate_rolling_beta(x):
            # x is the window of asset returns (Series)
            bench_window = aligned_benchmark_returns.loc[x.index]
            if bench_window.var() > 0:
                return x.cov(bench_window) / bench_window.var()
            return np.nan

        rolling_beta = aligned_returns_df[col].rolling(window=window).apply(calculate_rolling_beta, raw=False)
        rolling_corr = aligned_returns_df[col].rolling(window=window).corr(aligned_benchmark_returns)
        
        fig_beta.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta, mode='lines', name=f'{col} Beta'))
        fig_corr.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode='lines', name=f'{col} Correlation'))

    fig_beta.update_layout(title_text=f"{window}-Day Rolling Beta (vs {BENCHMARK_SYMBOL})", yaxis_title="Beta", template="plotly_dark", height=400, hovermode="x unified")
    fig_corr.update_layout(title_text=f"{window}-Day Rolling Correlation (vs {BENCHMARK_SYMBOL})", yaxis_title="Correlation", template="plotly_dark", height=400, hovermode="x unified", yaxis_range=[-1, 1])

    return fig_beta, fig_corr


# Function to generate factsheet as multi-section CSV (includes historical data)
def generate_factsheet_csv_content(
    factsheet_constituents_df_final: pd.DataFrame, 
    factsheet_history_df_final: pd.DataFrame,     
    last_comparison_df: pd.DataFrame,
    last_comparison_metrics: dict,
    current_live_value: float,
    index_name: str = "Custom Index",
    ai_agent_embed_snippet: str = None
) -> str:
    content = []
    
    content.append(f"Factsheet for {index_name}\n")
    content.append(f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    content.append("\n--- Index Overview ---\n")
    if current_live_value > 0 and not factsheet_constituents_df_final.empty:
        content.append(f"Current Live Calculated Index Value,₹{current_live_value:,.2f}\n")
    else:
        content.append("Current Live Calculated Index Value,N/A (Constituent data not available or comparison report only)\n")
    
    content.append("\n--- Constituents ---\n")
    if not factsheet_constituents_df_final.empty:
        const_export_df = factsheet_constituents_df_final.copy()
        if 'Last Price' not in const_export_df.columns: const_export_df['Last Price'] = np.nan
        if 'Weighted Price' not in const_export_df.columns: const_export_df['Weighted Price'] = np.nan
        
        const_export_df['Last Price'] = const_export_df['Last Price'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
        const_export_df['Weighted Price'] = const_export_df['Weighted Price'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
        
        content.append(const_export_df[['symbol', 'Name', 'Weights', 'Last Price', 'Weighted Price']].to_csv(index=False))
    else:
        content.append("No constituent data available.\n")

    content.append("\n--- Historical Performance (Normalized to 100) ---\n")
    if not factsheet_history_df_final.empty:
        content.append(factsheet_history_df_final.to_csv())
    else:
        content.append("No historical performance data available.\n")

    content.append("\n--- Performance Metrics ---\n")
    if last_comparison_metrics:
        # Flatten metrics dictionary (index names as columns)
        metrics_df = pd.DataFrame(last_comparison_metrics).T
        metrics_df = metrics_df.applymap(lambda x: f"{x:.4f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")
        
        # Transpose so metric names are rows in the CSV
        content.append(metrics_df.T.to_csv()) 
    else:
        content.append("No performance metrics available (run a comparison first).\n")

    content.append("\n--- Comparison Data (Normalized to 100) ---\n")
    if not last_comparison_df.empty:
        content.append(last_comparison_df.to_csv())
    else:
        content.append("No comparison data available.\n")

    return "".join(content)

# Function to generate factsheet as HTML (without historical time series)
def generate_factsheet_html_content(
    factsheet_constituents_df_final: pd.DataFrame,
    factsheet_history_df_final: pd.DataFrame,
    last_comparison_df: pd.DataFrame,
    last_comparison_metrics: dict,
    current_live_value: float,
    index_name: str = "Custom Index",
    ai_agent_embed_snippet: str = None
) -> str:
    """Generates a comprehensive factsheet as an HTML string, including visualizations but NOT raw historical data."""
    html_content_parts = []

    # Basic HTML structure and styling
    html_content_parts.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Invsion Connect Factsheet</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #1a1a1a; color: #e0e0e0; }
            .container { max-width: 900px; margin: auto; padding: 20px; background-color: #2b2b2b; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
            h1, h2, h3, h4 { color: #f0f0f0; border-bottom: 2px solid #444; padding-bottom: 5px; margin-top: 20px; }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; }
            th, td { border: 1px solid #444; padding: 8px; text-align: left; }
            th { background-color: #3a3a3a; }
            .metric { font-size: 1.1em; margin-bottom: 5px; }
            .plotly-graph { margin-top: 20px; border: 1px solid #444; border-radius: 5px; overflow: hidden; }
            .info-box { background-color: #334455; border-left: 5px solid #6699cc; padding: 10px; margin-top: 10px; border-radius: 4px; }
            .warning-box { background-color: #554433; border-left: 5px solid #cc9966; padding: 10px; margin-top: 10px; border-radius: 4px; }
            .ai-agent-section { margin-top: 30px; padding: 15px; background-color: #333344; border-radius: 8px; }
            .ai-agent-section h3 { color: #add8e6; border-bottom: 1px solid #555; padding-bottom: 5px; }
            @media print {
                body { background-color: #fff; color: #000; }
                .container { box-shadow: none; border: 1px solid #eee; background-color: #fff; }
                h1, h2, h3, h4 { color: #000; border-bottom-color: #ccc; }
                th, td { border-color: #ccc; }
                .plotly-graph { border: none; }
                .ai-agent-section { display: none; }
            }
        </style>
    </head>
    <body>
        <div class="container">
    """)

    # --- Factsheet Header ---
    html_content_parts.append(f"<h1>Invsion Connect Factsheet: {index_name}</h1>")
    html_content_parts.append(f"<p><strong>Generated On:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html_content_parts.append("<h2>Index Overview</h2>")
    
    if current_live_value > 0 and not factsheet_constituents_df_final.empty:
        html_content_parts.append(f"<p class='metric'><strong>Current Live Calculated Index Value:</strong> ₹{current_live_value:,.2f}</p>")
    else:
        html_content_parts.append("<p class='warning-box'>Current Live Calculated Index Value: N/A (Constituent data not available or comparison report only)</p>")

    # --- Constituents ---
    html_content_parts.append("<h3>Constituents</h3>")
    if not factsheet_constituents_df_final.empty:
        const_display_df = factsheet_constituents_df_final.copy()
        
        if 'Name' not in const_display_df.columns: const_display_df['Name'] = const_display_df['symbol'] 

        if 'Last Price' in const_display_df.columns: const_display_df['Last Price'] = const_display_df['Last Price'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
        else: const_display_df['Last Price'] = "N/A"
        if 'Weighted Price' in const_display_df.columns: const_display_df['Weighted Price'] = const_display_df['Weighted Price'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
        else: const_display_df['Weighted Price'] = "N/A"

        # Ensure only relevant columns are in the HTML table
        html_content_parts.append(const_display_df[['symbol', 'Name', 'Weights', 'Last Price', 'Weighted Price']].to_html(index=False, classes='table'))

        # Index Composition Pie Chart
        fig_pie = go.Figure(data=[go.Pie(labels=const_display_df['Name'], values=const_display_df['Weights'], hole=.3)])
        fig_pie.update_layout(title_text='Constituent Weights', height=400, template="plotly_dark")
        html_content_parts.append("<h3>Index Composition</h3>")
        html_content_parts.append(f"<div class='plotly-graph'>{fig_pie.to_html(full_html=False, include_plotlyjs='cdn')}</div>") 
    else:
        html_content_parts.append("<p class='warning-box'>No constituent data available for this index.</p>")
    
    # --- Performance Metrics ---
    html_content_parts.append("<h3>Performance Metrics Summary</h3>")
    if last_comparison_metrics:
        metrics_df = pd.DataFrame(last_comparison_metrics).T
        metrics_html = metrics_df.style.format("{:.4f}", na_rep="N/A").to_html(classes='table')
        html_content_parts.append(metrics_html)
    else:
        html_content_parts.append("<p class='warning-box'>No performance metrics available (run a comparison first).</p>")

    # --- Comparison Data (Chart Only) ---
    html_content_parts.append("<h3>Cumulative Performance Comparison (Normalized to 100)</h3>")
    if not last_comparison_df.empty:
        fig_comparison = go.Figure()
        for col in last_comparison_df.columns:
            fig_comparison.add_trace(go.Scatter(x=last_comparison_df.index, y=last_comparison_df[col], mode='lines', name=col))
        
        chart_title = "Multi-Index & Benchmark Performance"
        if index_name != "Consolidated Report" and index_name != "Comparison Report" and index_name != "Combined Index Constituents Report":
            chart_title = f"{index_name} vs Benchmarks Performance"

        fig_comparison.update_layout(
            title_text=chart_title,
            xaxis_title="Date",
            yaxis_title="Normalized Value (Base 100)",
            height=600,
            template="plotly_dark",
            hovermode="x unified"
        )
        html_content_parts.append(f"<div class='plotly-graph'>{fig_comparison.to_html(full_html=False, include_plotlyjs='cdn')}</div>") 
        
        # ENHANCEMENT: Risk Analysis Charts embedded in HTML
        html_content_parts.append("<h3>Risk Analysis Charts</h3>")
        
        fig_drawdown = plot_drawdown_chart(last_comparison_df)
        html_content_parts.append(f"<div class='plotly-graph'>{fig_drawdown.to_html(full_html=False, include_plotlyjs=False)}</div>")
        
        fig_rolling_vol = plot_rolling_volatility_chart(last_comparison_df)
        html_content_parts.append(f"<div class='plotly-graph'>{fig_rolling_vol.to_html(full_html=False, include_plotlyjs=False)}</div>")
        
        # ENHANCEMENT: Rolling Beta & Correlation Charts
        benchmark_returns_data = st.session_state.get("benchmark_historical_data", pd.DataFrame()).get('close', pd.Series()).pct_change().dropna()
        if not benchmark_returns_data.empty:
             fig_beta, fig_corr = plot_rolling_risk_charts(last_comparison_df, benchmark_returns_data, window=60)
             if fig_beta.data: html_content_parts.append(f"<div class='plotly-graph'>{fig_beta.to_html(full_html=False, include_plotlyjs=False)}</div>")
             if fig_corr.data: html_content_parts.append(f"<div class='plotly-graph'>{fig_corr.to_html(full_html=False, include_plotlyjs=False)}</div>")


    else:
        html_content_parts.append("<p class='warning-box'>No comparison data available.</p>")

    # --- Optional: Historical Performance Chart for the main index ---
    if (len(st.session_state["factsheet_selected_constituents_index_names"]) == 1 and 
        not factsheet_history_df_final.empty and 
        factsheet_history_df_final.shape[0] < 730): 
        html_content_parts.append("<h3>Index Historical Performance (Normalized to 100)</h3>")
        fig_hist_index = go.Figure(data=[go.Scatter(x=factsheet_history_df_final.index, y=factsheet_history_df_final['index_value'], mode='lines', name=index_name)])
        fig_hist_index.update_layout(title_text=f"{index_name} Historical Performance", template="plotly_dark", height=400)
        html_content_parts.append(f"<div class='plotly-graph'>{fig_hist_index.to_html(full_html=False, include_plotlyjs='cdn')}</div>")
    elif not factsheet_history_df_final.empty and len(st.session_state["factsheet_selected_constituents_index_names"]) == 1:
        html_content_parts.append(f"<p class='info-box'>Historical performance chart for {index_name} is too large (>2 years) for the HTML factsheet. Please refer to the CSV download.</p>")
    elif len(st.session_state["factsheet_selected_constituents_index_names"]) > 1:
         html_content_parts.append(f"<p class='info-box'>Historical performance chart for individual index constituents is not shown when multiple indexes are selected for the constituents section. Please refer to the CSV download for full historical data or the comparison chart above.</p>")


    # --- AI Agent Embed Snippet ---
    if ai_agent_embed_snippet:
        html_content_parts.append("""
            <div class="ai-agent-section">
                <h3>Embedded AI Agent Insights</h3>
        """)
        # Note: We must trust the user input here as Streamlit does not sanitize HTML
        html_content_parts.append(ai_agent_embed_snippet)
        html_content_parts.append("</div>")

    html_content_parts.append("""
        <div class="info-box">
            <p><strong>Note:</strong> Raw historical time series data (tables) is intentionally excluded from this HTML/PDF factsheet to keep it concise and visually focused. For the full historical data, please download the CSV factsheet.</p>
            <p>To convert this HTML file to PDF, open it in your web browser (e.g., Chrome, Firefox) and use the browser's "Print" function (Ctrl+P or Cmd+P). Then select "Save as PDF" from the printer options.</p>
        </div>
        </div>
    </body>
    </html>
    """)
    return "".join(html_content_parts)


# --- Sidebar: Kite Login ---
with st.sidebar:
    st.markdown("### 1. Login to Kite Connect")
    st.write("Click to open Kite login. You'll be redirected back with a `request_token`.")
    st.markdown(f"[🔗 Open Kite login]({login_url})")

    # Handle request_token from URL
    request_token_param = st.query_params.get("request_token")
    if request_token_param and not st.session_state["kite_access_token"]:
        st.info("Received request_token — exchanging for access token...")
        try:
            data = kite_unauth_client.generate_session(request_token_param, api_secret=KITE_CREDENTIALS["api_secret"])
            st.session_state["kite_access_token"] = data.get("access_token")
            st.session_state["kite_login_response"] = data
            st.sidebar.success("Kite Access token obtained.")
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to generate Kite session: {e}")

    if st.session_state["kite_access_token"]:
        if st.sidebar.button("Logout from Kite", key=f"kite_logout_btn_{st.session_state['kite_access_token'][:5]}"):
            st.session_state["kite_access_token"] = None
            st.session_state["kite_login_response"] = None
            st.session_state["instruments_df"] = pd.DataFrame()
            st.success("Logged out from Kite. Please login again.")
            st.rerun()
        st.success("Kite Authenticated ✅")
    else:
        st.info("Not authenticated with Kite yet.")


# --- Sidebar: Supabase Authentication ---
with st.sidebar:
    st.markdown("### 2. Supabase User Account")
    
    _refresh_supabase_session()

    if st.session_state["user_session"]:
        st.success(f"Logged into Supabase as: {st.session_state['user_session'].user.email}")
        if st.button("Logout from Supabase", key=f"supabase_logout_btn_{st.session_state['user_id']}"):
            try:
                supabase.auth.sign_out()
                _refresh_supabase_session()
                st.sidebar.success("Logged out from Supabase.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error logging out: {e}")
    else:
        with st.form("supabase_auth_form_logged_out_static_key"):
            st.markdown("##### Email/Password Login/Sign Up")
            email = st.text_input("Email", key="supabase_email_input", help="Your email for Supabase authentication.")
            password = st.text_input("Password", type="password", key="supabase_password_input", help="Your password for Supabase authentication.")
            
            col_auth1, col_auth2 = st.columns(2)
            with col_auth1:
                login_submitted = st.form_submit_button("Login")
            with col_auth2:
                signup_submitted = st.form_submit_button("Sign Up")

            if login_submitted:
                if email and password:
                    try:
                        with st.spinner("Logging in..."):
                            supabase.auth.sign_in_with_password({"email": email, "password": password})
                        _refresh_supabase_session()
                        st.success("Login successful! Welcome.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: {e}")
                else:
                    st.warning("Please enter both email and password for login.")
            
            if signup_submitted:
                if email and password:
                    try:
                        with st.spinner("Signing up..."):
                            # In Supabase's default configuration, sign_up also signs the user in immediately.
                            supabase.auth.sign_up({"email": email, "password": password})
                        _refresh_supabase_session()
                        st.success("Sign up successful! Please check your email to confirm your account.")
                        st.info("After confirming your email, you can log in.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sign up failed: {e}")
                else:
                    st.warning("Please enter both email and password for sign up.")

    st.markdown("---")
    st.markdown("### 3. Quick Data Access (Kite)")
    if st.session_state["kite_access_token"]:
        current_k_client_for_sidebar = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

        if st.button("Fetch Current Holdings", key="sidebar_fetch_holdings_btn"):
            try:
                holdings = current_k_client_for_sidebar.holdings()
                st.session_state["holdings_data"] = pd.DataFrame(holdings)
                st.success(f"Fetched {len(holdings)} holdings.")
            except Exception as e:
                st.error(f"Error fetching holdings: {e}")
        if st.session_state.get("holdings_data") is not None and not st.session_state["holdings_data"].empty:
            with st.expander("Show Holdings"):
                st.dataframe(st.session_state["holdings_data"])
    else:
        st.info("Login to Kite to access quick data.")


# --- Authenticated KiteConnect client (used by main tabs) ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])


# --- Main UI - Tabs for modules ---
tabs = st.tabs(["Market & Historical", "Custom Index"])
tab_market, tab_custom_index = tabs

# --- Tab Logic Functions ---

def render_market_historical_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("📈 Market Data & Historical Candles with TA")
    if not kite_client:
        st.info("Login first to fetch market data.")
        return
    if not api_key or not access_token:
        st.info("Kite authentication details required for cached data access.")
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
                st.success(f"Fetched LTP for {q_symbol}.")
            else:
                st.error(f"Market data fetch failed for {q_symbol}: {ltp_data.get('_error', 'Unknown error')}")
    with col_market_quote2:
        if st.session_state.get("current_market_data"):
            st.markdown("##### Latest Quote Details")
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
            with st.spinner(f"Fetching {interval} historical data for {hist_symbol}..."):
                df_hist = get_historical_data_cached(api_key, access_token, hist_symbol, from_date, to_date, interval, DEFAULT_EXCHANGE) 
                if isinstance(df_hist, pd.DataFrame) and "_error" not in df_hist.columns:
                    st.session_state["historical_data"] = df_hist
                    st.session_state["last_fetched_symbol"] = hist_symbol
                    st.success(f"Fetched {len(df_hist)} records for {hist_symbol}.")
                else:
                    st.error(f"Historical fetch failed: {df_hist.get('_error', 'Unknown error')}")

    if not st.session_state.get("historical_data", pd.DataFrame()).empty:
        df = st.session_state["historical_data"]

        with st.expander("Technical Indicator Settings and Plotting Options", expanded=False):
            st.markdown("#### Indicator Parameters")
            ta_c1, ta_c2, ta_c3 = st.columns(3)
            with ta_c1:
                sma_periods_str = st.text_input("SMA Periods (comma-sep)", "20,50")
                ema_periods_str = st.text_input("EMA Periods (comma-sep)", "12,26")
                rsi_window = st.number_input("RSI Window", 5, 50, 14)
            with ta_c2:
                macd_fast = st.number_input("MACD Fast", 5, 50, 12, key='macdf')
                macd_slow = st.number_input("MACD Slow", 10, 100, 26, key='macds')
                macd_signal = st.number_input("MACD Signal", 5, 50, 9, key='macdsign')
            with ta_c3:
                bb_window = st.number_input("Bollinger Band Window", 5, 50, 20, key='bbw')
                bb_std_dev = st.number_input("Bollinger Band Std Dev", 1.0, 4.0, 2.0, 0.5, key='bbstd')
                chart_type = st.selectbox("Chart Style", ["Candlestick", "Line"])
                indicators_to_plot = st.multiselect("Plot on Price Chart", ["SMA", "EMA", "Bollinger Bands"])

            try:
                sma_periods = [int(p.strip()) for p in sma_periods_str.split(',') if p.strip().isdigit()]
                ema_periods = [int(p.strip()) for p in ema_periods_str.split(',') if p.strip().isdigit()]
            except ValueError:
                st.error("Invalid input for moving average periods.")
                return

            df_with_ta = add_technical_indicators(df, sma_periods, ema_periods, rsi_window, macd_fast, macd_slow, macd_signal, bb_window, bb_std_dev)

        st.subheader(f"Technical Analysis for {st.session_state['last_fetched_symbol']} ({interval})")
        
        # Multi-panel chart
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.1, 0.2, 0.2])
        
        # Panel 1: Price and MAs/BB
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Price'), row=1, col=1)

        if "SMA" in indicators_to_plot:
            for p in sma_periods: fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta.get(f'SMA_{p}'), mode='lines', name=f'SMA {p}'), row=1, col=1)
        if "EMA" in indicators_to_plot:
            for p in ema_periods: fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta.get(f'EMA_{p}'), mode='lines', name=f'EMA {p}'), row=1, col=1)
        if "Bollinger Bands" in indicators_to_plot:
            fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['Bollinger_High'], mode='lines', line=dict(width=0.5, color='gray'), name='BB High'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['Bollinger_Low'], mode='lines', line=dict(width=0.5, color='gray'), fill='tonexty', fillcolor='rgba(128,128,128,0.2)', name='BB Low'), row=1, col=1)
        
        # Panel 2: Volume
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume'), row=2, col=1)
        
        # Panel 3: RSI
        fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['RSI'], mode='lines', name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)
        
        # Panel 4: MACD
        fig.add_trace(go.Bar(x=df_with_ta.index, y=df_with_ta['MACD_hist'], name='MACD Hist', marker_color='orange'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['MACD'], mode='lines', name='MACD Line', line=dict(color='blue')), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['MACD_signal'], mode='lines', name='MACD Signal', line=dict(color='red')), row=4, col=1)
        
        fig.update_layout(height=1000, xaxis_rangeslider_visible=False, title_text=f"{st.session_state['last_fetched_symbol']} Technical Analysis", template="plotly_white")
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=4, col=1)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Performance Snapshot (Basic Ratios)")
        # Calculate returns in decimal form (needed for the function)
        daily_returns_decimal = df['close'].pct_change().dropna() 
        metrics = calculate_performance_metrics(daily_returns_decimal, risk_free_rate=6.0) # Assume 6% RF for quick check
        if metrics:
            st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}).style.format("{:.4f}"))
        else:
            st.info("Not enough data to calculate performance metrics.")


def render_custom_index_tab(kite_client: KiteConnect | None, supabase_client: Client, api_key: str | None, access_token: str | None):
    st.header("📊 Custom Index Creation, Benchmarking & Export")
    st.markdown("Create your own weighted index, analyze its historical performance, compare it against benchmarks, and calculate key financial metrics.")
    
    if not kite_client:
        st.info("Login to Kite first to fetch live and historical prices for index constituents.")
        return
    if not st.session_state["user_id"]:
        st.info("Login with your Supabase account in the sidebar to save and load custom indexes.")
        return
    if not api_key or not access_token:
        st.info("Kite authentication details required for data access.")
        return

    # Helper function to load historical data for a given index/symbol
    @st.cache_data(ttl=3600, show_spinner="Fetching historical data for comparison...")
    def _fetch_and_normalize_data_for_comparison(
        name: str,
        data_type: str,
        comparison_start_date: datetime.date,
        comparison_end_date: datetime.date,
        constituents_df: pd.DataFrame = None,
        symbol: str = None,
        exchange: str = DEFAULT_EXCHANGE,
        api_key: str = None,
        access_token: str = None
    ) -> pd.DataFrame:
        """ Fetches and normalized historical data for comparison. """
        hist_df = pd.DataFrame()
        if data_type == "custom_index":
            if constituents_df is None or constituents_df.empty: return pd.DataFrame({"_error": [f"No constituents for custom index {name}."]})
            hist_df = _calculate_historical_index_value(api_key, access_token, constituents_df, comparison_start_date, comparison_end_date, exchange)
            if "_error" in hist_df.columns: return hist_df
            data_series = hist_df['index_value']
        elif data_type == "benchmark":
            if symbol is None: return pd.DataFrame({"_error": [f"No symbol for benchmark {name}."]})
            
            # Robust logic relies on get_historical_data_cached handling index lookup robustly
            # Pass the specified exchange for external benchmarks
            hist_df = get_historical_data_cached(api_key, access_token, symbol, comparison_start_date, comparison_end_date, "day", exchange)
            
            if "_error" in hist_df.columns: return hist_df
            data_series = hist_df['close']
        else:
            return pd.DataFrame({"_error": ["Invalid data_type for comparison."]})

        if data_series.empty:
            return pd.DataFrame({"_error": [f"No historical data for {name} within the selected range."]})

        first_valid_index = data_series.first_valid_index()
        if first_valid_index is not None and data_series[first_valid_index] != 0:
            normalized_series = (data_series / data_series[first_valid_index]) * 100
            # Raw values are needed for performance metric calculation
            return pd.DataFrame({'normalized_value': normalized_series, 'raw_values': data_series}).rename_axis('date')
        return pd.DataFrame({"_error": [f"Could not normalize {name} (first value is zero or no valid data in range)."]})


    # Helper function to render an index's details, charts, and export options
    def display_single_index_details(index_name: str, constituents_df: pd.DataFrame, index_history_df: pd.DataFrame, index_id: str | None = None, is_recalculated_live=False):
        st.markdown(f"#### Details for Index: **{index_name}** {'(Recalculated Live)' if is_recalculated_live else ''}")
        
        st.subheader("Constituents and Current Live Value")
        
        live_quotes = {}
        symbols_for_ltp = [sym for sym in constituents_df["symbol"]]
        
        # Load instruments for symbol to token lookup if not already loaded
        if st.session_state["instruments_df"].empty:
            st.session_state["instruments_df"] = load_instruments_cached(api_key, access_token, DEFAULT_EXCHANGE)
        
        if "_error" not in st.session_state["instruments_df"].columns:
            if symbols_for_ltp:
                try:
                    kc_client = get_authenticated_kite_client(api_key, access_token)
                    if kc_client:
                        instrument_identifiers = [f"{DEFAULT_EXCHANGE}:{s}" for s in symbols_for_ltp]
                        ltp_data_batch = kc_client.ltp(instrument_identifiers)
                        for sym in symbols_for_ltp:
                            key = f"{DEFAULT_EXCHANGE}:{sym}"
                            live_quotes[sym] = ltp_data_batch.get(key, {}).get("last_price", np.nan)
                except Exception: pass

        if 'Name' not in constituents_df.columns:
            inst_names = st.session_state["instruments_df"].set_index('tradingsymbol')['name'].to_dict() if not st.session_state["instruments_df"].empty else {}
            constituents_df['Name'] = constituents_df['symbol'].map(inst_names).fillna(constituents_df['symbol'])

        constituents_df_display = constituents_df.copy()
        constituents_df_display["Last Price"] = constituents_df_display["symbol"].map(live_quotes)
        constituents_df_display["Weighted Price"] = constituents_df_display["Last Price"] * constituents_df_display["Weights"]
        current_live_value = constituents_df_display["Weighted Price"].sum()

        st.dataframe(constituents_df_display[['symbol', 'Name', 'Weights', 'Last Price', 'Weighted Price']].style.format({
            "Weights": "{:.4f}",
            "Last Price": "₹{:,.2f}",
            "Weighted Price": "₹{:,.2f}"
        }), use_container_width=True)
        st.success(f"Current Live Calculated Index Value: **₹{current_live_value:,.2f}**")

        st.markdown("---")
        st.subheader("Index Composition")
        fig_pie = go.Figure(data=[go.Pie(labels=constituents_df_display['Name'], values=constituents_df_display['Weights'], hole=.3)])
        fig_pie.update_layout(title_text='Constituent Weights', height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        st.subheader("Export Options")
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            csv_constituents = constituents_df_display[['symbol', 'Name', 'Weights', 'Last Price', 'Weighted Price']].to_csv(index=False).encode('utf-8')
            st.download_button(label="Export Constituents to CSV", data=csv_constituents, file_name=f"{index_name}_constituents.csv", mime="text/csv", key=f"export_constituents_{index_id or index_name}")
        with col_export2:
            if not index_history_df.empty:
                csv_history = index_history_df.to_csv().encode('utf-8')
                st.download_button(label="Export Historical Performance to CSV", data=csv_history, file_name=f"{index_name}_historical_performance.csv", mime="text/csv", key=f"export_history_{index_id or index_name}")
            else: st.info("No historical data to export for this index.")

    # --- Section: Index Creation ---
    st.markdown("---")
    st.subheader("1. Create New Index")
    
    create_meth_c1, create_meth_c2 = st.columns(2)
    
    with create_meth_c1:
        st.markdown("##### From CSV Upload")
        uploaded_file = st.file_uploader("Upload CSV with index constituents", type=["csv"], key="index_upload_csv")
        if uploaded_file:
            try:
                df_constituents_new = pd.read_csv(uploaded_file)
                required_cols = {"symbol", "Weights"} # Name is optional, can be derived
                if not required_cols.issubset(set(df_constituents_new.columns)):
                    st.error(f"CSV must contain columns: `symbol`, `Weights`. Recommended: `Name`. Missing: {required_cols - set(df_constituents_new.columns)}")
                    return

                df_constituents_new["Weights"] = pd.to_numeric(df_constituents_new["Weights"], errors='coerce')
                df_constituents_new.dropna(subset=["Weights", "symbol"], inplace=True)
                
                if df_constituents_new.empty:
                    st.error("No valid constituents found in the CSV. Ensure 'symbol' and numeric 'Weights' columns are present.")
                    return

                total_weights = df_constituents_new["Weights"].sum()
                if total_weights <= 0:
                    st.error("Sum of weights must be positive.")
                    return
                df_constituents_new["Weights"] = df_constituents_new["Weights"] / total_weights
                if 'Name' not in df_constituents_new.columns:
                     df_constituents_new['Name'] = df_constituents_new['symbol']
                st.info(f"Loaded {len(df_constituents_new)} constituents from CSV. Weights have been normalized to sum to 1.")
                st.session_state["current_calculated_index_data"] = df_constituents_new[['symbol', 'Name', 'Weights']]

            except pd.errors.EmptyDataError:
                st.error("The uploaded CSV file is empty.")
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}.")
    
    with create_meth_c2:
        st.markdown("##### From Kite Holdings")
        weighting_scheme = st.selectbox("Weighting Scheme", ["Equal Weight", "Value Weighted (Investment Value)"], key="holdings_weight_scheme")
        if st.button("Create Index from Holdings", key="create_from_holdings_btn"):
            holdings_df = st.session_state.get("holdings_data")
            if holdings_df is None or holdings_df.empty:
                st.warning("Please fetch holdings from the sidebar first.")
            else:
                df_constituents_new = holdings_df[holdings_df['product'] != 'CDS'].copy()
                df_constituents_new.rename(columns={'tradingsymbol': 'symbol'}, inplace=True)
                df_constituents_new['Name'] = df_constituents_new['symbol']
                
                if weighting_scheme == "Equal Weight":
                    if len(df_constituents_new) == 0:
                        st.error("No valid holdings found to create an index.")
                        return 
                    df_constituents_new['Weights'] = 1 / len(df_constituents_new)
                else: # Value Weighted
                    df_constituents_new['investment_value'] = df_constituents_new['average_price'] * df_constituents_new['quantity']
                    total_value = df_constituents_new['investment_value'].sum()
                    if total_value == 0:
                         st.error("Total investment value is zero, cannot calculate value weights.")
                         return 
                    df_constituents_new['Weights'] = df_constituents_new['investment_value'] / total_value
                
                st.session_state["current_calculated_index_data"] = df_constituents_new[['symbol', 'Name', 'Weights']]
                st.success(f"Index created from {len(df_constituents_new)} holdings using {weighting_scheme}.")


    # --- Calculation and Saving Logic ---
    current_calculated_index_data_df = st.session_state.get("current_calculated_index_data", pd.DataFrame())
    current_calculated_index_history_df = st.session_state.get("current_calculated_index_history", pd.DataFrame())
    
    if not current_calculated_index_data_df.empty:
        st.subheader("Configure Historical Calculation for New Index")
        calc_c1, calc_c2 = st.columns(2)
        with calc_c1:
             hist_start_date = st.date_input("Historical Start Date", value=datetime.now().date() - timedelta(days=365), key="new_index_hist_start_date")
        with calc_c2:
             hist_end_date = st.date_input("Historical End Date", value=datetime.now().date(), key="new_index_hist_end_date")

        if st.button("Calculate Historical Index Values", key="calculate_new_index_btn_final"):
            if hist_start_date >= hist_end_date:
                st.error("Historical start date must be before end date.")
            else:
                index_history_df_new = _calculate_historical_index_value(api_key, access_token, current_calculated_index_data_df, hist_start_date, hist_end_date, DEFAULT_EXCHANGE)
            
                if not index_history_df_new.empty and "_error" not in index_history_df_new.columns:
                    st.session_state["current_calculated_index_history"] = index_history_df_new
                    st.success("Historical index values calculated successfully.")
                    st.session_state["factsheet_selected_constituents_index_names"] = ["Newly Calculated Index"] 
                else:
                    st.error(f"Failed to calculate historical index values for new index: {index_history_df_new.get('_error', ['Unknown error'])[0]}")
                    st.session_state["current_calculated_index_history"] = pd.DataFrame()
                    st.session_state["factsheet_selected_constituents_index_names"] = []
    
    
    if not current_calculated_index_data_df.empty and not current_calculated_index_history_df.empty:

        constituents_df_for_live = current_calculated_index_data_df.copy()
        live_quotes = {}
        symbols_for_ltp = [sym for sym in constituents_df_for_live["symbol"]]

        if symbols_for_ltp:
            try:
                kc_client = get_authenticated_kite_client(api_key, access_token)
                instrument_identifiers = [f"{DEFAULT_EXCHANGE}:{s}" for s in symbols_for_ltp]
                ltp_data_batch = kc_client.ltp(instrument_identifiers)
                for sym in symbols_for_ltp:
                    key = f"{DEFAULT_EXCHANGE}:{sym}"
                    live_quotes[sym] = ltp_data_batch.get(key, {}).get("last_price", np.nan)
            except Exception: pass
        
        if 'Name' not in constituents_df_for_live.columns:
            inst_names = st.session_state["instruments_df"].set_index('tradingsymbol')['name'].to_dict() if not st.session_state["instruments_df"].empty else {}
            constituents_df_for_live['Name'] = constituents_df_for_live['symbol'].map(inst_names).fillna(constituents_df_for_live['symbol'])

        constituents_df_for_live["Last Price"] = constituents_df_for_live["symbol"].map(live_quotes)
        constituents_df_for_live["Weighted Price"] = constituents_df_for_live["Last Price"] * constituents_df_for_live["Weights"]
        current_live_value_for_factsheet_display = constituents_df_for_live["Weighted Price"].sum() if not constituents_df_for_live["Weighted Price"].empty else 0.0

        display_single_index_details("Newly Calculated Index", constituents_df_for_live, current_calculated_index_history_df, index_id="new_index")
        
        st.markdown("---")
        st.subheader("Save Newly Created Index")
        index_name_to_save = st.text_input("Enter a unique name for this index to save it:", value="MyCustomIndex", key="new_index_save_name")
        if st.button("Save New Index to DB", key="save_new_index_to_db_btn"):
            if index_name_to_save and st.session_state["user_id"]:
                try:
                    with st.spinner("Saving index..."):
                        check_response = supabase_client.table("custom_indexes").select("id").eq("user_id", st.session_state["user_id"]).eq("index_name", index_name_to_save).execute()
                        if check_response.data:
                            st.warning(f"An index named '{index_name_to_save}' already exists. Please choose a different name.")
                        else:
                            history_df_to_save = current_calculated_index_history_df.reset_index()
                            history_df_to_save['date'] = history_df_to_save['date'].dt.strftime('%Y-%m-%dT%H:%M:%S') 

                            index_data = {
                                "user_id": st.session_state["user_id"],
                                "index_name": index_name_to_save,
                                "constituents": current_calculated_index_data_df[['symbol', 'Name', 'Weights']].to_dict(orient='records'),
                                "historical_performance": history_df_to_save.to_dict(orient='records')
                            }
                            supabase_client.table("custom_indexes").insert(index_data).execute()
                            st.success(f"Index '{index_name_to_save}' saved successfully!")
                            st.session_state["saved_indexes"] = [] 
                            st.session_state["current_calculated_index_data"] = pd.DataFrame()
                            st.session_state["current_calculated_index_history"] = pd.DataFrame()
                            st.session_state["factsheet_selected_constituents_index_names"] = []
                            st.rerun()
                except Exception as e:
                    st.error(f"Error saving new index: {e}")
            else:
                st.warning("Please enter an index name and ensure you are logged into Supabase.")
    
    st.markdown("---")
    st.subheader("2. Load & Manage Saved Indexes")
    if st.button("Load My Indexes from DB", key="load_my_indexes_db_btn"):
        try:
            with st.spinner("Loading indexes..."):
                response = supabase_client.table("custom_indexes").select("id, index_name, constituents, historical_performance").eq("user_id", st.session_state["user_id"]).execute()
            if response.data:
                st.session_state["saved_indexes"] = response.data
                st.success(f"Loaded {len(response.data)} indexes.")
            else:
                st.session_state["saved_indexes"] = []
                st.info("No saved indexes found for your account.")
        except Exception as e: st.error(f"Error loading indexes: {e}")
    
    saved_indexes = st.session_state.get("saved_indexes", [])
    if saved_indexes:
        index_names_from_db = [idx['index_name'] for idx in saved_indexes]
        
        selected_custom_indexes_names = st.multiselect(
            "Select saved custom indexes to include in comparison:", 
            options=index_names_from_db, 
            key="select_saved_indexes_for_comparison"
        )

        st.markdown("---")
        st.subheader("3. Configure & Run Multi-Index & Benchmark Comparison")
        
        col_comp_dates, col_comp_bench = st.columns(2)
        with col_comp_dates:
            comparison_start_date = st.date_input("Comparison Start Date", value=datetime.now().date() - timedelta(days=365), key="comparison_start_date")
            comparison_end_date = st.date_input("Comparison End Date", value=datetime.now().date(), key="comparison_end_date")
            if comparison_start_date >= comparison_end_date:
                st.error("Comparison start date must be before end date.")
                # Reset to valid dates to prevent calculation error
                comparison_start_date = datetime.now().date() - timedelta(days=365)
                comparison_end_date = datetime.now().date()


        with col_comp_bench:
            benchmark_symbols_str = st.text_area(
                f"Enter External Benchmark Symbols (comma-separated, {BENCHMARK_SYMBOL} is automatically used for Alpha/Beta)",
                value=f"{BENCHMARK_SYMBOL}, NIFTY BANK",
                height=80,
                key="comparison_benchmark_symbols_input"
            )
            external_benchmark_symbols = [s.strip().upper() for s in benchmark_symbols_str.split(',') if s.strip()]
            comparison_exchange = st.selectbox("Exchange for External Benchmarks", ["NSE", "BSE", "NFO"], key="comparison_bench_exchange_select")
        
        # ENHANCEMENT: Risk-free rate input
        risk_free_rate = st.number_input("Risk-Free Rate (%) for Ratios (e.g., 6.0)", min_value=0.0, max_value=20.0, value=6.0, step=0.1)

        if st.button("Run Multi-Index & Benchmark Comparison", key="run_multi_comparison_btn"):
            if not selected_custom_indexes_names and not external_benchmark_symbols:
                st.warning("Please select at least one custom index or enter at least one benchmark symbol for comparison.")
            else:
                all_normalized_data = {}
                all_performance_metrics = {}
                
                # --- Prepare Benchmark Data for Advanced Ratios (NIFTY 50) ---
                benchmark_returns = None
                with st.spinner(f"Fetching primary benchmark ({BENCHMARK_SYMBOL}) for risk ratios..."):
                    # Use robust fetching for the primary benchmark (always try NSE)
                    benchmark_df = get_historical_data_cached(api_key, access_token, BENCHMARK_SYMBOL, comparison_start_date, comparison_end_date, "day", "NSE")
                    if "_error" in benchmark_df.columns:
                        st.warning(f"Could not fetch primary benchmark '{BENCHMARK_SYMBOL}'. Risk ratios will be N/A. Error: {benchmark_df.loc[0, '_error']}")
                        st.session_state["benchmark_historical_data"] = pd.DataFrame()
                    else:
                        # Returns in decimal form (0.01 for 1%)
                        benchmark_returns = benchmark_df['close'].pct_change().dropna() 
                        st.session_state["benchmark_historical_data"] = benchmark_df
                
                if st.session_state["instruments_df"].empty:
                    with st.spinner("Loading instruments for comparison lookup..."):
                        st.session_state["instruments_df"] = load_instruments_cached(api_key, access_token, DEFAULT_EXCHANGE)
                
                if "_error" in st.session_state["instruments_df"].columns:
                    st.error(f"Failed to load instruments for comparison lookup: {st.session_state['instruments_df'].loc[0, '_error']}")
                    return

                comparison_items = selected_custom_indexes_names + external_benchmark_symbols
                
                # Process all items
                for item_name in comparison_items:
                    data_type = "custom_index" if item_name in selected_custom_indexes_names else "benchmark"
                    constituents_df = None
                    symbol = None
                    exchange = comparison_exchange
                    
                    if data_type == "custom_index":
                        db_index_data = next((idx for idx in saved_indexes if idx['index_name'] == item_name), None)
                        if db_index_data: constituents_df = pd.DataFrame(db_index_data['constituents'])
                    else:
                        symbol = item_name

                    
                    # Using dedicated function for fetch and normalize
                    with st.spinner(f"Processing {item_name} data..."):
                        normalized_df_result = _fetch_and_normalize_data_for_comparison(
                            name=item_name, data_type=data_type, comparison_start_date=comparison_start_date,
                            comparison_end_date=comparison_end_date, constituents_df=constituents_df,
                            symbol=symbol, exchange=exchange, api_key=api_key, access_token=access_token
                        )
                    
                    if "_error" not in normalized_df_result.columns:
                        all_normalized_data[item_name] = normalized_df_result['normalized_value']
                        
                        # Asset daily returns in decimal form (raw_values column holds non-normalized close/index value)
                        asset_daily_returns_decimal = normalized_df_result['raw_values'].pct_change().dropna()
                        
                        # Calculate all enhanced metrics
                        all_performance_metrics[item_name] = calculate_performance_metrics(
                            asset_daily_returns_decimal, 
                            risk_free_rate=risk_free_rate, 
                            benchmark_returns=benchmark_returns
                        )
                    else:
                        st.error(f"Error processing {item_name}: {normalized_df_result.loc[0, '_error']}")


                if all_normalized_data:
                    combined_comparison_df = pd.DataFrame(all_normalized_data)
                    combined_comparison_df.dropna(how='all', inplace=True)
                    
                    if not combined_comparison_df.empty:
                        st.session_state["last_comparison_df"] = combined_comparison_df
                        st.session_state["last_comparison_metrics"] = all_performance_metrics
                        st.success("Comparison data generated successfully.")
                    else:
                        st.warning("No common or sufficient data found for comparison. Please check selected indexes/benchmarks and date range.")
                else:
                    st.info("No data selected or fetched for comparison.")

        # Ensure last_comparison_df is a DataFrame
        last_comparison_df = st.session_state.get("last_comparison_df", pd.DataFrame())

        if not last_comparison_df.empty:
            st.markdown("#### Cumulative Performance Comparison (Normalized to 100)")
            fig_comparison = go.Figure()
            for col in last_comparison_df.columns:
                fig_comparison.add_trace(go.Scatter(x=last_comparison_df.index, y=last_comparison_df[col], mode='lines', name=col))
            
            fig_comparison.update_layout(
                title_text="Multi-Index & Benchmark Performance",
                xaxis_title="Date",
                yaxis_title="Normalized Value (Base 100)",
                height=600,
                template="plotly_dark",
                hovermode="x unified"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)

            st.markdown("#### Performance Metrics Summary")
            metrics_df = pd.DataFrame(st.session_state["last_comparison_metrics"]).T
            st.dataframe(metrics_df.style.format("{:.4f}", na_rep="N/A"), use_container_width=True) 
            
            # Display Risk Analysis Charts
            st.markdown("#### Risk Analysis Charts")
            risk_c1, risk_c2 = st.columns(2)
            with risk_c1:
                st.plotly_chart(plot_drawdown_chart(last_comparison_df), use_container_width=True)
            with risk_c2:
                st.plotly_chart(plot_rolling_volatility_chart(last_comparison_df), use_container_width=True)

            # Display Rolling Beta/Correlation Charts
            if not st.session_state["benchmark_historical_data"].empty:
                 # Returns need to be in decimal form for plot_rolling_risk_charts
                 benchmark_returns_for_rolling = st.session_state["benchmark_historical_data"]['close'].pct_change().dropna()
                 
                 if not benchmark_returns_for_rolling.empty:
                     beta_chart, corr_chart = plot_rolling_risk_charts(last_comparison_df, benchmark_returns_for_rolling, window=60)
                     
                     st.markdown("##### Rolling Relative Metrics (60-Day Window)")
                     rel_c1, rel_c2 = st.columns(2)
                     with rel_c1:
                         if beta_chart.data: st.plotly_chart(beta_chart, use_container_width=True)
                         else: st.info(f"Not enough common data points ({len(benchmark_returns_for_rolling)} points) for rolling beta calculation.")
                     with rel_c2:
                         if corr_chart.data: st.plotly_chart(corr_chart, use_container_width=True)
                         else: st.info(f"Not enough common data points ({len(benchmark_returns_for_rolling)} points) for rolling correlation calculation.")


        st.markdown("---")
        st.subheader("5. Generate and Download Consolidated Factsheet")
        st.info("This will generate a factsheet. If a new index is calculated or a single saved index is selected, it will create a detailed report for that index. Otherwise, it will generate a comparison-only factsheet if comparison data is available.")
        
        # --- Factsheet data preparation logic ---
        factsheet_constituents_df_final = pd.DataFrame()
        factsheet_history_df_final = pd.DataFrame()
        factsheet_index_name_final = "Consolidated Report"
        current_live_value_for_factsheet_final = 0.0
        
        available_constituents_for_factsheet = ["None"]
        if not current_calculated_index_data_df.empty: available_constituents_for_factsheet.append("Newly Calculated Index")
        if saved_indexes: available_constituents_for_factsheet.extend(index_names_from_db)
        
        st.markdown("---")
        st.subheader("Factsheet Content Selection")
        
        selected_constituents_for_factsheet = st.multiselect(
            "Select which custom index(es) constituents and live value to include in the factsheet:",
            options=available_constituents_for_factsheet,
            default=st.session_state.get("factsheet_selected_constituents_index_names", []),
            key="factsheet_constituents_selector"
        )
        st.session_state["factsheet_selected_constituents_index_names"] = selected_constituents_for_factsheet

        all_constituents_dfs = []
        
        if selected_constituents_for_factsheet and "None" not in selected_constituents_for_factsheet:
            if "Newly Calculated Index" in selected_constituents_for_factsheet and not current_calculated_index_data_df.empty:
                all_constituents_dfs.append(current_calculated_index_data_df.copy())
            
            for index_name in selected_constituents_for_factsheet:
                if index_name == "Newly Calculated Index": continue
                selected_db_index_data = next((idx for idx in saved_indexes if idx['index_name'] == index_name), None)
                if selected_db_index_data:
                    all_constituents_dfs.append(pd.DataFrame(selected_db_index_data['constituents']).copy())

            if all_constituents_dfs:
                factsheet_constituents_df_final = pd.concat(all_constituents_dfs, ignore_index=True)
                
                # Combine constituents if multiple indexes are selected
                factsheet_constituents_df_final = factsheet_constituents_df_final.groupby(['symbol', 'Name'])['Weights'].sum().reset_index()
                factsheet_constituents_df_final['Weights'] = factsheet_constituents_df_final['Weights'] / factsheet_constituents_df_final['Weights'].sum()

                if len(selected_constituents_for_factsheet) == 1:
                    factsheet_index_name_final = selected_constituents_for_factsheet[0]
                    if factsheet_index_name_final == "Newly Calculated Index":
                         factsheet_history_df_final = current_calculated_index_history_df.copy()
                    else:
                        db_data = next((idx for idx in saved_indexes if idx['index_name'] == factsheet_index_name_final), None)
                        if db_data and db_data.get('historical_performance'):
                            history_from_db = pd.DataFrame(db_data['historical_performance'])
                            if not history_from_db.empty:
                                history_from_db['date'] = pd.to_datetime(history_from_db['date'])
                                history_from_db.set_index('date', inplace=True)
                                history_from_db.sort_index(inplace=True)
                                factsheet_history_df_final = history_from_db
                else:
                    factsheet_index_name_final = "Combined Index Constituents Report"

                # Fetch live prices for constituents (used in factsheet constituent table)
                live_quotes_for_factsheet_final = {}
                symbols_for_ltp_for_factsheet_final = [sym for sym in factsheet_constituents_df_final["symbol"]]
                if not st.session_state["instruments_df"].empty and symbols_for_ltp_for_factsheet_final:
                    try:
                        kc_client = get_authenticated_kite_client(api_key, access_token)
                        if kc_client:
                            instrument_identifiers = [f"{DEFAULT_EXCHANGE}:{s}" for s in symbols_for_ltp_for_factsheet_final]
                            ltp_data_batch_for_factsheet_final = kc_client.ltp(instrument_identifiers)
                            for sym in symbols_for_ltp_for_factsheet_final:
                                key = f"{DEFAULT_EXCHANGE}:{sym}"
                                live_quotes_for_factsheet_final[sym] = ltp_data_batch_for_factsheet_final.get(key, {}).get("last_price", np.nan)
                    except Exception as e:
                        st.warning(f"Error fetching batch LTP for factsheet live value: {e}. Live prices might be partial.")
                
                if 'Name' not in factsheet_constituents_df_final.columns and not st.session_state["instruments_df"].empty:
                    instrument_names_for_factsheet_final = st.session_state["instruments_df"].set_index('tradingsymbol')['name'].to_dict()
                    factsheet_constituents_df_final['Name'] = factsheet_constituents_df_final['symbol'].map(instrument_names_for_factsheet_final).fillna(factsheet_constituents_df_final['symbol'])
                elif 'Name' not in factsheet_constituents_df_final.columns:
                    factsheet_constituents_df_final['Name'] = factsheet_constituents_df_final['symbol']

                factsheet_constituents_df_final["Last Price"] = factsheet_constituents_df_final["symbol"].map(live_quotes_for_factsheet_final)
                factsheet_constituents_df_final["Weighted Price"] = factsheet_constituents_df_final["Last Price"] * factsheet_constituents_df_final["Weights"]
                current_live_value_for_factsheet_final = factsheet_constituents_df_final["Weighted Price"].sum() if not factsheet_constituents_df_final["Weighted Price"].empty else 0.0
            else:
                # Fallback if selection was made but data was empty
                factsheet_constituents_df_final = pd.DataFrame()
                factsheet_history_df_final = pd.DataFrame()
                factsheet_index_name_final = "Comparison Report" if not last_comparison_df.empty else "Consolidated Report"
                current_live_value_for_factsheet_final = 0.0
        else:
            # If "None" or nothing selected
            factsheet_constituents_df_final = pd.DataFrame()
            factsheet_history_df_final = pd.DataFrame()
            factsheet_index_name_final = "Comparison Report" if not last_comparison_df.empty else "Consolidated Report"
            current_live_value_for_factsheet_final = 0.0

        ai_agent_snippet_input = st.text_area(
            "Optional: Paste HTML snippet for an embedded AI Agent (e.g., iframe code)",
            height=150,
            key="ai_agent_embed_snippet_input",
            value="" # Removed default value to keep output clean unless user provides one
        )

        col_factsheet_download_options_1, col_factsheet_download_options_2 = st.columns(2)

        with col_factsheet_download_options_1:
            if st.button("Generate & Download Factsheet (CSV)", key="generate_download_factsheet_csv_btn"):
                if not factsheet_constituents_df_final.empty or not factsheet_history_df_final.empty or not last_comparison_df.empty:
                    factsheet_csv_content = generate_factsheet_csv_content(
                        factsheet_constituents_df_final=factsheet_constituents_df_final,
                        factsheet_history_df_final=factsheet_history_df_final,
                        last_comparison_df=last_comparison_df,
                        last_comparison_metrics=st.session_state.get("last_comparison_metrics", {}),
                        current_live_value=current_live_value_for_factsheet_final,
                        index_name=factsheet_index_name_final,
                        ai_agent_embed_snippet=None # CSV doesn't include AI snippet
                    )
                    st.session_state["last_facts_data"] = factsheet_csv_content.encode('utf-8')
                    st.download_button(
                        label="Download CSV Factsheet",
                        data=st.session_state["last_facts_data"],
                        file_name=f"InvsionConnect_Factsheet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="factsheet_download_button_final_csv_trigger",
                        help="Includes constituents, historical data, comparison data, and metrics."
                    )
                    st.success("CSV Factsheet generated and ready for download!")
                else:
                    st.warning("No data available to generate a factsheet. Please calculate a new index, load a saved index, or run a comparison first.")

        with col_factsheet_download_options_2:
            if st.button("Generate & Download Factsheet (HTML/PDF)", key="generate_download_factsheet_html_btn"):
                if not factsheet_constituents_df_final.empty or not factsheet_history_df_final.empty or not last_comparison_df.empty:
                    factsheet_html_content = generate_factsheet_html_content(
                        factsheet_constituents_df_final=factsheet_constituents_df_final,
                        factsheet_history_df_final=factsheet_history_df_final,
                        last_comparison_df=last_comparison_df,
                        last_comparison_metrics=st.session_state.get("last_comparison_metrics", {}),
                        current_live_value=current_live_value_for_factsheet_final,
                        index_name=factsheet_index_name_final,
                        ai_agent_embed_snippet=ai_agent_snippet_input if ai_agent_snippet_input.strip() else None
                    )
                    st.session_state["last_factsheet_html_data"] = factsheet_html_content.encode('utf-8')

                    st.download_button(
                        label="Download HTML Factsheet",
                        data=st.session_state["last_factsheet_html_data"],
                        file_name=f"InvsionConnect_Factsheet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        key="factsheet_download_button_final_html_trigger",
                        help="Includes charts for performance and composition, and optional embedded AI agent. Open in browser to Print to PDF."
                    )
                    st.success("HTML Factsheet generated and ready for download! (Open in browser, then 'Print to PDF')")
                else:
                    st.warning("No data available to generate a factsheet. Please calculate a new index, load a saved index, or run a comparison first.")

        st.markdown("---")
        st.subheader("6. View/Delete Individual Saved Indexes")
        
        index_names_from_db_for_selector = [idx['index_name'] for idx in saved_indexes] if saved_indexes else []

        selected_index_to_manage = st.selectbox(
            "Select a single saved index to view details or delete:", 
            ["--- Select ---"] + index_names_from_db_for_selector, 
            key="select_single_saved_index_to_manage"
        )

        selected_db_index_data = None
        if selected_index_to_manage != "--- Select ---":
            selected_db_index_data = next((idx for idx in saved_indexes if idx['index_name'] == selected_index_to_manage), None)
            if selected_db_index_data:
                loaded_constituents_df = pd.DataFrame(selected_db_index_data['constituents'])
                loaded_historical_performance_raw = selected_db_index_data.get('historical_performance')

                loaded_historical_df = pd.DataFrame()
                is_recalculated_live = False

                if loaded_historical_performance_raw:
                    try:
                        loaded_historical_df = pd.DataFrame(loaded_historical_performance_raw)
                        loaded_historical_df['date'] = pd.to_datetime(loaded_historical_df['date'])
                        loaded_historical_df.set_index('date', inplace=True)
                        loaded_historical_df.sort_index(inplace=True)
                        if loaded_historical_df.empty or 'index_value' not in loaded_historical_df.columns:
                            raise ValueError("Loaded historical data is invalid.")
                    except Exception:
                        st.warning(f"Saved historical data for '{selected_index_to_manage}' is invalid or outdated. Attempting live recalculation for display...")
                        loaded_historical_df = pd.DataFrame()

                if loaded_historical_df.empty:
                    min_date = (datetime.now().date() - timedelta(days=365))
                    max_date = datetime.now().date()
                    recalculated_historical_df = _calculate_historical_index_value(api_key, access_token, loaded_constituents_df, min_date, max_date, DEFAULT_EXCHANGE)
                    
                    if not recalculated_historical_df.empty and "_error" not in recalculated_historical_df.columns:
                        loaded_historical_df = recalculated_historical_df
                        is_recalculated_live = True
                        st.success("Historical data recalculated live successfully.")
                    else:
                        st.error(f"Failed to recalculate historical data: {recalculated_historical_df.get('_error', ['Unknown error'])}")

                display_single_index_details(selected_index_to_manage, loaded_constituents_df, loaded_historical_df, selected_db_index_data['id'], is_recalculated_live)
                
                st.markdown("---")
                if st.button(f"Delete Index '{selected_index_to_manage}'", key=f"delete_index_{selected_db_index_data['id']}", type="primary"):
                    try:
                        supabase_client.table("custom_indexes").delete().eq("id", selected_db_index_data['id']).execute()
                        st.success(f"Index '{selected_index_to_manage}' deleted successfully.")
                        st.session_state["saved_indexes"] = []
                        st.rerun()
                    except Exception as e: st.error(f"Error deleting index: {e}")
    else:
        st.info("No saved indexes to manage yet. Load them using the button above.")


# --- Main Application Logic (Tab Rendering) ---
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

with tab_market: render_market_historical_tab(k, api_key, access_token)
with tab_custom_index: render_custom_index_tab(k, supabase, api_key, access_token)
