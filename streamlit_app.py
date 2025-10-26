import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import json
import threading # Kept for potential future async operations, though not directly used for websockets anymore
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import ta  # Technical Analysis library

# Supabase imports
from supabase import create_client, Client

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Kite Connect - Advanced Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("Invsion Connect")
st.markdown("A comprehensive platform for fetching market data, performing ML-driven analysis, risk assessment, and live data streaming.")

# --- Global Constants & Session State Initialization ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"

# Initialize session state variables if they don't exist
if "kite_access_token" not in st.session_state:
    st.session_state["kite_access_token"] = None
if "kite_login_response" not in st.session_state:
    st.session_state["kite_login_response"] = None
if "instruments_df" not in st.session_state:
    st.session_state["instruments_df"] = pd.DataFrame()
if "historical_data" not in st.session_state:
    st.session_state["historical_data"] = pd.DataFrame()
if "last_fetched_symbol" not in st.session_state:
    st.session_state["last_fetched_symbol"] = None
if "user_session" not in st.session_state:
    st.session_state["user_session"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "saved_indexes" not in st.session_state:
    st.session_state["saved_indexes"] = []
if "current_calculated_index_data" not in st.session_state: # To store current CSV's index data
    st.session_state["current_calculated_index_data"] = None
if "current_calculated_index_history" not in st.session_state: # To store historical index values for plotting
    st.session_state["current_calculated_index_history"] = pd.DataFrame()
# Removed WebSocket-related session state variables


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
@st.cache_resource(ttl=3600) # Cache for 1 hour to prevent re-initializing on every rerun
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

# Helper to create an authenticated KiteConnect instance
def get_authenticated_kite_client(api_key: str | None, access_token: str | None) -> KiteConnect | None:
    if api_key and access_token:
        k_instance = KiteConnect(api_key=api_key)
        k_instance.set_access_token(access_token)
        return k_instance
    return None


@st.cache_data(ttl=86400, show_spinner="Loading instruments...") # Cache for 24 hours
def load_instruments_cached(api_key: str, access_token: str, exchange: str = None) -> pd.DataFrame:
    """Returns pandas.DataFrame of instrument data, using an internally created Kite instance."""
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        # Return an empty DataFrame with an error indicator
        return pd.DataFrame({"_error": ["Kite not authenticated to load instruments."]})
    try:
        instruments = kite_instance.instruments(exchange) if exchange else kite_instance.instruments()
        df = pd.DataFrame(instruments)
        if "instrument_token" in df.columns:
            df["instrument_token"] = df["instrument_token"].astype("int64")
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [f"Failed to load instruments for {exchange or 'all exchanges'}: {e}"]})

@st.cache_data(ttl=60) # Cache LTP for 1 minute
def get_ltp_price_cached(api_key: str, access_token: str, symbol: str, exchange: str = DEFAULT_EXCHANGE):
    """Fetches LTP for a symbol, using an internally created Kite instance."""
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return {"_error": "Kite not authenticated to fetch LTP."}
    
    exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
    try:
        ltp_data = kite_instance.ltp([exchange_symbol])
        return ltp_data.get(exchange_symbol)
    except Exception as e:
        return {"_error": str(e)}

@st.cache_data(ttl=3600) # Cache historical data for 1 hour
def get_historical_data_cached(api_key: str, access_token: str, symbol: str, from_date: datetime.date, to_date: datetime.date, interval: str, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    """Fetches historical data for a symbol, using an internally created Kite instance."""
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame({"_error": ["Kite not authenticated to fetch historical data."]})

    # Load instruments for token lookup (this calls the *cached* load_instruments_cached)
    instruments_df = load_instruments_cached(api_key, access_token, exchange)
    if "_error" in instruments_df.columns:
        return pd.DataFrame({"_error": [instruments_df.loc[0, '_error']]}) # Access the error message correctly

    token = find_instrument_token(instruments_df, symbol, exchange)
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
    if df.empty:
        return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & \
           (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    hits = df[mask]
    return int(hits.iloc[0]["instrument_token"]) if not hits.empty else None


# No caching for this as it modifies df and generates many dynamic columns
def add_technical_indicators(df: pd.DataFrame, sma_short=10, sma_long=50, rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9, bb_window=20, bb_std_dev=2) -> pd.DataFrame:
    if df.empty or 'close' not in df.columns:
        st.warning("Insufficient data or missing 'close' column for indicator calculation.")
        return pd.DataFrame()

    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    df_copy['SMA_Short'] = ta.trend.sma_indicator(df_copy['close'], window=sma_short)
    df_copy['SMA_Long'] = ta.trend.sma_indicator(df_copy['close'], window=sma_long)
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
    df_copy['Lag_1_Close'] = df_copy['close'].shift(1)
    
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(method='ffill', inplace=True)
    return df_copy

def calculate_performance_metrics(returns_series: pd.Series, risk_free_rate: float = 0.0) -> dict:
    if returns_series.empty or len(returns_series) < 2:
        return {}
    
    # Ensure returns are not already in percentage form for cumulative calculation
    daily_returns_decimal = returns_series / 100.0

    cumulative_returns = (1 + daily_returns_decimal).cumprod() - 1
    total_return = cumulative_returns.iloc[-1] * 100

    num_periods = len(returns_series)
    if num_periods > 0 and daily_returns_decimal.min() > -1: # Avoid issues with -100% return
        annualized_return = ((1 + daily_returns_decimal).prod())**(TRADING_DAYS_PER_YEAR/num_periods) - 1
    else:
        annualized_return = 0
    annualized_return *= 100 # Convert to percentage

    daily_volatility = returns_series.std()
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR) if daily_volatility is not None else 0

    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else np.nan

    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / (peak + 1e-9)
    max_drawdown = drawdown.min() * 100

    negative_returns = returns_series[returns_series < 0]
    downside_std_dev = negative_returns.std()
    sortino_ratio = (annualized_return - risk_free_rate) / (downside_std_dev * np.sqrt(TRADING_DAYS_PER_YEAR)) if downside_std_dev != 0 else np.nan

    return {
        "Total Return (%)": total_return,
        "Annualized Return (%)": annualized_return,
        "Annualized Volatility (%)": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Sortino Ratio": sortino_ratio
    }

@st.cache_data(ttl=3600, show_spinner="Calculating historical index values...")
def _calculate_historical_index_value(api_key: str, access_token: str, constituents_df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    """
    Calculates the historical value of a custom index based on its constituents and weights.
    Returns a DataFrame with 'date' and 'index_value'.
    """
    if constituents_df.empty:
        return pd.DataFrame({"_error": ["No constituents provided for historical index calculation."]})

    all_historical_closes = {}
    
    # Use a single progress bar for all fetches
    progress_bar_placeholder = st.empty()
    progress_text_placeholder = st.empty()
    
    # Ensure instruments are loaded for symbol to token lookup
    if st.session_state["instruments_df"].empty:
        st.session_state["instruments_df"] = load_instruments_cached(api_key, access_token, exchange)
        if "_error" in st.session_state["instruments_df"].columns:
            return pd.DataFrame({"_error": [st.session_state["instruments_df"].loc[0, '_error']]})

    for i, row in constituents_df.iterrows():
        symbol = row['symbol']
        progress_text_placeholder.text(f"Fetching historical data for {symbol} ({i+1}/{len(constituents_df)})...")
        
        # Use the cached historical data function
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
    
    # Forward-fill and then back-fill any missing daily prices to be more robust
    combined_closes = combined_closes.ffill().bfill()
    combined_closes.dropna(inplace=True) # Drop rows where all are still NaN

    if combined_closes.empty:
        return pd.DataFrame({"_error": ["Insufficient common historical data for index calculation after cleaning."]})

    # Calculate daily weighted prices
    weights_series = constituents_df.set_index('symbol')['Weights']
    
    # Reindex combined_closes columns to match weights_series index
    aligned_combined_closes = combined_closes[weights_series.index.intersection(combined_closes.columns)]

    if aligned_combined_closes.empty:
        return pd.DataFrame({"_error": ["No common symbols between historical data and constituent weights after alignment."]})

    # Apply weights
    weighted_closes = aligned_combined_closes.mul(weights_series[aligned_combined_closes.columns], axis=1)

    # Sum the weighted prices for each day to get the index value
    index_history_series = weighted_closes.sum(axis=1)

    # Normalize the index to a base value (e.g., 100 on the first common date)
    if not index_history_series.empty:
        base_value = index_history_series.iloc[0]
        if base_value != 0:
            index_history_df = pd.DataFrame({
                "index_value": (index_history_series / base_value) * 100
            })
            index_history_df.index.name = 'date' # Ensure index name for later merging/plotting
            return index_history_df
        else:
            return pd.DataFrame({"_error": ["First day's index value is zero, cannot normalize."]})
    return pd.DataFrame({"_error": ["Error in calculating or normalizing historical index values."]})


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
            st.query_params.clear() # Clear request_token from URL
            st.rerun() # Rerun to refresh UI
        except Exception as e:
            st.sidebar.error(f"Failed to generate Kite session: {e}")

    if st.session_state["kite_access_token"]:
        st.success("Kite Authenticated ✅")
        if st.sidebar.button("Logout from Kite", key="kite_logout_btn"):
            st.session_state["kite_access_token"] = None
            st.session_state["kite_login_response"] = None
            st.session_state["instruments_df"] = pd.DataFrame() # Clear cached instruments
            st.success("Logged out from Kite. Please login again.")
            st.rerun()
    else:
        st.info("Not authenticated with Kite yet.")


# --- Sidebar: Supabase Authentication ---
with st.sidebar:
    st.markdown("### 2. Supabase User Account")
    
    # Check/refresh Supabase session
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

    _refresh_supabase_session()

    if st.session_state["user_session"]:
        st.success(f"Logged into Supabase as: {st.session_state['user_session'].user.email}")
        if st.button("Logout from Supabase", key="supabase_logout_btn"):
            try:
                supabase.auth.sign_out()
                _refresh_supabase_session() # Update session state immediately
                st.sidebar.success("Logged out from Supabase.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error logging out: {e}")
    else:
        with st.form("supabase_auth_form"):
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
                            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
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
                            response = supabase.auth.sign_up({"email": email, "password": password})
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
                holdings = current_k_client_for_sidebar.holdings() # Direct call
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
# Removed Websocket and Instrument Lookup tabs
tabs = st.tabs(["Dashboard", "Portfolio", "Orders", "Market & Historical", "Machine Learning Analysis", "Risk & Stress Testing", "Performance Analysis", "Multi-Asset Analysis", "Custom Index"])
tab_dashboard, tab_portfolio, tab_orders, tab_market, tab_ml, tab_risk, tab_performance, tab_multi_asset, tab_custom_index = tabs

# --- Tab Logic Functions ---

def render_dashboard_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Personalized Dashboard")
    st.write("Welcome to your advanced financial analysis dashboard.")

    if not kite_client:
        st.info("Please login to Kite Connect to view your personalized dashboard.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account Summary")
        try:
            profile = kite_client.profile() # Direct call, not cached
            margins = kite_client.margins() # Direct call, not cached
            st.metric("Account Holder", profile.get("user_name", "N/A"))
            st.metric("Available Equity Margin", f"₹{margins.get('equity', {}).get('available', {}).get('live_balance', 0):,.2f}")
            st.metric("Available Commodity Margin", f"₹{margins.get('commodity', {}).get('available', {}).get('live_balance', 0):,.2f}")
        except Exception as e:
            st.warning(f"Could not fetch full account summary: {e}")

    with col2:
        st.subheader("Market Insight (NIFTY 50)")
        if api_key and access_token:
            nifty_ltp_data = get_ltp_price_cached(api_key, access_token, "NIFTY 50", DEFAULT_EXCHANGE) # Use cached LTP
            if nifty_ltp_data and "_error" not in nifty_ltp_data:
                nifty_ltp = nifty_ltp_data.get("last_price", 0.0)
                nifty_change = nifty_ltp_data.get("change", 0.0)
                st.metric("NIFTY 50 (LTP)", f"₹{nifty_ltp:,.2f}", delta=f"{nifty_change:.2f}%")
            else:
                st.warning(f"Could not fetch NIFTY 50 LTP: {nifty_ltp_data.get('_error', 'Unknown error')}")
        else:
            st.info("Kite not authenticated to fetch NIFTY 50 LTP.")

        if st.session_state.get("historical_data_NIFTY", pd.DataFrame()).empty:
            if st.button("Load NIFTY 50 Historical for Chart", key="dashboard_load_nifty_hist_btn"):
                if api_key and access_token:
                    with st.spinner("Fetching NIFTY 50 historical data..."):
                        nifty_df = get_historical_data_cached(api_key, access_token, "NIFTY 50", datetime.now().date() - timedelta(days=180), datetime.now().date(), "day", DEFAULT_EXCHANGE)
                        if isinstance(nifty_df, pd.DataFrame) and "_error" not in nifty_df.columns:
                            st.session_state["historical_data_NIFTY"] = nifty_df
                            st.success("NIFTY 50 historical data loaded.")
                        else:
                            st.error(f"Error fetching NIFTY 50 historical: {nifty_df.get('_error', 'Unknown error')}")
                else:
                    st.warning("Kite not authenticated to fetch historical data.")

        if not st.session_state.get("historical_data_NIFTY", pd.DataFrame()).empty:
            nifty_df = st.session_state["historical_data_NIFTY"]
            fig_nifty = go.Figure(data=[go.Candlestick(x=nifty_df.index, open=nifty_df['open'], high=nifty_df['high'], low=nifty_df['low'], close=nifty_df['close'], name='NIFTY 50')])
            fig_nifty.update_layout(title_text="NIFTY 50 Last 6 Months", xaxis_rangeslider_visible=False, height=300, template="plotly_white")
            st.plotly_chart(fig_nifty, use_container_width=True)

    with col3:
        st.subheader("Quick Performance")
        if st.session_state.get("last_fetched_symbol") and not st.session_state.get("historical_data", pd.DataFrame()).empty:
            last_symbol = st.session_state["last_fetched_symbol"]
            returns = st.session_state["historical_data"]["close"].pct_change().dropna() * 100
            if not returns.empty:
                perf = calculate_performance_metrics(returns)
                st.write(f"**{last_symbol}** (Last Fetched)")
                st.metric("Total Return", f"{perf.get('Total Return (%)', 0):.2f}%")
                st.metric("Annualized Volatility", f"{perf.get('Annualized Volatility (%)', 0):.2f}%")
                st.metric("Sharpe Ratio", f"{perf.get('Sharpe Ratio', 0):.2f}")
            else:
                st.info("No sufficient historical data for quick performance calculation.")
        else:
            st.info("Fetch some historical data in 'Market & Historical' tab to see quick performance here.")

def render_portfolio_tab(kite_client: KiteConnect | None):
    st.header("Portfolio (Future Enhancements)")
    st.info("This tab is being re-designed to incorporate more advanced portfolio analytics, scenario modeling, and optimization features. For now, basic holdings and positions can be fetched via the sidebar.")
    if not kite_client:
        st.warning("Login to Kite Connect to enable future portfolio features.")
        return
    
    # Placeholder for future portfolio features (e.g., portfolio analytics, risk breakdown)
    st.markdown("---")
    st.subheader("Current Kite Portfolio Summary (Basic)")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Fetch Current Holdings", key="portfolio_fetch_holdings_tab_btn"):
            try:
                holdings = kite_client.holdings()
                st.session_state["holdings_data"] = pd.DataFrame(holdings)
                st.success(f"Fetched {len(holdings)} holdings.")
            except Exception as e:
                st.error(f"Error fetching holdings: {e}")
        if not st.session_state.get("holdings_data", pd.DataFrame()).empty:
            st.dataframe(st.session_state["holdings_data"], use_container_width=True)
    with col2:
        if st.button("Fetch Current Positions", key="portfolio_fetch_positions_tab_btn"):
            try:
                positions = kite_client.positions()
                st.session_state["net_positions"] = pd.DataFrame(positions.get("net", []))
                st.success(f"Fetched {len(positions.get('net', []))} net positions.")
            except Exception as e:
                st.error(f"Error fetching positions: {e}")
        if not st.session_state.get("net_positions", pd.DataFrame()).empty:
            st.dataframe(st.session_state["net_positions"], use_container_width=True)


def render_orders_tab(kite_client: KiteConnect | None):
    st.header("Orders — Place, Modify, Cancel & View")
    if not kite_client:
        st.info("Login first to use orders API.")
        return

    st.subheader("Place New Order")
    with st.form("place_order_form", clear_on_submit=False):
        col_order1, col_order2 = st.columns(2)
        with col_order1:
            variety = st.selectbox("Variety", ["regular", "amo", "co", "iceberg"], key="order_variety")
            exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO", "CDS", "MCX"], key="order_exchange")
            tradingsymbol = st.text_input("Tradingsymbol", value="INFY", key="order_tradingsymbol")
            transaction_type = st.radio("Transaction Type", ["BUY", "SELL"], horizontal=True, key="order_transaction_type")
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="order_quantity")
        with col_order2:
            order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "SL", "SL-M"], key="order_type")
            product = st.selectbox("Product Type", ["CNC", "MIS", "NRML", "CO", "MTF"], key="order_product")
            price = st.text_input("Price (for LIMIT/SL)", value="", key="order_price")
            trigger_price = st.text_input("Trigger Price (for SL/SL-M)", value="", key="order_trigger_price")
            validity = st.selectbox("Validity", ["DAY", "IOC", "TTL"], key="order_validity")
            tag = st.text_input("Tag (optional, max 20 chars)", value="", key="order_tag")
        
        submit_place = st.form_submit_button("Place Order", key="submit_place_order")
        if submit_place:
            try:
                params = dict(variety=variety, exchange=exchange, tradingsymbol=tradingsymbol,
                              transaction_type=transaction_type, order_type=order_type,
                              quantity=int(quantity), product=product, validity=validity)
                if price: params["price"] = float(price)
                if trigger_price: params["trigger_price"] = float(trigger_price)
                if tag: params["tag"] = tag[:20]

                with st.spinner("Placing order..."):
                    resp = kite_client.place_order(**params) # Direct call
                    st.success(f"Order placed! ID: {resp.get('order_id')}")
            except Exception as e:
                st.error(f"Failed to place order: {e}")

    st.markdown("---")
    st.subheader("Manage Existing Orders & Trades")
    col_view_orders, col_manage_single = st.columns(2)

    with col_view_orders:
        if st.button("Fetch All Orders (Today)", key="fetch_all_orders_btn"):
            try:
                orders = kite_client.orders() # Direct call
                st.session_state["all_orders"] = pd.DataFrame(orders)
                st.success(f"Fetched {len(orders)} orders.")
            except Exception as e: st.error(f"Error fetching orders: {e}")
        if not st.session_state.get("all_orders", pd.DataFrame()).empty:
            with st.expander("Show Orders"): st.dataframe(st.session_state["all_orders"], use_container_width=True)

        if st.button("Fetch All Trades (Today)", key="fetch_all_trades_btn"):
            try:
                trades = kite_client.trades() # Direct call
                st.session_state["all_trades"] = pd.DataFrame(trades)
                st.success(f"Fetched {len(trades)} trades.")
            except Exception as e: st.error(f"Error fetching trades: {e}")
        if not st.session_state.get("all_trades", pd.DataFrame()).empty:
            with st.expander("Show Trades"): st.dataframe(st.session_state["all_trades"], use_container_width=True)

    with col_manage_single:
        order_id_action = st.text_input("Order ID for action", key="order_id_action")
        if st.button("Get Order History", key="get_order_history_btn"):
            if order_id_action:
                try: st.json(kite_client.order_history(order_id_action)) # Direct call
                except Exception as e: st.error(f"Failed to get order history: {e}")
            else: st.warning("Provide an Order ID.")
        
        with st.form("modify_order_form"):
            mod_variety = st.selectbox("Variety (for Modify)", ["regular", "amo", "co", "iceberg"], key="mod_variety_selector")
            mod_new_price = st.text_input("New Price (optional)", key="mod_new_price")
            mod_new_qty = st.number_input("New Quantity (optional)", min_value=0, value=0, step=1, key="mod_new_qty")
            submit_modify = st.form_submit_button("Modify Order", key="submit_modify_order")
            if submit_modify:
                if order_id_action:
                    try:
                        modify_args = {}
                        if mod_new_price: modify_args["price"] = float(mod_new_price)
                        if mod_new_qty > 0: modify_args["quantity"] = int(mod_new_qty)
                        if not modify_args: st.warning("No new price or quantity.")
                        else: st.json(kite_client.modify_order(variety=mod_variety, order_id=order_id_action, **modify_args)) # Direct call
                    except Exception as e: st.error(f"Failed to modify order: {e}")
                else: st.warning("Provide an Order ID to modify.")
        
        if st.button("Cancel Order", key="cancel_order_btn"):
            if order_id_action:
                try: st.json(kite_client.cancel_order(variety="regular", order_id=order_id_action)) # Direct call
                except Exception as e: st.error(f"Failed to cancel order: {e}")
            else: st.warning("Provide an Order ID to cancel.")

def render_market_historical_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Market Data & Historical Candles")
    if not kite_client:
        st.info("Login first to fetch market data.")
        return
    if not api_key or not access_token: # Additional check for cached functions
        st.info("Kite authentication details required for cached data access.")
        return

    st.subheader("Current Market Data Snapshot")
    col_market_quote1, col_market_quote2 = st.columns([1, 2])
    with col_market_quote1:
        q_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="market_exchange_tab")
        q_symbol = st.text_input("Tradingsymbol", value="NIFTY 50", key="market_symbol_tab") # Default to NIFTY 50 for quick demo
        if st.button("Get Market Data", key="get_market_data_btn"):
            ltp_data = get_ltp_price_cached(api_key, access_token, q_symbol, q_exchange) # Use cached LTP
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
    st.subheader("Historical Price Data")
    with st.expander("Load Instruments for Symbol Lookup (Recommended)"):
        exchange_for_lookup = st.selectbox("Exchange to load instruments", ["NSE", "BSE", "NFO"], key="hist_inst_load_exchange_selector")
        if st.button("Load Instruments into Cache", key="load_inst_cache_btn"):
            df_instruments = load_instruments_cached(api_key, access_token, exchange_for_lookup) # Use cached instruments
            if not df_instruments.empty and "_error" not in df_instruments.columns:
                st.session_state["instruments_df"] = df_instruments
                st.success(f"Loaded {len(df_instruments)} instruments.")
            else:
                st.error(f"Failed to load instruments: {df_instruments.get('_error', 'Unknown error')}")


    col_hist_controls, col_hist_plot = st.columns([1, 2])
    with col_hist_controls:
        hist_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="hist_ex_tab_selector")
        hist_symbol = st.text_input("Tradingsymbol", value="NIFTY 50", key="hist_sym_tab_input") # Default to NIFTY 50 for quick demo
        from_date = st.date_input("From Date", value=datetime.now().date() - timedelta(days=90), key="from_dt_tab_input")
        to_date = st.date_input("To Date", value=datetime.now().date(), key="to_dt_tab_input")
        interval = st.selectbox("Interval", ["minute", "5minute", "30minute", "day", "week", "month"], index=3, key="hist_interval_selector")

        if st.button("Fetch Historical Data", key="fetch_historical_data_btn"):
            with st.spinner(f"Fetching {interval} historical data for {hist_symbol}..."):
                df_hist = get_historical_data_cached(api_key, access_token, hist_symbol, from_date, to_date, interval, hist_exchange) # Use cached historical
                if isinstance(df_hist, pd.DataFrame) and "_error" not in df_hist.columns:
                    st.session_state["historical_data"] = df_hist
                    st.session_state["last_fetched_symbol"] = hist_symbol
                    st.success(f"Fetched {len(df_hist)} records for {hist_symbol}.")
                else:
                    st.error(f"Historical fetch failed: {df_hist.get('_error', 'Unknown error')}")

    with col_hist_plot:
        if not st.session_state.get("historical_data", pd.DataFrame()).empty:
            df = st.session_state["historical_data"]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='blue'), row=2, col=1)
            fig.update_layout(title_text=f"Historical Price & Volume for {st.session_state['last_fetched_symbol']}", xaxis_rangeslider_visible=False, height=600, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Historical chart will appear here.")

def render_ml_analysis_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Machine Learning Driven Price Analysis")
    if not kite_client:
        st.info("Login first to perform ML analysis.")
        return

    historical_data = st.session_state.get("historical_data")
    last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

    if historical_data.empty:
        st.warning("No historical data. Fetch from 'Market & Historical' first.")
        return

    st.subheader(f"1. Feature Engineering: Technical Indicators for {last_symbol}")
    col_indicator_params, col_indicator_data = st.columns([1,2])
    with col_indicator_params:
        sma_short_window = st.slider("SMA Short Window", 5, 50, 10, key="ml_sma_short_window")
        sma_long_window = st.slider("SMA Long Window", 20, 200, 50, key="ml_sma_long_window")
        rsi_window = st.slider("RSI Window", 7, 30, 14, key="ml_rsi_window")
        macd_fast = st.slider("MACD Fast Period", 5, 20, 12, key="ml_macd_fast")
        macd_slow = st.slider("MACD Slow Period", 20, 40, 26, key="ml_macd_slow")
        macd_signal = st.slider("MACD Signal Period", 5, 15, 9, key="ml_macd_signal")
        bb_window = st.slider("Bollinger Bands Window", 10, 50, 20, key="ml_bb_window")
        bb_std_dev = st.slider("Bollinger Bands Std Dev", 1.0, 3.0, 2.0, step=0.1, key="ml_bb_std_dev")
        
        if st.button("Apply Indicators", key="apply_indicators_btn"):
            df_with_indicators = add_technical_indicators(historical_data, sma_short_window, sma_long_window, 
                                                    rsi_window, macd_fast, macd_slow, macd_signal, 
                                                    bb_window, bb_std_dev)
            if not df_with_indicators.empty:
                st.session_state["ml_data"] = df_with_indicators
                st.success("Technical indicators applied.")
            else:
                st.error("Failed to add indicators. Data might be too short or invalid.")
                st.session_state["ml_data"] = pd.DataFrame()
    
    with col_indicator_data:
        ml_data = st.session_state.get("ml_data", pd.DataFrame())
        if not ml_data.empty:
            st.markdown("##### Data with Indicators (Head)")
            st.dataframe(ml_data.head(), use_container_width=True)
            # Plot indicators (abbreviated for brevity, full plot logic from original)
            fig_indicators = go.Figure(data=[
                go.Candlestick(x=ml_data.index, open=ml_data['open'], high=ml_data['high'], low=ml_data['low'], close=ml_data['close'], name='Price'),
                go.Scatter(x=ml_data.index, y=ml_data['SMA_Short'], mode='lines', name='SMA Short'),
                go.Scatter(x=ml_data.index, y=ml_data['SMA_Long'], mode='lines', name='SMA Long')
            ])
            fig_indicators.update_layout(title=f"Price with SMAs for {last_symbol}", height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_indicators, use_container_width=True)
        else:
            st.info("Apply indicators to see data and basic plots.")

    if not ml_data.empty:
        st.subheader(f"2. Machine Learning Model Training for {last_symbol}")
        col_ml_controls, col_ml_output = st.columns(2)
        with col_ml_controls:
            model_type_selected = st.selectbox("Select ML Model", ["Linear Regression", "Random Forest Regressor", "LightGBM Regressor"], key="ml_model_type_selector")
            ml_data_processed = ml_data.copy()
            ml_data_processed['target'] = ml_data_processed['close'].shift(-1)
            ml_data_processed.dropna(subset=['target'], inplace=True)
            
            features = [col for col in ml_data_processed.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target', 'MACD_hist']]
            selected_features = st.multiselect("Select Features for Model", options=features, default=features, key="ml_selected_features_multiselect")
            
            if not selected_features:
                st.warning("Please select at least one feature.")
                return

            X = ml_data_processed[selected_features]
            y = ml_data_processed['target']
            
            if X.empty or y.empty:
                st.error("Not enough clean data after preprocessing to train the model. Adjust parameters or fetch more data.")
                return

            test_size = st.slider("Test Set Size (%)", 10, 50, 20, step=5) / 100.0
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
            st.info(f"Training data: {len(X_train)} samples, Testing data: {len(X_test)} samples")

            if st.button(f"Train {model_type_selected} Model", key="train_ml_model_btn"):
                if len(X_train) == 0 or len(X_test) == 0:
                    st.error("Insufficient data for training/testing. Adjust test size or fetch more data.")
                    return
                model = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                    "LightGBM Regressor": lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                }.get(model_type_selected)

                if model:
                    with st.spinner(f"Training {model_type_selected} model..."):
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    st.session_state["ml_model"] = model
                    st.session_state["y_test"] = y_test
                    st.session_state["y_pred"] = y_pred
                    st.session_state["X_test_ml"] = X_test
                    st.session_state["ml_features"] = selected_features
                    st.session_state["ml_model_type"] = model_type_selected
                    st.success(f"{model_type_selected} Model Trained!")
        
        with col_ml_output:
            if st.session_state.get("ml_model") and st.session_state.get("y_test") is not None:
                mse = mean_squared_error(st.session_state['y_test'], st.session_state['y_pred'])
                r2 = r2_score(st.session_state['y_test'], st.session_state['y_pred'])
                st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
                st.metric("R2 Score", f"{r2:.4f}")

                pred_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': st.session_state['y_pred']}, index=st.session_state['y_test'].index)
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Actual'], mode='lines', name='Actual Price'))
                fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted'], mode='lines', name='Predicted Price', line=dict(dash='dot')))
                fig_pred.update_layout(title_text=f"Actual vs. Predicted Prices for {last_symbol}", height=500, template="plotly_white")
                st.plotly_chart(fig_pred, use_container_width=True)

        st.subheader(f"3. Real-time Price Prediction (Simulated for {last_symbol})")
        if st.session_state.get("ml_model") and not st.session_state.get("X_test_ml", pd.DataFrame()).empty:
            model = st.session_state["ml_model"]
            latest_features_df = st.session_state["X_test_ml"].iloc[[-1]][st.session_state["ml_features"]]
            if st.button("Simulate Next Period Prediction", key="simulate_prediction_btn"):
                simulated_prediction = model.predict(latest_features_df)[0]
                st.success(f"Simulated **next period** close price prediction: **₹{simulated_prediction:.2f}**")
        else:
            st.info("Train a machine learning model first for simulation.")
        
        st.markdown("---")
        st.subheader("4. Basic Backtesting: SMA Crossover Strategy")
        if not ml_data.empty:
            df_backtest = ml_data.copy()
            short_ma = st.slider("Short MA Window", 5, 50, 10, key="bt_short_ma")
            long_ma = st.slider("Long MA Window", 20, 200, 50, key="bt_long_ma")

            if st.button("Run Backtest", key="run_backtest_btn"):
                if len(df_backtest) < max(short_ma, long_ma):
                    st.error("Not enough data for MA windows.")
                    return
                df_backtest['SMA_Short_BT'] = ta.trend.sma_indicator(df_backtest['close'], window=short_ma)
                df_backtest['SMA_Long_BT'] = ta.trend.sma_indicator(df_backtest['close'], window=long_ma)
                df_backtest['Signal'] = (df_backtest['SMA_Short_BT'] > df_backtest['SMA_Long_BT']).astype(float)
                df_backtest['Position'] = df_backtest['Signal'].diff()
                df_backtest['Strategy_Return'] = df_backtest['Daily_Return'] * df_backtest['Signal'].shift(1)
                df_backtest['Cumulative_Strategy_Return'] = (1 + df_backtest['Strategy_Return'] / 100).cumprod() - 1
                df_backtest['Cumulative_Buy_Hold_Return'] = (1 + df_backtest['Daily_Return'] / 100).cumprod() - 1

                col_bt_metrics, col_bt_chart = st.columns(2)
                with col_bt_metrics:
                    strategy_metrics = calculate_performance_metrics(df_backtest['Strategy_Return'].dropna())
                    buy_hold_metrics = calculate_performance_metrics(df_backtest['Daily_Return'].dropna())
                    st.write("**Strategy Metrics**")
                    for k_m, v_m in strategy_metrics.items(): st.metric(k_m, f"{v_m:.2f}%" if "%" in k_m else f"{v_m:.2f}")
                    st.write("**Buy & Hold Metrics**")
                    for k_m, v_m in buy_hold_metrics.items(): st.metric(k_m, f"{v_m:.2f}%" if "%" in k_m else f"{v_m:.2f}")

                with col_bt_chart:
                    fig_backtest = go.Figure()
                    fig_backtest.add_trace(go.Scatter(x=df_backtest.index, y=df_backtest['Cumulative_Strategy_Return'] * 100, name='Strategy Return'))
                    fig_backtest.add_trace(go.Scatter(x=df_backtest.index, y=df_backtest['Cumulative_Buy_Hold_Return'] * 100, name='Buy & Hold Return', line=dict(dash='dash')))
                    fig_backtest.update_layout(title_text=f"SMA Crossover Strategy vs. Buy & Hold for {last_symbol}", height=450)
                    st.plotly_chart(fig_backtest, use_container_width=True)
        else:
            st.info("Apply technical indicators first to enable backtesting.")


def render_risk_stress_testing_tab(kite_client: KiteConnect | None):
    st.header("Risk & Stress Testing Models")
    if not kite_client:
        st.info("Login first to perform risk analysis.")
        return

    historical_data = st.session_state.get("historical_data")
    last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

    if historical_data.empty:
        st.warning("No historical data. Fetch from 'Market & Historical' first.")
        return
    
    historical_data['close'] = pd.to_numeric(historical_data['close'], errors='coerce')
    daily_returns = historical_data['close'].pct_change().dropna() * 100
    if daily_returns.empty or len(daily_returns) < 2:
        st.error("Not enough valid data for risk analysis.")
        return

    st.subheader(f"1. Volatility & Returns Analysis for {last_symbol}")
    col_vol_metrics, col_vol_dist = st.columns([1,2])
    with col_vol_metrics:
        st.dataframe(daily_returns.describe().to_frame().T, use_container_width=True)
        annualized_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        st.metric("Annualized Volatility", f"{annualized_volatility:.2f}%")
        st.metric("Mean Daily Return", f"{daily_returns.mean():.2f}%")

        rolling_window = st.slider("Rolling Volatility Window (days)", 10, 252, 30, key="risk_rolling_vol_window")
        if len(daily_returns) > rolling_window:
            rolling_vol = daily_returns.rolling(window=rolling_window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            fig_rolling_vol = go.Figure(go.Scatter(x=rolling_vol.index, y=rolling_vol, name='Rolling Volatility'))
            fig_rolling_vol.update_layout(title_text=f"Rolling {rolling_window}-Day Annualized Volatility", height=300)
            st.plotly_chart(fig_rolling_vol, use_container_width=True)
    with col_vol_dist:
        fig_volatility = go.Figure(go.Histogram(x=daily_returns, nbinsx=50, name='Daily Returns'))
        fig_volatility.update_layout(title_text=f'Distribution of Daily Returns for {last_symbol}', height=500)
        st.plotly_chart(fig_volatility, use_container_width=True)

    st.subheader(f"2. Value at Risk (VaR) Calculation for {last_symbol}")
    col_var_controls, col_var_plot = st.columns([1,2])
    with col_var_controls:
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 95, step=1, key="risk_confidence_level")
        holding_period_var = st.number_input("Holding Period for VaR (days)", min_value=1, value=1, step=1, key="risk_holding_period_var")
        var_percentile_1day = np.percentile(daily_returns, 100 - confidence_level)
        var_percentile_multiday = var_percentile_1day * np.sqrt(holding_period_var)
        st.write(f"With **{confidence_level}% confidence**, max loss over **{holding_period_var} day(s)**:")
        st.metric(f"VaR ({confidence_level}%)", f"{abs(var_percentile_multiday):.2f}%")
        current_price = historical_data['close'].iloc[-1]
        st.metric(f"Potential Loss (₹{current_price:.2f})", f"₹{(abs(var_percentile_multiday) / 100) * current_price:,.2f}")
    with col_var_plot:
        fig_var = go.Figure(go.Histogram(x=daily_returns, nbinsx=50, name='Daily Returns'))
        fig_var.add_vline(x=var_percentile_1day, line_dash="dash", line_color="red", annotation_text=f"1-Day VaR {confidence_level}%: {var_percentile_1day:.2f}%")
        fig_var.update_layout(title_text=f'Daily Returns Distribution with {confidence_level}% VaR for {last_symbol}', height=400)
        st.plotly_chart(fig_var, use_container_width=True)

    st.subheader(f"3. Stress Testing (Scenario Analysis) for {last_symbol}")
    col_stress_controls, col_stress_results = st.columns([1,2])
    with col_stress_controls:
        scenarios = {
            "Historical Worst Day Drop": {"type": "historical", "percent": daily_returns.min() if not daily_returns.empty else 0},
            "Global Financial Crisis (-20%)": {"type": "fixed", "percent": -20.0},
            "Custom % Change": {"type": "custom", "percent": 0.0}
        }
        scenario_key = st.selectbox("Select Stress Scenario", list(scenarios.keys()), key="risk_scenario_selector")
        custom_change_percent = st.number_input("Custom Percentage Change (%)", value=0.0, step=0.1, key="risk_custom_change_input") if scenario_key == "Custom % Change" else 0.0
        
        if st.button("Run Stress Test", key="run_stress_test_btn"):
            current_price = historical_data['close'].iloc[-1]
            scenario_data = scenarios[scenario_key]
            scenario_change_percent = scenario_data["percent"] if scenario_data["type"] != "custom" else custom_change_percent
            stressed_price = current_price * (1 + scenario_change_percent / 100)
            st.session_state["stress_test_results"] = {"scenario_key": scenario_key, "current_price": current_price, "stressed_price": stressed_price, "scenario_change_percent": scenario_change_percent}
            st.success("Stress test executed.")
    with col_stress_results:
        if st.session_state.get("stress_test_results"):
            results = st.session_state["stress_test_results"]
            st.markdown(f"##### Results for Scenario: **{results['scenario_key']}**")
            st.metric("Current Price", f"₹{results['current_price']:.2f}")
            st.metric("Stressed Price", f"₹{results['stressed_price']:.2f}")
            st.metric("Percentage Change", f"{results['scenario_change_percent']:.2f}%")

def render_performance_analysis_tab(kite_client: KiteConnect | None):
    st.header("Performance Analysis")
    if not kite_client:
        st.info("Login first to analyze performance.")
        return

    historical_data = st.session_state.get("historical_data")
    last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

    if historical_data.empty:
        st.warning("No historical data. Fetch from 'Market & Historical' first.")
        return
    
    historical_data['close'] = pd.to_numeric(historical_data['close'], errors='coerce')
    returns_series = historical_data['close'].pct_change().dropna() * 100
    if returns_series.empty or len(returns_series) < 2:
        st.error("Not enough valid data for performance metrics.")
        return

    st.subheader(f"Performance Metrics for {last_symbol}")
    col_metrics, col_chart = st.columns([1,2])
    with col_metrics:
        risk_free_rate = st.number_input("Risk-Free Rate (Annualized %)", 0.0, 10.0, 4.0, step=0.1, key="perf_risk_free_rate")
        performance_metrics = calculate_performance_metrics(returns_series, risk_free_rate)
        for k_m, v_m in performance_metrics.items(): st.metric(k_m, f"{v_m:.2f}%" if "%" in k_m else f"{v_m:.2f}")
    
    with col_chart:
        fig_cum_returns = go.Figure()
        cumulative_instrument_returns = (1 + returns_series / 100).cumprod() - 1
        fig_cum_returns.add_trace(go.Scatter(x=cumulative_instrument_returns.index, y=cumulative_instrument_returns * 100, name=f'{last_symbol} Returns'))
        
        st.subheader("Benchmark Comparison (e.g., NIFTY 50)")
        benchmark_symbol = st.text_input("Benchmark Symbol (e.g., NIFTY 50 from Zerodha)", "NIFTY 50", key="perf_benchmark_symbol")
        if st.button("Fetch & Compare Benchmark", key="fetch_compare_benchmark_btn"):
            with st.spinner(f"Fetching {benchmark_symbol} data..."):
                try:
                    # Fetch from Zerodha API via our cached function
                    benchmark_data_df = get_historical_data_cached(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"], benchmark_symbol, historical_data.index.min().date(), historical_data.index.max().date(), "day", DEFAULT_EXCHANGE)
                    
                    if isinstance(benchmark_data_df, pd.DataFrame) and "_error" not in benchmark_data_df.columns and not benchmark_data_df.empty:
                        benchmark_returns = benchmark_data_df['close'].pct_change().dropna() * 100
                        # Align dates
                        common_dates = returns_series.index.intersection(benchmark_returns.index)
                        if not common_dates.empty and len(common_dates) > 1:
                            returns_series_aligned = returns_series.loc[common_dates]
                            benchmark_returns_aligned = benchmark_returns.loc[common_dates]
                            cumulative_benchmark_returns = (1 + benchmark_returns_aligned / 100).cumprod() - 1
                            fig_cum_returns.add_trace(go.Scatter(x=cumulative_benchmark_returns.index, y=cumulative_benchmark_returns * 100, name=f'{benchmark_symbol} Returns', line=dict(dash='dash')))
                            
                            df_for_alpha_beta = pd.DataFrame({'Asset': returns_series_aligned, 'Benchmark': benchmark_returns_aligned}).dropna()
                            if len(df_for_alpha_beta) > 1:
                                covariance = df_for_alpha_beta['Asset'].cov(df_for_alpha_beta['Benchmark'])
                                benchmark_variance = df_for_alpha_beta['Benchmark'].var()
                                beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
                                alpha_annual = (performance_metrics['Annualized Return (%)'] - risk_free_rate - beta * (calculate_performance_metrics(benchmark_returns_aligned)['Annualized Return (%)'] - risk_free_rate)) if not np.isnan(beta) else np.nan
                                st.markdown("##### Alpha and Beta")
                                st.metric("Beta", f"{beta:.2f}" if not np.isnan(beta) else "N/A")
                                st.metric("Alpha (Annualized %)", f"{alpha_annual:.2f}%" if not np.isnan(alpha_annual) else "N/A")
                            else: st.warning("Not enough common data for Alpha/Beta.")
                        else: st.warning("No common historical data points with benchmark.")
                    else: st.warning(f"Could not fetch benchmark data for {benchmark_symbol}: {benchmark_data_df.get('_error', 'Unknown error')}")
                except Exception as e: st.error(f"Error fetching benchmark data: {e}")
        fig_cum_returns.update_layout(title_text=f"Cumulative Returns: {last_symbol} vs. Benchmark", height=500)
        st.plotly_chart(fig_cum_returns, use_container_width=True)

def render_multi_asset_analysis_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Multi-Asset Analysis: Correlation & Diversification")
    if not kite_client:
        st.info("Login first to perform multi-asset analysis.")
        return
    if not api_key or not access_token:
        st.info("Kite authentication details required for cached data access.")
        return

    st.subheader("Select Instruments for Analysis")
    selected_symbols_str = st.text_area("Enter Trading Symbols (comma-separated, e.g., INFY,RELIANCE)", "INFY,RELIANCE", height=80, key="multi_asset_symbols_input")
    symbols_to_analyze = [s.strip().upper() for s in selected_symbols_str.split(',') if s.strip()]
    
    multi_asset_exchange = st.selectbox("Exchange for all symbols", ["NSE", "BSE", "NFO"], key="multi_asset_exchange_selector")
    multi_asset_interval = st.selectbox("Interval for historical data", ["day", "week", "month"], key="multi_asset_interval_selector")
    from_date_multi = st.date_input("From Date", value=datetime.now().date() - timedelta(days=365), key="from_dt_multi_input")
    to_date_multi = st.date_input("To Date", value=datetime.now().date(), key="to_dt_multi_input")

    if st.button("Fetch Multi-Asset Data & Analyze", key="fetch_multi_asset_btn"):
        # Load instruments once for all lookups in this tab
        df_instruments_load = load_instruments_cached(api_key, access_token, multi_asset_exchange)
        if "_error" in df_instruments_load.columns:
            st.error(f"Failed to load instruments for lookup: {df_instruments_load.get('_error', 'Unknown error')}")
            return
        st.session_state["instruments_df"] = df_instruments_load


        all_historical_data = {}
        progress_bar = st.progress(0)
        for i, symbol in enumerate(symbols_to_analyze):
            with st.spinner(f"Fetching historical data for {symbol}..."):
                df = get_historical_data_cached(api_key, access_token, symbol, from_date_multi, to_date_multi, multi_asset_interval, multi_asset_exchange)
                if not df.empty and "_error" not in df.columns:
                    all_historical_data[symbol] = df['close']
                else:
                    st.warning(f"No historical data for {symbol} or error: {df.get('_error', 'Unknown error')}. Skipping.")
            progress_bar.progress((i + 1) / len(symbols_to_analyze))
        
        if len(all_historical_data) < 2:
            st.error("Select at least two instruments with data for correlation analysis.")
            return

        combined_df = pd.DataFrame(all_historical_data)
        combined_df.dropna(inplace=True)

        if combined_df.empty or len(combined_df) < 2:
            st.error("No common historical data. Adjust date range or symbols.")
            return

        returns_df = combined_df.pct_change().dropna()
        st.session_state["multi_asset_returns"] = returns_df
        st.session_state["multi_asset_correlation"] = returns_df.corr()
        st.success("Multi-asset data fetched and correlations calculated.")
        st.dataframe(combined_df.head(), use_container_width=True)

    if st.session_state.get("multi_asset_correlation") is not None:
        st.subheader("Correlation Matrix (Daily Returns)")
        correlation_matrix = st.session_state["multi_asset_correlation"]
        st.dataframe(correlation_matrix.style.background_gradient(cmap='RdBu', axis=None).format(precision=2), use_container_width=True)
        fig_corr_heatmap = go.Figure(data=go.Heatmap(z=correlation_matrix.values, x=correlation_matrix.columns, y=correlation_matrix.index, colorscale='RdBu', zmin=-1, zmax=1))
        fig_corr_heatmap.update_layout(title_text='Correlation Heatmap', height=600)
        st.plotly_chart(fig_corr_heatmap, use_container_width=True)

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

    # Helper function to render an index's details, charts, and export options
    def display_index_details(index_name: str, constituents_df: pd.DataFrame, index_history_df: pd.DataFrame, index_id: str | None = None, is_recalculated_live=False):
        st.markdown(f"### Details for Index: **{index_name}** {'(Recalculated Live)' if is_recalculated_live else ''}")
        
        st.subheader("Constituents and Current Live Value")
        
        live_quotes = {}
        symbols_for_ltp = [sym for sym in constituents_df["symbol"]]
        
        # Load instruments for symbol to token lookup if not already loaded
        if st.session_state["instruments_df"].empty:
            with st.spinner("Loading instruments for live price lookup..."):
                st.session_state["instruments_df"] = load_instruments_cached(api_key, access_token, DEFAULT_EXCHANGE)
        
        if "_error" in st.session_state["instruments_df"].columns:
            st.warning(f"Could not load instruments for live price lookup: {st.session_state['instruments_df'].loc[0, '_error']}")
        else:
            if symbols_for_ltp:
                try:
                    kc_client = get_authenticated_kite_client(api_key, access_token)
                    if kc_client:
                        instrument_identifiers = [f"{DEFAULT_EXCHANGE}:{s}" for s in symbols_for_ltp]
                        ltp_data_batch = kc_client.ltp(instrument_identifiers)
                        
                        for sym in symbols_for_ltp:
                            key = f"{DEFAULT_EXCHANGE}:{sym}"
                            if key in ltp_data_batch:
                                live_quotes[sym] = ltp_data_batch[key].get("last_price", np.nan)
                            else:
                                live_quotes[sym] = np.nan
                    else:
                        st.warning("Kite client not available for batch LTP fetch (internal error).")
                except Exception as e:
                    st.error(f"Error fetching batch LTP: {e}. Live prices might be partial.")

        # Ensure 'Name' column exists for display
        if 'Name' not in constituents_df.columns:
            if not st.session_state["instruments_df"].empty:
                instrument_names = st.session_state["instruments_df"].set_index('tradingsymbol')['name'].to_dict()
                constituents_df['Name'] = constituents_df['symbol'].map(instrument_names).fillna(constituents_df['symbol'])
            else:
                constituents_df['Name'] = constituents_df['symbol'] # Fallback if no names found

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
        st.subheader("Index Historical Performance and Benchmarking")

        if index_history_df.empty or "_error" in index_history_df.columns:
            st.warning(f"Historical performance data for '{index_name}' is not available or could not be calculated: {index_history_df.get('_error', ['Unknown Error'])[0]}")
            return # Cannot proceed with charting if no historical data

        fig_index_perf = go.Figure()
        fig_index_perf.add_trace(go.Scatter(x=index_history_df.index, y=index_history_df['index_value'], mode='lines', name=f'Custom Index ({index_name})', line=dict(color='blue', width=2)))
        
        # Performance Metrics for Custom Index
        st.markdown("##### Custom Index Performance Metrics")
        index_returns = index_history_df['index_value'].pct_change().dropna() * 100
        risk_free_rate_index = st.number_input("Risk-Free Rate (Annualized %) for Index", 0.0, 10.0, 4.0, step=0.1, key=f"index_rf_rate_{index_id or index_name}")
        index_perf_metrics = calculate_performance_metrics(index_returns, risk_free_rate_index)
        
        col_idx_perf1, col_idx_perf2, col_idx_perf3 = st.columns(3)
        with col_idx_perf1:
            st.metric("Total Return (%)", f"{index_perf_metrics.get('Total Return (%)', 0):.2f}%")
            st.metric("Annualized Return (%)", f"{index_perf_metrics.get('Annualized Return (%)', 0):.2f}%")
        with col_idx_perf2:
            st.metric("Annualized Volatility (%)", f"{index_perf_metrics.get('Annualized Volatility (%)', 0):.2f}%")
            st.metric("Sharpe Ratio", f"{index_perf_metrics.get('Sharpe Ratio', 0):.2f}")
        with col_idx_perf3:
            st.metric("Max Drawdown (%)", f"{index_perf_metrics.get('Max Drawdown (%)', 0):.2f}%")
            st.metric("Sortino Ratio", f"{index_perf_metrics.get('Sortino Ratio', 0):.2f}")


        st.markdown("---")
        st.markdown("##### Benchmark Comparison and Advanced Metrics")
        benchmark_symbols_str = st.text_input("Enter Benchmark Symbols (comma-separated, e.g., NIFTY 50,BANKNIFTY)", value="NIFTY 50", key=f"benchmark_symbols_{index_id or index_name}")
        benchmark_symbols = [s.strip().upper() for s in benchmark_symbols_str.split(',') if s.strip()]
        benchmark_exchange = st.selectbox("Benchmark Exchange", ["NSE", "BSE", "NFO"], key=f"bench_exchange_{index_id or index_name}")

        benchmark_metrics = {}
        
        # Store benchmark data to plot persistently until symbols change
        if f"active_benchmarks_data_{index_id or index_name}" not in st.session_state:
            st.session_state[f"active_benchmarks_data_{index_id or index_name}"] = {}

        if st.button("Analyze & Add Benchmarks to Chart", key=f"add_benchmarks_{index_id or index_name}"):
            # Clear previous benchmark data if new symbols are entered
            st.session_state[f"active_benchmarks_data_{index_id or index_name}"] = {}

            if "_error" in st.session_state["instruments_df"].columns:
                st.error(f"Failed to load instruments for benchmark lookup: {st.session_state['instruments_df'].loc[0, '_error']}")
                return

            for bench_symbol in benchmark_symbols:
                with st.spinner(f"Fetching historical data for benchmark {bench_symbol}..."):
                    bench_hist_df = get_historical_data_cached(api_key, access_token, bench_symbol, index_history_df.index.min().date(), index_history_df.index.max().date(), "day", benchmark_exchange)
                
                if isinstance(bench_hist_df, pd.DataFrame) and "_error" not in bench_hist_df.columns and not bench_hist_df.empty and 'close' in bench_hist_df.columns:
                    st.session_state[f"active_benchmarks_data_{index_id or index_name}"][bench_symbol] = bench_hist_df
                else:
                    error_msg = bench_hist_df.get('_error', ['Unknown error'])[0] if isinstance(bench_hist_df, pd.DataFrame) else 'Unknown error'
                    st.warning(f"No historical data obtained for benchmark {bench_symbol}. Skipping. Error: {error_msg}")
        
        # Plot and calculate metrics for all active benchmarks
        for bench_symbol, bench_hist_df in st.session_state[f"active_benchmarks_data_{index_id or index_name}"].items():
            bench_hist_df_renamed = bench_hist_df[['close']].rename(columns={'close': f'{bench_symbol}_close_bench'})
            
            aligned_df = pd.merge(index_history_df[['index_value']], bench_hist_df_renamed, left_index=True, right_index=True, how='inner')
            
            if not aligned_df.empty:
                benchmark_col_name = f'{bench_symbol}_close_bench'
                if benchmark_col_name in aligned_df.columns:
                    base_index_val = aligned_df['index_value'].iloc[0]
                    base_bench_val = aligned_df[benchmark_col_name].iloc[0]
                    if base_bench_val != 0:
                        aligned_df[f'{bench_symbol}_normalized'] = (aligned_df[benchmark_col_name] / base_bench_val) * base_index_val
                        fig_index_perf.add_trace(go.Scatter(x=aligned_df.index, y=aligned_df[f'{bench_symbol}_normalized'], mode='lines', name=f'Benchmark: {bench_symbol}', line=dict(dash='dash')))
                        
                        # Calculate Benchmark-specific metrics
                        bench_returns_aligned = aligned_df[benchmark_col_name].pct_change().dropna() * 100
                        benchmark_perf = calculate_performance_metrics(bench_returns_aligned, risk_free_rate_index)
                        benchmark_metrics[bench_symbol] = benchmark_perf

                        # Tracking Error, Information Ratio, Beta
                        if not index_returns.empty and not bench_returns_aligned.empty:
                            common_returns_df = pd.DataFrame({'Index': index_returns, 'Benchmark': bench_returns_aligned}).dropna()
                            if len(common_returns_df) > 1:
                                tracking_error = np.std(common_returns_df['Index'] - common_returns_df['Benchmark']) * np.sqrt(TRADING_DAYS_PER_YEAR)
                                
                                active_return = index_perf_metrics.get('Annualized Return (%)', 0) - benchmark_perf.get('Annualized Return (%)', 0)
                                information_ratio = active_return / tracking_error if tracking_error != 0 else np.nan

                                # Beta calculation
                                covariance = common_returns_df['Index'].cov(common_returns_df['Benchmark'])
                                benchmark_variance = common_returns_df['Benchmark'].var()
                                beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan

                                st.markdown(f"###### Metrics vs. {bench_symbol}")
                                col_bm1, col_bm2, col_bm3 = st.columns(3)
                                with col_bm1:
                                    st.metric("Benchmark Return (%)", f"{benchmark_perf.get('Annualized Return (%)', 0):.2f}%")
                                    st.metric("Benchmark Volatility (%)", f"{benchmark_perf.get('Annualized Volatility (%)', 0):.2f}%")
                                with col_bm2:
                                    st.metric("Tracking Error (%)", f"{tracking_error:.2f}%")
                                    st.metric("Information Ratio", f"{information_ratio:.2f}")
                                with col_bm3:
                                    st.metric("Beta vs. Benchmark", f"{beta:.2f}")
                            else:
                                st.warning(f"Not enough common return data for {bench_symbol} to calculate advanced metrics.")
                        else:
                            st.warning(f"Insufficient returns data for {bench_symbol} for advanced metrics.")

                    else:
                        st.warning(f"First historical value of {bench_symbol} is zero, cannot normalize. Skipping benchmark.")
                else:
                    st.warning(f"Benchmark '{bench_symbol}' data missing expected column '{benchmark_col_name}' after merge. This can happen if merge failed or data is insufficient. Skipping benchmark.")
            else:
                st.warning(f"No common historical data between custom index and benchmark {bench_symbol}. Skipping benchmark.")
        
        fig_index_perf.update_layout(title_text=f"Historical Performance: {index_name} vs. Benchmarks",
                                  xaxis_title="Date", yaxis_title="Index Value (Normalized to 100 on Start Date)",
                                  height=500, template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_index_perf, use_container_width=True)

        st.markdown("---")
        st.subheader("Export Options")
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            csv_constituents = constituents_df[['symbol', 'Name', 'Weights']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export Constituents to CSV",
                data=csv_constituents,
                file_name=f"{index_name}_constituents.csv",
                mime="text/csv",
                key=f"export_constituents_{index_id or index_name}"
            )
        with col_export2:
            csv_history = index_history_df.to_csv().encode('utf-8')
            st.download_button(
                label="Export Historical Performance to CSV",
                data=csv_history,
                file_name=f"{index_name}_historical_performance.csv",
                mime="text/csv",
                key=f"export_history_{index_id or index_name}"
            )

    # --- Section: Index Creation ---
    st.markdown("---")
    st.subheader("1. Create New Index from CSV")
    st.markdown("Upload a CSV file containing your desired index constituents. The CSV should have columns: `symbol`, `Name`, `Weights`.")
    st.markdown("Example CSV format:")
    st.code("symbol,Name,Weights\nINFY,Infosys,0.3\nRELIANCE,Reliance Industries,0.5\nHDFCBANK,HDFC Bank,0.2")

    uploaded_file = st.file_uploader("Upload CSV with index constituents", type=["csv"], key="index_upload_csv")
    
    if uploaded_file:
        try:
            df_constituents_new = pd.read_csv(uploaded_file)
            required_cols = {"symbol", "Name", "Weights"}
            if not required_cols.issubset(set(df_constituents_new.columns)):
                st.error(f"CSV must contain columns: `symbol`, `Name`, `Weights`. Missing: {required_cols - set(df_constituents_new.columns)}")
                return

            df_constituents_new["Weights"] = pd.to_numeric(df_constituents_new["Weights"], errors='coerce')
            df_constituents_new.dropna(subset=["Weights"], inplace=True)
            
            if df_constituents_new.empty:
                st.error("No valid constituents found in the CSV after processing weights. Ensure 'Weights' column contains numeric values.")
                return

            total_weights = df_constituents_new["Weights"].sum()
            if total_weights <= 0:
                st.error("Sum of weights must be positive. Please provide valid positive weights.")
                return
            df_constituents_new["Weights"] = df_constituents_new["Weights"] / total_weights # Normalize weights
            st.info(f"Loaded {len(df_constituents_new)} constituents. Weights have been normalized to sum to 1.")

            st.subheader("Configure Historical Calculation for New Index")
            hist_start_date = st.date_input("Historical Start Date (for new index)", value=datetime.now().date() - timedelta(days=365), key="new_index_hist_start_date")
            hist_end_date = st.date_input("Historical End Date (for new index)", value=datetime.now().date(), key="new_index_hist_end_date")

            if hist_start_date >= hist_end_date:
                st.error("Historical start date must be before end date.")
                return

            with st.spinner("Calculating historical index values... This may take some time depending on constituents and date range."):
                index_history_df_new = _calculate_historical_index_value(api_key, access_token, df_constituents_new, hist_start_date, hist_end_date, DEFAULT_EXCHANGE)
            
            if not index_history_df_new.empty and "_error" not in index_history_df_new.columns:
                st.session_state["current_calculated_index_data"] = df_constituents_new
                st.session_state["current_calculated_index_history"] = index_history_df_new
                st.success("Historical index values calculated successfully.")

                # Display details for the newly calculated index immediately
                display_index_details("Newly Calculated Index", df_constituents_new, index_history_df_new)
            else:
                st.error(f"Failed to calculate historical index values for new index: {index_history_df_new.get('_error', 'Unknown error')}")
                st.session_state["current_calculated_index_data"] = None
                st.session_state["current_calculated_index_history"] = pd.DataFrame()


            if st.session_state["current_calculated_index_data"] is not None and not st.session_state["current_calculated_index_history"].empty:
                st.markdown("---")
                st.subheader("Save Newly Created Index to Database")
                index_name_to_save = st.text_input("Enter a unique name for this new index to save", key="new_index_save_name")
                if st.button("Save New Index to DB", key="save_new_index_to_db_btn"):
                    if index_name_to_save and st.session_state["user_id"]:
                        # Check if an index with this name already exists for the user
                        try:
                            check_response = supabase_client.table("custom_indexes").select("id").eq("user_id", st.session_state["user_id"]).eq("index_name", index_name_to_save).execute()
                            if check_response.data:
                                st.warning(f"An index named '{index_name_to_save}' already exists. Please choose a different name.")
                                return

                            index_data = {
                                "user_id": st.session_state["user_id"],
                                "index_name": index_name_to_save,
                                "constituents": st.session_state["current_calculated_index_data"][['symbol', 'Name', 'Weights']].to_dict(orient='records'),
                                "historical_performance": st.session_state["current_calculated_index_history"].reset_index().to_dict(orient='records') # Store history
                            }
                            response = supabase_client.table("custom_indexes").insert(index_data).execute()
                            if response.data:
                                st.success(f"Index '{index_name_to_save}' saved successfully to Supabase!")
                                st.session_state["saved_indexes"] = [] # Clear to force reload
                                st.session_state["current_calculated_index_data"] = None # Clear new index data after saving
                                st.session_state["current_calculated_index_history"] = pd.DataFrame()
                                st.rerun()
                            else:
                                st.error(f"Failed to save index: {response.data}")
                        except Exception as e:
                            st.error(f"Error saving new index: {e}")
                    else:
                        st.warning("Please enter an index name and ensure you are logged into Supabase.")

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")
        except Exception as e:
            st.error(f"An unexpected error occurred while processing the file or calculating the index: {e}.")
    
    st.markdown("---")
    st.subheader("2. Load & Manage Saved Indexes")
    if st.button("Load My Indexes from DB", key="load_my_indexes_db_btn"):
        try:
            response = supabase_client.table("custom_indexes").select("id, index_name, constituents, historical_performance").eq("user_id", st.session_state["user_id"]).execute()
            if response.data:
                st.session_state["saved_indexes"] = response.data
                st.success(f"Loaded {len(response.data)} indexes.")
            else:
                st.session_state["saved_indexes"] = []
                st.info("No saved indexes found for your account. Create one above and save it!")
        except Exception as e: st.error(f"Error loading indexes: {e}")
    
    if st.session_state.get("saved_indexes"):
        index_names_from_db = [idx['index_name'] for idx in st.session_state["saved_indexes"]]
        selected_index_name_from_db = st.selectbox("Select a saved index to display and analyze:", ["--- Select ---"] + index_names_from_db, key="select_saved_index_from_db")

        if selected_index_name_from_db != "--- Select ---":
            selected_db_index_data = next((idx for idx in st.session_state["saved_indexes"] if idx['index_name'] == selected_index_name_from_db), None)
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
                        # Basic validation: check if there's enough data and 'index_value' column
                        if loaded_historical_df.empty or 'index_value' not in loaded_historical_df.columns or len(loaded_historical_df) < 2:
                            raise ValueError("Loaded historical data is insufficient or invalid.")
                    except (KeyError, ValueError, TypeError) as e:
                        st.warning(f"Problem with saved historical data for '{selected_index_name_from_db}': {e}. Attempting to recalculate live...")
                        loaded_historical_df = pd.DataFrame() # Clear invalid data
                
                # If historical data isn't saved or is empty/invalid, re-calculate it live
                if loaded_historical_df.empty or "_error" in loaded_historical_df.columns:
                    # Determine date range for recalculation based on current defaults
                    # Default to 1 year for recalculation if no historical data is saved.
                    min_date = (datetime.now().date() - timedelta(days=365))
                    max_date = datetime.now().date()
                    
                    with st.spinner("Recalculating historical index values (live)..."):
                        recalculated_historical_df = _calculate_historical_index_value(api_key, access_token, loaded_constituents_df, min_date, max_date, DEFAULT_EXCHANGE)
                    
                    if not recalculated_historical_df.empty and "_error" not in recalculated_historical_df.columns:
                        loaded_historical_df = recalculated_historical_df
                        is_recalculated_live = True
                        st.success("Historical data recalculated live successfully.")
                    else:
                        st.error(f"Failed to recalculate historical data: {recalculated_historical_df.get('_error', 'Unknown error')}")
                        loaded_historical_df = pd.DataFrame({"_error": ["Failed to recalculate historical data."]})


                display_index_details(selected_index_name_from_db, loaded_constituents_df, loaded_historical_df, selected_db_index_data['id'], is_recalculated_live)
                
                st.markdown("---")
                if st.button(f"Delete Index '{selected_index_name_from_db}' from DB", key=f"delete_index_{selected_db_index_data['id']}"):
                    try:
                        response = supabase_client.table("custom_indexes").delete().eq("id", selected_db_index_data['id']).execute()
                        if response.data:
                            st.success(f"Index '{selected_index_name_from_db}' deleted successfully.")
                            st.session_state["saved_indexes"] = []
                            st.rerun()
                        else: st.error(f"Failed to delete index: {response.data}")
                    except Exception as e: st.error(f"Error deleting index: {e}")
            else: st.error("Selected index data not found in session state.")
    else: st.info("No indexes loaded yet. Click 'Load My Indexes from DB' or 'Create New Index from CSV'.")


# --- Main Application Logic (Tab Rendering) ---
# Global api_key and access_token to pass to tab functions that use cached utility functions.
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

with tab_dashboard: render_dashboard_tab(k, api_key, access_token)
with tab_portfolio: render_portfolio_tab(k)
with tab_orders: render_orders_tab(k)
with tab_market: render_market_historical_tab(k, api_key, access_token)
with tab_ml: render_ml_analysis_tab(k, api_key, access_token)
with tab_risk: render_risk_stress_testing_tab(k)
with tab_performance: render_performance_analysis_tab(k)
with tab_multi_asset: render_multi_asset_analysis_tab(k, api_key, access_token)
with tab_custom_index: render_custom_index_tab(k, supabase, api_key, access_token)
