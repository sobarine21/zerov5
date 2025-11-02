import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import json
import threading 
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
from sklearn.preprocessing import MinMaxScaler 
import pickle
import io

# Supabase imports
from supabase import create_client, Client

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Invsion Connect - Algorithmic Trading Platform", layout="wide", initial_sidebar_state="expanded")
st.title("Invsion Connect - Algorithmic Trading & Price Prediction Platform with RL")
st.markdown("A focused platform with **Reinforcement Learning** that learns from actual market data and adapts predictions.")

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

# --- ML specific initializations ---
if "ml_data" not in st.session_state:
    st.session_state["ml_data"] = pd.DataFrame()
if "ml_model" not in st.session_state:
    st.session_state["ml_model"] = None
if "prediction_horizon" not in st.session_state:
    st.session_state["prediction_horizon"] = 5

# --- RL specific initializations ---
if "rl_memory" not in st.session_state:
    st.session_state["rl_memory"] = []  # Store (features, predicted, actual, error)
if "rl_adaptive_weights" not in st.session_state:
    st.session_state["rl_adaptive_weights"] = None
if "rl_correction_model" not in st.session_state:
    st.session_state["rl_correction_model"] = None
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []  # Track predictions over time
if "rl_learning_rate" not in st.session_state:
    st.session_state["rl_learning_rate"] = 0.01
if "rl_enabled" not in st.session_state:
    st.session_state["rl_enabled"] = True
if "uploaded_real_data" not in st.session_state:
    st.session_state["uploaded_real_data"] = pd.DataFrame()
if "rl_training_complete" not in st.session_state:
    st.session_state["rl_training_complete"] = False

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
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame({"_error": ["Kite not authenticated to fetch historical data."]})

    instruments_df = load_instruments_cached(api_key, access_token, exchange)
    if "_error" in instruments_df.columns:
        return pd.DataFrame({"_error": [instruments_df.loc[0, '_error']]}) 

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


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'close' not in df.columns:
        return pd.DataFrame()

    df_copy = df.copy() 
    
    # Technical Indicators
    df_copy['SMA_10'] = ta.trend.sma_indicator(df_copy['close'], window=10)
    df_copy['RSI_14'] = ta.momentum.rsi(df_copy['close'], window=14)
    macd_obj = ta.trend.MACD(df_copy['close'], window_fast=12, window_slow=26, window_sign=9)
    df_copy['MACD'] = macd_obj.macd()
    df_copy['Bollinger_High'] = ta.volatility.BollingerBands(df_copy['close'], window=20, window_dev=2).bollinger_hband()
    df_copy['Bollinger_Low'] = ta.volatility.BollingerBands(df_copy['close'], window=20, window_dev=2).bollinger_lband()
    df_copy['VWAP'] = ta.volume.volume_weighted_average_price(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], window=14)
    df_copy['OBV'] = ta.volume.on_balance_volume(df_copy['close'], df_copy['volume'])
    df_copy['ATR'] = ta.volatility.average_true_range(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
    
    # Lagged features
    for lag in [1, 2, 5]:
        df_copy[f'Lag_Close_{lag}'] = df_copy['close'].shift(lag)
        df_copy[f'Lag_Return_{lag}'] = df_copy['close'].pct_change(lag) * 100

    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(method='ffill', inplace=True)
    df_copy.dropna(inplace=True) 
    return df_copy


# --- CSV PARSING FUNCTION ---
def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Parse uploaded CSV file with flexible column mapping"""
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Create column mapping (case-insensitive)
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'date' in col_lower:
                col_map[col] = 'date'
            elif 'open' in col_lower:
                col_map[col] = 'open'
            elif 'high' in col_lower:
                col_map[col] = 'high'
            elif 'low' in col_lower:
                col_map[col] = 'low'
            elif 'close' in col_lower:
                col_map[col] = 'close'
            elif 'volume' in col_lower:
                col_map[col] = 'volume'
        
        # Rename columns
        df.rename(columns=col_map, inplace=True)
        
        # Check required columns
        required_cols = ['date', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Parse date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Set date as index
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close']
        if 'volume' in df.columns:
            numeric_cols.append('volume')
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing values
        df.dropna(subset=['close'], inplace=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
        return pd.DataFrame()


# --- REINFORCEMENT LEARNING FUNCTIONS ---

class AdaptivePredictionCorrector:
    """
    RL-based correction model that learns from prediction errors.
    Uses gradient descent to adjust predictions based on historical errors.
    """
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = np.zeros(n_features + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.error_history = []
        self.n_updates = 0
        
    def predict_correction(self, features, base_prediction):
        """Predict correction factor to apply to base model prediction"""
        X = np.append(features, 1)  # Add bias term
        correction = np.dot(self.weights, X)
        return base_prediction + correction
    
    def update(self, features, predicted_price, actual_price):
        """Update weights based on prediction error using gradient descent"""
        X = np.append(features, 1)
        error = actual_price - predicted_price
        self.error_history.append(abs(error))
        
        # Gradient descent update
        gradient = error * X
        self.weights += self.learning_rate * gradient
        self.n_updates += 1
        
        return error


def store_prediction_feedback(features, predicted, actual, symbol, timestamp):
    """Store prediction and actual outcome for learning"""
    error = actual - predicted
    error_pct = (error / predicted) * 100 if predicted != 0 else 0
    
    feedback_entry = {
        'timestamp': timestamp,
        'symbol': symbol,
        'features': features,
        'predicted': predicted,
        'actual': actual,
        'error': error,
        'error_pct': error_pct
    }
    
    st.session_state["rl_memory"].append(feedback_entry)
    
    # Keep only last 1000 entries to prevent memory overflow
    if len(st.session_state["rl_memory"]) > 1000:
        st.session_state["rl_memory"] = st.session_state["rl_memory"][-1000:]
    
    return feedback_entry


def train_rl_on_real_data(model, scaler, features_list, real_data_df, horizon, symbol):
    """
    Train RL model on real historical data by making predictions and learning from actual outcomes.
    This simulates forward-testing where we predict and then see the actual result.
    """
    if real_data_df.empty or model is None:
        return 0
    
    # Add features to real data
    real_data_with_features = add_advanced_features(real_data_df)
    
    if real_data_with_features.empty:
        return 0
    
    predictions_made = 0
    
    # Initialize RL model if not exists
    if st.session_state["rl_correction_model"] is None:
        n_features = len(features_list)
        st.session_state["rl_correction_model"] = AdaptivePredictionCorrector(
            n_features, 
            learning_rate=st.session_state["rl_learning_rate"]
        )
    
    corrector = st.session_state["rl_correction_model"]
    
    # Iterate through the data, making predictions and learning
    for i in range(len(real_data_with_features) - horizon):
        try:
            # Get current features
            current_features = real_data_with_features[features_list].iloc[[i]]
            
            # Scale features
            current_features_scaled = scaler.transform(current_features)
            
            # Make base prediction
            base_prediction = model.predict(current_features_scaled)[0]
            
            # Apply RL correction if available
            if corrector.n_updates > 0:
                corrected_prediction = corrector.predict_correction(
                    current_features_scaled[0], 
                    base_prediction
                )
            else:
                corrected_prediction = base_prediction
            
            # Get actual price after horizon periods
            actual_price = real_data_with_features['close'].iloc[i + horizon]
            
            # Store feedback
            store_prediction_feedback(
                features=current_features_scaled[0],
                predicted=corrected_prediction,
                actual=actual_price,
                symbol=symbol,
                timestamp=real_data_with_features.index[i]
            )
            
            # Update RL model with this error
            corrector.update(
                current_features_scaled[0],
                corrected_prediction,
                actual_price
            )
            
            predictions_made += 1
            
        except Exception as e:
            continue
    
    return predictions_made


def apply_rl_correction(base_prediction, features):
    """Apply RL correction to base model prediction"""
    if not st.session_state["rl_enabled"]:
        return base_prediction
    
    corrector = st.session_state["rl_correction_model"]
    if corrector is None or corrector.n_updates == 0:
        return base_prediction
    
    corrected_prediction = corrector.predict_correction(features, base_prediction)
    return corrected_prediction


def calculate_rl_metrics():
    """Calculate RL performance metrics"""
    memory = st.session_state["rl_memory"]
    
    if len(memory) < 5:
        return None
    
    recent_errors = [abs(entry['error_pct']) for entry in memory[-20:]]
    older_errors = [abs(entry['error_pct']) for entry in memory[:20]] if len(memory) > 20 else recent_errors
    
    metrics = {
        'total_predictions': len(memory),
        'avg_recent_error': np.mean(recent_errors),
        'avg_older_error': np.mean(older_errors),
        'improvement': np.mean(older_errors) - np.mean(recent_errors),
        'best_prediction_error': min([abs(e['error_pct']) for e in memory]),
        'worst_prediction_error': max([abs(e['error_pct']) for e in memory])
    }
    
    return metrics


# --- Sidebar: Authentication ---
with st.sidebar:
    st.markdown("### 1. Login to Kite Connect")
    st.write("Click to open Kite login. You'll be redirected back with a `request_token`.")
    st.markdown(f"[🔗 Open Kite login]({login_url})")

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
        st.success("Kite Authenticated ✅")
        if st.sidebar.button("Logout from Kite", key="kite_logout_btn"):
            st.session_state["kite_access_token"] = None
            st.session_state["kite_login_response"] = None
            st.session_state["instruments_df"] = pd.DataFrame() 
            st.success("Logged out from Kite. Please login again.")
            st.rerun()
    else:
        st.info("Not authenticated with Kite yet.")

    st.markdown("---")
    st.markdown("### 2. Reinforcement Learning Settings")
    
    st.session_state["rl_enabled"] = st.checkbox("Enable RL Adaptive Learning", value=st.session_state["rl_enabled"])
    st.session_state["rl_learning_rate"] = st.slider("RL Learning Rate", 0.001, 0.1, st.session_state["rl_learning_rate"], 0.001)
    
    if st.button("Reset RL Memory", key="reset_rl_btn"):
        st.session_state["rl_memory"] = []
        st.session_state["rl_correction_model"] = None
        st.session_state["rl_training_complete"] = False
        st.success("RL memory reset!")
        st.rerun()
    
    # Display RL stats
    rl_metrics = calculate_rl_metrics()
    if rl_metrics:
        st.markdown("##### RL Performance")
        st.metric("Total Predictions Tracked", rl_metrics['total_predictions'])
        st.metric("Recent Avg Error %", f"{rl_metrics['avg_recent_error']:.2f}%")
        improvement_val = rl_metrics['improvement']
        st.metric("Learning Improvement", f"{improvement_val:.2f}%", 
                 delta=f"{improvement_val:.2f}%" if improvement_val > 0 else None)
        
        if st.session_state["rl_training_complete"]:
            st.success("✅ RL Training Complete on Real Data")

    st.markdown("---")
    st.markdown("### 3. Supabase User Account (Optional)")
    
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
        st.success(f"Supabase Logged in: {st.session_state['user_session'].user.email}")
        if st.button("Logout from Supabase", key="supabase_logout_btn"):
            try:
                supabase.auth.sign_out()
                _refresh_supabase_session() 
                st.sidebar.success("Logged out from Supabase.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error logging out: {e}")
    else:
        st.info("Supabase login section hidden for brevity.")


k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])


# --- Tab Logic Functions ---

def render_market_historical_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("1. Market Data & Historical Data")
    if not kite_client:
        st.info("Login first to fetch market data.")
        return
    if not api_key or not access_token: 
        st.info("Kite authentication details required for data access.")
        return

    st.subheader("Current Market Data Snapshot (LTP)")
    col_market_quote1, col_market_quote2 = st.columns([1, 2])
    with col_market_quote1:
        q_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="market_exchange_tab")
        q_symbol = st.text_input("Tradingsymbol", value="RELIANCE", key="market_symbol_tab") 
        if st.button("Get Latest Price", key="get_market_data_btn"):
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
    st.subheader("Historical Price Data")
    
    col_hist_controls, col_hist_plot = st.columns([1, 2])
    with col_hist_controls:
        hist_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="hist_ex_tab_selector")
        hist_symbol = st.text_input("Tradingsymbol", value="INFY", key="hist_sym_tab_input") 
        from_date = st.date_input("From Date", value=datetime.now().date() - timedelta(days=365), key="from_dt_tab_input")
        to_date = st.date_input("To Date", value=datetime.now().date() - timedelta(days=30), key="to_dt_tab_input")
        
        st.info("💡 Set 'To Date' to 1 month before today to leave room for RL training with real data")
        
        interval = st.selectbox("Interval", ["day", "minute", "5minute", "30minute", "week", "month"], index=0, key="hist_interval_selector")

        if st.button("Fetch Historical Data", key="fetch_historical_data_btn"):
            if st.session_state["instruments_df"].empty:
                st.info("Loading instruments first...")
                df_instruments = load_instruments_cached(api_key, access_token, hist_exchange)
                if not df_instruments.empty and "_error" not in df_instruments.columns:
                    st.session_state["instruments_df"] = df_instruments
                else:
                    st.error(f"Failed to load instruments: {df_instruments.get('_error', 'Unknown error')}")
                    return

            with st.spinner(f"Fetching {interval} historical data for {hist_symbol}..."):
                df_hist = get_historical_data_cached(api_key, access_token, hist_symbol, from_date, to_date, interval, hist_exchange) 
                if isinstance(df_hist, pd.DataFrame) and "_error" not in df_hist.columns:
                    st.session_state["historical_data"] = df_hist
                    st.session_state["last_fetched_symbol"] = hist_symbol
                    st.session_state["ml_data"] = pd.DataFrame()
                    st.session_state["ml_model"] = None
                    st.session_state["rl_training_complete"] = False
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
            st.info("Historical chart will appear here. Please fetch data first.")


def render_price_predictor_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("2. Price Predictor with Reinforcement Learning")
    if not kite_client:
        st.info("Login first
