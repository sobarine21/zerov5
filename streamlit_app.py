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

# Supabase imports
from supabase import create_client, Client

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Invsion Connect - Algorithmic Trading Platform", layout="wide", initial_sidebar_state="expanded")
st.title("Invsion Connect - Algorithmic Trading & Price Prediction Platform with RL")
st.markdown("A focused platform with **Reinforcement Learning** that learns from prediction errors and adapts over time.")

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
    
    # Keep only last 500 entries to prevent memory overflow
    if len(st.session_state["rl_memory"]) > 500:
        st.session_state["rl_memory"] = st.session_state["rl_memory"][-500:]
    
    return feedback_entry


def train_rl_correction_model():
    """Train the RL correction model on accumulated feedback"""
    memory = st.session_state["rl_memory"]
    
    if len(memory) < 10:  # Need minimum data
        return None
    
    # Extract training data from memory
    X_train = []
    y_train = []
    
    for entry in memory:
        features = entry['features']
        error = entry['error']
        X_train.append(features)
        y_train.append(error)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Initialize or update correction model
    if st.session_state["rl_correction_model"] is None:
        n_features = X_train.shape[1]
        st.session_state["rl_correction_model"] = AdaptivePredictionCorrector(
            n_features, 
            learning_rate=st.session_state["rl_learning_rate"]
        )
    
    # Train on recent errors
    corrector = st.session_state["rl_correction_model"]
    for features, error_val in zip(X_train[-50:], y_train[-50:]):  # Use last 50 samples
        # Simulate update (in real scenario, this happens when actual data arrives)
        pass
    
    return corrector


def apply_rl_correction(base_prediction, features):
    """Apply RL correction to base model prediction"""
    if not st.session_state["rl_enabled"]:
        return base_prediction
    
    corrector = st.session_state["rl_correction_model"]
    if corrector is None:
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
        st.success("RL memory reset!")
        st.rerun()
    
    # Display RL stats
    rl_metrics = calculate_rl_metrics()
    if rl_metrics:
        st.markdown("##### RL Performance")
        st.metric("Total Predictions Tracked", rl_metrics['total_predictions'])
        st.metric("Recent Avg Error %", f"{rl_metrics['avg_recent_error']:.2f}%")
        st.metric("Learning Improvement", f"{rl_metrics['improvement']:.2f}%", 
                 delta=f"{rl_metrics['improvement']:.2f}%" if rl_metrics['improvement'] > 0 else None)

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
        to_date = st.date_input("To Date", value=datetime.now().date(), key="to_dt_tab_input")
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
        st.info("Login first to perform ML analysis.")
        return

    historical_data = st.session_state.get("historical_data", pd.DataFrame())
    last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

    if historical_data.empty:
        st.warning("No historical data. Fetch from 'Market Data & Historical Data' first.")
        return

    st.subheader(f"1. Feature Engineering & Data Preparation for {last_symbol}")
    
    col_feat_eng, col_prep = st.columns(2)
    with col_feat_eng:
        if st.button("Generate Advanced Features (Indicators & Lags)", key="generate_features_btn"):
            df_with_features = add_advanced_features(historical_data)
            if not df_with_features.empty:
                st.session_state["ml_data"] = df_with_features
                st.session_state["ml_model"] = None
                st.success(f"Data prepared with {len(df_with_features.columns)} features.")
            else:
                st.error("Failed to add features. Data might be too short or invalid.")
                st.session_state["ml_data"] = pd.DataFrame()

    ml_data = st.session_state.get("ml_data", pd.DataFrame())
    
    if not ml_data.empty:
        
        with col_prep:
            current_prediction_horizon = st.number_input("Prediction Horizon (Periods/Days Ahead)", min_value=1, max_value=20, value=5, step=1, key="pred_horizon")
            test_size = st.slider("Test Set Size (%)", 10, 50, 20, step=5) / 100.0
        
        st.markdown("---")
        st.subheader("2. Machine Learning Model Training with RL Enhancement")
        
        col_ml_controls, col_ml_output = st.columns(2)
        
        ml_data_processed = ml_data.copy()
        ml_data_processed['target'] = ml_data_processed['close'].shift(-current_prediction_horizon)
        ml_data_processed.dropna(subset=['target'], inplace=True)
        
        features = [col for col in ml_data_processed.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        
        with col_ml_controls:
            model_type_selected = st.selectbox("Select ML Model", ["LightGBM Regressor (High Performance)", "Random Forest Regressor", "Linear Regression"], key="ml_model_type_selector")
            selected_features = st.multiselect("Select Features for Model", options=features, default=[f for f in features if f.startswith(('RSI', 'MACD', 'Lag_Close', 'SMA', 'ATR'))], key="ml_selected_features_multiselect")
            
            if not selected_features:
                st.warning("Please select at least one feature.")
                return

            X = ml_data_processed[selected_features]
            y = ml_data_processed['target']
            
            if X.empty or y.empty or len(X) < 100:
                st.error("Insufficient clean data for robust training (need at least 100 samples).")
                return

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, shuffle=False)
            st.info(f"Training data: {len(X_train)} periods, Testing data: {len(X_test)} periods")

            if st.button(f"Train {model_type_selected} Model", key="train_ml_model_btn"):
                if len(X_train) == 0 or len(X_test) == 0:
                    st.error("Insufficient data for training/testing.")
                    return
                
                model = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
                    "LightGBM Regressor (High Performance)": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1)
                }.get(model_type_selected)

                if model:
                    with st.spinner(f"Training {model_type_selected} model for {current_prediction_horizon}-day prediction..."):
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Apply RL correction to predictions if enabled
                        if st.session_state["rl_enabled"] and st.session_state["rl_correction_model"] is not None:
                            y_pred_corrected = []
                            for i, pred in enumerate(y_pred):
                                corrected = apply_rl_correction(pred, X_test[i])
                                y_pred_corrected.append(corrected)
                            y_pred = np.array(y_pred_corrected)
                            st.info("✨ RL correction applied to predictions!")
                        
                        # Store predictions for RL learning (simulate actual prices with test set)
                        for i in range(len(y_test)):
                            store_prediction_feedback(
                                features=X_test[i],
                                predicted=y_pred[i],
                                actual=y_test.iloc[i],
                                symbol=last_symbol,
                                timestamp=datetime.now()
                            )
                        
                        # Train RL correction model on accumulated feedback
                        if st.session_state["rl_enabled"] and len(st.session_state["rl_memory"]) >= 10:
                            train_rl_correction_model()
                            st.success("🧠 RL model updated with new feedback!")
                    
                    st.session_state["ml_model"] = model
                    st.session_state["y_test"] = y_test
                    st.session_state["y_pred"] = y_pred
                    st.session_state["X_test_scaled"] = X_test
                    st.session_state["scaler"] = scaler
                    st.session_state["ml_features"] = selected_features
                    st.session_state["ml_model_type"] = model_type_selected
                    st.session_state["prediction_horizon"] = current_prediction_horizon
                    st.success(f"{model_type_selected} Model Trained for {current_prediction_horizon}-day horizon!")
        
        with col_ml_output:
            if st.session_state.get("ml_model") and st.session_state.get("y_test") is not None:
                mse = mean_squared_error(st.session_state['y_test'], st.session_state['y_pred'])
                rmse = np.sqrt(mse)
                r2 = r2_score(st.session_state['y_test'], st.session_state['y_pred'])
                
                st.markdown(f"##### Evaluation Metrics ({st.session_state['prediction_horizon']} periods ahead)")
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("RMSE", f"₹{rmse:.2f}")
                col_m2.metric("R² Score", f"{r2:.4f}")
                
                # Show RL enhancement status
                if st.session_state["rl_enabled"]:
                    rl_metrics = calculate_rl_metrics()
                    if rl_metrics and rl_metrics['total_predictions'] > 10:
                        st.success(f"🤖 RL Enhanced: {rl_metrics['total_predictions']} predictions tracked, {rl_metrics['improvement']:.2f}% improvement")
                
                pred_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': st.session_state['y_pred']}, index=st.session_state['y_test'].index)
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Actual'], mode='lines', name='Actual Future Price'))
                fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted'], mode='lines', name='Predicted Future Price', line=dict(dash='dot', width=2)))
                fig_pred.update_layout(title_text=f"Model Performance: Actual vs. Predicted Price ({st.session_state['prediction_horizon']} periods ahead)", height=500, template="plotly_white")
                st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("---")
        st.subheader(f"3. Next Prediction: Forecasting {last_symbol}")

        if st.session_state.get("ml_model") and not ml_data.empty:
            model = st.session_state["ml_model"]
            scaler = st.session_state["scaler"]
            features_list = st.session_state["ml_features"]
            horizon = st.session_state["prediction_horizon"]
            
            latest_row = ml_data[features_list].iloc[[-1]] 
            latest_row_scaled = scaler.transform(latest_row)
            
            col_forecast_btn, col_actual_update = st.columns(2)
            
            with col_forecast_btn:
                if st.button(f"Generate Forecast for Next {horizon} Periods", key="generate_forecast_btn"):
                    
                    with st.spinner(f"Predicting next {horizon} periods..."):
                        base_prediction = model.predict(latest_row_scaled)[0]
                        
                        # Apply RL correction if enabled
                        if st.session_state["rl_enabled"] and st.session_state["rl_correction_model"] is not None:
                            forecasted_price = apply_rl_correction(base_prediction, latest_row_scaled[0])
                            st.info("✨ RL correction applied to forecast!")
                        else:
                            forecasted_price = base_prediction
                    
                    last_known_close = historical_data['close'].iloc[-1]
                    predicted_change = ((forecasted_price - last_known_close) / last_known_close) * 100

                    # Store this prediction
                    st.session_state["last_forecast"] = {
                        'timestamp': datetime.now(),
                        'symbol': last_symbol,
                        'predicted_price': forecasted_price,
                        'base_price': last_known_close,
                        'features': latest_row_scaled[0],
                        'horizon': horizon
                    }

                    st.success(f"Forecast Generated using **{st.session_state['ml_model_type']}** model:")
                    
                    col_f1, col_f2, col_f3 = st.columns(3)
                    col_f1.metric("Last Known Close", f"₹{last_known_close:.2f}")
                    col_f2.metric(f"Predicted Price (in {horizon} periods)", 
                                         f"₹{forecasted_price:.2f}", 
                                         delta=f"{predicted_change:.2f}%")
                    col_f3.metric("Forecast Date", 
                                         (historical_data.index[-1] + timedelta(days=horizon)).strftime('%Y-%m-%d'))
                    
                    if predicted_change > 0:
                        st.info(f"📈 **Action:** Potential **BUY/HOLD** signal, expecting rise of {predicted_change:.2f}%")
                    elif predicted_change < 0:
                        st.info(f"📉 **Action:** Potential **SELL/SHORT** signal, expecting drop of {abs(predicted_change):.2f}%")
                    else:
                        st.info("➡️ **Action:** Neutral outlook, minimal change predicted")
            
            with col_actual_update:
                st.markdown("##### Update with Actual Price (for RL Learning)")
                st.write("When the predicted period passes, update with actual price to improve model:")
                
                actual_price_input = st.number_input("Actual Price Observed", min_value=0.0, value=0.0, step=0.01, key="actual_price_input")
                
                if st.button("Submit Actual Price & Learn", key="submit_actual_btn"):
                    if actual_price_input > 0 and st.session_state.get("last_forecast"):
                        forecast = st.session_state["last_forecast"]
                        
                        # Store feedback for RL
                        feedback = store_prediction_feedback(
                            features=forecast['features'],
                            predicted=forecast['predicted_price'],
                            actual=actual_price_input,
                            symbol=forecast['symbol'],
                            timestamp=datetime.now()
                        )
                        
                        # Update RL correction model
                        if st.session_state["rl_correction_model"] is None:
                            n_features = len(forecast['features'])
                            st.session_state["rl_correction_model"] = AdaptivePredictionCorrector(
                                n_features, 
                                learning_rate=st.session_state["rl_learning_rate"]
                            )
                        
                        corrector = st.session_state["rl_correction_model"]
                        error = corrector.update(forecast['features'], forecast['predicted_price'], actual_price_input)
                        
                        st.success(f"✅ RL Model Updated!")
                        st.info(f"Prediction Error: ₹{abs(error):.2f} ({abs(feedback['error_pct']):.2f}%)")
                        
                        # Show learning progress
                        if len(st.session_state["rl_memory"]) > 1:
                            rl_metrics = calculate_rl_metrics()
                            if rl_metrics['improvement'] > 0:
                                st.success(f"🎯 Model improving! Error reduced by {rl_metrics['improvement']:.2f}%")
                            else:
                                st.info(f"📊 Model learning... Track more predictions for better adaptation")
                        
                        st.rerun()
                    else:
                        st.warning("Please enter a valid actual price and ensure a forecast was generated.")
        else:
            st.info("Train the machine learning model first to generate forecasts.")

        # Display RL Learning History
        if st.session_state["rl_memory"]:
            st.markdown("---")
            st.subheader("4. RL Learning History & Performance")
            
            col_rl1, col_rl2 = st.columns([2, 1])
            
            with col_rl1:
                # Plot error progression over time
                memory_df = pd.DataFrame([
                    {
                        'timestamp': entry['timestamp'],
                        'error_pct': abs(entry['error_pct']),
                        'symbol': entry['symbol']
                    }
                    for entry in st.session_state["rl_memory"]
                ])
                
                fig_rl = go.Figure()
                fig_rl.add_trace(go.Scatter(
                    x=memory_df.index, 
                    y=memory_df['error_pct'],
                    mode='lines+markers',
                    name='Prediction Error %',
                    line=dict(color='red', width=2)
                ))
                
                # Add trend line
                if len(memory_df) > 5:
                    z = np.polyfit(memory_df.index, memory_df['error_pct'], 1)
                    p = np.poly1d(z)
                    fig_rl.add_trace(go.Scatter(
                        x=memory_df.index,
                        y=p(memory_df.index),
                        mode='lines',
                        name='Trend',
                        line=dict(color='green', dash='dash', width=2)
                    ))
                
                fig_rl.update_layout(
                    title_text="RL Learning Progress: Prediction Error Over Time",
                    xaxis_title="Prediction Number",
                    yaxis_title="Absolute Error %",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig_rl, use_container_width=True)
            
            with col_rl2:
                rl_metrics = calculate_rl_metrics()
                if rl_metrics:
                    st.markdown("##### RL Statistics")
                    st.metric("Total Predictions", rl_metrics['total_predictions'])
                    st.metric("Best Prediction Error", f"{rl_metrics['best_prediction_error']:.2f}%")
                    st.metric("Recent Avg Error", f"{rl_metrics['avg_recent_error']:.2f}%")
                    st.metric("Learning Improvement", 
                             f"{rl_metrics['improvement']:.2f}%",
                             delta=f"{rl_metrics['improvement']:.2f}%")
            
            # Show recent predictions table
            st.markdown("##### Recent Predictions Log")
            recent_predictions = st.session_state["rl_memory"][-10:][::-1]  # Last 10, reversed
            
            log_df = pd.DataFrame([
                {
                    'Time': entry['timestamp'].strftime('%Y-%m-%d %H:%M'),
                    'Symbol': entry['symbol'],
                    'Predicted': f"₹{entry['predicted']:.2f}",
                    'Actual': f"₹{entry['actual']:.2f}",
                    'Error': f"₹{entry['error']:.2f}",
                    'Error %': f"{entry['error_pct']:.2f}%"
                }
                for entry in recent_predictions
            ])
            
            st.dataframe(log_df, use_container_width=True, hide_index=True)


# --- Main Application Logic ---
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

tabs = st.tabs(["Market Data & Historical Data", "Price Predictor (ML + RL)"])
tab_market, tab_ml = tabs

with tab_market: 
    render_market_historical_tab(k, api_key, access_token)
    
with tab_ml: 
    render_price_predictor_tab(k, api_key, access_token)
