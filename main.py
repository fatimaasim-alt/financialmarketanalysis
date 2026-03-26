import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# --- PAGE SETUP ---
st.set_page_config(page_title="Financial Market Analysis Dashboard", layout="wide")

st.title("Financial Market Analysis Dashboard")
st.markdown("---")

# --- SIDEBAR OPTIONS ---
stocks = ("AAPL", "GOOG", "MSFT", "TSLA", "NVDA")
selected_stock = st.sidebar.selectbox("Select dataset", stocks)
n_days = st.sidebar.slider("Days of prediction:", 30, 365, 90)

# We use a long start date to ensure we have enough data for Moving Averages (50 days min)
START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# --- DATA LOAD (ROBUST VERSION) ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_data(ticker):
    try:
        # download with multi_level_index=False to avoid column name issues
        data = yf.download(ticker, start=START, end=TODAY, multi_level_index=False)
        
        if data.empty:
            return None
        
        data.reset_index(inplace=True)
        
        # Ensure column names are clean strings (fix for some yfinance versions)
        data.columns = [str(col) for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

data_load_state = st.info(f"Fetching data for {selected_stock}...")
df_raw = load_data(selected_stock)

if df_raw is None or len(df_raw) < 50:
    st.error("Could not fetch enough data. Yahoo Finance might be temporarily throttling requests. Please try again in a few minutes or select a different stock.")
    st.stop()
else:
    data_load_state.empty()

# --- TECHNICAL INDICATORS ---
def calculate_indicators(df_in):
    df = df_in.copy()
    # Ensure we use 'Close' column
    close_col = 'Close'
    
    delta = df[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MA20'] = df[close_col].rolling(window=20).mean()
    df['MA50'] = df[close_col].rolling(window=50).mean()
    return df

df = calculate_indicators(df_raw)
df.dropna(inplace=True)

# --- TIME FILTER ---
st.sidebar.subheader("Historical View")
range_option = st.sidebar.selectbox("Quick Select", ("All", "1 Month", "6 Months", "1 Year", "5 Years"))

df_display = df.copy()
if range_option == "1 Month": df_display = df.tail(30)
elif range_option == "6 Months": df_display = df.tail(180)
elif range_option == "1 Year": df_display = df.tail(365)
elif range_option == "5 Years": df_display = df.tail(1825)

# --- KPI METRICS ---
if len(df_display) > 1:
    curr_price = float(df_display['Close'].iloc[-1])
    prev_price = float(df_display['Close'].iloc[-2])
    delta_price = ((curr_price - prev_price) / prev_price) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${curr_price:,.2f}", f"{delta_price:.2f}%")
    c2.metric("RSI (14D)", f"{df_display['RSI'].iloc[-1]:.1f}")
    c3.metric("50D MA", f"${df_display['MA50'].iloc[-1]:,.2f}")
    c4.metric("Volume", f"{df_display['Volume'].iloc[-1]/1e6:.1f}M")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Technical Analysis", "🤖 Smart Trend Forecast", "📄 Raw Data"])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df_display['Date'], open=df_display['Open'], high=df_display['High'], 
                                 low=df_display['Low'], close=df_display['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['MA20'], name="20d MA", line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # PREDICTION LOGIC
    df_fc = df[['Date', 'Close']].copy()
    df_fc['Day_Index'] = np.arange(len(df_fc))
    
    X = df_fc[['Day_Index']].values
    y = df_fc['Close'].values
    
    # Polynomial features to allow for curves (Degree 2)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Generate future dates
    last_date = df_fc['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, n_days + 1)]
    future_indices = np.arange(len(df_fc), len(df_fc) + n_days).reshape(-1, 1)
    future_poly = poly.transform(future_indices)
    preds = model.predict(future_poly)
    
    # Chart
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=df_fc['Date'].tail(150), y=df_fc['Close'].tail(150), name="Actual Price"))
    fig_fc.add_trace(go.Scatter(x=future_dates, y=preds, name="Predicted Trend", line=dict(dash='dash', color='red')))
    fig_fc.update_layout(title="Smart Price Trend Prediction", template="plotly_dark")
    st.plotly_chart(fig_fc, use_container_width=True)

with tab3:
    st.dataframe(df.sort_values(by="Date", ascending=False), use_container_width=True)