import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime, timedelta
import os
import time

# ==========================================
# --- 1. INSTITUTIONAL UI THEME INJECTION ---
# ==========================================
def inject_institutional_css():
    st.markdown("""
    <style>
        /* Main Backgrounds */
        .stApp { background-color: #0B0F19; color: #F8FAFC; }
        [data-testid="stSidebar"] { background-color: #0F172A; border-right: 1px solid #1E293B; }
        
        /* Metric Cards */
        div[data-testid="metric-container"] {
            background-color: #1E293B; border: 1px solid #334155;
            padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        div[data-testid="metric-container"] label { color: #94A3B8 !important; font-weight: 600 !important; letter-spacing: 0.5px; }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #38BDF8 !important; font-size: 1.8rem !important; font-weight: 700 !important; }
        
        /* The Awesome Spot / Apex Box */
        .apex-box {
            background-color: #082F49; border-left: 5px solid #38BDF8;
            border-radius: 5px; padding: 20px; margin-top: 15px; margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        .apex-title { color: #BAE6FD; font-size: 1.4em; font-weight: 800; margin-bottom: 10px; }
        .apex-action { color: #38BDF8; font-size: 1.2em; font-weight: 700; margin-bottom: 10px; }
        .apex-logic { color: #94A3B8; font-size: 1em; font-style: italic; }
        
        /* Headers and Dividers */
        h1, h2, h3 { color: #F1F5F9 !important; font-weight: 700 !important; }
        hr { border-color: #334155 !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# --- 2. GLOBAL INSTITUTIONAL UNIVERSE ---
# ==========================================
TICKER_SETS = {
    "🔥 Magnificent 7 + Crypto": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "COIN", "MSTR", "MARA", "BTC-USD"],
    "💻 Semiconductors (AI)": ["NVDA", "AMD", "TSM", "INTC", "MU", "AVGO", "QCOM", "ARM", "AMAT", "SMH"],
    "🏦 Financials & Banking": ["JPM", "BAC", "WFC", "MS", "GS", "C", "V", "MA", "AXP", "XLF"]
}

# ==========================================
# --- 3. SYNTHETIC DATA GENERATOR (GBM) ---
# ==========================================
class DummyDataGenerator:
    @staticmethod
    def generate_synthetic_ohlcv(ticker="SYNTH", days=252, S0=150, mu=0.15, sigma=0.4):
        """Generates Geometric Brownian Motion data for UI testing without API limits."""
        dt = 1/252
        prices = np.zeros(days)
        prices[0] = S0
        
        # Simulate path
        for t in range(1, days):
            Z = np.random.standard_normal()
            prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            
        dates = pd.date_range(end=datetime.today(), periods=days)
        df = pd.DataFrame({'Close': prices}, index=dates)
        
        # Synthesize Open, High, Low, Volume based on Close
        df['Open'] = df['Close'] * np.random.normal(1, 0.005, days)
        df['High'] = df[['Open', 'Close']].max(axis=1) * np.random.normal(1.01, 0.005, days)
        df['Low'] = df[['Open', 'Close']].min(axis=1) * np.random.normal(0.99, 0.005, days)
        df['Volume'] = np.random.randint(1000000, 10000000, days)
        
        return df

# ==========================================
# --- 4. ADVANCED INSTITUTIONAL MATH ENGINE ---
# ==========================================
class AlphaEngine:
    @staticmethod
    def apply_kalman_filter(prices, noise_estimate=1.0, measure_noise=1.0):
        n = len(prices)
        kalman_gains = np.zeros(n)
        estimates = np.zeros(n)
        current_estimate = prices.iloc[0]
        err_estimate = noise_estimate
        
        for i in range(n):
            kalman_gains[i] = err_estimate / (err_estimate + measure_noise)
            current_estimate = current_estimate + kalman_gains[i] * (prices.iloc[i] - current_estimate)
            err_estimate = (1 - kalman_gains[i]) * err_estimate
            estimates[i] = current_estimate
            
        return pd.Series(estimates, index=prices.index)

class BacktestEngine:
    @staticmethod
    def run_quick_backtest(df, slippage_bps=5, commission_bps=2):
        bt_df = df.copy().dropna()
        if len(bt_df) < 50:
            return None 
            
        bt_df['Kalman_Price'] = AlphaEngine.apply_kalman_filter(bt_df['Close'])
        returns = bt_df['Close'].pct_change()
        bt_df['Vol_Regime'] = returns.ewm(span=20).std() * np.sqrt(252)
        
        bt_df['Band_Std'] = bt_df['Vol_Regime'] * bt_df['Kalman_Price'] / np.sqrt(252)
        bt_df['Upper_Band'] = bt_df['Kalman_Price'] + (2 * bt_df['Band_Std'])
        bt_df['Lower_Band'] = bt_df['Kalman_Price'] - (2 * bt_df['Band_Std'])

        bt_df['Signal'] = 0
        bt_df.loc[bt_df['Close'] < bt_df['Lower_Band'], 'Signal'] = 1
        bt_df.loc[bt_df['Close'] > bt_df['Upper_Band'], 'Signal'] = -1
        
        bt_df['Target_Position'] = bt_df['Signal'].replace(0, np.nan).ffill().fillna(0)
        bt_df['Actual_Position'] = bt_df['Target_Position'].shift(1).fillna(0)
        bt_df['Underlying_Return'] = returns.fillna(0)
        bt_df['Gross_Return'] = bt_df['Actual_Position'] * bt_df['Underlying_Return']
        
        turnover = bt_df['Actual_Position'].diff().abs().fillna(0)
        total_cost = (slippage_bps + commission_bps) / 10000
        
        bt_df['Net_Return'] = bt_df['Gross_Return'] - (turnover * total_cost * (1 + (bt_df['Vol_Regime'] > 0.35).astype(int)))
        
        win_rate = (bt_df['Net_Return'] > 0).mean() * 100
        cumulative = (1 + bt_df['Net_Return']).prod() - 1
        buy_hold = (1 + bt_df['Underlying_Return']).prod() - 1
        outperf = cumulative - buy_hold
        
        peak = (1 + bt_df['Net_Return']).cumprod().cummax()
        max_dd = (((1 + bt_df['Net_Return']).cumprod() - peak) / peak).min() * 100
        
        wins = bt_df[bt_df['Net_Return'] > 0]['Net_Return']
        losses = bt_df[bt_df['Net_Return'] < 0]['Net_Return']
        
        half_kelly = 0.0
        if len(wins) > 0 and len(losses) > 0:
            win_avg = wins.mean()
            loss_avg = abs(losses.mean())
            win_prob = len(wins) / (len(wins) + len(losses))
            
            if loss_avg > 0:
                kelly_fraction = win_prob - ((1 - win_prob) / (win_avg / loss_avg))
                half_kelly = max(0.0, kelly_fraction / 2.0) * 100 
                
        return {
            "Cumulative_Return": cumulative,
            "Alpha_vs_BuyHold": outperf,
            "Win_Rate": win_rate,
            "Max_Drawdown": max_dd,
            "Half_Kelly_Sizing_Pct": half_kelly,
            "Processed_Data": bt_df
        }

# ==========================================
# --- 5. UI RENDERING LAYER ---
# ==========================================
def render_dashboard(ticker, df, backtest_results):
    st.markdown(f"## 📊 Institutional Profile: **{ticker}**")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Cumulative Net Return", value=f"{backtest_results['Cumulative_Return'] * 100:.2f}%")
    with col2:
        st.metric(label="Alpha (vs Buy & Hold)", value=f"{backtest_results['Alpha_vs_BuyHold'] * 100:.2f}%",
                  delta=f"{backtest_results['Alpha_vs_BuyHold'] * 100:.2f}%")
    with col3:
        st.metric(label="System Win Rate", value=f"{backtest_results['Win_Rate']:.1f}%")
    with col4:
        st.metric(label="Max Drawdown", value=f"{backtest_results['Max_Drawdown']:.2f}%",
                  delta="Risk Limit", delta_color="inverse")

    kelly_pct = backtest_results['Half_Kelly_Sizing_Pct']
    allocation_action = "DEPLOY CAPITAL" if kelly_pct > 0 else "FLATTEN POSITION / NO EDGE"
    box_color = "#082F49" if kelly_pct > 0 else "#450a0a" # Turn red if no edge
    border_color = "#38BDF8" if kelly_pct > 0 else "#f87171"
    
    st.markdown(f"""
    <div class="apex-box" style="background-color: {box_color}; border-left: 5px solid {border_color};">
        <div class="apex-title">⚡ HQTA Allocation Directive</div>
        <div class="apex-action" style="color: {border_color};">ACTION: {allocation_action}</div>
        <div class="apex-logic">
            Optimal Half-Kelly Sizing: <strong>{kelly_pct:.2f}%</strong> of total portfolio equity.<br>
            <em>Calculated using EWMA GARCH-proxied variance and momentum-adjusted win rates.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📈 Quantitative Dynamics: Kalman Centerline & EWMA GARCH Bands")
    
    vis_df = backtest_results["Processed_Data"]
    fig_price = go.Figure()
    
    fig_price.add_trace(go.Candlestick(
        x=vis_df.index, open=vis_df['Open'], high=vis_df['High'], low=vis_df['Low'], close=vis_df['Close'],
        name='Market Price', increasing_line_color='#38BDF8', decreasing_line_color='#334155'
    ))
    fig_price.add_trace(go.Scatter(
        x=vis_df.index, y=vis_df['Kalman_Price'], mode='lines', 
        name='Kalman Filter', line=dict(color='#F8FAFC', width=1.5)
    ))
    fig_price.add_trace(go.Scatter(
        x=vis_df.index, y=vis_df['Upper_Band'], mode='lines', 
        name='Upper Vol Band', line=dict(color='#94A3B8', width=1, dash='dot')
    ))
    fig_price.add_trace(go.Scatter(
        x=vis_df.index, y=vis_df['Lower_Band'], mode='lines', 
        name='Lower Vol Band', line=dict(color='#94A3B8', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(148, 163, 184, 0.1)' 
    ))
    
    fig_price.update_layout(
        template='plotly_dark', paper_bgcolor='#0B0F19', plot_bgcolor='#0B0F19',
        margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False, height=500
    )
    st.plotly_chart(fig_price, use_container_width=True)

# ==========================================
# --- 6. MAIN APP EXECUTION ---
# ==========================================
def main():
    st.set_page_config(page_title="HQTA Alpha Engine", layout="wide", initial_sidebar_state="expanded")
    inject_institutional_css()
    
    with st.sidebar:
        st.title("⚙️ HQTA Control Panel")
        data_mode = st.radio("Data Source", ["Synthetic (Test Mode)", "Live (yfinance)"])
        
        if data_mode == "Live (yfinance)":
            sector = st.selectbox("Select Sector", list(TICKER_SETS.keys()))
            ticker = st.selectbox("Select Ticker", TICKER_SETS[sector])
        else:
            ticker = "SYNTH-GBM"
            st.info("Running on Geometric Brownian Motion dummy data to protect API limits.")
            
        st.markdown("---")
        if st.button("RUN ALPHA ENGINE", use_container_width=True):
            st.session_state['run'] = True

    if st.session_state.get('run', False):
        with st.spinner(f"Aggregating Order Flow & Quant Metrics for {ticker}..."):
            # Throttled Fetching Logic
            if data_mode == "Synthetic (Test Mode)":
                time.sleep(0.5) # Simulate latency
                data = DummyDataGenerator.generate_synthetic_ohlcv(ticker)
            else:
                try:
                    data = yf.download(ticker, period="1y", interval="1d", progress=False)
                    time.sleep(1) # Intentional throttle to prevent yfinance lock
                except Exception as e:
                    st.error(f"Market Scanner Error: {e}")
                    return
            
            if data is not None and len(data) > 50:
                results = BacktestEngine.run_quick_backtest(data)
                if results:
                    render_dashboard(ticker, data, results)
                else:
                    st.warning("Insufficient data to generate Kalman models.")

if __name__ == "__main__":
    main()
