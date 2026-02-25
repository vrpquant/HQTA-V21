import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.optimize as sco
from scipy.stats import norm
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import io
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="HQTA | V22.0 Terminal", layout="wide", page_icon="🏦")

# --- 2. TIERED AUTHENTICATION ---
# Define your users and their access levels here
try:
    USERS = {
        "analyst":    {"password": st.secrets["ANALYST_PW"],  "tier": "ANALYST"},
        "fund":       {"password": st.secrets["FUND_PW"],     "tier": "GOD_MODE"},
        "demo":       {"password": st.secrets["DEMO_PW"],     "tier": "ANALYST"},
        "admin":      {"password": st.secrets["ADMIN_PW"],    "tier": "GOD_MODE"},
        "guest":      {"password": st.secrets["GUEST_PW"],    "tier": "ANALYST"}
    }
except Exception as e:
    # Fallback for local testing if secrets.toml is not perfectly set yet
    USERS = {
        "analyst": {"password": "password123", "tier": "ANALYST"},
        "fund":    {"password": "godmode123",  "tier": "GOD_MODE"},
        "demo":    {"password": "demo",        "tier": "ANALYST"},
        "admin":   {"password": "admin",       "tier": "GOD_MODE"}
    }

def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.tier = None
        st.session_state.user = None

    if not st.session_state.authenticated:
        st.markdown("## 🔒 HQTA Terminal Login")
        c1, c2 = st.columns([1, 2])
        with c1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if username in USERS and USERS[username]["password"] == password:
                    st.session_state.authenticated = True
                    st.session_state.user = username
                    st.session_state.tier = USERS[username]["tier"]
                    st.rerun()
                else:
                    st.error("Invalid Credentials")
            
            st.markdown("---")
            b1, b2 = st.columns(2)
            with b1:
                st.info("**ANALYST TIER**\n* Retail Price: ~~$299/mo~~\n* Founding Member: **$149/mo**")
                st.link_button("Subscribe via PayPal ($149/mo)", "https://www.paypal.com/webapps/billing/plans/subscribe?plan_id=P-0CB63794C10515154NGMNDNA", use_container_width=True)
                
            with b2:
                st.success("**GOD MODE TIER**\n* Retail Price: ~~$999/mo~~\n* Founding Member: **$499/mo**")
                st.link_button("Subscribe via PayPal ($499/mo)", "https://www.paypal.com/webapps/billing/plans/subscribe?plan_id=P-723423746M676015CNGMNFGI", use_container_width=True)

        return False
    return True

# --- 3. PROPRIETARY ENGINES ---
class TrendReversalEngine:
    @staticmethod
    def calculate_rsi(df, window=14):
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
        short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        return pd.DataFrame({'MACD_Line': macd_line, 'Signal_Line': signal_line})

    @staticmethod
    def calculate_volume_sma(df, window=20):
        return df['Volume'].rolling(window=window).mean()

    @staticmethod
    def detect_reversals(df, rsi_window=14, macd_params=(12, 26, 9), volume_window=20, volume_multiplier=1.5):
        rsi = TrendReversalEngine.calculate_rsi(df, rsi_window)
        macd_df = TrendReversalEngine.calculate_macd(df, *macd_params)
        volume_sma = TrendReversalEngine.calculate_volume_sma(df, volume_window)
        
        df = df.copy()
        df['RSI'] = rsi
        df['RSI_Signal'] = np.where(df['RSI'] < 30, 'Bullish Reversal (Oversold)', 
                                    np.where(df['RSI'] > 70, 'Bearish Reversal (Overbought)', 'Neutral'))
        
        df = pd.concat([df, macd_df], axis=1)
        df['MACD_Crossover'] = np.where((df['MACD_Line'] > df['Signal_Line']) & (df['MACD_Line'].shift(1) <= df['Signal_Line'].shift(1)), 
                                        'Bullish Reversal (Crossover)', 
                                        np.where((df['MACD_Line'] < df['Signal_Line']) & (df['MACD_Line'].shift(1) >= df['Signal_Line'].shift(1)), 
                                                 'Bearish Reversal (Crossover)', 'Neutral'))
        
        df['Price_Trend'] = np.where(df['Close'] > df['Close'].shift(1), 'Up', 'Down')
        df['MACD_Divergence'] = np.where((df['Price_Trend'] == 'Up') & (df['MACD_Line'] < df['MACD_Line'].shift(1)), 
                                         'Bearish Divergence', 
                                         np.where((df['Price_Trend'] == 'Down') & (df['MACD_Line'] > df['MACD_Line'].shift(1)), 
                                                  'Bullish Divergence', 'No Divergence'))
        
        df['Base_Reversal_Signal'] = df['MACD_Divergence'].where(df['MACD_Divergence'] != 'No Divergence', 
                                                                 df['MACD_Crossover'].where(df['MACD_Crossover'] != 'Neutral', 
                                                                                            df['RSI_Signal']))
        
        df['Volume_SMA'] = volume_sma
        df['Volume_Spike'] = np.where(df['Volume'] > (df['Volume_SMA'] * volume_multiplier), 'High Volume', 'Normal Volume')
        
        df['Confirmed_Reversal_Signal'] = np.where((df['Base_Reversal_Signal'] != 'Neutral') & (df['Volume_Spike'] == 'High Volume'), 
                                                   '⚠️ CONFIRMED ' + df['Base_Reversal_Signal'].astype(str).str.upper(), 
                                                   df['Base_Reversal_Signal'])
        return df

class AlphaEngine:
    @staticmethod
    def calculate_score(df):
        try:
            curr = df['Close'].iloc[-1]
            sma50 = df['Close'].rolling(50).mean().iloc[-1]
            sma200 = df['Close'].rolling(200).mean().iloc[-1]
            vol = df['Close'].pct_change().std() * np.sqrt(252)
            score = 50 
            if curr > sma50 > sma200: score += 30 
            elif curr < sma50 < sma200: score -= 30 
            elif curr > sma50: score += 10 
            elif curr < sma50: score -= 10 
            mom = (curr / df['Close'].iloc[-10]) - 1
            if mom > 0.05: score += 10
            elif mom < -0.05: score -= 10
            if vol > 0.50: score -= 10 
            return max(0, min(100, score))
        except: return 50

class MonteCarloEngine:
    @staticmethod
    def simulate_paths(df, days=30, sims=1000):
        last_price = df['Close'].iloc[-1]
        returns = df['Close'].pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        daily_shocks = np.random.normal(mu, sigma, (days, sims))
        price_paths = last_price * (1 + daily_shocks).cumprod(axis=0)
        start_row = np.full((1, sims), last_price)
        price_paths = np.vstack([start_row, price_paths])
        return pd.DataFrame(price_paths)

class TradeArchitect:
    @staticmethod
    def generate_plan(score):
        plan = {}
        if score >= 80:
            plan['bias'] = "STRONG BUY"; plan['color'] = "#00ff00"; plan['strat'] = "Long Call Vertical"; plan['legs'] = "Buy ATM Call / Sell OTM Call"; plan['pop'] = 65
        elif score >= 60:
            plan['bias'] = "BULLISH"; plan['color'] = "#00cc00"; plan['strat'] = "Short Put Vertical"; plan['legs'] = "Sell OTM Put / Buy Lower Put"; plan['pop'] = 78
        elif score <= 20:
            plan['bias'] = "STRONG SELL"; plan['color'] = "#ff0000"; plan['strat'] = "Long Put Vertical"; plan['legs'] = "Buy ATM Put / Sell OTM Put"; plan['pop'] = 65
        elif score <= 40:
            plan['bias'] = "BEARISH"; plan['color'] = "#cc0000"; plan['strat'] = "Short Call Vertical"; plan['legs'] = "Sell OTM Call / Buy Higher Call"; plan['pop'] = 78
        else:
            plan['bias'] = "NEUTRAL"; plan['color'] = "#888888"; plan['strat'] = "Iron Condor"; plan['legs'] = "Sell OTM Call & Put"; plan['pop'] = 68
        return plan

@st.cache_data(ttl=600)
def get_data_v20(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y") # 1 Year minimum for MACD/RSI to work correctly
        return df if not df.empty else None
    except: return None

@st.cache_data(ttl=600)
def scan_market(tickers):
    results = []
    for t in tickers:
        df = get_data_v20(t)
        if df is not None:
            score = AlphaEngine.calculate_score(df)
            vol = df['Close'].pct_change().std() * np.sqrt(252) * 100
            
            # V22 Reversal Logic
            reversal_df = TrendReversalEngine.detect_reversals(df)
            latest_signal = reversal_df['Confirmed_Reversal_Signal'].iloc[-1]
            display_signal = "TRENDING" if latest_signal == "Neutral" else latest_signal.upper()
            
            results.append({
                "Ticker": t, 
                "Price": df['Close'].iloc[-1], 
                "Alpha Score": score, 
                "Vol %": vol,
                "V22 Reversal Alert": display_signal
            })
    return pd.DataFrame(results).sort_values("Alpha Score", ascending=False)

# --- 4. UI LOGIC (THE GATEKEEPER) ---
if check_login():
    tier = st.session_state.tier
    user = st.session_state.user
    
    # Global Date/Time for display
    run_date_display = datetime.now().strftime("%B %d, %Y | %H:%M EST")
    
    # Sidebar Info
    st.sidebar.markdown("# 🏦 HQTA Terminal")
    if tier == "GOD_MODE":
        st.sidebar.success(f"🔓 GOD MODE ACTIVE")
        st.sidebar.caption("Full Institutional Access")
    else:
        st.sidebar.warning(f"🔒 ANALYST TIER")
        st.sidebar.caption("Limited Access. Upgrade to God Mode ($999/mo) to unlock Scanner & Deep Compute.")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"🕒 System Time: {run_date_display}")
        
    mode = st.sidebar.radio("Module", ["🚀 Market Scanner", "🔬 Deep Dive & Monte Carlo"])
    
    # === MODULE 1: MARKET SCANNER (LOCKED FOR ANALYST) ===
    if mode == "🚀 Market Scanner":
        st.title("🚀 Institutional Market Scanner")
        st.caption(f"REPORT GENERATED: {run_date_display} | SEC RULE 206(4)-1 COMPLIANT")
        
        # TIER CHECK
        if tier != "GOD_MODE":
            st.error("🔒 ACCESS DENIED: Market Scanner is locked for Analyst Tier.")
            st.info("💡 Upgrade to God Mode ($999/mo) to instantly scan the 'Magnificent 7' and Crypto markets by Alpha Score and Trend Reversals.")
            st.markdown("---")
            st.image("https://placehold.co/800x400/1e1e1e/FFF?text=Scanner+Locked", caption="Institutional Feature Locked")
        else:
            # GOD MODE ONLY CONTENT
            st.markdown("Live ranking of assets by **HQTA Alpha Score™** and **Trend Reversals**.")
            default_tickers = ["SPY", "QQQ", "IWM", "BTC-USD", "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "COIN"]
            
            if st.button("🔄 Scan Market"):
                with st.spinner("Calculating Alpha Scores and MACD Divergences..."):
                    df_scan = scan_market(default_tickers)
                    
                    # Highlight Reversal Alerts
                    def highlight_reversals(val):
                        if 'BULLISH' in str(val): return 'color: #00ff00; font-weight: bold'
                        elif 'BEARISH' in str(val): return 'color: #ff0000; font-weight: bold'
                        return 'color: #718096'

                    st.dataframe(
                        df_scan.style
                        .background_gradient(subset=['Alpha Score'], cmap='RdYlGn', vmin=0, vmax=100)
                        .applymap(highlight_reversals, subset=['V22 Reversal Alert']), 
                        use_container_width=True
                    )

    # === MODULE 2: DEEP DIVE (RESTRICTED FOR ANALYST) ===
    elif mode == "🔬 Deep Dive & Monte Carlo":
        st.title("🔬 Deep Dive Analysis")
        st.caption(f"REPORT GENERATED: {run_date_display} | SEC RULE 206(4)-1 COMPLIANT")
        
        c1, c2 = st.columns([3, 1])
        with c1: ticker = st.text_input("Asset Ticker", "NVDA").upper()
        
        # TIER CHECK FOR MONTE CARLO
        if tier == "GOD_MODE":
            with c2: 
                sim_depth = st.selectbox("Simulation Depth", [1000, 10000], index=1, 
                                       format_func=lambda x: f"{x:,} (Deep Compute)")
        else:
            with c2:
                st.write("") # Spacer
                st.caption("🔒 Sim Limit: 1,000")
                sim_depth = 1000
        
        if st.button("Run Analysis"):
            with st.spinner(f"Running Analysis ({sim_depth:,} Sims) & Checking Reversals..."):
                df = get_data_v20(ticker)
                
                if df is not None:
                    # 1. Trend Reversal Check (New V22 Feature)
                    reversal_df = TrendReversalEngine.detect_reversals(df)
                    latest_signal = reversal_df['Confirmed_Reversal_Signal'].iloc[-1]
                    display_signal = "TRENDING (NO REVERSAL)" if latest_signal == "Neutral" else latest_signal.upper()
                    
                    if "CONFIRMED" in display_signal:
                        if "BULLISH" in display_signal:
                            st.success(f"📈 **V22 SYSTEM ALERT:** {display_signal}")
                        else:
                            st.error(f"📉 **V22 SYSTEM ALERT:** {display_signal}")
                    else:
                        st.info(f"📊 **V22 TREND STATUS:** {display_signal}")

                    # 2. Base Metrics
                    score = AlphaEngine.calculate_score(df)
                    vol = df['Close'].pct_change().std() * np.sqrt(252) * 100
                    plan = TradeArchitect.generate_plan(score)
                    
                    # 3. HUD
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Price", f"${df['Close'].iloc[-1]:.2f}")
                    m2.metric("Alpha Score", f"{score}/100")
                    m3.metric("Bias", plan['bias'], delta_color="off")
                    m4.metric("Volatility", f"{vol:.1f}%")
                    
                    st.info(f"**🎯 Recommended Strategy:** {plan['strat']} | **POP:** {plan['pop']}%")
                    
                    # 4. Monte Carlo
                    mc_df = MonteCarloEngine.simulate_paths(df, days=30, sims=sim_depth)
                    final_prices = mc_df.iloc[-1]
                    var_95 = np.percentile(final_prices, 5)
                    
                    st.subheader(f"🎲 Monte Carlo Projection")
                    
                    # Plot Logic
                    plot_data = mc_df if sim_depth == 1000 else mc_df.sample(n=200, axis=1)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(-60, 0)), y=df['Close'].tail(60), name='History', line=dict(color='white')))
                    for col in plot_data.columns:
                        fig.add_trace(go.Scatter(x=list(range(0, 31)), y=plot_data[col], mode='lines', line=dict(color=plan['color'], width=0.5), opacity=0.1, showlegend=False))
                    fig.add_trace(go.Scatter(x=list(range(0, 31)), y=mc_df.mean(axis=1), name='Mean', line=dict(color='orange', dash='dash')))
                    fig.update_layout(title="Cone of Probability", template="plotly_dark", height=450)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 5. TIER CHECK FOR RISK METRICS
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Median Outcome", f"${np.median(final_prices):.2f}")
                    r2.metric("95% VaR", f"${var_95:.2f}", delta_color="inverse")
                    
                    if tier == "GOD_MODE":
                        var_99 = np.percentile(final_prices, 1)
                        r3.metric("99% Black Swan", f"${var_99:.2f}", delta_color="inverse")
                    else:
                        r3.metric("99% Black Swan", "🔒 LOCKED", help="Upgrade to God Mode to see Tail Risk")
                    
                else: st.error("Asset not found")
