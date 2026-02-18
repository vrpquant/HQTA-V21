import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.optimize as sco
from scipy.stats import norm
import io
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="HQTA | V20.1 Tiered", layout="wide", page_icon="🏦")

# --- 2. TIERED AUTHENTICATION ---
# This is your "Database". Add new clients here.
USERS = {
    # "username": {"password": "pwd", "tier": "TIER_LEVEL"}
    "analyst":    {"password": "data2026",  "tier": "ANALYST"},   # $299 Tier
    "fund":       {"password": "alpha2026", "tier": "GOD_MODE"},  # $999 Tier
    "demo":       {"password": "demo",      "tier": "ANALYST"},   # Sales Demo
    "admin":      {"password": "admin",     "tier": "GOD_MODE"}   # Your Access
}

def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.tier = None
        st.session_state.user = None

    if not st.session_state.authenticated:
        st.markdown("## 🔒 HQTA Terminal Login")
        st.caption("Restricted Institutional Access")
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
        return False
    return True

# --- 3. PROPRIETARY ENGINES ---
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
        # Vectorized for speed
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
        df = yf.Ticker(ticker).history(period="2y")
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
            results.append({"Ticker": t, "Price": df['Close'].iloc[-1], "Alpha Score": score, "Vol %": vol})
    return pd.DataFrame(results).sort_values("Alpha Score", ascending=False)

# --- 4. UI LOGIC (THE GATEKEEPER) ---
if check_login():
    tier = st.session_state.tier
    user = st.session_state.user
    
    # Sidebar Info
    st.sidebar.markdown("# 🏦 HQTA Terminal")
    if tier == "GOD_MODE":
        st.sidebar.success(f"🔓 GOD MODE ACTIVE")
        st.sidebar.caption("Full Institutional Access")
    else:
        st.sidebar.warning(f"🔒 ANALYST TIER")
        st.sidebar.caption("Limited Access. Upgrade to God Mode ($999/mo) to unlock Scanner & Deep Compute.")
        
    mode = st.sidebar.radio("Module", ["🚀 Market Scanner", "🔬 Deep Dive & Monte Carlo"])
    
    # === MODULE 1: MARKET SCANNER (LOCKED FOR ANALYST) ===
    if mode == "🚀 Market Scanner":
        st.title("🚀 Institutional Market Scanner")
        
        # TIER CHECK
        if tier != "GOD_MODE":
            st.error("🔒 ACCESS DENIED: Market Scanner is locked for Analyst Tier.")
            st.info("💡 Upgrade to God Mode ($999/mo) to instantly scan the 'Magnificent 7' and Crypto markets by Alpha Score.")
            st.markdown("---")
            st.code("ERROR 403: PREMIUM_FEATURE_LOCKED", language="text")
        else:
            # GOD MODE ONLY CONTENT
            st.markdown("Live ranking of assets by **HQTA Alpha Score™**.")
            default_tickers = ["SPY", "QQQ", "IWM", "BTC-USD", "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "COIN"]
            
            if st.button("🔄 Scan Market"):
                with st.spinner("Calculating Alpha Scores..."):
                    df_scan = scan_market(default_tickers)
                    st.dataframe(df_scan.style.background_gradient(subset=['Alpha Score'], cmap='RdYlGn', vmin=0, vmax=100), use_container_width=True)

    # === MODULE 2: DEEP DIVE (RESTRICTED FOR ANALYST) ===
    elif mode == "🔬 Deep Dive & Monte Carlo":
        st.title("🔬 Deep Dive Analysis")
        
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
            with st.spinner(f"Running Analysis ({sim_depth:,} Sims)..."):
                df = get_data_v20(ticker)
                
                if df is not None:
                    score = AlphaEngine.calculate_score(df)
                    vol = df['Close'].pct_change().std() * np.sqrt(252) * 100
                    plan = TradeArchitect.generate_plan(score)
                    
                    # HUD
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Price", f"${df['Close'].iloc[-1]:.2f}")
                    m2.metric("Alpha Score", f"{score}/100")
                    m3.metric("Bias", plan['bias'], delta_color="off")
                    m4.metric("Volatility", f"{vol:.1f}%")
                    
                    st.info(f"**🎯 Strategy:** {plan['strat']} | **POP:** {plan['pop']}%")
                    
                    # Monte Carlo
                    mc_df = MonteCarloEngine.simulate_paths(df, days=30, sims=sim_depth)
                    final_prices = mc_df.iloc[-1]
                    var_95 = np.percentile(final_prices, 5)
                    
                    st.subheader(f"🎲 Monte Carlo Projection")
                    
                    # Plot Logic (Optimized for Browser)
                    plot_data = mc_df if sim_depth == 1000 else mc_df.sample(n=200, axis=1)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(-60, 0)), y=df['Close'].tail(60), name='History', line=dict(color='white')))
                    for col in plot_data.columns:
                        fig.add_trace(go.Scatter(x=list(range(0, 31)), y=plot_data[col], mode='lines', line=dict(color=plan['color'], width=0.5), opacity=0.1, showlegend=False))
                    fig.add_trace(go.Scatter(x=list(range(0, 31)), y=mc_df.mean(axis=1), name='Mean', line=dict(color='orange', dash='dash')))
                    fig.update_layout(title="Cone of Probability", template="plotly_dark", height=450)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # TIER CHECK FOR METRICS
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Median Outcome", f"${np.median(final_prices):.2f}")
                    r2.metric("95% VaR", f"${var_95:.2f}", delta_color="inverse")
                    
                    if tier == "GOD_MODE":
                        var_99 = np.percentile(final_prices, 1)
                        r3.metric("99% Black Swan", f"${var_99:.2f}", delta_color="inverse", help="The 1% Tail Risk Event")
                    else:
                        r3.metric("99% Black Swan", "🔒 LOCKED", help="Upgrade to God Mode to see Tail Risk")
                    
                    # REPORT GENERATION & SAVE (V20 Feature Restored)
                    report_txt = f"""HQTA V20.1 INSTITUTIONAL REPORT
--------------------------------
DATE: {datetime.now().strftime('%Y-%m-%d %H:%M')}
TICKER: {ticker}
ALPHA SCORE: {score}/100
BIAS: {plan['bias']}
VOLATILITY: {vol:.1f}%

STRATEGY RECOMMENDATION:
Structure: {plan['strat']}
Legs: {plan['legs']}
Probability of Profit: {plan['pop']}%

RISK ANALYSIS:
95% Value at Risk (VaR): ${var_95:.2f}
"""
                    if tier == "GOD_MODE":
                        var_99 = np.percentile(final_prices, 1)
                        report_txt += f"99% Black Swan Level: ${var_99:.2f}\n"
                        report_txt += "STATUS: GOD MODE - DEEP COMPUTE VERIFIED"
                    else:
                        report_txt += "STATUS: ANALYST TIER - STANDARD RESOLUTION"

                    st.download_button("💾 Download Institutional Report", report_txt, f"{ticker}_HQTA_Report.txt")
                    
                else: st.error("Asset not found")

# --- 5. LOGOUT ---
if check_login():
    with st.sidebar:
        st.markdown("---")
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.rerun()