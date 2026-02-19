import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime

# ==========================================
# --- PART 1: THE CORE ENGINE (BACKEND) ---
# ==========================================

class DataHandler:
    @staticmethod
    @st.cache_data(ttl=600)
    def fetch(ticker):
        try:
            df = yf.Ticker(ticker).history(period="2y")
            return df if not df.empty else None
        except: return None

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

class QuantLogic:
    @staticmethod
    def get_support_resistance(df):
        res = df['High'].rolling(window=50).max().iloc[-1]
        sup = df['Low'].rolling(window=50).min().iloc[-1]
        return sup, res

    @staticmethod
    def calculate_vol(df):
        return df['Close'].pct_change().std() * np.sqrt(252) * 100

class TradeArchitect:
    @staticmethod
    def generate_plan(ticker, price, score, vol, sup, res):
        plan = {}
        if score >= 60: bias = "BULLISH"
        elif score <= 40: bias = "BEARISH"
        else: bias = "NEUTRAL"
        
        vol_regime = "HIGH" if vol > 35 else "LOW"
        
        if bias == "BULLISH":
            if vol_regime == "LOW":
                plan['name'] = "Long Call Vertical (Debit)"
                plan['legs'] = f"Buy ATM Call (${price:.0f}) / Sell Res Call (${res:.0f})"
                plan['pop'] = 62
            else:
                plan['name'] = "Short Put Vertical (Credit)"
                plan['legs'] = f"Sell Supp Put (${sup:.0f}) / Buy Lower Put"
                plan['pop'] = 78
        elif bias == "BEARISH":
            if vol_regime == "LOW":
                plan['name'] = "Long Put Vertical (Debit)"
                plan['legs'] = f"Buy ATM Put (${price:.0f}) / Sell Supp Put (${sup:.0f})"
                plan['pop'] = 62
            else:
                plan['name'] = "Short Call Vertical (Credit)"
                plan['legs'] = f"Sell Res Call (${res:.0f}) / Buy Higher Call"
                plan['pop'] = 78
        else:
            if vol_regime == "HIGH":
                plan['name'] = "Iron Condor"
                plan['legs'] = f"Sell Put (${sup:.0f}) / Sell Call (${res:.0f})"
                plan['pop'] = 68
            else:
                plan['name'] = "Calendar Spread"
                plan['legs'] = f"Sell 30D Call / Buy 60D Call (${price:.0f})"
                plan['pop'] = 55
                
        plan['dte'] = "30-45 Days"
        plan['bias'] = bias
        return plan

class MonteCarloEngine:
    @staticmethod
    def simulate_paths(df, days=30, sims=1000):
        last_price = df['Close'].iloc[-1]
        returns = df['Close'].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        daily_shocks = np.random.normal(mu, sigma, (days, sims))
        price_paths = last_price * (1 + daily_shocks).cumprod(axis=0)
        start_row = np.full((1, sims), last_price)
        price_paths = np.vstack([start_row, price_paths])
        return pd.DataFrame(price_paths)

class MarketScanner:
    @staticmethod
    def run_scan(tickers):
        results = []
        for t in tickers:
            df = DataHandler.fetch(t)
            if df is not None:
                score = AlphaEngine.calculate_score(df)
                vol = QuantLogic.calculate_vol(df)
                results.append({
                    "Ticker": t,
                    "Price": df['Close'].iloc[-1],
                    "Alpha Score": score,
                    "Vol %": vol
                })
        return pd.DataFrame(results).sort_values("Alpha Score", ascending=False)


# ==========================================
# --- PART 2: THE STREAMLIT APP (UI) ---
# ==========================================

st.set_page_config(page_title="HQTA | V21.2 Command", layout="wide", page_icon="🏦")

USERS = {
    "analyst":    {"password": "data2026",  "tier": "ANALYST"},
    "fund":       {"password": "alpha2026", "tier": "GOD_MODE"},
    "demo":       {"password": "demo",      "tier": "ANALYST"},
    "admin":      {"password": "admin",     "tier": "GOD_MODE"},
    "guest":      {"password": "start_risk_free", "tier": "ANALYST"}
}

DISCLAIMER_TEXT = """
**REGULATORY DISCLAIMER & COMPLIANCE NOTICE**
1. **No Financial Advice:** HQTA is a quantitative tool for informational purposes only.
2. **Risk Warning:** Trading involves substantial risk. You are solely responsible for your trades.
3. **Hypothetical Results:** Alpha Scores and Monte Carlo projections are theoretical.
"""

def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.tier = None
        
    if not st.session_state.authenticated:
        st.markdown("## 🔒 HQTA Terminal Login")
        c1, c2 = st.columns([1, 2])
        with c1:
            user = st.text_input("Username", key="login_u")
            pwd = st.text_input("Password", type="password", key="login_p")
            if st.button("Login"):
                if user in USERS and USERS[user]["password"] == pwd:
                    st.session_state.authenticated = True
                    st.session_state.tier = USERS[user]["tier"]
                    st.rerun()
                else: st.error("Invalid Credentials")
        
        # --- THE WAITLIST FIX ---
        st.markdown("---")
        st.caption("New Client? The Payment Gateway opens next week.")
        b1, b2 = st.columns(2)
        
        if b1.button("Subscribe Analyst ($299)"):
            st.info("🚧 The Stripe Gateway is currently locked for Beta Testing. DM the founder 'WAITLIST' to secure your spot.")
            
        if b2.button("Subscribe God Mode ($999)"):
            st.info("🚧 The Stripe Gateway is currently locked for Beta Testing. DM the founder 'WAITLIST' to secure your spot.")
            
        return False
    return True

if check_login():
    tier = st.session_state.tier
    
    st.sidebar.markdown("# 🏦 HQTA V21.2")
    if tier == "GOD_MODE": st.sidebar.success("🔓 GOD MODE ACTIVE")
    else: st.sidebar.warning("🔒 ANALYST TIER")
        
    mode = st.sidebar.radio("Module", ["🚀 Market Scanner", "🔬 Deep Dive Analysis"])

    # === MODULE 1: MARKET SCANNER ===
    if mode == "🚀 Market Scanner":
        st.title("🚀 Institutional Market Scanner")
        
        TICKER_SETS = {
            "🔥 Magnificent 7 + Crypto": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "BTC-USD", "ETH-USD", "COIN"],
            "💻 Semiconductors (AI)": ["NVDA", "AMD", "AVGO", "TSM", "INTC", "QCOM", "MU", "SMH", "SOXL"],
            "🛢️ Energy & Commodities": ["XLE", "USO", "GLD", "SLV", "CCJ", "URA", "CVX", "XOM", "UNG"],
            "📉 Volatility & Hedges": ["VIXY", "UVXY", "TLT", "SH", "SQQQ", "SPXU"],
            "🏦 Financials": ["JPM", "GS", "BAC", "MS", "C", "XLF", "KRE"]
        }

        if tier != "GOD_MODE":
            st.error("🔒 ACCESS DENIED: Market Scanner is locked for Analyst Tier.")
            st.info("Subscribe to God Mode ($999/mo) to unlock.")
            st.code("ERROR 403: PREMIUM_FEATURE_LOCKED", language="text")
        else:
            st.markdown("### Select Institutional Universe")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                options = list(TICKER_SETS.keys()) + ["✨ Custom Watchlist"]
                sector_choice = st.selectbox("Select Sector:", options)
            
            selected_tickers = []
            if sector_choice == "✨ Custom Watchlist":
                with col2:
                    st.info("Paste your own list of tickers to scan.")
                    custom_input = st.text_area("Enter Tickers (comma separated):", "PLTR, SOFI, DKNG")
                    if custom_input: 
                        selected_tickers = [t.strip().upper() for t in custom_input.split(',')]
            else: 
                selected_tickers = TICKER_SETS[sector_choice]
            
            if st.button("🔄 Run Live Scan") and selected_tickers:
                with st.spinner(f"Scanning {len(selected_tickers)} Assets..."):
                    df_scan = MarketScanner.run_scan(selected_tickers)
                    st.dataframe(df_scan, column_config={
                        "Ticker": st.column_config.TextColumn("Ticker"),
                        "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                        "Alpha Score": st.column_config.ProgressColumn("Alpha Score", format="%d", min_value=0, max_value=100),
                        "Vol %": st.column_config.NumberColumn("Vol %", format="%.1f%%"),
                    }, use_container_width=True)
                    
                    csv = df_scan.to_csv(index=False).encode('utf-8')
                    st.download_button("💾 Download Results (CSV)", csv, "HQTA_Scan.csv", "text/csv")

    # === MODULE 2: DEEP DIVE ===
    elif mode == "🔬 Deep Dive Analysis":
        st.title("🔬 Deep Dive & Trade Architect")
        ticker = st.text_input("Asset Ticker", "NVDA").upper()
        
        if st.button("Run Analysis"):
            with st.spinner("Architecting Trade Strategy..."):
                df = DataHandler.fetch(ticker)
                if df is not None:
                    curr_price = df['Close'].iloc[-1]
                    score = AlphaEngine.calculate_score(df)
                    vol = QuantLogic.calculate_vol(df)
                    sup, res = QuantLogic.get_support_resistance(df)
                    
                    plan = TradeArchitect.generate_plan(ticker, curr_price, score, vol, sup, res)
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Price", f"${curr_price:.2f}")
                    m2.metric("Alpha Score", f"{score}/100")
                    m3.metric("Trend Bias", plan['bias'])
                    m4.metric("Volatility", f"{vol:.1f}%")
                    
                    st.info(f"**🎯 OPTIMAL STRATEGY:** {plan['name']}")
                    s1, s2, s3 = st.columns(3)
                    s1.caption(f"**Legs:** {plan['legs']}")
                    s2.metric("Prob. of Profit (POP)", f"{plan['pop']}%")
                    s3.metric("Ideal DTE", plan['dte'])
                    
                    sims = 10000 if tier == "GOD_MODE" else 1000
                    mc_df = MonteCarloEngine.simulate_paths(df, days=30, sims=sims)
                    var_95 = np.percentile(mc_df.iloc[-1], 5)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=df['Close'], name='History', line=dict(color='white')))
                    fig.add_trace(go.Scatter(x=list(range(len(df), len(df)+30)), y=mc_df.mean(axis=1), name='Mean Projection', line=dict(dash='dash', color='orange')))
                    fig.add_hline(y=sup, line_dash="dot", line_color="green", annotation_text="Support")
                    fig.add_hline(y=res, line_dash="dot", line_color="red", annotation_text="Resistance")
                    
                    fig.update_layout(template="plotly_dark", height=500, title="Institutional Chart (History + Projection)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    r1, r2 = st.columns(2)
                    r1.metric("95% Value at Risk (Stop)", f"${var_95:.2f}", delta_color="inverse")
                    
                    if tier == "GOD_MODE":
                        var_99 = np.percentile(mc_df.iloc[-1], 1)
                        r2.metric("99% Black Swan Level", f"${var_99:.2f}", delta_color="inverse", help="1% Tail Risk")
                        status = "GOD MODE - DEEP COMPUTE VERIFIED"
                    else:
                        r2.metric("99% Black Swan", "🔒 LOCKED", help="Upgrade to God Mode")
                        status = "ANALYST TIER - STANDARD"

                    report_txt = f"""HQTA V21.2 INSTITUTIONAL REPORT
--------------------------------
DATE: {datetime.now().strftime('%Y-%m-%d %H:%M')}
TICKER: {ticker}
ALPHA SCORE: {score}/100
BIAS: {plan['bias']}

STRATEGY ARCHITECT:
Strategy: {plan['name']}
Legs: {plan['legs']}
POP: {plan['pop']}%
DTE: {plan['dte']}

TECHNICALS:
Support: ${sup:.2f}
Resistance: ${res:.2f}
Volatility: {vol:.1f}%

RISK ANALYSIS:
95% Value at Risk: ${var_95:.2f}
Status: {status}

--------------------------------
{DISCLAIMER_TEXT.replace('**', '')}
"""
                    st.download_button("💾 Download Trade Plan (TXT)", report_txt, f"{ticker}_Trade_Plan.txt")
                    
                else: st.error("Asset not found")
        
        st.markdown("---")
        st.caption(DISCLAIMER_TEXT)

    with st.sidebar:
        st.markdown("---")
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.rerun()

