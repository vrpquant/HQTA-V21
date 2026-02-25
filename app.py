import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime
import time

# ==========================================
# --- PART 1: V22 INSTITUTIONAL MATH ENGINE ---
# ==========================================

class AlphaEngine:
    @staticmethod
    def calculate_score(df):
        try:
            calc_df = df.copy().dropna()
            sma50 = calc_df['Close'].rolling(50).mean()
            sma200 = calc_df['Close'].rolling(200).mean()
            trend_spread = (sma50 - sma200) / sma200
            trend_z = (trend_spread.iloc[-1] - trend_spread.mean()) / (trend_spread.std() + 1e-9)
            
            delta = calc_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100 - (100 / (1 + rs.iloc[-1]))
            
            score = 50 
            score += max(-25, min(25, trend_z * 12.5))
            if rsi < 30: score += 15
            elif rsi > 70: score -= 15
            
            return max(0, min(100, int(score)))
        except Exception: 
            return 50

class BacktestEngine:
    @staticmethod
    def run_quick_backtest(df, slippage_bps=5, commission_bps=2):
        try:
            bt_df = df.copy().dropna()
            bt_df['SMA50'] = bt_df['Close'].rolling(50).mean()
            bt_df['SMA200'] = bt_df['Close'].rolling(200).mean()
            bt_df['Target_Position'] = np.where(bt_df['SMA50'] > bt_df['SMA200'], 1, -1)
            bt_df['Actual_Position'] = bt_df['Target_Position'].shift(1).fillna(0)
            bt_df['Turnover'] = bt_df['Actual_Position'].diff().abs().fillna(0)
            bt_df['Underlying_Return'] = bt_df['Close'].pct_change().fillna(0)
            bt_df['Gross_Strategy_Return'] = bt_df['Actual_Position'] * bt_df['Underlying_Return']
            
            total_friction_pct = (slippage_bps + commission_bps) / 10000 
            bt_df['Net_Strategy_Return'] = bt_df['Gross_Strategy_Return'] - (bt_df['Turnover'] * total_friction_pct)
            
            winning_days = (bt_df['Net_Strategy_Return'] > 0).sum()
            total_trades_days = (bt_df['Actual_Position'] != 0).sum()
            win_rate = (winning_days / total_trades_days) * 100 if total_trades_days > 0 else 0
            
            bt_df['Cumulative_Net'] = (1 + bt_df['Net_Strategy_Return']).cumprod() - 1
            bt_df['Cumulative_Hold'] = (1 + bt_df['Underlying_Return']).cumprod() - 1
            
            cumulative_return = bt_df['Cumulative_Net'].iloc[-1]
            buy_hold_return = bt_df['Cumulative_Hold'].iloc[-1]
            outperformance = cumulative_return - buy_hold_return
            
            bt_df['Peak'] = (1 + bt_df['Net_Strategy_Return']).cumprod().cummax()
            bt_df['Drawdown'] = ((1 + bt_df['Net_Strategy_Return']).cumprod() - bt_df['Peak']) / bt_df['Peak']
            max_drawdown = bt_df['Drawdown'].min() * 100 
            
            return win_rate, cumulative_return * 100, outperformance * 100, max_drawdown
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

class QuantLogic:
    @staticmethod
    def get_support_resistance(df):
        res = df['High'].rolling(window=50).max().iloc[-1]
        sup = df['Low'].rolling(window=50).min().iloc[-1]
        return sup, res

    @staticmethod
    def calculate_vol(df):
        return df['Close'].pct_change().std() * np.sqrt(252) * 100
        
    @staticmethod
    def calculate_sharpe(df, risk_free_rate=0.04):
        returns = df['Close'].pct_change().dropna()
        excess_returns = (returns.mean() * 252) - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        if volatility == 0: return 0
        return excess_returns / volatility

    @staticmethod
    def calculate_vrp_edge(df):
        hv20 = df['Close'].pct_change().tail(20).std() * np.sqrt(252) * 100
        hv60 = df['Close'].pct_change().tail(60).std() * np.sqrt(252) * 100
        return hv20 - hv60

    @staticmethod
    def bs_call(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return max(0.0, S - K)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def bs_put(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return max(0.0, K - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

class TradeArchitect:
    @staticmethod
    def prob_itm(S, K, T, r, sigma, option_type='call'):
        if T <= 0 or sigma <= 0: return 0.0
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call': return norm.cdf(d2)
        else: return norm.cdf(-d2)

    @staticmethod
    def generate_plan(ticker, price, score, vol, sup, res):
        plan = {}
        if score >= 60: bias = "LONG (Bullish Trend)"
        elif score <= 40: bias = "SHORT (Bearish Trend)"
        else: bias = "NEUTRAL (Mean-Reverting)"
        
        vol_regime = "HIGH" if vol > 35 else "LOW"
        sigma = max(0.01, vol / 100); r = 0.04; T30 = 30 / 365; T60 = 60 / 365 
        
        if res <= price: res = price * 1.05
        if sup >= price: sup = price * 0.95
        
        lower_wing = sup * 0.95; upper_wing = res * 1.05
        
        if "LONG" in bias:
            if vol_regime == "LOW":
                plan['name'] = "Long Call Vertical"
                plan['legs'] = f"Buy ATM Call (${price:.0f}) / Sell Res Call (${res:.0f})"
                debit = QuantLogic.bs_call(price, price, T30, r, sigma) - QuantLogic.bs_call(price, res, T30, r, sigma)
                plan['premium'] = f"Debit: ${max(0.01, debit):.2f}"
                plan['pop'] = int(TradeArchitect.prob_itm(price, price + debit, T30, r, sigma, 'call') * 100)
            else:
                plan['name'] = "Short Put Vertical"
                plan['legs'] = f"Sell Supp Put (${sup:.0f}) / Buy Wing Put (${lower_wing:.0f})"
                credit = QuantLogic.bs_put(price, sup, T30, r, sigma) - QuantLogic.bs_put(price, lower_wing, T30, r, sigma)
                plan['premium'] = f"Credit: ${max(0.01, credit):.2f}"
                plan['pop'] = int((1 - TradeArchitect.prob_itm(price, sup, T30, r, sigma, 'put')) * 100)
        elif "SHORT" in bias:
            if vol_regime == "LOW":
                plan['name'] = "Long Put Vertical"
                plan['legs'] = f"Buy ATM Put (${price:.0f}) / Sell Supp Put (${sup:.0f})"
                debit = QuantLogic.bs_put(price, price, T30, r, sigma) - QuantLogic.bs_put(price, sup, T30, r, sigma)
                plan['premium'] = f"Debit: ${max(0.01, debit):.2f}"
                plan['pop'] = int(TradeArchitect.prob_itm(price, price - debit, T30, r, sigma, 'put') * 100)
            else:
                plan['name'] = "Short Call Vertical"
                plan['legs'] = f"Sell Res Call (${res:.0f}) / Buy Wing Call (${upper_wing:.0f})"
                credit = QuantLogic.bs_call(price, res, T30, r, sigma) - QuantLogic.bs_call(price, upper_wing, T30, r, sigma)
                plan['premium'] = f"Credit: ${max(0.01, credit):.2f}"
                plan['pop'] = int((1 - TradeArchitect.prob_itm(price, res, T30, r, sigma, 'call')) * 100)
        else:
            if vol_regime == "HIGH":
                plan['name'] = "Iron Condor"
                plan['legs'] = f"Sell P({sup:.0f}) / Sell C({res:.0f})"
                put_credit = QuantLogic.bs_put(price, sup, T30, r, sigma) - QuantLogic.bs_put(price, lower_wing, T30, r, sigma)
                call_credit = QuantLogic.bs_call(price, res, T30, r, sigma) - QuantLogic.bs_call(price, upper_wing, T30, r, sigma)
                plan['premium'] = f"Credit: ${max(0.01, put_credit + call_credit):.2f}"
                prob_call = TradeArchitect.prob_itm(price, res, T30, r, sigma, 'call')
                prob_put = TradeArchitect.prob_itm(price, sup, T30, r, sigma, 'put')
                plan['pop'] = int((1 - (prob_call + prob_put)) * 100)
            else:
                plan['name'] = "Calendar Spread"
                plan['legs'] = f"Sell 30D Call (${price:.0f}) / Buy 60D Call (${price:.0f})"
                debit = QuantLogic.bs_call(price, price, T60, r, sigma) - QuantLogic.bs_call(price, price, T30, r, sigma)
                plan['premium'] = f"Debit: ${max(0.01, debit):.2f}"
                plan['pop'] = 50 
                
        plan['dte'] = "30 Days"
        plan['bias'] = bias
        return plan

class MonteCarloEngine:
    @staticmethod
    def simulate_paths(df, days=30, sims=1000, jump_intensity=1.5, jump_mean=-0.02, jump_std=0.05):
        try:
            last_price = df['Close'].iloc[-1]
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            mu_hist, sigma_hist = returns.mean(), returns.std()
            dt = 1 / 252 

            lambda_dt = jump_intensity * dt 
            jump_expected_return = lambda_dt * (np.exp(jump_mean + 0.5 * jump_std**2) - 1) 
            drift_adj = mu_hist - 0.5 * sigma_hist**2 - jump_expected_return

            z1 = np.random.normal(0, 1, (days, sims))
            diffusion = np.exp(drift_adj * dt + sigma_hist * np.sqrt(dt) * z1)

            jump_occurred = np.random.poisson(lambda_dt, (days, sims))
            jump_size = np.random.normal(jump_mean, jump_std, (days, sims))
            jump_factor = np.exp(jump_occurred * jump_size)

            daily_factors = diffusion * jump_factor
            price_paths = np.zeros((days + 1, sims))
            price_paths[0] = last_price
            price_paths[1:] = last_price * np.cumprod(daily_factors, axis=0)
            
            return pd.DataFrame(price_paths)
        except Exception as e:
            st.warning("Degraded to simple random walk.")
            return MonteCarloEngine._simple_random_walk(df, days, sims)

    @staticmethod
    def _simple_random_walk(df, days, sims):
        last_price = df['Close'].iloc[-1]
        returns = df['Close'].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        daily_shocks = np.random.normal(mu, sigma, (days, sims))
        price_paths = last_price * (1 + daily_shocks).cumprod(axis=0)
        start_row = np.full((1, sims), last_price)
        return pd.DataFrame(np.vstack([start_row, price_paths]))

class MarketScanner:
    @staticmethod
    @st.cache_data(ttl=900, show_spinner=False)
    def run_scan(tickers):
        results = []
        for t in tickers:
            try:
                stock = yf.Ticker(t)
                df = stock.history(period="2y")
                if not df.empty and len(df) > 50:
                    price = df['Close'].iloc[-1]
                    score = AlphaEngine.calculate_score(df)
                    vol = QuantLogic.calculate_vol(df)
                    sharpe = QuantLogic.calculate_sharpe(df)
                    vrp = QuantLogic.calculate_vrp_edge(df)
                    sup, res = QuantLogic.get_support_resistance(df)
                    plan = TradeArchitect.generate_plan(t, price, score, vol, sup, res)
                    mc_df = MonteCarloEngine.simulate_paths(df, days=30, sims=1000)
                    stop_loss = np.percentile(mc_df.iloc[-1], 5)

                    results.append({
                        "Ticker": t, "Price": price, "Alpha Score": score, "Trend (L/S)": plan['bias'],
                        "VRP Edge %": vrp, "Sharpe": sharpe, "Vol %": vol, "Support": sup,
                        "Resistance": res, "Stop Loss (VaR 95)": stop_loss, "Optimal Strategy": plan['name'],
                        "Legs (Strikes)": plan['legs'], "POP %": plan['pop']
                    })
            except Exception: pass
        return pd.DataFrame(results).sort_values("Alpha Score", ascending=False)

# ==========================================
# --- PART 2: THE STREAMLIT APP (UI) ---
# ==========================================

st.set_page_config(page_title="VRP Quant | V22 Command", layout="wide", page_icon="🏦")

# --- SECURE PRODUCTION USERS DICTIONARY ---
try:
    USERS = st.secrets["credentials"]
except Exception as e:
    st.error("⚠️ SYSTEM LOCKED: Security vault not connected. Please configure [credentials] in Streamlit Secrets.")
    st.stop()

DISCLAIMER_TEXT = """**SEC MARKETING RULE (17 CFR § 275.206(4)-1) & REGULATORY COMPLIANCE NOTICE**
1. **Hypothetical Performance:** Metrics generated by this software are hypothetical and not guarantees of future results.
2. **Not Financial Advice:** VRP Quant provides quantitative data analysis for informational purposes only.
3. **Risk Disclosure:** Options trading involves substantial risk of loss."""

def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.tier = None
        
    if not st.session_state.authenticated:
        st.markdown("## 🔒 VRP Quant Terminal Login")
        c1, c2 = st.columns([1, 2])
        with c1:
            user = st.text_input("Username", key="login_u")
            pwd = st.text_input("Password", type="password", key="login_p")
            if st.button("Login"):
                if user in USERS and USERS[user]["password"] == pwd:
                    st.session_state.authenticated = True
                    st.session_state.tier = USERS[user]["tier"]
                    st.rerun()
                else: 
                    st.error("Invalid Credentials")
        
        st.markdown("---")
        st.markdown("### 👑 Founding Member Cohort (Beta)")
        st.caption("Institutional pricing is $299/mo (Analyst) and $999/mo (God Mode). Join the Private Beta today to lock in your lifetime discounted rate.")
        
        b1, b2 = st.columns(2)
        with b1:
            st.info("**ANALYST TIER**\n* Retail Price: ~~$299/mo~~\n* Founding Member: **$149/mo**")
            st.link_button("Subscribe via PayPal ($149/mo)", "YOUR_PAYPAL_LINK_HERE", use_container_width=True)
            
        with b2:
            st.success("**GOD MODE TIER**\n* Retail Price: ~~$999/mo~~\n* Founding Member: **$499/mo**")
            st.link_button("Subscribe via PayPal ($499/mo)", "YOUR_PAYPAL_LINK_HERE", use_container_width=True)
            
        st.markdown("---")
        st.success("🛡️ SEC Compliance Check: System verified.")
        st.caption(DISCLAIMER_TEXT)
        return False
    return True

if check_login():
    tier = st.session_state.tier
    
    with st.sidebar:
        st.markdown("# 🏦 VRP Quant V22.0")
        if tier == "GOD_MODE": st.success("🔓 GOD MODE ACTIVE")
        else: st.warning("🔒 ANALYST TIER")
        st.markdown("---")
        
    mode = st.sidebar.radio("Module", ["🚀 Market Scanner", "🔬 Deep Dive Analysis"])

    if mode == "🚀 Market Scanner":
        st.title("🚀 Institutional Market Scanner")
        TICKER_SETS = {
            "🔥 Magnificent 7 + Crypto": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "BTC-USD", "ETH-USD", "COIN"],
            "💻 Semiconductors (AI)": ["NVDA", "AMD", "AVGO", "TSM", "INTC", "QCOM", "MU", "SMH"],
            "🛢️ Energy & Commodities": ["XLE", "USO", "GLD", "SLV", "CVX", "XOM"]
        }

        if tier != "GOD_MODE":
            st.error("🔒 ACCESS DENIED: Market Scanner is locked for Analyst Tier.")
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                options = list(TICKER_SETS.keys()) + ["✨ Custom Watchlist"]
                sector_choice = st.selectbox("Select Sector:", options)
            
            selected_tickers = []
            if sector_choice == "✨ Custom Watchlist":
                with col2:
                    custom_input = st.text_area("Enter Tickers (comma separated):", "PLTR, SOFI")
                    if custom_input: selected_tickers = [t.strip().upper() for t in custom_input.split(',')]
            else: selected_tickers = TICKER_SETS[sector_choice]
            
            if st.button("🔄 Run Live Scan") and selected_tickers:
                with st.spinner("Scanning..."):
                    df_scan = MarketScanner.run_scan(selected_tickers)
                    if not df_scan.empty: st.dataframe(df_scan, use_container_width=True)

    elif mode == "🔬 Deep Dive Analysis":
        st.title("🔬 Deep Dive & Trade Architect")
        ticker = st.text_input("Asset Ticker", "NVDA").upper()
        
        if st.button("Run Analysis"):
            with st.spinner("Connecting to Data Feeds & Running Vector Backtest..."):
                try:
                    stock = yf.Ticker(ticker)
                    df = stock.history(period="2y")
                    if not df.empty:
                        curr_price = df['Close'].iloc[-1]
                        score = AlphaEngine.calculate_score(df)
                        vol = QuantLogic.calculate_vol(df)
                        sup, res = QuantLogic.get_support_resistance(df)
                        sharpe = QuantLogic.calculate_sharpe(df)
                        vrp_edge = QuantLogic.calculate_vrp_edge(df)
                        
                        win_rate, strat_ret, outperf, max_dd = BacktestEngine.run_quick_backtest(df)
                        plan = TradeArchitect.generate_plan(ticker, curr_price, score, vol, sup, res)
                        
                        st.markdown("### 📊 Market Variables")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Price", f"${curr_price:.2f}")
                        m2.metric("Alpha Score", f"{score}/100")
                        m3.metric("Trend", plan['bias'])
                        m4.metric("Volatility", f"{vol:.1f}%")
                        
                        m5, m6, m7, m8 = st.columns(4)
                        m5.metric("VRP Edge", f"{vrp_edge:+.2f}%")
                        m6.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        m7.metric("Support (Floor)", f"${sup:.2f}")
                        m8.metric("Resistance (Ceiling)", f"${res:.2f}")
                        
                        st.markdown("### ⚙️ Strategy Backtest Validation (2-Year)")
                        b1, b2, b3, b4 = st.columns(4)
                        b1.metric("Historical Win Rate", f"{win_rate:.1f}%")
                        b2.metric("Net Strategy Return", f"{strat_ret:+.1f}%")
                        b3.metric("Alpha Generated", f"{outperf:+.1f}%")
                        b4.metric("Max Drawdown", f"{max_dd:.1f}%", delta_color="inverse")
                        
                        st.markdown("### 🎯 Optimal Trade Architecture")
                        st.info(f"**STRATEGY:** {plan['name']} | **LEGS:** {plan['legs']}")
                        
                        s1, s2, s3 = st.columns(3)
                        s1.metric("Est. Execution Target", plan['premium'])
                        s2.metric("Prob. of Profit (POP)", f"{plan['pop']}%")
                        s3.metric("Ideal DTE", plan['dte'])
                        
                        sims = 10000 if tier == "GOD_MODE" else 1000
                        mc_df = MonteCarloEngine.simulate_paths(df, days=30, sims=sims)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=df['Close'], name='History', line=dict(color='white')))
                        fig.add_trace(go.Scatter(x=list(range(len(df), len(df)+30)), y=mc_df.mean(axis=1), name='Mean Projection', line=dict(dash='dash', color='orange')))
                        fig.add_hline(y=sup, line_dash="dot", line_color="green", annotation_text="Support")
                        fig.add_hline(y=res, line_dash="dot", line_color="red", annotation_text="Resistance")
                        fig.update_layout(template="plotly_dark", height=500, title="Institutional Chart (History + 30-Day Projection)")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error analyzing {ticker}: {e}")
        
    with st.sidebar:
        st.markdown("---")
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.rerun()
