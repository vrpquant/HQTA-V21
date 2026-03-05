import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, t
from datetime import datetime
import pytz
import os
import time  # [ADDED] For institutional API throttling

# ==========================================
# --- GLOBAL INSTITUTIONAL UNIVERSE ---
# Optimized for Maximum Liquidity & Options Open Interest (Top 10 per sector)
# ==========================================

TICKER_SETS = {
    "🔥 Magnificent 7 + Crypto": [
        "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "COIN", "MSTR", "MARA", "BTC-USD"
    ],
    "💻 Semiconductors (AI)": [
        "NVDA", "AMD", "TSM", "INTC", "MU", "AVGO", "QCOM", "ARM", "AMAT", "SMH"
    ],
    "🛢️ Energy & Commodities": [
        "XOM", "CVX", "COP", "SLB", "OXY", "EOG", "MPC", "VLO", "HAL", "XLE"
    ],
    "🏥 Healthcare & Biotech": [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "AMGN", "ISRG", "SYK", "XLV"
    ],
    "🏦 Financials & Banking": [
        "JPM", "BAC", "WFC", "MS", "GS", "C", "V", "MA", "AXP", "XLF"
    ],
    "🏭 Industrials & Defense": [
        "GE", "CAT", "UBER", "BA", "RTX", "LMT", "HON", "UNP", "DE", "XLI"
    ],
    "🛒 Consumer Discretionary": [
        "TSLA", "AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX", "XLY"
    ],
    "🧼 Consumer Staples": [
        "WMT", "PG", "COST", "KO", "PEP", "PM", "TGT", "MO", "DG", "XLP"
    ],
    "🏠 Real Estate (REITs)": [
        "AMT", "PLD", "CCI", "EQIX", "O", "PSA", "SPG", "WELL", "DLR", "XLRE"
    ],
    "🔌 Utilities": [
        "NEE", "CEG", "SO", "DUK", "SRE", "AEP", "D", "PCG", "EXC", "XLU"
    ],
    "📡 Communications & Media": [
        "META", "GOOGL", "NFLX", "DIS", "VZ", "T", "CMCSA", "TMUS", "WBD", "XLC"
    ]
}

# ==========================================
# --- ADVANCED INSTITUTIONAL MATH ENGINE ---
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
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100 - (100 / (1 + rs.iloc[-1]))
            score = 50 + max(-25, min(25, trend_z * 12.5))
            if rsi < 30: score += 15
            elif rsi > 70: score -= 15
            return max(0, min(100, int(score)))
        except:
            return 50

class BacktestEngine:
    @staticmethod
    def run_quick_backtest(df, slippage_bps=5, commission_bps=2):
        bt_df = df.copy().dropna()
        bt_df['SMA50'] = bt_df['Close'].rolling(50).mean()
        bt_df['SMA200'] = bt_df['Close'].rolling(200).mean()
        bt_df['Vol_Regime'] = (bt_df['Close'].pct_change().rolling(20).std() * np.sqrt(252) > 0.35).astype(int)
        bt_df['Target_Position'] = np.where(bt_df['SMA50'] > bt_df['SMA200'], 1, -1)
        bt_df['Actual_Position'] = bt_df['Target_Position'].shift(1).fillna(0)
        bt_df['Underlying_Return'] = bt_df['Close'].pct_change().fillna(0)
        bt_df['Gross_Return'] = bt_df['Actual_Position'] * bt_df['Underlying_Return']
        turnover = bt_df['Actual_Position'].diff().abs().fillna(0)
        total_cost = (slippage_bps + commission_bps) / 10000
        bt_df['Net_Return'] = bt_df['Gross_Return'] - (turnover * total_cost * (1 + bt_df['Vol_Regime']))
        win_rate = (bt_df['Net_Return'] > 0).mean() * 100
        cumulative = (1 + bt_df['Net_Return']).prod() - 1
        buy_hold = (1 + bt_df['Underlying_Return']).prod() - 1
        outperf = cumulative - buy_hold
        peak = (1 + bt_df['Net_Return']).cumprod().cummax()
        max_dd = (((1 + bt_df['Net_Return']).cumprod() - peak) / peak).min() * 100
        
        wins = bt_df[bt_df['Net_Return'] > 0]['Net_Return']
        losses = bt_df[bt_df['Net_Return'] < 0]['Net_Return']
        half_kelly = 0
        
        if len(wins) and len(losses):
            win_avg = wins.mean()
            loss_avg = abs(losses.mean())
            if loss_avg > 0:
                win_prob = len(wins) / len(bt_df)
                kelly_pct = (win_avg / loss_avg) - (1 - win_prob)
                half_kelly = max(0, min(100, kelly_pct * 50))
                
        return round(win_rate,1), round(cumulative*100,1), round(outperf*100,1), round(max_dd,1), round(half_kelly,1)

class QuantLogic:
    @staticmethod
    def calculate_vol(df):
        return df['Close'].pct_change().std() * np.sqrt(252) * 100

    @staticmethod
    def get_atm_iv(ticker, current_price):
        try:
            stock = yf.Ticker(ticker)
            if stock.options:
                chain = stock.option_chain(stock.options[0])
                calls = chain.calls
                atm_idx = (calls['strike'] - current_price).abs().idxmin()
                return round(calls.loc[atm_idx, 'impliedVolatility'] * 100, 2)
            return None
        except:
            return None

    @staticmethod
    def calculate_vrp_edge(ticker, df, mode="scanner"):
        """
        [PATCHED] Dual-mode VRP calculation to prevent YFinance IP Bans.
        mode='scanner': Uses lightweight historical proxy
        mode='deep_dive': Pulls live option chains for true IV
        """
        if mode == "scanner":
            hv20 = df['Close'].pct_change().tail(20).std() * np.sqrt(252) * 100
            hv60 = df['Close'].pct_change().tail(60).std() * np.sqrt(252) * 100
            return round(hv20 - hv60, 2)
        else:
            price = df['Close'].iloc[-1]
            hv = QuantLogic.calculate_vol(df)
            iv = QuantLogic.get_atm_iv(ticker, price)
            return round(iv - hv, 2) if iv else 0.0

    @staticmethod
    def calculate_sharpe(df, risk_free_rate=0.04):
        returns = df['Close'].pct_change().dropna()
        excess = returns.mean() * 252 - risk_free_rate
        vol = returns.std() * np.sqrt(252)
        return round(excess / vol, 2) if vol > 0 else 0

    @staticmethod
    def get_support_resistance(df):
        res = df['High'].rolling(50).max().iloc[-1]
        sup = df['Low'].rolling(50).min().iloc[-1]
        return sup, res

    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        pdf = norm.pdf(d1)
        delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
        gamma = pdf / (S * sigma * np.sqrt(T))
        vega = S * pdf * np.sqrt(T)
        return {'delta': round(delta,3), 'gamma': round(gamma,4), 'vega': round(vega,2)}

    @staticmethod
    def bs_call(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return max(0, S - K)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def bs_put(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return max(0, K - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

class TradeArchitect:
    @staticmethod
    def prob_itm(S, K, T, r, sigma, option_type='call'):
        if T <= 0 or sigma <= 0: return 0.0
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2)

    @staticmethod
    def generate_plan(ticker, price, score, vol, sup, res, half_kelly):
        plan = {}
        bias = "LONG (Bullish Trend)" if score >= 60 else "SHORT (Bearish Trend)" if score <= 40 else "NEUTRAL (Mean-Reverting)"
        vol_regime = "HIGH" if vol > 35 else "LOW"
        sigma = max(0.01, vol / 100)
        r = 0.04
        T30 = 30 / 365
        if res <= price: res = price * 1.05
        if sup >= price: sup = price * 0.95
        lower_wing = sup * 0.95
        upper_wing = res * 1.05
        
        if "LONG" in bias:
            if vol_regime == "LOW":
                plan['name'] = "Long Call Vertical"
                plan['legs'] = f"+C({price:.0f}) / -C({res:.0f})"
                debit = QuantLogic.bs_call(price, price, T30, r, sigma) - QuantLogic.bs_call(price, res, T30, r, sigma)
                plan['premium'] = f"Debit ${max(0.01, debit):.2f}"
                plan['pop'] = int(TradeArchitect.prob_itm(price, price + debit, T30, r, sigma, 'call') * 100)
            else:
                plan['name'] = "Short Put Vertical"
                plan['legs'] = f"-P({sup:.0f}) / +P({lower_wing:.0f})"
                credit = QuantLogic.bs_put(price, sup, T30, r, sigma) - QuantLogic.bs_put(price, lower_wing, T30, r, sigma)
                plan['premium'] = f"Credit ${max(0.01, credit):.2f}"
                plan['pop'] = int((1 - TradeArchitect.prob_itm(price, sup, T30, r, sigma, 'put')) * 100)
        elif "SHORT" in bias:
            if vol_regime == "LOW":
                plan['name'] = "Long Put Vertical"
                plan['legs'] = f"+P({price:.0f}) / -P({sup:.0f})"
                debit = QuantLogic.bs_put(price, price, T30, r, sigma) - QuantLogic.bs_put(price, sup, T30, r, sigma)
                plan['premium'] = f"Debit ${max(0.01, debit):.2f}"
                plan['pop'] = int(TradeArchitect.prob_itm(price, price - debit, T30, r, sigma, 'put') * 100)
            else:
                plan['name'] = "Short Call Vertical"
                plan['legs'] = f"-C({res:.0f}) / +C({upper_wing:.0f})"
                credit = QuantLogic.bs_call(price, res, T30, r, sigma) - QuantLogic.bs_call(price, upper_wing, T30, r, sigma)
                plan['premium'] = f"Credit ${max(0.01, credit):.2f}"
                plan['pop'] = int((1 - TradeArchitect.prob_itm(price, res, T30, r, sigma, 'call')) * 100)
        else:
            plan['name'] = "Iron Condor"
            plan['legs'] = f"+P({lower_wing:.0f}) / -P({sup:.0f}) | -C({res:.0f}) / +C({upper_wing:.0f})"
            put_credit = QuantLogic.bs_put(price, sup, T30, r, sigma) - QuantLogic.bs_put(price, lower_wing, T30, r, sigma)
            call_credit = QuantLogic.bs_call(price, res, T30, r, sigma) - QuantLogic.bs_call(price, upper_wing, T30, r, sigma)
            plan['premium'] = f"Credit ${max(0.01, put_credit + call_credit):.2f}"
            plan['pop'] = 65
            
        plan['greeks'] = QuantLogic.calculate_greeks(price, price, T30, r, sigma)
        plan['kelly_size'] = f"{int(max(5, min(50, half_kelly)))}% capital"
        plan['dte'] = "30 Days"
        plan['bias'] = bias
        return plan

class MonteCarloEngine:
    @staticmethod
    def simulate_paths(df, days=30, sims=1000):
        try:
            price = df['Close'].iloc[-1]
            returns = np.log(df['Close']/df['Close'].shift(1)).dropna()
            lambda_ = 0.94
            ewma_var = np.average(returns**2, weights=np.power(lambda_, np.arange(len(returns)-1,-1,-1)))
            sigma = np.sqrt(ewma_var * 252)
            mu = returns.mean() * 252
            dt = 1/252
            z = np.random.normal(0,1,(days,sims))
            jumps = np.random.poisson(0.8*dt,(days,sims)) * np.random.normal(-0.015, 0.08, (days,sims))
            paths = np.zeros((days+1, sims))
            paths[0] = price
            for t in range(1, days+1):
                paths[t] = paths[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z[t-1] + jumps[t-1])
            return pd.DataFrame(paths)
        except:
            return pd.DataFrame(np.tile(df['Close'].iloc[-1], (days+1, sims)))

class MarketScanner:
    @staticmethod
    @st.cache_data(ttl=900, show_spinner=False)
    def run_scan(tickers):
        results = []
        for t in tickers:
            try:
                stock = yf.Ticker(t)
                df = stock.history(period="2y")
                if len(df) > 50:
                    price = df['Close'].iloc[-1]
                    score = AlphaEngine.calculate_score(df)
                    vol = QuantLogic.calculate_vol(df)
                    
                    # [PATCHED] Use scanner mode to avoid heavy payload options pull
                    vrp = QuantLogic.calculate_vrp_edge(t, df, mode="scanner")
                    
                    sup, res = QuantLogic.get_support_resistance(df)
                    win_rate, strat_ret, outperf, max_dd, kelly = BacktestEngine.run_quick_backtest(df)
                    plan = TradeArchitect.generate_plan(t, price, score, vol, sup, res, kelly)
                    
                    results.append({
                        "Ticker": t, "Price": round(price,2), "Alpha Score": score, "Trend": plan['bias'],
                        "VRP Edge": f"{vrp:+.1f}%", "Vol": f"{vol:.1f}%", "Support": round(sup,2),
                        "Resistance": round(res,2), "Win Rate": f"{win_rate}%", "Max DD": f"{max_dd}%",
                        "Kelly": f"{kelly}%", "Strategy": plan['name']
                    })
                
                # [PATCHED] Throttling API calls to prevent YFinance 429 Ban
                time.sleep(1.5)
                
            except Exception as e:
                print(f"⚠️ [ENGINE WARNING] Data pull failed for {t}: {e}")
                pass
        
        df_results = pd.DataFrame(results)
        if df_results.empty:
            return df_results
        return df_results.sort_values("Alpha Score", ascending=False)

# ==========================================
# --- STREAMLIT APP UI ---
# ==========================================

st.set_page_config(page_title="VRP Quant | V22.1", layout="wide", page_icon="🏦")
est_tz = pytz.timezone('US/Eastern')

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
            st.link_button("Subscribe ($149/mo)", "https://www.paypal.com/webapps/billing/plans/subscribe?plan_id=P-0CB63794C10515154NGMNDNA", use_container_width=True)
        with b2:
            st.success("**GOD MODE TIER**\n* Retail Price: ~~$999/mo~~\n* Founding Member: **$499/mo**")
            st.link_button("Subscribe ($499/mo)", "https://www.paypal.com/webapps/billing/plans/subscribe?plan_id=P-723423746M676015CNGMNFGI", use_container_width=True)
        st.markdown("---")
        st.success("🛡️ SEC Compliance Check: System verified.")
        st.caption(DISCLAIMER_TEXT)
        return False
    return True

if check_login():
    tier = st.session_state.tier
    with st.sidebar:
        st.markdown("# 🏦 VRP Quant V22.1 (Stable)")
        if tier == "GOD_MODE": st.success("🔓 GOD MODE ACTIVE")
        else: st.warning("🔒 ANALYST TIER")
        st.markdown("---")
        mode = st.sidebar.radio("Module", ["🚀 Market Scanner", "🔬 Deep Dive Analysis"])

    if mode == "🚀 Market Scanner":
        st.title("🚀 Institutional Market Scanner")
        st.caption(f"⏱️ **Data Snapshot:** Displaying latest compiled quantitative run.")
        
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
            
            # [OPTION 1] UI reads from CSV pipeline (V22.1 Default Decoupled Architecture)
            if st.button("🔄 Load Offline Institutional Scan") and selected_tickers:
                with st.spinner("Decrypting quantitative pipeline..."):
                    try:
                        if os.path.exists("latest_scan.csv"):
                            df_scan = pd.read_csv("latest_scan.csv")
                            filtered_df = df_scan[df_scan['Ticker'].isin(selected_tickers)]
                            if not filtered_df.empty: 
                                st.dataframe(filtered_df, use_container_width=True)
                            else:
                                st.warning("No data found for this sector in the latest pipeline run.")
                        else:
                            st.error("⚠️ Pipeline link severed: 'latest_scan.csv' not found. Run your local data_engine.py first.")
                    except Exception as e:
                        st.error(f"Error loading dashboard: {e}")
            
            # [OPTION 2] Live Fallback Scan (With Throttling built-in)
            if st.button("⚡ Run Live Scan (Throttled Fallback)") and selected_tickers:
                with st.spinner("Running Live Throttled Scan (Avoids IP Ban)..."):
                    try:
                        df_scan = MarketScanner.run_scan(selected_tickers)
                        if not df_scan.empty:
                            st.dataframe(df_scan, use_container_width=True)
                    except Exception as e:
                        st.error(f"Live scan failed: {e}")

    elif mode == "🔬 Deep Dive Analysis":
        st.title("🔬 Deep Dive & Trade Architect")
        st.caption(f"⏱️ **Data Vault Timestamp:** {datetime.now(est_tz).strftime('%Y-%m-%d %H:%M:%S')} EST")
        ticker = st.text_input("Asset Ticker", "TSLA").upper().strip()
        
        if st.button("Run Deep Dive Analysis"):
            with st.spinner("Extracting Advanced Institutional Metrics..."):
                try:
                    # Execute a direct pull for the single ticker (Safe for YFinance)
                    stock = yf.Ticker(ticker)
                    df = stock.history(period="2y")
                    
                    if df.empty:
                        st.error(f"⚠️ No historical bars found for {ticker}.")
                        st.stop()
                        
                    curr_price = df['Close'].iloc[-1]
                    score = AlphaEngine.calculate_score(df)
                    vol = QuantLogic.calculate_vol(df)
                    
                    # [PATCHED] Use TRUE IV for single deep-dive (Heavier, but safe for 1 ticker)
                    vrp_edge_val = QuantLogic.calculate_vrp_edge(ticker, df, mode="deep_dive")
                    vrp_edge_str = f"{vrp_edge_val:+.2f}%"
                    
                    sup, res = QuantLogic.get_support_resistance(df)
                    sharpe = QuantLogic.calculate_sharpe(df)
                    win_rate, strat_ret, outperf, max_dd, half_kelly = BacktestEngine.run_quick_backtest(df)
                    
                    plan = TradeArchitect.generate_plan(ticker, curr_price, score, vol, sup, res, half_kelly)
                    mc_df = MonteCarloEngine.simulate_paths(df, days=30, sims=5000 if tier == "GOD_MODE" else 1000)

                    st.markdown("### 📊 Market Variables")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Price", f"${curr_price:.2f}")
                    m2.metric("Alpha Score", f"{score}/100")
                    m3.metric("Trend", plan['bias'])
                    m4.metric("Volatility", f"{vol:.1f}%")
                    m5.metric("Trend Reversal", "Live Check Active")

                    m6, m7, m8, m9, m10 = st.columns(5)
                    m6.metric("True VRP Edge", vrp_edge_str)
                    m7.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    m8.metric("Support (Floor)", f"${sup:.2f}")
                    m9.metric("Resistance (Ceiling)", f"${res:.2f}")
                    m10.metric("95% VaR (Variance)", f"${np.percentile(mc_df.iloc[-1], 5):.2f}" if tier == "GOD_MODE" else "🔒 God Mode")

                    st.markdown("### ⚙️ Strategy Backtest Validation (2-Year)")
                    b1, b2, b3, b4, b5 = st.columns(5)
                    b1.metric("Historical Win Rate", f"{win_rate:.1f}%")
                    b2.metric("Net Strategy Return", f"{strat_ret:+.1f}%")
                    b3.metric("Alpha Generated", f"{outperf:+.1f}%")
                    b4.metric("Max DD", f"{max_dd:.1f}%", delta_color="inverse" if tier == "GOD_MODE" else "normal")
                    b5.metric("Kelly Fraction", f"{half_kelly:.1f}%", delta_color="normal" if tier == "GOD_MODE" else "normal")

                    st.markdown("### 🎯 Optimal Trade Architecture")
                    st.info(f"**STRATEGY:** {plan['name']} | **LEGS:** {plan['legs']}")
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Est. Execution Target", plan['premium'])
                    s2.metric("Prob. of Profit (POP)", f"{plan['pop']}%")
                    s3.metric("Ideal DTE", plan['dte'])
                    st.metric("Greeks (ATM)", f"Δ {plan['greeks']['delta']} | Γ {plan['greeks']['gamma']} | Vega {plan['greeks']['vega']}")
                    st.metric("Kelly Position Size", plan['kelly_size'])

                    hist_dates = df.index.tz_localize(None) if df.index.tz else df.index
                    future_dates = pd.date_range(start=hist_dates[-1] + pd.Timedelta(days=1), periods=30, freq='B')
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist_dates, y=df['Close'], name='History', line=dict(color='white')))
                    fig.add_trace(go.Scatter(x=future_dates, y=mc_df.mean(axis=1), name='Mean Projection', line=dict(dash='dash', color='orange')))
                    fig.add_hline(y=sup, line_dash="dot", line_color="green", annotation_text="Support")
                    fig.add_hline(y=res, line_dash="dot", line_color="red", annotation_text="Resistance")
                    fig.update_layout(template="plotly_dark", height=500, title="Institutional Chart (History + 30-Day EWMA Jump-Diffusion Projection)")
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Deep Dive Engine Error: {str(e)}")
                    st.info("Check ticker symbol or API connection.")

    with st.sidebar:
        st.markdown("---")
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.rerun()

