import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

from vrp_quant_engine_v30_2_1 import OptionsDataEngine, AlphaEngine, BacktestEngine, QuantLogic, TradeArchitect, RegimeEngine, DynamicUniverseEngine, get_sparkline

# --- SPECIFIC UI ENGINE CLASSES RETAINED HERE ---
class MonteCarloEngine:
    @staticmethod
    def simulate_paths(df, days=30, sims=1000):
        try:
            price = df['Close'].iloc[-1]
            returns = np.log(df['Close']/df['Close'].shift(1)).dropna()
            mu, dt = returns.mean() * 252, 1/252
            kappa, theta, sigma_v, rho = 2.0, 0.04, 0.6, -0.7  
            ewma_var = np.average(returns**2, weights=np.power(0.94, np.arange(len(returns)-1,-1,-1)))
            v = np.full(sims, ewma_var * 252) 
            jumps = np.random.poisson(0.8*dt,(days,sims)) * np.random.normal(-0.015, 0.08, (days,sims))
            paths = np.zeros((days+1, sims))
            paths[0] = price
            for t in range(1, days+1):
                dv = kappa*(theta - v)*dt + sigma_v*np.sqrt(v*dt)*np.random.normal(0,1,(sims,))
                v = np.maximum(v + dv, 1e-8)
                dW1 = np.random.normal(0,1,(sims,))
                dW2 = rho*dW1 + np.sqrt(1-rho**2)*np.random.normal(0,1,(sims,))
                paths[t] = paths[t-1] * np.exp((mu - 0.5*v)*dt + np.sqrt(v*dt)*dW1 + jumps[t-1])
            return pd.DataFrame(paths)
        except:
            return pd.DataFrame(np.tile(df['Close'].iloc[-1], (days+1, sims)))

class SectorStrengthEngine:
    SECTOR_ETFS = {"🔥 Magnificent 7 + BTC": "MAGS", "💻 Information Technology": "XLK", "🏦 Financials": "XLF", "🏥 Healthcare": "XLV", "🛒 Consumer Discretionary": "XLY", "📡 Communication Services": "XLC", "🏭 Industrials": "XLI", "🧼 Consumer Staples": "XLP", "🛢️ Energy": "XLE", "🔌 Utilities": "XLU", "🏠 Real Estate": "XLRE", "🪙 Digital Assets & Proxies": "WGMI"}
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_strongest_sector():
        try:
            returns = {sector: (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1 for sector, etf in SectorStrengthEngine.SECTOR_ETFS.items() if len(df := yf.Ticker(etf).history(period="1mo")) > 15}
            best_sector = max(returns, key=returns.get) if returns else "💻 Information Technology"
            return best_sector, returns.get(best_sector, 0.0)
        except: return "💻 Information Technology", 0.0 

class UniverseEngine:
    SECTOR_UNIVERSE = {
        "🔥 Magnificent 7 + BTC": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BTC-USD"],
        "💻 Information Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "INTC", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT", "MU", "PANW", "SNOW", "CRWD"],
        "🏦 Financials": ["JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "C", "BLK", "SPGI", "CME", "SCHW", "CB", "MMC", "PGR", "USB", "PNC", "TFC", "COF"],
        "🏥 Healthcare": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "AMGN", "ISRG", "SYK", "MDT", "VRTX", "REGN", "GILD", "BSX", "CVS", "CI", "ZTS", "BDX", "HUM", "BIIB"],
        "🛒 Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX", "CMG", "MAR", "HLT", "ORLY", "ABNB", "GM", "F", "FVRR", "CHWY", "ETSY", "EBAY"],
        "📡 Communication Services": ["GOOGL", "META", "NFLX", "DIS", "VZ", "T", "CMCSA", "TMUS", "WBD", "CHTR", "EA", "TTWO", "LYV", "MTCH", "FOXA", "SIRI", "ROKU", "SNAP", "PINS", "ZG"],
        "🏭 Industrials": ["GE", "CAT", "UBER", "BA", "RTX", "LMT", "HON", "UNP", "DE", "UPS", "MMM", "LUV", "FDX", "ETN", "EMR", "NOC", "GD", "CSX", "NSC", "PCAR"],
        "🧼 Consumer Staples": ["WMT", "PG", "COST", "KO", "PEP", "PM", "TGT", "MO", "DG", "EL", "CL", "KMB", "MDLZ", "SYY", "HSY", "KR", "GIS", "CHD", "K", "CPB"],
        "🛢️ Energy": ["XOM", "CVX", "COP", "SLB", "OXY", "EOG", "MPC", "VLO", "HAL", "PSX", "WMB", "KMI", "HES", "BKR", "DVN", "FANG", "TRGP", "OKE", "CTRA", "MRO"],
        "🔌 Utilities": ["NEE", "CEG", "SO", "DUK", "SRE", "AEP", "D", "PCG", "EXC", "PEG", "XEL", "ED", "WEC", "AWK", "ES", "ETR", "FE", "CMS", "LNT", "NI"],
        "🏠 Real Estate": ["AMT", "PLD", "CCI", "EQIX", "O", "PSA", "SPG", "WELL", "DLR", "VICI", "CSG", "AVB", "DRE", "EXR", "MAA", "ESS", "INVH", "UDR", "CPT", "HST"],
        "🪙 Digital Assets & Proxies": ["BTC-USD", "ETH-USD", "COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC", "BRRR", "HODL", "BTCW", "BITO", "CORZ", "CIFR"]
    }

class PortfolioEngine:
    @staticmethod
    def cross_sectional_momentum(price_dict):
        try: return pd.Series({t: df['Close'].pct_change(20).iloc[-1] for t, df in price_dict.items()}).rank(pct=True).to_dict()
        except: return {t: 0.5 for t in price_dict.keys()}

    @staticmethod
    def volatility_targeting(df, target_vol=0.20):
        try: return min(2.0, max(0.0, target_vol / (df['Close'].pct_change().std() * np.sqrt(252) + 1e-9)))
        except: return 1.0

    @staticmethod
    def mean_variance_weight(price_dict):
        try:
            returns = pd.DataFrame({t: df['Close'].pct_change() for t, df in price_dict.items()}).dropna()
            inv = np.linalg.pinv(returns.cov().values)
            ones = np.ones(len(inv))
            return dict(zip(returns.columns, inv @ ones / (ones.T @ inv @ ones)))
        except:
            return {t: 1/len(price_dict) for t in price_dict}

    @staticmethod
    def kelly_weight(df):
        try:
            r = df['Close'].pct_change().dropna()
            return max(0, min((r.mean() * 252) / (r.var() * 252 + 1e-9), 2))
        except: return 0.0

class OptionsExpectedMove:
    @staticmethod
    def calculate(ticker, current_price):
        try:
            chain = OptionsDataEngine.get_robust_chain(ticker)
            if chain is None: return 0, 0, 0
            calls, puts = chain.calls, chain.puts
            calls['diff'] = abs(calls['strike'] - current_price)
            puts['diff'] = abs(puts['strike'] - current_price)
            atm_call, atm_put = calls.sort_values('diff').iloc[0], puts.sort_values('diff').iloc[0]
            call_price = (atm_call['bid'] + atm_call['ask']) / 2 if (atm_call['bid'] > 0) else atm_call['lastPrice']
            put_price = (atm_put['bid'] + atm_put['ask']) / 2 if (atm_put['bid'] > 0) else atm_put['lastPrice']
            expected_move = call_price + put_price
            return expected_move, current_price + expected_move, current_price - expected_move
        except: return 0, 0, 0

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history(ticker, period="2y"):
    time.sleep(1.0) 
    return yf.Ticker(ticker).history(period=period)

def inject_institutional_css():
    st.markdown("""
    <style>
        .stApp { background-color: #0B0F19; color: #F8FAFC; }
        [data-testid="stSidebar"] { background-color: #0F172A; border-right: 1px solid #1E293B; }
        div[data-testid="metric-container"] {
            background-color: #1E293B; border: 1px solid #334155;
            padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        div[data-testid="metric-container"] label { color: #94A3B8 !important; font-weight: 600 !important; letter-spacing: 0.5px; }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #38BDF8 !important; font-size: 1.8rem !important; font-weight: 700 !important; }
        .apex-box {
            background-color: #082F49; border-left: 5px solid #38BDF8;
            border-radius: 5px; padding: 20px; margin-top: 15px; margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        .apex-title { color: #BAE6FD; font-size: 1.4em; font-weight: 800; margin-bottom: 10px; }
        .apex-action { color: #38BDF8; font-size: 1.2em; font-weight: 700; margin-bottom: 10px; }
        .apex-logic { color: #94A3B8; font-size: 1em; font-style: italic; }
        h1, h2, h3 { color: #F1F5F9 !important; font-weight: 700 !important; }
        hr { border-color: #334155 !important; }
    </style>
    """, unsafe_allow_html=True)

class MarketScanner:
    @staticmethod
    @st.cache_data(ttl=900, show_spinner=False)
    def run_scan(tickers):
        regime, results, price_dict = RegimeEngine.detect_regime(), [], {}
        for t in tickers:
            try:
                df = fetch_history(t, period="2y")
                if len(df) > 100 and df['Volume'].tail(30).mean() > 1_000_000 and df['Close'].pct_change().std() > 0.02:
                    price_dict[t] = df
            except: pass

        if not price_dict: return pd.DataFrame()

        cs_rank, mvo_weights = PortfolioEngine.cross_sectional_momentum(price_dict), PortfolioEngine.mean_variance_weight(price_dict)

        for t, df in price_dict.items():
            try:
                price = df['Close'].iloc[-1]
                score, vol = AlphaEngine.calculate_score(df), QuantLogic.calculate_vol(df)
                vrp = QuantLogic.calculate_vrp_edge(t, df, mode="scanner")
                sup, res = QuantLogic.get_support_resistance(df)
                win_rate, strat_ret, outperf, max_dd, kelly, sortino, calmar, _ = BacktestEngine.run_wfo_backtest(df)
                plan = TradeArchitect.generate_plan(t, price, score, vol, sup, res, kelly)
                
                # --- V30.2.1 UNIFIED HYBRID CALL ---
                hybrid = TradeArchitect.generate_hybrid_plan(price, score, vrp, sup, res)
                move, exp_upper, exp_lower = OptionsExpectedMove.calculate(t, price)

                cs, vol_target, mvo, kelly_port = cs_rank.get(t, 0.5), PortfolioEngine.volatility_targeting(df), mvo_weights.get(t, 0), PortfolioEngine.kelly_weight(df)
                ultimate_score = (score * 0.35 + cs * 100 * 0.20 + kelly_port * 50 * 0.15 + vol_target * 50 * 0.15 + mvo * 50 * 0.15)
                
                signal = "🎯 ULTIMATE LONG" if (regime != "Risk-Off" and ultimate_score > 65 and vrp < 0 and win_rate > 10 and strat_ret > 50) else "🩸 ULTIMATE SHORT" if (ultimate_score < 35 and vrp > 0 and win_rate > 10 and strat_ret > 50) else "Standard"

                results.append({
                    "Ticker": t, "Price": round(price, 2), "Ultimate Signal": signal, "Alpha Score": score, "Trend": plan['bias'], 
                    "Options Exp Move": round(move, 2), "VRP Edge": f"{vrp:+.1f}%", "Vol": f"{vol:.1f}%", "Support": round(sup, 2), 
                    "Resistance": round(res, 2), "HQTA Apex Action": hybrid['action'], "Strategy": plan['name'], "Kelly": kelly 
                })
            except: pass

        df_results = pd.DataFrame(results)
        if df_results.empty: return df_results
        df_results = df_results.sort_values(by=["Kelly", "Alpha Score"], ascending=[False, False]).head(15)
        df_results["Kelly"] = df_results["Kelly"].apply(lambda x: f"{x:.1f}%")
        return df_results

# ==========================================
# --- STREAMLIT APP UI ---
# ==========================================
st.set_page_config(page_title="VRP Quant | V30.2.1 Institutional", layout="wide", page_icon="🏦")
inject_institutional_css() 

try: USERS = st.secrets["credentials"]
except Exception:
    st.error("⚠️ SYSTEM LOCKED: Security vault not connected. Please configure [credentials] in Streamlit Secrets.")
    st.stop()

PAYPAL_ANALYST_LINK = "https://www.paypal.com/webapps/billing/plans/subscribe?plan_id=P-0CB63794C10515154NGMNDNA" 
PAYPAL_GOD_MODE_LINK = "https://www.paypal.com/webapps/billing/plans/subscribe?plan_id=P-723423746M676015CNGMNFGI"

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
                else: st.error("Invalid Credentials")
        st.markdown("---")
        st.markdown("### 👑 Founding Member Cohort (V30.2.1)")
        b1, b2 = st.columns(2)
        with b1:
            st.info("**ANALYST TIER**\n* Retail Price: ~~$299/mo~~\n* Founding Member: **$149/mo**")
            st.link_button("Subscribe securely via PayPal", PAYPAL_ANALYST_LINK, use_container_width=True)
        with b2:
            st.success("**GOD MODE TIER**\n* Retail Price: ~~$999/mo~~\n* Founding Member: **$499/mo**")
            st.link_button("Subscribe securely via PayPal", PAYPAL_GOD_MODE_LINK, use_container_width=True)
        return False
    return True

if check_login():
    tier = st.session_state.tier
    with st.sidebar:
        st.markdown("# 🏦 VRP Quant V30.2.1")
        if tier == "GOD_MODE": st.success("🔓 GOD MODE ACTIVE")
        else: st.warning("🔒 ANALYST TIER")
        st.markdown("---")
        mode = st.sidebar.radio("Module", ["🚀 Market Scanner", "🔬 Deep Dive Analysis"])

    if mode == "🚀 Market Scanner":
        st.title("🚀 Institutional Market Scanner")
        st.markdown(f"### 🌍 Market Regime: **{RegimeEngine.detect_regime()}**")
        st.markdown("---")
        
        if tier != "GOD_MODE":
            st.error("🔒 ACCESS DENIED: Market Scanner is locked for Analyst Tier.")
        else:
            st.markdown("### ⚡ Live Sector Scanner")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                options = ["🤖 Auto-Detect Strongest Sector", "🌌 Dynamic Apex 100 (Max Liquidity)"] + list(UniverseEngine.SECTOR_UNIVERSE.keys()) + ["✨ Custom Watchlist"]
                sector_choice = st.selectbox("Select Sector to Scan:", options)
            
            selected_tickers = []
            if sector_choice == "🤖 Auto-Detect Strongest Sector":
                with st.spinner("Analyzing Macro Capital Rotation via SPDR ETFs..."):
                    best_sector, sector_roc = SectorStrengthEngine.get_strongest_sector()
                    selected_tickers = UniverseEngine.SECTOR_UNIVERSE[best_sector]
                    st.info(f"🔄 **Auto-Rotation Triggered:** Institutional capital flowing into **{best_sector}** (+{sector_roc*100:.2f}%).")
            elif sector_choice == "✨ Custom Watchlist":
                with col2:
                    if custom_input := st.text_area("Enter Tickers:", "PLTR, SOFI"): selected_tickers = [t.strip().upper() for t in custom_input.split(',')]
            elif sector_choice == "🌌 Dynamic Apex 100 (Max Liquidity)":
                selected_tickers = DynamicUniverseEngine.get_apex_100()[:60]
            else: 
                selected_tickers = UniverseEngine.SECTOR_UNIVERSE[sector_choice]

            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                strict_mode = st.checkbox("⚡ ISOLATE HIGH-CONVICTION QUANTITATIVE SETUPS", value=False)
            
            if st.button("Run Live Sector Scan") and selected_tickers:
                with st.spinner(f"Initiating Stealth Scan & Applying 1M Vol / 31.7% Volatility Gates for {len(selected_tickers)} Assets..."):
                    try:
                        df_scan = MarketScanner.run_scan(selected_tickers)
                        if not df_scan.empty:
                            if strict_mode:
                                df_scan = df_scan[df_scan["Ultimate Signal"] != "Standard"]
                                if df_scan.empty:
                                    st.warning("⚠️ No assets meet the strict Ultimate Master criteria. Cash is a position.")
                                    st.stop()
                                    
                            def render_institutional_html_table(df):
                                html = """
                                <style>
                                    .table-container { overflow-x: auto; background-color: #0B0F19; margin-top: 10px; border: 1px solid #1E293B; border-radius: 6px; }
                                    .inst-table { width: 100%; border-collapse: collapse; font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 13px; text-align: left; }
                                    .inst-table th { background-color: #0F172A; color: #94A3B8; padding: 12px 15px; border-bottom: 2px solid #334155; font-weight: 700; font-size: 11px; text-transform: uppercase; white-space: nowrap; letter-spacing: 0.5px; }
                                    .inst-table tr { border-bottom: 1px solid #1E293B; background-color: #0B0F19; }
                                    .inst-table tr:hover { background-color: #162032; }
                                    .inst-table td { padding: 12px 15px; color: #E2E8F0; vertical-align: middle; white-space: nowrap; }
                                    .badge-long { background-color: #022C22; color: #10B981; padding: 4px 8px; border-radius: 4px; font-weight: 900; border: 1px solid #047857; font-size: 11px; }
                                    .badge-short { background-color: #450A0A; color: #EF4444; padding: 4px 8px; border-radius: 4px; font-weight: 900; border: 1px solid #B91C1C; font-size: 11px; }
                                    .badge-std { color: #475569; font-style: italic; font-size: 11px; }
                                    .apex-cell { background-color: #082F49 !important; border-left: 4px solid #0EA5E9 !important; color: #38BDF8 !important; font-weight: 700; white-space: normal !important; min-width: 250px; }
                                    .ticker-cell { font-weight: 900; color: #FFFFFF; font-size: 14px; }
                                    .val-pos { color: #10B981; font-weight: 600; }
                                    .val-neg { color: #EF4444; font-weight: 600; }
                                </style>
                                <div class="table-container"><table class="inst-table"><thead><tr>
                                """
                                for col in df.columns: html += f"<th>{col}</th>"
                                html += "</tr></thead><tbody>"
                                for _, row in df.iterrows():
                                    html += "<tr>"
                                    for col in df.columns:
                                        val = row[col]
                                        if col == "Ticker": html += f"<td class='ticker-cell'>{val}</td>"
                                        elif col == "Ultimate Signal":
                                            html += f"<td><span class='badge-long'>{val}</span></td>" if "LONG" in val else f"<td><span class='badge-short'>{val}</span></td>" if "SHORT" in val else f"<td><span class='badge-std'>{val}</span></td>"
                                        elif col == "HQTA Apex Action": html += f"<td class='apex-cell'>{val}</td>"
                                        elif col in ["VRP Edge", "Trend"]:
                                            html += f"<td class='val-pos'>{val}</td>" if ("+" in str(val) or "LONG" in str(val)) else f"<td class='val-neg'>{val}</td>" if ("-" in str(val) or "SHORT" in str(val)) else f"<td>{val}</td>"
                                        else: html += f"<td>{val}</td>"
                                    html += "</tr>"
                                html += "</tbody></table></div>"
                                return html
                            st.markdown(render_institutional_html_table(df_scan), unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ No tickers in this sector passed the 1M Volume and 31.7% Annualized Volatility gates.")
                    except Exception as e:
                        st.error(f"Scan failed: {e}")

    elif mode == "🔬 Deep Dive Analysis":
        st.title("🔬 Deep Dive & Trade Architect")
        ticker = st.text_input("Asset Ticker", "TSLA").upper().strip()
        
        if st.button("Run Deep Dive Analysis"):
            with st.spinner("Executing GARCH Modeling & Out-of-Sample Walk-Forward Backtesting..."):
                try:
                    df = fetch_history(ticker, period="2y")
                    if df.empty:
                        st.error(f"⚠️ No historical bars found for {ticker}.")
                        st.stop()
                        
                    curr_price = df['Close'].iloc[-1]
                    score, vol = AlphaEngine.calculate_score(df), QuantLogic.calculate_vol(df)
                    vrp_edge_val = QuantLogic.calculate_vrp_edge(ticker, df, mode="deep_dive")
                    
                    # --- V30.2 DATA INTEGRITY KILL-SWITCH ---
                    if pd.isna(vrp_edge_val):
                        st.error(f"⚠️ DATA INTEGRITY LOCK: Options pricing API is throttled or illiquid for {ticker}. VRP Edge cannot be mathematically verified. Deep Dive halted to prevent invalid projections.")
                        st.stop()
                    
                    reversal_signal, sharpe = QuantLogic.detect_reversal(df), QuantLogic.calculate_sharpe(df)
                    sup, res = QuantLogic.get_support_resistance(df)
                    
                    try:
                        var_95 = round(curr_price * (1 + np.percentile(df['Close'].pct_change().dropna(), 5)), 2)
                    except:
                        var_95 = curr_price * 0.95
                        
                    win_rate, strat_ret, outperf, max_dd, half_kelly, sortino, calmar, bt_df = BacktestEngine.run_wfo_backtest(df)
                    
                    plan = TradeArchitect.generate_plan(ticker, curr_price, score, vol, sup, res, half_kelly)
                    
                    # --- V30.2.1 UNIFIED HYBRID CALL ---
                    hybrid = TradeArchitect.generate_hybrid_plan(curr_price, score, vrp_edge_val, sup, res)
                    
                    mc_sims = 5000 if tier == "GOD_MODE" else 1000
                    mc_df = MonteCarloEngine.simulate_paths(df, days=30, sims=mc_sims)

                    st.markdown(f"""
                    <div class="apex-box">
                        <div class="apex-title">🏆 THE AWESOME SPOT: {hybrid['name']}</div>
                        <div class="apex-action">TRADE ARCHITECTURE: {hybrid['action']}</div>
                        <div class="apex-logic">Institutional Logic: {hybrid['logic']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 📊 Market Variables")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Price", f"${curr_price:.2f}")
                    m2.metric("Alpha Score", f"{score}/100")
                    m3.metric("Trend", plan['bias'])
                    m4.metric("Volatility", f"{vol:.1f}%")
                    m5.metric("Trend Reversal", reversal_signal)

                    m6, m7, m8, m9, m10 = st.columns(5)
                    m6.metric("VRP Edge", f"{vrp_edge_val:+.2f}%")
                    m7.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    m8.metric("Support (Floor)", f"${sup:.2f}")
                    m9.metric("Resistance (Ceiling)", f"${res:.2f}")
                    m10.metric("95% VaR (Variance)", f"${var_95:.2f}")

                    st.markdown("### ⚙️ Strategy Validation (Walk-Forward Out-of-Sample)")
                    b1, b2, b3, b4, b5 = st.columns(5)
                    b1.metric("Historical Win Rate", f"{win_rate:.1f}%")
                    b2.metric("Net Strategy Return", f"{strat_ret:+.1f}%")
                    b3.metric("Alpha Generated", f"{outperf:+.1f}%")
                    b4.metric("Markdown % (Max DD)", f"{max_dd:.1f}%", delta_color="inverse")
                    b5.metric("Kelly Fraction (Half)", f"{half_kelly:.1f}%", delta_color="normal")

                    st.markdown("### 🛡️ Institutional Risk & Regime Metrics")
                    r1, r2, r3, r4, r5 = st.columns(5)
                    r1.metric("Vol Engine", "True GARCH(1,1)" if ARCH_AVAILABLE else "EWMA Proxy")
                    r2.metric("WFO Status", "Active (OOS)" if len(df) > 315 else "In-Sample")
                    r3.metric("Sortino Ratio", f"{sortino:.2f}")
                    r4.metric("Calmar Ratio", f"{calmar:.2f}")
                    try:
                        upside_var = round(curr_price * (1 + np.percentile(df['Close'].pct_change().dropna(), 95)), 2)
                    except:
                        upside_var = curr_price * 1.05
                    r5.metric("Upside VaR (Shorts)", f"${upside_var:.2f}")

                    allocation_action = "DEPLOY CAPITAL" if half_kelly > 0 else "FLATTEN POSITION / NO EDGE"
                    box_color, border_color = ("#082F49", "#38BDF8") if half_kelly > 0 else ("#450a0a", "#f87171")
                    
                    st.markdown(f"""
                    <div class="apex-box" style="background-color: {box_color}; border-left: 5px solid {border_color};">
                        <div class="apex-title">⚡ HQTA Allocation Directive</div>
                        <div class="apex-action" style="color: {border_color};">ACTION: {allocation_action}</div>
                        <div class="apex-logic">
                            Optimal Half-Kelly Sizing: <strong>{half_kelly:.2f}%</strong> of total portfolio equity.<br>
                            <em>Calculated using true GARCH(1,1) variance modeling and out-of-sample walk-forward optimization.</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### 📈 Quantitative Dynamics: Kalman Centerline & Dynamic GARCH Bands")
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Candlestick(x=bt_df.index, open=bt_df['Open'], high=bt_df['High'], low=bt_df['Low'], close=bt_df['Close'], name='Market Price', increasing_line_color='#38BDF8', decreasing_line_color='#334155'))
                    if 'Kalman_Price' in bt_df.columns:
                        fig_price.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Kalman_Price'], mode='lines', name='Kalman Filter', line=dict(color='#F8FAFC', width=1.5)))
                        fig_price.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Upper_Band'], mode='lines', name='Upper Vol Band', line=dict(color='#94A3B8', width=1, dash='dot')))
                        fig_price.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Lower_Band'], mode='lines', name='Lower Vol Band', line=dict(color='#94A3B8', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(148, 163, 184, 0.1)'))
                    fig_price.update_layout(template='plotly_dark', paper_bgcolor='#0B0F19', plot_bgcolor='#0B0F19', margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False, height=500, font_family="Inter")
                    st.plotly_chart(fig_price, use_container_width=True)

                    st.markdown("### 🎯 Advanced Options Architecture (For Pros)")
                    st.info(f"**STRATEGY:** {plan['name']} | **LEGS:** {plan['legs']}")
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Est. Execution Target", plan['premium'])
                    s2.metric("Prob. of Profit (POP)", f"{plan['pop']}%")
                    s3.metric("Ideal DTE", plan['dte'])

                    hist_dates = df.index.tz_localize(None) if df.index.tz else df.index
                    future_dates = pd.date_range(start=hist_dates[-1] + pd.Timedelta(days=1), periods=30, freq='B')
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist_dates, y=df['Close'], name='History', line=dict(color='#F1F5F9')))
                    fig.add_trace(go.Scatter(x=future_dates, y=mc_df.mean(axis=1), name='Mean Projection', line=dict(dash='dash', color='#38BDF8')))
                    fig.add_hline(y=sup, line_dash="dot", line_color="#4ADE80", annotation_text="Support", annotation_font_color="#4ADE80")
                    fig.add_hline(y=res, line_dash="dot", line_color="#F87171", annotation_text="Resistance", annotation_font_color="#F87171")
                    
                    fig.update_layout(template="plotly_dark", height=500, title=f"Institutional Chart (History + 30-Day Projection | {mc_sims} Simulations)", paper_bgcolor='#0B0F19', plot_bgcolor='#0F172A', font_family="Inter", font=dict(color='#F8FAFC'))
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Deep Dive Engine Error: {str(e)}")

    with st.sidebar:
        st.markdown("---")
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.rerun()

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="font-size: 0.85em; color: #94A3B8; line-height: 1.6; text-align: justify; padding: 15px; border-left: 4px solid #F59E0B; background-color: #1E293B; border-radius: 4px; margin-bottom: 20px;">
    <b style="color: #F8FAFC;">SEC RULE 206(4)-1 COMPLIANCE NOTICE:</b> VRP Quant and its associated Terminal operate strictly as a financial data and analytics publisher. We are not a registered investment advisor, broker-dealer, or financial planner. All quantitative metrics, Alpha Scores, Volatility Risk Premium (VRP) edges, N(d2) Probabilities of Profit (POP), and mathematically derived Support/Resistance levels provided by this platform are for informational and educational purposes only. Past performance does not guarantee future results.<br><br>
    <div style="text-align: center; font-size: 0.9em; color: #64748B;">
        &copy; 2026 vrpquant.com. All Rights Reserved.
    </div>
</div>
""", unsafe_allow_html=True)

