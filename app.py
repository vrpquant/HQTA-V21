import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime
import pytz
import os
import time

# ==========================================
# --- INSTITUTIONAL UI THEME INJECTION ---
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
# --- GLOBAL INSTITUTIONAL UNIVERSE ---
# ==========================================
TICKER_SETS = {
    "🔥 Magnificent 7 + Crypto": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "COIN", "MSTR", "MARA", "BTC-USD"],
    "💻 Semiconductors (AI)": ["NVDA", "AMD", "TSM", "INTC", "MU", "AVGO", "QCOM", "ARM", "AMAT", "SMH"],
    "🛢️ Energy & Commodities": ["XOM", "CVX", "COP", "SLB", "OXY", "EOG", "MPC", "VLO", "HAL", "XLE"],
    "🏥 Healthcare & Biotech": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "AMGN", "ISRG", "SYK", "XLV"],
    "🏦 Financials & Banking": ["JPM", "BAC", "WFC", "MS", "GS", "C", "V", "MA", "AXP", "XLF"],
    "🏭 Industrials & Defense": ["GE", "CAT", "UBER", "BA", "RTX", "LMT", "HON", "UNP", "DE", "XLI"],
    "🛒 Consumer Discretionary": ["TSLA", "AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX", "XLY"],
    "🧼 Consumer Staples": ["WMT", "PG", "COST", "KO", "PEP", "PM", "TGT", "MO", "DG", "XLP"],
    "🏠 Real Estate (REITs)": ["AMT", "PLD", "CCI", "EQIX", "O", "PSA", "SPG", "WELL", "DLR", "XLRE"],
    "🔌 Utilities": ["NEE", "CEG", "SO", "DUK", "SRE", "AEP", "D", "PCG", "EXC", "XLU"],
    "📡 Communications & Media": ["META", "GOOGL", "NFLX", "DIS", "VZ", "T", "CMCSA", "TMUS", "WBD", "XLC"]
}

# ==========================================
# --- ADVANCED INSTITUTIONAL MATH ENGINE ---
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

    @staticmethod
    def calculate_score(df):
        try:
            calc_df = df.copy().dropna()
            if len(calc_df) < 50: return 50
                
            calc_df['Kalman_Price'] = AlphaEngine.apply_kalman_filter(calc_df['Close'])
            returns = calc_df['Close'].pct_change()
            calc_df['GARCH_Proxy_Vol'] = returns.ewm(span=20).std() * np.sqrt(252) 
            
            calc_df['Band_Std'] = calc_df['GARCH_Proxy_Vol'] * calc_df['Kalman_Price'] / np.sqrt(252)
            calc_df['Upper_Band'] = calc_df['Kalman_Price'] + (2 * calc_df['Band_Std'])
            calc_df['Lower_Band'] = calc_df['Kalman_Price'] - (2 * calc_df['Band_Std'])
            calc_df['BBW'] = (calc_df['Upper_Band'] - calc_df['Lower_Band']) / calc_df['Kalman_Price']
            
            current_price = calc_df['Close'].iloc[-1]
            lower_band = calc_df['Lower_Band'].iloc[-1]
            upper_band = calc_df['Upper_Band'].iloc[-1]
            kalman_trend = calc_df['Kalman_Price'].iloc[-1] > calc_df['Kalman_Price'].iloc[-2]
            
            score = 50 
            if current_price < lower_band and kalman_trend: score += 35 
            elif current_price > upper_band: score -= 35
                
            bbw_z = (calc_df['BBW'].iloc[-1] - calc_df['BBW'].mean()) / (calc_df['BBW'].std() + 1e-9)
            score += max(-15, min(15, bbw_z * 10))
            return max(0, min(100, int(score)))
        except:
            return 50

class BacktestEngine:
    @staticmethod
    def run_quick_backtest(df, slippage_bps=5, commission_bps=2):
        try:
            bt_df = df.copy().dropna()
            if len(bt_df) < 50:
                return 0.0, 0.0, 0.0, 0.0, 0.0, bt_df
                
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
                    
            return round(win_rate,1), round(cumulative*100,1), round(outperf*100,1), round(max_dd,1), round(half_kelly,1), bt_df
        except:
            return 0.0, 0.0, 0.0, 0.0, 0.0, df.copy()

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
    def detect_reversal(df):
        try:
            if len(df) < 201: return "Insufficient Data"
            sma50 = df['Close'].rolling(50).mean()
            sma200 = df['Close'].rolling(200).mean()
            if sma50.iloc[-2] < sma200.iloc[-2] and sma50.iloc[-1] >= sma200.iloc[-1]: return "Golden Cross (Bull)"
            elif sma50.iloc[-2] > sma200.iloc[-2] and sma50.iloc[-1] <= sma200.iloc[-1]: return "Death Cross (Bear)"
                
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            if rsi.iloc[-2] < 30 and rsi.iloc[-1] >= 30: return "RSI Bull Bounce"
            elif rsi.iloc[-2] > 70 and rsi.iloc[-1] <= 70: return "RSI Bear Rejection"
            return "No Active Reversal"
        except:
            return "No Active Reversal"

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
    def calculate_var(df, confidence=0.95):
        """Standard 95% Downside Value at Risk"""
        try:
            price = df['Close'].iloc[-1]
            daily_returns = df['Close'].pct_change().dropna()
            var_pct = np.percentile(daily_returns, (1 - confidence) * 100)
            return round(price * (1 + var_pct), 2)
        except:
            return df['Close'].iloc[-1] * 0.95

    @staticmethod
    def calculate_upside_var(df, confidence=0.95):
        """95% Upside Variance (Used for sizing Short positions below Resistance)"""
        try:
            price = df['Close'].iloc[-1]
            daily_returns = df['Close'].pct_change().dropna()
            var_pct = np.percentile(daily_returns, confidence * 100)
            return round(price * (1 + var_pct), 2)
        except:
            return df['Close'].iloc[-1] * 1.05

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

    @staticmethod
    def generate_hybrid_plan(price, score, vrp, sup, res):
        hybrid = {}
        if score >= 60:
            if vrp > 0:
                hybrid['name'] = "The Institutional Buy-Write (Yield Harvest)"
                hybrid['action'] = f"Buy 100 Shares @ Market AND Sell 1 Call Option @ ${res:.2f} Strike."
                hybrid['logic'] = "Trend is strong, but options are expensive. We buy the stock and sell overpriced calls to institutions to lower our risk."
            else:
                hybrid['name'] = "The Bulletproof Bull (Protected Upside)"
                hybrid['action'] = f"Buy 100 Shares @ Market AND Buy 1 Put Option @ ${sup:.2f} Strike."
                hybrid['logic'] = "Trend is strong and options are cheap. We ride the stock up, but buy cheap insurance at Support to make this mathematically low-stress."
        elif score <= 40:
            if vrp > 0:
                hybrid['name'] = "The Warren Buffett Entry (Discount Acquisition)"
                hybrid['action'] = f"Hold Cash AND Sell 1 Cash-Secured Put @ ${sup:.2f} Strike."
                hybrid['logic'] = "Momentum is weak and fear is high. Do not buy the stock yet. Sell puts to get paid upfront while waiting to buy it at the Support floor."
            else:
                hybrid['name'] = "The Smart-Money Short (Risk-Defined Bear)"
                hybrid['action'] = f"Do NOT Buy Stock. Buy 1 Put Option Vertical Spread targeting ${sup:.2f}."
                hybrid['logic'] = "Momentum is broken and options are cheap. We use low-risk put options to profit from the drop without shorting shares."
        else:
            hybrid['name'] = "The Floor-to-Ceiling Swing (Mean Reversion)"
            hybrid['action'] = f"Place Limit Buy Order for Shares @ ${sup:.2f} AND Set Sell Target @ ${res:.2f}."
            hybrid['logic'] = "Stock is trapped in a channel. We refuse to buy at current prices. We set traps at the floor and sell at the ceiling."
            
        return hybrid

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
                    vrp = QuantLogic.calculate_vrp_edge(t, df, mode="scanner")
                    reversal = QuantLogic.detect_reversal(df)
                    sup, res = QuantLogic.get_support_resistance(df)
                    
                    var_95 = QuantLogic.calculate_var(df)          # Standard Downside VaR
                    upside_var = QuantLogic.calculate_upside_var(df) # Upside VaR for shorting
                    
                    win_rate, strat_ret, outperf, max_dd, kelly, _ = BacktestEngine.run_quick_backtest(df)
                    plan = TradeArchitect.generate_plan(t, price, score, vol, sup, res, kelly)
                    hybrid = TradeArchitect.generate_hybrid_plan(price, score, vrp, sup, res)
                    
                    # ==========================================
                    # --- THE ULTIMATE GOD-MODE METRIC LOGIC ---
                    # ==========================================
                    
                    # ULTIMATE LONG CONDITIONS
                    is_ult_long = (
                        var_95 > sup and
                        score > 58 and
                        ("Bullish" in plan['bias'] or "Bear Rejection" in reversal) and
                        vrp < 0 and
                        win_rate > 10.0 and
                        strat_ret > 50.0 and
                        kelly > 2.0
                    )
                    
                    # ULTIMATE SHORT CONDITIONS (Inverted Math)
                    is_ult_short = (
                        upside_var < res and
                        score < 42 and
                        ("Bearish" in plan['bias'] or "Bull Bounce" in reversal) and
                        vrp > 0 and 
                        win_rate > 10.0 and
                        strat_ret > 50.0 and 
                        kelly > 2.0
                    )

                    ultimate_signal = "🎯 ULTIMATE LONG" if is_ult_long else "🩸 ULTIMATE SHORT" if is_ult_short else "Standard"

                    results.append({
                        "Ticker": t, "Price": round(price,2), "Ultimate Signal": ultimate_signal,
                        "Alpha Score": score, "Trend": plan['bias'],
                        "Reversal": reversal, "VRP Edge": f"{vrp:+.1f}%", "Vol": f"{vol:.1f}%", 
                        "Support": round(sup,2), "Resistance": round(res,2), 
                        "HQTA Apex Action": hybrid['action'],
                        "Strategy": plan['name'], "Kelly": f"{kelly}%"
                    })
                
                time.sleep(1.5)
            except Exception as e:
                pass
        
        df_results = pd.DataFrame(results)
        if df_results.empty:
            return df_results
        return df_results.sort_values("Alpha Score", ascending=False)

# ==========================================
# --- STREAMLIT APP UI ---
# ==========================================

st.set_page_config(page_title="VRP Quant | V22.2 Institutional", layout="wide", page_icon="🏦")
inject_institutional_css() 
est_tz = pytz.timezone('US/Eastern')

try:
    USERS = st.secrets["credentials"]
except Exception as e:
    st.error("⚠️ SYSTEM LOCKED: Security vault not connected. Please configure [credentials] in Streamlit Secrets.")
    st.stop()

# ==========================================
# --- PAYMENT GATEWAY LINKS ---
# ==========================================
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
                else: 
                    st.error("Invalid Credentials")
        st.markdown("---")
        st.markdown("### 👑 Founding Member Cohort (V22.2)")
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
        st.markdown("# 🏦 VRP Quant V22.2")
        if tier == "GOD_MODE": st.success("🔓 GOD MODE ACTIVE")
        else: st.warning("🔒 ANALYST TIER")
        st.markdown("---")
        mode = st.sidebar.radio("Module", ["🚀 Market Scanner", "🔬 Deep Dive Analysis"])

    if mode == "🚀 Market Scanner":
        st.title("🚀 Institutional Market Scanner")
        
        if tier != "GOD_MODE":
            st.error("🔒 ACCESS DENIED: Market Scanner is locked for Analyst Tier.")
        else:
            st.markdown("### ⚡ Live Sector Scanner")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                options = list(TICKER_SETS.keys()) + ["✨ Custom Watchlist"]
                sector_choice = st.selectbox("Select Sector to Scan:", options)
            
            selected_tickers = []
            if sector_choice == "✨ Custom Watchlist":
                with col2:
                    custom_input = st.text_area("Enter Tickers:", "PLTR, SOFI")
                    if custom_input: selected_tickers = [t.strip().upper() for t in custom_input.split(',')]
            else: selected_tickers = TICKER_SETS[sector_choice]

            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                strict_mode = st.checkbox("⚡ ISOLATE ULTIMATE MASTER SETUPS ONLY", value=False, help="Filters out all standard setups. Only shows tickers passing the strict quantitative logic.")
            
            if st.button("Run Live Sector Scan") and selected_tickers:
                with st.spinner("Running Live Sector Scan & Checking Ultimate Conditions..."):
                    try:
                        df_scan = MarketScanner.run_scan(selected_tickers)
                        if not df_scan.empty:
                            
                            # Filter logic for Ultimate Checkbox
                            if strict_mode:
                                df_scan = df_scan[df_scan["Ultimate Signal"] != "Standard"]
                                if df_scan.empty:
                                    st.warning("⚠️ No assets currently meet the strict Ultimate Master criteria. Cash is a position.")
                                    st.stop()
                                    
                            # Styling logic for Ultimate Signals and Apex Box
                            def highlight_ultimate(row):
                                if row.get("Ultimate Signal") == "🎯 ULTIMATE LONG":
                                    return ['background-color: #064E3B; color: #34D399; font-weight: bold'] * len(row)
                                elif row.get("Ultimate Signal") == "🩸 ULTIMATE SHORT":
                                    return ['background-color: #450A0A; color: #F87171; font-weight: bold'] * len(row)
                                return [''] * len(row)

                            styled_df = df_scan.style.apply(highlight_ultimate, axis=1).set_properties(**{
                                'background-color': '#1E293B',
                                'color': '#F8FAFC',
                                'border-color': '#334155'
                            })
                            
                            if 'HQTA Apex Action' in df_scan.columns:
                                styled_df = styled_df.set_properties(subset=['HQTA Apex Action'], **{
                                    'background-color': '#0C4A6E',
                                    'color': '#38BDF8',
                                    'font-weight': 'bold'
                                })
                                
                            st.dataframe(styled_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Scan failed: {e}")

    elif mode == "🔬 Deep Dive Analysis":
        st.title("🔬 Deep Dive & Trade Architect")
        ticker = st.text_input("Asset Ticker", "TSLA").upper().strip()
        
        if st.button("Run Deep Dive Analysis"):
            with st.spinner("Extracting Advanced Institutional Metrics..."):
                try:
                    # SURGICAL STRIKE: Single ping, no throttling needed.
                    stock = yf.Ticker(ticker)
                    df = stock.history(period="2y")
                    
                    if df.empty:
                        st.error(f"⚠️ No historical bars found for {ticker}.")
                        st.stop()
                        
                    curr_price = df['Close'].iloc[-1]
                    score = AlphaEngine.calculate_score(df)
                    vol = QuantLogic.calculate_vol(df)
                    vrp_edge_val = QuantLogic.calculate_vrp_edge(ticker, df, mode="deep_dive")
                    reversal_signal = QuantLogic.detect_reversal(df)
                    sup, res = QuantLogic.get_support_resistance(df)
                    sharpe = QuantLogic.calculate_sharpe(df)
                    var_95 = QuantLogic.calculate_var(df)
                    win_rate, strat_ret, outperf, max_dd, half_kelly, bt_df = BacktestEngine.run_quick_backtest(df)
                    
                    plan = TradeArchitect.generate_plan(ticker, curr_price, score, vol, sup, res, half_kelly)
                    hybrid = TradeArchitect.generate_hybrid_plan(curr_price, score, vrp_edge_val, sup, res)
                    
                    # 10,000 MONTE CARLO SIMULATIONS FOR GOD MODE TIER
                    mc_df = MonteCarloEngine.simulate_paths(df, days=30, sims=10000 if tier == "GOD_MODE" else 1000)

                    # --- THE AWESOME SPOT UI BOX ---
                    st.markdown(f"""
                    <div class="apex-box">
                        <div class="apex-title">🏆 THE AWESOME SPOT: {hybrid['name']}</div>
                        <div class="apex-action">TRADE ARCHITECTURE: {hybrid['action']}</div>
                        <div class="apex-logic">Institutional Logic: {hybrid['logic']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Exact 10-Metric Layout match from user screenshot
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

                    # Exact Backtest layout match from user screenshot
                    st.markdown("### ⚙️ Strategy Backtest Validation (2-Year)")
                    b1, b2, b3, b4, b5 = st.columns(5)
                    b1.metric("Historical Win Rate", f"{win_rate:.1f}%")
                    b2.metric("Net Strategy Return", f"{strat_ret:+.1f}%")
                    b3.metric("Alpha Generated", f"{outperf:+.1f}%")
                    b4.metric("Markdown % (Max DD)", f"{max_dd:.1f}%", delta_color="inverse")
                    b5.metric("Kelly Fraction (Half)", f"{half_kelly:.1f}%", delta_color="normal")

                    # --- HQTA DIRECTIVE INJECTION ---
                    allocation_action = "DEPLOY CAPITAL" if half_kelly > 0 else "FLATTEN POSITION / NO EDGE"
                    box_color = "#082F49" if half_kelly > 0 else "#450a0a"
                    border_color = "#38BDF8" if half_kelly > 0 else "#f87171"
                    
                    st.markdown(f"""
                    <div class="apex-box" style="background-color: {box_color}; border-left: 5px solid {border_color};">
                        <div class="apex-title">⚡ HQTA Allocation Directive</div>
                        <div class="apex-action" style="color: {border_color};">ACTION: {allocation_action}</div>
                        <div class="apex-logic">
                            Optimal Half-Kelly Sizing: <strong>{half_kelly:.2f}%</strong> of total portfolio equity.<br>
                            <em>Calculated using EWMA GARCH-proxied variance and momentum-adjusted win rates.</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # --- PLOTLY QUANTITATIVE DYNAMICS CHART ---
                    st.markdown("### 📈 Quantitative Dynamics: Kalman Centerline & EWMA GARCH Bands")
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Candlestick(
                        x=bt_df.index, open=bt_df['Open'], high=bt_df['High'], low=bt_df['Low'], close=bt_df['Close'],
                        name='Market Price', increasing_line_color='#38BDF8', decreasing_line_color='#334155'
                    ))
                    if 'Kalman_Price' in bt_df.columns:
                        fig_price.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Kalman_Price'], mode='lines', name='Kalman Filter', line=dict(color='#F8FAFC', width=1.5)))
                        fig_price.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Upper_Band'], mode='lines', name='Upper Vol Band', line=dict(color='#94A3B8', width=1, dash='dot')))
                        fig_price.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Lower_Band'], mode='lines', name='Lower Vol Band', line=dict(color='#94A3B8', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(148, 163, 184, 0.1)'))
                    
                    fig_price.update_layout(template='plotly_dark', paper_bgcolor='#0B0F19', plot_bgcolor='#0B0F19', margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False, height=500)
                    st.plotly_chart(fig_price, use_container_width=True)

                    # --- REST OF ORIGINAL UI ---
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
                    
                    fig.update_layout(
                        template="plotly_dark", 
                        height=500, 
                        title=f"Institutional Chart (History + 30-Day Projection | {10000 if tier == 'GOD_MODE' else 1000} Simulations)",
                        paper_bgcolor='#0B0F19',
                        plot_bgcolor='#0F172A',
                        font=dict(color='#F8FAFC')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Deep Dive Engine Error: {str(e)}")

    with st.sidebar:
        st.markdown("---")
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.rerun()

# ==========================================
# --- SEC REGULATORY COMPLIANCE FOOTER ---
# ==========================================
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="font-size: 0.85em; color: #94A3B8; line-height: 1.6; text-align: justify; padding: 15px; border-left: 4px solid #F59E0B; background-color: #1E293B; border-radius: 4px; margin-bottom: 20px;">
    <b style="color: #F8FAFC;">SEC RULE 206(4)-1 COMPLIANCE NOTICE:</b> VRP Quant and its associated V22.2 Terminal operate strictly as a financial data and analytics publisher. We are not a registered investment advisor, broker-dealer, or financial planner. All quantitative metrics, Alpha Scores, Volatility Risk Premium (VRP) edges, N(d2) Probabilities of Profit (POP), and mathematically derived Support/Resistance levels provided by this platform are for informational and educational purposes only. Past performance does not guarantee future results.<br><br>
    <div style="text-align: center; font-size: 0.9em; color: #64748B;">
        &copy; 2026 vrpquant.com. All Rights Reserved.
    </div>
</div>
""", unsafe_allow_html=True)
