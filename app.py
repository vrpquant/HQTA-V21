import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime
import pytz
import time

# --- NEW: ARCH LIBRARY FOR TRUE GARCH(1,1) ---
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

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
    def apply_garch(returns):
        if ARCH_AVAILABLE and len(returns.dropna()) > 50:
            try:
                scaled_rets = returns.dropna() * 100
                am = arch_model(scaled_rets, vol='Garch', p=1, q=1, rescale=False)
                res = am.fit(disp='off', show_warning=False)
                cond_vol = res.conditional_volatility / 100
                return pd.Series(cond_vol, index=scaled_rets.index) * np.sqrt(252)
            except:
                pass
        return returns.ewm(span=20).std() * np.sqrt(252)

    @staticmethod
    def calculate_score(df):
        try:
            calc_df = df.copy().dropna()
            if len(calc_df) < 50: return 50
                
            calc_df['Kalman_Price'] = AlphaEngine.apply_kalman_filter(calc_df['Close'])
            returns = calc_df['Close'].pct_change().fillna(0)
            
            calc_df['GARCH_Vol'] = AlphaEngine.apply_garch(returns)
            calc_df['Band_Std'] = calc_df['GARCH_Vol'] * calc_df['Kalman_Price'] / np.sqrt(252)
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
    def run_wfo_backtest(df, slippage_bps=5, commission_bps=2):
        try:
            bt_df = df.copy().dropna()
            if len(bt_df) < 50:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, bt_df
                
            bt_df['Kalman_Price'] = AlphaEngine.apply_kalman_filter(bt_df['Close'])
            returns = bt_df['Close'].pct_change().fillna(0)
            bt_df['Vol_Regime'] = AlphaEngine.apply_garch(returns)
            bt_df['Underlying_Return'] = returns
            
            train_size = 252 
            step_size = 63   
            multipliers = [1.5, 2.0, 2.5] 
            
            oos_positions = pd.Series(0.0, index=bt_df.index)
            
            if len(bt_df) > train_size + step_size:
                for start_idx in range(0, len(bt_df) - train_size, step_size):
                    train_end = start_idx + train_size
                    test_end = min(train_end + step_size, len(bt_df))
                    
                    train_df = bt_df.iloc[start_idx:train_end]
                    best_sharpe = -999
                    best_mult = 2.0
                    
                    for m in multipliers:
                        band_std = train_df['Vol_Regime'] * train_df['Kalman_Price'] / np.sqrt(252)
                        upper = train_df['Kalman_Price'] + (m * band_std)
                        lower = train_df['Kalman_Price'] - (m * band_std)
                        
                        sig = pd.Series(0, index=train_df.index)
                        sig[train_df['Close'] < lower] = 1
                        sig[train_df['Close'] > upper] = -1
                        pos = sig.replace(0, np.nan).ffill().fillna(0).shift(1).fillna(0)
                        
                        strat_rets = pos * train_df['Underlying_Return']
                        sharpe = np.sqrt(252) * strat_rets.mean() / (strat_rets.std() + 1e-9)
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_mult = m
                            
                    test_df = bt_df.iloc[train_end:test_end]
                    band_std_oos = test_df['Vol_Regime'] * test_df['Kalman_Price'] / np.sqrt(252)
                    upper_oos = test_df['Kalman_Price'] + (best_mult * band_std_oos)
                    lower_oos = test_df['Kalman_Price'] - (best_mult * band_std_oos)
                    
                    sig_oos = pd.Series(0, index=test_df.index)
                    sig_oos[test_df['Close'] < lower_oos] = 1
                    sig_oos[test_df['Close'] > upper_oos] = -1
                    oos_pos = sig_oos.replace(0, np.nan).ffill().fillna(0)
                    oos_positions.iloc[train_end:test_end] = oos_pos
                    
                bt_df['Target_Position'] = oos_positions
            else:
                band_std = bt_df['Vol_Regime'] * bt_df['Kalman_Price'] / np.sqrt(252)
                bt_df['Upper_Band'] = bt_df['Kalman_Price'] + (2 * band_std)
                bt_df['Lower_Band'] = bt_df['Kalman_Price'] - (2 * band_std)
                
                sig = pd.Series(0, index=bt_df.index)
                sig[bt_df['Close'] < bt_df['Lower_Band']] = 1
                sig[bt_df['Close'] > bt_df['Upper_Band']] = -1
                bt_df['Target_Position'] = sig.replace(0, np.nan).ffill().fillna(0)

            bt_df['Actual_Position'] = bt_df['Target_Position'].shift(1).fillna(0)
            bt_df['Gross_Return'] = bt_df['Actual_Position'] * bt_df['Underlying_Return']
            
            turnover = bt_df['Actual_Position'].diff().abs().fillna(0)
            total_cost = (slippage_bps + commission_bps) / 10000
            bt_df['Net_Return'] = bt_df['Gross_Return'] - (turnover * total_cost * (1 + (bt_df['Vol_Regime'] > 0.35).astype(int)))
            
            eval_df = bt_df.iloc[train_size:] if len(bt_df) > train_size + step_size else bt_df
            
            win_rate = (eval_df['Net_Return'] > 0).mean() * 100
            cumulative = (1 + eval_df['Net_Return']).prod() - 1
            buy_hold = (1 + eval_df['Underlying_Return']).prod() - 1
            outperf = cumulative - buy_hold
            
            peak = (1 + eval_df['Net_Return']).cumprod().cummax()
            max_dd = (((1 + eval_df['Net_Return']).cumprod() - peak) / peak).min() * 100
            
            ann_return = eval_df['Net_Return'].mean() * 252
            downside_std = eval_df[eval_df['Net_Return'] < 0]['Net_Return'].std() * np.sqrt(252)
            sortino = ann_return / (downside_std + 1e-9)
            calmar = ann_return / (abs(max_dd)/100 + 1e-9)
            
            wins = eval_df[eval_df['Net_Return'] > 0]['Net_Return']
            losses = eval_df[eval_df['Net_Return'] < 0]['Net_Return']
            
            half_kelly = 0.0
            if len(wins) > 0 and len(losses) > 0:
                win_avg = wins.mean()
                loss_avg = abs(losses.mean())
                win_prob = len(wins) / (len(wins) + len(losses))
                if loss_avg > 0:
                    kelly_fraction = win_prob - ((1 - win_prob) / (win_avg / loss_avg))
                    half_kelly = max(0.0, kelly_fraction / 2.0) * 100 
            
            last_band_std = bt_df['Vol_Regime'] * bt_df['Kalman_Price'] / np.sqrt(252)
            bt_df['Upper_Band'] = bt_df['Kalman_Price'] + (2 * last_band_std)
            bt_df['Lower_Band'] = bt_df['Kalman_Price'] - (2 * last_band_std)
                    
            return round(win_rate,1), round(cumulative*100,1), round(outperf*100,1), round(max_dd,1), round(half_kelly,1), round(sortino, 2), round(calmar, 2), bt_df
        except:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, df.copy()

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
        try:
            price = df['Close'].iloc[-1]
            daily_returns = df['Close'].pct_change().dropna()
            var_pct = np.percentile(daily_returns, (1 - confidence) * 100)
            return round(price * (1 + var_pct), 2)
        except:
            return df['Close'].iloc[-1] * 0.95

    @staticmethod
    def calculate_upside_var(df, confidence=0.95):
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

class PortfolioEngine:
    @staticmethod
    def cross_sectional_momentum(price_dict):
        try:
            returns = {t: df['Close'].pct_change(20).iloc[-1] for t, df in price_dict.items()}
            ranks = pd.Series(returns).rank(pct=True)
            return ranks.to_dict()
        except:
            return {t: 0.5 for t in price_dict.keys()}

    @staticmethod
    def volatility_targeting(df, target_vol=0.20):
        try:
            realized = df['Close'].pct_change().std() * np.sqrt(252)
            weight = target_vol / (realized + 1e-9)
            return min(2.0, max(0.0, weight))
        except:
            return 1.0

    @staticmethod
    def mean_variance_weight(price_dict):
        try:
            returns = pd.DataFrame({t: df['Close'].pct_change() for t, df in price_dict.items()}).dropna()
            cov = returns.cov()
            inv = np.linalg.pinv(cov.values)
            ones = np.ones(len(inv))
            weights = inv @ ones / (ones.T @ inv @ ones)
            return dict(zip(returns.columns, weights))
        except:
            return {t: 1/len(price_dict) for t in price_dict}

    @staticmethod
    def kelly_weight(df):
        try:
            r = df['Close'].pct_change().dropna()
            mu = r.mean() * 252
            var = r.var() * 252
            kelly = mu / (var + 1e-9)
            return max(0, min(kelly, 2))
        except:
            return 0.0

class RegimeEngine:
    @staticmethod
    @st.cache_data(ttl=900)
    def detect_regime():
        try:
            spy = yf.Ticker("SPY").history(period="1y")
            vix = yf.Ticker("^VIX").history(period="6mo")
            spy_close = spy['Close']
            vix_level = vix['Close'].iloc[-1]

            ma50 = spy_close.rolling(50).mean().iloc[-1]
            ma200 = spy_close.rolling(200).mean().iloc[-1]
            price = spy_close.iloc[-1]
            momentum = spy_close.pct_change(60).iloc[-1]

            if price > ma50 > ma200 and vix_level < 20 and momentum > 0:
                return "Risk-On"
            elif price < ma200 and vix_level > 25:
                return "Risk-Off"
            else:
                return "Neutral"
        except Exception as e:
            return "Neutral"

class OptionsExpectedMove:
    @staticmethod
    def calculate(ticker, current_price):
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options

            if len(expirations) == 0:
                return 0, 0, 0

            nearest_exp = expirations[0]
            time.sleep(0.5) 
            chain = tk.option_chain(nearest_exp)
            
            calls = chain.calls
            puts = chain.puts

            calls['diff'] = abs(calls['strike'] - current_price)
            atm_call = calls.sort_values('diff').iloc[0]

            puts['diff'] = abs(puts['strike'] - current_price)
            atm_put = puts.sort_values('diff').iloc[0]

            call_price = atm_call['lastPrice']
            put_price = atm_put['lastPrice']

            expected_move = call_price + put_price
            upper = current_price + expected_move
            lower = current_price - expected_move

            return expected_move, upper, lower
        except:
            return 0, 0, 0

# ==========================================
# --- V30 SECTOR STRENGTH ENGINE ---
# ==========================================

class SectorStrengthEngine:
    # SPDR ETFs & Proxies used to gauge institutional capital flow
    SECTOR_ETFS = {
        "🔥 Magnificent 7 + BTC": "MAGS", # The Roundhill Magnificent Seven ETF proxy
        "🪙 Digital Assets & Proxies": "WGMI", # Bitcoin Miners Proxy
        "💻 Semiconductors (AI)": "SMH",
        "🛢️ Energy & Commodities": "XLE",
        "🏥 Healthcare & Biotech": "XLV",
        "🏦 Financials & Banking": "XLF",
        "🏭 Industrials & Defense": "XLI",
        "🛒 Consumer Discretionary": "XLY",
        "🧼 Consumer Staples": "XLP",
        "🏠 Real Estate (REITs)": "XLRE",
        "🔌 Utilities": "XLU",
        "📡 Communications & Media": "XLC"
    }

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_strongest_sector():
        """Calculates 20-Day ROC for sector ETFs to auto-detect capital rotation."""
        try:
            returns = {}
            for sector, etf in SectorStrengthEngine.SECTOR_ETFS.items():
                df = yf.Ticker(etf).history(period="1mo")
                if not df.empty and len(df) > 15:
                    roc = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
                    returns[sector] = roc

            if returns:
                best_sector = max(returns, key=returns.get)
                return best_sector, returns[best_sector]
            
            return "💻 Semiconductors (AI)", 0.0 
        except:
            return "💻 Semiconductors (AI)", 0.0 

# ==========================================
# --- V30 DYNAMIC SECTOR UNIVERSE ENGINE ---
# ==========================================

class UniverseEngine:
    # The Full Institutional Dictionary (Corrected Mag 7)
    SECTOR_UNIVERSE = {
        "🔥 Magnificent 7 + BTC": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BTC-USD"],
        "🪙 Digital Assets & Proxies": ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "IBIT"],
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

class DynamicUniverseEngine:
    @staticmethod
    def get_apex_10():
        return ["SPY", "QQQ", "NVDA", "TSLA", "MSTR", "COIN", "AAPL", "AMD", "AMZN", "META"]

class MarketScanner:
    @staticmethod
    @st.cache_data(ttl=900, show_spinner=False)
    def run_scan(tickers):
        regime = RegimeEngine.detect_regime()
        results = []
        price_dict = {}

        for t in tickers:
            try:
                stock = yf.Ticker(t)
                df = stock.history(period="2y")
                
                if len(df) > 100:
                    avg_vol = df['Volume'].tail(30).mean()
                    daily_vol = df['Close'].pct_change().std()
                    
                    if avg_vol > 1_000_000 and daily_vol > 0.02:
                        price_dict[t] = df
                        
                time.sleep(1.5) 
            except:
                pass

        if not price_dict:
            return pd.DataFrame()

        cs_rank = PortfolioEngine.cross_sectional_momentum(price_dict)
        mvo_weights = PortfolioEngine.mean_variance_weight(price_dict)

        for t, df in price_dict.items():
            try:
                price = df['Close'].iloc[-1]
                score = AlphaEngine.calculate_score(df)
                vol = QuantLogic.calculate_vol(df)
                vrp = QuantLogic.calculate_vrp_edge(t, df, mode="scanner")
                reversal = QuantLogic.detect_reversal(df)
                sup, res = QuantLogic.get_support_resistance(df)
                
                win_rate, strat_ret, outperf, max_dd, kelly, sortino, calmar, _ = BacktestEngine.run_wfo_backtest(df)
                plan = TradeArchitect.generate_plan(t, price, score, vol, sup, res, kelly)
                hybrid = TradeArchitect.generate_hybrid_plan(price, score, vrp, sup, res)
                move, exp_upper, exp_lower = OptionsExpectedMove.calculate(t, price)

                cs = cs_rank.get(t, 0.5)
                vol_target = PortfolioEngine.volatility_targeting(df)
                mvo = mvo_weights.get(t, 0)
                kelly_port = PortfolioEngine.kelly_weight(df)

                ultimate_score = (score * 0.35 + cs * 100 * 0.20 + kelly_port * 50 * 0.15 + vol_target * 50 * 0.15 + mvo * 50 * 0.15)

                is_long = (regime != "Risk-Off" and ultimate_score > 65 and vrp < 0 and win_rate > 10 and strat_ret > 50)
                is_short = (ultimate_score < 35 and vrp > 0 and win_rate > 10 and strat_ret > 50)
                ultimate_signal = "🎯 ULTIMATE LONG" if is_long else "🩸 ULTIMATE SHORT" if is_short else "Standard"

                results.append({
                    "Ticker": t, "Price": round(price, 2), "Ultimate Signal": ultimate_signal,
                    "Alpha Score": score, "Trend": plan['bias'], "Options Exp Move": round(move, 2),
                    "Options Upper": round(exp_upper, 2), "Options Lower": round(exp_lower, 2),
                    "Reversal": reversal, "VRP Edge": f"{vrp:+.1f}%", "Vol": f"{vol:.1f}%",
                    "Support": round(sup, 2), "Resistance": round(res, 2), "HQTA Apex Action": hybrid['action'],
                    "Strategy": plan['name'], "Kelly": f"{kelly}%"
                })
            except:
                pass

        df_results = pd.DataFrame(results)
        if df_results.empty: return df_results
        return df_results.sort_values("Alpha Score", ascending=False).head(10)

# ==========================================
# --- STREAMLIT APP UI ---
# ==========================================
st.set_page_config(page_title="VRP Quant | V30 Institutional", layout="wide", page_icon="🏦")
inject_institutional_css() 
est_tz = pytz.timezone('US/Eastern')

try:
    USERS = st.secrets["credentials"]
except Exception as e:
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
                else: 
                    st.error("Invalid Credentials")
        st.markdown("---")
        st.markdown("### 👑 Founding Member Cohort (V30.0)")
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
        st.markdown("# 🏦 VRP Quant V30.0")
        if tier == "GOD_MODE": st.success("🔓 GOD MODE ACTIVE")
        else: st.warning("🔒 ANALYST TIER")
        st.markdown("---")
        mode = st.sidebar.radio("Module", ["🚀 Market Scanner", "🔬 Deep Dive Analysis"])

    if mode == "🚀 Market Scanner":
        st.title("🚀 Institutional Market Scanner")
        
        regime = RegimeEngine.detect_regime()
        st.markdown(f"### 🌍 Market Regime: **{regime}**")
        st.markdown("---")
        
        if tier != "GOD_MODE":
            st.error("🔒 ACCESS DENIED: Market Scanner is locked for Analyst Tier.")
        else:
            st.markdown("### ⚡ Live Sector Scanner")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                options = ["🤖 Auto-Detect Strongest Sector", "🌌 Dynamic Apex 10 (Max Liquidity)"] + list(UniverseEngine.SECTOR_UNIVERSE.keys()) + ["✨ Custom Watchlist"]
                sector_choice = st.selectbox("Select Sector to Scan:", options)
            
            selected_tickers = []
            if sector_choice == "🤖 Auto-Detect Strongest Sector":
                with st.spinner("Analyzing Macro Capital Rotation via SPDR ETFs..."):
                    best_sector, sector_roc = SectorStrengthEngine.get_strongest_sector()
                    selected_tickers = UniverseEngine.SECTOR_UNIVERSE[best_sector]
                    st.info(f"🔄 **Auto-Rotation Triggered:** The engine detected institutional capital flowing into **{best_sector}** (20-Day ROC: +{sector_roc*100:.2f}%). Targeting {len(selected_tickers)} underlying assets.")
            elif sector_choice == "✨ Custom Watchlist":
                with col2:
                    custom_input = st.text_area("Enter Tickers:", "PLTR, SOFI")
                    if custom_input: selected_tickers = [t.strip().upper() for t in custom_input.split(',')]
            elif sector_choice == "🌌 Dynamic Apex 10 (Max Liquidity)":
                selected_tickers = DynamicUniverseEngine.get_apex_10()
            else: 
                selected_tickers = UniverseEngine.SECTOR_UNIVERSE[sector_choice]

            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                strict_mode = st.checkbox("⚡ ISOLATE HIGH-CONVICTION QUANTITATIVE SETUPS", value=False, help="Filters out all standard setups. Only shows tickers passing the strict quantitative logic.")
            
            if st.button("Run Live Sector Scan") and selected_tickers:
                with st.spinner(f"Initiating Stealth Scan & Applying 1M Vol / 31.7% Volatility Gates for {len(selected_tickers)} Assets..."):
                    try:
                        df_scan = MarketScanner.run_scan(selected_tickers)
                        if not df_scan.empty:
                            if strict_mode:
                                df_scan = df_scan[df_scan["Ultimate Signal"] != "Standard"]
                                if df_scan.empty:
                                    st.warning("⚠️ No assets currently meet the strict Ultimate Master criteria. Cash is a position.")
                                    st.stop()
                                    
                            def render_institutional_html_table(df):
                                html = """
                                <style>
                                    .table-container { overflow-x: auto; background-color: #0B0F19; margin-top: 10px; border: 1px solid #1E293B; border-radius: 6px; }
                                    .inst-table { width: 100%; border-collapse: collapse; font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 13px; text-align: left; }
                                    .inst-table th { background-color: #0F172A; color: #94A3B8; padding: 12px 15px; border-bottom: 2px solid #334155; font-weight: 700; font-size: 11px; text-transform: uppercase; white-space: nowrap; letter-spacing: 0.5px; }
                                    .inst-table tr { border-bottom: 1px solid #1E293B; transition: background-color 0.15s; background-color: #0B0F19; }
                                    .inst-table tr:hover { background-color: #162032; }
                                    .inst-table td { padding: 12px 15px; color: #E2E8F0; vertical-align: middle; white-space: nowrap; }
                                    
                                    .badge-long { background-color: #022C22; color: #10B981; padding: 4px 8px; border-radius: 4px; font-weight: 900; border: 1px solid #047857; font-size: 11px; letter-spacing: 0.5px; }
                                    .badge-short { background-color: #450A0A; color: #EF4444; padding: 4px 8px; border-radius: 4px; font-weight: 900; border: 1px solid #B91C1C; font-size: 11px; letter-spacing: 0.5px; }
                                    .badge-std { color: #475569; font-style: italic; font-size: 11px; }
                                    
                                    .apex-cell { background-color: #082F49 !important; border-left: 4px solid #0EA5E9 !important; border-right: 1px solid #0369A1 !important; color: #38BDF8 !important; font-weight: 700; white-space: normal !important; min-width: 250px; line-height: 1.5; }
                                    
                                    .ticker-cell { font-weight: 900; color: #FFFFFF; font-size: 14px; letter-spacing: 0.5px; }
                                    .val-pos { color: #10B981; font-weight: 600; }
                                    .val-neg { color: #EF4444; font-weight: 600; }
                                    .val-neu { color: #94A3B8; }
                                </style>
                                <div class="table-container">
                                <table class="inst-table">
                                    <thead><tr>
                                """
                                for col in df.columns: html += f"<th>{col}</th>"
                                html += "</tr></thead><tbody>"
                                
                                for _, row in df.iterrows():
                                    html += "<tr>"
                                    for col in df.columns:
                                        val = row[col]
                                        if col == "Ticker": html += f"<td class='ticker-cell'>{val}</td>"
                                        elif col == "Ultimate Signal":
                                            if val == "🎯 ULTIMATE LONG": html += f"<td><span class='badge-long'>{val}</span></td>"
                                            elif val == "🩸 ULTIMATE SHORT": html += f"<td><span class='badge-short'>{val}</span></td>"
                                            else: html += f"<td><span class='badge-std'>{val}</span></td>"
                                        elif col == "HQTA Apex Action": html += f"<td class='apex-cell'>{val}</td>"
                                        elif col == "VRP Edge":
                                            if isinstance(val, str) and val.startswith("+"): html += f"<td class='val-pos'>{val}</td>"
                                            elif isinstance(val, str) and val.startswith("-"): html += f"<td class='val-neg'>{val}</td>"
                                            else: html += f"<td>{val}</td>"
                                        elif col == "Trend":
                                            if isinstance(val, str) and "LONG" in val: html += f"<td class='val-pos'>{val}</td>"
                                            elif isinstance(val, str) and "SHORT" in val: html += f"<td class='val-neg'>{val}</td>"
                                            else: html += f"<td class='val-neu'>{val}</td>"
                                        else: html += f"<td>{val}</td>"
                                    html += "</tr>"
                                html += "</tbody></table></div>"
                                return html

                            html_table = render_institutional_html_table(df_scan)
                            st.markdown(html_table, unsafe_allow_html=True)
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
                    win_rate, strat_ret, outperf, max_dd, half_kelly, sortino, calmar, bt_df = BacktestEngine.run_wfo_backtest(df)
                    
                    plan = TradeArchitect.generate_plan(ticker, curr_price, score, vol, sup, res, half_kelly)
                    hybrid = TradeArchitect.generate_hybrid_plan(curr_price, score, vrp_edge_val, sup, res)
                    mc_df = MonteCarloEngine.simulate_paths(df, days=30, sims=10000 if tier == "GOD_MODE" else 1000)

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
                    r5.metric("Upside VaR (Shorts)", f"${QuantLogic.calculate_upside_var(df):.2f}")

                    allocation_action = "DEPLOY CAPITAL" if half_kelly > 0 else "FLATTEN POSITION / NO EDGE"
                    box_color = "#082F49" if half_kelly > 0 else "#450a0a"
                    border_color = "#38BDF8" if half_kelly > 0 else "#f87171"
                    
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

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="font-size: 0.85em; color: #94A3B8; line-height: 1.6; text-align: justify; padding: 15px; border-left: 4px solid #F59E0B; background-color: #1E293B; border-radius: 4px; margin-bottom: 20px;">
    <b style="color: #F8FAFC;">SEC RULE 206(4)-1 COMPLIANCE NOTICE:</b> VRP Quant and its associated Terminal operate strictly as a financial data and analytics publisher. We are not a registered investment advisor, broker-dealer, or financial planner. All quantitative metrics, Alpha Scores, Volatility Risk Premium (VRP) edges, N(d2) Probabilities of Profit (POP), and mathematically derived Support/Resistance levels provided by this platform are for informational and educational purposes only. Past performance does not guarantee future results.<br><br>
    <div style="text-align: center; font-size: 0.9em; color: #64748B;">
        &copy; 2026 vrpquant.com. All Rights Reserved.
    </div>
</div>
""", unsafe_allow_html=True)
