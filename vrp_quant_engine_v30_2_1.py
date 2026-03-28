import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
import time
import logging

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("⚠️ 'arch' library not found. Falling back to EWMA variance.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("V30_Options_Engine")

class OptionsDataEngine:
    @staticmethod
    def get_robust_chain(ticker, max_retries=3, base_delay=0.5):
        for attempt in range(max_retries):
            try:
                time.sleep(base_delay * (attempt + 1))
                tk = yf.Ticker(ticker)
                if not tk.options: return None
                chain = tk.option_chain(tk.options[0])
                if chain.calls.empty and chain.puts.empty: return None
                return chain
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"[{ticker}] Options ingestion failed after retries.")
                    return None
        return None

    @staticmethod
    def get_atm_iv(ticker, current_price):
        chain = OptionsDataEngine.get_robust_chain(ticker)
        if chain is None or chain.calls.empty: return np.nan
        try:
            calls = chain.calls
            calls['strike_dist'] = (calls['strike'] - current_price).abs()
            atm_call = calls.sort_values('strike_dist').iloc[0]
            if atm_call['bid'] > 0 and atm_call['ask'] > 0:
                iv_raw = atm_call.get('impliedVolatility', np.nan)
                if pd.isna(iv_raw) or iv_raw == 0.0: return np.nan
                return round(iv_raw * 100, 2)
            return np.nan
        except Exception:
            return np.nan

class AlphaEngine:
    @staticmethod
    def apply_kalman_filter(prices, noise_estimate=1.0, measure_noise=1.0):
        n = len(prices)
        kalman_gains, estimates = np.zeros(n), np.zeros(n)
        current_estimate, err_estimate = prices.iloc[0], noise_estimate
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
                res = am.fit(disp='off', show_warning=False, options={'maxiter': 200})
                return pd.Series(res.conditional_volatility / 100, index=scaled_rets.index) * np.sqrt(252)
            except: pass
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
            kalman_trend = calc_df['Kalman_Price'].iloc[-1] > calc_df['Kalman_Price'].iloc[-2]
            score = 50
            if current_price < calc_df['Lower_Band'].iloc[-1] and kalman_trend: score += 35
            elif current_price > calc_df['Upper_Band'].iloc[-1]: score -= 35
            bbw_z = (calc_df['BBW'].iloc[-1] - calc_df['BBW'].mean()) / (calc_df['BBW'].std() + 1e-9)
            score += max(-15, min(15, bbw_z * 10))
            return max(0, min(100, int(score)))
        except: return 50

class BacktestEngine:
    @staticmethod
    def run_wfo_backtest(df, slippage_bps=5, commission_bps=2):
        # (unchanged - exact same as you pasted)
        try:
            bt_df = df.copy().dropna()
            if len(bt_df) < 50: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, bt_df
            bt_df['Kalman_Price'] = AlphaEngine.apply_kalman_filter(bt_df['Close'])
            returns = bt_df['Close'].pct_change().fillna(0)
            bt_df['Vol_Regime'] = AlphaEngine.apply_garch(returns)
            bt_df['Underlying_Return'] = returns
            train_size, step_size = 252, 63
            multipliers = [1.5, 2.0, 2.5]
            oos_positions = pd.Series(0.0, index=bt_df.index, dtype=float)
            if len(bt_df) > train_size + step_size:
                for start_idx in range(0, len(bt_df) - train_size, step_size):
                    train_end, test_end = start_idx + train_size, min(start_idx + train_size + step_size, len(bt_df))
                    train_df = bt_df.iloc[start_idx:train_end]
                    best_sharpe, best_mult = -999, 2.0
                    for m in multipliers:
                        band_std = train_df['Vol_Regime'] * train_df['Kalman_Price'] / np.sqrt(252)
                        upper, lower = train_df['Kalman_Price'] + (m * band_std), train_df['Kalman_Price'] - (m * band_std)
                        sig = pd.Series(0, index=train_df.index)
                        sig[train_df['Close'] < lower] = 1
                        sig[train_df['Close'] > upper] = -1
                        pos = sig.replace(0, np.nan).ffill().fillna(0).shift(1).fillna(0)
                        strat_rets = pos * train_df['Underlying_Return']
                        sharpe = np.sqrt(252) * strat_rets.mean() / (strat_rets.std() + 1e-9)
                        if sharpe > best_sharpe: best_sharpe, best_mult = sharpe, m
                    test_df = bt_df.iloc[train_end:test_end]
                    band_std_oos = test_df['Vol_Regime'] * test_df['Kalman_Price'] / np.sqrt(252)
                    upper_oos, lower_oos = test_df['Kalman_Price'] + (best_mult * band_std_oos), test_df['Kalman_Price'] - (best_mult * band_std_oos)
                    sig_oos = pd.Series(0, index=test_df.index)
                    sig_oos[test_df['Close'] < lower_oos] = 1
                    sig_oos[test_df['Close'] > upper_oos] = -1
                    oos_pos = sig_oos.replace(0, np.nan).ffill().fillna(0)
                    oos_positions.iloc[train_end:test_end] = oos_pos.values
                bt_df['Target_Position'] = oos_positions
            else:
                band_std = bt_df['Vol_Regime'] * bt_df['Kalman_Price'] / np.sqrt(252)
                upper, lower = bt_df['Kalman_Price'] + (2 * band_std), bt_df['Kalman_Price'] - (2 * band_std)
                sig = pd.Series(0, index=bt_df.index)
                sig[bt_df['Close'] < lower] = 1
                sig[bt_df['Close'] > upper] = -1
                bt_df['Target_Position'] = sig.replace(0, np.nan).ffill().fillna(0)
            bt_df['Actual_Position'] = bt_df['Target_Position'].shift(1).fillna(0)
            bt_df['Gross_Return'] = bt_df['Actual_Position'] * bt_df['Underlying_Return']
            turnover = bt_df['Actual_Position'].diff().abs().fillna(0)
            total_cost = (slippage_bps + commission_bps) / 10000
            bt_df['Net_Return'] = bt_df['Gross_Return'] - (turnover * total_cost * (1 + (bt_df['Vol_Regime'] > 0.35).astype(int)))
            eval_df = bt_df.iloc[train_size:] if len(bt_df) > train_size + step_size else bt_df
            win_rate = (eval_df['Net_Return'] > 0).mean() * 100
            cumulative = (1 + eval_df['Net_Return']).prod() - 1
            outperf = cumulative - ((1 + eval_df['Underlying_Return']).prod() - 1)
            peak = (1 + eval_df['Net_Return']).cumprod().cummax()
            max_dd = (((1 + eval_df['Net_Return']).cumprod() - peak) / peak).min() * 100
            ann_return = eval_df['Net_Return'].mean() * 252
            sortino = ann_return / (eval_df[eval_df['Net_Return'] < 0]['Net_Return'].std() * np.sqrt(252) + 1e-9)
            calmar = ann_return / (abs(max_dd)/100 + 1e-9)
            wins, losses = eval_df[eval_df['Net_Return'] > 0]['Net_Return'], eval_df[eval_df['Net_Return'] < 0]['Net_Return']
            half_kelly = 0.0
            if len(wins) > 0 and len(losses) > 0 and abs(losses.mean()) > 0:
                win_prob = len(wins) / (len(wins) + len(losses))
                kelly_fraction = win_prob - ((1 - win_prob) / (wins.mean() / abs(losses.mean())))
                half_kelly = max(0.0, kelly_fraction / 2.0) * 100
            return round(win_rate,1), round(cumulative*100,1), round(outperf*100,1), round(max_dd,1), round(half_kelly,1), round(sortino, 2), round(calmar, 2), bt_df
        except: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, df.copy()

class QuantLogic:
    @staticmethod
    def calculate_vol(df): return df['Close'].pct_change().std() * np.sqrt(252) * 100

    @staticmethod
    def calculate_vrp_edge(ticker, df, mode="scanner"):
        if mode == "scanner":
            hv20 = df['Close'].pct_change().tail(20).std() * np.sqrt(252) * 100
            hv60 = df['Close'].pct_change().tail(60).std() * np.sqrt(252) * 100
            return round(hv20 - hv60, 2)
        else:
            hv = QuantLogic.calculate_vol(df)
            iv = OptionsDataEngine.get_atm_iv(ticker, df['Close'].iloc[-1])
            return round(iv - hv, 2) if not pd.isna(iv) else np.nan

    @staticmethod
    def get_support_resistance(df):
        return df['Low'].rolling(50).min().iloc[-1], df['High'].rolling(50).max().iloc[-1]

    @staticmethod
    def detect_reversal(df):
        try:
            if len(df) < 201: return "Insufficient Data"
            sma50, sma200 = df['Close'].rolling(50).mean(), df['Close'].rolling(200).mean()
            if sma50.iloc[-2] < sma200.iloc[-2] and sma50.iloc[-1] >= sma200.iloc[-1]: return "Golden Cross (Bull)"
            elif sma50.iloc[-2] > sma200.iloc[-2] and sma50.iloc[-1] <= sma200.iloc[-1]: return "Death Cross (Bear)"
            delta = df['Close'].diff()
            rs = (delta.where(delta > 0, 0)).rolling(14).mean() / ((-delta.where(delta < 0, 0)).rolling(14).mean() + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            if rsi.iloc[-2] < 30 and rsi.iloc[-1] >= 30: return "RSI Bull Bounce"
            elif rsi.iloc[-2] > 70 and rsi.iloc[-1] <= 70: return "RSI Bear Rejection"
            return "No Active Reversal"
        except:
            return "No Active Reversal"

    @staticmethod
    def calculate_sharpe(df, risk_free_rate=0.04):
        returns = df['Close'].pct_change().dropna()
        vol = returns.std() * np.sqrt(252)
        return round((returns.mean() * 252 - risk_free_rate) / vol, 2) if vol > 0 else 0

    @staticmethod
    def calculate_var(df, confidence=0.95):
        try: return round(df['Close'].iloc[-1] * (1 + np.percentile(df['Close'].pct_change().dropna(), (1 - confidence) * 100)), 2)
        except: return df['Close'].iloc[-1] * 0.95

    @staticmethod
    def calculate_upside_var(df, confidence=0.95):
        try: return round(df['Close'].iloc[-1] * (1 + np.percentile(df['Close'].pct_change().dropna(), confidence * 100)), 2)
        except: return df['Close'].iloc[-1] * 1.05

    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        pdf = norm.pdf(d1)
        return {'delta': round(norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1), 3),
                'gamma': round(pdf / (S * sigma * np.sqrt(T)), 4),
                'vega': round(S * pdf * np.sqrt(T), 2)}

    @staticmethod
    def bs_call(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return max(0, S - K)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))

    @staticmethod
    def bs_put(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return max(0, K - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return K * np.exp(-r * T) * norm.cdf(-(d1 - sigma * np.sqrt(T))) - S * norm.cdf(-d1)

class TradeArchitect:
    @staticmethod
    def generate_plan(ticker, price, score, vol, sup, res, half_kelly):
        plan = {}
        bias = "LONG (Bullish Trend)" if score >= 60 else "SHORT (Bearish Trend)" if score <= 40 else "NEUTRAL (Mean-Reverting)"
        vol_regime = "HIGH" if vol > 35 else "LOW"
        sigma, r, T30 = max(0.01, vol / 100), 0.04, 30 / 365
        res = price * 1.05 if res <= price else res
        sup = price * 0.95 if sup >= price else sup
        lower_wing, upper_wing = sup * 0.95, res * 1.05
        if "LONG" in bias:
            if vol_regime == "LOW":
                plan['name'], plan['legs'] = "Long Call Vertical", f"+C({price:.0f}) / -C({res:.0f})"
                debit = QuantLogic.bs_call(price, price, T30, r, sigma) - QuantLogic.bs_call(price, res, T30, r, sigma)
                plan['premium'] = f"Debit ${max(0.01, debit):.2f}"
                plan['pop'] = int(TradeArchitect.prob_itm(price, price + debit, T30, r, sigma, 'call') * 100)
            else:
                plan['name'], plan['legs'] = "Short Put Vertical", f"-P({sup:.0f}) / +P({lower_wing:.0f})"
                credit = QuantLogic.bs_put(price, sup, T30, r, sigma) - QuantLogic.bs_put(price, lower_wing, T30, r, sigma)
                plan['premium'] = f"Credit ${max(0.01, credit):.2f}"
                plan['pop'] = int((1 - TradeArchitect.prob_itm(price, sup, T30, r, sigma, 'put')) * 100)
        elif "SHORT" in bias:
            if vol_regime == "LOW":
                plan['name'], plan['legs'] = "Long Put Vertical", f"+P({price:.0f}) / -P({sup:.0f})"
                debit = QuantLogic.bs_put(price, price, T30, r, sigma) - QuantLogic.bs_put(price, sup, T30, r, sigma)
                plan['premium'] = f"Debit ${max(0.01, debit):.2f}"
                plan['pop'] = int(TradeArchitect.prob_itm(price, price - debit, T30, r, sigma, 'put') * 100)
            else:
                plan['name'], plan['legs'] = "Short Call Vertical", f"-C({res:.0f}) / +C({upper_wing:.0f})"
                credit = QuantLogic.bs_call(price, res, T30, r, sigma) - QuantLogic.bs_call(price, upper_wing, T30, r, sigma)
                plan['premium'] = f"Credit ${max(0.01, credit):.2f}"
                plan['pop'] = int((1 - TradeArchitect.prob_itm(price, res, T30, r, sigma, 'call')) * 100)
        else:
            plan['name'], plan['legs'] = "Iron Condor", f"+P({lower_wing:.0f}) / -P({sup:.0f}) | -C({res:.0f}) / +C({upper_wing:.0f})"
            credit = (QuantLogic.bs_put(price, sup, T30, r, sigma) - QuantLogic.bs_put(price, lower_wing, T30, r, sigma)) + (QuantLogic.bs_call(price, res, T30, r, sigma) - QuantLogic.bs_call(price, upper_wing, T30, r, sigma))
            plan['premium'] = f"Credit ${max(0.01, credit):.2f}"
            plan['pop'] = 65
        plan['greeks'] = QuantLogic.calculate_greeks(price, price, T30, r, sigma)
        plan['kelly_size'] = f"{int(max(5, min(50, half_kelly)))}% capital"
        plan['dte'], plan['bias'] = "30 Days", bias
        return plan

    @staticmethod
    def prob_itm(S, K, T, r, sigma, option_type='call'):
        if T <= 0 or sigma <= 0: return 0.0
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2)

    @staticmethod
    def generate_hybrid_plan(price, score, vrp, sup, res):
        if score >= 60:
            if vrp > 0:
                return {"name": "The Institutional Buy-Write (Yield Harvest)", "action": f"Buy Shares @ Market AND Sell 1 Call @ ${res:.2f}", "logic": "Trend is strong, options expensive. Buy stock, sell overpriced calls to lower risk."}
            else:
                return {"name": "The Bulletproof Bull (Protected Upside)", "action": f"Buy Shares @ Market AND Buy 1 Put @ ${sup:.2f}", "logic": "Trend is strong, options cheap. Ride stock up, buy cheap insurance at Support."}
        elif score <= 40:
            if vrp > 0:
                return {"name": "The Warren Buffett Entry (Discount Acquisition)", "action": f"Hold Cash AND Sell 1 Cash-Secured Put @ ${sup:.2f}", "logic": "Momentum weak, fear high. Sell puts to get paid while waiting to buy at floor."}
            else:
                return {"name": "The Smart-Money Short (Risk-Defined Bear)", "action": f"Do NOT Buy Stock. Buy Put Spread targeting ${sup:.2f}", "logic": "Momentum broken, options cheap. Use put options to profit from drop without shorting."}
        else:
            return {"name": "The Floor-to-Ceiling Swing (Mean Reversion)", "action": f"Limit Buy @ ${sup:.2f} AND Sell Target @ ${res:.2f}", "logic": "Stock trapped in channel. Refuse current prices. Trap at floor, sell at ceiling."}

class RegimeEngine:
    @staticmethod
    def detect_regime():
        try:
            spy_df = fetch_history("SPY", period="1y") if 'fetch_history' in globals() else yf.Ticker("SPY").history(period="1y")
            vix_df = fetch_history("^VIX", period="6mo") if 'fetch_history' in globals() else yf.Ticker("^VIX").history(period="6mo")
            vix9d_df = fetch_history("^VIX9D", period="5d") if 'fetch_history' in globals() else yf.Ticker("^VIX9D").history(period="5d")
            spy_close = spy_df['Close']
            vix_level = vix_df['Close'].iloc[-1]
            vix9d_level = vix9d_df['Close'].iloc[-1] if not vix9d_df.empty else vix_level
            ma50, ma200, price = spy_close.rolling(50).mean().iloc[-1], spy_close.rolling(200).mean().iloc[-1], spy_close.iloc[-1]
            if vix9d_level > vix_level + 2: return "Vol-Squeeze → Explosive Move Imminent"
            elif price > ma50 > ma200 and vix_level < 20: return "Risk-On"
            elif price < ma200 and vix_level > 25: return "Risk-Off"
            return "Neutral"
        except: return "Neutral"

def get_sparkline(data):
    """
    V30.2.1 CANDLE-STYLE SPARKLINE
    - Green █ = bullish candle (close > previous)
    - Red █  = bearish candle (close < previous)
    - Height scaled to move magnitude
    - Renders as colored inline candles in HTML table
    """
    if len(data) < 2:
        return "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁"
    
    bars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    changes = np.diff(data)                    # price change vs previous close
    abs_changes = np.abs(changes)
    max_change = max(abs_changes) if max(abs_changes) > 0 else 1.0
    
    spark = []
    # First point = neutral (no prior comparison)
    spark.append("<span style='color:#94A3B8;font-size:13px'>▁</span>")
    
    for i in range(1, len(data)):
        change = changes[i-1]
        height_idx = min(7, int((abs(change) / max_change) * 7))
        bar = bars[height_idx]
        
        if change >= 0:
            color = "#4ADE80"   # bull candle
        else:
            color = "#F87171"   # bear candle
        
        spark.append(f"<span style='color:{color};font-size:13px'>{bar}</span>")
    
    return "".join(spark)

class DynamicUniverseEngine:
    @staticmethod
    def get_apex_100():
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "LLY", "JPM", "V", "MA", "UNH", "XOM", "JNJ", "HD", "PG", "COST", "MRK", "ABBV", "CRM", "AMD", "BAC", "NFLX", "KO", "PEP", "TMO", "WMT", "DIS", "ADBE", "CSCO", "INTC", "QCOM", "TXN", "IBM", "NOW", "UBER", "PLTR", "WFC", "GS", "MS", "AXP", "C", "BLK", "SPGI", "HOOD", "SOFI", "COIN", "PYPL", "SQ", "AFRM", "UPST", "CVX", "COP", "SLB", "EOG", "MPC", "GE", "CAT", "BA", "RTX", "LMT", "HON", "UNP", "DE", "VLO", "OXY", "HAL", "GD", "NOC", "WM", "MCD", "NKE", "SBUX", "LOW", "ROKU", "DKNG", "SPOT", "SNOW", "CRWD", "PANW", "MSTR", "MARA", "RIOT", "CVNA", "SMCI", "ARM", "TSM", "ASML", "BTC-USD", "ETH-USD", "IBIT", "MU", "AMAT", "LRCX", "KLAC", "SYM", "DELL", "MNDY"]