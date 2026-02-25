import sys
import subprocess
import importlib
import time
import gc 
import random
import os
import hashlib
import uuid
import json
import base64
import warnings
import concurrent.futures
from datetime import datetime
from pathlib import Path
from io import BytesIO

# ========================================================
# 🔧 WINDOWS ENCODING FIX
# ========================================================
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass 

# ========================================================
# 🛠️ DEPENDENCY CHECK
# ========================================================
def auto_install_dependencies():
    required_packages = {
        'pandas': 'pandas', 'numpy': 'numpy', 'requests': 'requests',
        'yfinance': 'yfinance', 'scipy': 'scipy', 'matplotlib': 'matplotlib',
        'tqdm': 'tqdm', 'selenium': 'selenium', 
        'webdriver_manager': 'webdriver_manager', 'PIL': 'Pillow'
    }
    missing = []
    for module, package in required_packages.items():
        try: importlib.import_module(module)
        except ImportError: missing.append(package)
    
    if missing:
        print(f"🚀 HQTA V22.0 Pipeline: Installing Dependencies: {', '.join(missing)}")
        try: 
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("✅ Dependencies installed.")
        except subprocess.CalledProcessError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", *missing])
            except:
                print(f"❌ ERROR: Run manually: pip install {' '.join(missing)}")
                sys.exit(1)

auto_install_dependencies()

# ========================================================
# IMPORTS & CONFIG
# ========================================================
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from PIL import Image

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# CONFIGURATION
VERSION = "v22.0 Institutional Pipeline"
MAX_WORKERS = 4 # Reduced to 4 to prevent Yahoo Finance IP Bans
TIMEOUT_SECONDS = 30 

# DIRECTORY STRUCTURE (FIXED: Now creates a unique folder per run)
run_timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = Path(f"./HQTA_WEEKLY_DROP_{run_timestamp_folder}")
BASE_DIR.mkdir(parents=True, exist_ok=True)

SUBSTACK_DIR = BASE_DIR / "1_SUBSTACK_READY"
META_DIR = BASE_DIR / "2_INSTAGRAM_FACEBOOK_READY"
XLINK_DIR = BASE_DIR / "3_TWITTER_LINKEDIN_READY"
AUDIT_DIR = BASE_DIR / "4_COMPLIANCE_LOGS"

for d in [SUBSTACK_DIR, META_DIR, XLINK_DIR, AUDIT_DIR]: d.mkdir(exist_ok=True)

rejection_reasons = { "Approved": 0, "Rejected": 0 }

# ========================================================
# 🧠 V22 INSTITUTIONAL MATH ENGINE 
# ========================================================

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
                
        plan['bias'] = bias
        return plan

# ========================================================
# 🎨 VISUAL ENGINE & BROWSER BOT
# ========================================================
class VisualEngine:
    def process_images(self, wide_screenshot_path, run_id):
        try:
            img = Image.open(wide_screenshot_path)
            wide_path_substack = SUBSTACK_DIR / f"Dashboard_Wide_{run_id}.png"
            wide_path_xlink = XLINK_DIR / f"Dashboard_Wide_{run_id}.png"
            img.save(wide_path_substack); img.save(wide_path_xlink)
            
            target_size = (1080, 1080); bg_color = (11, 14, 17) 
            canvas = Image.new("RGB", target_size, bg_color)
            w_percent = (target_size[0] / float(img.size[0]))
            h_size = int((float(img.size[1]) * float(w_percent)))
            img_resized = img.resize((target_size[0], h_size), Image.Resampling.LANCZOS)
            y_offset = (target_size[1] - h_size) // 2
            canvas.paste(img_resized, (0, y_offset))
            
            square_path = META_DIR / f"IG_Square_{run_id}.jpg"
            canvas.save(square_path, quality=95)
            return True
        except Exception as e:
            print(f"⚠️ Image Processing Error: {e}")
            return False

class BrowserBot:
    def screenshot(self, html_path, output_path):
        opts = Options()
        opts.add_argument("--headless"); opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--disable-gpu"); opts.add_argument("--no-sandbox")
        opts.add_argument("--log-level=3") 
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
        try:
            driver.get(f"file://{os.path.abspath(html_path)}")
            time.sleep(2)
            h = driver.execute_script("return document.body.parentNode.scrollHeight")
            driver.set_window_size(1920, h + 100)
            driver.save_screenshot(str(output_path))
        finally: driver.quit()

# ========================================================
# 📡 DATA WORKER
# ========================================================
def analyze_ticker(ticker):
    time.sleep(random.uniform(0.1, 0.5)) 
    try:
        stock = yf.Ticker(ticker.replace('.', '-'))
        hist = stock.history(period="2y")
        if hist.empty or len(hist) < 200: 
            rejection_reasons["Rejected"] += 1; return None
            
        price = hist['Close'].iloc[-1]
        
        # --- V22 MATH CALCULATIONS ---
        score = AlphaEngine.calculate_score(hist)
        win_rate, net_ret, outperf, max_dd = BacktestEngine.run_quick_backtest(hist)
        vol = QuantLogic.calculate_vol(hist)
        sup, res = QuantLogic.get_support_resistance(hist)
        vrp = QuantLogic.calculate_vrp_edge(hist)
        sharpe = QuantLogic.calculate_sharpe(hist)
        
        # We only want assets with distinct structural edges for our report
        if score > 40 and score < 60 and abs(vrp) < 2.0:
            rejection_reasons["Rejected"] += 1; return None

        plan = TradeArchitect.generate_plan(ticker, price, score, vol, sup, res)
        
        # --- TIER LOGIC ---
        tier = "SILVER"
        if score >= 75 or score <= 25: tier = "PLATINUM"
        elif score >= 65 or score <= 35: tier = "GOLD"

        # --- CHART GENERATOR ---
        chart_base64 = ""
        try:
            fig, ax = plt.subplots(figsize=(3.0, 1.0), dpi=75)
            data_slice = hist.tail(100)
            ax.plot(data_slice.index, data_slice['Close'], color='#00ff9d' if "LONG" in plan['bias'] else '#ff4d4d', linewidth=1.5)
            ax.axhline(y=sup, color='#63b3ed', linestyle=':', linewidth=0.8)
            ax.axhline(y=res, color='#f56565', linestyle=':', linewidth=0.8)
            ax.set_facecolor('#0b0e11'); fig.patch.set_facecolor('#0b0e11'); ax.axis('off')
            buf = BytesIO(); plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
            chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig); plt.clf(); gc.collect() 
        except: pass 

        rejection_reasons["Approved"] += 1
        return {
            "Ticker": ticker, "Price": price, "Score": score, "Trend": plan['bias'], 
            "Strategy": plan['name'], "Legs": plan['legs'], "Premium": plan['premium'],
            "Tier": tier, "VRP": vrp, "POP": plan['pop'], "Net_Return": net_ret, 
            "Max_DD": max_dd, "Sharpe": sharpe, "Support": sup, "Resistance": res,
            "Chart": chart_base64
        }
    except Exception as e: 
        rejection_reasons["Rejected"] += 1; return None

# ========================================================
# 📝 COPYWRITING ENGINE
# ========================================================
def generate_copy(df, run_id, timestamp, audit_hash):
    top = df.iloc[0]
    
    substack = f"""
# HQTA V22 Command Brief: {top['Ticker']} Edge Detected
**Run ID:** {run_id} | **Audit Hash:** {audit_hash[:16]}...
**Date:** {timestamp} EST

---

## 📊 Executive Summary
The V22.0 Jump-Diffusion Engine scanned {rejection_reasons['Approved'] + rejection_reasons['Rejected']} assets.
* **Signals Approved:** {rejection_reasons['Approved']}
* **Noise Rejected:** {rejection_reasons['Rejected']}

---
## 💎 Institutional Focus: {top['Ticker']}
* **Regime Score:** {top['Score']}/100 ({top['Trend']})
* **Execution Architecture:** {top['Strategy']} | {top['Legs']}
* **VRP Edge:** {top['VRP']:+.2f}% 
* **N(d2) Risk-Neutral POP:** {top['POP']}% 
* **Friction-Adjusted Max Drawdown:** {top['Max_DD']:.1f}%

[INSERT DASHBOARD IMAGE HERE]

### The Quant Breakdown:
Stop trading like retail. On {top['Ticker']}, our friction-adjusted backtests indicate a Net Strategy Return of **{top['Net_Return']:+.1f}%** with a Max Drawdown of **{top['Max_DD']:.1f}%**. Target execution is a **{top['Strategy']}** yielding **{top['Premium']}**.

Generate this math for any ticker. Get on the waitlist for the HQTA V22.0 Terminal today.
---
## ⚖️ Compliance Audit Log
This signal was generated algorithmically.
**Full SHA-256 Hash:** {audit_hash}
*Disclaimer: Quantitative research only. Not financial advice.*
    """
    with open(SUBSTACK_DIR / f"Substack_Text_{run_id}.md", "w", encoding="utf-8") as f: f.write(substack)

    meta = f"""
Institutional Math > Retail Emotion // {timestamp}

🔎 **Spotlight: ${top['Ticker']}**
The V22.0 terminal detected a structural anomaly.
• Regime: {top['Score']}/100 Alpha Score
• True Max Drawdown: {top['Max_DD']:.1f}%
• VRP Edge: {top['VRP']:+.2f}%
• Optimal Structuring: {top['Strategy']}
• N(d2) Prob of Profit: {top['POP']}%

The goal isn't to gamble. It's to operate like the house. 

👇 **ACCESS THE FULL DASHBOARD**
Link in Bio for the full report and to join the V22 Terminal Beta Waitlist.

#VRPQuant #QuantitativeTrading #OptionsTrading #{top['Ticker'].lower()}
    """
    with open(META_DIR / f"IG_FB_Caption_{run_id}.txt", "w", encoding="utf-8") as f: f.write(meta)

# ========================================================
# 🚀 MAIN COMMAND
# ========================================================
def get_sp500_tickers():
    defaults = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "META", "COIN"]
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        tickers = pd.read_html(BytesIO(r.content))[0]['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except: return defaults

def run_unified_command():
    run_id = str(uuid.uuid4())[:8].upper()
    print("\n" + "="*60)
    print(f" 🧬 HQTA V22.0 INSTITUTIONAL PIPELINE | RUN ID: {run_id}")
    print("="*60 + "\n")

    tickers = get_sp500_tickers()[:100] # Adjust batch size here
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_ticker, t): t for t in tickers}
        with tqdm(total=len(tickers), ncols=100, desc="⚙️ Processing V22 Math", unit="ticker") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try: 
                    res = future.result(timeout=TIMEOUT_SECONDS)
                    if res: results.append(res)
                except: pass
                finally: pbar.update(1)
    
    if results:
        tier_priority = {"PLATINUM": 0, "GOLD": 1, "SILVER": 2}
        df = pd.DataFrame(results)
        df['Rank'] = df['Tier'].map(tier_priority)
        df = df.sort_values(by=['Rank', 'Score'], ascending=[True, False]).drop('Rank', axis=1)
        
        table_rows = ""
        for _, row in df.iterrows():
            tier_class = f"badge-{row['Tier'].lower()}"
            trend_class = "text-bull" if "LONG" in row['Trend'] else "text-bear"
            table_rows += f"""<tr class="hover-row"><td class="ticker-cell">{row['Ticker']} <span class="price-sub">${row['Price']:.2f}</span></td><td><span class="{trend_class}">{row['Trend']}</span><br><span class="sub-text">Score: {row['Score']}/100</span></td><td><span class="badge {tier_class}">{row['Tier']}</span></td><td><span class="strat-text">{row['Strategy']}</span><br><span class="sub-text" style="color:#a0aec0; font-family:monospace; font-size:10px">{row['Legs']}</span></td><td><div class="data-box"><span class="highlight">{row['VRP']:+.2f}%</span></div></td><td><div class="progress-bar"><div class="fill" style="width:{row['POP']}%"></div></div><span class="sub-text">{row['POP']}%</span></td><td><span style="color:#ff4d4d">{row['Max_DD']:.1f}%</span></td><td>{row['Premium']}</td><td><span style="font-size:10px; color:#718096">S:{row['Support']}<br>R:{row['Resistance']}</span></td><td><img src="data:image/png;base64,{row['Chart']}" class="sparkline"></td></tr>"""

        html_content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>HQTA V22.0</title><style>:root {{ --bg: #0b0e11; --card: #151a21; --border: #2d3748; --text: #e2e8f0; --bull: #00ff9d; --bear: #ff4d4d; --accent: #3182ce; }} body {{ background-color: var(--bg); color: var(--text); font-family: 'Consolas', 'Monaco', monospace; margin: 0; padding: 20px; }} .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 2px solid var(--border); padding-bottom: 15px; }} .logo {{ font-size: 24px; font-weight: bold; letter-spacing: 2px; color: var(--accent); }} .meta {{ font-size: 12px; color: #718096; text-align: right; }} table {{ width: 100%; border-collapse: separate; border-spacing: 0 4px; }} th {{ text-align: left; font-size: 10px; color: #718096; text-transform: uppercase; padding: 10px; border-bottom: 1px solid var(--border); }} td {{ background-color: var(--card); padding: 10px; font-size: 13px; border-top: 1px solid #1e2530; border-bottom: 1px solid #1e2530; }} .ticker-cell {{ font-weight: bold; color: #fff; font-size: 15px; }} .price-sub {{ font-size: 11px; color: #718096; display: block; }} .sub-text {{ font-size: 10px; color: #718096; }} .text-bull {{ color: var(--bull); font-weight: bold; }} .text-bear {{ color: var(--bear); font-weight: bold; }} .strat-text {{ color: #63b3ed; font-weight: bold; }} .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; color: #000; }} .badge-platinum {{ background: linear-gradient(135deg, #e2e8f0, #a0aec0); box-shadow: 0 0 8px rgba(226, 232, 240, 0.5); }} .badge-gold {{ background: #ecc94b; }} .badge-silver {{ background: #cbd5e0; }} .data-box {{ background: rgba(255,255,255,0.05); padding: 4px; border-radius: 4px; text-align: center; }} .highlight {{ font-weight: bold; color: #fff; }} .progress-bar {{ width: 50px; height: 4px; background: #2d3748; border-radius: 2px; display: inline-block; margin-right: 5px; }} .fill {{ height: 100%; background: var(--accent); border-radius: 2px; }} .sparkline {{ height: 35px; vertical-align: middle; opacity: 0.9; width: 120px; }} .footer {{ margin-top: 30px; font-size: 10px; color: #4a5568; border-top: 1px solid var(--border); padding-top: 10px; }}</style></head><body><div class="header"><div class="logo">VRP QUANT <span style="color:#fff">V22.0</span> // INSTITUTIONAL</div><div class="meta">SCAN TIME: {datetime.now().strftime("%Y-%m-%d %H:%M")} EST<br>RUN ID: {run_id}<br>APPROVED: {len(df)}</div></div><table><thead><tr><th>Ticker</th><th>Regime</th><th>Alpha Tier</th><th>Strategy / Legs</th><th>VRP Edge</th><th>N(d2) POP</th><th>Max Drawdown</th><th>Pricing</th><th>Levels</th><th>Tech Chart</th></tr></thead><tbody>{table_rows}</tbody></table><div class="footer"><p><strong>INSTITUTIONAL DISCLAIMER:</strong> HQTA V22.0 COMMAND. COMPLIANCE AUDIT ID {run_id}.</p></div></body></html>"""
        
        html_path = BASE_DIR / f"Dashboard_{run_id}.html"
        with open(html_path, "w", encoding="utf-8") as f: f.write(html_content)
        
        print("📸 Capturing High-Res Dashboard...")
        raw_screenshot_path = BASE_DIR / f"Raw_Snap_{run_id}.png"
        bot = BrowserBot()
        bot.screenshot(html_path, raw_screenshot_path)
        
        print("🎨 Formatting Images for All Platforms...")
        viz = VisualEngine()
        viz.process_images(raw_screenshot_path, run_id)
        
        print("✍️ Writing Platform-Specific Copy...")
        data_manifest = json.dumps(df.to_dict(), sort_keys=True)
        audit_hash = hashlib.sha256(data_manifest.encode('utf-8')).hexdigest()
        generate_copy(df, run_id, datetime.now().strftime("%Y-%m-%d %H:%M"), audit_hash)
        
        with open(AUDIT_DIR / f"SEC_AUDIT_{datetime.now().strftime('%Y%m%d')}.log", "a", encoding="utf-8") as f:
            f.write(f"RUN: {run_id} | HASH: {audit_hash} | SIGNALS: {len(df)}\n")

        print("\n" + "="*60)
        print("✅ HQTA V22.0 WEEKLY PIPELINE COMPLETE.")
        print(f"📂 OUTPUT FOLDER: {BASE_DIR.absolute()}")
        print("="*60)
        
        if os.name == 'nt': os.startfile(BASE_DIR)
        elif os.name == 'posix': subprocess.call(['open', str(BASE_DIR)])
        
    else:
        print("\n❌ No signals found. Aborted.")

if __name__ == "__main__":
    run_unified_command()
