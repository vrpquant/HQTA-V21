# =============================================================================
# vrp_quant_engine_v30_2_1.py
# VRP QUANT — CORE ENGINE PACKAGE — v6.5
# =============================================================================
# DIFF SUMMARY (vs Script 1 embedded engine):
#   [PKG-01] fetch_history() and LOCAL_CACHE now live HERE as the single
#            authoritative implementation. Scripts 2 & 3 import from here.
#   [PKG-02] get_candle_sparkline() migrated from Script 2 into package.
#   [PKG-03] ARCH_AVAILABLE flag exported so UI can import it cleanly.
#   [PKG-04] Full PEP-484 type hints added to all public methods.
#   [PKG-05] Google-style docstrings added to all classes and public methods.
#   [PKG-06] All bare `except:` → `except Exception:` for production hygiene.
#   [PKG-07] __all__ defined to control namespace when using `from pkg import *`.
# =============================================================================

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

try:
    from arch import arch_model
    ARCH_AVAILABLE: bool = True
except ImportError:
    ARCH_AVAILABLE = False
    print("⚠️ 'arch' library not found. Falling back to EWMA variance.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("V30_Options_Engine")

# [PKG-07] Explicit public API — controls `from vrp_quant_engine_v30_2_1 import *`
__all__ = [
    "ARCH_AVAILABLE",
    "fetch_history",
    "get_sparkline",
    "get_candle_sparkline",
    "OptionsDataEngine",
    "AlphaEngine",
    "BacktestEngine",
    "QuantLogic",
    "TradeArchitect",
    "RegimeEngine",
    "DynamicUniverseEngine",
]


# ---------------------------------------------------------------------------
# [PKG-01] SHARED DATA FETCHER — single source of truth for both scripts
# ---------------------------------------------------------------------------

LOCAL_CACHE: dict[str, pd.DataFrame] = {}


def fetch_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch OHLCV history via yfinance with an in-process LRU-style cache.

    Uses a module-level dict so both the pipeline runner (Script 2) and the
    Streamlit app (Script 3, which wraps this in @st.cache_data at call-site)
    share the same hot path.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. "AAPL", "BTC-USD").
        period:  yfinance period string (e.g. "2y", "1mo", "6mo").

    Returns:
        pd.DataFrame: OHLCV DataFrame. Empty DataFrame on failure.
    """
    cache_key = f"{ticker}::{period}"
    if cache_key not in LOCAL_CACHE:
        time.sleep(1.0)  # polite rate-limit guard
        try:
            LOCAL_CACHE[cache_key] = yf.Ticker(ticker).history(period=period)
        except Exception as e:
            logger.warning(f"[{ticker}] fetch_history failed: {e}")
            LOCAL_CACHE[cache_key] = pd.DataFrame()
    return LOCAL_CACHE[cache_key]


# ---------------------------------------------------------------------------
# SPARKLINE HELPERS
# ---------------------------------------------------------------------------

def get_sparkline(data: pd.Series | list) -> str:
    """
    Render a Unicode block-character sparkline from a 1-D price series.

    Args:
        data: Sequence of numeric values (e.g. closing prices).

    Returns:
        str: Unicode bar string, length == len(data).
    """
    bars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    s_min, s_max = min(data), max(data)
    if s_max == s_min:
        return bars[0] * len(data)
    scaled = np.clip(np.int_((np.array(data) - s_min) / (s_max - s_min) * 7), 0, 7)
    return "".join(bars[i] for i in scaled)


# [PKG-02] Migrated from Script 2 into the shared package
def get_candle_sparkline(df_subset: pd.DataFrame) -> str:
    """
    True OHLC sparkline: height maps to absolute Close level; colour maps to
    intraday direction (green = Close >= Open, red = Close < Open).

    Args:
        df_subset: DataFrame slice containing at minimum Open, High, Low, Close
                   columns. Typically the trailing 20 bars.

    Returns:
        str: HTML string of coloured Unicode block characters. Safe to render
             with ``unsafe_allow_html=True`` in Streamlit.
    """
    if len(df_subset) < 2:
        return "▁" * 20

    bars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    s_min = df_subset["Low"].min()
    s_max = df_subset["High"].max()
    spark: list[str] = []

    for _, row in df_subset.iterrows():
        idx = (
            0
            if s_max == s_min
            else int(np.clip((row["Close"] - s_min) / (s_max - s_min) * 7, 0, 7))
        )
        char = bars[idx]
        color = "#4ADE80" if row["Close"] >= row["Open"] else "#F87171"
        spark.append(f"<span style='color:{color};'>{char}</span>")

    return "".join(spark)


# ---------------------------------------------------------------------------
# OPTIONS DATA ENGINE
# ---------------------------------------------------------------------------

class OptionsDataEngine:
    """
    Resilient yfinance options chain ingestion with exponential-back-off retry.
    """

    @staticmethod
    def get_robust_chain(
        ticker: str,
        max_retries: int = 3,
        base_delay: float = 0.5,
    ) -> Optional[object]:
        """
        Fetch the nearest-expiry options chain with retry logic.

        Args:
            ticker:      Yahoo Finance ticker symbol.
            max_retries: Maximum number of fetch attempts before giving up.
            base_delay:  Base sleep duration (seconds); multiplied by attempt index.

        Returns:
            yfinance OptionChain object or None if all attempts fail.
        """
        for attempt in range(max_retries):
            try:
                time.sleep(base_delay * (attempt + 1))
                tk = yf.Ticker(ticker)
                if not tk.options:
                    return None
                chain = tk.option_chain(tk.options[0])
                if chain.calls.empty and chain.puts.empty:
                    return None
                return chain
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(
                        f"[{ticker}] Options ingestion failed after {max_retries} retries: {e}"
                    )
                    return None
        return None

    @staticmethod
    def get_atm_iv(ticker: str, current_price: float) -> float:
        """
        Return the at-the-money implied volatility (%) from the nearest-expiry
        call option chain.

        Args:
            ticker:        Yahoo Finance ticker symbol.
            current_price: Current underlying spot price.

        Returns:
            float: ATM IV in percentage points (e.g. 32.5), or np.nan on failure.
        """
        chain = OptionsDataEngine.get_robust_chain(ticker)
        if chain is None or chain.calls.empty:
            return np.nan
        try:
            calls = chain.calls.copy()
            calls["strike_dist"] = (calls["strike"] - current_price).abs()
            atm_call = calls.sort_values("strike_dist").iloc[0]
            if atm_call["bid"] > 0 and atm_call["ask"] > 0:
                iv_raw = atm_call.get("impliedVolatility", np.nan)
                if pd.isna(iv_raw) or iv_raw == 0.0:
                    return np.nan
                return round(iv_raw * 100, 2)
            return np.nan
        except Exception:
            return np.nan


# ---------------------------------------------------------------------------
# ALPHA ENGINE
# ---------------------------------------------------------------------------

class AlphaEngine:
    """
    Signal generation engine: Kalman filter trend estimation + GARCH(1,1)
    conditional volatility + composite Alpha Score.
    """

    @staticmethod
    def apply_kalman_filter(
        prices: pd.Series,
        noise_estimate: float = 1.0,
        measure_noise: float = 1.0,
    ) -> pd.Series:
        """
        Scalar Kalman filter (steady-state 1-D) applied to a price series.

        State model:  x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
        Observation:  z_t = x_t + v_t,        v_t ~ N(0, R)

        Args:
            prices:         pd.Series of closing prices (ordered oldest → newest).
            noise_estimate: Process noise covariance Q (controls filter lag).
            measure_noise:  Measurement noise covariance R (controls smoothing).

        Returns:
            pd.Series: Filtered price estimates, same index as ``prices``.
        """
        n = len(prices)
        kalman_gains = np.zeros(n)
        estimates = np.zeros(n)
        current_estimate = prices.iloc[0]
        err_estimate = noise_estimate

        for i in range(n):
            kalman_gains[i] = err_estimate / (err_estimate + measure_noise)
            current_estimate = current_estimate + kalman_gains[i] * (
                prices.iloc[i] - current_estimate
            )
            err_estimate = (1 - kalman_gains[i]) * err_estimate
            estimates[i] = current_estimate

        return pd.Series(estimates, index=prices.index)

    @staticmethod
    def apply_garch(returns: pd.Series) -> pd.Series:
        """
        Fit GARCH(1,1) and return the annualised conditional volatility series.

        Falls back to EWMA (span=20) if the ``arch`` library is unavailable or
        the optimiser diverges.

        Args:
            returns: pd.Series of arithmetic or log daily returns.

        Returns:
            pd.Series: Annualised conditional volatility, same index as the
                       non-NaN subset of ``returns``.
        """
        if ARCH_AVAILABLE and len(returns.dropna()) > 50:
            try:
                scaled = returns.dropna() * 100
                am = arch_model(scaled, vol="Garch", p=1, q=1, rescale=False)
                res = am.fit(disp="off", show_warning=False, options={"maxiter": 200})
                return (
                    pd.Series(res.conditional_volatility / 100, index=scaled.index)
                    * np.sqrt(252)
                )
            except Exception:
                pass
        # EWMA fallback
        return returns.ewm(span=20).std() * np.sqrt(252)

    @staticmethod
    def calculate_score(df: pd.DataFrame) -> int:
        """
        Compute the proprietary Alpha Score (0–100) combining:
          - Kalman trend direction
          - GARCH-derived Bollinger Band position
          - Bollinger Band Width (BBW) z-score for volatility regime

        Args:
            df: OHLCV DataFrame with at minimum a ``Close`` column and ≥ 50 rows.

        Returns:
            int: Alpha Score in [0, 100]. Returns 50 (neutral) on failure.
        """
        try:
            calc_df = df.copy().dropna()
            if len(calc_df) < 50:
                return 50

            calc_df["Kalman_Price"] = AlphaEngine.apply_kalman_filter(calc_df["Close"])
            returns = calc_df["Close"].pct_change().fillna(0)
            calc_df["GARCH_Vol"] = AlphaEngine.apply_garch(returns)

            calc_df["Band_Std"] = (
                calc_df["GARCH_Vol"] * calc_df["Kalman_Price"] / np.sqrt(252)
            )
            calc_df["Upper_Band"] = calc_df["Kalman_Price"] + 2 * calc_df["Band_Std"]
            calc_df["Lower_Band"] = calc_df["Kalman_Price"] - 2 * calc_df["Band_Std"]
            calc_df["BBW"] = (
                (calc_df["Upper_Band"] - calc_df["Lower_Band"]) / calc_df["Kalman_Price"]
            )

            current_price = calc_df["Close"].iloc[-1]
            kalman_trend = (
                calc_df["Kalman_Price"].iloc[-1] > calc_df["Kalman_Price"].iloc[-2]
            )

            score = 50
            if current_price < calc_df["Lower_Band"].iloc[-1] and kalman_trend:
                score += 35
            elif current_price > calc_df["Upper_Band"].iloc[-1]:
                score -= 35

            bbw_z = (calc_df["BBW"].iloc[-1] - calc_df["BBW"].mean()) / (
                calc_df["BBW"].std() + 1e-9
            )
            score += max(-15, min(15, bbw_z * 10))
            return max(0, min(100, int(score)))
        except Exception:
            return 50


# ---------------------------------------------------------------------------
# BACKTEST ENGINE
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Walk-Forward Optimisation (WFO) backtester with transaction cost modelling
    and a half-Kelly position-sizing recommendation.
    """

    @staticmethod
    def run_wfo_backtest(
        df: pd.DataFrame,
        slippage_bps: int = 5,
        commission_bps: int = 2,
    ) -> tuple[float, float, float, float, float, float, float, pd.DataFrame]:
        """
        Run an anchored Walk-Forward Optimisation backtest.

        Training window : 252 bars
        Step size       : 63 bars (quarterly re-optimisation)
        Optimised param : Bollinger Band multiplier ∈ {1.5, 2.0, 2.5}
        Cost model      : (slippage_bps + commission_bps) × turnover;
                          doubled during high-vol regimes (GARCH > 35%)

        Args:
            df:             OHLCV DataFrame with a ``Close`` column.
            slippage_bps:   One-way slippage assumption in basis points.
            commission_bps: One-way commission assumption in basis points.

        Returns:
            Tuple of:
                win_rate    (float): % of profitable trading days (OOS).
                cumulative  (float): Net cumulative return % (OOS).
                outperf     (float): Alpha vs buy-and-hold % (OOS).
                max_dd      (float): Maximum drawdown % (OOS, negative convention).
                half_kelly  (float): Half-Kelly fraction % for position sizing.
                sortino     (float): Annualised Sortino ratio (OOS).
                calmar      (float): Calmar ratio (OOS).
                bt_df       (pd.DataFrame): Full backtest DataFrame with band columns.
        """
        _null = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, df.copy())
        try:
            bt_df = df.copy().dropna()
            if len(bt_df) < 50:
                return _null

            bt_df["Kalman_Price"] = AlphaEngine.apply_kalman_filter(bt_df["Close"])
            returns = bt_df["Close"].pct_change().fillna(0)
            bt_df["Vol_Regime"] = AlphaEngine.apply_garch(returns)
            bt_df["Underlying_Return"] = returns

            train_size, step_size = 252, 63
            multipliers = [1.5, 2.0, 2.5]
            oos_positions = pd.Series(0.0, index=bt_df.index, dtype=float)

            if len(bt_df) > train_size + step_size:
                for start_idx in range(0, len(bt_df) - train_size, step_size):
                    train_end = start_idx + train_size
                    test_end = min(start_idx + train_size + step_size, len(bt_df))
                    train_df = bt_df.iloc[start_idx:train_end]

                    best_sharpe, best_mult = -999.0, 2.0
                    for m in multipliers:
                        band_std = (
                            train_df["Vol_Regime"] * train_df["Kalman_Price"] / np.sqrt(252)
                        )
                        upper = train_df["Kalman_Price"] + m * band_std
                        lower = train_df["Kalman_Price"] - m * band_std
                        sig = pd.Series(0, index=train_df.index)
                        sig[train_df["Close"] < lower] = 1
                        sig[train_df["Close"] > upper] = -1
                        pos = sig.replace(0, np.nan).ffill().fillna(0).shift(1).fillna(0)
                        strat_rets = pos * train_df["Underlying_Return"]
                        sharpe = np.sqrt(252) * strat_rets.mean() / (strat_rets.std() + 1e-9)
                        if sharpe > best_sharpe:
                            best_sharpe, best_mult = sharpe, m

                    test_df = bt_df.iloc[train_end:test_end]
                    band_std_oos = (
                        test_df["Vol_Regime"] * test_df["Kalman_Price"] / np.sqrt(252)
                    )
                    upper_oos = test_df["Kalman_Price"] + best_mult * band_std_oos
                    lower_oos = test_df["Kalman_Price"] - best_mult * band_std_oos
                    sig_oos = pd.Series(0, index=test_df.index)
                    sig_oos[test_df["Close"] < lower_oos] = 1
                    sig_oos[test_df["Close"] > upper_oos] = -1
                    oos_pos = sig_oos.replace(0, np.nan).ffill().fillna(0)
                    oos_positions.iloc[train_end:test_end] = oos_pos.values

                bt_df["Target_Position"] = oos_positions
            else:
                # Short series fallback: single in-sample pass
                band_std = bt_df["Vol_Regime"] * bt_df["Kalman_Price"] / np.sqrt(252)
                upper = bt_df["Kalman_Price"] + 2 * band_std
                lower = bt_df["Kalman_Price"] - 2 * band_std
                sig = pd.Series(0, index=bt_df.index)
                sig[bt_df["Close"] < lower] = 1
                sig[bt_df["Close"] > upper] = -1
                bt_df["Target_Position"] = sig.replace(0, np.nan).ffill().fillna(0)

            bt_df["Actual_Position"] = bt_df["Target_Position"].shift(1).fillna(0)
            bt_df["Gross_Return"] = bt_df["Actual_Position"] * bt_df["Underlying_Return"]
            turnover = bt_df["Actual_Position"].diff().abs().fillna(0)
            total_cost = (slippage_bps + commission_bps) / 10_000
            bt_df["Net_Return"] = bt_df["Gross_Return"] - (
                turnover
                * total_cost
                * (1 + (bt_df["Vol_Regime"] > 0.35).astype(int))
            )

            eval_df = (
                bt_df.iloc[train_size:]
                if len(bt_df) > train_size + step_size
                else bt_df
            )

            win_rate = (eval_df["Net_Return"] > 0).mean() * 100
            cumulative = (1 + eval_df["Net_Return"]).prod() - 1
            outperf = cumulative - ((1 + eval_df["Underlying_Return"]).prod() - 1)
            peak = (1 + eval_df["Net_Return"]).cumprod().cummax()
            max_dd = (
                ((1 + eval_df["Net_Return"]).cumprod() - peak) / peak
            ).min() * 100

            ann_return = eval_df["Net_Return"].mean() * 252
            downside = eval_df[eval_df["Net_Return"] < 0]["Net_Return"]
            sortino = ann_return / (downside.std() * np.sqrt(252) + 1e-9)
            calmar = ann_return / (abs(max_dd) / 100 + 1e-9)

            wins = eval_df[eval_df["Net_Return"] > 0]["Net_Return"]
            losses = eval_df[eval_df["Net_Return"] < 0]["Net_Return"]
            half_kelly = 0.0
            if len(wins) > 0 and len(losses) > 0 and abs(losses.mean()) > 0:
                win_prob = len(wins) / (len(wins) + len(losses))
                kelly_fraction = win_prob - (
                    (1 - win_prob) / (wins.mean() / abs(losses.mean()))
                )
                half_kelly = max(0.0, kelly_fraction / 2.0) * 100

            return (
                round(win_rate, 1),
                round(cumulative * 100, 1),
                round(outperf * 100, 1),
                round(max_dd, 1),
                round(half_kelly, 1),
                round(sortino, 2),
                round(calmar, 2),
                bt_df,
            )
        except Exception:
            return _null


# ---------------------------------------------------------------------------
# QUANT LOGIC
# ---------------------------------------------------------------------------

class QuantLogic:
    """
    Ancillary quantitative metrics: volatility, VRP edge, support/resistance,
    reversal detection, Sharpe, VaR, Black-Scholes Greeks.
    """

    @staticmethod
    def calculate_vol(df: pd.DataFrame) -> float:
        """
        Annualised close-to-close historical volatility (%).

        Args:
            df: OHLCV DataFrame with a ``Close`` column.

        Returns:
            float: Annualised volatility in percentage points.
        """
        return df["Close"].pct_change().std() * np.sqrt(252) * 100

    @staticmethod
    def calculate_vrp_edge(
        ticker: str,
        df: pd.DataFrame,
        mode: str = "scanner",
    ) -> float:
        """
        Compute the Volatility Risk Premium (VRP) edge.

        Scanner mode: HV20 − HV60 (proxy; no live options fetch)
        Deep-dive mode: ATM IV − HV252 (requires live options chain)

        Args:
            ticker: Yahoo Finance ticker symbol.
            df:     OHLCV DataFrame.
            mode:   ``"scanner"`` (fast proxy) or ``"deep_dive"`` (live IV).

        Returns:
            float: VRP edge in percentage points. Returns np.nan in deep-dive
                   mode when IV is unavailable.
        """
        if mode == "scanner":
            hv20 = df["Close"].pct_change().tail(20).std() * np.sqrt(252) * 100
            hv60 = df["Close"].pct_change().tail(60).std() * np.sqrt(252) * 100
            return round(hv20 - hv60, 2)
        else:
            hv = QuantLogic.calculate_vol(df)
            iv = OptionsDataEngine.get_atm_iv(ticker, df["Close"].iloc[-1])
            return round(iv - hv, 2) if not pd.isna(iv) else np.nan

    @staticmethod
    def get_support_resistance(df: pd.DataFrame) -> tuple[float, float]:
        """
        Rolling 50-bar support (Low min) and resistance (High max).

        Args:
            df: OHLCV DataFrame.

        Returns:
            tuple: (support_level, resistance_level)
        """
        support = df["Low"].rolling(50).min().iloc[-1]
        resistance = df["High"].rolling(50).max().iloc[-1]
        return support, resistance

    @staticmethod
    def detect_reversal(df: pd.DataFrame) -> str:
        """
        Identify the most recent structural reversal signal.

        Priority order:
          1. Golden Cross / Death Cross (SMA50 vs SMA200 crossover)
          2. RSI 30/70 bounce / rejection

        Args:
            df: OHLCV DataFrame with ≥ 201 rows.

        Returns:
            str: Human-readable reversal label.
        """
        try:
            if len(df) < 201:
                return "Insufficient Data"
            sma50 = df["Close"].rolling(50).mean()
            sma200 = df["Close"].rolling(200).mean()
            if sma50.iloc[-2] < sma200.iloc[-2] and sma50.iloc[-1] >= sma200.iloc[-1]:
                return "Golden Cross (Bull)"
            elif sma50.iloc[-2] > sma200.iloc[-2] and sma50.iloc[-1] <= sma200.iloc[-1]:
                return "Death Cross (Bear)"
            delta = df["Close"].diff()
            rs = (delta.where(delta > 0, 0)).rolling(14).mean() / (
                (-delta.where(delta < 0, 0)).rolling(14).mean() + 1e-9
            )
            rsi = 100 - 100 / (1 + rs)
            if rsi.iloc[-2] < 30 and rsi.iloc[-1] >= 30:
                return "RSI Bull Bounce"
            elif rsi.iloc[-2] > 70 and rsi.iloc[-1] <= 70:
                return "RSI Bear Rejection"
            return "No Active Reversal"
        except Exception:
            return "No Active Reversal"

    @staticmethod
    def calculate_sharpe(df: pd.DataFrame, risk_free_rate: float = 0.04) -> float:
        """
        Annualised Sharpe ratio.

        Args:
            df:               OHLCV DataFrame.
            risk_free_rate:   Annual risk-free rate (decimal, default 4%).

        Returns:
            float: Sharpe ratio, or 0.0 if volatility is zero.
        """
        returns = df["Close"].pct_change().dropna()
        vol = returns.std() * np.sqrt(252)
        return round((returns.mean() * 252 - risk_free_rate) / vol, 2) if vol > 0 else 0.0

    @staticmethod
    def calculate_var(df: pd.DataFrame, confidence: float = 0.95) -> float:
        """
        Historical simulation Value-at-Risk (downside) at the given confidence level.

        Args:
            df:         OHLCV DataFrame.
            confidence: Confidence level (e.g. 0.95 → 95% VaR).

        Returns:
            float: Price-level VaR.
        """
        try:
            return round(
                df["Close"].iloc[-1]
                * (1 + np.percentile(df["Close"].pct_change().dropna(), (1 - confidence) * 100)),
                2,
            )
        except Exception:
            return df["Close"].iloc[-1] * 0.95

    @staticmethod
    def calculate_upside_var(df: pd.DataFrame, confidence: float = 0.95) -> float:
        """
        Historical simulation Value-at-Risk (upside) — used for short-side risk.

        Args:
            df:         OHLCV DataFrame.
            confidence: Confidence level (e.g. 0.95 → 95% upside VaR).

        Returns:
            float: Price-level upside VaR.
        """
        try:
            return round(
                df["Close"].iloc[-1]
                * (1 + np.percentile(df["Close"].pct_change().dropna(), confidence * 100)),
                2,
            )
        except Exception:
            return df["Close"].iloc[-1] * 1.05

    @staticmethod
    def calculate_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> dict[str, float]:
        """
        Black-Scholes Delta, Gamma, and Vega.

        Args:
            S:           Spot price.
            K:           Strike price.
            T:           Time to expiry in years.
            r:           Continuously compounded risk-free rate.
            sigma:       Implied volatility (decimal, e.g. 0.30).
            option_type: ``"call"`` or ``"put"``.

        Returns:
            dict: Keys ``delta``, ``gamma``, ``vega``.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        pdf = norm.pdf(d1)
        return {
            "delta": round(norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1), 3),
            "gamma": round(pdf / (S * sigma * np.sqrt(T)), 4),
            "vega": round(S * pdf * np.sqrt(T), 2),
        }

    @staticmethod
    def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Black-Scholes European call price.

        Args:
            S:     Spot price.
            K:     Strike price.
            T:     Time to expiry (years).
            r:     Risk-free rate (decimal).
            sigma: Implied volatility (decimal).

        Returns:
            float: Theoretical call price.
        """
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Black-Scholes European put price.

        Args:
            S:     Spot price.
            K:     Strike price.
            T:     Time to expiry (years).
            r:     Risk-free rate (decimal).
            sigma: Implied volatility (decimal).

        Returns:
            float: Theoretical put price.
        """
        if T <= 0 or sigma <= 0:
            return max(0.0, K - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# TRADE ARCHITECT
# ---------------------------------------------------------------------------

class TradeArchitect:
    """
    Options structure selector and hybrid stock+options plan generator.
    """

    @staticmethod
    def generate_plan(
        ticker: str,
        price: float,
        score: int,
        vol: float,
        sup: float,
        res: float,
        half_kelly: float,
    ) -> dict:
        """
        Select and price the appropriate directional options strategy.

        Logic matrix:
          Bullish  + Low  Vol  → Long Call Vertical  (debit)
          Bullish  + High Vol  → Short Put Vertical   (credit)
          Bearish  + Low  Vol  → Long Put Vertical    (debit)
          Bearish  + High Vol  → Short Call Vertical  (credit)
          Neutral              → Iron Condor           (credit)

        Args:
            ticker:     Ticker symbol (unused in pricing but kept for logging).
            price:      Current spot price.
            score:      Alpha Score (0–100).
            vol:        Annualised historical volatility (%).
            sup:        50-bar Support level.
            res:        50-bar Resistance level.
            half_kelly: Half-Kelly position-sizing recommendation (%).

        Returns:
            dict: Strategy name, legs, theoretical premium, POP %, Greeks,
                  Kelly size, DTE, and directional bias string.
        """
        plan: dict = {}
        bias = (
            "LONG (Bullish Trend)"
            if score >= 60
            else "SHORT (Bearish Trend)"
            if score <= 40
            else "NEUTRAL (Mean-Reverting)"
        )
        vol_regime = "HIGH" if vol > 35 else "LOW"
        sigma = max(0.01, vol / 100)
        r, T30 = 0.04, 30 / 365

        res = price * 1.05 if res <= price else res
        sup = price * 0.95 if sup >= price else sup
        lower_wing = sup * 0.95
        upper_wing = res * 1.05

        if "LONG" in bias:
            if vol_regime == "LOW":
                plan["name"] = "Long Call Vertical"
                plan["legs"] = f"+C({price:.0f}) / -C({res:.0f})"
                debit = QuantLogic.bs_call(price, price, T30, r, sigma) - QuantLogic.bs_call(
                    price, res, T30, r, sigma
                )
                plan["premium"] = f"Debit ${max(0.01, debit):.2f}"
                plan["pop"] = int(
                    TradeArchitect.prob_itm(price, price + debit, T30, r, sigma, "call") * 100
                )
            else:
                plan["name"] = "Short Put Vertical"
                plan["legs"] = f"-P({sup:.0f}) / +P({lower_wing:.0f})"
                credit = QuantLogic.bs_put(price, sup, T30, r, sigma) - QuantLogic.bs_put(
                    price, lower_wing, T30, r, sigma
                )
                plan["premium"] = f"Credit ${max(0.01, credit):.2f}"
                plan["pop"] = int(
                    (1 - TradeArchitect.prob_itm(price, sup, T30, r, sigma, "put")) * 100
                )
        elif "SHORT" in bias:
            if vol_regime == "LOW":
                plan["name"] = "Long Put Vertical"
                plan["legs"] = f"+P({price:.0f}) / -P({sup:.0f})"
                debit = QuantLogic.bs_put(price, price, T30, r, sigma) - QuantLogic.bs_put(
                    price, sup, T30, r, sigma
                )
                plan["premium"] = f"Debit ${max(0.01, debit):.2f}"
                plan["pop"] = int(
                    TradeArchitect.prob_itm(price, price - debit, T30, r, sigma, "put") * 100
                )
            else:
                plan["name"] = "Short Call Vertical"
                plan["legs"] = f"-C({res:.0f}) / +C({upper_wing:.0f})"
                credit = QuantLogic.bs_call(price, res, T30, r, sigma) - QuantLogic.bs_call(
                    price, upper_wing, T30, r, sigma
                )
                plan["premium"] = f"Credit ${max(0.01, credit):.2f}"
                plan["pop"] = int(
                    (1 - TradeArchitect.prob_itm(price, res, T30, r, sigma, "call")) * 100
                )
        else:
            plan["name"] = "Iron Condor"
            plan["legs"] = (
                f"+P({lower_wing:.0f}) / -P({sup:.0f}) | "
                f"-C({res:.0f}) / +C({upper_wing:.0f})"
            )
            credit = (
                QuantLogic.bs_put(price, sup, T30, r, sigma)
                - QuantLogic.bs_put(price, lower_wing, T30, r, sigma)
            ) + (
                QuantLogic.bs_call(price, res, T30, r, sigma)
                - QuantLogic.bs_call(price, upper_wing, T30, r, sigma)
            )
            plan["premium"] = f"Credit ${max(0.01, credit):.2f}"
            plan["pop"] = 65

        plan["greeks"] = QuantLogic.calculate_greeks(price, price, T30, r, sigma)
        plan["kelly_size"] = f"{int(max(5, min(50, half_kelly)))}% capital"
        plan["dte"] = "30 Days"
        plan["bias"] = bias
        return plan

    @staticmethod
    def prob_itm(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """
        Risk-neutral probability of finishing in-the-money, N(d2).

        Args:
            S:           Spot price.
            K:           Strike price.
            T:           Time to expiry (years).
            r:           Risk-free rate (decimal).
            sigma:       Implied volatility (decimal).
            option_type: ``"call"`` or ``"put"``.

        Returns:
            float: Probability in [0, 1].
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d2) if option_type == "call" else norm.cdf(-d2)

    @staticmethod
    def generate_hybrid_plan(
        price: float,
        score: int,
        vrp: float,
        sup: float,
        res: float,
    ) -> dict[str, str]:
        """
        Generate the HQTA (Hybrid Quantitative Trade Architecture) directive.

        Combines directional trend bias with the VRP regime to select the
        optimal stock+options hybrid structure.

        Args:
            price: Current spot price.
            score: Alpha Score (0–100).
            vrp:   VRP Edge (IV − HV or HV20 − HV60, %).
            sup:   Support level.
            res:   Resistance level.

        Returns:
            dict: Keys ``name``, ``action``, ``logic``.
        """
        if score >= 60:
            if vrp > 0:
                return {
                    "name": "The Institutional Buy-Write (Yield Harvest)",
                    "action": f"Buy Shares @ Market AND Sell 1 Call @ ${res:.2f}",
                    "logic": (
                        "Trend is strong, options expensive. Buy stock, "
                        "sell overpriced calls to lower risk."
                    ),
                }
            else:
                return {
                    "name": "The Bulletproof Bull (Protected Upside)",
                    "action": f"Buy Shares @ Market AND Buy 1 Put @ ${sup:.2f}",
                    "logic": (
                        "Trend is strong, options cheap. Ride stock up, "
                        "buy cheap insurance at Support."
                    ),
                }
        elif score <= 40:
            if vrp > 0:
                return {
                    "name": "The Warren Buffett Entry (Discount Acquisition)",
                    "action": f"Hold Cash AND Sell 1 Cash-Secured Put @ ${sup:.2f}",
                    "logic": (
                        "Momentum weak, fear high. Sell puts to get paid "
                        "while waiting to buy at floor."
                    ),
                }
            else:
                return {
                    "name": "The Smart-Money Short (Risk-Defined Bear)",
                    "action": f"Do NOT Buy Stock. Buy Put Spread targeting ${sup:.2f}",
                    "logic": (
                        "Momentum broken, options cheap. Use put options to profit "
                        "from drop without shorting."
                    ),
                }
        else:
            return {
                "name": "The Floor-to-Ceiling Swing (Mean Reversion)",
                "action": f"Limit Buy @ ${sup:.2f} AND Sell Target @ ${res:.2f}",
                "logic": (
                    "Stock trapped in channel. Refuse current prices. "
                    "Trap at floor, sell at ceiling."
                ),
            }


# ---------------------------------------------------------------------------
# REGIME ENGINE
# ---------------------------------------------------------------------------

class RegimeEngine:
    """
    Macro market regime classifier using SPY trend and VIX term structure.
    """

    @staticmethod
    def detect_regime() -> str:
        """
        Classify the current macro regime into one of four states:
          - ``"Vol-Squeeze → Explosive Move Imminent"``
          - ``"Risk-On"``
          - ``"Risk-Off"``
          - ``"Neutral"``

        Logic:
          1. VIX9D > VIX + 2 → vol term-structure inversion → squeeze.
          2. SPY > SMA50 > SMA200 AND VIX < 20 → risk-on.
          3. SPY < SMA200 AND VIX > 25 → risk-off.
          4. Otherwise → neutral.

        Returns:
            str: Regime label string.
        """
        try:
            spy_df = fetch_history("SPY", period="1y")
            vix_df = fetch_history("^VIX", period="6mo")
            vix9d_df = fetch_history("^VIX9D", period="5d")

            spy_close = spy_df["Close"]
            vix_level = vix_df["Close"].iloc[-1]
            vix9d_level = (
                vix9d_df["Close"].iloc[-1] if not vix9d_df.empty else vix_level
            )

            ma50 = spy_close.rolling(50).mean().iloc[-1]
            ma200 = spy_close.rolling(200).mean().iloc[-1]
            price = spy_close.iloc[-1]

            if vix9d_level > vix_level + 2:
                return "Vol-Squeeze → Explosive Move Imminent"
            elif price > ma50 > ma200 and vix_level < 20:
                return "Risk-On"
            elif price < ma200 and vix_level > 25:
                return "Risk-Off"
            return "Neutral"
        except Exception:
            return "Neutral"


# ---------------------------------------------------------------------------
# DYNAMIC UNIVERSE ENGINE
# ---------------------------------------------------------------------------

class DynamicUniverseEngine:
    """
    Curated high-liquidity universe covering equities, ETFs, and crypto proxies.
    """

    @staticmethod
    def get_apex_100() -> list[str]:
        """
        Return the APEX-100 ticker universe.

        Returns:
            list[str]: 100 ticker symbols sorted by approximate institutional
                       liquidity preference.
        """
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO",
            "LLY", "JPM", "V", "MA", "UNH", "XOM", "JNJ", "HD", "PG", "COST",
            "MRK", "ABBV", "CRM", "AMD", "BAC", "NFLX", "KO", "PEP", "TMO",
            "WMT", "DIS", "ADBE", "CSCO", "INTC", "QCOM", "TXN", "IBM", "NOW",
            "UBER", "PLTR", "WFC", "GS", "MS", "AXP", "C", "BLK", "SPGI",
            "HOOD", "SOFI", "COIN", "PYPL", "SHOP", "AFRM", "UPST", "CVX", "COP",
            "SLB", "EOG", "MPC", "GE", "CAT", "BA", "RTX", "LMT", "HON", "UNP",
            "DE", "VLO", "OXY", "HAL", "GD", "NOC", "WM", "MCD", "NKE", "SBUX",
            "LOW", "ROKU", "DKNG", "SPOT", "SNOW", "CRWD", "PANW", "MSTR",
            "MARA", "RIOT", "CVNA", "SMCI", "ARM", "TSM", "ASML", "BTC-USD",
            "ETH-USD", "IBIT", "MU", "AMAT", "LRCX", "KLAC", "SYM", "DELL", "MNDY",
        ]

