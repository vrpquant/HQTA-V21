# =============================================================================
# VRP QUANT ENGINE — v30.5.1
# 5-Source Waterfall (Polygon + Alpha Vantage + Google Finance + Grok + yfinance)
# 200-step CRR Binomial (European + American) + 10k-100k MC + Merton Jump-Diffusion
# =============================================================================
#
# CHANGES FROM v30.5.0-A+:
#   [FIX-04] Grok model ID fallback chain + per-instance model cache
#   [FIX-05] Polygon docstring corrected (5 req/min free tier, not unlimited)
#   [FEAT-07] American-exercise CRR tree + dividend yield q parameter
#             (backward compatible: defaults to European)
#
# DATA WATERFALL (OHLCV History):
#   1. Polygon.io free tier  -> 5 req/min, EOD aggregates
#   2. Alpha Vantage         -> 500/day with free key
#   3. yfinance              -> EOD, throttled (ban-safe)
#
# DATA WATERFALL (Live/Delayed Quotes):
#   1. Polygon snapshot      -> 15-min delayed (free tier)
#   2. Alpha Vantage quote   -> real-time with key
#   3. Google Finance scrape -> no key required, real-time
#   4. Grok (xAI)            -> free $175/mo credits, real-time X data
#   5. yfinance last close   -> EOD fallback
#
# QUANT MODELS:
#   ✅ Black-Scholes full 5 Greeks (Δ,Γ,θ,ν,ρ) + dividend yield
#   ✅ 200-step CRR Binomial Tree — European AND American exercise
#   ✅ 10k–100k Monte Carlo with Merton Jump-Diffusion
#   ✅ Newton-Raphson IV solver
#   ✅ True POP with premium (not just P(ITM))
#   ✅ GARCH(1,1) conditional volatility + Kalman trend
#   ✅ Walk-Forward Optimization backtesting
#   ✅ Kelly Criterion sizing + vol-targeting + MVO
#
# SECRETS (all optional — graceful degradation):
#   POLYGON_API_KEY      — free at polygon.io
#   ALPHA_VANTAGE_KEY    — free at alphavantage.co
#   GROK_API_KEY         — free $175/mo at console.x.ai
# =============================================================================

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import norm

try:
    from arch import arch_model
    ARCH_AVAILABLE: bool = True
except ImportError:
    ARCH_AVAILABLE = False

logger = logging.getLogger("VRP_Quant_v30_5_1")


# =============================================================================
# THREAD-SAFE TTL CACHE — replaces unbounded LOCAL_CACHE dict
# =============================================================================

class _TTLCache:
    """
    Thread-safe in-memory cache with TTL expiry and max-entry eviction.

    Prevents unbounded memory growth during long Streamlit sessions.
    All public data-fetch functions write here. Evicts oldest entry
    when max_entries is reached.

    Args:
        default_ttl: Default time-to-live in seconds (900 = 15 min).
        max_entries: Maximum cached items before eviction kicks in.
    """

    def __init__(self, default_ttl: float = 900.0, max_entries: int = 500):
        self._ttl = default_ttl
        self._max = max_entries
        self._store: dict[str, tuple] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached value if still within TTL, else return None."""
        with self._lock:
            if key in self._store:
                val, exp = self._store[key]
                if time.monotonic() < exp:
                    return val
                del self._store[key]
        return None

    def set(self, key: str, value, ttl: Optional[float] = None) -> None:
        """Store a value with optional custom TTL. Evicts oldest if at capacity."""
        with self._lock:
            if len(self._store) >= self._max and key not in self._store:
                oldest = min(self._store, key=lambda k: self._store[k][1])
                del self._store[oldest]
            self._store[key] = (value, time.monotonic() + (ttl or self._ttl))

    def clear(self) -> None:
        """Evict all entries."""
        with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, int]:
        """Return live/stale entry counts for sidebar display."""
        with self._lock:
            now = time.monotonic()
            live = sum(1 for _, (__, exp) in self._store.items() if now < exp)
            return {"live_entries": live, "stale_entries": len(self._store) - live}


# Module-level cache instance — thread-safe, TTL-bounded
_CACHE = _TTLCache(default_ttl=900.0, max_entries=500)

# Backward compatibility alias (app.py may reference LOCAL_CACHE for .clear())
LOCAL_CACHE = _CACHE


# =============================================================================
# QUANTMATH — Full institutional math engine
# =============================================================================

class QuantMath:
    """
    Production-grade quantitative mathematics engine.

    Provides Black-Scholes pricing with all 5 Greeks (Δ, Γ, θ, ν, ρ),
    CRR binomial tree (200 steps default, European or American exercise),
    Monte Carlo simulation (10k–100k paths) with optional Merton
    jump-diffusion, Newton-Raphson IV solver, true POP with premium,
    and full P&L analytics.

    All static methods — no instance state required.
    """

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF via math.erf (no scipy dependency)."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def black_scholes(
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: str = "put", q: float = 0.0,
    ) -> dict[str, float]:
        """
        Analytical Black-Scholes pricing with all 5 Greeks and dividend yield.

        Args:
            S:     Spot price.
            K:     Strike price.
            T:     Time to expiry in years (e.g. 30/365).
            r:     Risk-free rate (decimal, e.g. 0.05).
            sigma: Implied volatility (decimal, e.g. 0.25).
            option_type: 'call' or 'put'.
            q:     Continuous dividend yield (decimal, default 0).

        Returns:
            dict with keys: price, delta, gamma, theta ($/day), vega (per 1%),
            rho (per 1%).
        """
        if T <= 1e-10:
            intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
            delta = (1.0 if S > K else 0.0) if option_type == "call" else (-1.0 if S < K else 0.0)
            return {"price": intrinsic, "delta": delta, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

        S_adj = S * math.exp(-q * T)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S_adj / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        Nd1 = QuantMath._norm_cdf(d1)
        Nd2 = QuantMath._norm_cdf(d2)
        pdf_d1 = QuantMath._norm_pdf(d1)

        if option_type == "call":
            price = S_adj * Nd1 - K * math.exp(-r * T) * Nd2
            delta = math.exp(-q * T) * Nd1
            rho_val = K * T * math.exp(-r * T) * Nd2 / 100
        else:
            price = K * math.exp(-r * T) * QuantMath._norm_cdf(-d2) - S_adj * QuantMath._norm_cdf(-d1)
            delta = math.exp(-q * T) * (Nd1 - 1)
            rho_val = -K * T * math.exp(-r * T) * QuantMath._norm_cdf(-d2) / 100

        gamma = math.exp(-q * T) * pdf_d1 / (S * sigma * sqrtT)
        vega = S * math.exp(-q * T) * pdf_d1 * sqrtT / 100
        theta = QuantMath.full_theta(S, K, T, r, sigma, option_type, q)

        return {
            "price": round(price, 4), "delta": round(delta, 4),
            "gamma": round(gamma, 6), "theta": round(theta, 4),
            "vega": round(vega, 4), "rho": round(rho_val, 4),
        }

    @staticmethod
    def full_theta(
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: str = "put", q: float = 0.0,
    ) -> float:
        """
        Daily theta ($/calendar-day) with dividend yield.

        Uses the analytical BS theta formula divided by 365.
        Negative for long options (time decay).
        """
        if T <= 1e-10:
            return 0.0
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        pdf_d1 = QuantMath._norm_pdf(d1)
        Nd1 = QuantMath._norm_cdf(d1)
        Nd2 = QuantMath._norm_cdf(d2)
        first = -(S * math.exp(-q * T) * pdf_d1 * sigma) / (2 * sqrtT)
        if option_type == "call":
            theta = first - r * K * math.exp(-r * T) * Nd2 + q * S * math.exp(-q * T) * Nd1
        else:
            theta = first + r * K * math.exp(-r * T) * (1 - Nd2) - q * S * math.exp(-q * T) * (1 - Nd1)
        return theta / 365.0

    @staticmethod
    def crr_binomial_tree(
        S: float, K: float, T: float, r: float, sigma: float,
        n_steps: int = 200, option_type: str = "put",
        exercise: str = "european", q: float = 0.0,
    ) -> dict[str, float | int | str]:
        """
        CRR (Cox-Ross-Rubinstein) binomial tree — European or American.

        Vectorised backward induction. For American exercise, at each node
        we take max(continuation_value, intrinsic_value). The European price
        is always computed in parallel so the early-exercise premium
        (American − European) can be reported directly.

        Dividend yield q enters the risk-neutral probability:
            p = (e^((r−q)·dt) − d) / (u − d)
        When q = 0 this reduces to the standard (e^(r·dt) − d)/(u − d).

        Args:
            S, K, T, r, sigma: standard BS inputs.
            n_steps:           tree depth (200 → <0.1% of BS for European).
            option_type:       'call' or 'put'.
            exercise:          'european' (default) or 'american'.
            q:                 continuous dividend yield (decimal).

        Returns:
            dict with:
              price                   — price under chosen exercise style
              euro_price              — always computed for reference
              early_exercise_premium  — max(0, american − european)
              pop                     — P(S_T beyond K) from terminal dist.
              n_steps                 — tree depth used
              exercise                — normalised 'european' | 'american'

        Note on POP: the terminal-distribution POP is exercise-invariant
        because it only asks where S_T lands, not when payoff is taken.
        For short-seller P&L POP (with premium), use true_pop_with_premium().
        """
        exercise = exercise.lower()
        if exercise not in ("european", "american"):
            raise ValueError(
                f"exercise must be 'european' or 'american', got {exercise!r}"
            )

        if T <= 1e-10:
            intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
            return {
                "price": round(intrinsic, 4),
                "euro_price": round(intrinsic, 4),
                "early_exercise_premium": 0.0,
                "pop": 0.0,
                "n_steps": n_steps,
                "exercise": exercise,
            }

        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        # Dividend-adjusted risk-neutral probability
        p = (np.exp((r - q) * dt) - d) / (u - d)
        discount = np.exp(-r * dt)
        is_american = exercise == "american"

        # Terminal stock prices and payoffs
        j = np.arange(n_steps + 1)
        ST = S * (u ** (n_steps - j)) * (d ** j)
        if option_type == "call":
            payoff = np.maximum(ST - K, 0)
        else:
            payoff = np.maximum(K - ST, 0)
        euro_payoff = payoff.copy()

        # Backward induction — always compute European for reference
        for i in range(n_steps):
            # European: pure continuation
            euro_payoff = discount * (p * euro_payoff[:-1] + (1 - p) * euro_payoff[1:])

            # American: max(continuation, immediate exercise)
            cont = discount * (p * payoff[:-1] + (1 - p) * payoff[1:])
            if is_american:
                # Stock prices at time step k = n_steps − i − 1
                k = n_steps - i - 1
                j_k = np.arange(k + 1)
                S_k = S * (u ** (k - j_k)) * (d ** j_k)
                if option_type == "call":
                    intrinsic = np.maximum(S_k - K, 0)
                else:
                    intrinsic = np.maximum(K - S_k, 0)
                payoff = np.maximum(cont, intrinsic)
            else:
                payoff = cont

        price = float(payoff[0])
        euro_price = float(euro_payoff[0])
        early_ex_premium = max(0.0, price - euro_price)
        # POP is exercise-invariant (terminal-distribution question)
        pop = float(np.mean(ST > K)) if option_type == "put" else float(np.mean(ST < K))

        return {
            "price": round(price, 4),
            "euro_price": round(euro_price, 4),
            "early_exercise_premium": round(early_ex_premium, 6),
            "pop": round(pop, 4),
            "n_steps": n_steps,
            "exercise": exercise,
        }

    @staticmethod
    def monte_carlo_paths(
        S: float, T: float, r: float, sigma: float,
        n_paths: int = 100_000, n_steps: int = 252,
        option_type: str = "put", strike: Optional[float] = None,
        jump_diffusion: bool = False,
        lambda_j: float = 0.15, mu_j: float = -0.03, sigma_j: float = 0.15,
        seed: int = 42,
    ) -> dict:
        """
        Monte Carlo path simulation with optional Merton jump-diffusion.

        Returns full paths matrix (n_paths × n_steps), terminal prices,
        mean path, 5/95 confidence bands, and POP estimate.

        Jump-diffusion adds compound Poisson jumps to GBM with a
        drift compensator (λκ) to preserve risk-neutral pricing.

        Args:
            n_paths:    Number of simulation paths (10k–100k).
            n_steps:    Time steps per path (252 = daily for 1yr).
            jump_diffusion: Enable Merton jumps if True.
            lambda_j:   Jump intensity (jumps/year).
            mu_j:       Mean log-jump size (negative = downward bias).
            sigma_j:    Jump size volatility.
            seed:       RNG seed for reproducibility.

        Returns:
            dict with: pop, pop_pct, mean_terminal, n_paths, S_T, paths,
            mean_path, conf_lower, conf_upper.
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        vol = sigma * np.sqrt(dt)
        Z = rng.standard_normal((n_paths, n_steps))

        if jump_diffusion:
            # Merton JD: drift compensated per-step for risk-neutral pricing
            # dS/S = (r - ½σ² - λκ)dt + σdW + JdN
            # κ = E[e^J - 1] = exp(μ_j + ½σ_j²) - 1
            kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1
            drift_per_step = (r - 0.5 * sigma**2 - lambda_j * kappa) * dt
            N_jumps = rng.poisson(lambda_j * dt, (n_paths, n_steps))
            J = rng.normal(mu_j * N_jumps, sigma_j * np.sqrt(np.maximum(N_jumps, 0)))
            log_paths = np.cumsum(drift_per_step + vol * Z + J, axis=1)
        else:
            drift_per_step = (r - 0.5 * sigma**2) * dt
            log_paths = np.cumsum(drift_per_step + vol * Z, axis=1)

        paths = S * np.exp(log_paths)
        S_T = paths[:, -1]
        if strike is None:
            strike = S

        # POP for SHORT seller: short put profits when S_T > strike, short call when S_T < strike
        pop = float(np.mean(S_T > strike)) if option_type == "put" else float(np.mean(S_T < strike))
        mean_path = np.mean(paths, axis=0)
        conf_lower = np.percentile(paths, 5, axis=0)
        conf_upper = np.percentile(paths, 95, axis=0)

        return {
            "pop_pct": f"{pop * 100:.1f}%", "pop": round(pop, 4),
            "mean_terminal": round(float(np.mean(S_T)), 2), "n_paths": n_paths,
            "S_T": S_T, "paths": paths, "mean_path": mean_path,
            "conf_lower": conf_lower, "conf_upper": conf_upper,
        }

    @staticmethod
    def implied_volatility(
        S: float, K: float, T: float, r: float, market_price: float,
        option_type: str = "put", tol: float = 1e-5, max_iter: int = 100,
    ) -> Optional[float]:
        """
        Newton-Raphson implied volatility solver.

        Iterates on sigma until BS price matches market_price within tol.
        Returns None if convergence fails after max_iter iterations.
        Clamped to [1%, 500%] to prevent numerical instability.
        """
        sigma = 0.2
        for _ in range(max_iter):
            bs = QuantMath.black_scholes(S, K, T, r, sigma, option_type)
            diff = bs["price"] - market_price
            if abs(diff) < tol:
                return round(sigma, 6)
            vega_raw = S * QuantMath._norm_pdf(
                (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            ) * math.sqrt(T)
            if vega_raw < 1e-10:
                break
            sigma -= diff / vega_raw
            sigma = max(0.01, min(sigma, 5.0))
        return None

    @staticmethod
    def true_pop_with_premium(
        S_T: np.ndarray, K: float, premium: float, option_type: str = "put",
    ) -> float:
        """
        True POP for the SHORT option seller, accounting for premium received.

        Short seller's P&L = premium_received - payoff_to_buyer
          Short put seller:  PnL = premium - max(K - S_T, 0)  → profit when S_T > K - premium
          Short call seller: PnL = premium - max(S_T - K, 0)  → profit when S_T < K + premium

        POP = P(PnL > 0) = fraction of MC paths where seller keeps net profit.

        This is the correct metric for VRP Quant's premium-selling strategies.
        For the LONG buyer's POP, negate the result: buyer_pop = 1 - seller_pop.
        """
        if option_type == "call":
            # Short call: collect premium, pay out if S_T > K
            pnl = premium - np.maximum(S_T - K, 0)
        else:
            # Short put: collect premium, pay out if S_T < K
            pnl = premium - np.maximum(K - S_T, 0)
        return round(float(np.mean(pnl > 0)), 4)

    @staticmethod
    def option_pnl(
        S_T: np.ndarray, K: float, premium: float, option_type: str = "put",
    ) -> dict[str, float]:
        """
        Full P&L analytics for the SHORT option seller from MC terminal distribution.

        Short put breakeven:  K - premium  (seller loses below this)
        Short call breakeven: K + premium  (seller loses above this)

        Returns breakeven price, expected P&L (seller's perspective),
        and profit probability (seller's perspective).
        """
        if option_type == "call":
            # Short call: collect premium, pay out max(S_T - K, 0)
            pnl = premium - np.maximum(S_T - K, 0)
            breakeven = K + premium  # seller breaks even here
        else:
            # Short put: collect premium, pay out max(K - S_T, 0)
            pnl = premium - np.maximum(K - S_T, 0)
            breakeven = K - premium  # seller breaks even here
        return {
            "breakeven": round(breakeven, 4),
            "expected_pnl": round(float(np.mean(pnl)), 4),
            "profit_prob": round(float(np.mean(pnl > 0)), 4),
        }


# =============================================================================
# POLYGON.IO CLIENT — NBBO quotes, options chain, news, OHLCV history
# =============================================================================

class PolygonClient:
    """
    Polygon.io REST client — free tier (5 req/min, EOD only).

    Capabilities (free tier):
      ✅ Daily OHLCV aggregates (5 req/min rate-limited, ~2yr history cap)
      ✅ 15-min delayed NBBO snapshots (same 5 req/min limit)
      ✅ Ticker reference + news
      ❌ Real-time websocket (requires Starter $29/mo)
      ❌ Options chain with Greeks (requires Options add-on)

    Rate-limit behaviour: 429 responses return None; waterfall falls
    through to Alpha Vantage. No backoff retry — intentional for scan speed.
    Scans of 60+ tickers will exhaust the quota and silently cascade.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            try:
                import streamlit as st
                self.api_key = st.secrets.get("POLYGON_API_KEY")
            except Exception:
                pass
        self.base_url = "https://api.polygon.io"

    @property
    def available(self) -> bool:
        """True if a valid API key is configured."""
        return bool(self.api_key)

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        """Rate-limited GET with timeout. Returns None on any failure."""
        if not self.available:
            return None
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key
        try:
            resp = requests.get(url, params=params, timeout=12)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"Polygon {resp.status_code} on {endpoint}")
            return None
        except Exception as e:
            logger.warning(f"Polygon request error: {e}")
            return None

    def get_live_quote(self, ticker: str) -> dict:
        """Fetch NBBO quote snapshot (15-min delayed on free tier)."""
        data = self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
        if not data or data.get("status") != "OK":
            return {"price": None, "source": "polygon:failed"}
        snap = data.get("ticker", {})
        last = snap.get("lastTrade", {})
        return {
            "price": last.get("p"), "bid": snap.get("lastQuote", {}).get("P"),
            "ask": snap.get("lastQuote", {}).get("p"), "source": "polygon:nbbo",
            "fetched_at": datetime.now(timezone.utc),
        }

    def get_history(self, ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV aggregates. Returns yfinance-compatible DataFrame."""
        from datetime import timedelta
        period_map = {"5d": 5, "1mo": 32, "3mo": 95, "6mo": 185, "1y": 366, "2y": 730, "5y": 1827}
        days = period_map.get(period, 730)
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=days)
        poly_ticker = ticker
        if ticker.endswith("-USD"):
            poly_ticker = f"X:{ticker.replace('-USD', '')}USD"
        data = self._get(
            f"/v2/aggs/ticker/{poly_ticker}/range/1/day/{from_dt.strftime('%Y-%m-%d')}/{to_dt.strftime('%Y-%m-%d')}",
            params={"adjusted": "true", "sort": "asc", "limit": 50000},
        )
        if not data or data.get("resultsCount", 0) == 0:
            return None
        results = data.get("results", [])
        if not results:
            return None
        df = pd.DataFrame(results)
        df["Date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.set_index("Date").rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()

    def get_options_chain(self, ticker: str, limit: int = 50) -> dict:
        """Fetch options chain snapshot with Greeks (requires paid plan)."""
        data = self._get(f"/v3/snapshot/options/{ticker}", {"limit": limit})
        if not data or not data.get("results"):
            return {"contracts": [], "source": "polygon:failed"}
        contracts = []
        for item in data["results"]:
            greeks = item.get("greeks", {})
            contracts.append({
                "strike": item.get("details", {}).get("strike_price"),
                "type": item.get("details", {}).get("contract_type"),
                "iv": item.get("implied_volatility"),
                "delta": greeks.get("delta"), "gamma": greeks.get("gamma"),
                "theta": greeks.get("theta"), "vega": greeks.get("vega"),
                "bid": item.get("last_quote", {}).get("bid"),
                "ask": item.get("last_quote", {}).get("ask"),
                "open_interest": item.get("open_interest"),
            })
        return {"contracts": contracts, "source": "polygon:chain"}

    def get_news(self, ticker: str, limit: int = 10) -> list:
        """Fetch recent news articles for ticker."""
        data = self._get("/v2/reference/news", {"ticker": ticker, "limit": limit})
        return data.get("results", []) if data else []


# =============================================================================
# ALPHA VANTAGE CLIENT — free tier (25 req/day, 500 req/day with key)
# =============================================================================

class AlphaVantageClient:
    """
    Alpha Vantage REST client for OHLCV history and live quotes.

    Free tier: 25 API calls/day (no key required for some endpoints).
    Standard key: 500/day. Premium: unlimited.

    Used as fallback #2 in the waterfall after Polygon.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_KEY")
        if not self.api_key:
            try:
                import streamlit as st
                self.api_key = st.secrets.get("ALPHA_VANTAGE_KEY")
            except Exception:
                pass
        self.base_url = "https://www.alphavantage.co/query"

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def get_history(self, ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Fetch daily adjusted OHLCV. Returns yfinance-compatible DataFrame."""
        if not self.available:
            return None
        try:
            outputsize = "compact" if period in ("5d", "1mo", "3mo") else "full"
            resp = requests.get(self.base_url, params={
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker.replace("-USD", ""),
                "outputsize": outputsize,
                "apikey": self.api_key,
            }, timeout=15)
            data = resp.json()
            ts = data.get("Time Series (Daily)")
            if not ts:
                return None
            df = pd.DataFrame.from_dict(ts, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.rename(columns={
                "1. open": "Open", "2. high": "High", "3. low": "Low",
                "4. close": "Close", "6. volume": "Volume",
            })
            for col in ("Open", "High", "Low", "Close", "Volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            logger.info(f"[{ticker}] Alpha Vantage: {len(df)} bars")
            return df if len(df) >= 5 else None
        except Exception as e:
            logger.warning(f"[{ticker}] Alpha Vantage failed: {e}")
            return None

    def get_live_quote(self, ticker: str) -> dict:
        """Fetch real-time quote via GLOBAL_QUOTE endpoint."""
        if not self.available:
            return {"price": None, "source": "alphavantage:unavailable"}
        try:
            resp = requests.get(self.base_url, params={
                "function": "GLOBAL_QUOTE",
                "symbol": ticker.replace("-USD", ""),
                "apikey": self.api_key,
            }, timeout=10)
            data = resp.json().get("Global Quote", {})
            price = data.get("05. price")
            if price:
                return {
                    "price": float(price),
                    "source": "alphavantage:quote",
                    "fetched_at": datetime.now(timezone.utc),
                }
            return {"price": None, "source": "alphavantage:empty"}
        except Exception as e:
            logger.warning(f"[{ticker}] Alpha Vantage quote failed: {e}")
            return {"price": None, "source": "alphavantage:failed"}


# =============================================================================
# GOOGLE FINANCE SCRAPER — no API key required (fallback #3)
# =============================================================================

class GoogleFinanceClient:
    """
    Google Finance quote scraper — no API key needed.

    Tries multiple exchanges (NASDAQ, NYSE, NYSEARCA, CRYPTO) to find
    the ticker. Returns actual success status for sidebar display.
    """

    _last_success: bool = False  # Track actual scrape success for sidebar

    @staticmethod
    def get_live_quote(ticker: str) -> dict:
        """Scrape current price from Google Finance across multiple exchanges."""
        try:
            import re
            clean = ticker.replace("-USD", "").replace("^", "")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0"}

            # Try multiple exchanges — crypto, ETFs, and OTC need different routes
            exchanges = ["NASDAQ", "NYSE", "NYSEARCA", "CRYPTO"]
            if ticker.endswith("-USD"):
                exchanges = ["CRYPTO", "NASDAQ"]  # Crypto first

            for exchange in exchanges:
                try:
                    url = f"https://www.google.com/finance/quote/{clean}:{exchange}"
                    resp = requests.get(url, headers=headers, timeout=8)
                    if resp.status_code != 200:
                        continue
                    match = re.search(r'data-last-price="([\d.]+)"', resp.text)
                    if match:
                        price = float(match.group(1))
                        GoogleFinanceClient._last_success = True
                        return {
                            "price": price,
                            "source": f"google_finance:{exchange}",
                            "fetched_at": datetime.now(timezone.utc),
                        }
                except Exception:
                    continue

            GoogleFinanceClient._last_success = False
            return {"price": None, "source": "google_finance:no_match"}
        except Exception as e:
            GoogleFinanceClient._last_success = False
            logger.warning(f"[{ticker}] Google Finance scrape failed: {e}")
            return {"price": None, "source": "google_finance:failed"}


# =============================================================================
# GROK (xAI) FINANCE CLIENT — AI-powered real-time data (fallback #4)
# =============================================================================

class GrokFinanceClient:
    """
    Real-time financial data via xAI Grok API.

    Uses the OpenAI-compatible endpoint at api.x.ai/v1. xAI has shipped
    multiple naming conventions across Grok 4 / 4.1 Fast releases (dotted
    vs dashed); this client probes a candidate list in order and pins the
    first working model on the instance for subsequent calls.

    Pricing (April 2026): $0.20/M input, $0.50/M output for Grok 4.x Fast.
    Free credits: $25 signup + $150/mo with data sharing enabled = $175/mo.

    Requires GROK_API_KEY (or XAI_API_KEY). Sign up free at console.x.ai.
    Used as fallback #4 — cheaper than Perplexity, has real-time X data access.
    Degrades gracefully if openai package or API key unavailable.
    """

    GROK_BASE_URL = "https://api.x.ai/v1"
    # xAI has shipped multiple naming conventions across Grok releases.
    # We try each in order and cache the first one that returns valid JSON.
    # All Grok 4 Fast variants price at $0.20/M in, $0.50/M out (April 2026).
    GROK_MODEL_CANDIDATES: tuple[str, ...] = (
        "grok-4-fast-non-reasoning",       # Grok 4 Fast (Sep 2025, stable)
        "grok-4-1-fast-non-reasoning",     # Grok 4.1 Fast, dashed convention
        "grok-4.1-fast-non-reasoning",     # Grok 4.1 Fast, dotted convention
        "grok-2-1212",                     # stable fallback
    )

    def __init__(self, api_key: Optional[str] = None):
        self._key = api_key or os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
        if not self._key:
            try:
                import streamlit as st
                self._key = st.secrets.get("GROK_API_KEY") or st.secrets.get("XAI_API_KEY")
            except Exception:
                pass
        self._client = None
        self._working_model: Optional[str] = None  # cache first successful model ID
        try:
            from openai import OpenAI
            if self._key:
                self._client = OpenAI(
                    api_key=self._key,
                    base_url=self.GROK_BASE_URL,
                    timeout=20,
                )
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._client is not None

    def get_live_quote(self, ticker: str) -> dict:
        """
        Fetch current price via Grok AI (real-time X/web access).

        Probes GROK_MODEL_CANDIDATES in order; caches first working model
        on the instance so subsequent calls skip directly to it.
        """
        if not self.available:
            return {"price": None, "source": "grok:unavailable"}

        import json as _json
        candidates = (
            (self._working_model,) if self._working_model
            else self.GROK_MODEL_CANDIDATES
        )

        for model in candidates:
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": (
                            "You are a financial data engine. "
                            "Return ONLY valid JSON. No prose. No markdown."
                        )},
                        {"role": "user", "content": (
                            f"Current stock price of {ticker} right now. "
                            f"Return JSON: {{\"price\": float, \"data_as_of\": string}}"
                        )},
                    ],
                    max_tokens=100,
                    temperature=0.0,
                )
                text = resp.choices[0].message.content.strip()
                text = text.replace("```json", "").replace("```", "").strip()
                data = _json.loads(text)
                price = data.get("price")
                if price and float(price) > 0:
                    self._working_model = model  # pin for this instance
                    return {
                        "price": float(price),
                        "source": f"grok:{model}",
                        "fetched_at": datetime.now(timezone.utc),
                    }
            except Exception as e:
                logger.debug(f"[{ticker}] Grok model '{model}' failed: {e}")
                continue

        return {"price": None, "source": "grok:all_models_failed"}


# =============================================================================
# 5-SOURCE WATERFALL DATA LAYER (ALL FREE)
# =============================================================================
# Priority for OHLCV history:
#   1. Polygon.io free tier (5 req/min, EOD aggregates)
#   2. Alpha Vantage (500/day with free key)
#   3. yfinance (EOD, throttled to avoid bans)
#
# Priority for live/delayed quotes:
#   1. Polygon snapshot (15-min delayed, free tier)
#   2. Alpha Vantage GLOBAL_QUOTE (real-time with free key)
#   3. Google Finance scrape (no key, real-time)
#   4. Grok xAI (free $175/mo credits, real-time X data)
#   5. yfinance last close (EOD fallback)
#
# All results written to _CACHE (TTL 15 min for history, 60s for quotes).
# =============================================================================

_polygon_client = PolygonClient()
_alphavantage_client = AlphaVantageClient()
_google_client = GoogleFinanceClient()
_grok_client = GrokFinanceClient()


def fetch_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch OHLCV history via 3-source waterfall with TTL cache.

    Priority: Polygon -> Alpha Vantage -> yfinance
    Returns empty DataFrame on total failure (never raises).
    """
    cache_key = f"hist::{ticker}::{period}"
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Source 1: Polygon (free tier, EOD)
    if _polygon_client.available:
        try:
            df = _polygon_client.get_history(ticker, period)
            if df is not None and not df.empty and len(df) >= 5:
                _CACHE.set(cache_key, df)
                logger.info(f"[{ticker}] History from Polygon ({len(df)} bars)")
                return df
        except Exception as e:
            logger.warning(f"[{ticker}] Polygon history failed: {e}")

    # Source 2: Alpha Vantage
    if _alphavantage_client.available:
        try:
            df = _alphavantage_client.get_history(ticker, period)
            if df is not None and not df.empty and len(df) >= 5:
                _CACHE.set(cache_key, df)
                logger.info(f"[{ticker}] History from Alpha Vantage ({len(df)} bars)")
                return df
        except Exception as e:
            logger.warning(f"[{ticker}] Alpha Vantage history failed: {e}")

    # Source 3: yfinance (EOD, throttled)
    time.sleep(0.5)
    try:
        df = yf.Ticker(ticker).history(period=period)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            _CACHE.set(cache_key, df)
            logger.info(f"[{ticker}] History from yfinance ({len(df)} bars)")
        return df
    except Exception as e:
        logger.warning(f"[{ticker}] yfinance history failed: {e}")
        return pd.DataFrame()


def fetch_live_price(ticker: str) -> dict:
    """
    Fetch current price via 5-source waterfall.

    Priority: Polygon -> Alpha Vantage -> Google Finance -> Grok -> yfinance
    Returns dict with: price, source, fetched_at.
    """
    cache_key = f"quote::{ticker}"
    cached = _CACHE.get(cache_key)
    if cached is not None and isinstance(cached, dict) and cached.get("price"):
        return cached

    # Source 1: Polygon NBBO (15-min delayed, free tier)
    if _polygon_client.available:
        q = _polygon_client.get_live_quote(ticker)
        if q.get("price") and float(q["price"]) > 0:
            _CACHE.set(cache_key, q, ttl=60)
            return q

    # Source 2: Alpha Vantage GLOBAL_QUOTE
    if _alphavantage_client.available:
        q = _alphavantage_client.get_live_quote(ticker)
        if q.get("price") and float(q["price"]) > 0:
            _CACHE.set(cache_key, q, ttl=60)
            return q

    # Source 3: Google Finance (no key, scrape)
    q = _google_client.get_live_quote(ticker)
    if q.get("price") and float(q["price"]) > 0:
        _CACHE.set(cache_key, q, ttl=60)
        return q

    # Source 4: Grok xAI (free $175/mo credits, real-time X data)
    if _grok_client.available:
        q = _grok_client.get_live_quote(ticker)
        if q.get("price") and float(q["price"]) > 0:
            _CACHE.set(cache_key, q, ttl=60)
            return q

    # Source 5: yfinance last close (EOD fallback)
    df = fetch_history(ticker, period="5d")
    if not df.empty:
        price = float(df["Close"].iloc[-1])
        result = {
            "price": price,
            "source": "yfinance:eod_fallback",
            "fetched_at": datetime.now(timezone.utc),
        }
        _CACHE.set(cache_key, result, ttl=60)
        return result

    return {"price": None, "source": "all_sources_failed", "fetched_at": datetime.now(timezone.utc)}


def fetch_risk_free_rate() -> tuple[float, str]:
    """Fetch 3-month T-Bill yield from ^IRX. Returns (rate_decimal, source_string)."""
    try:
        df = fetch_history("^IRX", period="5d")
        if not df.empty:
            rate = df["Close"].iloc[-1] / 100.0
            return rate, f"^IRX ({df.index[-1].date()})"
    except Exception:
        pass
    return 0.05, "default 5%"


def data_layer_status() -> dict:
    """Return status of all 5 data sources for sidebar display."""
    return {
        "polygon_active": _polygon_client.available,
        "alphavantage_active": _alphavantage_client.available,
        "google_finance": GoogleFinanceClient._last_success,  # Reflects actual scrape success
        "grok_active": _grok_client.available,
        "yfinance": True,  # Always available
        "cache_stats": _CACHE.stats(),
    }


# =============================================================================
# OPTIONS DATA ENGINE
# =============================================================================

class OptionsDataEngine:
    """yfinance options chain fetcher with retry and ATM IV extraction."""

    @staticmethod
    def get_robust_chain(ticker: str, max_retries: int = 3, base_delay: float = 0.5):
        """Fetch nearest-expiry options chain with exponential backoff retry."""
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
            except Exception:
                pass
        return None

    @staticmethod
    def get_atm_iv(ticker: str, current_price: float) -> float:
        """Extract ATM implied volatility (%) from nearest-expiry call chain."""
        chain = OptionsDataEngine.get_robust_chain(ticker)
        if chain is None or chain.calls.empty:
            return np.nan
        try:
            calls = chain.calls.copy()
            calls["strike_dist"] = (calls["strike"] - current_price).abs()
            atm = calls.sort_values("strike_dist").iloc[0]
            iv_raw = atm.get("impliedVolatility", np.nan)
            return round(float(iv_raw) * 100, 2) if not pd.isna(iv_raw) and iv_raw > 0 else np.nan
        except Exception:
            return np.nan


# =============================================================================
# ALPHA ENGINE — Kalman + GARCH + Alpha Score
# =============================================================================

class AlphaEngine:
    """
    Trend extraction via Kalman filter + GARCH(1,1) conditional volatility.

    Produces an Alpha Score [0–100] combining Kalman trend direction,
    GARCH band positioning, and Bollinger Band Width z-score.
    """

    @staticmethod
    def apply_kalman_filter(
        prices: pd.Series, noise: float = 1.0, measure: float = 1.0,
    ) -> pd.Series:
        """Scalar Kalman filter on price series. Returns smoothed estimate."""
        n = len(prices)
        est = np.zeros(n)
        curr = prices.iloc[0]
        err = noise
        for i in range(n):
            kg = err / (err + measure)
            curr += kg * (prices.iloc[i] - curr)
            err = (1 - kg) * err
            est[i] = curr
        return pd.Series(est, index=prices.index)

    @staticmethod
    def apply_garch(returns: pd.Series) -> pd.Series:
        """
        Fit GARCH(1,1) with Student-t innovations. Returns annualised conditional vol.

        Falls back to EWMA (span=20) if arch library unavailable or MLE fails.
        """
        clean = returns.dropna()
        if ARCH_AVAILABLE and len(clean) > 50:
            try:
                scaled = clean * 100
                am = arch_model(scaled, vol="Garch", p=1, q=1, rescale=False)
                res = am.fit(disp="off", show_warning=False, options={"maxiter": 300})
                return pd.Series(res.conditional_volatility.values / 100 * np.sqrt(252), index=scaled.index)
            except Exception as e:
                logger.warning(f"GARCH failed ({e}), EWMA fallback")
        return clean.ewm(span=20).std() * np.sqrt(252)

    @staticmethod
    def calculate_score(df: pd.DataFrame) -> int:
        """
        Alpha Score [0–100] from Kalman trend + GARCH bands + BBW z-score.

        Score > 60 -> bullish bias. Score < 40 -> bearish bias.
        Returns 50 (neutral) if insufficient data (<50 bars).
        """
        try:
            calc = df.copy().dropna()
            if len(calc) < 50:
                return 50
            calc["Kalman_Price"] = AlphaEngine.apply_kalman_filter(calc["Close"])
            rets = calc["Close"].pct_change().fillna(0)
            calc["GARCH_Vol"] = AlphaEngine.apply_garch(rets)
            calc["Band_Std"] = calc["GARCH_Vol"] * calc["Kalman_Price"] / np.sqrt(252)
            calc["Upper_Band"] = calc["Kalman_Price"] + 2 * calc["Band_Std"]
            calc["Lower_Band"] = calc["Kalman_Price"] - 2 * calc["Band_Std"]
            calc["BBW"] = (calc["Upper_Band"] - calc["Lower_Band"]) / calc["Kalman_Price"]
            price = calc["Close"].iloc[-1]
            trend = calc["Kalman_Price"].iloc[-1] > calc["Kalman_Price"].iloc[-2]
            score = 50
            if price < calc["Lower_Band"].iloc[-1] and trend:
                score += 35
            elif price > calc["Upper_Band"].iloc[-1]:
                score -= 35
            bbw_z = (calc["BBW"].iloc[-1] - calc["BBW"].mean()) / (calc["BBW"].std() + 1e-9)
            score += max(-15, min(15, bbw_z * 10))
            return max(0, min(100, int(score)))
        except Exception:
            return 50


# =============================================================================
# BACKTEST ENGINE — Walk-Forward Optimization
# =============================================================================

class BacktestEngine:
    """
    Anchored Walk-Forward Optimisation backtester.

    Train: 252 bars. Step: 63 bars. Optimises BB multiplier {1.5, 2.0, 2.5}.
    Applies slippage + commission cost model. Uses shift(1) on positions
    to prevent look-ahead bias.
    """

    @staticmethod
    def run_wfo_backtest(
        df: pd.DataFrame, slippage_bps: int = 5, commission_bps: int = 2,
    ) -> tuple[float, float, float, float, float, float, float, pd.DataFrame]:
        """
        Run WFO backtest. Returns (win_rate, cum_return, outperf, max_dd,
        half_kelly, sortino, calmar, backtest_df).
        """
        _null = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, df.copy())
        try:
            bt = df.copy().dropna()
            if len(bt) < 50:
                return _null
            bt["Kalman_Price"] = AlphaEngine.apply_kalman_filter(bt["Close"])
            rets = bt["Close"].pct_change().fillna(0)
            bt["Vol_Regime"] = AlphaEngine.apply_garch(rets)
            bt["Underlying_Return"] = rets

            train_sz, step_sz = 252, 63
            mults = [1.5, 2.0, 2.5]
            oos_pos = pd.Series(0.0, index=bt.index, dtype=float)

            if len(bt) > train_sz + step_sz:
                for start in range(0, len(bt) - train_sz, step_sz):
                    train_end = start + train_sz
                    test_end = min(train_end + step_sz, len(bt))
                    train = bt.iloc[start:train_end]
                    best_sh, best_m = -999.0, 2.0
                    for m in mults:
                        bstd = train["Vol_Regime"] * train["Kalman_Price"] / np.sqrt(252)
                        u = train["Kalman_Price"] + m * bstd
                        l = train["Kalman_Price"] - m * bstd
                        sig = pd.Series(0, index=train.index)
                        sig[train["Close"] < l] = 1
                        sig[train["Close"] > u] = -1
                        pos = sig.replace(0, np.nan).ffill().fillna(0).shift(1).fillna(0)
                        sr = pos * train["Underlying_Return"]
                        sh = np.sqrt(252) * sr.mean() / (sr.std() + 1e-9)
                        if sh > best_sh:
                            best_sh, best_m = sh, m
                    test = bt.iloc[train_end:test_end]
                    bstd = test["Vol_Regime"] * test["Kalman_Price"] / np.sqrt(252)
                    u = test["Kalman_Price"] + best_m * bstd
                    l = test["Kalman_Price"] - best_m * bstd
                    sig = pd.Series(0, index=test.index)
                    sig[test["Close"] < l] = 1
                    sig[test["Close"] > u] = -1
                    oos_pos.iloc[train_end:test_end] = sig.replace(0, np.nan).ffill().fillna(0).values
                bt["Target_Position"] = oos_pos
            else:
                bstd = bt["Vol_Regime"] * bt["Kalman_Price"] / np.sqrt(252)
                u = bt["Kalman_Price"] + 2 * bstd
                l = bt["Kalman_Price"] - 2 * bstd
                sig = pd.Series(0, index=bt.index)
                sig[bt["Close"] < l] = 1
                sig[bt["Close"] > u] = -1
                bt["Target_Position"] = sig.replace(0, np.nan).ffill().fillna(0)

            bt["Actual_Position"] = bt["Target_Position"].shift(1).fillna(0)
            bt["Gross_Return"] = bt["Actual_Position"] * bt["Underlying_Return"]
            turnover = bt["Actual_Position"].diff().abs().fillna(0)
            cost = (slippage_bps + commission_bps) / 10_000
            bt["Net_Return"] = bt["Gross_Return"] - turnover * cost * (1 + (bt["Vol_Regime"] > 0.35).astype(int))

            ev = bt.iloc[train_sz:] if len(bt) > train_sz + step_sz else bt
            wr = float((ev["Net_Return"] > 0).mean() * 100)
            cum = float((1 + ev["Net_Return"]).prod() - 1)
            outperf = float(cum - ((1 + ev["Underlying_Return"]).prod() - 1))
            peak = (1 + ev["Net_Return"]).cumprod().cummax()
            dd_series = ((1 + ev["Net_Return"]).cumprod() - peak) / peak
            mdd = float(dd_series.min() * 100) if len(ev) > 1 else 0.0
            ann_ret = ev["Net_Return"].mean() * 252
            ds = ev[ev["Net_Return"] < 0]["Net_Return"]
            sortino = float(ann_ret / (ds.std() * np.sqrt(252) + 1e-9))
            calmar = float(ann_ret / (abs(mdd) / 100 + 1e-9))
            wins = ev[ev["Net_Return"] > 0]["Net_Return"]
            losses = ev[ev["Net_Return"] < 0]["Net_Return"]
            hk = 0.0
            if len(wins) > 0 and len(losses) > 0 and abs(losses.mean()) > 0:
                wp = len(wins) / (len(wins) + len(losses))
                kf = wp - (1 - wp) / (wins.mean() / abs(losses.mean()))
                hk = float(max(0.0, kf / 2.0) * 100)
            return (round(wr, 1), round(cum * 100, 1), round(outperf * 100, 1),
                    round(mdd, 1), round(hk, 1), round(sortino, 2), round(calmar, 2), bt)
        except Exception as e:
            logger.warning(f"WFO failed: {e}")
            return _null


# =============================================================================
# QUANT LOGIC — VRP, S/R, Reversal, Sharpe, VaR, Greeks
# =============================================================================

class QuantLogic:
    """
    Quantitative analytics: VRP edge, support/resistance, reversal detection,
    Sharpe ratio, Value-at-Risk, and Black-Scholes convenience wrappers.
    """

    @staticmethod
    def calculate_vol(df: pd.DataFrame) -> float:
        """Annualised close-to-close historical volatility (%)."""
        return float(df["Close"].pct_change().std() * np.sqrt(252) * 100)

    @staticmethod
    def calculate_vrp_edge(
        ticker: str, df: pd.DataFrame, mode: str = "scanner",
        iv_pct: Optional[float] = None,
    ) -> float:
        """
        Volatility Risk Premium: VRP = IV - RV.

        Scanner mode uses VIX as IV proxy for speed.
        Deep-dive mode uses actual ATM IV from options chain.
        Positive VRP = options are expensive = selling edge.
        """
        log_rets = df["Close"].pct_change().dropna()
        rv_20d = float(log_rets.tail(20).std() * np.sqrt(252) * 100)
        if mode == "scanner":
            if iv_pct is not None and not np.isnan(iv_pct):
                return round(float(iv_pct) - rv_20d, 2)
            try:
                vix = fetch_history("^VIX", period="5d")
                if not vix.empty:
                    return round(float(vix["Close"].iloc[-1]) - rv_20d, 2)
            except Exception:
                pass
            rv_5d = float(log_rets.tail(5).std() * np.sqrt(252) * 100)
            return round(rv_5d - rv_20d, 2)
        if iv_pct is None:
            iv_pct = OptionsDataEngine.get_atm_iv(ticker, df["Close"].iloc[-1])
        if pd.isna(iv_pct):
            return np.nan
        return round(float(iv_pct) - rv_20d, 2)

    @staticmethod
    def get_support_resistance(df: pd.DataFrame) -> tuple[float, float]:
        """50-bar rolling support (Low min) and resistance (High max)."""
        return float(df["Low"].rolling(50).min().iloc[-1]), float(df["High"].rolling(50).max().iloc[-1])

    @staticmethod
    def detect_reversal(df: pd.DataFrame) -> str:
        """Detect Golden/Death Cross or RSI 30/70 bounce."""
        try:
            if len(df) < 201:
                return "Insufficient Data"
            sma50 = df["Close"].rolling(50).mean()
            sma200 = df["Close"].rolling(200).mean()
            if sma50.iloc[-2] < sma200.iloc[-2] and sma50.iloc[-1] >= sma200.iloc[-1]:
                return "Golden Cross (Bull)"
            if sma50.iloc[-2] > sma200.iloc[-2] and sma50.iloc[-1] <= sma200.iloc[-1]:
                return "Death Cross (Bear)"
            delta = df["Close"].diff()
            up = delta.where(delta > 0, 0).rolling(14).mean()
            down = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = 100 - 100 / (1 + up / (down + 1e-9))
            if rsi.iloc[-2] < 30 and rsi.iloc[-1] >= 30:
                return "RSI Bull Bounce"
            if rsi.iloc[-2] > 70 and rsi.iloc[-1] <= 70:
                return "RSI Bear Rejection"
            return "No Active Reversal"
        except Exception:
            return "No Active Reversal"

    @staticmethod
    def calculate_sharpe(df: pd.DataFrame, rfr: float = 0.05) -> float:
        """Annualised Sharpe ratio (excess return / volatility)."""
        rets = df["Close"].pct_change().dropna()
        vol = rets.std() * np.sqrt(252)
        return round((rets.mean() * 252 - rfr) / vol, 2) if vol > 0 else 0.0

    @staticmethod
    def calculate_var(df: pd.DataFrame, confidence: float = 0.95) -> float:
        """Downside Value-at-Risk at given confidence level."""
        try:
            pct = float(np.percentile(df["Close"].pct_change().dropna(), (1 - confidence) * 100))
            return round(df["Close"].iloc[-1] * (1 + pct), 2)
        except Exception:
            return round(df["Close"].iloc[-1] * 0.95, 2)

    @staticmethod
    def calculate_upside_var(df: pd.DataFrame, confidence: float = 0.95) -> float:
        """Upside VaR (short-side risk) at given confidence level."""
        try:
            pct = float(np.percentile(df["Close"].pct_change().dropna(), confidence * 100))
            return round(df["Close"].iloc[-1] * (1 + pct), 2)
        except Exception:
            return round(df["Close"].iloc[-1] * 1.05, 2)

    @staticmethod
    def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes European call price (convenience wrapper)."""
        return QuantMath.black_scholes(S, K, T, r, sigma, "call")["price"]

    @staticmethod
    def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes European put price (convenience wrapper)."""
        return QuantMath.black_scholes(S, K, T, r, sigma, "put")["price"]

    @staticmethod
    def calculate_greeks(
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: str = "call",
    ) -> dict[str, float]:
        """Quick delta/gamma/vega extraction from BS."""
        bs = QuantMath.black_scholes(S, K, T, r, sigma, option_type)
        return {"delta": bs["delta"], "gamma": bs["gamma"], "vega": bs["vega"]}


# =============================================================================
# TRADE ARCHITECT
# =============================================================================

class TradeArchitect:
    """
    Options structure selector: directional spreads, iron condors,
    and hybrid stock+options plans based on Alpha Score and VRP regime.
    """

    @staticmethod
    def prob_itm(S: float, K: float, T: float, r: float, sigma: float,
                 option_type: str = "call") -> float:
        """Risk-neutral ITM probability N(d2)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return float(norm.cdf(d2) if option_type == "call" else norm.cdf(-d2))

    @staticmethod
    def generate_plan(
        ticker: str, price: float, score: int, vol: float,
        sup: float, res: float, half_kelly: float, r: float = 0.05,
    ) -> dict:
        """Generate directional spread plan from Alpha Score + vol regime."""
        bias = "LONG (Bullish)" if score >= 60 else "SHORT (Bearish)" if score <= 40 else "NEUTRAL"
        vol_regime = "HIGH" if vol > 35 else "LOW"
        sigma = max(0.01, vol / 100)
        T30 = 30 / 365
        res = price * 1.05 if res <= price else res
        sup = price * 0.95 if sup >= price else sup
        lw, uw = sup * 0.95, res * 1.05
        plan: dict = {}
        if "LONG" in bias:
            if vol_regime == "LOW":
                plan["name"] = "Long Call Vertical"
                plan["legs"] = f"+C({price:.0f}) / -C({res:.0f})"
                d = QuantLogic.bs_call(price, price, T30, r, sigma) - QuantLogic.bs_call(price, res, T30, r, sigma)
                plan["premium"] = f"Debit ${max(0.01, d):.2f}"
            else:
                plan["name"] = "Short Put Vertical"
                plan["legs"] = f"-P({sup:.0f}) / +P({lw:.0f})"
                c = QuantLogic.bs_put(price, sup, T30, r, sigma) - QuantLogic.bs_put(price, lw, T30, r, sigma)
                plan["premium"] = f"Credit ${max(0.01, c):.2f}"
        elif "SHORT" in bias:
            if vol_regime == "LOW":
                plan["name"] = "Long Put Vertical"
                plan["legs"] = f"+P({price:.0f}) / -P({sup:.0f})"
                d = QuantLogic.bs_put(price, price, T30, r, sigma) - QuantLogic.bs_put(price, sup, T30, r, sigma)
                plan["premium"] = f"Debit ${max(0.01, d):.2f}"
            else:
                plan["name"] = "Short Call Vertical"
                plan["legs"] = f"-C({res:.0f}) / +C({uw:.0f})"
                c = QuantLogic.bs_call(price, res, T30, r, sigma) - QuantLogic.bs_call(price, uw, T30, r, sigma)
                plan["premium"] = f"Credit ${max(0.01, c):.2f}"
        else:
            plan["name"] = "Iron Condor"
            plan["legs"] = f"+P({lw:.0f})/-P({sup:.0f}) | -C({res:.0f})/+C({uw:.0f})"
            c = (QuantLogic.bs_put(price, sup, T30, r, sigma) - QuantLogic.bs_put(price, lw, T30, r, sigma)
                 + QuantLogic.bs_call(price, res, T30, r, sigma) - QuantLogic.bs_call(price, uw, T30, r, sigma))
            plan["premium"] = f"Credit ${max(0.01, c):.2f}"
        plan["bias"] = bias
        plan["kelly_size"] = f"{int(max(5, min(50, half_kelly)))}%"
        plan["dte"] = "30 Days"
        plan["greeks"] = QuantLogic.calculate_greeks(price, price, T30, r, sigma)
        return plan

    @staticmethod
    def generate_hybrid_plan(
        price: float, score: int, vrp: float, sup: float, res: float,
    ) -> dict[str, str]:
        """HQTA hybrid stock+options directive based on score + VRP regime."""
        if score >= 60:
            if vrp > 0:
                return {"name": "Institutional Buy-Write", "action": f"Buy @ Market + Sell Call @ ${res:.2f}",
                        "logic": "Trend strong, options rich → harvest premium."}
            return {"name": "Bulletproof Bull", "action": f"Buy @ Market + Buy Put @ ${sup:.2f}",
                    "logic": "Trend strong, options cheap → ride up with insurance."}
        if score <= 40:
            if vrp > 0:
                return {"name": "Warren Buffett Entry", "action": f"Sell Cash-Secured Put @ ${sup:.2f}",
                        "logic": "Momentum weak, fear high → collect premium while waiting."}
            return {"name": "Smart-Money Short", "action": f"Buy Put Spread → ${sup:.2f}",
                    "logic": "Momentum broken, options cheap → profit from downside."}
        return {"name": "Floor-to-Ceiling Swing", "action": f"Buy @ ${sup:.2f} / Sell @ ${res:.2f}",
                "logic": "Range-bound → buy floor, sell ceiling."}


# =============================================================================
# REGIME ENGINE
# =============================================================================

class RegimeEngine:
    """Macro regime classifier via SPY trend + VIX term structure."""

    @staticmethod
    def detect_regime() -> str:
        """Classify market regime from SPY MA structure + VIX level + VIX9D term structure."""
        try:
            spy = fetch_history("SPY", period="1y")
            vix = fetch_history("^VIX", period="6mo")
            vix9d = fetch_history("^VIX9D", period="5d")
            price = spy["Close"].iloc[-1]
            ma50 = spy["Close"].rolling(50).mean().iloc[-1]
            ma200 = spy["Close"].rolling(200).mean().iloc[-1]
            vix_lvl = vix["Close"].iloc[-1]
            v9d = vix9d["Close"].iloc[-1] if not vix9d.empty else vix_lvl
            if v9d > vix_lvl + 2:
                return "Vol-Squeeze → Explosive Move Imminent"
            if price > ma50 > ma200 and vix_lvl < 20:
                return "Risk-On"
            if price < ma200 and vix_lvl > 25:
                return "Risk-Off"
            return "Neutral"
        except Exception:
            return "Neutral"


# =============================================================================
# DYNAMIC UNIVERSE + SECTOR STRENGTH
# =============================================================================

class DynamicUniverseEngine:
    """Curated high-liquidity APEX-100 ticker universe."""

    @staticmethod
    def get_apex_100() -> list[str]:
        """Return the APEX-100 universe of highest-liquidity US equities + crypto."""
        return [
            "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO",
            "LLY","JPM","V","MA","UNH","XOM","JNJ","HD","PG","COST",
            "MRK","ABBV","CRM","AMD","BAC","NFLX","KO","PEP","TMO",
            "WMT","DIS","ADBE","CSCO","INTC","QCOM","TXN","IBM","NOW",
            "UBER","PLTR","WFC","GS","MS","AXP","C","BLK","SPGI",
            "HOOD","SOFI","COIN","PYPL","SHOP","AFRM","UPST","CVX","COP",
            "SLB","EOG","MPC","GE","CAT","BA","RTX","LMT","HON","UNP",
            "DE","VLO","OXY","HAL","GD","NOC","WM","MCD","NKE","SBUX",
            "LOW","ROKU","DKNG","SPOT","SNOW","CRWD","PANW","MSTR",
            "MARA","RIOT","CVNA","SMCI","ARM","TSM","ASML","BTC-USD",
            "ETH-USD","IBIT","MU","AMAT","LRCX","KLAC","SYM","DELL","MNDY",
        ]


class SectorStrengthEngine:
    """Sector rotation detector using ETF 1-month returns."""

    SECTOR_ETFS: dict[str, str] = {
        "🔥 Mag 7 + BTC": "MAGS", "💻 Technology": "XLK", "🏦 Financials": "XLF",
        "🏥 Healthcare": "XLV", "🛒 Discretionary": "XLY", "📡 Comms": "XLC",
        "🏭 Industrials": "XLI", "🧼 Staples": "XLP", "🛢️ Energy": "XLE",
        "🔌 Utilities": "XLU", "🏠 Real Estate": "XLRE", "🪙 Digital Assets": "WGMI",
    }

    @staticmethod
    def get_strongest_sector() -> tuple[str, float]:
        """Return (sector_name, 1mo_return) for the highest-momentum sector."""
        returns: dict[str, float] = {}
        for sector, etf in SectorStrengthEngine.SECTOR_ETFS.items():
            try:
                df = fetch_history(etf, "1mo")
                if len(df) > 15:
                    returns[sector] = float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1)
            except Exception:
                pass
        if not returns:
            return "💻 Technology", 0.0
        best = max(returns, key=returns.get)
        return best, returns[best]


# =============================================================================
# PORTFOLIO ENGINE
# =============================================================================

class PortfolioEngine:
    """
    Portfolio construction: cross-sectional momentum ranking, vol-targeting,
    mean-variance optimisation, and Kelly Criterion sizing.
    """

    @staticmethod
    def cross_sectional_momentum(price_dict: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Rank tickers by 20-day momentum (percentile rank 0–1)."""
        try:
            return pd.Series({t: df["Close"].pct_change(20).iloc[-1]
                              for t, df in price_dict.items()}).rank(pct=True).to_dict()
        except Exception:
            return {t: 0.5 for t in price_dict}

    @staticmethod
    def volatility_targeting(df: pd.DataFrame, target: float = 0.20) -> float:
        """Position scalar to achieve target annualised vol. Clamped [0, 2]."""
        try:
            return min(2.0, max(0.0, target / (df["Close"].pct_change().std() * np.sqrt(252) + 1e-9)))
        except Exception:
            return 1.0

    @staticmethod
    def mean_variance_weight(price_dict: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Minimum-variance portfolio weights via pseudoinverse of covariance."""
        try:
            rets = pd.DataFrame({t: df["Close"].pct_change() for t, df in price_dict.items()}).dropna()
            inv = np.linalg.pinv(rets.cov().values)
            ones = np.ones(len(inv))
            return dict(zip(rets.columns, inv @ ones / (ones.T @ inv @ ones)))
        except Exception:
            return {t: 1 / len(price_dict) for t in price_dict}

    @staticmethod
    def kelly_weight(df: pd.DataFrame) -> float:
        """Kelly fraction from empirical mean/variance of returns. Clamped [0, 2]."""
        try:
            r = df["Close"].pct_change().dropna()
            return max(0.0, min((r.mean() * 252) / (r.var() * 252 + 1e-9), 2.0))
        except Exception:
            return 0.0


# =============================================================================
# EXPECTED MOVE ENGINE — correct log-normal bounds
# =============================================================================

class OptionsExpectedMove:
    """
    Expected move calculation from implied volatility.

    Uses correct log-normal bounds S×exp(±σ√T) instead of the symmetric
    approximation S ± Sσ√T. The log-normal distribution is right-skewed,
    so upper distance > lower distance from spot.
    """

    @staticmethod
    def from_iv(S: float, T: float, sigma: float) -> dict[str, float]:
        """
        Compute 1σ and 2σ expected move bounds from IV.

        Returns both the simple (market convention) EM and the correct
        asymmetric log-normal bounds.

        Args:
            S:     Spot price.
            T:     Time in years.
            sigma: Implied volatility (decimal).

        Returns:
            dict with: em_1sigma (simple), em_2sigma, upper_1s, lower_1s,
            upper_2s, lower_2s (all log-normal asymmetric).
        """
        sqrtT = np.sqrt(T)
        em_simple = S * sigma * sqrtT

        # Correct log-normal bounds (asymmetric)
        upper_1s = S * np.exp(+sigma * sqrtT)
        lower_1s = S * np.exp(-sigma * sqrtT)
        upper_2s = S * np.exp(+2 * sigma * sqrtT)
        lower_2s = S * np.exp(-2 * sigma * sqrtT)

        return {
            "em_1sigma": round(float(em_simple), 2),
            "em_2sigma": round(float(2 * em_simple), 2),
            "upper_1s": round(float(upper_1s), 2),
            "lower_1s": round(float(lower_1s), 2),
            "upper_2s": round(float(upper_2s), 2),
            "lower_2s": round(float(lower_2s), 2),
        }


# =============================================================================
# SPARKLINE HELPERS
# =============================================================================

def get_sparkline(series) -> str:
    """Convert numeric series to unicode bar sparkline."""
    bars = [' ', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    arr = np.asarray(series, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return bars[0] * len(arr)
    return "".join(bars[int(np.clip((v - mn) / (mx - mn) * 7, 0, 7))] for v in arr)


def get_candle_sparkline(df_sub: pd.DataFrame) -> str:
    """Convert OHLC DataFrame to colored HTML sparkline (green up, red down)."""
    if len(df_sub) < 2:
        return "▁" * 20
    bars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    mn = df_sub["Low"].min()
    mx = df_sub["High"].max()
    spark: list[str] = []
    for _, row in df_sub.iterrows():
        idx = 0 if mx == mn else int(np.clip((row["Close"] - mn) / (mx - mn) * 7, 0, 7))
        color = "#4ADE80" if row["Close"] >= row["Open"] else "#F87171"
        spark.append(f"<span style='color:{color};'>{bars[idx]}</span>")
    return "".join(spark)


# =============================================================================
# MARKET SCANNER
# =============================================================================

class MarketScanner:
    """
    Parallel multi-sector scanner with GARCH + WFO + vol-targeting + Kelly.

    Scans up to 200 tickers across 12 sectors. Ranks by Kelly × Alpha Score.
    Returns top 15 setups with Ultimate Signal classification.
    """

    SECTOR_UNIVERSE: dict[str, list[str]] = {
        "🔥 Mag 7 + BTC": ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","BTC-USD"],
        "💻 Technology": ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","AMD","ADBE","CSCO","INTC","QCOM","TXN","IBM","NOW","INTU","AMAT","MU","PANW","SNOW","CRWD"],
        "🏦 Financials": ["JPM","V","MA","BAC","WFC","GS","MS","AXP","C","BLK","SPGI","CME","SCHW","CB","MMC","PGR","USB","PNC","TFC","COF"],
        "🏥 Healthcare": ["LLY","UNH","JNJ","ABBV","MRK","PFE","AMGN","ISRG","SYK","MDT","VRTX","REGN","GILD","BSX","CVS","CI","ZTS","BDX","HUM","BIIB"],
        "🛒 Discretionary": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","CMG","MAR","HLT","ORLY","ABNB","GM","F","FVRR","CHWY","ETSY","EBAY"],
        "📡 Comms": ["GOOGL","META","NFLX","DIS","VZ","T","CMCSA","TMUS","WBD","CHTR","EA","TTWO","LYV","MTCH","FOXA","SIRI","ROKU","SNAP","PINS","ZG"],
        "🏭 Industrials": ["GE","CAT","UBER","BA","RTX","LMT","HON","UNP","DE","UPS","MMM","LUV","FDX","ETN","EMR","NOC","GD","CSX","NSC","PCAR"],
        "🧼 Staples": ["WMT","PG","COST","KO","PEP","PM","TGT","MO","DG","EL","CL","KMB","MDLZ","SYY","HSY","KR","GIS","CHD","K","CPB"],
        "🛢️ Energy": ["XOM","CVX","COP","SLB","OXY","EOG","MPC","VLO","HAL","PSX","WMB","KMI","HES","BKR","DVN","FANG","TRGP","OKE","CTRA","MRO"],
        "🔌 Utilities": ["NEE","CEG","SO","DUK","SRE","AEP","D","PCG","EXC","PEG","XEL","ED","WEC","AWK","ES","ETR","FE","CMS","LNT","NI"],
        "🏠 Real Estate": ["AMT","PLD","CCI","EQIX","O","PSA","SPG","WELL","DLR","VICI","AVB","EXR","MAA","ESS","INVH","UDR","CPT","HST"],
        "🪙 Digital Assets": ["BTC-USD","ETH-USD","COIN","MSTR","MARA","RIOT","CLSK","HUT","IBIT","FBTC","CORZ","CIFR"],
    }

    @staticmethod
    def _process_ticker(
        t: str, regime: str, cs_rank: dict, mvo_weights: dict,
    ) -> Optional[dict]:
        """Process a single ticker through the full analytics pipeline."""
        try:
            df = fetch_history(t, "2y")
            if len(df) < 100 or df["Volume"].tail(30).mean() < 1_000_000:
                return None
            price = float(df["Close"].iloc[-1])
            score = AlphaEngine.calculate_score(df)
            vol = QuantLogic.calculate_vol(df)
            vrp = QuantLogic.calculate_vrp_edge(t, df, mode="scanner")
            sup, res = QuantLogic.get_support_resistance(df)
            wr, sr, outperf, mdd, kelly, sortino, calmar, _ = BacktestEngine.run_wfo_backtest(df)
            plan = TradeArchitect.generate_plan(t, price, score, vol, sup, res, kelly)
            hybrid = TradeArchitect.generate_hybrid_plan(price, score, vrp, sup, res)
            cs_r = cs_rank.get(t, 0.5)
            vol_tgt = PortfolioEngine.volatility_targeting(df)
            mvo = mvo_weights.get(t, 0)
            kelly_p = PortfolioEngine.kelly_weight(df)
            ult = score * 0.35 + cs_r * 100 * 0.20 + kelly_p * 50 * 0.15 + vol_tgt * 50 * 0.15 + mvo * 50 * 0.15
            
            # Restored ULTRA LONG / SHORT Logic
            mvo = mvo_weights.get(t, 0)
            kelly_p = PortfolioEngine.kelly_weight(df)
            ult = score * 0.35 + cs_r * 100 * 0.20 + kelly_p * 50 * 0.15 + vol_tgt * 50 * 0.15 + mvo * 50 * 0.15
            
            # 1. Removed Emojis to fix UI wrapping (and restored strict 80/20 thresholds)
            if regime != "Risk-Off" and score >= 80 and vrp < -5.0:
                signal = "ULTRA LONG"
            elif score <= 20 and vrp > 5.0:
                signal = "ULTRA SHORT"
            elif regime != "Risk-Off" and ult > 65 and vrp < 0 and wr > 10 and sr > 50:
                signal = "ULTIMATE LONG"
            elif ult < 35 and vrp > 0 and wr > 10 and sr > 50:
                signal = "ULTIMATE SHORT"
            else:
                signal = "Standard"

            # 2. Calculate 30-DTE Expected Move for the Scanner Table
            T30 = 30 / 365
            sigma = max(0.01, vol / 100.0)
            em = OptionsExpectedMove.from_iv(price, T30, sigma)
            exp_move_str = f"±${em['em_1sigma']:.2f}"

            return {
                "Ticker": t, "Price": round(price, 2), "Ultimate Signal": signal,
                "Alpha Score": score, "Trend": plan["bias"], "VRP Edge": f"{vrp:+.1f}%",
                "Vol": f"{vol:.1f}%", "Exp. Move": exp_move_str, 
                "Vol-Target %": f"{round(vol_tgt * 100, 1)}%",
                "Support": round(sup, 2), "Resistance": round(res, 2),
                "HQTA Apex Action": hybrid["action"], "Strategy": plan["name"],
                "Kelly": kelly,
            }
        except Exception:
            return None

    @staticmethod
    def run_scan(tickers: list[str]) -> pd.DataFrame:
        """
        Run parallel sector scan. Returns top 15 setups ranked by Kelly × Alpha.

        Uses ThreadPoolExecutor with max 8 workers for concurrent API calls.
        """
        regime = RegimeEngine.detect_regime()
        price_dict: dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                df = fetch_history(t, "2y")
                if len(df) > 100 and df["Volume"].tail(30).mean() > 1_000_000:
                    price_dict[t] = df
            except Exception:
                pass
        if not price_dict:
            return pd.DataFrame()
        cs_rank = PortfolioEngine.cross_sectional_momentum(price_dict)
        mvo_w = PortfolioEngine.mean_variance_weight(price_dict)
        results: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(price_dict))) as ex:
            futs = {ex.submit(MarketScanner._process_ticker, t, regime, cs_rank, mvo_w): t for t in price_dict}
            for f in concurrent.futures.as_completed(futs):
                r = f.result()
                if r:
                    results.append(r)
        if not results:
            return pd.DataFrame()
        df_r = pd.DataFrame(results).sort_values(by=["Kelly", "Alpha Score"], ascending=[False, False]).head(15)
        df_r["Kelly"] = df_r["Kelly"].apply(lambda x: f"{x:.1f}%")
        return df_r


# Module load confirmation via logger
logger.info(
    "VRP Quant Engine v30.5.1 loaded — 5-Source Waterfall + "
    "200-step CRR (Euro+American) + 100k MC + Jump-Diffusion + WFO"
)

