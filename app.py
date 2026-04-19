# =============================================================================
# VRP QUANT LIVE TERMINAL — v30.5.1 (COMPLETE PRODUCTION)
# 5-Source Waterfall + 200-step CRR (Euro+American) + 10k-100k MC + bcrypt
# =============================================================================
#
# CHANGES FROM v30.5.0-A+:
#   [FIX-01]  Market Scanner radio label gets proper emoji (📡)
#   [FIX-02]  Duplicate sidebar separator removed for ANALYST tier
#   [FIX-03]  Raw Supabase errors masked from login page; logged server-side
#   [FEAT-07] Deep Dive shows European vs American CRR + early-ex premium
#   [VERSION] Sidebar heading + login title bumped to v30.5.1
# =============================================================================

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timezone
from supabase import create_client, Client

import bcrypt

from vrp_quant_engine_v30_5_1 import (
    QuantMath,
    PolygonClient,
    AlphaVantageClient,
    GoogleFinanceClient,
    GrokFinanceClient,
    OptionsDataEngine,
    AlphaEngine,
    BacktestEngine,
    QuantLogic,
    TradeArchitect,
    RegimeEngine,
    DynamicUniverseEngine,
    SectorStrengthEngine,
    PortfolioEngine,
    OptionsExpectedMove,
    MarketScanner,
    get_candle_sparkline,
    fetch_history,
    fetch_live_price,
    fetch_risk_free_rate,
    data_layer_status,
    ARCH_AVAILABLE,
)

logger = logging.getLogger("VRP_Terminal")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="VRP Quant | Institutional Terminal",
    layout="wide", page_icon="🏦", initial_sidebar_state="expanded",
)


# =============================================================================
# BCRYPT AUTH
# =============================================================================

def hash_password(plaintext: str, rounds: int = 12) -> str:
    """Hash plaintext with bcrypt (work factor 12, ~250ms/hash)."""
    return bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt(rounds=rounds)).decode("utf-8")

def verify_password(plaintext: str, stored_hash: str) -> bool:
    """Constant-time bcrypt verification. Returns False on malformed hash."""
    try:
        return bcrypt.checkpw(plaintext.encode("utf-8"), stored_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False

def _is_bcrypt_hash(value: str) -> bool:
    """Check if string is a bcrypt hash ($2b$/$2a$/$2y$ prefix)."""
    return isinstance(value, str) and value.startswith(("$2b$", "$2a$", "$2y$"))

@st.cache_resource
def _init_supabase() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def verify_login(email: str, password: str) -> bool:
    """Authenticate via Supabase with bcrypt. Auto-upgrades plaintext passwords."""
    try:
        sb = _init_supabase()
        resp = sb.table("subscribers").select("*").eq("email", email.lower()).execute()
        if not resp.data:
            return False
        user = resp.data[0]
        stored_pw = user.get("app_password", "")
        if user.get("status") != "ACTIVE":
            return False

        if _is_bcrypt_hash(stored_pw):
            if not verify_password(password, stored_pw):
                return False
        else:
            if stored_pw != password:
                return False
            try:
                sb.table("subscribers").update(
                    {"app_password": hash_password(password)}
                ).eq("email", email.lower()).execute()
            except Exception:
                pass

        st.session_state["authenticated"] = True
        st.session_state["user_tier"] = user["tier"]
        st.session_state.tier = user["tier"]
        st.session_state.paypal_active = True
        st.session_state.user_email = email.lower()
        return True
    except Exception as e:
        # [FIX-03] Mask raw error from user; log full traceback server-side
        logger.error(
            f"Supabase auth error for {email.lower() if email else '<empty>'}: {e}",
            exc_info=True,
        )
        st.error("Authentication service temporarily unavailable. Please try again.")
    return False

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

PAYPAL_ANALYST = st.secrets.get("PAYPAL_ANALYST_LINK",
    "https://www.paypal.com/webapps/billing/plans/subscribe?plan_id=P-0CB63794C10515154NGMNDNA")
PAYPAL_GOD = st.secrets.get("PAYPAL_GOD_MODE_LINK",
    "https://www.paypal.com/webapps/billing/plans/subscribe?plan_id=P-723423746M676015CNGMNFGI")

if not st.session_state["authenticated"]:
    st.title("🔒 VRP Quant Terminal v30.5.1")
    col_form, _ = st.columns([1, 2])
    with col_form:
        email_in = st.text_input("Email")
        pass_in = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if verify_login(email_in, pass_in):
                st.success("Access Granted.")
                st.rerun()
            else:
                st.error("Invalid email, password, or inactive subscription.")
    st.markdown("---")
    st.markdown("### Founding Member Cohort")
    b1, b2 = st.columns(2)
    with b1:
        st.info("**ANALYST TIER**\n* Retail: ~~$299/mo~~\n* Founding: **$149/mo**")
        st.link_button("Subscribe via PayPal", PAYPAL_ANALYST, use_container_width=True)
    with b2:
        st.success("**GOD MODE TIER**\n* Retail: ~~$999/mo~~\n* Founding: **$499/mo**")
        st.link_button("Subscribe via PayPal", PAYPAL_GOD, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-size: 0.85em; color: #888;'>
    <b>SEC & RISK DISCLOSURE:</b> VRP Quant is a quantitative research tool, not a registered investment advisor, broker-dealer, or financial planner. All outputs generated by this terminal, including Monte Carlo simulations and HQTA Directives, are strictly for <b>informational and educational purposes only</b>. Options trading carries a substantial risk of total capital loss and is not suitable for all investors. Past performance is not indicative of future results. Always verify data independently and consult a licensed professional before deploying capital.
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# =============================================================================
# INSTITUTIONAL CSS
# =============================================================================
st.markdown("""
<style>
    .stApp { background-color: #0B0F19; color: #F8FAFC; }
    [data-testid="stSidebar"] { background-color: #0F172A; border-right: 1px solid #1E293B; }
    div[data-testid="metric-container"] {
        background-color: #1E293B; border: 1px solid #334155;
        padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label { color: #94A3B8 !important; font-weight: 600 !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #38BDF8 !important; font-size: 1.8rem !important; font-weight: 700 !important;
    }
    .apex-box { background-color: #082F49; border-left: 5px solid #38BDF8; border-radius: 5px;
        padding: 20px; margin: 15px 0 25px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
    .apex-title { color: #BAE6FD; font-size: 1.4em; font-weight: 800; margin-bottom: 10px; }
    .apex-action { color: #38BDF8; font-size: 1.2em; font-weight: 700; margin-bottom: 10px; }
    .apex-logic { color: #94A3B8; font-size: 1em; font-style: italic; }
    h1, h2, h3 { color: #F1F5F9 !important; font-weight: 700 !important; }
    .badge-long { background:#022C22; color:#10B981; padding:4px 8px; border-radius:4px; font-weight:900; border:1px solid #047857; font-size:11px; }
    .badge-short { background:#450A0A; color:#EF4444; padding:4px 8px; border-radius:4px; font-weight:900; border:1px solid #B91C1C; font-size:11px; }
    .badge-std { color:#475569; font-style:italic; font-size:11px; }
    .apex-cell { background:#082F49 !important; border-left:4px solid #0EA5E9 !important; color:#38BDF8 !important; font-weight:700; }
    .ticker-cell { font-weight:900; color:#FFFFFF; font-size:14px; }
    .val-pos { color:#10B981; font-weight:600; }
    .val-neg { color:#EF4444; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HTML TABLE RENDERER
# =============================================================================

def render_scanner_table(df: pd.DataFrame) -> str:
    """Render scanner results as styled HTML table with signal badges."""
    html = """<div style="overflow-x:auto;background:#0B0F19;margin-top:10px;border:1px solid #1E293B;border-radius:6px;">
    <table style="width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;font-size:13px;text-align:left;">
    <thead><tr>"""
    for col in df.columns:
        html += f"<th style='background:#0F172A;color:#94A3B8;padding:12px 15px;border-bottom:2px solid #334155;font-weight:700;font-size:11px;text-transform:uppercase;'>{col}</th>"
    html += "</tr></thead><tbody>"
    for _, row in df.iterrows():
        html += "<tr style='border-bottom:1px solid #1E293B;'>"
        for col in df.columns:
            val = str(row[col])
            if col == "Ticker":
                html += f"<td class='ticker-cell' style='padding:12px 15px;'>{val}</td>"
            elif col == "Ultimate Signal":
                badge = "badge-long" if "LONG" in val else "badge-short" if "SHORT" in val else "badge-std"
                html += f"<td style='padding:12px 15px;'><span class='{badge}'>{val}</span></td>"
            elif col == "HQTA Apex Action":
                html += f"<td class='apex-cell' style='padding:12px 15px;'>{val}</td>"
            else:
                cls = "val-pos" if "+" in val else "val-neg" if "-" in val and col in ("VRP Edge","Trend") else ""
                html += f"<td class='{cls}' style='padding:12px 15px;color:#E2E8F0;'>{val}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    return html


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

_DK = dict(template="plotly_dark", paper_bgcolor="#0B0F19", plot_bgcolor="#0F172A",
           font=dict(family="Inter", color="#F8FAFC"))

def chart_price_cone(ticker: str, df: pd.DataFrame, S: float, sigma: float) -> None:
    """Candlestick chart with ±1σ/2σ expected move lines from IV."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#4ADE80", decreasing_line_color="#F87171"))
    T = 30 / 365
    em = OptionsExpectedMove.from_iv(S, T, sigma)
    fig.add_hline(y=em["upper_1s"], line_dash="dot", line_color="#4ADE80", annotation_text=f"+1σ ${em['upper_1s']}")
    fig.add_hline(y=em["lower_1s"], line_dash="dot", line_color="#F87171", annotation_text=f"-1σ ${em['lower_1s']}")
    fig.add_hline(y=em["upper_2s"], line_dash="dash", line_color="#60A5FA", annotation_text="+2σ")
    fig.add_hline(y=em["lower_2s"], line_dash="dash", line_color="#60A5FA", annotation_text="-2σ")
    fig.update_layout(**_DK, height=550, xaxis_rangeslider_visible=False,
                      title=f"{ticker} — Price + 30-DTE Expected Move Cone")
    st.plotly_chart(fig, use_container_width=True)

def chart_greeks_bar(bs: dict) -> None:
    """Greeks bar chart with scaled absolute values for visual comparison."""
    g = {"Delta": abs(bs["delta"]), "Gamma": bs["gamma"] * 100,
         "Theta": abs(bs["theta"]) * 10, "Vega": bs["vega"], "Rho": abs(bs["rho"])}
    fig = go.Figure(go.Bar(x=list(g.keys()), y=list(g.values()), marker_color="#4ADE80"))
    fig.update_layout(**_DK, height=350, title="Greeks (Scaled Absolute Impact)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("ℹ️ Gamma scaled ×100, Theta ×10 for visual comparability. Raw values shown in metrics above.")

def chart_mc_fan(mc: dict, S: float, K: float) -> None:
    """MC path fan (100 sample paths) + terminal distribution histogram."""
    paths = mc["paths"]
    mean_path = mc["mean_path"]
    lower = mc["conf_lower"]
    upper = mc["conf_upper"]
    fig = go.Figure()
    rng = np.random.default_rng(42)
    idx = rng.integers(0, paths.shape[0], 100)
    for i in idx:
        fig.add_trace(go.Scatter(x=list(range(paths.shape[1])), y=paths[i],
            mode="lines", line=dict(color="#64748B", width=0.8), opacity=0.12, showlegend=False))
    fig.add_trace(go.Scatter(y=mean_path, mode="lines", line=dict(color="#4ADE80", width=3), name="Mean"))
    fig.add_trace(go.Scatter(y=lower, mode="lines", line=dict(color="#F87171", dash="dot"), name="5th %ile"))
    fig.add_trace(go.Scatter(y=upper, mode="lines", line=dict(color="#F87171", dash="dot"),
                             name="95th %ile", fill="tonexty", fillcolor="rgba(248,113,113,0.06)"))
    fig.update_layout(**_DK, height=520, title=f"Monte Carlo — {mc['n_paths']:,} Paths")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure(go.Histogram(x=mc["S_T"], nbinsx=100, marker_color="#A78BFA"))
    fig2.add_vline(x=K, line_dash="dash", line_color="#F87171", annotation_text=f"Strike ${K:.2f}")
    fig2.add_vline(x=S, line_dash="dot", line_color="#4ADE80", annotation_text=f"Spot ${S:.2f}")
    fig2.update_layout(**_DK, height=380, title="Terminal Price Distribution at Expiry")
    st.plotly_chart(fig2, use_container_width=True)

def chart_kalman_garch(bt_df: pd.DataFrame) -> None:
    """Candlestick with Kalman centerline and GARCH ±2σ bands overlay."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=bt_df.index, open=bt_df["Open"], high=bt_df["High"],
        low=bt_df["Low"], close=bt_df["Close"], name="Price",
        increasing_line_color="#38BDF8", decreasing_line_color="#334155"))
    if "Kalman_Price" in bt_df.columns:
        fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["Kalman_Price"], mode="lines",
                                  name="Kalman", line=dict(color="#F8FAFC", width=1.5)))
    if "Upper_Band" in bt_df.columns:
        fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["Upper_Band"], mode="lines",
                                  name="Upper Band", line=dict(color="#94A3B8", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["Lower_Band"], mode="lines",
                                  name="Lower Band", line=dict(color="#94A3B8", width=1, dash="dot"),
                                  fill="tonexty", fillcolor="rgba(148,163,184,0.08)"))
    fig.update_layout(**_DK, height=500, xaxis_rangeslider_visible=False,
                      title="Kalman Centerline & GARCH Bands")
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SIDEBAR
# =============================================================================

tier = st.session_state.get("tier", "ANALYST")
with st.sidebar:
    st.markdown("# 🏦 VRP Quant v30.5.1")
    if tier == "GOD_MODE":
        st.success("🔓 GOD MODE ACTIVE")
    else:
        st.warning("🔒 ANALYST TIER")

    # [FIX-01] proper emoji on radio label   [FIX-02] single separator only
    st.markdown("---")
    mode = st.radio("Module", ["📡 Market Scanner", "🔬 Deep Dive Analysis"])
    st.markdown("---")
    if st.button("🔄 Clear Cache"):
        from vrp_quant_engine_v30_5_1 import LOCAL_CACHE
        LOCAL_CACHE.clear()
        st.cache_data.clear()
        st.success("Cache cleared.")
    if st.button("Log Out"):
        st.session_state["authenticated"] = False
        st.rerun()


# =============================================================================
# CACHED SCANNER (module-level to avoid redefinition on each button press)
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def _cached_scan(tickers_tuple: tuple) -> pd.DataFrame:
    """Cached scanner — 15-min TTL prevents redundant re-scans."""
    return MarketScanner.run_scan(list(tickers_tuple))


# =============================================================================
# MARKET SCANNER
# =============================================================================

# [FIX-01] match the new emoji-prefixed label
if mode == "📡 Market Scanner":
    st.title("📡 Institutional Market Scanner")
    regime = RegimeEngine.detect_regime()
    st.markdown(f"### 🌍 Market Regime: **{regime}**")
    st.markdown("---")

    if tier != "GOD_MODE":
        st.error("🔒 Market Scanner requires GOD MODE tier.")
        st.stop()

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        options = (
            ["🤖 Auto-Detect Strongest", "🌌 APEX-100"]
            + list(MarketScanner.SECTOR_UNIVERSE.keys())
            + ["✨ Custom Watchlist"]
        )
        sector = st.selectbox("Sector:", options)

    selected = []
    if sector == "🤖 Auto-Detect Strongest":
        with st.spinner("Detecting strongest sector..."):
            best, roc = SectorStrengthEngine.get_strongest_sector()
            selected = MarketScanner.SECTOR_UNIVERSE.get(best, [])
            st.info(f"🔄 Rotating into **{best}** ({roc*100:+.2f}%)")
    elif sector == "✨ Custom Watchlist":
        with c2:
            custom = st.text_area("Tickers (comma-sep):", "PLTR, SOFI, NVDA")
            if custom:
                selected = [t.strip().upper() for t in custom.split(",") if t.strip()]
    elif sector == "🌌 APEX-100":
        selected = DynamicUniverseEngine.get_apex_100()[:60]
    else:
        selected = MarketScanner.SECTOR_UNIVERSE.get(sector, [])

    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        strict = st.checkbox("⚡ High-conviction only", value=False)

    if st.button("▶ Run Scan", use_container_width=True) and selected:
        n = len(selected)
        with st.spinner(f"Parallel scan — {n} assets × GARCH + WFO + vol-targeting..."):
            try:
                df_scan = _cached_scan(tuple(selected))
                if df_scan.empty:
                    st.warning("No tickers passed volume/volatility gates.")
                else:
                    if strict:
                        df_scan = df_scan[df_scan["Ultimate Signal"] != "Standard"]
                        if df_scan.empty:
                            st.warning("No high-conviction setups. Cash is a position.")
                            st.stop()
                    st.success(f"✅ {len(df_scan)} setups from {n} candidates")
                    st.markdown(render_scanner_table(df_scan), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Scan error: {e}")


# =============================================================================
# DEEP DIVE
# =============================================================================

elif mode == "🔬 Deep Dive Analysis":
    st.title("🔬 Deep Dive & Trade Architect")
    ticker = st.text_input("Asset Ticker", "TSLA").upper().strip()

    if st.button("▶ Run Deep Dive", use_container_width=True):
        with st.spinner("Running GARCH + 200-step CRR + 50k MC + Jump-Diffusion..."):
            try:
                df = fetch_history(ticker, "2y")
                if df.empty:
                    st.error(f"No data for {ticker}.")
                    st.stop()

                # Live price via 5-source waterfall
                price_data = fetch_live_price(ticker)
                curr = price_data.get("price") or float(df["Close"].iloc[-1])
                price_source = price_data.get("source", "unknown")

                rfr, rfr_src = fetch_risk_free_rate()
                iv_pct = OptionsDataEngine.get_atm_iv(ticker, curr)
                sigma = float(iv_pct) / 100 if iv_pct and not np.isnan(iv_pct) else float(df["Close"].pct_change().std() * np.sqrt(252))
                T30 = 30 / 365

                # Data provenance badge
                st.markdown(
                    f"<div style='background:#0f1a2e;border-left:4px solid #475569;border-radius:4px;"
                    f"padding:10px 14px;margin-bottom:12px;font-family:monospace;font-size:11px;color:#94A3B8;'>"
                    f"<b>${curr:.2f}</b> via <b>{price_source}</b> | "
                    f"IV: {f'{iv_pct:.1f}%' if iv_pct and not np.isnan(iv_pct) else 'N/A'} | "
                    f"r: {rfr*100:.3f}% [{rfr_src}] | Bars: {len(df)}"
                    f"</div>", unsafe_allow_html=True,
                )

                score = AlphaEngine.calculate_score(df)
                vol = QuantLogic.calculate_vol(df)
                vrp = QuantLogic.calculate_vrp_edge(ticker, df, mode="deep_dive")
                reversal = QuantLogic.detect_reversal(df)
                sharpe = QuantLogic.calculate_sharpe(df, rfr)
                sup, res = QuantLogic.get_support_resistance(df)
                wr, sr, outperf, mdd, hk, sortino, calmar, bt_df = BacktestEngine.run_wfo_backtest(df)
                plan = TradeArchitect.generate_plan(ticker, curr, score, vol, sup, res, hk)
                hybrid = TradeArchitect.generate_hybrid_plan(curr, score, vrp if not np.isnan(vrp) else 0.0, sup, res)

                # Apex Box
                st.markdown(f"""<div class="apex-box">
                    <div class="apex-title">🏆 {hybrid['name']}</div>
                    <div class="apex-action">TRADE ARCHITECTURE: {hybrid['action']}</div>
                    <div class="apex-logic">{hybrid['logic']}</div>
                </div>""", unsafe_allow_html=True)

                # Market Variables
                st.markdown("### 📊 Market Variables")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Price", f"${curr:.2f}")
                m2.metric("Alpha Score", f"{score}/100")
                m3.metric("Trend", plan["bias"])
                m4.metric("Ann. Vol", f"{vol:.1f}%")
                m5.metric("Reversal", reversal)

                m6, m7, m8, m9, m10 = st.columns(5)
                m6.metric("VRP Edge", f"{vrp:+.2f}%" if not np.isnan(vrp) else "N/A")
                m7.metric("Sharpe", f"{sharpe:.2f}")
                m8.metric("Support", f"${sup:.2f}")
                m9.metric("Resistance", f"${res:.2f}")
                m10.metric("95% VaR", f"${QuantLogic.calculate_var(df):.2f}")

                # Full Greeks
                st.markdown("### 📐 Black-Scholes Full Greeks")
                bs = QuantMath.black_scholes(curr, sup, T30, rfr, sigma, "put")
                g1, g2, g3, g4, g5, g6 = st.columns(6)
                g1.metric("BS Price", f"${bs['price']:.4f}")
                g2.metric("Delta (Δ)", f"{bs['delta']:.4f}")
                g3.metric("Gamma (Γ)", f"{bs['gamma']:.5f}")
                g4.metric("Theta (θ)", f"${bs['theta']:.4f}/day")
                g5.metric("Vega (ν)", f"${bs['vega']:.4f}/1%")
                g6.metric("Rho (ρ)", f"${bs['rho']:.4f}/1%")

                # [FEAT-07] CRR Binomial — European vs American exercise comparison
                st.markdown("### 🌳 Pricing Convergence & Early-Exercise Premium")
                binom_euro = QuantMath.crr_binomial_tree(
                    curr, sup, T30, rfr, sigma,
                    n_steps=200, option_type="put", exercise="european",
                )
                binom_amer = QuantMath.crr_binomial_tree(
                    curr, sup, T30, rfr, sigma,
                    n_steps=200, option_type="put", exercise="american",
                )
                euro_err = abs(binom_euro["price"] - bs["price"]) / max(bs["price"], 0.001) * 100
                st.caption(
                    f"CRR European: **${binom_euro['price']:.4f}**  |  "
                    f"CRR American: **${binom_amer['price']:.4f}**  |  "
                    f"Early-ex premium: **${binom_amer['early_exercise_premium']:.4f}**  |  "
                    f"BS: ${bs['price']:.4f}  |  Euro vs BS error: {euro_err:.3f}%"
                )

                # WFO Backtest
                st.markdown("### ⚙️ Walk-Forward Backtest (OOS)")
                b1, b2, b3, b4, b5, b6, b7 = st.columns(7)
                b1.metric("Win Rate", f"{wr:.1f}%")
                b2.metric("Net Return", f"{sr:+.1f}%")
                b3.metric("vs B&H", f"{outperf:+.1f}%")
                b4.metric("Max DD", f"{mdd:.1f}%")
                b5.metric("Half-Kelly", f"{hk:.1f}%")
                b6.metric("Sortino", f"{sortino:.2f}")
                b7.metric("Calmar", f"{calmar:.2f}")

                # Allocation Directive
                action = "DEPLOY CAPITAL" if hk > 0 else "FLATTEN / NO EDGE"
                box_c = "#082F49" if hk > 0 else "#450a0a"
                bdr_c = "#38BDF8" if hk > 0 else "#f87171"
                st.markdown(f"""<div class="apex-box" style="background:{box_c};border-left:5px solid {bdr_c};">
                    <div class="apex-title">⚡ HQTA Allocation Directive</div>
                    <div class="apex-action" style="color:{bdr_c};">ACTION: {action}</div>
                    <div class="apex-logic">Half-Kelly: <strong>{hk:.2f}%</strong> | GARCH(1,1) + OOS WFO</div>
                </div>""", unsafe_allow_html=True)

                # Charts
                st.markdown("---")
                chart_price_cone(ticker, df.tail(60), curr, sigma)
                chart_greeks_bar(bs)
                chart_kalman_garch(bt_df)

                # Monte Carlo — configurable path count for GOD_MODE
                st.markdown("### 🎲 Monte Carlo + Jump-Diffusion")
                mc_paths = 50_000
                if tier == "GOD_MODE":
                    mc_paths = st.slider("MC Paths", 10_000, 100_000, 50_000, 10_000,
                                         help="More paths = higher accuracy, slower compute")
                mc = QuantMath.monte_carlo_paths(
                    curr, T30, rfr, sigma, n_paths=mc_paths,
                    option_type="put", strike=sup, jump_diffusion=True, seed=42)
                chart_mc_fan(mc, curr, sup)

                # MC POP + P&L
                st.markdown("### 🎯 Monte Carlo Analytics")
                pnl = QuantMath.option_pnl(mc["S_T"], sup, bs["price"], "put")
                true_pop = QuantMath.true_pop_with_premium(mc["S_T"], sup, bs["price"], "put")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("MC POP (with premium)", f"{true_pop*100:.1f}%")
                p2.metric("Breakeven", f"${pnl['breakeven']:.2f}")
                p3.metric("Expected P&L", f"${pnl['expected_pnl']:.2f}")
                p4.metric("Profit Prob", f"{pnl['profit_prob']*100:.1f}%")

            except Exception as e:
                st.error(f"Deep Dive Error: {e}")
                st.exception(e)


# =============================================================================
# COMPLIANCE FOOTER
# =============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="font-size:0.82em;color:#94A3B8;line-height:1.7;text-align:justify;
            padding:15px;border-left:4px solid #F59E0B;background:#1E293B;border-radius:4px;">
    <b style="color:#F8FAFC;">RISK DISCLOSURE</b><br>
    VRP Quant is a quantitative research tool. It is not a registered investment advisor,
    broker-dealer, or financial planner. All outputs are for <strong>informational and educational
    purposes only</strong>. Options trading carries substantial risk. Verify all data independently.<br>
    <div style="text-align:center;font-size:0.85em;color:#64748B;margin-top:8px;">
        © 2026 vrpquant.com. All Rights Reserved.
    </div>
</div>
""", unsafe_allow_html=True)
