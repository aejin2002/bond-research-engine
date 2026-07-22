from __future__ import annotations

from datetime import datetime
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from backtest import MAIN_STRATEGY_NAME, NORMAL_CARRY_BASKET, infer_long_only_exposures, performance_metrics, run_backtest
from data_sources import CASH_PROXY_TICKERS, ETF_TICKERS, FRED_SERIES, STRUCTURED_PROXY_TICKERS, FetchStatus, cache_age_minutes, demo_data, fetch_etfs, fetch_fred
from pimco_holdings import fetch_bond_exposure_history, mbs_gap_diagnostic, position_alignment
from rule_discovery import discover_if_then_rules
from rule_library import RULE_FIELDS, RULES, SOURCES
from signals import FACTOR_WEIGHTS, recommend_weights, score_etfs


st.set_page_config(page_title="Bond Research Engine", page_icon="📚", layout="wide")

try:
    secret_key = st.secrets.get("FRED_API_KEY", "")
except FileNotFoundError:
    secret_key = ""
if secret_key and not os.getenv("FRED_API_KEY"):
    os.environ["FRED_API_KEY"] = str(secret_key)


def signal_background(value):
    if pd.isna(value):
        return ""
    if value >= 0.5:
        return "background-color:#dff0e5;color:#173b2b"
    if value <= -0.5:
        return "background-color:#f8d9d5;color:#682c20"
    return "background-color:#f5f1e8;color:#5b4a20"


def line_chart(data: pd.DataFrame | pd.Series, y_title: str = "", height: int = 330):
    frame = data.to_frame() if isinstance(data, pd.Series) else data.copy()
    frame = frame.dropna(how="all")
    if frame.empty:
        st.info("표시할 공통 데이터가 없습니다.")
        return
    index_name = frame.index.name or "index"
    long = frame.reset_index().rename(columns={index_name: "Date"}).melt("Date", var_name="Series", value_name="Value").dropna()
    chart = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Value:Q", title=y_title, scale=alt.Scale(zero=False, padding=12)),
            color=alt.Color("Series:N", legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("Date:T"), "Series:N", alt.Tooltip("Value:Q", format=".2f")],
        )
        .properties(height=height)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def common_rebased_nav(data: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    frame = data.copy().dropna(how="all")
    if frame.empty:
        return frame
    common = frame.dropna(how="any")
    if common.empty:
        return frame
    common = common.loc[common.index >= common.index.min()]
    return common / common.iloc[0] * base


def regime_timeline(regimes: pd.DataFrame, agg_prices: pd.Series):
    frame = regimes.reset_index().rename(columns={regimes.index.name or "index": "Start"})
    frame["End"] = frame["Start"] + pd.offsets.MonthEnd(1)
    frame["Changed"] = frame["Regime"].ne(frame["Regime"].shift())
    colors = ["#525b66", "#c55a5a", "#5b7cba", "#efaa45", "#57a773"]
    domain = ["Risk Off", "Stagflation", "Disinflation / Recession", "Reflation", "Goldilocks"]
    agg = agg_prices.resample("ME").last().ffill().loc[regimes.index.min(): regimes.index.max() + pd.offsets.MonthEnd(1)]
    agg = (agg / agg.iloc[0] * 100).rename("AGG Total Return Index").reset_index()
    agg.columns = ["Date", "AGG Total Return Index"]
    bands = alt.Chart(frame).mark_rect(opacity=.17).encode(
        x=alt.X("Start:T", title="Time", axis=alt.Axis(format="%Y-%m", labelAngle=-35)),
        x2="End:T",
        color=alt.Color("Regime:N", scale=alt.Scale(domain=domain, range=colors), legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("Start:T", title="Month"), "Regime:N", alt.Tooltip("Inflation 3M:Q", format=".2%"), alt.Tooltip("Growth 6M:Q", format=".2%")],
    )
    agg_line = alt.Chart(agg).mark_line(strokeWidth=3, color="#111827").encode(
        x=alt.X("Date:T", title="Time"),
        y=alt.Y("AGG Total Return Index:Q", title="AGG Total Return Index", scale=alt.Scale(zero=False, padding=12)),
        tooltip=[alt.Tooltip("Date:T", title="Month"), alt.Tooltip("AGG Total Return Index:Q", format=".2f")],
    )
    change_rules = alt.Chart(frame[frame["Changed"]]).mark_rule(color="#374151", strokeDash=[4, 4], opacity=.7).encode(
        x="Start:T", tooltip=[alt.Tooltip("Start:T", title="Transition"), "Regime:N"]
    )
    chart = alt.layer(bands, agg_line, change_rules).properties(height=340).interactive()
    st.altair_chart(chart, use_container_width=True)


def metrics_frame(results: dict) -> pd.DataFrame:
    rows = {
        MAIN_STRATEGY_NAME: results["bond_core_overlay_metrics"],
    }
    if results["structured_v2_metrics"]:
        rows["Structured Proxy v2 Candidate"] = results["structured_v2_metrics"]
    if results["target_metrics"]:
        rows["PIMCO Active Bond (BOND)"] = results["target_metrics"]
    if results["benchmark_metrics"]:
        rows["US Aggregate Bond (AGG)"] = results["benchmark_metrics"]
    return pd.DataFrame(rows).T


DURATION_ESTIMATES = {
    "SHY": 1.8, "IEF": 7.3, "TLT": 16.0, "TIP": 6.5, "LQD": 8.0,
    "HYG": 3.2, "MBB": 5.5, "JAAA": 0.4, "BKLN": 0.3, "CMBS": 5.0,
    "SGOV": 0.1, "BIL": 0.1, "USFR": 0.1, "BOND": 6.2,
}
CONVEXITY_ESTIMATES = {
    "SHY": 4.0, "IEF": 60.0, "TLT": 290.0, "TIP": 55.0, "LQD": 70.0,
    "HYG": 18.0, "MBB": -25.0, "JAAA": 1.0, "BKLN": 1.0, "CMBS": 40.0,
    "SGOV": 0.0, "BIL": 0.0, "USFR": 0.0, "BOND": 45.0,
}


def portfolio_risk_estimates(weights: pd.Series) -> dict:
    w = weights.reindex(sorted(set(DURATION_ESTIMATES) | set(weights.index)), fill_value=0.0).fillna(0.0)
    duration = float(sum(w.get(ticker, 0.0) * DURATION_ESTIMATES.get(ticker, 0.0) for ticker in w.index))
    convexity = float(sum(w.get(ticker, 0.0) * CONVEXITY_ESTIMATES.get(ticker, 0.0) for ticker in w.index))
    shock = -duration * 0.01 + 0.5 * convexity * (0.01 ** 2)
    structured = float(sum(w.get(ticker, 0.0) for ticker in STRUCTURED_PROXY_TICKERS))
    credit = float(sum(w.get(ticker, 0.0) for ticker in ["LQD", "HYG", "JAAA", "BKLN", "CMBS"]) + w.get("BOND", 0.0) * 0.35)
    return {
        "Duration": duration,
        "Convexity": convexity,
        "+100bp Rate Shock": shock,
        "Structured Sleeve": structured,
        "Credit + Structured": credit,
    }


def etf_return_snapshot(prices: pd.DataFrame) -> pd.DataFrame:
    universe = ["BOND", "AGG"] + ETF_TICKERS + STRUCTURED_PROXY_TICKERS + CASH_PROXY_TICKERS
    rows = []
    today = prices.index.max()
    for ticker in universe:
        if ticker not in prices:
            continue
        series = prices[ticker].dropna()
        if series.empty:
            continue
        latest = float(series.iloc[-1])

        def total_return(days: int) -> float:
            window = series.loc[series.index >= today - pd.DateOffset(days=days)].dropna()
            if len(window) < 2:
                return np.nan
            return float(window.iloc[-1] / window.iloc[0] - 1)

        ytd = series.loc[series.index >= pd.Timestamp(today.year, 1, 1)].dropna()
        since_years = max((series.index[-1] - series.index[0]).days / 365.25, 1 / 12)
        rows.append({
            "ETF": ticker,
            "Role": {
                "BOND": "PIMCO active bond core",
                "AGG": "US aggregate benchmark",
                "SHY": "Cash / short Treasury defense",
                "IEF": "Intermediate Treasury duration",
                "TLT": "Long duration convexity",
                "TIP": "TIPS / real yield",
                "LQD": "IG credit",
                "HYG": "High yield carry",
                "MBB": "Agency MBS",
                "JAAA": "AAA CLO / structured carry",
                "BKLN": "Senior loan / floating-rate carry",
                "CMBS": "Commercial MBS / structured spread",
                "SGOV": "0-3M T-Bill cash defense",
                "BIL": "1-3M T-Bill cash defense",
                "USFR": "Floating-rate Treasury cash defense",
            }.get(ticker, "ETF"),
            "Latest": latest,
            "1M": total_return(31),
            "3M": total_return(93),
            "YTD": float(ytd.iloc[-1] / ytd.iloc[0] - 1) if len(ytd) >= 2 else np.nan,
            "1Y": total_return(366),
            "3Y Ann.": (1 + total_return(365 * 3)) ** (1 / 3) - 1 if np.isfinite(total_return(365 * 3)) else np.nan,
            "Since Ann.": float((series.iloc[-1] / series.iloc[0]) ** (1 / since_years) - 1),
            "Start": series.index[0].date().isoformat(),
        })
    return pd.DataFrame(rows).set_index("ETF")


def period_strategy_metrics(results: dict, fred: pd.DataFrame, start: str, end: str | None = None) -> dict:
    returns = results["bond_core_overlay_returns"].dropna()
    if returns.empty:
        return {}
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else returns.index.max()
    sample = returns.loc[start_ts:end_ts]
    if len(sample) < 3:
        return {}
    rf = fred["DGS3MO"].resample("ME").last().reindex(sample.index, method="ffill") / 100 / 12
    benchmark = results["benchmark_returns"].reindex(sample.index) if results.get("benchmark_returns") is not None else None
    return performance_metrics(sample, rf, benchmark)


st.markdown(
    """
    <style>
    :root {--ink:#17191c;--muted:#687078;--line:#e3e6eb;--accent:#ef625e;}
    .stApp {background:#f7f8fb;color:var(--ink);}
    .block-container {max-width:1220px;padding-top:2.4rem;}
    [data-testid="stSidebar"] {background:#f1f3f7;border-right:1px solid #dde1e7;}
    [data-testid="stSidebar"] button {background:#3467df;color:white;border:0;font-weight:700;}
    [data-testid="stMetric"] {background:white;border:1px solid var(--line);border-radius:12px;padding:13px 15px;}
    [data-baseweb="tab-list"] {gap:18px;border-bottom:1px solid #d9dde3;}
    [data-baseweb="tab"] {height:48px;padding:0 3px;}
    [aria-selected="true"] {color:#d64f4b !important;border-bottom-color:var(--accent) !important;}
    h1 {letter-spacing:-.045em;font-weight:850 !important;} h2,h3 {letter-spacing:-.025em;}
    .pipeline {background:white;border:1px solid var(--line);border-radius:12px;padding:15px 18px;font-weight:750;margin:12px 0 24px;}
    .regime-card {padding:23px 25px;border-radius:13px;background:#e8f0ff;border:1px solid #6c8ab7;margin:8px 0 14px;}
    .regime-card h2 {font-size:2rem;margin:0 0 8px;color:#121820;}
    .eyebrow {font-size:.75rem;letter-spacing:.12em;font-weight:800;color:#6b7480;margin-bottom:12px;}
    .rule-card {background:white;border:1px solid var(--line);border-radius:11px;padding:14px 16px;margin:7px 0;}
    .muted {color:var(--muted);font-size:.86rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900, show_spinner=False)
def load_data(fred_live: bool = False):
    if os.getenv("BOND_DASHBOARD_DEMO") == "1":
        fred, prices, quotes, fred_status, etf_status = demo_data()
        agg = prices[["IEF", "MBB", "LQD"]].mean(axis=1).rename("AGG")
        bond = (prices["IEF"] * .45 + prices["MBB"] * .30 + prices["LQD"] * .25).rename("BOND")
        return fred, prices, quotes, fred_status, etf_status, bond, agg, []
    fred, fred_status = fetch_fred(force_live=fred_live)
    prices, quotes, etf_status = fetch_etfs()
    try:
        extra_prices, _, extra_status = fetch_etfs(["^MOVE"])
        prices = prices.join(extra_prices.rename(columns={"^MOVE": "MOVE"}), how="outer").ffill()
        etf_status += extra_status
    except Exception:
        pass
    try:
        structured_prices, _, structured_status = fetch_etfs(STRUCTURED_PROXY_TICKERS)
        prices = prices.join(structured_prices, how="outer").ffill()
        etf_status += structured_status
    except Exception as exc:
        for ticker in STRUCTURED_PROXY_TICKERS:
            etf_status.append(FetchStatus(ticker, False, False, "—", type(exc).__name__))
    try:
        cash_prices, _, cash_status = fetch_etfs(CASH_PROXY_TICKERS)
        prices = prices.join(cash_prices, how="outer").ffill()
        etf_status += cash_status
    except Exception as exc:
        for ticker in CASH_PROXY_TICKERS:
            etf_status.append(FetchStatus(ticker, False, False, "—", type(exc).__name__))
    bond, agg, comparator_status = None, None, []
    try:
        comp_prices, _, comparator_status = fetch_etfs(["BOND", "AGG"])
        bond, agg = comp_prices["BOND"], comp_prices["AGG"]
    except Exception:
        pass
    return fred, prices, quotes, fred_status, etf_status, bond, agg, comparator_status


@st.cache_data(ttl=86400, show_spinner=False)
def load_pimco_positions():
    if os.getenv("BOND_DASHBOARD_DEMO") == "1":
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=12, freq="QE")
        x = np.arange(len(dates))
        history = pd.DataFrame(
            {
                "Effective Duration": 6.2 + np.sin(x / 2) * .45,
                "Gross / Net": 1.55 + np.cos(x / 3) * .12,
                "Number of Positions": 1500 + x * 12,
                "Government Related": .16 + np.sin(x / 3) * .03,
                "Agency MBS": .36 + np.cos(x / 4) * .04,
                "Non-Agency MBS": .06 + np.sin(x / 5) * .01,
                "ABS / CLO": .10 + np.cos(x / 5) * .015,
                "Credit & Loans": .25 + np.sin(x / 4) * .025,
                "Municipal / Other": .04,
                "Cash & Repo": .03,
            },
            index=dates,
        )
        history.index.name = "Report Date"
        history["High Quality Proxy"] = history["Government Related"] + history["Agency MBS"]
        history["Spread Risk Proxy"] = history["Non-Agency MBS"] + history["ABS / CLO"] + history["Credit & Loans"]
        status = FetchStatus("SEC BOND Holdings", True, False, dates[-1].date().isoformat(), "offline UI fixture")
        return history, pd.DataFrame(), status
    return fetch_bond_exposure_history()


with st.sidebar:
    st.header("Research Control")
    lookback = st.slider("시장 차트 기간(년)", 1, 10, 5)
    transaction_cost = st.select_slider("편도 거래비용(bp)", options=[0, 2, 5, 10, 15], value=5)
    show_zero = st.toggle("0% 자산 포함", value=True)
    st.caption("매일 신호 감시 · 조건부 거래 · 배당재투자 총수익")
    st.divider()
    fred_live = st.toggle("FRED 실시간 갱신 시도", value=False, help="네트워크가 느리면 끄고 번들 스냅샷으로 즉시 실행하세요.")
    refresh = st.button("데이터 새로고침", width="stretch")
    st.caption("FRED API + ETF 공개 지연시세\n\n15분 캐시 및 마지막 정상 데이터 백업")
    st.divider()
    st.warning("연구용 프로토타입입니다. 현재 백테스트는 revised data를 포함하므로 point-in-time 검증 전 실거래 규칙이 아닙니다.")

if refresh:
    st.cache_data.clear()

try:
    with st.spinner("Rule Engine과 백테스트를 계산하는 중…"):
        fred, prices, quotes, fred_status, etf_status, bond, agg, comparator_status = load_data(fred_live)
        scored, ctx = score_etfs(prices, fred)
        weights = recommend_weights(scored, ctx)
        results = run_backtest(prices, fred, bond, agg, transaction_cost)
        try:
            bond_exposures, latest_bond_holdings, holdings_status = load_pimco_positions()
            discovered_rules, rule_panel = discover_if_then_rules(bond_exposures, fred, max_per_target=2)
        except Exception as holdings_exc:
            bond_exposures, latest_bond_holdings = pd.DataFrame(), pd.DataFrame()
            discovered_rules, rule_panel = pd.DataFrame(), pd.DataFrame()
            holdings_status = FetchStatus("SEC BOND Holdings", False, False, "—", type(holdings_exc).__name__)
except Exception as exc:
    st.error("데이터 또는 계산 단계에서 오류가 발생했습니다. FRED API 키와 인터넷 연결을 확인해 주세요.")
    st.exception(exc)
    st.stop()

regime_name = "Risk Off" if ctx["risk_off"] else ctx["regime"]
latest_date = fred.dropna(how="all").index.max().date().isoformat()
if not results["bond_core_overlay_weights"].empty:
    current_strategy_name = MAIN_STRATEGY_NAME
    current_strategy_weights = results["bond_core_overlay_weights"].iloc[-1].sort_values(ascending=False)
    current_strategy_signal = results["bond_core_overlay_signals"].iloc[-1] if not results["bond_core_overlay_signals"].empty else pd.Series(dtype=object)
    current_strategy_universe = ["BOND"] + ETF_TICKERS + [ticker for ticker in STRUCTURED_PROXY_TICKERS + CASH_PROXY_TICKERS if ticker in current_strategy_weights.index]
elif not results["structured_v2_weights"].empty:
    current_strategy_name = "Structured Proxy v2 Candidate"
    current_strategy_weights = results["structured_v2_weights"].iloc[-1].sort_values(ascending=False)
    current_strategy_signal = results["structured_v2_signals"].iloc[-1] if not results["structured_v2_signals"].empty else pd.Series(dtype=object)
    current_strategy_universe = ETF_TICKERS + [ticker for ticker in STRUCTURED_PROXY_TICKERS + CASH_PROXY_TICKERS if ticker in current_strategy_weights.index]
elif not results["structured_proxy_weights"].empty:
    current_strategy_name = "Structured Proxy Engine"
    current_strategy_weights = results["structured_proxy_weights"].iloc[-1].sort_values(ascending=False)
    current_strategy_signal = results["structured_proxy_signals"].iloc[-1] if not results["structured_proxy_signals"].empty else pd.Series(dtype=object)
    current_strategy_universe = ETF_TICKERS + [ticker for ticker in STRUCTURED_PROXY_TICKERS + CASH_PROXY_TICKERS if ticker in current_strategy_weights.index]
else:
    current_strategy_name = "Base Rule Engine"
    current_strategy_weights = weights.sort_values(ascending=False)
    current_strategy_signal = pd.Series(dtype=object)
    current_strategy_universe = ETF_TICKERS
current_strategy_risk = portfolio_risk_estimates(current_strategy_weights)
current_strategy_metrics = (
    results["bond_core_overlay_metrics"] if current_strategy_name == MAIN_STRATEGY_NAME
    else results["structured_v2_metrics"] if current_strategy_name == "Structured Proxy v2 Candidate"
    else results["structured_proxy_metrics"] if current_strategy_name == "Structured Proxy Engine"
    else results["metrics"]
)
current_strategy_bond_metrics = results.get("bond_core_overlay_bond_metrics", {}) if current_strategy_name == MAIN_STRATEGY_NAME else {}
st.title("Bond Research Engine")
st.caption(f"운용사 연구를 실행 가능한 롱온리 ETF Rule로 변환 · 기준일 {latest_date} · 계산 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.markdown(
    "<div class='pipeline'>운용사·학술 연구　→　Rule Library　→　ETF Mapping　→　Python Engine　→　Backtest　→　PIMCO BOND 역설계</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs(["📊 Signal", "🗓 Regime", "📈 Backtest", "📚 Rule Library", "🔬 PIMCO Compare", "🌐 Data"])

with tabs[0]:
    st.markdown("## Current Decision")
    st.markdown(
        f"<div class='regime-card'><div class='eyebrow'>CURRENT REGIME</div><h2>{regime_name}</h2>"
        f"<p>Latest-data decision as of {latest_date}<br>"
        f"VIX percentile {ctx['vix_pct']:.0%} · HY OAS 3M {ctx['hy_3m']:+.2f}%p · Unemployment gap {ctx['unrate_delta']:+.2f}%p</p></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Representative Strategy")
    structured_weight = current_strategy_risk["Structured Sleeve"]
    current_ir = current_strategy_metrics.get("Information Ratio", np.nan)
    bond_ir = current_strategy_bond_metrics.get("Information Ratio", np.nan)
    current_te = current_strategy_metrics.get("Tracking Error", np.nan)
    m0, m1, m2, m3, m4 = st.columns(5)
    ir_caption = f"IR vs BOND {bond_ir:.2f}" if np.isfinite(bond_ir) else (f"IR vs AGG {current_ir:.2f}" if np.isfinite(current_ir) else "IR —")
    m0.metric("Strategy", current_strategy_name, ir_caption)
    m1.metric("Est. Duration", f"{current_strategy_risk['Duration']:.2f}년")
    m2.metric("Est. Convexity", f"{current_strategy_risk['Convexity']:.1f}")
    m3.metric("+100bp Shock", f"{current_strategy_risk['+100bp Rate Shock']:.1%}")
    m4.metric("Structured Sleeve", f"{structured_weight:.0%}")
    st.caption("듀레이션·컨벡시티는 ETF 대표값으로 계산한 근사치입니다. MBB/CMBS는 금리·스프레드·조기상환 영향 때문에 실제 컨벡시티가 시장 상황에 따라 크게 달라질 수 있습니다.")

    full_metrics = results["bond_core_overlay_metrics"]
    shock_2022_metrics = period_strategy_metrics(results, fred, "2022-01-31", "2022-12-31")
    recovery_metrics = period_strategy_metrics(results, fred, "2023-07-31")
    k0, k1, k2, k3 = st.columns(4)
    k0.metric("CAGR since 2020-10", f"{full_metrics.get('CAGR', np.nan):.2%}", "risk-gated")
    k1.metric("Max Drawdown", f"{full_metrics.get('Max Drawdown', np.nan):.2%}")
    k2.metric("2022 Shock MDD", f"{shock_2022_metrics.get('Max Drawdown', np.nan):.2%}")
    k3.metric("CAGR since 2023-07", f"{recovery_metrics.get('CAGR', np.nan):.2%}")
    st.caption("2020-10 이후 성과는 JAAA 상장 이후 공통기간입니다. 2023-07 이후 숫자는 최근 회복장만 자른 참고값입니다.")

    left, right = st.columns([1, 1.15])
    with left:
        allocation = current_strategy_weights.rename("Weight").to_frame()
        if show_zero:
            allocation = allocation.reindex(current_strategy_universe, fill_value=0)
        st.subheader("Current Strategy Allocation")
        st.bar_chart(allocation, horizontal=True, color="#ef625e")
        st.dataframe(allocation.style.format({"Weight": "{:.1%}"}), width="stretch")
    with right:
        st.subheader("Core Rules")
        if current_strategy_name == MAIN_STRATEGY_NAME:
            st.markdown(
                f"<div class='rule-card'><b>Engine · {current_strategy_name}</b><br>"
                f"<span class='muted'>Risk-first bond allocation: severe stress can move 100% to SGOV/BIL/USFR; normal markets use BOND/structured carry · "
                f"Overlay rule: {current_strategy_signal.get('Overlay Rule', '—')} · "
                f"Cash gate: {current_strategy_signal.get('Cash Gate', '—')}</span></div>",
                unsafe_allow_html=True,
            )
        elif current_strategy_name == "Structured Proxy v2 Candidate":
            st.markdown(
                f"<div class='rule-card'><b>Engine · {current_strategy_name}</b><br>"
                f"<span class='muted'>Best config: {results['structured_v2_best_name']} · "
                f"Rule: {current_strategy_signal.get('Structured v2 Rule', '—')} · "
                f"Stress: {current_strategy_signal.get('Structured v2 Stress', '—')}</span></div>",
                unsafe_allow_html=True,
            )
        mbs_score = current_strategy_signal.get("MBS RV Score", np.nan)
        credit_score = current_strategy_signal.get("Credit Cushion Score", np.nan)
        stress_label = current_strategy_signal.get("Overlay Risk", current_strategy_signal.get("Structured v2 Stress", "—"))
        risk_votes = current_strategy_signal.get("Risk Votes", np.nan)
        core_bucket = current_strategy_signal.get("BOND Core Weight", current_strategy_weights.get("BOND", 0.0))
        core_asset = current_strategy_signal.get("Core Defense Asset", "BOND")
        core_rule = current_strategy_signal.get("Core Defense Rule", "Normal: core bucket in BOND")
        crisis_type = current_strategy_signal.get("Crisis Type", "Normal carry")
        hedge_etf = current_strategy_signal.get("Hedge ETF", core_asset)
        return_mode = current_strategy_signal.get("Return Mode", "Risk-first defense/base")
        carry_basket_weight = current_strategy_signal.get("Carry Basket Weight", 0.0)
        carry_basket_assets = current_strategy_signal.get("Carry Basket Assets", ", ".join(NORMAL_CARRY_BASKET))
        if carry_basket_weight > 0:
            core_now = (
                f"Now: {crisis_type} → normal carry basket {carry_basket_weight:.0%}; "
                f"defensive core rule is on standby: {core_rule}."
            )
        else:
            core_now = (
                f"Now: {crisis_type} → hedge {hedge_etf}, Core {core_bucket:.0%} in {core_asset}, "
                f"Active overlay {1-core_bucket:.0%}. {core_rule}."
            )
        st.markdown(
            f"<div class='rule-card'><b>1. Core / Satellite</b><br>"
            f"<span class='muted'>IF rate/inflation shock, THEN hedge with SGOV/BIL/USFR; IF recession/flight-to-quality, THEN hedge with IEF/TLT; IF normal carry, THEN use BOND/structured carry. "
            f"{core_now}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='rule-card'><b>1-b. Normal Carry Basket</b><br>"
            f"<span class='muted'>IF crisis type is Normal carry, THEN use a JAAA-heavy carry basket instead of sitting too passively in BOND. "
            f"Basket target: JAAA 50%, HYG 20%, BKLN 20%, CMBS 10%. "
            f"Now: basket {carry_basket_weight:.0%} · {return_mode} · available {carry_basket_assets}.</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='rule-card'><b>2. Aggressive RA Overlay</b><br>"
            f"<span class='muted'>IF MBS RV, credit carry, duration, curve, or real-yield signal is strong, THEN concentrate the active sleeve. "
            f"Active sleeve means the non-BOND satellite sleeve; SHY/IEF are not deleted, they are defensive assets used when risk rises. "
            f"Now: MBS {mbs_score:+.1f}, Credit {credit_score:+.1f}, Structured {structured_weight:.0%}.</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='rule-card'><b>3. Risk Gate</b><br>"
            f"<span class='muted'>IF NFCI, HY spread, MOVE, labor, or credit price stress appears, THEN the core bucket and overlay both de-risk. "
            f"Rate/MOVE shock uses SGOV/BIL/USFR; recession or flight-to-quality can use IEF/TLT; normal carry uses BOND/structured carry. "
            f"Now: Stress {stress_label}, Risk votes {risk_votes:.0f}/5.</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='rule-card'><b>4. Portfolio Risk Budget</b><br>"
            f"<span class='muted'>The first job is not to beat BOND every month; it is to avoid bond-crash regimes before adding carry. "
            f"Now: Duration {current_strategy_risk['Duration']:.2f}y, Convexity {current_strategy_risk['Convexity']:.1f}, "
            f"Credit+Structured {current_strategy_risk['Credit + Structured']:.0%}, AGG Tracking Error {current_te:.1%}.</span></div>",
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Top Position", current_strategy_weights.index[0], f"{current_strategy_weights.iloc[0]:.0%}")
    c2.metric("10Y–2Y", f"{ctx['slope']:.2f}%p")
    c3.metric("10Y Real Yield", f"{ctx['real_yield']:.2f}%")
    c4.metric("IG OAS", f"{ctx['ig_oas']:.2f}%")
    c5.metric("HY OAS", f"{ctx['hy_oas']:.2f}%")
    with st.expander("Diagnostic details"):
        st.caption("아래는 현재 대표 전략의 실제 overlay 유니버스입니다. JAAA/BKLN/CMBS는 기본 7개 ETF scorecard가 아니라 overlay rule과 현재 비중으로 봅니다.")
        overlay_rows = []
        for ticker in [ticker for ticker in current_strategy_universe if ticker != "BOND"]:
            if ticker not in current_strategy_weights.index:
                continue
            overlay_rows.append({
                "Ticker": ticker,
                "Current Weight": current_strategy_weights.get(ticker, 0.0),
                "Role": {
                    "JAAA": "AAA CLO / structured carry",
                    "BKLN": "Senior loan / floating-rate carry",
                    "CMBS": "Commercial MBS / structured spread",
                    "HYG": "High yield carry",
                    "MBB": "Agency MBS",
                    "IEF": "Intermediate duration",
                    "TLT": "Long duration convexity",
                    "TIP": "Real-yield / inflation",
                    "LQD": "IG credit",
                    "SHY": "Short Treasury",
                    "SGOV": "0-3M T-Bill cash defense",
                    "BIL": "1-3M T-Bill cash defense",
                    "USFR": "Floating-rate Treasury cash defense",
                }.get(ticker, "ETF"),
                "Signal Note": {
                    "JAAA": f"JAAA vs SHY 1M {current_strategy_signal.get('JAAA vs SHY 1M', np.nan):+.1%}",
                    "BKLN": f"BKLN vs SHY 1M {current_strategy_signal.get('BKLN vs SHY 1M', np.nan):+.1%}",
                    "CMBS": f"CMBS vs MBB 3M {current_strategy_signal.get('CMBS vs MBB 3M', np.nan):+.1%}",
                }.get(ticker, current_strategy_signal.get("Overlay Rule", "—")),
            })
        if overlay_rows:
            overlay_diag = pd.DataFrame(overlay_rows).sort_values("Current Weight", ascending=False)
            st.dataframe(overlay_diag.style.format({"Current Weight": "{:.1%}"}), width="stretch", hide_index=True)
        with st.expander("Base 7-ETF factor scorecard"):
            factors = list(FACTOR_WEIGHTS)
            st.dataframe(scored[factors + ["Total Score"]].style.map(signal_background).format({"Total Score": "{:+.2f}"}), width="stretch")

    st.subheader("ETF Historical Return Snapshot")
    st.caption("현재 전략 유니버스의 과거 총수익률입니다. 구조화 ETF는 상장일이 짧아 표본 시작점이 JAAA의 2020년 10월로 제한됩니다.")
    display_prices = prices.copy()
    if bond is not None:
        display_prices = display_prices.join(bond.rename("BOND"), how="outer").ffill()
    if agg is not None:
        display_prices = display_prices.join(agg.rename("AGG"), how="outer").ffill()
    history = etf_return_snapshot(display_prices)
    st.dataframe(
        history.style.format({
            "Latest": "{:.2f}",
            "1M": "{:+.1%}",
            "3M": "{:+.1%}",
            "YTD": "{:+.1%}",
            "1Y": "{:+.1%}",
            "3Y Ann.": "{:+.1%}",
            "Since Ann.": "{:+.1%}",
        }),
        width="stretch",
    )

with tabs[1]:
    st.markdown("## Regime Change Map")
    st.caption("x축은 시간, y축은 AGG 총수익지수입니다. 배경색은 해당 월의 레짐이고 점선은 레짐 전환 시점입니다.")
    chart_regime = results["regimes"].iloc[-1] if not results["regimes"].empty else pd.Series(dtype=object)
    a0, b0 = st.columns(2)
    with a0:
        st.metric("현재 의사결정 레짐", regime_name, f"as of {latest_date}")
    with b0:
        if not chart_regime.empty:
            st.metric("차트 마지막 월말 레짐", chart_regime["Regime"], f"as of {results['regimes'].index[-1].date().isoformat()}")
    st.info("Current Decision은 최신 가용 데이터 기준이고, Regime Change Map은 백테스트용 월말 히스토리입니다. 차트 tooltip에 보이는 레짐은 마우스를 올린 과거 월의 레짐이라 현재 레짐과 다를 수 있습니다.")
    if agg is not None:
        regime_timeline(results["regimes"], agg)
    regime_changes = results["regimes"][results["regimes"]["Regime"].ne(results["regimes"]["Regime"].shift())]
    st.subheader("Regime Transition Log")
    transition = regime_changes[["Regime", "Inflation 3M", "Growth 6M"]].copy()
    st.dataframe(transition.style.format({"Inflation 3M":"{:.2%}", "Growth 6M":"{:.2%}"}), width="stretch")
    a, b = st.columns(2)
    with a:
        st.subheader("Inflation Momentum")
        monthly_macro = fred[["CPILFESL"]].resample("ME").last().dropna()
        macro = pd.DataFrame({"Core CPI 3M ann.": (monthly_macro["CPILFESL"] / monthly_macro["CPILFESL"].shift(3)) ** 4 - 1,
                              "Core CPI YoY": monthly_macro["CPILFESL"].pct_change(12)})
        line_chart(macro.loc[results["common_start"]:], "Rate")
    with b:
        st.subheader("Growth & Labor")
        macro2 = fred[["INDPRO", "UNRATE"]].resample("ME").last().dropna(how="all")
        macro2["Industrial Production 6M ann."] = (macro2["INDPRO"] / macro2["INDPRO"].shift(6)) ** 2 - 1
        unemployment_rate = macro2["UNRATE"].ffill(limit=1)
        macro2["Unemployment gap"] = unemployment_rate - unemployment_rate.rolling(12, min_periods=6).min()
        line_chart(macro2[["Industrial Production 6M ann.", "Unemployment gap"]].loc[results["common_start"]:], "Rate / %p")
        st.caption("Unemployment gap은 실업률 - 최근 12개월 최저 실업률입니다. FRED 스냅샷의 1개월 결측 때문에 뒤 데이터가 사라지지 않도록 실업률은 최대 1개월만 보간하고, 최소 6개월 관측치로 계산합니다.")

with tabs[2]:
    st.markdown("## Walk-Forward Backtest")
    st.markdown(
        f"**{MAIN_STRATEGY_NAME}**가 현재 대표 전략입니다. "
        "위험 신호가 먼저 포트폴리오의 방어 강도를 정하고, 위험이 낮을 때만 BOND·structured carry·credit carry를 사용합니다. "
        "MOVE/금리 급등이나 credit/liquidity stress가 강하면 BOND를 고집하지 않고 SGOV/BIL/USFR 같은 cash defense bucket까지 이동합니다."
    )
    start = results.get("main_strategy_start", results["common_start"]).date().isoformat()
    st.caption(f"대표 전략 공통 데이터가 실제 존재하는 {start}부터 시작 · 월별 리밸런싱 · 편도 비용 {transaction_cost}bp · 신호 산출 후 익월 수익 적용")
    st.info("CAGR이 5%대에서 낮아진 주된 이유는 표본 시작이 2023-07 회복장 중심에서 JAAA 상장 이후인 2020-10로 앞당겨졌기 때문입니다. 이 구간에는 2022년 금리 급등/채권 약세장이 포함됩니다.")
    metrics = metrics_frame(results)
    display_metrics = metrics[[col for col in [
        "CAGR", "Ann. Vol", "Sharpe", "Sortino", "Max Drawdown", "Calmar", "Hit Rate",
        "Excess t-stat", "Correlation", "Beta", "Tracking Error", "Information Ratio",
    ] if col in metrics]].copy()
    pct_cols = [col for col in ["CAGR", "Ann. Vol", "Max Drawdown", "Hit Rate", "Tracking Error"] if col in display_metrics]
    formats = {col:"{:.1%}" for col in pct_cols} | {col:"{:.2f}" for col in display_metrics if col not in pct_cols}
    st.dataframe(display_metrics.style.format(formats), width="stretch")
    st.caption("Correlation, Beta, Tracking Error, Information Ratio는 전략 행에서는 AGG 대비 기준입니다. BOND 대비 알파는 아래 별도 카드에서 봅니다.")
    st.caption("Sharpe는 3개월 T-Bill 수익률을 차감한 초과수익 기준입니다. 수익률이 양수여도 현금보다 위험 대비 보상이 낮으면 음수가 됩니다.")
    overlay_metrics = results["bond_core_overlay_metrics"]
    overlay_bond_metrics = results["bond_core_overlay_bond_metrics"]
    if overlay_metrics:
        st.success(
            f"{MAIN_STRATEGY_NAME} · CAGR {overlay_metrics.get('CAGR', np.nan):.1%} · "
            f"Sharpe {overlay_metrics.get('Sharpe', np.nan):.2f} · MDD {overlay_metrics.get('Max Drawdown', np.nan):.1%} · "
            f"BOND 대비 IR {overlay_bond_metrics.get('Information Ratio', np.nan):.2f}."
        )
    st.caption("이 버전은 BOND를 항상 코어로 들고 가지 않습니다. 목표는 채권 급락 레짐을 먼저 피하고, 위험이 낮을 때만 carry/structured sleeve로 수익을 더하는 것입니다.")
    comparison_nav = results["nav"][[col for col in [
        MAIN_STRATEGY_NAME,
        "Structured Proxy v2 Candidate",
        "PIMCO Active Bond (BOND)",
        "US Aggregate Bond (AGG)",
    ] if col in results["nav"]]]
    comparison_nav = common_rebased_nav(comparison_nav)
    st.subheader("Growth of $100 · Common Start")
    if not comparison_nav.empty:
        st.caption(f"모든 라인이 같은 시작점 {comparison_nav.index.min().date().isoformat()} = 100에서 출발합니다.")
    line_chart(comparison_nav, "Portfolio Value", 370)
    main_returns = results["bond_core_overlay_returns"].dropna()
    main_nav = ((1 + main_returns).cumprod() * 100).rename(MAIN_STRATEGY_NAME)
    drawdown = (main_nav / main_nav.cummax() - 1).rename(f"{MAIN_STRATEGY_NAME} Drawdown")
    st.subheader("Drawdown · Main Strategy")
    line_chart(drawdown, "Drawdown", 230)
    a, b = st.columns(2)
    with a:
        st.subheader("Main Strategy Allocation History")
        if not results["bond_core_overlay_weights"].empty:
            weight_long = results["bond_core_overlay_weights"].reset_index().rename(columns={"index":"Date"}).melt("Date", var_name="ETF", value_name="Weight")
            area = alt.Chart(weight_long).mark_area().encode(
                x=alt.X("Date:T", title=None),
                y=alt.Y("Weight:Q", stack="normalize", axis=alt.Axis(format="%")),
                color="ETF:N",
                tooltip=["Date:T", "ETF:N", alt.Tooltip("Weight:Q", format=".1%")],
            ).properties(height=300)
            st.altair_chart(area, use_container_width=True)
    with b:
        st.subheader("Main Strategy Turnover")
        if not results["bond_core_overlay_turnover"].empty:
            monthly_turn = results["bond_core_overlay_turnover"].resample("ME").sum()
            st.metric("연환산 평균 회전율", f"{monthly_turn.mean()*12:.0%}")
            line_chart(monthly_turn, "One-way Turnover", 245)

    st.subheader("Main Strategy Signals")
    source_cols = ["MBS RV Score", "Credit Cushion Score", "Duration/MOVE Score", "Curve Score", "Real Yield Score", "Overlay Structured"]
    if all(column in results["bond_core_overlay_signals"] for column in source_cols):
        line_chart(results["bond_core_overlay_signals"][source_cols].tail(36), "Score / Weight", 260)
    signal_cols = [
        "Regime", "Dominant Alpha", "Crisis Type", "Hedge ETF", "Core Allocation Rule", "Core Defense Asset", "Core Defense Rule", "Overlay Rule", "Overlay Risk", "BOND Core Weight", "Active Overlay Weight",
        "Return Mode", "Carry Basket Weight", "Carry Basket Assets",
        "Overlay HYG + BKLN", "Overlay Structured", "Overlay Duration Tilt",
        "MBS RV Score", "Credit Cushion Score", "Duration/MOVE Score", "Curve Score", "Real Yield Score",
        "MOVE Percentile", "VIX Percentile", "NFCI", "HY OAS 3M", "10Y 3M Change",
        "JAAA vs SHY 1M", "BKLN vs SHY 1M", "CMBS vs MBB 3M", "Risk Votes", "Triggered", "Cash Gate",
    ]
    signal_cols = [col for col in signal_cols if col in results["bond_core_overlay_signals"]]
    main_signals = results["bond_core_overlay_signals"][signal_cols].tail(24).sort_index(ascending=False)
    st.dataframe(
        main_signals.style.format({
            "BOND Core Weight":"{:.0%}", "Active Overlay Weight":"{:.0%}",
            "Carry Basket Weight":"{:.0%}",
            "Overlay HYG + BKLN":"{:.0%}", "Overlay Structured":"{:.0%}", "Overlay Duration Tilt":"{:.0%}",
            "MBS RV Score":"{:+.1f}", "Credit Cushion Score":"{:+.1f}",
            "Duration/MOVE Score":"{:+.1f}", "Curve Score":"{:+.1f}", "Real Yield Score":"{:+.1f}",
            "MOVE Percentile":"{:.0%}", "VIX Percentile":"{:.0%}", "HY OAS 3M":"{:+.2f}",
            "10Y 3M Change":"{:+.2f}",
            "JAAA vs SHY 1M":"{:+.1%}", "BKLN vs SHY 1M":"{:+.1%}", "CMBS vs MBB 3M":"{:+.1%}",
            "Risk Votes":"{:.0f}",
        }),
        width="stretch",
    )
    st.subheader("Main Strategy Trade Log")
    if results["bond_core_overlay_trades"].empty:
        st.info(f"{MAIN_STRATEGY_NAME} 거래가 없습니다.")
    else:
        trade_log = results["bond_core_overlay_trades"].tail(30).sort_index(ascending=False)
        st.dataframe(trade_log.style.format({"Turnover":"{:.1%}", "HY OAS 3M":"{:+.2f}", "MOVE":"{:.1f}", "MOVE Percentile":"{:.0%}", "VIX Percentile":"{:.0%}", "HYG vs SHY 1M":"{:+.1%}", "Risk Votes":"{:.0f}"}), width="stretch")
    with st.expander("Structured Proxy v2 candidate grid"):
        if not results["structured_v2_summary"].empty:
            st.caption("인샘플 비교입니다. Joint Score는 AGG 초과수익을 보상하고, AGG보다 나쁜 MDD와 AGG 추적오차를 벌점으로 줍니다.")
            grid_cols = [col for col in [
                "Joint Score", "Ann. AGG Alpha", "CAGR", "Sharpe", "Max Drawdown", "Tracking Error",
                "Avg Structured Sleeve", "Ann. Turnover", "Cap", "Mix", "Gate",
            ] if col in results["structured_v2_summary"]]
            st.dataframe(
                results["structured_v2_summary"][grid_cols].head(12).style.format({
                    "Joint Score":"{:.3f}", "Ann. AGG Alpha":"{:+.1%}", "CAGR":"{:.1%}",
                    "Sharpe":"{:.2f}", "Max Drawdown":"{:.1%}", "Tracking Error":"{:.1%}",
                    "Avg Structured Sleeve":"{:.1%}", "Ann. Turnover":"{:.0%}", "Cap":"{:.0%}",
                }),
                width="stretch",
            )

with tabs[3]:
    st.markdown("## Rule Library")
    st.caption("각 Rule은 같은 15개 질문을 통과해야 의사결정 엔진에 들어갑니다.")
    selected_rule = st.selectbox("Rule 선택", [item["Rule"] for item in RULES])
    item = next(entry for entry in RULES if entry["Rule"] == selected_rule)
    for number, field in enumerate(RULE_FIELDS, 1):
        st.markdown(f"<div class='rule-card'><b>{number}. {field}</b><br><span class='muted'>{item[field]}</span></div>", unsafe_allow_html=True)
    st.subheader("Evidence")
    for source_name in item["근거 자료"]:
        st.markdown(f"- [{source_name}]({SOURCES[source_name]})")
    st.subheader("Library Matrix")
    matrix = pd.DataFrame([{field: entry[field] for field in ["Rule", "팩터 정의", "한계점", "ETF 근사 가능성", "PIMCO와의 연결"]} for entry in RULES])
    st.dataframe(matrix, width="stretch", hide_index=True)

with tabs[4]:
    st.markdown("## PIMCO BOND Reverse Engineering")
    st.markdown(f"`BOND`는 비교 대상이지만 더 이상 무조건 코어는 아닙니다. 여기서는 **{MAIN_STRATEGY_NAME}**이 BOND/AGG 대비 위기 방어와 알파를 만들 수 있는지 봅니다.")
    if results["target_returns"] is None:
        st.warning("BOND 데이터를 불러오지 못해 역설계 비교를 생략했습니다.")
    else:
        compare_cols = [col for col in ["Correlation", "Beta", "Tracking Error", "Information Ratio"] if col in metrics]
        compare_names = [MAIN_STRATEGY_NAME, "Structured Proxy v2 Candidate"]
        compare = metrics.loc[[name for name in compare_names if name in metrics.index], compare_cols]
        st.caption("아래 Correlation, Beta, Tracking Error, Information Ratio는 AGG 대비 기준입니다. BOND와의 실제 포지션 유사도는 아래 N-PORT 섹션에서 따로 봅니다.")
        st.dataframe(compare.style.format({"Correlation":"{:.2f}", "Beta":"{:.2f}", "Tracking Error":"{:.1%}", "Information Ratio":"{:.2f}"}), width="stretch")
        bond_relative = results["bond_core_overlay_bond_metrics"]
        if bond_relative:
            br1, br2, br3 = st.columns(3)
            br1.metric("BOND 대비 Tracking Error", f"{bond_relative.get('Tracking Error', np.nan):.1%}")
            br2.metric("BOND 대비 Information Ratio", f"{bond_relative.get('Information Ratio', np.nan):.2f}")
            br3.metric("BOND 대비 Beta", f"{bond_relative.get('Beta', np.nan):.2f}")
        st.subheader("Main Strategy vs Structured Proxy v2 vs BOND vs AGG")
        compare_nav = common_rebased_nav(results["nav"][[col for col in [MAIN_STRATEGY_NAME, "Structured Proxy v2 Candidate", "PIMCO Active Bond (BOND)", "US Aggregate Bond (AGG)"] if col in results["nav"]]])
        line_chart(compare_nav, "Portfolio Value")
        st.subheader("BOND Return-Implied Long-Only Exposures")
        inferred = infer_long_only_exposures(prices, bond)
        if not inferred.empty:
            inferred_long = inferred.reset_index().rename(columns={"index":"Date"}).melt("Date", var_name="ETF", value_name="Weight")
            exposure_chart = alt.Chart(inferred_long).mark_area().encode(x=alt.X("Date:T", title=None), y=alt.Y("Weight:Q", stack="normalize", axis=alt.Axis(format="%")), color="ETF:N", tooltip=["Date:T", "ETF:N", alt.Tooltip("Weight:Q", format=".1%")]).properties(height=330)
            st.altair_chart(exposure_chart, use_container_width=True)
            st.caption("24개월 rolling ridge return replication의 비음수 정규화 추정치입니다. 실제 보유비중이 아니라 수익률로부터 추론한 근사 노출입니다.")
        st.subheader("Actual Position Reverse Engineering")
        if bond_exposures.empty:
            st.warning("BOND의 SEC N-PORT 포지션 이력을 불러오지 못했습니다. 수익률 기반 비교만 표시합니다.")
        else:
            latest_exposure = bond_exposures.iloc[-1]
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("공시 기준일", bond_exposures.index[-1].date().isoformat())
            p2.metric("BOND Effective Duration", f"{latest_exposure['Effective Duration']:.2f}년")
            p3.metric("Gross / Net", f"{latest_exposure['Gross / Net']:.2f}x")
            p4.metric("공시 포지션", f"{latest_exposure['Number of Positions']:,.0f}개")

            aligned_positions, position_metrics = position_alignment(results["bond_core_overlay_weights"], bond_exposures)
            if position_metrics:
                a1, a2, a3 = st.columns(3)
                a1.metric("포지션 변화 방향 일치율", f"{position_metrics['Direction Match']:.0%}")
                a2.metric("Position Fit", f"{position_metrics['Position Fit']:.0%}")
                a3.metric("정규화 노출 MAE", f"{position_metrics['Exposure MAE']:.2f}")
            st.caption("SEC 공개 N-PORT의 분기말 전체 포지션과 DV01을 사용합니다. Effective Duration = 전체 DV01 ÷ 순자산 × 10,000.")

            if not aligned_positions.empty:
                duration_compare = aligned_positions[["BOND Effective Duration", "Rule Effective Duration"]]
                line_chart(duration_compare, "Effective Duration (years)", 280)
            krd_cols = [column for column in ["1Y KRD", "5Y KRD", "10Y KRD", "30Y KRD"] if column in bond_exposures]
            if krd_cols:
                st.markdown("#### BOND Key Rate Duration")
                line_chart(bond_exposures[krd_cols], "Duration contribution (years)", 280)

            sector_cols = ["Government Related", "Agency MBS", "Non-Agency MBS", "ABS / CLO", "Credit & Loans", "Municipal / Other", "Cash & Repo"]
            sector_long = bond_exposures[sector_cols].reset_index().melt("Report Date", var_name="Sector", value_name="Weight")
            sector_chart = alt.Chart(sector_long).mark_area().encode(
                x=alt.X("Report Date:T", title=None),
                y=alt.Y("Weight:Q", stack="normalize", axis=alt.Axis(format="%")),
                color=alt.Color("Sector:N", legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("Report Date:T"), "Sector:N", alt.Tooltip("Weight:Q", format=".1%")],
            ).properties(height=330)
            st.markdown("#### BOND Funded Holdings Sector Mix")
            st.altair_chart(sector_chart, use_container_width=True)
            st.caption("파생상품의 명목 노출은 포함하지 않은 funded holdings 구성입니다. PIMCO 공식 GMV sector allocation과 정의가 다르므로 방향 비교용입니다.")

            quality = bond_exposures[["High Quality Proxy", "Spread Risk Proxy"]]
            st.markdown("#### Quality Proxy")
            line_chart(quality, "Share of funded holdings", 260)
            st.caption("N-PORT에는 신용등급 필드가 없어 Government+Agency MBS를 High Quality, Non-Agency MBS+ABS/CLO+Credit/Loans를 Spread Risk 프록시로 사용합니다. 실제 평균 신용등급으로 해석하면 안 됩니다.")

            mbs_gap_weights = results["bond_core_overlay_weights"]
            mbs_gap_signals = results["bond_core_overlay_signals"]
            mbs_gap, mbs_gap_metrics = mbs_gap_diagnostic(mbs_gap_weights, bond_exposures, mbs_gap_signals)
            if not mbs_gap.empty:
                st.markdown("### MBS RV Gap Diagnostic")
                st.caption("포트폴리오는 risk-first gate가 먼저 SHY/IEF/BOND 방어자산을 고르고, 위험이 낮을 때만 MBB·JAAA·BKLN·CMBS로 추가 틸트를 줍니다.")
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("최근 Agency Gap", f"{mbs_gap_metrics.get('Latest Agency Gap', np.nan):+.1%}")
                g2.metric("최근 Missing Structured", f"{mbs_gap_metrics.get('Latest Missing Structured Sleeve', np.nan):.1%}")
                g3.metric("최근 Total Securitized Gap", f"{mbs_gap_metrics.get('Latest Total Securitized Gap', np.nan):+.1%}")
                g4.metric("확장 ETF 반영 후 Gap", f"{mbs_gap_metrics.get('Latest Total Gap After Structured Proxy', np.nan):+.1%}")
                gap_chart_cols = [
                    "BOND Agency MBS", "Rule MBB", "BOND Non-Agency MBS",
                    "BOND ABS / CLO", "BOND Total Securitized", "Rule Explicit Securitized",
                    "Rule Structured Proxy", "Rule Total Securitized Proxy",
                ]
                line_chart(mbs_gap[gap_chart_cols], "Share of funded holdings", 300)
                gap_table_cols = [
                    "BOND Agency MBS", "Rule MBB", "Agency Gap", "Missing Structured Sleeve",
                    "Rule Structured Proxy", "BOND Total Securitized", "Rule Total Securitized Proxy",
                    "Total Securitized Gap", "Total Gap After Structured Proxy", "MBS RV Score",
                    "MBS Spread Percentile", "MBB vs IEF 3M", "MOVE Percentile",
                ]
                st.dataframe(
                    mbs_gap[gap_table_cols].tail(12).sort_index(ascending=False).style.format({
                        "BOND Agency MBS":"{:.1%}", "Rule MBB":"{:.1%}", "Agency Gap":"{:+.1%}",
                        "Missing Structured Sleeve":"{:.1%}", "BOND Total Securitized":"{:.1%}",
                        "Rule Structured Proxy":"{:.1%}", "Rule Total Securitized Proxy":"{:.1%}",
                        "Total Securitized Gap":"{:+.1%}", "Total Gap After Structured Proxy":"{:+.1%}", "MBS RV Score":"{:+.1f}",
                        "MBS Spread Percentile":"{:.0%}", "MBB vs IEF 3M":"{:+.1%}", "MOVE Percentile":"{:.0%}",
                    }),
                    width="stretch",
                )
                if np.isfinite(mbs_gap_metrics.get("MBS Score vs Next Total Securitized Change Corr", np.nan)):
                    st.info(
                        f"MBS RV Score와 다음 공시의 BOND 총 securitized 비중 변화 상관은 "
                        f"{mbs_gap_metrics['MBS Score vs Next Total Securitized Change Corr']:.2f}입니다. "
                        "표본이 작아 확정 규칙은 아니지만, 신호가 BOND의 실제 MBS/structured 움직임을 따라가는지 보는 단서입니다."
                    )

            if not latest_bond_holdings.empty:
                top = latest_bond_holdings[latest_bond_holdings["Market Value"] > 0].nlargest(15, "Market Value").copy()
                top["Funded Weight"] = top["Market Value"] / latest_bond_holdings.loc[latest_bond_holdings["Market Value"] > 0, "Market Value"].sum()
                st.markdown("#### Latest Disclosed Positions")
                st.dataframe(top[["Name", "Title", "Sector", "Asset Category", "Country", "Maturity", "Funded Weight"]].style.format({"Funded Weight":"{:.2%}"}), width="stretch", hide_index=True)
            st.markdown("### Observed IF–THEN Rulebook")
            st.caption(
                f"{bond_exposures.index.min().date().isoformat()}~{bond_exposures.index.max().date().isoformat()}의 "
                f"{len(bond_exposures)}개 분기 공시를 사용합니다. 각 IF는 해당 분기 말 시장상태, THEN은 다음 분기 BOND 포지션 변화입니다."
            )
            st.warning("내부 규칙의 증명이 아니라 다음 공시에서 검증할 행동 가설입니다. 작은 표본에서 여러 임계값을 탐색했기 때문에 raw p-value와 탐색횟수 보정값, 표본 수, 기간 안정성을 함께 봐야 합니다.")
            if discovered_rules.empty:
                st.info("현재 표본에서 최소 적중률과 Lift 조건을 통과한 IF–THEN 후보가 없습니다.")
            else:
                rule_target = st.selectbox("역설계 대상", ["전체"] + discovered_rules["Target"].drop_duplicates().tolist(), key="observed_rule_target")
                shown_rules = discovered_rules if rule_target == "전체" else discovered_rules[discovered_rules["Target"] == rule_target]
                for _, rule in shown_rules.head(6).iterrows():
                    oos_text = "N/A" if pd.isna(rule["OOS 적중률"]) else f"{rule['OOS 적중률']:.0%}"
                    st.markdown(
                        f"<div class='rule-card'><b>{rule['IF']}<br>{rule['THEN']}</b><br>"
                        f"<span class='muted'>학습 적중률 {rule['적중률']:.0%} · Baseline {rule['기준 적중률']:.0%} · "
                        f"Lift {rule['Lift']:+.0%} · 학습 n={int(rule['관측 수'])} · "
                        f"OOS {oos_text} (n={int(rule['OOS 관측 수'])}) · 등급 {rule['신뢰도']}<br>"
                        f"경제적 논리: {rule['경제적 논리']}</span></div>",
                        unsafe_allow_html=True,
                    )
                rule_table = shown_rules[["Target", "IF", "THEN", "관측 수", "적중률", "기준 적중률", "Lift", "OOS 관측 수", "OOS 적중률", "p-value", "탐색보정 p-value", "기간 안정성", "신뢰도"]]
                st.dataframe(
                    rule_table.style.format({"적중률":"{:.0%}", "기준 적중률":"{:.0%}", "Lift":"{:+.0%}", "OOS 적중률":"{:.0%}", "p-value":"{:.3f}", "탐색보정 p-value":"{:.3f}"}),
                    width="stretch",
                )
        st.info("현재는 SEC가 공개하는 분기말 실제 포지션을 사용합니다. PIMCO의 현재 일별 holdings는 월말마다 스냅샷을 누적해야 이후 월별 역사가 만들어지며, 공개되지 않은 과거 월은 보간하지 않습니다.")

with tabs[5]:
    st.markdown("## Data Availability")
    rows = []
    for name in fred.columns:
        clean = fred[name].dropna()
        rows.append({"Series":name, "Description":FRED_SERIES.get(name, name), "Start":clean.index.min().date() if len(clean) else None, "End":clean.index.max().date() if len(clean) else None, "Observations":len(clean)})
    for name in ETF_TICKERS + [ticker for ticker in STRUCTURED_PROXY_TICKERS + CASH_PROXY_TICKERS if ticker in prices]:
        clean = prices[name].dropna()
        rows.append({"Series":name, "Description":"ETF total return price", "Start":clean.index.min().date(), "End":clean.index.max().date(), "Observations":len(clean)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.success(f"백테스트와 레짐 차트는 모든 필수 데이터가 겹치는 {results['common_start'].date().isoformat()}부터 자동 절단했습니다.")
    st.subheader("Market Monitor")
    cutoff = fred.index.max() - pd.DateOffset(years=lookback)
    a, b = st.columns(2)
    with a:
        spread = fred[["BAMLC0A0CM", "BAMLH0A0HYM2"]].dropna(how="all").loc[cutoff:]
        spread.columns = ["IG OAS", "HY OAS"]
        line_chart(spread, "OAS (%)")
    with b:
        real = fred[["DFII10", "T10YIE"]].dropna(how="all").loc[cutoff:]
        real.columns = ["10Y Real Yield", "10Y Breakeven"]
        line_chart(real, "Rate (%)")
    normalized = prices[ETF_TICKERS].dropna(how="all").loc[prices.index.max() - pd.DateOffset(years=lookback):].ffill()
    normalized = normalized.loc[:, normalized.notna().any()]
    normalized = normalized / normalized.iloc[0] * 100
    st.subheader("ETF Total Return Index")
    line_chart(normalized, "Index", 380)
    statuses = [fred_status] + etf_status + comparator_status + [holdings_status]
    st.subheader("Source Status")
    st.dataframe(pd.DataFrame([status.__dict__ for status in statuses]), width="stretch")
    with st.expander("Local Cache Age"):
        st.json(cache_age_minutes())
