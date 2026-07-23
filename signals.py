from __future__ import annotations

import numpy as np
import pandas as pd

from data_sources import ETF_TICKERS
from risk_config import RISK_CONFIG


ETF_META = {
    "SHY": ("Short Treasury", "단기 국채·대기자산", "DGS2", 1.8),
    "IEF": ("Intermediate Treasury", "7–10년 중기 국채", "DGS10", 7.3),
    "TLT": ("Long Treasury", "20년+ 장기 듀레이션", "DGS30", 16.0),
    "LQD": ("IG Credit", "투자등급 회사채", "IG_YIELD", 8.0),
    "HYG": ("High Yield", "하이일드 회사채", "HY_YIELD", 3.2),
    "MBB": ("Agency MBS", "기관 MBS", "MORTGAGE30US", 5.5),
    "TIP": ("TIPS", "물가연동국채", "DFII10", 6.5),
}

FACTOR_WEIGHTS = {
    "Carry": 0.20,
    "Momentum": 0.20,
    "Value": 0.15,
    "Real Yield": 0.10,
    "Yield Curve": 0.10,
    "Credit Spread": 0.10,
    "Defensive": 0.05,
    "Regime": 0.10,
}

CAPS = {"SHY": 1.0, "IEF": 0.40, "TLT": 0.25, "TIP": 0.30,
        "LQD": 0.35, "HYG": 0.20, "MBB": 0.30}


def _latest(series: pd.Series, default: float = np.nan) -> float:
    clean = series.dropna()
    return float(clean.iloc[-1]) if len(clean) else default


def _asof(series: pd.Series, date: pd.Timestamp) -> float:
    clean = series.dropna().loc[:date]
    return float(clean.iloc[-1]) if len(clean) else np.nan


def _return_at(prices: pd.DataFrame, ticker: str, months: int) -> float:
    s = prices[ticker].dropna()
    if len(s) < 2:
        return np.nan
    target = s.index[-1] - pd.DateOffset(months=months)
    base = _asof(s, target)
    return float(s.iloc[-1] / base - 1) if np.isfinite(base) and base else np.nan


def _percentile(series: pd.Series, years: int = 10) -> float:
    s = series.dropna()
    if not len(s):
        return 0.5
    cutoff = s.index.max() - pd.DateOffset(years=years)
    s = s.loc[cutoff:]
    return float((s <= s.iloc[-1]).mean())


def build_market_series(fred: pd.DataFrame) -> pd.DataFrame:
    x = fred.copy()
    x["IG_YIELD"] = x["DGS10"] + x["BAMLC0A0CM"]
    x["HY_YIELD"] = x["DGS5"] + x["BAMLH0A0HYM2"]
    return x


def regime_snapshot(fred: pd.DataFrame) -> dict:
    monthly = fred[["CPILFESL", "INDPRO", "UNRATE"]].resample("ME").last().ffill()
    core = monthly["CPILFESL"].dropna()
    ip = monthly["INDPRO"].dropna()
    unrate = monthly["UNRATE"].dropna()
    inf_3m = (core.iloc[-1] / core.iloc[-4]) ** 4 - 1 if len(core) >= 4 else np.nan
    inf_yoy = core.iloc[-1] / core.iloc[-13] - 1 if len(core) >= 13 else np.nan
    growth_series = (ip / ip.shift(6)) ** 2 - 1
    growth_6m = float(growth_series.iloc[-1]) if len(growth_series.dropna()) else np.nan
    growth_recent = growth_series.tail(3).mean()
    growth_rising = growth_recent > 0.0
    inflation_3m_series = (core / core.shift(3)) ** 4 - 1
    inflation_rising = inflation_3m_series.tail(3).mean() > 0.03
    if growth_rising and not inflation_rising:
        regime = "Goldilocks"
    elif growth_rising and inflation_rising:
        regime = "Reflation"
    elif not growth_rising and inflation_rising:
        regime = "Stagflation"
    else:
        regime = "Disinflation / Recession"
    unrate_delta = float(unrate.iloc[-1] - unrate.tail(13).min()) if len(unrate) else np.nan
    return {
        "regime": regime,
        "inflation_3m": inf_3m,
        "inflation_yoy": inf_yoy,
        "growth_6m": growth_6m,
        "growth_rising": bool(growth_rising),
        "inflation_rising": bool(inflation_rising),
        "unrate_delta": unrate_delta,
    }


def score_etfs(prices: pd.DataFrame, fred: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    market = build_market_series(fred)
    regime = regime_snapshot(fred)
    shy_6 = _return_at(prices, "SHY", 6)
    shy_12 = _return_at(prices, "SHY", 12)
    slope = _latest(market["T10Y2Y"])
    real_yield = _latest(market["DFII10"])
    ig_oas = _latest(market["BAMLC0A0CM"])
    hy_oas = _latest(market["BAMLH0A0HYM2"])
    hy_3m = hy_oas - _asof(market["BAMLH0A0HYM2"], market.index.max() - pd.DateOffset(months=3))
    vix_pct = _percentile(market["VIXCLS"], 3)
    risk_pillars = {
        "Credit": hy_3m >= RISK_CONFIG.hy_spread_3m_warning,
        "Equity Vol": vix_pct >= RISK_CONFIG.vix_percentile_warning,
        "Labor": regime["unrate_delta"] >= RISK_CONFIG.unemployment_gap_warning,
    }
    risk_votes = sum(bool(vote) for vote in risk_pillars.values())
    risk_off = risk_votes >= RISK_CONFIG.entry_votes

    daily_rets = prices.pct_change(fill_method=None)
    vols = daily_rets.tail(756).std() * np.sqrt(252)
    rows = []
    for ticker in ETF_TICKERS:
        if ticker not in prices:
            continue
        label, role, yield_key, duration = ETF_META[ticker]
        r6, r12 = _return_at(prices, ticker, 6), _return_at(prices, ticker, 12)
        ex6, ex12 = r6 - shy_6, r12 - shy_12
        momentum = 1 if ex6 > 0 and ex12 > 0 else (-1 if ex6 < 0 and ex12 < 0 else 0)
        yld_series = market[yield_key].dropna()
        yld = _latest(yld_series)
        shy_yield = _latest(market["DGS2"])
        vol = float(vols.get(ticker, np.nan))
        carry_edge = yld - shy_yield
        carry = 1 if carry_edge >= 0.75 and momentum >= 0 else (-1 if carry_edge < 0.25 else 0)
        ypct = _percentile(yld_series, 5)
        inflation_ok = not regime["inflation_rising"]
        value = 1 if ypct >= 0.80 and momentum >= 0 and inflation_ok else (-1 if ypct <= 0.20 else 0)

        real = 0
        if ticker == "TIP":
            real = 1 if real_yield >= 2.0 and momentum >= 0 else (-1 if real_yield < 0 else 0)

        curve = 0
        if slope > 0.75 and ticker == "IEF":
            curve = 1
        if slope < -0.25 and inflation_ok and regime["unrate_delta"] >= 0.3 and ticker in {"IEF", "TLT"}:
            curve = 1 if momentum >= 0 else 0
        if slope < -0.25 and ticker == "SHY":
            curve = -1

        credit = 0
        if ticker == "LQD":
            credit = -1 if ig_oas < 0.90 else (1 if ig_oas > 1.50 and momentum >= 0 else 0)
        if ticker == "HYG":
            credit = -1 if hy_oas < 3.00 or hy_3m >= 1.00 else (1 if hy_oas > 5.00 and hy_3m <= 0 and momentum > 0 else 0)

        defensive = 0
        if risk_off:
            defensive = 1 if ticker == "SHY" else (-1 if ticker in {"HYG", "LQD"} else 0)
            if ticker in {"IEF", "TLT"} and momentum > 0:
                defensive = 1

        reg_score = 0
        pref = {
            "Goldilocks": {"LQD", "HYG", "MBB", "IEF"},
            "Reflation": {"TIP", "SHY", "HYG"},
            "Stagflation": {"TIP", "SHY"},
            "Disinflation / Recession": {"IEF", "TLT", "SHY"},
        }[regime["regime"]]
        avoid = {
            "Goldilocks": {"TLT"},
            "Reflation": {"TLT", "LQD"},
            "Stagflation": {"TLT", "HYG", "LQD"},
            "Disinflation / Recession": {"HYG"},
        }[regime["regime"]]
        reg_score = 1 if ticker in pref else (-1 if ticker in avoid else 0)

        factors = {"Carry": carry, "Momentum": momentum, "Value": value, "Real Yield": real,
                   "Yield Curve": curve, "Credit Spread": credit, "Defensive": defensive, "Regime": reg_score}
        total = sum(FACTOR_WEIGHTS[k] * v for k, v in factors.items())
        rows.append({"Ticker": ticker, "Bucket": label, "Role": role, "Yield %": yld, "Duration": duration,
                     "6M %": r6 * 100, "12M %": r12 * 100, "Vol %": vol * 100,
                     "Yield pct": ypct * 100, **factors, "Total Score": total})

    scored = pd.DataFrame(rows).set_index("Ticker").sort_values("Total Score", ascending=False)
    context = {**regime, "slope": slope, "real_yield": real_yield, "ig_oas": ig_oas,
               "hy_oas": hy_oas, "hy_3m": hy_3m, "vix_pct": vix_pct,
               "risk_votes": risk_votes, "risk_pillars": risk_pillars, "risk_off": risk_off}
    return scored, context


def recommend_weights(scored: pd.DataFrame, context: dict) -> pd.Series:
    eligible = scored[scored["Total Score"] > 0].copy()
    if eligible.empty:
        return pd.Series({"SHY": 1.0})
    # Remove duplicate implementation vehicles; keep the stronger signal, then lower volatility.
    for group in [{"IEF", "TLT"}]:
        members = [x for x in group if x in eligible.index]
        if len(members) > 1:
            keep = eligible.loc[members].sort_values(["Total Score", "Vol %"], ascending=[False, True]).index[0]
            eligible = eligible.drop([x for x in members if x != keep])
    raw = eligible["Total Score"].clip(lower=0.05) / eligible["Vol %"].clip(lower=1.0)
    weights = raw / raw.sum()
    caps = pd.Series(CAPS).reindex(weights.index)
    weights = weights.clip(upper=caps)
    if context["risk_off"]:
        for t in ["HYG", "LQD"]:
            if t in weights:
                weights[t] = min(weights[t], 0.10 if t != "HYG" else 0.0)
    residual = max(0.0, 1.0 - weights.sum())
    weights["SHY"] = weights.get("SHY", 0.0) + residual
    if weights.sum() > 1:
        weights = weights / weights.sum()
    return weights.sort_values(ascending=False)
