from __future__ import annotations

import numpy as np
import pandas as pd

from data_sources import CASH_PROXY_TICKERS, ETF_TICKERS, STRUCTURED_PROXY_TICKERS
from risk_config import RISK_CONFIG
from signals import ETF_META, FACTOR_WEIGHTS, recommend_weights, score_etfs


STATIC_CORE = {"SHY": 0.05, "IEF": 0.30, "TLT": 0.00, "TIP": 0.10, "LQD": 0.20, "HYG": 0.10, "MBB": 0.25}
MAIN_STRATEGY_NAME = "Risk-First Bond Signal Engine"
DEFAULT_BOND_CORE_WEIGHT = 0.70
DEFAULT_ACTIVE_OVERLAY_WEIGHT = 0.30
NORMAL_CARRY_BASKET = {"JAAA": 0.50, "HYG": 0.20, "BKLN": 0.20, "CMBS": 0.10}
BOND_REPLICA_BASE = {"SHY": 0.05, "IEF": 0.25, "TLT": 0.05, "TIP": 0.05, "LQD": 0.15, "HYG": 0.05, "MBB": 0.40}
BOND_REPLICA_BOUNDS = {
    "SHY": (0.00, 0.15), "IEF": (0.15, 0.35), "TLT": (0.00, 0.10),
    "TIP": (0.00, 0.12), "LQD": (0.08, 0.22), "HYG": (0.00, 0.10),
    "MBB": (0.30, 0.45),
}
HYBRID_RULES = {
    "confirmation_days": 5,
    "min_holding_days": 20,
    "min_weight_gap": 0.10,
    "max_asset_change": 0.15,
    "annual_turnover_cap": 1.00,
    "entry_score": 0.15,
    "exit_score": -0.05,
}


def _first_valid(series: pd.Series) -> pd.Timestamp | None:
    clean = series.dropna()
    return clean.index.min() if len(clean) else None


def performance_metrics(returns: pd.Series, rf: pd.Series | None = None, benchmark: pd.Series | None = None) -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {}
    rf_aligned = pd.Series(0.0, index=r.index) if rf is None else rf.reindex(r.index).ffill().fillna(0.0)
    excess = r - rf_aligned
    nav = (1 + r).cumprod()
    years = len(r) / 12
    cagr = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(12)
    downside = excess[excess < 0].std(ddof=1) * np.sqrt(12)
    drawdown = nav / nav.cummax() - 1
    result = {
        "CAGR": cagr,
        "Ann. Vol": vol,
        "Sharpe": excess.mean() / r.std(ddof=1) * np.sqrt(12) if r.std(ddof=1) else np.nan,
        "Sortino": excess.mean() * 12 / downside if downside else np.nan,
        "Max Drawdown": drawdown.min(),
        "Calmar": cagr / abs(drawdown.min()) if drawdown.min() < 0 else np.nan,
        "Hit Rate": (r > 0).mean(),
        "Excess t-stat": excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess))) if excess.std(ddof=1) else np.nan,
    }
    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        aligned = pd.concat([r, b], axis=1).dropna()
        if len(aligned) > 2:
            active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
            te = active.std(ddof=1) * np.sqrt(12)
            result.update(
                {
                    "Correlation": aligned.iloc[:, 0].corr(aligned.iloc[:, 1]),
                    "Beta": aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var(),
                    "Tracking Error": te,
                    "Information Ratio": active.mean() * 12 / te if te else np.nan,
                }
            )
    return result


def _hysteresis_target(scored: pd.DataFrame, context: dict, current: pd.Series) -> pd.Series:
    """Require a stronger score to enter than to keep an existing ETF."""
    adjusted = scored.copy()
    held = current.reindex(adjusted.index, fill_value=0.0) >= .01
    keep = held & (adjusted["Total Score"] >= HYBRID_RULES["exit_score"])
    enter = (~held) & (adjusted["Total Score"] >= HYBRID_RULES["entry_score"])
    eligible = keep | enter
    adjusted.loc[~eligible, "Total Score"] = -1.0
    adjusted.loc[keep & (adjusted["Total Score"] <= 0), "Total Score"] = .01
    return recommend_weights(adjusted, context).reindex(ETF_TICKERS, fill_value=0.0)


def _bounded_normalize(weights: pd.Series) -> pd.Series:
    lower = pd.Series({ticker: BOND_REPLICA_BOUNDS[ticker][0] for ticker in ETF_TICKERS})
    upper = pd.Series({ticker: BOND_REPLICA_BOUNDS[ticker][1] for ticker in ETF_TICKERS})
    result = weights.reindex(ETF_TICKERS, fill_value=0.0).clip(lower=lower, upper=upper)
    for _ in range(10):
        residual = 1.0 - float(result.sum())
        if abs(residual) < 1e-10:
            break
        room = upper - result if residual > 0 else result - lower
        room = room.clip(lower=0.0)
        if room.sum() <= 0:
            break
        result += np.sign(residual) * room * min(abs(residual) / room.sum(), 1.0)
    return result / result.sum()


def _duration_adjust(weights: pd.Series, target_duration: float) -> pd.Series:
    """Move weight between existing long-only ETFs until the duration target is met."""
    result = _bounded_normalize(weights)
    duration = pd.Series({ticker: ETF_META[ticker][3] for ticker in ETF_TICKERS})
    lower = pd.Series({ticker: BOND_REPLICA_BOUNDS[ticker][0] for ticker in ETF_TICKERS})
    upper = pd.Series({ticker: BOND_REPLICA_BOUNDS[ticker][1] for ticker in ETF_TICKERS})
    for _ in range(30):
        gap = target_duration - float((result * duration).sum())
        if abs(gap) < 1e-6:
            break
        if gap > 0:
            donors = [ticker for ticker in ETF_TICKERS if result[ticker] > lower[ticker] + 1e-9]
            receivers = [ticker for ticker in ETF_TICKERS if result[ticker] < upper[ticker] - 1e-9]
            pair = max(((duration[r] - duration[d], d, r) for d in donors for r in receivers), default=(0, None, None))
        else:
            donors = [ticker for ticker in ETF_TICKERS if result[ticker] > lower[ticker] + 1e-9]
            receivers = [ticker for ticker in ETF_TICKERS if result[ticker] < upper[ticker] - 1e-9]
            pair = max(((duration[d] - duration[r], d, r) for d in donors for r in receivers), default=(0, None, None))
        duration_change, donor, receiver = pair
        if donor is None or duration_change <= 0:
            break
        amount = min(abs(gap) / duration_change, result[donor] - lower[donor], upper[receiver] - result[receiver])
        if amount <= 1e-9:
            break
        result[donor] -= amount
        result[receiver] += amount
    return result


def _bond_replica_target(scored: pd.DataFrame, context: dict) -> tuple[pd.Series, float]:
    """Stable core plus small, economically constrained tactical tilts."""
    target = pd.Series(BOND_REPLICA_BASE).reindex(ETF_TICKERS).astype(float)
    score = scored["Total Score"].reindex(ETF_TICKERS).clip(-1.0, 1.0).fillna(0.0)
    target += 0.025 * (score - score.mean())

    regime = context["regime"]
    duration_targets = {
        "Goldilocks": 6.20, "Reflation": 5.80,
        "Stagflation": 5.60, "Disinflation / Recession": 6.50,
    }
    duration_target = duration_targets[regime]
    if regime == "Goldilocks":
        target[["LQD", "HYG"]] += [0.025, 0.015]
        target["SHY"] -= 0.040
    elif regime == "Reflation":
        target["TIP"] += 0.040
        target[["IEF", "TLT"]] -= [0.025, 0.015]
    elif regime == "Stagflation":
        target["TIP"] += 0.050
        target[["LQD", "HYG"]] -= [0.030, 0.020]
    else:
        target[["IEF", "TLT"]] += [0.035, 0.015]
        target[["LQD", "HYG"]] -= [0.020, 0.030]

    if context["risk_off"]:
        target[["LQD", "HYG"]] -= [0.030, 0.050]
        target[["IEF", "MBB"]] += [0.040, 0.040]
        duration_target = 6.50
    elif context["hy_oas"] >= 5.0 and context["hy_3m"] <= 0:
        target[["LQD", "HYG"]] += [0.020, 0.025]
        target["SHY"] -= 0.045

    return _duration_adjust(target, duration_target), duration_target


def run_bond_replica_backtest(
    monthly: pd.DataFrame,
    prices: pd.DataFrame,
    fred: pd.DataFrame,
    dates: pd.DatetimeIndex,
    transaction_cost_bps: float,
) -> dict:
    """Quarterly core tilts plus a monthly risk transition overlay."""
    returns: dict[pd.Timestamp, float] = {}
    turnover: dict[pd.Timestamp, float] = {}
    weight_rows: list[pd.Series] = []
    duration_rows: dict[pd.Timestamp, float] = {}
    previous = pd.Series(BOND_REPLICA_BASE).reindex(ETF_TICKERS).astype(float)
    prior_risk_off = False
    initialized = False

    for current, following in zip(dates[:-1], dates[1:]):
        scored, context = score_etfs(prices.loc[:current], fred.loc[:current])
        review = (not initialized) or current.month in {3, 6, 9, 12} or bool(context["risk_off"]) != prior_risk_off
        if review:
            desired, _ = _bond_replica_target(scored, context)
            step = 0.75 if bool(context["risk_off"]) != prior_risk_off else 0.50
            proposed = previous + step * (desired - previous)
            max_change = float((proposed - previous).abs().max())
            if max_change > 0.05:
                proposed = previous + (proposed - previous) * (0.05 / max_change)
            weights = _bounded_normalize(proposed)
        else:
            weights = previous.copy()
        next_returns = monthly.loc[following] / monthly.loc[current] - 1
        one_way = float((weights - previous).abs().sum() / 2)
        returns[following] = float((weights * next_returns).sum() - one_way * transaction_cost_bps / 10_000)
        turnover[current] = one_way
        weight_rows.append(weights.rename(current))
        duration_rows[current] = float(sum(weights[ticker] * ETF_META[ticker][3] for ticker in ETF_TICKERS))
        previous = weights
        prior_risk_off = bool(context["risk_off"])
        initialized = True

    return {
        "returns": pd.Series(returns, name="BOND Exposure Replica").sort_index(),
        "weights": pd.DataFrame(weight_rows),
        "turnover": pd.Series(turnover, name="Replica Turnover").sort_index(),
        "duration": pd.Series(duration_rows, name="Replica Effective Duration").sort_index(),
    }


def _asof_value(series: pd.Series, date: pd.Timestamp, default: float = np.nan) -> float:
    clean = series.loc[:date].dropna()
    return float(clean.iloc[-1]) if len(clean) else default


def _period_return(series: pd.Series, date: pd.Timestamp, days: int) -> float:
    clean = series.loc[:date].dropna()
    if len(clean) <= days:
        return 0.0
    return float(clean.iloc[-1] / clean.iloc[-days - 1] - 1)


def _lookback_percentile(series: pd.Series, date: pd.Timestamp, years: int = 5) -> float:
    window = series.loc[:date].dropna()
    window = window.loc[window.index >= date - pd.DateOffset(years=years)]
    if len(window) < 2:
        return 0.5
    return float((window <= window.iloc[-1]).mean())


def _fred_change(fred: pd.DataFrame, column: str, date: pd.Timestamp, days: int = 63) -> float:
    series = fred[column].ffill().loc[:date].dropna()
    if len(series) <= days:
        return 0.0
    return float(series.iloc[-1] - series.iloc[-days - 1])


def _clip_score(value: float, low: float = -2.0, high: float = 2.0) -> float:
    return float(np.clip(value, low, high))


def _safe_normalize(weights: pd.Series) -> pd.Series:
    result = weights.reindex(ETF_TICKERS, fill_value=0.0).clip(lower=0.0)
    total = float(result.sum())
    if total <= 0:
        result["SHY"] = 1.0
        return result
    return result / total


def _strategy_metadata(
    returns: pd.Series,
    benchmark: str,
    parameter_version: str,
    signal_lag: str,
    uses_bond_core: bool = False,
    executed: bool | None = None,
) -> dict:
    clean = returns.dropna() if returns is not None else pd.Series(dtype=float)
    did_execute = bool(len(clean)) if executed is None else bool(executed)
    return {
        "strategy_executed": did_execute,
        "data_start": clean.index.min() if len(clean) else None,
        "data_end": clean.index.max() if len(clean) else None,
        "number_of_observations": int(len(clean)),
        "parameter_version": parameter_version,
        "signal_lag": signal_lag,
        "benchmark": benchmark,
        "uses_bond_core": uses_bond_core,
    }


def _fund_overlay(weights: pd.Series, overlay_amount: float, donors: list[str]) -> pd.Series:
    result = weights.copy()
    remaining = float(max(0.0, overlay_amount))
    for ticker in donors:
        if remaining <= 1e-10:
            break
        if ticker not in result:
            continue
        take = min(float(result[ticker]), remaining)
        result[ticker] -= take
        remaining -= take
    return result


def _available_cash_proxies(prices: pd.DataFrame, date: pd.Timestamp | None = None, min_obs: int = 20) -> list[str]:
    available = []
    for ticker in CASH_PROXY_TICKERS:
        if ticker not in prices:
            continue
        series = prices[ticker].dropna() if date is None else prices[ticker].loc[:date].dropna()
        if len(series) >= min_obs:
            available.append(ticker)
    return available


def _defensive_cash_asset(prices: pd.DataFrame, date: pd.Timestamp, universe: list[str] | pd.Index | None = None) -> str:
    allowed = set(universe) if universe is not None else set(prices.columns)
    candidates = [ticker for ticker in _available_cash_proxies(prices, date) if ticker in allowed]
    if "SHY" in allowed:
        candidates.append("SHY")
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return "SHY"
    scores = {}
    for ticker in candidates:
        score = _period_return(prices[ticker], date, 21)
        dd_window = prices[ticker].loc[:date].dropna().tail(63)
        if len(dd_window) >= 2:
            nav = dd_window / dd_window.iloc[0]
            score += float((nav / nav.cummax() - 1).min())
        scores[ticker] = score
    return max(scores, key=scores.get)


def _alpha_replica_target(prices: pd.DataFrame, fred: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.Series, dict]:
    """Alpha-source scorecard: MBS RV, credit cushion, duration/MOVE, curve and real yield."""
    scored, context = score_etfs(prices.loc[:date], fred.loc[:date])
    mortgage_spread = (fred["MORTGAGE30US"].ffill() - fred["DGS10"].ffill()).loc[:date].dropna()
    window = mortgage_spread.loc[mortgage_spread.index >= date - pd.DateOffset(years=5)]
    mbs_percentile = _lookback_percentile(mortgage_spread, date, 5)
    mbs_relative_3m = _period_return(prices["MBB"], date, 63) - _period_return(prices["IEF"], date, 63)
    move = _asof_value(prices["MOVE"], date) if "MOVE" in prices else np.nan
    move_pct = _lookback_percentile(prices["MOVE"], date, 3) if "MOVE" in prices else np.nan
    move_3m = _period_return(prices["MOVE"], date, 63) if "MOVE" in prices else 0.0
    move_extreme = bool(np.isfinite(move) and move >= 110 and np.isfinite(move_pct) and move_pct >= .85)

    mbs_score = 0.0
    mbs_score += 1.0 if mbs_percentile >= 0.75 else (-1.0 if mbs_percentile <= 0.25 else 0.0)
    mbs_score += 0.5 if mbs_relative_3m >= 0.00 else (-0.5 if mbs_relative_3m <= -0.015 else 0.0)
    mbs_score += 0.5 if move_3m <= 0.00 else (-0.5 if move_extreme else 0.0)
    mbs_score = _clip_score(mbs_score)
    if mbs_score >= 1.0:
        mbs_rule = "Cheap MBS + stable volatility"
    elif mbs_score <= -1.0:
        mbs_rule = "Rich/weak MBS or convexity stress"
    else:
        mbs_rule = "Neutral MBS RV"

    five_ten = _asof_value(fred["DGS10"] - fred["DGS5"], date)
    ten_thirty = _asof_value(fred["DGS30"] - fred["DGS10"], date)
    ten_year_3m_change = _fred_change(fred, "DGS10", date, 63)
    tlt_momentum_6m = _period_return(prices["TLT"], date, 126)
    duration_score = 0.0
    duration_score += 1.0 if ten_year_3m_change <= -0.25 else (-1.0 if ten_year_3m_change >= 0.25 else 0.0)
    duration_score += 0.5 if tlt_momentum_6m > 0 else (-0.5 if tlt_momentum_6m < -0.04 else 0.0)
    duration_score += 0.5 if context["regime"] == "Disinflation / Recession" else (-0.5 if context["inflation_rising"] else 0.0)
    duration_score += -0.5 if move_extreme and ten_year_3m_change > 0 else 0.0
    duration_score = _clip_score(duration_score)
    if duration_score >= 1.0:
        krd_rule = "Extend duration"
    elif duration_score <= -1.0:
        krd_rule = "Cut duration"
    else:
        krd_rule = "Neutral duration"

    slope = _asof_value(fred["T10Y2Y"], date)
    slope_3m_change = _fred_change(fred, "T10Y2Y", date, 63)
    curve_score = 0.0
    curve_score += 0.5 if five_ten >= 0.20 or ten_thirty >= 0.20 else 0.0
    curve_score += 0.5 if slope_3m_change >= 0.25 else (-0.5 if slope <= -0.75 and ten_year_3m_change > 0 else 0.0)
    curve_score = _clip_score(curve_score)
    curve_rule = "Roll-down/steepening support" if curve_score > 0 else ("Deep inversion caution" if curve_score < 0 else "Neutral curve")

    real_yield = _asof_value(fred["DFII10"], date)
    real_yield_pct = _lookback_percentile(fred["DFII10"].ffill(), date, 5)
    tip_relative_3m = _period_return(prices["TIP"], date, 63) - _period_return(prices["IEF"], date, 63)
    real_yield_score = 0.0
    real_yield_score += 1.0 if real_yield_pct >= 0.70 and real_yield >= 1.0 else (-0.75 if real_yield_pct <= 0.25 else 0.0)
    real_yield_score += 0.5 if tip_relative_3m >= 0 or context["inflation_rising"] else (-0.5 if tip_relative_3m <= -0.02 else 0.0)
    real_yield_score = _clip_score(real_yield_score)
    real_rule = "Real-yield carry/TIPS support" if real_yield_score > 0 else ("Low real-yield value" if real_yield_score < 0 else "Neutral real yield")

    hy_oas_pct = _lookback_percentile(fred["BAMLH0A0HYM2"].ffill(), date, 5)
    ig_oas_pct = _lookback_percentile(fred["BAMLC0A0CM"].ffill(), date, 5)
    hyg_relative_3m = _period_return(prices["HYG"], date, 63) - _period_return(prices["SHY"], date, 63)
    lqd_relative_3m = _period_return(prices["LQD"], date, 63) - _period_return(prices["IEF"], date, 63)
    credit_stable = context["hy_3m"] <= 0.25 and hyg_relative_3m > -0.02
    credit_score = 0.0
    credit_score += 1.0 if hy_oas_pct >= 0.65 and credit_stable else (-1.0 if context["hy_3m"] >= 0.75 or hyg_relative_3m <= -0.04 else 0.0)
    credit_score += 0.5 if ig_oas_pct >= 0.60 and lqd_relative_3m > -0.02 else (-0.5 if context["ig_oas"] < 0.90 else 0.0)
    credit_score += -0.5 if context["unrate_delta"] >= 0.50 else 0.0
    credit_score = _clip_score(credit_score)
    if credit_score >= 1.0:
        credit_rule = "Spread cushion + price confirmation"
    elif credit_score <= -1.0:
        credit_rule = "Credit drawdown risk"
    else:
        credit_rule = "Neutral credit carry"

    source_scores = {
        "MBS RV": mbs_score,
        "Credit Cushion": credit_score,
        "Duration/MOVE": duration_score,
        "Curve Roll-down": curve_score,
        "Real Yield": real_yield_score,
    }

    weights = pd.Series({"SHY": 0.08, "IEF": 0.24, "TLT": 0.02, "TIP": 0.06, "LQD": 0.16, "HYG": 0.04, "MBB": 0.40})

    weights["MBB"] += 0.045 * mbs_score
    weights["TIP"] += 0.035 * real_yield_score
    if duration_score >= 0:
        weights["TLT"] += 0.035 * duration_score
        weights["IEF"] += 0.020 * duration_score
        weights["SHY"] -= 0.055 * duration_score
    else:
        weights["SHY"] += 0.055 * abs(duration_score)
        weights["TLT"] -= 0.035 * abs(duration_score)
        weights["IEF"] -= 0.020 * abs(duration_score)
    if curve_score >= 0:
        weights["IEF"] += 0.030 * curve_score
        weights["SHY"] -= 0.020 * curve_score
    else:
        weights["SHY"] += 0.030 * abs(curve_score)
        weights["TLT"] -= 0.020 * abs(curve_score)
    if credit_score >= 0:
        weights["LQD"] += 0.025 * min(credit_score, 1.0)
        weights["HYG"] += 0.030 * max(credit_score - 0.25, 0.0)
        weights["SHY"] -= 0.030 * credit_score
    else:
        weights["LQD"] -= 0.035 * abs(credit_score)
        weights["HYG"] -= 0.040 * abs(credit_score)
        weights["SHY"] += 0.075 * abs(credit_score)

    if move_extreme and duration_score < 0:
        weights["TLT"] = min(weights["TLT"], 0.04)
    if context["hy_3m"] >= 1.00 or hyg_relative_3m <= -0.04:
        weights["HYG"] = 0.0

    caps = pd.Series({"SHY": 1.00, "IEF": 0.45, "TLT": 0.20, "TIP": 0.18, "LQD": 0.28, "HYG": 0.15, "MBB": 0.52})
    floors = pd.Series({"SHY": 0.00, "IEF": 0.05, "TLT": 0.00, "TIP": 0.00, "LQD": 0.00, "HYG": 0.00, "MBB": 0.20})
    weights = weights.clip(lower=floors, upper=caps)
    if weights.sum() > 1:
        weights = weights / weights.sum()
    elif weights.sum() < 1:
        weights["SHY"] += 1.0 - float(weights.sum())
    weights = _safe_normalize(weights)

    dominant = max(source_scores, key=lambda key: abs(source_scores[key]))
    pimco_anchor_gap = float((weights - pd.Series(BOND_REPLICA_BASE).reindex(ETF_TICKERS)).abs().sum() / 2)

    diagnostics = {
        "MBS Spread": float(window.iloc[-1]) if len(window) else np.nan,
        "MBS Spread Percentile": mbs_percentile,
        "MBB vs IEF 3M": mbs_relative_3m,
        "MBS RV Score": mbs_score,
        "MBS Rule": mbs_rule,
        "Duration/MOVE Score": duration_score,
        "10Y 3M Change": ten_year_3m_change,
        "KRD Rule": krd_rule,
        "Curve Score": curve_score,
        "Curve Rule": curve_rule,
        "Real Yield Score": real_yield_score,
        "Real Yield Rule": real_rule,
        "Credit Cushion Score": credit_score,
        "Credit Rule": credit_rule,
        "Dominant Alpha": dominant,
        "PIMCO Anchor Gap": pimco_anchor_gap,
        "KRD ETF": "TLT" if duration_score >= 1.0 else ("SHY" if duration_score <= -1.0 else "IEF"),
        "Credit ETF": "HYG" if credit_score >= 1.25 else ("SHY" if credit_score <= -1.0 else "LQD"),
        "MOVE": move,
        "MOVE Percentile": move_pct,
        "Regime": context["regime"],
    }
    return weights, diagnostics


def _structured_proxy_target(prices: pd.DataFrame, fred: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.Series, dict]:
    base_weights, diagnostics = _alpha_replica_target(prices, fred, date)
    available = [ticker for ticker in STRUCTURED_PROXY_TICKERS if ticker in prices and prices[ticker].loc[:date].dropna().shape[0] > 63]
    universe = ETF_TICKERS + available
    weights = base_weights.reindex(universe, fill_value=0.0)
    if not available:
        diagnostics = {**diagnostics, "Structured Proxy Rule": "No ETF data", "Structured Proxy Weight": 0.0}
        return weights, diagnostics

    mbs_score = float(diagnostics.get("MBS RV Score", 0.0))
    credit_score = float(diagnostics.get("Credit Cushion Score", 0.0))
    move_pct = float(diagnostics.get("MOVE Percentile", np.nan))
    structured_target = 0.0
    structured_rule = "Off"
    if mbs_score > 0 and credit_score > -1.0:
        structured_target = 0.06 + 0.035 * min(mbs_score, 2.0)
        structured_target += 0.020 * max(credit_score, 0.0)
        structured_rule = "MBS gap proxy"
    if credit_score <= -1.0 or (np.isfinite(move_pct) and move_pct >= 0.90):
        structured_target = min(structured_target, 0.04)
        structured_rule = "Stress cap"
    structured_target = float(np.clip(structured_target, 0.0, 0.20))

    weights = _fund_overlay(weights, structured_target, ["SHY", "IEF", "LQD", "MBB", "TIP"])
    sleeve = pd.Series(0.0, index=available)
    if structured_target > 0:
        if "JAAA" in available:
            sleeve["JAAA"] = structured_target * 0.60
        if "CMBS" in available:
            sleeve["CMBS"] = structured_target * 0.25
        if "BKLN" in available and credit_score >= 0.0:
            sleeve["BKLN"] = structured_target * 0.15
        unassigned = structured_target - float(sleeve.sum())
        if unassigned > 0:
            if "JAAA" in sleeve:
                sleeve["JAAA"] += unassigned
            else:
                sleeve.iloc[0] += unassigned
    weights.loc[sleeve.index] = sleeve
    weights = weights.clip(lower=0.0)
    weights = weights / weights.sum() if weights.sum() else weights
    diagnostics = {
        **diagnostics,
        "Structured Proxy Rule": structured_rule,
        "Structured Proxy Weight": float(sleeve.sum()),
        "JAAA Weight": float(weights.get("JAAA", 0.0)),
        "BKLN Weight": float(weights.get("BKLN", 0.0)),
        "CMBS Weight": float(weights.get("CMBS", 0.0)),
        "Structured ETFs": ", ".join(available),
    }
    return weights, diagnostics


STRUCTURED_V2_MIXES = {
    "JAAA Defensive": {"JAAA": 0.85, "CMBS": 0.15, "BKLN": 0.00},
    "Balanced": {"JAAA": 0.70, "CMBS": 0.20, "BKLN": 0.10},
    "Carry Tilt": {"JAAA": 0.55, "CMBS": 0.25, "BKLN": 0.20},
}

STRUCTURED_V2_GATES = {
    "Light Gate": {
        "nfci_cut": 0.55, "hy3m_cut": 1.10, "move_pct_cut": 0.95,
        "jaaa_1m_cut": -0.009, "bkln_1m_cut": -0.020, "cmbs_3m_cut": -0.030,
        "stress_scale": 0.75,
    },
    "Medium Gate": {
        "nfci_cut": 0.35, "hy3m_cut": 0.90, "move_pct_cut": 0.92,
        "jaaa_1m_cut": -0.006, "bkln_1m_cut": -0.015, "cmbs_3m_cut": -0.020,
        "stress_scale": 0.50,
    },
    "Strong Gate": {
        "nfci_cut": 0.20, "hy3m_cut": 0.70, "move_pct_cut": 0.88,
        "jaaa_1m_cut": -0.003, "bkln_1m_cut": -0.010, "cmbs_3m_cut": -0.015,
        "stress_scale": 0.25,
    },
}


def structured_v2_configs() -> list[dict]:
    configs: list[dict] = []
    for cap in [0.10, 0.15, 0.20]:
        for mix_name, mix in STRUCTURED_V2_MIXES.items():
            for gate_name, gate in STRUCTURED_V2_GATES.items():
                configs.append({
                    "name": f"v2 {int(cap * 100)}% · {mix_name} · {gate_name}",
                    "cap": cap,
                    "mix_name": mix_name,
                    "mix": mix,
                    "gate_name": gate_name,
                    "gate": gate,
                })
    return configs


def _structured_proxy_target_v2(prices: pd.DataFrame, fred: pd.DataFrame, date: pd.Timestamp, config: dict) -> tuple[pd.Series, dict]:
    base_weights, diagnostics = _alpha_replica_target(prices, fred, date)
    available = [ticker for ticker in STRUCTURED_PROXY_TICKERS if ticker in prices and prices[ticker].loc[:date].dropna().shape[0] > 63]
    universe = ETF_TICKERS + available
    weights = base_weights.reindex(universe, fill_value=0.0)
    if not available:
        diagnostics = {**diagnostics, "Structured v2 Rule": "No ETF data", "Structured v2 Weight": 0.0}
        return weights, diagnostics

    _, context = score_etfs(prices.loc[:date], fred.loc[:date])
    gate = config["gate"]
    cap = float(config["cap"])
    mbs_score = float(diagnostics.get("MBS RV Score", 0.0))
    credit_score = float(diagnostics.get("Credit Cushion Score", 0.0))
    move_pct = float(diagnostics.get("MOVE Percentile", np.nan))
    nfci = _asof_value(fred["NFCI"], date, 0.0)

    activation = 0.0
    if mbs_score > 0 and credit_score > -1.0:
        activation = min(1.0, 0.35 + 0.25 * max(mbs_score, 0.0) + 0.15 * max(credit_score, 0.0))
    structured_target = cap * activation

    stress_flags = []
    if nfci >= gate["nfci_cut"]:
        stress_flags.append("NFCI")
    if context["hy_3m"] >= gate["hy3m_cut"]:
        stress_flags.append("HY spread")
    if np.isfinite(move_pct) and move_pct >= gate["move_pct_cut"]:
        stress_flags.append("MOVE")
    if stress_flags:
        structured_target *= gate["stress_scale"]

    sleeve = pd.Series(0.0, index=available)
    raw_mix = pd.Series({ticker: config["mix"].get(ticker, 0.0) for ticker in available}, dtype=float)
    if raw_mix.sum() <= 0:
        raw_mix[:] = 1 / len(raw_mix)

    jaaa_1m = _period_return(prices["JAAA"], date, 21) - _period_return(prices["SHY"], date, 21) if "JAAA" in available else 0.0
    bkln_1m = _period_return(prices["BKLN"], date, 21) - _period_return(prices["SHY"], date, 21) if "BKLN" in available else 0.0
    cmbs_3m = _period_return(prices["CMBS"], date, 63) - _period_return(prices["MBB"], date, 63) if "CMBS" in available else 0.0

    if "JAAA" in raw_mix and jaaa_1m <= gate["jaaa_1m_cut"]:
        raw_mix["JAAA"] = 0.0
        stress_flags.append("JAAA price")
    if "BKLN" in raw_mix and (bkln_1m <= gate["bkln_1m_cut"] or context["hy_3m"] >= gate["hy3m_cut"] or credit_score < 0):
        raw_mix["BKLN"] = 0.0
        stress_flags.append("BKLN credit")
    if "CMBS" in raw_mix and cmbs_3m <= gate["cmbs_3m_cut"]:
        raw_mix["CMBS"] = 0.0
        stress_flags.append("CMBS momentum")

    if raw_mix.sum() <= 0:
        structured_target = 0.0
    else:
        raw_mix = raw_mix / raw_mix.sum()
        sleeve.loc[raw_mix.index] = structured_target * raw_mix

    weights = _fund_overlay(weights, float(sleeve.sum()), ["SHY", "IEF", "LQD", "MBB", "TIP"])
    weights.loc[sleeve.index] = sleeve
    weights = weights.clip(lower=0.0)
    weights = weights / weights.sum() if weights.sum() else weights
    rule = "Off" if sleeve.sum() <= 0 else ("Stress-scaled" if stress_flags else "Active")
    diagnostics = {
        **diagnostics,
        "Structured v2 Rule": rule,
        "Structured v2 Weight": float(sleeve.sum()),
        "Structured v2 Config": config["name"],
        "Structured v2 Stress": ", ".join(dict.fromkeys(stress_flags)) or "None",
        "JAAA Weight": float(weights.get("JAAA", 0.0)),
        "BKLN Weight": float(weights.get("BKLN", 0.0)),
        "CMBS Weight": float(weights.get("CMBS", 0.0)),
        "JAAA vs SHY 1M": jaaa_1m,
        "BKLN vs SHY 1M": bkln_1m,
        "CMBS vs MBB 3M": cmbs_3m,
        "Structured ETFs": ", ".join(available),
    }
    return weights, diagnostics


def _dynamic_bond_core_weight(
    context: dict,
    diagnostics: dict,
    risk_flags: list[str],
    overlay_rule: str,
    overlay_weights: pd.Series,
    move_pct: float,
    nfci: float,
) -> tuple[float, str]:
    """Dynamic 50~90% BOND core weight for the core/satellite strategy."""
    credit_score = float(diagnostics.get("Credit Cushion Score", 0.0))
    mbs_score = float(diagnostics.get("MBS RV Score", 0.0))
    duration_score = float(diagnostics.get("Duration/MOVE Score", 0.0))
    ten_year_3m = float(diagnostics.get("10Y 3M Change", np.nan))
    structured = float(sum(overlay_weights.get(ticker, 0.0) for ticker in STRUCTURED_PROXY_TICKERS))
    carry = float(overlay_weights.get("HYG", 0.0) + overlay_weights.get("BKLN", 0.0) + overlay_weights.get("JAAA", 0.0))

    if len(risk_flags) >= 3 or context["risk_off"] or context["hy_3m"] >= 1.00 or nfci >= 0.50:
        return 1.00, "Risk-off: defensive core 100 / Overlay 0"
    if len(risk_flags) >= 2 or context["hy_3m"] >= 0.70 or (np.isfinite(move_pct) and move_pct >= 0.88) or nfci >= 0.20:
        return 0.90, "Risk warning: defensive core 90 / Overlay 10"
    if context["hy_3m"] >= 0.35 or (np.isfinite(move_pct) and move_pct >= 0.82) or nfci >= 0.10:
        return 0.80, "Early warning: defensive core 80 / Overlay 20"
    if (
        overlay_rule in {"Credit carry / structured carry", "MBS / structured relative value", "Carry with curve support", "Balanced carry"}
        and structured >= 0.50
        and carry >= 0.50
        and credit_score >= -0.5
        and context["hy_3m"] <= 0.35
        and (not np.isfinite(move_pct) or move_pct <= 0.85)
        and nfci < 0.20
    ):
        if credit_score >= 0.5 or mbs_score >= 1.0:
            return 0.50, "Risk-on strong carry/structured: BOND 50 / Overlay 50"
        return 0.60, "Risk-on carry/structured: BOND 60 / Overlay 40"
    return 0.70, "Neutral: BOND 70 / Overlay 30"


def _aggressive_ra_overlay_target(prices: pd.DataFrame, fred: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.Series, dict]:
    """Concentrated active sleeve designed to sit on top of a dynamic 50~90% BOND core."""
    _, diagnostics = _alpha_replica_target(prices, fred, date)
    _, context = score_etfs(prices.loc[:date], fred.loc[:date])
    available = [ticker for ticker in STRUCTURED_PROXY_TICKERS if ticker in prices and prices[ticker].loc[:date].dropna().shape[0] > 63]
    cash_available = _available_cash_proxies(prices, date)
    universe = ETF_TICKERS + available + cash_available
    weights = pd.Series(0.0, index=universe)
    cash_asset = _defensive_cash_asset(prices, date, universe)

    mbs_score = float(diagnostics.get("MBS RV Score", 0.0))
    credit_score = float(diagnostics.get("Credit Cushion Score", 0.0))
    duration_score = float(diagnostics.get("Duration/MOVE Score", 0.0))
    curve_score = float(diagnostics.get("Curve Score", 0.0))
    real_yield_score = float(diagnostics.get("Real Yield Score", 0.0))
    move_pct = float(diagnostics.get("MOVE Percentile", np.nan))
    nfci = _asof_value(fred["NFCI"], date, 0.0)
    hyg_1m = _period_return(prices["HYG"], date, 21) - _period_return(prices["SHY"], date, 21)
    jaaa_1m = _period_return(prices["JAAA"], date, 21) - _period_return(prices["SHY"], date, 21) if "JAAA" in available else 0.0
    bkln_1m = _period_return(prices["BKLN"], date, 21) - _period_return(prices["SHY"], date, 21) if "BKLN" in available else 0.0
    cmbs_3m = _period_return(prices["CMBS"], date, 63) - _period_return(prices["MBB"], date, 63) if "CMBS" in available else 0.0

    risk_flags = []
    if context["risk_off"]:
        risk_flags.append("regime risk-off")
    if nfci >= 0.20:
        risk_flags.append("NFCI")
    if context["hy_3m"] >= 0.70:
        risk_flags.append("HY spread")
    if np.isfinite(move_pct) and move_pct >= 0.88:
        risk_flags.append("MOVE")
    if hyg_1m <= -0.025:
        risk_flags.append("credit price")

    if len(risk_flags) >= 3:
        weights[cash_asset] = 0.75
        weights["IEF"] = 0.25
        overlay_rule = "Defensive cash/short-duration"
    elif credit_score >= -0.5 and context["hy_3m"] <= 0.35 and (not np.isfinite(move_pct) or move_pct <= 0.85):
        weights["HYG"] = 0.25
        if "BKLN" in available:
            weights["BKLN"] = 0.25
        else:
            weights["HYG"] += 0.15
            weights["LQD"] += 0.10
        if "JAAA" in available:
            weights["JAAA"] = 0.35
        else:
            weights["LQD"] += 0.35
        if "CMBS" in available:
            weights["CMBS"] = 0.15
        else:
            weights["MBB"] += 0.15
        overlay_rule = "Credit carry / structured carry"
    elif mbs_score >= 1.0 and (not np.isfinite(move_pct) or move_pct <= 0.80):
        weights["MBB"] = 0.25
        if "CMBS" in available:
            weights["CMBS"] = 0.25
        else:
            weights["MBB"] += 0.15
            weights["IEF"] += 0.10
        if "JAAA" in available:
            weights["JAAA"] = 0.35
        else:
            weights["LQD"] += 0.35
        if credit_score >= 0 and "BKLN" in available:
            weights["BKLN"] = 0.15
        else:
            weights[cash_asset] += 0.15
        overlay_rule = "MBS / structured relative value"
    elif duration_score >= 1.5 and context["regime"] == "Disinflation / Recession" and context["hy_3m"] < 0.70:
        weights["TLT"] = 0.35
        weights["IEF"] = 0.40
        weights["MBB"] = 0.10
        weights["SHY"] = 0.15
        overlay_rule = "Duration convexity"
    elif real_yield_score >= 1.0 and credit_score < -0.5:
        weights["TIP"] = 0.35
        weights["IEF"] = 0.25
        if "JAAA" in available:
            weights["JAAA"] = 0.25
        else:
            weights[cash_asset] = 0.25
        weights[cash_asset] += 0.15
        overlay_rule = "Defensive real-yield value"
    elif curve_score > 0 or duration_score > 0:
        if "JAAA" in available:
            weights["JAAA"] = 0.45
        else:
            weights["LQD"] = 0.35
        if "CMBS" in available:
            weights["CMBS"] = 0.20
        else:
            weights["MBB"] = 0.20
        weights["BKLN" if "BKLN" in available else "HYG"] += 0.20
        weights["IEF"] = 0.15
        overlay_rule = "Carry with curve support"
    else:
        if "JAAA" in available:
            weights["JAAA"] = 0.45
        else:
            weights["LQD"] = 0.35
        if "BKLN" in available:
            weights["BKLN"] = 0.25
        else:
            weights["HYG"] = 0.20
        if "CMBS" in available:
            weights["CMBS"] = 0.15
        else:
            weights["MBB"] = 0.15
        weights[cash_asset] = 0.15
        overlay_rule = "Balanced carry"

    if context["hy_3m"] >= 1.00 or hyg_1m <= -0.03:
        credit_cut = float(weights.get("HYG", 0.0) + weights.get("BKLN", 0.0))
        weights["HYG"] = 0.0
        if "BKLN" in weights:
            weights["BKLN"] = 0.0
        weights[cash_asset] += credit_cut
        if credit_cut > 0:
            risk_flags.append("credit sleeve cut")
    if "JAAA" in weights and jaaa_1m <= -0.003:
        weights[cash_asset] += weights["JAAA"]
        weights["JAAA"] = 0.0
        risk_flags.append("JAAA price")
    if "CMBS" in weights and cmbs_3m <= -0.015:
        weights["MBB"] += weights["CMBS"]
        weights["CMBS"] = 0.0
        risk_flags.append("CMBS momentum")
    if float(weights.sum()) <= 0:
        weights[cash_asset] = 1.0
    weights = weights.clip(lower=0.0)
    weights = weights / weights.sum()
    bond_core_weight, core_rule = _dynamic_bond_core_weight(
        context, diagnostics, risk_flags, overlay_rule, weights, move_pct, nfci
    )
    active_overlay_weight = 1.0 - bond_core_weight

    diagnostics = {
        **diagnostics,
        "Overlay Rule": overlay_rule,
        "Overlay Risk": ", ".join(dict.fromkeys(risk_flags)) or "None",
        "Core Allocation Rule": core_rule,
        "BOND Core Weight": bond_core_weight,
        "Active Overlay Weight": active_overlay_weight,
        "Overlay HYG + BKLN": float(weights.get("HYG", 0.0) + weights.get("BKLN", 0.0)),
        "Overlay Structured": float(sum(weights.get(ticker, 0.0) for ticker in STRUCTURED_PROXY_TICKERS)),
        "Overlay Duration Tilt": float(weights.get("IEF", 0.0) + weights.get("TLT", 0.0)),
        "Cash Defense Asset": cash_asset,
        "JAAA vs SHY 1M": jaaa_1m,
        "BKLN vs SHY 1M": bkln_1m,
        "CMBS vs MBB 3M": cmbs_3m,
        "Structured ETFs": ", ".join(available),
    }
    return weights, diagnostics


def _severe_risk_snapshot(prices: pd.DataFrame, fred: pd.DataFrame, date: pd.Timestamp, vol_signal: str = "VIX or MOVE") -> tuple[bool, dict]:
    _, context = score_etfs(prices.loc[:date], fred.loc[:date])
    hyg_relative_1m = _period_return(prices["HYG"], date, 21) - _period_return(prices["SHY"], date, 21)
    ten_year_3m = _fred_change(fred, "DGS10", date, 63)
    nfci = _asof_value(fred["NFCI"], date, 0.0)
    move = _asof_value(prices["MOVE"], date) if "MOVE" in prices else np.nan
    move_history = prices["MOVE"].loc[:date].dropna() if "MOVE" in prices else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    if len(move_history) and isinstance(move_history.index, pd.DatetimeIndex):
        move_history = move_history.loc[move_history.index >= date - pd.DateOffset(years=3)]
    move_pct = float((move_history <= move_history.iloc[-1]).mean()) if len(move_history) else np.nan
    move_extreme = bool(
        np.isfinite(move)
        and move >= RISK_CONFIG.move_level_crisis
        and move_pct >= RISK_CONFIG.move_percentile_warning
    )
    vix_extreme = context["vix_pct"] >= RISK_CONFIG.vix_percentile_crisis
    if vol_signal == "MOVE" and len(move_history):
        volatility_extreme = move_extreme
    elif vol_signal == "VIX":
        volatility_extreme = vix_extreme
    else:
        volatility_extreme = vix_extreme or move_extreme
    pillars = {
        "Credit": context["hy_3m"] >= RISK_CONFIG.hy_spread_3m_crisis or hyg_relative_1m <= RISK_CONFIG.hyg_relative_1m_crisis,
        "Rates": move_extreme or (volatility_extreme and ten_year_3m >= RISK_CONFIG.ten_year_3m_rate_shock),
        "Liquidity": nfci >= RISK_CONFIG.nfci_crisis,
        "Labor": context["unrate_delta"] >= RISK_CONFIG.unemployment_gap_crisis,
        "Equity Vol": vix_extreme,
    }
    warning = (
        (np.isfinite(move_pct) and move_pct >= RISK_CONFIG.move_percentile_crisis and ten_year_3m >= 0.15)
        or context["hy_3m"] >= 0.90
        or nfci >= 0.35
    )
    vote_count = int(sum(pillars.values()))
    severe = vote_count >= RISK_CONFIG.entry_votes or warning or (context["hy_3m"] >= 1.25 and volatility_extreme)
    return bool(severe), {
        "Risk Votes": vote_count, "Risk Pillars": ", ".join(name for name, active in pillars.items() if active) or "None",
        "HY OAS 3M": context["hy_3m"],
        "Risk Vol Source": vol_signal, "MOVE": move, "MOVE Percentile": move_pct,
        "10Y 3M Change": ten_year_3m,
        "VIX Percentile": context["vix_pct"], "Unemployment Gap": context["unrate_delta"],
        "NFCI": nfci, "HYG vs SHY 1M": hyg_relative_1m,
        "Triggered": ", ".join(name for name, active in pillars.items() if active) or "None",
    }


def run_alpha_replica_backtest(
    prices: pd.DataFrame,
    fred: pd.DataFrame,
    common_start: pd.Timestamp,
    transaction_cost_bps: float = 5.0,
    vol_signal: str = "VIX or MOVE",
    target_func=_alpha_replica_target,
    label: str = "Alpha Replica v2",
    tickers: list[str] | None = None,
) -> dict:
    """Monthly concentrated alpha target with an immediate severe-risk cash gate."""
    universe = tickers or ETF_TICKERS
    asset_returns = prices[universe].pct_change(fill_method=None)
    dates = asset_returns.index[(asset_returns.index >= common_start) & asset_returns.notna().all(axis=1)]
    if len(dates) < 20:
        return {"monthly_returns": pd.Series(dtype=float), "weights": pd.DataFrame(),
                "turnover": pd.Series(dtype=float), "signals": pd.DataFrame(), "trades": pd.DataFrame()}

    current, target = target_func(prices, fred, dates[0])[0].reindex(universe, fill_value=0.0), None
    in_cash = False
    cash_start = -10_000
    clear_days = 0
    daily_result: dict[pd.Timestamp, float] = {}
    turnover_rows: dict[pd.Timestamp, float] = {}
    weight_rows: list[pd.Series] = [current.rename(dates[0])]
    signal_rows: list[dict] = []
    trades: list[dict] = []

    for position, date in enumerate(dates):
        day_return = float((current * asset_returns.loc[date].fillna(0.0)).sum())
        daily_result[date] = day_return
        if 1 + day_return > 0:
            current = current.mul(1 + asset_returns.loc[date].fillna(0.0)).div(1 + day_return)

        severe, risk = _severe_risk_snapshot(prices, fred, date, vol_signal=vol_signal)
        next_date = dates[position + 1] if position + 1 < len(dates) else date + pd.Timedelta(days=7)
        month_end = next_date.month != date.month
        reason = None
        proposed = current.copy()
        diagnostics = None
        if severe and not in_cash:
            cash_asset = _defensive_cash_asset(prices, date, universe)
            proposed = pd.Series(0.0, index=universe); proposed[cash_asset] = 1.0
            in_cash, cash_start, clear_days, reason = True, position, 0, "Severe Risk → Cash"
        elif in_cash:
            clear_days = clear_days + 1 if not severe else 0
            if position - cash_start >= RISK_CONFIG.minimum_hold_days and clear_days >= RISK_CONFIG.exit_clear_days:
                proposed, diagnostics = target_func(prices, fred, date)
                proposed = proposed.reindex(universe, fill_value=0.0)
                in_cash, reason = False, "Risk Clear → Re-enter"
        elif month_end and date.month in {3, 6, 9, 12}:
            proposed, diagnostics = target_func(prices, fred, date)
            proposed = proposed.reindex(universe, fill_value=0.0)
            reason = "Quarterly Alpha Review"

        if reason is not None:
            one_way = float((proposed - current).abs().sum() / 2)
            if one_way > 1e-6:
                daily_result[date] -= one_way * transaction_cost_bps / 10_000
                turnover_rows[date] = one_way
                current = proposed
                weight_rows.append(current.rename(date))
                trades.append({"Date": date, "Reason": reason, "Turnover": one_way, **risk})
        if month_end:
            if diagnostics is None:
                _, diagnostics = target_func(prices, fred, date)
            signal_rows.append({"Date": date, **diagnostics, **risk, "Cash Gate": in_cash})

    daily = pd.Series(daily_result, name=label).sort_index()
    return {
        "monthly_returns": (1 + daily).resample("ME").prod() - 1,
        "weights": pd.DataFrame(weight_rows).groupby(level=0).last().sort_index(),
        "turnover": pd.Series(turnover_rows, name="Alpha Replica Turnover").sort_index(),
        "signals": pd.DataFrame(signal_rows).set_index("Date"),
        "trades": pd.DataFrame(trades).set_index("Date") if trades else pd.DataFrame(),
    }


def run_bond_core_overlay_backtest(
    prices: pd.DataFrame,
    fred: pd.DataFrame,
    bond_returns: pd.Series | None,
    common_start: pd.Timestamp,
    transaction_cost_bps: float = 5.0,
    vol_signal: str = "VIX or MOVE",
) -> dict:
    """Dynamic 50~90% BOND ETF core plus aggressive RA-style overlay."""
    available = [
        ticker for ticker in STRUCTURED_PROXY_TICKERS
        if ticker in prices and prices[ticker].dropna().shape[0] >= 252
    ]
    cash_available = _available_cash_proxies(prices, min_obs=252)
    universe = ETF_TICKERS + available + cash_available
    empty = {
        "monthly_returns": pd.Series(dtype=float, name=MAIN_STRATEGY_NAME),
        "weights": pd.DataFrame(),
        "turnover": pd.Series(dtype=float, name=f"{MAIN_STRATEGY_NAME} Turnover"),
        "signals": pd.DataFrame(),
        "trades": pd.DataFrame(),
        "overlay": {},
    }
    if bond_returns is None or not len(bond_returns.dropna()):
        return empty

    overlay = run_alpha_replica_backtest(
        prices,
        fred,
        common_start,
        transaction_cost_bps,
        vol_signal,
        target_func=_aggressive_ra_overlay_target,
        label="Aggressive RA Overlay",
        tickers=universe,
    )
    overlay_returns = overlay["monthly_returns"].dropna()
    aligned = pd.concat(
        [
            bond_returns.rename("BOND Core"),
            overlay_returns.rename("Aggressive RA Overlay"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return empty

    signals = overlay["signals"].copy()
    if signals.empty or "BOND Core Weight" not in signals:
        signal_core = pd.Series(DEFAULT_BOND_CORE_WEIGHT, index=aligned.index)
        signal_core_asset = pd.Series("BOND", index=aligned.index)
        signal_core_rule = pd.Series("Neutral BOND core", index=aligned.index)
        signal_crisis_type = pd.Series("Normal carry", index=aligned.index)
        signal_hedge_etf = pd.Series("BOND", index=aligned.index)
    else:
        signal_core = signals["BOND Core Weight"].clip(0.50, 0.90).copy()

        def core_defense(row: pd.Series) -> tuple[str, str, float, str, str]:
            move_pct = float(row.get("MOVE Percentile", np.nan))
            vix_pct = float(row.get("VIX Percentile", np.nan))
            duration_score = float(row.get("Duration/MOVE Score", 0.0))
            ten_year_3m = float(row.get("10Y 3M Change", np.nan))
            hy_3m = float(row.get("HY OAS 3M", 0.0))
            risk_votes = float(row.get("Risk Votes", 0.0))
            nfci = float(row.get("NFCI", 0.0))
            credit_score = float(row.get("Credit Cushion Score", 0.0))
            unrate_gap = float(row.get("Unemployment Gap", 0.0))
            cash_gate = bool(row.get("Cash Gate", False))
            regime = row.get("Regime", "")
            cash_asset = _defensive_cash_asset(prices, row.name, universe)
            if cash_gate:
                return cash_asset, f"Cash gate active: 100% defensive {cash_asset}", 1.00, "Cooldown / cash gate", cash_asset
            rate_shock = (
                np.isfinite(move_pct) and move_pct >= 0.85
                and (
                    duration_score <= -0.5
                    or (np.isfinite(ten_year_3m) and ten_year_3m >= 0.15)
                )
            )
            severe_credit = risk_votes >= 2 or hy_3m >= 0.90 or nfci >= 0.35
            flight_to_quality = (
                (risk_votes >= 1 or (np.isfinite(vix_pct) and vix_pct >= 0.85) or hy_3m >= 0.45 or unrate_gap >= 0.50)
                and np.isfinite(ten_year_3m) and ten_year_3m <= -0.15
                and duration_score >= 0.5
            )
            early_stress = (
                risk_votes >= 1
                or hy_3m >= 0.45
                or nfci >= 0.15
                or credit_score <= -1.0
                or (np.isfinite(move_pct) and move_pct >= 0.82 and duration_score < 0)
            )
            recession_duration = (
                (regime == "Disinflation / Recession" or unrate_gap >= 0.50)
                and duration_score >= 0.75
                and (not np.isfinite(move_pct) or move_pct < 0.85)
            )
            if rate_shock:
                return cash_asset, f"Rate/MOVE shock: 100% defensive core to {cash_asset}", 1.00, "Rate / inflation shock", cash_asset
            if flight_to_quality:
                tlt_allowed = duration_score >= 1.5 and (risk_votes >= 1 or (np.isfinite(vix_pct) and vix_pct >= 0.85)) and hy_3m < 0.70
                hedge_asset = "TLT" if tlt_allowed and "TLT" in universe else "IEF"
                hedge_weight = 0.85 if severe_credit else 0.75
                return hedge_asset, f"Flight-to-quality: {hedge_weight:.0%} core to {hedge_asset}", hedge_weight, "Recession / flight-to-quality", hedge_asset
            if severe_credit:
                return cash_asset, f"Severe credit/liquidity stress: 100% defensive core to {cash_asset}", 1.00, "Credit / liquidity shock", cash_asset
            if recession_duration:
                hedge_asset = "IEF"
                return hedge_asset, f"Disinflation/recession hedge: 70% core to {hedge_asset}", 0.70, "Recession / duration hedge", hedge_asset
            if early_stress:
                return cash_asset, f"Early stress: 70% defensive core to {cash_asset}", 0.70, "Early warning", cash_asset
            return "BOND", "Normal: core bucket in BOND", float(row.get("BOND Core Weight", DEFAULT_BOND_CORE_WEIGHT)), "Normal carry", "BOND"

        defense = signals.apply(core_defense, axis=1, result_type="expand")
        defense.columns = ["Core Defense Asset", "Core Defense Rule", "Risk-Gated Core Weight", "Crisis Type", "Hedge ETF"]
        signals = pd.concat([signals, defense], axis=1)
        signal_core = signals["Risk-Gated Core Weight"].clip(0.50, 1.00)
        signal_core_asset = signals["Core Defense Asset"]
        signal_core_rule = signals["Core Defense Rule"]
        signal_crisis_type = signals["Crisis Type"]
        signal_hedge_etf = signals["Hedge ETF"]
    core_weight = signal_core.shift(1).reindex(aligned.index).ffill().fillna(DEFAULT_BOND_CORE_WEIGHT)
    core_asset = signal_core_asset.shift(1).reindex(aligned.index).ffill().fillna("BOND")
    core_rule = signal_core_rule.shift(1).reindex(aligned.index).ffill().fillna("Normal: core bucket in BOND")
    crisis_type = signal_crisis_type.shift(1).reindex(aligned.index).ffill().fillna("Normal carry")
    active_weight = 1.0 - core_weight
    defensive_assets = [ticker for ticker in CASH_PROXY_TICKERS + ["SHY", "IEF", "TLT"] if ticker in prices]
    monthly_prices = prices[defensive_assets].resample("ME").last().ffill()
    if len(monthly_prices) and monthly_prices.index[-1] > prices.index.max().normalize():
        monthly_prices = monthly_prices.iloc[:-1]
    fallback_returns = monthly_prices.pct_change(fill_method=None).reindex(aligned.index)
    core_return = aligned["BOND Core"].copy()
    if "SHY" in fallback_returns:
        core_return = core_return.where(core_asset != "SHY", fallback_returns["SHY"])
    for ticker in CASH_PROXY_TICKERS:
        if ticker in fallback_returns:
            core_return = core_return.where(core_asset != ticker, fallback_returns[ticker])
    if "IEF" in fallback_returns:
        core_return = core_return.where(core_asset != "IEF", fallback_returns["IEF"])
    if "TLT" in fallback_returns:
        core_return = core_return.where(core_asset != "TLT", fallback_returns["TLT"])
    core_return = core_return.fillna(aligned["BOND Core"])
    core_switch_cost = (core_asset != core_asset.shift(1)).astype(float).fillna(0.0) * core_weight * transaction_cost_bps / 10_000
    core_shift_cost = core_weight.diff().abs().fillna(0.0) * transaction_cost_bps / 10_000 + core_switch_cost
    combined = (
        core_weight * core_return
        + active_weight * aligned["Aggressive RA Overlay"]
        - core_shift_cost
    ).rename(MAIN_STRATEGY_NAME)

    carry_basket_assets = [
        ticker for ticker in NORMAL_CARRY_BASKET
        if ticker in prices and prices[ticker].dropna().shape[0] >= 252
    ]
    carry_basket_weight = pd.Series(0.0, index=aligned.index, name="Normal Carry Basket Weight")
    carry_basket_return = pd.Series(np.nan, index=aligned.index, name="Normal Carry Basket")
    if carry_basket_assets:
        carry_prices = prices[carry_basket_assets].resample("ME").last().ffill()
        if len(carry_prices) and carry_prices.index[-1] > prices.index.max().normalize():
            carry_prices = carry_prices.iloc[:-1]
        raw_weights = pd.Series(NORMAL_CARRY_BASKET).reindex(carry_basket_assets).astype(float)
        raw_weights = raw_weights / raw_weights.sum()
        carry_returns = carry_prices.pct_change(fill_method=None).reindex(aligned.index)
        carry_basket_return = carry_returns.mul(raw_weights, axis=1).sum(axis=1, min_count=1).rename("Normal Carry Basket")
        carry_active = (
            crisis_type.eq("Normal carry")
            & core_asset.eq("BOND")
            & carry_basket_return.notna()
        )
        carry_basket_weight.loc[carry_active] = 1.0
        carry_switch_cost = carry_basket_weight.diff().abs().fillna(carry_basket_weight).mul(transaction_cost_bps / 10_000)
        combined = (
            combined.where(~carry_active, carry_basket_return)
            - carry_switch_cost
        ).rename(MAIN_STRATEGY_NAME)

    overlay_weights = overlay["weights"].reindex(columns=universe, fill_value=0.0)
    if not signals.empty:
        overlay_weights = overlay_weights.reindex(overlay_weights.index.union(signals.index).sort_values()).ffill().fillna(0.0)
    if signals.empty or "BOND Core Weight" not in signals:
        display_core = pd.Series(DEFAULT_BOND_CORE_WEIGHT, index=overlay_weights.index)
        display_core_asset = pd.Series("BOND", index=overlay_weights.index)
    else:
        display_core = signal_core.reindex(overlay_weights.index).ffill().fillna(DEFAULT_BOND_CORE_WEIGHT)
        display_core_asset = signal_core_asset.reindex(overlay_weights.index).ffill().fillna("BOND")
    display_active = 1.0 - display_core
    full_weights = overlay_weights.mul(display_active, axis=0)
    full_weights.insert(0, "BOND", 0.0)
    for date, asset in display_core_asset.items():
        if asset not in full_weights.columns:
            full_weights[asset] = 0.0
        full_weights.loc[date, asset] = full_weights.loc[date, asset] + display_core.loc[date]

    display_crisis_type = signal_crisis_type.reindex(full_weights.index).ffill().fillna("Normal carry") if not signals.empty else pd.Series("Normal carry", index=full_weights.index)
    display_carry_basket = (
        display_crisis_type.eq("Normal carry")
        & display_core_asset.reindex(full_weights.index).ffill().fillna("BOND").eq("BOND")
    )
    display_carry_weight = pd.Series(0.0, index=full_weights.index)
    display_carry_weight.loc[display_carry_basket] = 1.0
    if display_carry_weight.any():
        basket_weights = pd.Series(NORMAL_CARRY_BASKET).reindex(full_weights.columns, fill_value=0.0)
        available_basket = basket_weights[basket_weights.index.isin(carry_basket_assets)]
        if available_basket.sum() > 0:
            basket_weights.loc[available_basket.index] = available_basket / available_basket.sum()
        full_weights = full_weights.mul(1.0 - display_carry_weight, axis=0)
        full_weights = full_weights.add(
            pd.DataFrame(
                np.outer(display_carry_weight.to_numpy(), basket_weights.reindex(full_weights.columns, fill_value=0.0).to_numpy()),
                index=full_weights.index,
                columns=full_weights.columns,
            ),
            fill_value=0.0,
        )
    full_weights = full_weights.loc[full_weights.index >= combined.index.min()]

    turnover = overlay["turnover"].reindex(display_active.index).fillna(0.0).mul(display_active, axis=0)
    turnover = turnover.add(display_core.diff().abs().fillna(0.0), fill_value=0.0)
    turnover = turnover.add(display_carry_weight.diff().abs().fillna(display_carry_weight), fill_value=0.0)
    turnover.name = f"{MAIN_STRATEGY_NAME} Turnover"
    trades = overlay["trades"].copy()
    if not trades.empty and "Turnover" in trades:
        trade_active = display_active.reindex(trades.index).ffill().fillna(DEFAULT_ACTIVE_OVERLAY_WEIGHT)
        trades["Turnover"] = trades["Turnover"] * trade_active

    if not signals.empty:
        signal_carry_basket = (
            signal_crisis_type.reindex(signals.index).ffill().fillna("Normal carry").eq("Normal carry")
            & signal_core_asset.reindex(signals.index).ffill().fillna("BOND").eq("BOND")
        )
        signals["BOND Core Weight"] = signal_core.reindex(signals.index).ffill().fillna(DEFAULT_BOND_CORE_WEIGHT)
        signals["Active Overlay Weight"] = 1.0 - signals["BOND Core Weight"]
        signals["Return Mode"] = np.where(signal_carry_basket, "JAAA-heavy normal carry basket", "Risk-first defense/base")
        signals["Carry Basket Weight"] = np.where(signal_carry_basket, 1.0, 0.0)
        signals["Carry Basket Assets"] = ", ".join(carry_basket_assets)
        signals["Strategy"] = MAIN_STRATEGY_NAME
    return {
        "monthly_returns": combined,
        "weights": full_weights.sort_index(),
        "turnover": turnover.sort_index(),
        "signals": signals,
        "trades": trades,
        "overlay": overlay,
    }


def run_hybrid_backtest(
    prices: pd.DataFrame,
    fred: pd.DataFrame,
    common_start: pd.Timestamp,
    transaction_cost_bps: float = 5.0,
) -> dict:
    """Asymmetric clock: monthly alpha, weekly momentum, daily risk-off."""
    asset_returns = prices[ETF_TICKERS].pct_change(fill_method=None)
    dates = asset_returns.index[(asset_returns.index >= common_start) & asset_returns.notna().all(axis=1)]
    if len(dates) < 20:
        return {"daily_returns": pd.Series(dtype=float), "monthly_returns": pd.Series(dtype=float),
                "weights": pd.DataFrame(), "turnover": pd.Series(dtype=float), "trades": pd.DataFrame()}

    current = pd.Series(0.0, index=ETF_TICKERS)
    current["SHY"] = 1.0
    last_trade_position = -10_000
    prior_risk_off = False
    slow_factors: pd.DataFrame | None = None
    weekly_momentum: pd.Series | None = None
    candidate_target = current.copy()
    pending_signature = None
    pending_reason = "Initial Review"
    confirmation_count = 0
    daily_result: dict[pd.Timestamp, float] = {}
    turnover_rows: dict[pd.Timestamp, float] = {}
    weight_rows: list[pd.Series] = [current.rename(dates[0])]
    trades: list[dict] = []
    slow_columns = ["Carry", "Value", "Real Yield", "Yield Curve", "Credit Spread", "Regime"]

    for position, date in enumerate(dates):
        day_asset_return = asset_returns.loc[date].fillna(0.0)
        gross_return = float((current * day_asset_return).sum())
        daily_result[date] = gross_return
        if 1 + gross_return > 0:
            current = current.mul(1 + day_asset_return).div(1 + gross_return)

        scored, context = score_etfs(prices.loc[:date], fred.loc[:date])
        next_date = dates[position + 1] if position + 1 < len(dates) else date + pd.Timedelta(days=7)
        month_end = next_date.month != date.month
        week_end = next_date.isocalendar().week != date.isocalendar().week or next_date.year != date.year

        update_reason = None
        if slow_factors is None or month_end:
            slow_factors = scored[slow_columns].reindex(ETF_TICKERS).copy()
            update_reason = "Monthly Alpha Review"
        if weekly_momentum is None or week_end:
            weekly_momentum = scored["Momentum"].reindex(ETF_TICKERS).copy()
            update_reason = update_reason or "Weekly Momentum Review"

        mixed = scored.copy().reindex(ETF_TICKERS)
        mixed[slow_columns] = slow_factors
        mixed["Momentum"] = weekly_momentum
        mixed["Total Score"] = sum(mixed[factor] * weight for factor, weight in FACTOR_WEIGHTS.items())
        if context["risk_off"]:
            candidate_target = recommend_weights(scored, context).reindex(ETF_TICKERS, fill_value=0.0)
        elif update_reason is not None:
            candidate_target = _hysteresis_target(mixed, context, current)
            pending_reason = update_reason

        selected = tuple(sorted(candidate_target[candidate_target >= .01].index.tolist()))
        signature = ("Risk Off" if context["risk_off"] else context["regime"], selected)
        if signature == pending_signature:
            confirmation_count += 1
        else:
            pending_signature = signature
            confirmation_count = 1

        weight_gap = float((candidate_target - current).abs().max())
        held_days = position - last_trade_position
        risk_transition = bool(context["risk_off"] and not prior_risk_off)
        confirmed_signal = (
            confirmation_count >= HYBRID_RULES["confirmation_days"]
            and weight_gap >= HYBRID_RULES["min_weight_gap"]
        )
        can_trade = held_days >= HYBRID_RULES["min_holding_days"]
        reason = "Risk Off" if risk_transition else pending_reason
        should_trade = (risk_transition and weight_gap >= .05) or (can_trade and confirmed_signal)

        if should_trade:
            delta = candidate_target - current
            max_delta = float(delta.abs().max())
            if not risk_transition and max_delta > HYBRID_RULES["max_asset_change"]:
                delta *= HYBRID_RULES["max_asset_change"] / max_delta
            proposed = current + delta
            proposed = proposed.clip(lower=0.0)
            proposed /= proposed.sum()
            turnover = float((proposed - current).abs().sum() / 2)
            cutoff = date - pd.Timedelta(days=365)
            recent_turnover = sum(value for trade_date, value in turnover_rows.items() if trade_date > cutoff)
            remaining = max(0.0, HYBRID_RULES["annual_turnover_cap"] - recent_turnover)
            if not risk_transition and turnover > remaining and turnover > 0:
                scale = remaining / turnover
                proposed = current + (proposed - current) * scale
                proposed /= proposed.sum()
                turnover = float((proposed - current).abs().sum() / 2)
            if turnover > 1e-6:
                daily_result[date] -= turnover * transaction_cost_bps / 10_000
                turnover_rows[date] = turnover
                current = proposed
                last_trade_position = position
                weight_rows.append(current.rename(date))
                trades.append(
                    {"Date": date, "Reason": reason, "Turnover": turnover,
                     "Regime": signature[0], "Max Weight Gap": weight_gap,
                     "Hold Days": held_days, "Rolling 1Y Turnover": recent_turnover + turnover}
                )
        prior_risk_off = bool(context["risk_off"])

    daily = pd.Series(daily_result, name="Hybrid Engine").sort_index()
    monthly = (1 + daily).resample("ME").prod() - 1
    monthly.name = "Hybrid Engine"
    weights = pd.DataFrame(weight_rows).groupby(level=0).last().sort_index()
    trade_frame = pd.DataFrame(trades).set_index("Date") if trades else pd.DataFrame()
    return {
        "daily_returns": daily,
        "monthly_returns": monthly,
        "weights": weights,
        "turnover": pd.Series(turnover_rows, name="Hybrid Turnover").sort_index(),
        "trades": trade_frame,
    }


def alpha_source_attribution(
    alpha_returns: pd.Series,
    signals: pd.DataFrame,
    target_returns: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    """Explain next-month alpha returns by the prior month-end source scorecard."""
    if signals.empty or alpha_returns.dropna().empty:
        return {"by_source": pd.DataFrame(), "score_tests": pd.DataFrame(), "monthly": pd.DataFrame()}

    signal_cols = [
        "Dominant Alpha", "MBS RV Score", "Credit Cushion Score", "Duration/MOVE Score",
        "Curve Score", "Real Yield Score", "Cash Gate", "Risk Votes", "PIMCO Anchor Gap",
    ]
    available = [column for column in signal_cols if column in signals]
    with pd.option_context("future.no_silent_downcasting", True):
        explanatory = signals[available].reindex(alpha_returns.index).ffill().infer_objects(copy=False).shift(1)
    frame = pd.DataFrame({"Alpha Return": alpha_returns}).join(explanatory)
    if benchmark_returns is not None:
        frame["AGG Return"] = benchmark_returns.reindex(frame.index)
        frame["Active vs AGG"] = frame["Alpha Return"] - frame["AGG Return"]
    if target_returns is not None:
        frame["BOND Return"] = target_returns.reindex(frame.index)
        frame["Active vs BOND"] = frame["Alpha Return"] - frame["BOND Return"]
    frame = frame.dropna(subset=["Alpha Return", "Dominant Alpha"])

    rows: list[dict] = []
    for source, group in frame.groupby("Dominant Alpha"):
        dominant_score_col = f"{source} Score"
        row = {
            "Months": len(group),
            "Avg Dominant Score": group[dominant_score_col].mean() if dominant_score_col in group else np.nan,
            "Avg Return": group["Alpha Return"].mean(),
            "Cumulative Return": (1 + group["Alpha Return"]).prod() - 1,
            "Return Hit Rate": (group["Alpha Return"] > 0).mean(),
            "Cash Gate %": group["Cash Gate"].mean() if "Cash Gate" in group else np.nan,
        }
        if "Active vs AGG" in group:
            row.update({
                "Avg vs AGG": group["Active vs AGG"].mean(),
                "Total vs AGG": group["Active vs AGG"].sum(),
                "AGG Hit Rate": (group["Active vs AGG"] > 0).mean(),
            })
        if "Active vs BOND" in group:
            row.update({
                "Avg vs BOND": group["Active vs BOND"].mean(),
                "Total vs BOND": group["Active vs BOND"].sum(),
                "BOND Hit Rate": (group["Active vs BOND"] > 0).mean(),
            })
        rows.append({"Dominant Alpha": source, **row})
    by_source = pd.DataFrame(rows).set_index("Dominant Alpha") if rows else pd.DataFrame()
    if not by_source.empty and "Total vs AGG" in by_source:
        by_source = by_source.sort_values("Total vs AGG", ascending=False)

    score_rows: list[dict] = []
    score_cols = ["MBS RV Score", "Credit Cushion Score", "Duration/MOVE Score", "Curve Score", "Real Yield Score"]
    for column in [item for item in score_cols if item in frame]:
        valid = frame[[column, "Alpha Return"] + [c for c in ["Active vs AGG", "Active vs BOND"] if c in frame]].dropna()
        if len(valid) < 4:
            continue
        positive = valid[valid[column] > 0]
        negative = valid[valid[column] < 0]
        row = {
            "Score": column.replace(" Score", ""),
            "Observations": len(valid),
            "Corr with Return": valid[column].corr(valid["Alpha Return"]),
            "Avg Return if Positive": positive["Alpha Return"].mean() if len(positive) else np.nan,
            "Avg Return if Negative": negative["Alpha Return"].mean() if len(negative) else np.nan,
            "Positive Months": len(positive),
            "Negative Months": len(negative),
        }
        if "Active vs AGG" in valid:
            row["Corr with AGG Alpha"] = valid[column].corr(valid["Active vs AGG"])
            row["Avg AGG Alpha if Positive"] = positive["Active vs AGG"].mean() if len(positive) else np.nan
            row["Avg AGG Alpha if Negative"] = negative["Active vs AGG"].mean() if len(negative) else np.nan
        if "Active vs BOND" in valid:
            row["Corr with BOND Gap"] = valid[column].corr(valid["Active vs BOND"])
            row["Avg BOND Gap if Positive"] = positive["Active vs BOND"].mean() if len(positive) else np.nan
            row["Avg BOND Gap if Negative"] = negative["Active vs BOND"].mean() if len(negative) else np.nan
        score_rows.append(row)
    score_tests = pd.DataFrame(score_rows).set_index("Score") if score_rows else pd.DataFrame()

    return {"by_source": by_source, "score_tests": score_tests, "monthly": frame}


def simulate_structured_proxy_v2(
    prices: pd.DataFrame,
    fred: pd.DataFrame,
    common_start: pd.Timestamp,
    target_returns: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    transaction_cost_bps: float = 5.0,
    vol_signal: str = "VIX or MOVE",
) -> dict:
    available = [
        ticker for ticker in STRUCTURED_PROXY_TICKERS
        if ticker in prices and prices[ticker].dropna().shape[0] >= 252
    ]
    if not available:
        return {"summary": pd.DataFrame(), "returns": pd.DataFrame(), "weights": {}, "signals": {}, "best_name": None}

    universe = ETF_TICKERS + available
    monthly = prices[universe].resample("ME").last().ffill()
    if len(monthly) and monthly.index[-1] > prices.index.max().normalize():
        monthly = monthly.iloc[:-1]
    dates = monthly.index[(monthly.index >= common_start) & monthly[universe].notna().all(axis=1)]
    if len(dates) < 6:
        return {"summary": pd.DataFrame(), "returns": pd.DataFrame(), "weights": {}, "signals": {}, "best_name": None}

    configs = structured_v2_configs()
    config_returns: dict[str, dict[pd.Timestamp, float]] = {config["name"]: {} for config in configs}
    config_turnover: dict[str, dict[pd.Timestamp, float]] = {config["name"]: {} for config in configs}
    config_weights: dict[str, dict[pd.Timestamp, pd.Series]] = {config["name"]: {} for config in configs}
    config_signals: dict[str, list[dict]] = {config["name"]: [] for config in configs}
    previous = {config["name"]: pd.Series(0.0, index=universe) for config in configs}
    current_weights = {config["name"]: pd.Series(0.0, index=universe) for config in configs}
    initialized = {config["name"]: False for config in configs}

    for current, following in zip(dates[:-1], dates[1:]):
        next_returns = monthly.loc[following] / monthly.loc[current] - 1
        severe, risk = _severe_risk_snapshot(prices, fred, current, vol_signal=vol_signal)
        review = (current.month in {3, 6, 9, 12})
        for config in configs:
            name = config["name"]
            diagnostics = None
            if severe:
                weights = pd.Series(0.0, index=universe)
                weights["SHY"] = 1.0
                diagnostics = {"Structured v2 Rule": "Severe Risk → SHY", "Structured v2 Config": name}
            elif (not initialized[name]) or review:
                weights, diagnostics = _structured_proxy_target_v2(prices, fred, current, config)
                weights = weights.reindex(universe, fill_value=0.0)
                initialized[name] = True
            else:
                weights = current_weights[name].copy()
            if float(weights.sum()) <= 0:
                weights["SHY"] = 1.0
            weights = weights / weights.sum()
            prior = previous[name]
            turnover = float((weights - prior).abs().sum()) if prior.sum() == 0 else float((weights - prior).abs().sum() / 2)
            config_returns[name][following] = float((weights * next_returns).sum() - turnover * transaction_cost_bps / 10_000)
            config_turnover[name][current] = turnover
            config_weights[name][current] = weights
            if diagnostics is not None:
                config_signals[name].append({"Date": current, **diagnostics, **risk})
            previous[name] = weights
            current_weights[name] = weights

    return_frame = pd.DataFrame({name: pd.Series(values) for name, values in config_returns.items()}).sort_index()
    rf = fred["DGS3MO"].resample("ME").last().reindex(return_frame.index, method="ffill") / 100 / 12
    agg_metrics = performance_metrics(benchmark_returns, rf) if benchmark_returns is not None else {}
    rows = []
    for config in configs:
        name = config["name"]
        series = return_frame[name].dropna()
        metrics = performance_metrics(series, rf, benchmark_returns)
        active_vs_agg = pd.Series(dtype=float)
        if benchmark_returns is not None:
            aligned_agg = pd.concat([series, benchmark_returns.reindex(series.index)], axis=1).dropna()
            active_vs_agg = aligned_agg.iloc[:, 0] - aligned_agg.iloc[:, 1] if len(aligned_agg) else pd.Series(dtype=float)
        ann_agg_alpha = active_vs_agg.mean() * 12 if len(active_vs_agg) else np.nan
        mdd = metrics.get("Max Drawdown", np.nan)
        agg_mdd = agg_metrics.get("Max Drawdown", np.nan)
        mdd_penalty = max(0.0, abs(mdd) - abs(agg_mdd)) if np.isfinite(mdd) and np.isfinite(agg_mdd) else 0.0
        tracking_error = metrics.get("Tracking Error", np.nan)
        joint_score = (
            (ann_agg_alpha if np.isfinite(ann_agg_alpha) else 0.0)
            - 0.50 * mdd_penalty
            - 0.20 * (tracking_error if np.isfinite(tracking_error) else 0.0)
            + 0.01 * (metrics.get("Sharpe", 0.0) if np.isfinite(metrics.get("Sharpe", np.nan)) else 0.0)
        )
        annual_turnover = float(pd.Series(config_turnover[name]).mean() * 12)
        avg_sleeve = float(pd.DataFrame.from_dict(config_weights[name], orient="index").reindex(columns=STRUCTURED_PROXY_TICKERS, fill_value=0.0).sum(axis=1).mean())
        rows.append({
            "Combination": name,
            "Cap": config["cap"],
            "Mix": config["mix_name"],
            "Gate": config["gate_name"],
            "Ann. AGG Alpha": ann_agg_alpha,
            "Joint Score": joint_score,
            "Avg Structured Sleeve": avg_sleeve,
            "Ann. Turnover": annual_turnover,
            **metrics,
        })

    summary = pd.DataFrame(rows).set_index("Combination").sort_values("Joint Score", ascending=False)
    weight_frames = {
        name: pd.DataFrame.from_dict(rows, orient="index").reindex(columns=universe, fill_value=0.0).sort_index()
        for name, rows in config_weights.items()
    }
    signal_frames = {
        name: pd.DataFrame(rows).set_index("Date").sort_index() if rows else pd.DataFrame()
        for name, rows in config_signals.items()
    }
    best_name = summary.index[0] if not summary.empty else None
    return {"summary": summary, "returns": return_frame, "weights": weight_frames, "signals": signal_frames, "best_name": best_name}



def run_backtest(
    prices: pd.DataFrame,
    fred: pd.DataFrame,
    target: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    transaction_cost_bps: float = 5.0,
) -> dict:
    monthly = prices[ETF_TICKERS].resample("ME").last().ffill()
    if len(monthly) and monthly.index[-1] > prices.index.max().normalize():
        monthly = monthly.iloc[:-1]
    required_fred = [
        "DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y", "DFII10", "T10YIE",
        "BAMLC0A0CM", "BAMLH0A0HYM2", "CPILFESL", "INDPRO", "UNRATE", "VIXCLS", "MORTGAGE30US",
    ]
    starts = [_first_valid(monthly[col]) for col in ETF_TICKERS]
    starts += [_first_valid(fred[col]) for col in required_fred]
    starts = [date for date in starts if date is not None]
    raw_start = max(starts)
    dates = monthly.index[monthly.index >= raw_start]
    if len(dates) < 3:
        raise ValueError("공통 데이터 구간이 너무 짧아 백테스트할 수 없습니다.")
    common_start = dates.min()

    strategy_returns: dict[pd.Timestamp, float] = {}
    static_returns: dict[pd.Timestamp, float] = {}
    weight_rows: list[pd.Series] = []
    regime_rows: list[dict] = []
    turnover_rows: dict[pd.Timestamp, float] = {}
    previous = pd.Series(0.0, index=ETF_TICKERS)

    for current, following in zip(dates[:-1], dates[1:]):
        scored, context = score_etfs(prices.loc[:current], fred.loc[:current])
        weights = recommend_weights(scored, context).reindex(ETF_TICKERS, fill_value=0.0)
        next_returns = monthly.loc[following] / monthly.loc[current] - 1
        turnover = float((weights - previous).abs().sum()) if previous.sum() == 0 else float((weights - previous).abs().sum() / 2)
        cost = turnover * transaction_cost_bps / 10_000
        strategy_returns[following] = float((weights * next_returns).sum() - cost)
        static_returns[following] = float((pd.Series(STATIC_CORE) * next_returns).sum())
        weight_rows.append(weights.rename(current))
        regime_rows.append(
            {
                "date": current,
                "Regime": "Risk Off" if context["risk_off"] else context["regime"],
                "Risk Off": context["risk_off"],
                "Inflation 3M": context["inflation_3m"],
                "Growth 6M": context["growth_6m"],
            }
        )
        turnover_rows[current] = turnover
        previous = weights

    returns = pd.Series(strategy_returns, name="Rule Engine").sort_index()
    static = pd.Series(static_returns, name="Static Core").sort_index()
    rf = fred["DGS3MO"].resample("ME").last().reindex(returns.index, method="ffill") / 100 / 12
    alpha_vol_source = "VIX or MOVE" if "MOVE" in prices and prices["MOVE"].notna().any() else "VIX"
    empty_engine = {
        "monthly_returns": pd.Series(dtype=float), "weights": pd.DataFrame(),
        "turnover": pd.Series(dtype=float), "signals": pd.DataFrame(), "trades": pd.DataFrame(),
    }
    hybrid = {"monthly_returns": pd.Series(dtype=float), "weights": pd.DataFrame(), "turnover": pd.Series(dtype=float), "trades": pd.DataFrame()}
    hybrid_returns = pd.Series(dtype=float, name="Hybrid Engine").reindex(returns.index)
    replica = {"returns": pd.Series(dtype=float), "weights": pd.DataFrame(), "turnover": pd.Series(dtype=float), "duration": pd.Series(dtype=float)}
    replica_returns = pd.Series(dtype=float, name="BOND Exposure Replica").reindex(returns.index)
    alpha_replica = empty_engine.copy()
    alpha_vix = empty_engine.copy()
    alpha_move = empty_engine.copy()
    alpha_replica_returns = alpha_replica["monthly_returns"].reindex(returns.index)
    alpha_replica_returns.name = "Alpha Replica v2"
    structured_available = [
        ticker for ticker in STRUCTURED_PROXY_TICKERS
        if ticker in prices and prices[ticker].dropna().shape[0] >= 252
    ]
    structured_universe = ETF_TICKERS + structured_available
    structured_proxy = empty_engine.copy()
    structured_returns = structured_proxy["monthly_returns"].reindex(returns.index)
    structured_returns.name = "Structured Proxy Engine"
    def aligned_returns(series: pd.Series | None, name: str, index: pd.Index | None = None) -> pd.Series | None:
        if series is None or not len(series.dropna()):
            return None
        monthly_prices = series.resample("ME").last().ffill()
        if len(monthly_prices) and monthly_prices.index[-1] > series.index.max().normalize():
            monthly_prices = monthly_prices.iloc[:-1]
        aligned = monthly_prices.pct_change().reindex(returns.index if index is None else index)
        aligned.name = name
        return aligned

    target_returns = aligned_returns(target, "PIMCO Active Bond (BOND)")
    benchmark_returns = aligned_returns(benchmark, "US Aggregate Bond (AGG)")
    bond_core_overlay = run_bond_core_overlay_backtest(
        prices,
        fred,
        target_returns,
        common_start,
        transaction_cost_bps,
        alpha_vol_source,
    )
    bond_core_overlay_returns = bond_core_overlay["monthly_returns"].reindex(returns.index)
    bond_core_overlay_returns.name = MAIN_STRATEGY_NAME
    alpha_attribution = {"by_source": pd.DataFrame(), "score_tests": pd.DataFrame(), "monthly": pd.DataFrame()}
    structured_v2 = simulate_structured_proxy_v2(
        prices,
        fred,
        common_start,
        target_returns,
        benchmark_returns,
        transaction_cost_bps,
        alpha_vol_source,
    )
    structured_v2_best = structured_v2["best_name"]
    if structured_v2_best:
        structured_v2_daily = {
            "monthly_returns": structured_v2["returns"][structured_v2_best].dropna(),
            "weights": structured_v2["weights"].get(structured_v2_best, pd.DataFrame()),
            "turnover": pd.Series(dtype=float),
            "signals": structured_v2["signals"].get(structured_v2_best, pd.DataFrame()),
            "trades": pd.DataFrame(),
        }
    else:
        structured_v2_daily = empty_engine.copy()
    structured_v2_returns = structured_v2_daily["monthly_returns"].reindex(returns.index)
    structured_v2_returns.name = "Structured Proxy v2 Candidate"
    nav_inputs = [bond_core_overlay_returns, structured_v2_returns] + [item for item in [target_returns, benchmark_returns] if item is not None]
    main_strategy_index = bond_core_overlay_returns.dropna().index
    main_target_returns = target_returns.reindex(main_strategy_index) if target_returns is not None and len(main_strategy_index) else target_returns
    main_benchmark_returns = benchmark_returns.reindex(main_strategy_index) if benchmark_returns is not None and len(main_strategy_index) else benchmark_returns

    return {
        "returns": returns,
        "alpha_replica_returns": alpha_replica_returns,
        "structured_proxy_returns": structured_returns,
        "structured_v2_returns": structured_v2_returns,
        "bond_core_overlay_returns": bond_core_overlay_returns,
        "replica_returns": replica_returns,
        "hybrid_returns": hybrid_returns,
        "static_returns": static,
        "target_returns": target_returns,
        "benchmark_returns": benchmark_returns,
        "nav": (1 + pd.concat(nav_inputs, axis=1).dropna(how="all")).cumprod() * 100,
        "weights": pd.DataFrame(weight_rows),
        "regimes": pd.DataFrame(regime_rows).set_index("date"),
        "turnover": pd.Series(turnover_rows, name="Turnover"),
        "hybrid_weights": hybrid["weights"],
        "hybrid_turnover": hybrid["turnover"],
        "hybrid_trades": hybrid["trades"],
        "replica_weights": replica["weights"],
        "replica_turnover": replica["turnover"],
        "replica_duration": replica["duration"],
        "alpha_replica_weights": alpha_replica["weights"],
        "alpha_replica_turnover": alpha_replica["turnover"],
        "alpha_replica_signals": alpha_replica["signals"],
        "alpha_replica_trades": alpha_replica["trades"],
        "alpha_source_attribution": alpha_attribution,
        "structured_proxy_weights": structured_proxy["weights"],
        "structured_proxy_turnover": structured_proxy["turnover"],
        "structured_proxy_signals": structured_proxy["signals"],
        "structured_proxy_trades": structured_proxy["trades"],
        "structured_proxy_available": structured_available,
        "structured_v2_summary": structured_v2["summary"],
        "structured_v2_returns_all": structured_v2["returns"],
        "structured_v2_weights": structured_v2_daily["weights"],
        "structured_v2_turnover": structured_v2_daily["turnover"],
        "structured_v2_signals": structured_v2_daily["signals"],
        "structured_v2_trades": structured_v2_daily["trades"],
        "structured_v2_best_name": structured_v2_best,
        "bond_core_overlay_weights": bond_core_overlay["weights"],
        "bond_core_overlay_turnover": bond_core_overlay["turnover"],
        "bond_core_overlay_signals": bond_core_overlay["signals"],
        "bond_core_overlay_trades": bond_core_overlay["trades"],
        "bond_core_overlay_overlay": bond_core_overlay["overlay"],
        "metrics": performance_metrics(returns, rf, benchmark_returns),
        "alpha_replica_metrics": performance_metrics(alpha_replica_returns, rf, benchmark_returns),
        "structured_proxy_metrics": performance_metrics(structured_returns, rf, benchmark_returns),
        "structured_v2_metrics": performance_metrics(structured_v2_returns, rf, benchmark_returns),
        "bond_core_overlay_metrics": performance_metrics(bond_core_overlay_returns, rf, benchmark_returns),
        "bond_core_overlay_bond_metrics": performance_metrics(bond_core_overlay_returns, rf, target_returns),
        "alpha_vix_metrics": performance_metrics(alpha_vix["monthly_returns"].reindex(returns.index), rf, benchmark_returns),
        "alpha_move_metrics": performance_metrics(alpha_move["monthly_returns"].reindex(returns.index), rf, benchmark_returns),
        "alpha_vol_source": alpha_vol_source,
        "replica_metrics": performance_metrics(replica_returns, rf, benchmark_returns),
        "hybrid_metrics": performance_metrics(hybrid_returns, rf, benchmark_returns),
        "static_metrics": performance_metrics(static, rf, benchmark_returns),
        "target_metrics": performance_metrics(main_target_returns, rf) if main_target_returns is not None else {},
        "benchmark_metrics": performance_metrics(main_benchmark_returns, rf) if main_benchmark_returns is not None else {},
        "strategy_metadata": {
            "Rule Engine": _strategy_metadata(returns, "AGG", "static-rule-v1", "month-end signal, next-month return"),
            "Structured Proxy v2 Candidate": _strategy_metadata(
                structured_v2_returns, "AGG", "structured-v2-grid-selected", "month-end/quarter-end signal, next-period return"
            ),
            MAIN_STRATEGY_NAME: _strategy_metadata(
                bond_core_overlay_returns,
                "AGG for market performance; BOND only for relative overlay diagnostics",
                "bond-core-overlay-v1",
                "prior signal applied to next monthly return",
                uses_bond_core=True,
            ),
            "PIMCO Active Bond (BOND)": _strategy_metadata(
                main_target_returns if main_target_returns is not None else pd.Series(dtype=float),
                "reference asset",
                "observed",
                "observed return",
                executed=main_target_returns is not None,
            ),
            "US Aggregate Bond (AGG)": _strategy_metadata(
                main_benchmark_returns if main_benchmark_returns is not None else pd.Series(dtype=float),
                "benchmark",
                "observed",
                "observed return",
                executed=main_benchmark_returns is not None,
            ),
            "Hybrid Engine": _strategy_metadata(hybrid_returns, "AGG", "disabled", "not executed", executed=False),
            "BOND Exposure Replica": _strategy_metadata(replica_returns, "BOND", "disabled", "not executed", executed=False),
            "Alpha Replica v2": _strategy_metadata(alpha_replica_returns, "AGG", "disabled", "not executed", executed=False),
            "Structured Proxy Engine": _strategy_metadata(structured_returns, "AGG", "disabled", "not executed", executed=False),
        },
        "common_start": common_start,
        "main_strategy_start": main_strategy_index.min() if len(main_strategy_index) else common_start,
        "transaction_cost_bps": transaction_cost_bps,
    }


def infer_long_only_exposures(prices: pd.DataFrame, comparator: pd.Series, window: int = 24, ridge: float = 0.01) -> pd.DataFrame:
    """Rolling ridge return replication; clips short estimates and normalizes to long-only weights."""
    asset_returns = prices[ETF_TICKERS].resample("ME").last().ffill().pct_change()
    target_returns = comparator.resample("ME").last().ffill().pct_change().rename("Target")
    joined = pd.concat([asset_returns, target_returns], axis=1).dropna()
    rows: list[pd.Series] = []
    for end in range(window, len(joined) + 1):
        sample = joined.iloc[end - window:end]
        x = sample[ETF_TICKERS].to_numpy()
        y = sample["Target"].to_numpy()
        x = x - x.mean(axis=0)
        y = y - y.mean()
        beta = np.linalg.solve(x.T @ x + ridge * np.eye(len(ETF_TICKERS)), x.T @ y)
        beta = np.clip(beta, 0, None)
        weights = beta / beta.sum() if beta.sum() else np.repeat(1 / len(beta), len(beta))
        rows.append(pd.Series(weights, index=ETF_TICKERS, name=sample.index[-1]))
    return pd.DataFrame(rows)


FACTOR_COMBINATIONS = {
    "Equal Factor": {"Carry": .125, "Momentum": .125, "Value": .125, "Real Yield": .125, "Yield Curve": .125, "Credit Spread": .125, "Defensive": .125, "Regime": .125},
    "Carry + Momentum": {"Carry": .35, "Momentum": .35, "Defensive": .15, "Regime": .15},
    "Valuation + Curve": {"Value": .30, "Real Yield": .20, "Yield Curve": .25, "Credit Spread": .15, "Defensive": .10},
    "Credit + Defensive": {"Carry": .15, "Credit Spread": .30, "Defensive": .30, "Regime": .15, "Momentum": .10},
    "Macro Regime": {"Yield Curve": .25, "Real Yield": .15, "Defensive": .25, "Regime": .35},
    "PIMCO Core Proxy": {"Carry": .20, "Value": .10, "Yield Curve": .20, "Credit Spread": .20, "Defensive": .15, "Regime": .15},
}


def simulate_rule_combinations(
    prices: pd.DataFrame,
    fred: pd.DataFrame,
    target: pd.Series,
    common_start: pd.Timestamp,
    transaction_cost_bps: float = 5.0,
) -> dict:
    monthly = prices[ETF_TICKERS].resample("ME").last().ffill()
    if monthly.index[-1] > prices.index.max().normalize():
        monthly = monthly.iloc[:-1]
    dates = monthly.index[monthly.index >= common_start]
    target_monthly = target.resample("ME").last().ffill()
    if target_monthly.index[-1] > target.index.max().normalize():
        target_monthly = target_monthly.iloc[:-1]
    target_returns = target_monthly.pct_change().reindex(dates[1:])
    rf = fred["DGS3MO"].resample("ME").last().reindex(dates[1:], method="ffill") / 100 / 12

    config_returns = {name: {} for name in FACTOR_COMBINATIONS}
    config_turnover = {name: {} for name in FACTOR_COMBINATIONS}
    config_weights = {name: {} for name in FACTOR_COMBINATIONS}
    previous = {name: pd.Series(0.0, index=ETF_TICKERS) for name in FACTOR_COMBINATIONS}

    for current, following in zip(dates[:-1], dates[1:]):
        scored, context = score_etfs(prices.loc[:current], fred.loc[:current])
        next_returns = monthly.loc[following] / monthly.loc[current] - 1
        for name, mix in FACTOR_COMBINATIONS.items():
            variant = scored.copy()
            variant["Total Score"] = sum(variant[factor] * weight for factor, weight in mix.items())
            weights = recommend_weights(variant, context).reindex(ETF_TICKERS, fill_value=0.0)
            prior = previous[name]
            turnover = float((weights - prior).abs().sum()) if prior.sum() == 0 else float((weights - prior).abs().sum() / 2)
            config_returns[name][following] = float((weights * next_returns).sum() - turnover * transaction_cost_bps / 10_000)
            config_turnover[name][current] = turnover
            config_weights[name][current] = weights
            previous[name] = weights

    return_frame = pd.DataFrame({name: pd.Series(values) for name, values in config_returns.items()}).sort_index()
    target_metrics = performance_metrics(target_returns, rf)
    rows = []
    for name in FACTOR_COMBINATIONS:
        series = return_frame[name]
        metrics = performance_metrics(series, rf, target_returns)
        aligned = pd.concat([series, target_returns], axis=1).dropna()
        rmse = float(np.sqrt(np.mean((aligned.iloc[:, 0] - aligned.iloc[:, 1]) ** 2)) * np.sqrt(12))
        annual_turnover = float(pd.Series(config_turnover[name]).mean() * 12)
        loss = (
            .35 * rmse
            + .20 * abs(metrics.get("Ann. Vol", 0) - target_metrics.get("Ann. Vol", 0))
            + .20 * abs(metrics.get("Max Drawdown", 0) - target_metrics.get("Max Drawdown", 0))
            + .15 * abs(metrics.get("Beta", 0) - 1) * .05
            + .10 * annual_turnover * transaction_cost_bps / 10_000
        )
        rows.append({"Combination": name, **metrics, "RMSE": rmse, "Ann. Turnover": annual_turnover, "Replication Loss": loss})
    summary = pd.DataFrame(rows).set_index("Combination").sort_values("Replication Loss")
    weight_frames = {
        name: pd.DataFrame.from_dict(rows, orient="index").reindex(columns=ETF_TICKERS, fill_value=0.0).sort_index()
        for name, rows in config_weights.items()
    }
    return {"summary": summary, "returns": return_frame, "target_returns": target_returns, "weights": weight_frames}
