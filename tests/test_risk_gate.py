"""Pinning tests for the severe-risk cash gate (backtest._severe_risk_snapshot)
and its hold/clear-day state machine (backtest.run_alpha_replica_backtest).

Constraint: production code (backtest.py, signals.py, risk_config.py, app.py) is
never modified or monkeypatched here. Every test builds a synthetic prices/fred
DataFrame and feeds it into the real, unmodified functions -- only the INPUT
DATA is manufactured.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from backtest import (  # noqa: E402
    _severe_risk_snapshot,
    _first_valid,
    run_alpha_replica_backtest,
    run_bond_core_overlay_backtest,
    performance_metrics,
)
from risk_config import RISK_CONFIG  # noqa: E402
from data_sources import ETF_TICKERS, STRUCTURED_PROXY_TICKERS, CASH_PROXY_TICKERS  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic fixture builder
# ---------------------------------------------------------------------------
_DRIFTS = {"SHY": 0.00008, "IEF": 0.00012, "TLT": 0.00015, "TIP": 0.0001,
           "LQD": 0.00013, "HYG": 0.00014, "MBB": 0.0001}


def _make_baseline(n_days: int = 900, end: str = "2024-01-31", oscillate: bool = True):
    """A multi-year calm synthetic prices/fred pair covering every column and
    lookback window score_etfs / _severe_risk_snapshot / _alpha_replica_target
    need. `oscillate=True` gives VIX/MOVE gentle sinusoidal variation (needed so
    percentile-based conditions behave sensibly rather than pinning at 100%
    against a perfectly flat series); the tail is flattened to a calm constant
    so "calm" assertions don't depend on where the sine wave happens to sit.
    `oscillate=False` gives a fully flat series, used by the hold/clear-day
    tests where only one deliberately injected spike should ever move severe().
    """
    idx = pd.bdate_range(end=end, periods=n_days)
    t = np.arange(n_days)

    fred = pd.DataFrame(index=idx)
    fred["DGS3MO"] = 3.0
    fred["DGS2"] = 2.5
    fred["DGS5"] = 2.8
    fred["DGS10"] = 3.0
    fred["DGS30"] = 3.5
    fred["T10Y2Y"] = fred["DGS10"] - fred["DGS2"]
    fred["DFII10"] = 1.0
    fred["T10YIE"] = 2.0
    fred["BAMLC0A0CM"] = 1.0
    fred["BAMLH0A0HYM2"] = 3.0
    fred["VIXCLS"] = 15.0 + (3.0 * np.sin(2 * np.pi * t / 63) if oscillate else 0.0)
    fred["NFCI"] = -0.2 + (0.02 * np.sin(2 * np.pi * t / 126) if oscillate else 0.0)
    fred["MORTGAGE30US"] = 5.0
    fred["CPILFESL"] = 260.0 * np.exp(0.0015 * t / 21)
    fred["INDPRO"] = 100.0 * np.exp(0.0008 * t / 21)
    fred["UNRATE"] = 4.0

    prices = pd.DataFrame(index=idx)
    for ticker, drift in _DRIFTS.items():
        prices[ticker] = 100.0 * np.exp(drift * t)
    prices["MOVE"] = 80.0 + (5.0 * np.sin(2 * np.pi * t / 63) if oscillate else 0.0)
    prices["BOND"] = 100.0 * np.exp(0.0001 * t)

    if oscillate:
        flat_window = 100
        fred.iloc[-flat_window:, fred.columns.get_loc("VIXCLS")] = 15.0
        prices.iloc[-flat_window:, prices.columns.get_loc("MOVE")] = 80.0

    return fred, prices


def _trading_dates(prices: pd.DataFrame, fred: pd.DataFrame, tickers: list[str], common_start: pd.Timestamp) -> pd.DatetimeIndex:
    asset_returns = prices[tickers].pct_change(fill_method=None)
    return asset_returns.index[(asset_returns.index >= common_start) & asset_returns.notna().all(axis=1)]


# ---------------------------------------------------------------------------
# 1. vote_count >= entry_votes triggers severe with no override active
# ---------------------------------------------------------------------------
def test_vote_ge_2_triggers_severe_without_override():
    fred, prices = _make_baseline()
    date = fred.index[-1]

    # Credit pillar via HYG underperforming SHY (not via HY OAS, so C2 stays off).
    hyg_col = prices.columns.get_loc("HYG")
    anchor = prices["HYG"].iloc[-22]
    decline = np.linspace(0.0, -0.06, 21)
    prices.iloc[-21:, hyg_col] = anchor * (1 + decline)
    # Labor pillar via a sustained rise in unemployment.
    fred.iloc[-25:, fred.columns.get_loc("UNRATE")] = 5.0

    severe, risk = _severe_risk_snapshot(prices, fred, date, vol_signal="VIX or MOVE")

    assert risk["Risk Votes"] >= RISK_CONFIG.entry_votes
    assert severe is True
    # confirm this is a pure vote-count trigger, not also an override
    assert risk["HY OAS 3M"] < 0.90
    assert risk["NFCI"] < 0.35
    assert not (np.isfinite(risk["MOVE Percentile"]) and risk["MOVE Percentile"] >= RISK_CONFIG.move_percentile_crisis)


# ---------------------------------------------------------------------------
# 2. C1 override (MOVE percentile-crisis + 10Y rate-shock) triggers severe even
#    with vote_count below entry_votes
# ---------------------------------------------------------------------------
def test_c1_override_triggers_severe_with_vote_below_entry_votes():
    fred, prices = _make_baseline()
    date = fred.index[-1]

    prices.loc[date, "MOVE"] = 130.0
    fred.iloc[-5:, fred.columns.get_loc("DGS10")] = 3.30

    severe, risk = _severe_risk_snapshot(prices, fred, date, vol_signal="VIX or MOVE")

    assert risk["Risk Votes"] < RISK_CONFIG.entry_votes
    assert severe is True
    assert risk["MOVE"] >= RISK_CONFIG.move_level_crisis
    assert risk["MOVE Percentile"] >= RISK_CONFIG.move_percentile_crisis
    assert risk["10Y 3M Change"] >= 0.15


# ---------------------------------------------------------------------------
# 3. C2 / C3 override independently, and C4's structural invariant
# ---------------------------------------------------------------------------
def test_c2_override_alone_triggers_severe():
    fred, prices = _make_baseline()
    date = fred.index[-1]
    fred.iloc[-10:, fred.columns.get_loc("BAMLH0A0HYM2")] = 3.95  # +0.95, below Credit pillar's 1.00

    severe, risk = _severe_risk_snapshot(prices, fred, date, vol_signal="VIX or MOVE")

    assert risk["Risk Votes"] == 0
    assert severe is True
    assert risk["HY OAS 3M"] >= 0.90


def test_c3_override_alone_triggers_severe():
    fred, prices = _make_baseline()
    date = fred.index[-1]
    fred.iloc[-3:, fred.columns.get_loc("NFCI")] = 0.40  # in [0.35, 0.50): C3 only, not Liquidity pillar

    severe, risk = _severe_risk_snapshot(prices, fred, date, vol_signal="VIX or MOVE")

    assert risk["Risk Votes"] == 0
    assert severe is True
    assert risk["NFCI"] >= 0.35


def test_c4_structurally_implies_c2_and_volatility_extreme():
    """C4 (hy_3m>=1.25 and volatility_extreme) can never fire in isolation: hy_3m>=1.25
    always satisfies C2's hy_3m>=0.90, and reaching volatility_extreme requires either
    vix_extreme (identical to the Equity Vol pillar's own condition) or move_extreme
    (one of the Rates pillar's own OR-branches). This mirrors what the real historical
    data showed: C4 co-occurred with C2 on 12/12 of its activation days. This test
    verifies the invariant directly instead of attempting an impossible "C4 alone" case.
    """
    fred, prices = _make_baseline()
    date = fred.index[-1]
    fred.iloc[-10:, fred.columns.get_loc("BAMLH0A0HYM2")] = 4.30  # hy_3m ~= 1.30 >= 1.25
    prices.loc[date, "MOVE"] = 130.0  # drives volatility_extreme via move_extreme

    severe, risk = _severe_risk_snapshot(prices, fred, date, vol_signal="VIX or MOVE")

    hy_3m = risk["HY OAS 3M"]
    assert hy_3m >= 1.25, "fixture must actually reach the C4 threshold"
    assert hy_3m >= 0.90, "C4 implies C2: hy_3m>=1.25 always satisfies hy_3m>=0.90"

    move, move_pct = risk["MOVE"], risk["MOVE Percentile"]
    move_extreme = bool(np.isfinite(move) and move >= RISK_CONFIG.move_level_crisis
                         and np.isfinite(move_pct) and move_pct >= RISK_CONFIG.move_percentile_warning)
    vix_extreme = risk["VIX Percentile"] >= RISK_CONFIG.vix_percentile_crisis
    assert move_extreme or vix_extreme, "C4 implies volatility_extreme"
    assert severe is True


# ---------------------------------------------------------------------------
# 4. Neither vote_count nor any override -> severe is False
# ---------------------------------------------------------------------------
def test_calm_baseline_is_not_severe():
    fred, prices = _make_baseline()
    date = fred.index[-1]

    severe, risk = _severe_risk_snapshot(prices, fred, date, vol_signal="VIX or MOVE")

    assert risk["Risk Votes"] < RISK_CONFIG.entry_votes
    assert risk["HY OAS 3M"] < 0.90
    assert risk["NFCI"] < 0.35
    assert severe is False


# ---------------------------------------------------------------------------
# 5. minimum_hold_days=20, verified by trading-day index position (not calendar days)
# ---------------------------------------------------------------------------
def test_minimum_hold_days_enforced_by_trading_day_index():
    fred, prices = _make_baseline(n_days=900, end="2024-06-30", oscillate=False)
    common_start = fred.index[500]
    dates = _trading_dates(prices, fred, ETF_TICKERS, common_start)
    spike_pos = 50
    spike_date = dates[spike_pos]
    fred.loc[spike_date, "BAMLH0A0HYM2"] = 3.95  # single-day C2 spike, clean recovery after

    result = run_alpha_replica_backtest(prices, fred, common_start, 5.0, "VIX or MOVE", tickers=ETF_TICKERS)
    trades = result["trades"]

    entry_date = trades[trades["Reason"] == "Severe Risk → Cash"].index[0]
    exit_date = trades[trades["Reason"] == "Risk Clear → Re-enter"].index[0]
    entry_pos = dates.get_loc(entry_date)
    exit_pos = dates.get_loc(exit_date)

    assert entry_pos == spike_pos
    assert exit_pos - entry_pos == RISK_CONFIG.minimum_hold_days


# ---------------------------------------------------------------------------
# 6. exit_clear_days=5 consecutive clear trading days, with a mid-hold severe
#    recurrence that resets the clear-day count
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def clear_days_scenario():
    fred, prices = _make_baseline(n_days=900, end="2024-06-30", oscillate=False)
    common_start = fred.index[500]
    dates = _trading_dates(prices, fred, ETF_TICKERS, common_start)
    spike_pos = 50
    spike_date = dates[spike_pos]
    # A second, single-day recurrence at +16 trading days: still inside the 20-day
    # hold window, so it never opens a new "Severe Risk -> Cash" trade (already in
    # cash) -- but it resets clear_days back to 0 that day.
    retrigger_date = dates[spike_pos + 16]
    fred.loc[spike_date, "BAMLH0A0HYM2"] = 3.95
    fred.loc[retrigger_date, "BAMLH0A0HYM2"] = 3.95

    result = run_alpha_replica_backtest(prices, fred, common_start, 5.0, "VIX or MOVE", tickers=ETF_TICKERS)
    trades = result["trades"]
    entry_date = trades[trades["Reason"] == "Severe Risk → Cash"].index[0]
    exit_date = trades[trades["Reason"] == "Risk Clear → Re-enter"].index[0]
    return {
        "spike_pos": spike_pos,
        "entry_pos": dates.get_loc(entry_date),
        "exit_pos": dates.get_loc(exit_date),
        "trades": trades,
    }


def test_no_reentry_before_minimum_hold_days(clear_days_scenario):
    s = clear_days_scenario
    assert s["exit_pos"] >= s["spike_pos"] + RISK_CONFIG.minimum_hold_days


def test_no_reentry_with_only_4_consecutive_clear_days(clear_days_scenario):
    s = clear_days_scenario
    # Because of the +16 recurrence, clear_days is only 4 (not 5) at the hold
    # boundary (spike_pos + 20); re-entry must not happen exactly there.
    boundary_pos = s["spike_pos"] + RISK_CONFIG.minimum_hold_days
    assert s["exit_pos"] != boundary_pos


def test_reentry_on_5th_consecutive_clear_trading_day(clear_days_scenario):
    s = clear_days_scenario
    assert s["exit_pos"] == s["spike_pos"] + RISK_CONFIG.minimum_hold_days + 1


def test_severe_recurrence_resets_clear_day_count(clear_days_scenario):
    s = clear_days_scenario
    # Without the +16 recurrence, exit happens at spike_pos+20 exactly (see
    # test_minimum_hold_days_enforced_by_trading_day_index). With it, exit is
    # pushed one trading day later to +21 -- proof the recurrence reset clear_days
    # rather than being ignored. Also confirm it did not open a second cash-entry
    # trade (the state machine only logs an entry on the *first* transition into cash).
    assert s["exit_pos"] == s["spike_pos"] + RISK_CONFIG.minimum_hold_days + 1
    cash_entries = s["trades"][s["trades"]["Reason"] == "Severe Risk → Cash"]
    assert len(cash_entries) == 1


# ---------------------------------------------------------------------------
# 7. Real bundled-data baseline: metrics within tolerance, plus structural checks
# ---------------------------------------------------------------------------
BASELINE_TOLERANCE_ABS = 1e-4

EXPECTED_METRICS = {
    "CAGR": 0.0332854982360149,
    "Ann. Vol": 0.020097662337606764,
    "Sharpe": 0.02038465416144357,
    "Sortino": 0.0265038539940607,
    "Max Drawdown": -0.017089955617956343,
    "Calmar": 1.9476644047595986,
    "Hit Rate": 0.6521739130434783,
    "Excess t-stat": 0.04888068350963944,
}
EXPECTED_START = pd.Timestamp("2020-10-31")
EXPECTED_END = pd.Timestamp("2026-06-30")
EXPECTED_N = 69
REQUIRED_KEYS = {"monthly_returns", "weights", "turnover", "signals", "trades", "overlay"}


def _load_real_bundled_data():
    fred = pd.read_csv(os.path.join(REPO_ROOT, "data", "fred_snapshot.csv"), index_col=0, parse_dates=True).sort_index()
    prices = pd.read_csv(
        os.path.join(REPO_ROOT, "data", "etf_total_return_snapshot.csv"), index_col=0, parse_dates=True
    ).sort_index().ffill()
    return fred, prices


def _real_data_common_start(prices: pd.DataFrame, fred: pd.DataFrame) -> pd.Timestamp:
    available = [t for t in STRUCTURED_PROXY_TICKERS if t in prices and prices[t].dropna().shape[0] >= 252]
    cash_available = [t for t in CASH_PROXY_TICKERS if t in prices and prices[t].dropna().shape[0] >= 252]
    universe = ETF_TICKERS + available + cash_available
    required_fred = [
        "DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y", "DFII10", "T10YIE",
        "BAMLC0A0CM", "BAMLH0A0HYM2", "CPILFESL", "INDPRO", "UNRATE", "VIXCLS", "MORTGAGE30US", "NFCI",
    ]
    starts = [_first_valid(prices[t]) for t in universe]
    starts += [_first_valid(fred[c]) for c in required_fred if c in fred]
    return max(d for d in starts if d is not None)


def _real_bond_returns(prices: pd.DataFrame) -> pd.Series:
    series = prices["BOND"].dropna()
    monthly = series.resample("ME").last().ffill()
    if len(monthly) and monthly.index[-1] > series.index.max().normalize():
        monthly = monthly.iloc[:-1]
    r = monthly.pct_change()
    r.name = "PIMCO Active Bond (BOND)"
    return r


@pytest.fixture(scope="module")
def real_bundled_result():
    fred, prices = _load_real_bundled_data()
    common_start = _real_data_common_start(prices, fred)
    bond_returns = _real_bond_returns(prices)
    return run_bond_core_overlay_backtest(prices, fred, bond_returns, common_start, 5.0, "VIX or MOVE")


def test_baseline_required_keys_present(real_bundled_result):
    assert REQUIRED_KEYS.issubset(real_bundled_result.keys())


def test_baseline_returns_have_no_nan(real_bundled_result):
    r = real_bundled_result["monthly_returns"].dropna()
    assert len(r) > 0
    assert not r.isna().any()


def test_baseline_date_range_and_row_count(real_bundled_result):
    r = real_bundled_result["monthly_returns"].dropna()
    assert r.index.min() == EXPECTED_START
    assert r.index.max() == EXPECTED_END
    assert len(r) == EXPECTED_N


def test_baseline_metrics_within_tolerance(real_bundled_result):
    r = real_bundled_result["monthly_returns"].dropna()
    fred, _ = _load_real_bundled_data()
    rf = fred["DGS3MO"].resample("ME").last().reindex(r.index, method="ffill") / 100 / 12
    metrics = performance_metrics(r, rf)
    for key, expected in EXPECTED_METRICS.items():
        assert metrics[key] == pytest.approx(expected, abs=BASELINE_TOLERANCE_ABS), (
            f"{key}: got {metrics[key]}, expected {expected} (+/-{BASELINE_TOLERANCE_ABS})"
        )
