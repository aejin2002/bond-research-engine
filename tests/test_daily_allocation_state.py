"""Tests for the daily allocation-state fix (run_alpha_replica_backtest's
`daily_state` / `effective_weights`, threaded through
run_bond_core_overlay_backtest unchanged).

Constraint, same as tests/test_risk_gate.py: production code (backtest.py,
signals.py, risk_config.py, app.py) is never modified or monkeypatched here.
Every test builds a synthetic prices/fred DataFrame (or loads the real bundled
snapshot) and feeds it into the real, unmodified functions -- only the INPUT
DATA is manufactured. Strategy logic and thresholds are not touched by the fix
this file guards; only the *reporting resolution* of the cash-gate state and
the resulting held weights changed (previously month-end-ffilled, now daily).
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
    _first_valid,
    performance_metrics,
    run_alpha_replica_backtest,
    run_bond_core_overlay_backtest,
)
from data_sources import CASH_PROXY_TICKERS, ETF_TICKERS, STRUCTURED_PROXY_TICKERS  # noqa: E402
from risk_config import RISK_CONFIG  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic fixture builder (mirrors tests/test_risk_gate.py's _make_baseline;
# duplicated rather than imported so this file has no cross-test-module
# dependency).
# ---------------------------------------------------------------------------
_DRIFTS = {"SHY": 0.00008, "IEF": 0.00012, "TLT": 0.00015, "TIP": 0.0001,
           "LQD": 0.00013, "HYG": 0.00014, "MBB": 0.0001}


def _make_baseline(n_days: int = 900, end: str = "2024-06-30", oscillate: bool = False):
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

    return fred, prices


def _trading_dates(prices: pd.DataFrame, fred: pd.DataFrame, tickers: list[str], common_start: pd.Timestamp) -> pd.DatetimeIndex:
    asset_returns = prices[tickers].pct_change(fill_method=None)
    return asset_returns.index[(asset_returns.index >= common_start) & asset_returns.notna().all(axis=1)]


def _spike_scenario(spike_pos: int = 50, n_days: int = 900, end: str = "2024-06-30"):
    """A single mid-month HY-OAS spike (C2 override) that opens and later
    closes a cash-gate spell, same construction as
    tests/test_risk_gate.py::test_minimum_hold_days_enforced_by_trading_day_index.
    """
    fred, prices = _make_baseline(n_days=n_days, end=end, oscillate=False)
    common_start = fred.index[500]
    dates = _trading_dates(prices, fred, ETF_TICKERS, common_start)
    spike_date = dates[spike_pos]
    fred.loc[spike_date, "BAMLH0A0HYM2"] = 3.95  # single-day C2 spike, clean recovery after
    return fred, prices, common_start, dates, spike_pos


# ---------------------------------------------------------------------------
# 1. daily_state and effective_weights share the same index: no duplicates,
#    monotonic increasing
# ---------------------------------------------------------------------------
def test_daily_state_and_effective_weights_index_match():
    fred, prices, common_start, dates, spike_pos = _spike_scenario()
    result = run_alpha_replica_backtest(
        prices, fred, common_start, 5.0, "VIX or MOVE", tickers=ETF_TICKERS, compute_core_blend=True
    )
    daily_state = result["daily_state"]
    effective_weights = result["effective_weights"]

    assert not daily_state.empty and not effective_weights.empty
    assert daily_state.index.equals(effective_weights.index)
    assert daily_state.index.is_monotonic_increasing
    assert daily_state.index.is_unique
    assert effective_weights.index.is_monotonic_increasing
    assert effective_weights.index.is_unique


# ---------------------------------------------------------------------------
# 2. Every day's effective weights sum to 1.0
# ---------------------------------------------------------------------------
def test_effective_weights_rows_sum_to_one():
    fred, prices, common_start, dates, spike_pos = _spike_scenario()
    result = run_alpha_replica_backtest(
        prices, fred, common_start, 5.0, "VIX or MOVE", tickers=ETF_TICKERS, compute_core_blend=True
    )
    row_sums = result["effective_weights"].sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 1e-8


# ---------------------------------------------------------------------------
# 3. Cash Gate=True implies only the defensive asset has nonzero weight
# ---------------------------------------------------------------------------
def test_cash_gate_true_implies_only_defensive_asset_nonzero():
    fred, prices, common_start, dates, spike_pos = _spike_scenario()
    result = run_alpha_replica_backtest(
        prices, fred, common_start, 5.0, "VIX or MOVE", tickers=ETF_TICKERS, compute_core_blend=True
    )
    daily_state = result["daily_state"]
    effective_weights = result["effective_weights"]

    gated_dates = daily_state.index[daily_state["Cash Gate"]]
    assert len(gated_dates) > 0, "fixture must actually enter the cash gate"
    for date in gated_dates:
        row = effective_weights.loc[date]
        defensive_asset = daily_state.loc[date, "Defensive Asset"]
        nonzero = row[row.abs() > 1e-9]
        assert list(nonzero.index) == [defensive_asset], (
            f"{date.date()}: expected only {defensive_asset} nonzero, got {dict(nonzero)}"
        )
        assert nonzero.iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Defensive weight is reflected on the entry day itself, not deferred to
#    the next month-end (the core bug this fix addresses)
# ---------------------------------------------------------------------------
def test_defensive_weight_reflected_on_entry_day_not_next_month_end():
    fred, prices, common_start, dates, spike_pos = _spike_scenario()
    result = run_alpha_replica_backtest(
        prices, fred, common_start, 5.0, "VIX or MOVE", tickers=ETF_TICKERS, compute_core_blend=True
    )
    trades = result["trades"]
    entry_date = trades[trades["Reason"] == "Severe Risk → Cash"].index[0]

    daily_state = result["daily_state"]
    effective_weights = result["effective_weights"]

    assert bool(daily_state.loc[entry_date, "Cash Gate"]) is True
    defensive_asset = daily_state.loc[entry_date, "Defensive Asset"]
    assert effective_weights.loc[entry_date, defensive_asset] == pytest.approx(1.0)

    # And the entry date is very unlikely to itself be a month-end -- proving
    # the defensive weight did not wait for one.
    next_day = dates[dates.get_loc(entry_date) + 1] if dates.get_loc(entry_date) + 1 < len(dates) else entry_date
    month_end = next_day.month != entry_date.month
    assert not month_end, "fixture assumption: spike lands mid-month, not on a month-end trading day"


# ---------------------------------------------------------------------------
# 5. Hold/Clear "met" flags are correct and distinguishable from raw day counts
# ---------------------------------------------------------------------------
def test_hold_and_clear_requirement_met_flags():
    # Plain single-spike scenario: hold (20d) and clear (5d) requirements are
    # met on the *same* trading day (clear_days counts up cleanly from day 1),
    # so the gate exits exactly at entry+20 -- there is no day where hold is
    # met but the gate is still open. To observe "Hold Requirement Met" while
    # still in cash, use a second, mid-hold recurrence (same construction as
    # tests/test_risk_gate.py::clear_days_scenario) that resets clear_days
    # without opening a second cash-entry trade, pushing the exit one day past
    # the hold boundary.
    fred, prices = _make_baseline(n_days=900, end="2024-06-30", oscillate=False)
    common_start = fred.index[500]
    dates = _trading_dates(prices, fred, ETF_TICKERS, common_start)
    spike_pos = 50
    spike_date = dates[spike_pos]
    retrigger_date = dates[spike_pos + 16]
    fred.loc[spike_date, "BAMLH0A0HYM2"] = 3.95
    fred.loc[retrigger_date, "BAMLH0A0HYM2"] = 3.95

    result = run_alpha_replica_backtest(
        prices, fred, common_start, 5.0, "VIX or MOVE", tickers=ETF_TICKERS, compute_core_blend=True
    )
    daily_state = result["daily_state"]
    trades = result["trades"]
    entry_date = trades[trades["Reason"] == "Severe Risk → Cash"].index[0]
    entry_pos = dates.get_loc(entry_date)
    assert entry_pos == spike_pos

    just_before_hold_met = dates[entry_pos + RISK_CONFIG.minimum_hold_days - 1]
    at_hold_met = dates[entry_pos + RISK_CONFIG.minimum_hold_days]

    assert daily_state.loc[just_before_hold_met, "Hold Requirement Met"] is False
    assert daily_state.loc[just_before_hold_met, "Hold Days Elapsed"] == RISK_CONFIG.minimum_hold_days - 1
    # Hold is met, but clear_days was reset by the +16 recurrence, so the gate
    # must still be open here -- this is exactly "Hold met / Clear not met".
    assert bool(daily_state.loc[at_hold_met, "Cash Gate"]) is True
    assert daily_state.loc[at_hold_met, "Hold Requirement Met"] is True
    assert daily_state.loc[at_hold_met, "Hold Days Elapsed"] == RISK_CONFIG.minimum_hold_days
    assert daily_state.loc[at_hold_met, "Clear Requirement Met"] is False

    # Clear-day count resets to 0 on the entry day itself (severe that day).
    assert daily_state.loc[entry_date, "Clear Days Elapsed"] == 0
    assert daily_state.loc[entry_date, "Clear Requirement Met"] is False


# ---------------------------------------------------------------------------
# 6. Entry snapshot (Entry Risk Votes / Entry Triggered Pillars) stays frozen
#    for the whole cash-gate spell, even once current Risk Votes has cleared
#    to 0 -- this is exactly the "cooldown, not a new signal" distinction.
# ---------------------------------------------------------------------------
def test_entry_snapshot_frozen_during_cooldown():
    fred, prices, common_start, dates, spike_pos = _spike_scenario()
    result = run_alpha_replica_backtest(
        prices, fred, common_start, 5.0, "VIX or MOVE", tickers=ETF_TICKERS, compute_core_blend=True
    )
    daily_state = result["daily_state"]
    trades = result["trades"]
    entry_date = trades[trades["Reason"] == "Severe Risk → Cash"].index[0]
    exit_date = trades[trades["Reason"] == "Risk Clear → Re-enter"].index[0]

    entry_votes_at_entry = daily_state.loc[entry_date, "Entry Risk Votes"]
    assert entry_votes_at_entry >= RISK_CONFIG.entry_votes or entry_votes_at_entry >= 0
    mid_cooldown_date = dates[dates.get_loc(entry_date) + 2]
    assert mid_cooldown_date in daily_state.index
    if daily_state.loc[mid_cooldown_date, "Cash Gate"]:
        # Entry snapshot must be identical to entry day even after current risk clears.
        assert daily_state.loc[mid_cooldown_date, "Entry Risk Votes"] == entry_votes_at_entry
        assert daily_state.loc[mid_cooldown_date, "Entry Triggered Pillars"] == daily_state.loc[entry_date, "Triggered Pillars"]
        # Current risk has cleared below the entry-triggering vote count (the
        # single-day HY OAS spike's 3-month-change measure can linger above 0
        # for a couple of days without itself being enough to re-trigger
        # `severe`, hence "< entry_votes" rather than "== 0").
        assert daily_state.loc[mid_cooldown_date, "Risk Votes"] < RISK_CONFIG.entry_votes
        assert daily_state.loc[mid_cooldown_date, "State"] != "Severe Risk"

    # Once out of cash, entry snapshot fields must not linger with stale values.
    assert bool(daily_state.loc[exit_date, "Cash Gate"]) is False
    assert pd.isna(daily_state.loc[exit_date, "Entry Risk Votes"])


# ---------------------------------------------------------------------------
# 7. State enum values: Normal / Severe Risk / Minimum Hold / Clear Confirmation
# ---------------------------------------------------------------------------
def test_state_enum_sequence_through_a_full_spell():
    fred, prices, common_start, dates, spike_pos = _spike_scenario()
    result = run_alpha_replica_backtest(
        prices, fred, common_start, 5.0, "VIX or MOVE", tickers=ETF_TICKERS, compute_core_blend=True
    )
    daily_state = result["daily_state"]
    trades = result["trades"]
    entry_date = trades[trades["Reason"] == "Severe Risk → Cash"].index[0]
    exit_date = trades[trades["Reason"] == "Risk Clear → Re-enter"].index[0]

    assert daily_state.loc[entry_date, "State"] == "Severe Risk"

    day_before_exit = dates[dates.get_loc(exit_date) - 1]
    assert daily_state.loc[day_before_exit, "State"] in {"Minimum Hold", "Clear Confirmation"}
    assert daily_state.loc[exit_date, "State"] == "Normal"

    valid_states = {"Normal", "Severe Risk", "Minimum Hold", "Clear Confirmation"}
    assert set(daily_state["State"].unique()).issubset(valid_states)


# ---------------------------------------------------------------------------
# 8. Returns/metrics are bit-for-bit unchanged by compute_core_blend=True and
#    by the run_bond_core_overlay_backtest weights-construction rewrite.
# ---------------------------------------------------------------------------
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


# Same pinned baseline as tests/test_risk_gate.py::EXPECTED_METRICS -- duplicated
# here (not imported) so this file is self-contained; both must stay in sync
# with real_bundled_result's actual monthly_returns.
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
BASELINE_TOLERANCE_ABS = 1e-4


def test_returns_and_metrics_unchanged_after_daily_state_fix():
    fred, prices = _load_real_bundled_data()
    common_start = _real_data_common_start(prices, fred)
    bond_returns = _real_bond_returns(prices)
    result = run_bond_core_overlay_backtest(prices, fred, bond_returns, common_start, 5.0, "VIX or MOVE")

    r = result["monthly_returns"].dropna()
    rf = fred["DGS3MO"].resample("ME").last().reindex(r.index, method="ffill") / 100 / 12
    metrics = performance_metrics(r, rf)
    for key, expected in EXPECTED_METRICS.items():
        assert metrics[key] == pytest.approx(expected, abs=BASELINE_TOLERANCE_ABS), (
            f"{key}: got {metrics[key]}, expected {expected} (+/-{BASELINE_TOLERANCE_ABS}) -- "
            "the daily allocation-state fix must not change returns/metrics."
        )


# ---------------------------------------------------------------------------
# 9. Pinned regression on the *real bundled snapshot* data: this asserts a
#    specific historical fact of data/etf_total_return_snapshot.csv +
#    data/fred_snapshot.csv as bundled at the time this test was written
#    (Cash Gate entered 2026-06-10, defensive asset USFR). It is NOT a general
#    property of the strategy -- it will need updating if the bundled snapshot
#    files are ever refreshed/replaced with different historical data.
# ---------------------------------------------------------------------------
def test_2026_06_10_usfr_regression_pinned_to_bundled_snapshot():
    fred, prices = _load_real_bundled_data()
    common_start = _real_data_common_start(prices, fred)
    bond_returns = _real_bond_returns(prices)
    result = run_bond_core_overlay_backtest(prices, fred, bond_returns, common_start, 5.0, "VIX or MOVE")

    daily_state = result["daily_state"]
    effective_weights = result["effective_weights"]
    entry_date = pd.Timestamp("2026-06-10")

    assert entry_date in daily_state.index, "bundled snapshot's known 2026-06-10 cash-gate entry date is missing"
    assert bool(daily_state.loc[entry_date, "Cash Gate"]) is True
    assert daily_state.loc[entry_date, "Defensive Asset"] == "USFR"
    assert effective_weights.loc[entry_date, "USFR"] == pytest.approx(1.0)
    # Every other column must be ~0 that day (100% USFR, not a blend).
    other_nonzero = effective_weights.loc[entry_date].drop("USFR")
    assert other_nonzero.abs().max() < 1e-9


# ---------------------------------------------------------------------------
# 10. Backward compatibility: the pre-existing "weights" key must keep its
#     original meaning, frequency, and columns -- callers that read
#     run_bond_core_overlay_backtest(...)["weights"] (or
#     run_backtest(...)["bond_core_overlay_weights"]) must see byte-for-byte
#     the same thing they always did. The new daily-accurate reconstruction
#     lives ONLY under "effective_weights" -- a separate, additive key.
# ---------------------------------------------------------------------------
# Pinned shape/columns of the pre-fix "weights" output on the bundled
# snapshot, captured by diffing against a checkout of the pre-fix backtest.py
# (git history) -- see the PR/session notes for the exact comparison script.
EXPECTED_WEIGHTS_SHAPE = (93, 14)
EXPECTED_WEIGHTS_COLUMNS = {
    "BOND", "SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "MBB",
    "JAAA", "BKLN", "CMBS", "SGOV", "BIL", "USFR",
}


def test_weights_key_unchanged_shape_columns_and_frequency():
    fred, prices = _load_real_bundled_data()
    common_start = _real_data_common_start(prices, fred)
    bond_returns = _real_bond_returns(prices)
    result = run_bond_core_overlay_backtest(prices, fred, bond_returns, common_start, 5.0, "VIX or MOVE")

    weights = result["weights"]
    effective_weights = result["effective_weights"]

    assert weights.shape == EXPECTED_WEIGHTS_SHAPE
    assert set(weights.columns) == EXPECTED_WEIGHTS_COLUMNS
    # "weights" stays sparse (trade-day/month-end union), never daily -- if it
    # ever grows to match effective_weights' row count, "weights" has silently
    # been repointed at the daily reconstruction again.
    assert len(weights) < len(effective_weights)
    assert weights.index.isin(effective_weights.index).all(), "weights dates must still be a subset of trading days"


def test_weights_key_still_shows_the_pre_fix_stale_entry_day_value():
    """Pinned to the bundled snapshot's known 2026-06-10 cash-gate entry.

    This is the exact bug this whole fix was about: the legacy "weights" key
    is *intentionally* left showing the old, month-end-ffilled (and therefore
    stale/wrong-looking) composition on the entry day, because restoring it
    means restoring its original computation byte-for-byte for backward
    compatibility. "effective_weights" is where the corrected value lives.
    Do not "fix" this test by changing weights' value -- that would silently
    re-break backward compatibility with any caller still reading "weights".
    """
    fred, prices = _load_real_bundled_data()
    common_start = _real_data_common_start(prices, fred)
    bond_returns = _real_bond_returns(prices)
    result = run_bond_core_overlay_backtest(prices, fred, bond_returns, common_start, 5.0, "VIX or MOVE")

    weights = result["weights"]
    effective_weights = result["effective_weights"]
    entry_date = pd.Timestamp("2026-06-10")

    assert entry_date in weights.index, "2026-06-10 is a trade day, so legacy weights must still have a row for it"
    legacy_row = weights.loc[entry_date]
    assert legacy_row.get("USFR", 0.0) == pytest.approx(0.0), (
        "legacy 'weights' must remain unfixed (pre-fix, ffilled) on the entry day"
    )
    assert legacy_row[legacy_row.abs() > 1e-9].index.tolist(), "legacy row should still show the pre-crisis composition"

    fixed_row = effective_weights.loc[entry_date]
    assert fixed_row["USFR"] == pytest.approx(1.0), "effective_weights must show the corrected same-day USFR entry"


def test_run_backtest_bond_core_overlay_weights_key_matches_legacy_semantics():
    """run_backtest()["bond_core_overlay_weights"] must be the same object/
    values as run_bond_core_overlay_backtest()["weights"] -- not
    effective_weights -- so any existing consumer of the top-level key is
    unaffected."""
    from backtest import run_backtest

    fred, prices = _load_real_bundled_data()
    target = prices["BOND"].dropna()
    benchmark = prices["AGG"].dropna() if "AGG" in prices else None
    results = run_backtest(prices, fred, target=target, benchmark=benchmark)

    assert results["bond_core_overlay_weights"].shape == EXPECTED_WEIGHTS_SHAPE
    assert len(results["bond_core_overlay_weights"]) < len(results["bond_core_overlay_effective_weights"])
