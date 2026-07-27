"""Shared factor metadata for signals.py (execution) and rule_library.py (documentation).

Only data that both sides need to agree on lives here: thresholds, applicable
ETFs, factor weight, whether a factor's positive signal is gated by Momentum,
and the threshold's origin (developer-chosen vs. literature). The actual
scoring branches (if/else logic) stay in signals.py; rule_library.py's
qualitative prose (limitations, PIMCO linkage, practitioner adoption) stays
hand-written.

Defensive's thresholds are read live from RISK_CONFIG rather than duplicated
here, since RISK_CONFIG is already the single source of truth for those values
(see risk_config.py) -- copying them into a second constant here would recreate
the exact drift problem this file exists to avoid.
"""
from __future__ import annotations

from dataclasses import dataclass

from risk_config import RISK_CONFIG


@dataclass(frozen=True)
class FactorConfig:
    name: str
    weight: float
    applicable_etfs: tuple[str, ...]
    momentum_gated: bool
    momentum_operator: str  # ">=" | ">" | "mixed" | "" (not gated)
    thresholds: dict[str, float]  # same units signals.py already computes in (percentage points, not bp)
    threshold_origin: str  # "developer_initial" | "self_disclosed_sample_dependent"
    origin_note: str = ""


ALL_ETFS = ("SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "MBB")

CARRY = FactorConfig(
    name="Carry",
    weight=0.20,
    applicable_etfs=ALL_ETFS,
    momentum_gated=True,
    momentum_operator=">=",
    thresholds={"positive": 0.75, "negative": 0.25},
    threshold_origin="developer_initial",
)

MOMENTUM = FactorConfig(
    name="Momentum",
    weight=0.20,
    applicable_etfs=ALL_ETFS,
    momentum_gated=False,
    momentum_operator="",
    thresholds={"lookback_short_m": 6, "lookback_long_m": 12},
    threshold_origin="developer_initial",
)

VALUE = FactorConfig(
    name="Value",
    weight=0.15,
    applicable_etfs=ALL_ETFS,
    momentum_gated=True,
    momentum_operator=">=",
    thresholds={"positive_pct": 0.80, "negative_pct": 0.20, "lookback_years": 5},
    threshold_origin="developer_initial",
)

REAL_YIELD = FactorConfig(
    name="Real Yield",
    weight=0.10,
    applicable_etfs=("TIP",),
    momentum_gated=True,
    momentum_operator=">=",
    thresholds={"positive": 2.0, "negative": 0.0},
    threshold_origin="self_disclosed_sample_dependent",
)

YIELD_CURVE = FactorConfig(
    name="Yield Curve",
    weight=0.10,
    applicable_etfs=("SHY", "IEF", "TLT"),
    momentum_gated=True,
    momentum_operator=">=",
    thresholds={"steep": 0.75, "inverted": -0.25, "unrate_gap": 0.30},
    threshold_origin="developer_initial",
)

CREDIT_SPREAD = FactorConfig(
    name="Credit Spread",
    weight=0.10,
    applicable_etfs=("LQD", "HYG"),
    momentum_gated=True,
    momentum_operator="mixed",
    thresholds={
        "lqd_negative": 0.90,
        "lqd_positive": 1.50,
        "hyg_negative": 3.00,
        "hyg_negative_3m_change": 1.00,
        "hyg_positive": 5.00,
    },
    threshold_origin="developer_initial",
)

DEFENSIVE = FactorConfig(
    name="Defensive",
    weight=0.05,
    applicable_etfs=("SHY", "IEF", "TLT", "LQD", "HYG"),
    momentum_gated=True,
    momentum_operator=">",
    thresholds={
        "hy_3m": RISK_CONFIG.hy_spread_3m_warning,
        "vix_pct": RISK_CONFIG.vix_percentile_warning,
        "unrate_gap": RISK_CONFIG.unemployment_gap_warning,
        "vote_threshold": RISK_CONFIG.entry_votes,
    },
    threshold_origin="developer_initial",
    origin_note="Values come from RISK_CONFIG (risk_config.py), not from this file.",
)

REGIME_OVERLAY = FactorConfig(
    name="Regime Overlay",
    weight=0.10,
    applicable_etfs=ALL_ETFS,
    momentum_gated=False,
    momentum_operator="",
    thresholds={"growth": 0.0, "inflation": 0.03},
    threshold_origin="developer_initial",
)

REGIME_PREFERENCES = {
    "Goldilocks": {"prefer": ("LQD", "HYG", "MBB", "IEF"), "avoid": ("TLT",)},
    "Reflation": {"prefer": ("TIP", "HYG", "SHY"), "avoid": ("TLT", "LQD")},
    "Stagflation": {"prefer": ("TIP", "SHY"), "avoid": ("TLT", "HYG", "LQD")},
    "Disinflation / Recession": {"prefer": ("IEF", "TLT", "SHY"), "avoid": ("HYG",)},
}

FACTORS = {
    cfg.name: cfg
    for cfg in [CARRY, MOMENTUM, VALUE, REAL_YIELD, YIELD_CURVE, CREDIT_SPREAD, DEFENSIVE, REGIME_OVERLAY]
}
