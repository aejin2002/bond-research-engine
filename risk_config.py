from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    """Shared risk gate settings used by rule scoring, backtests, and UI copy."""

    entry_votes: int = 2
    crisis_votes: int = 3
    minimum_hold_days: int = 20
    exit_clear_days: int = 5

    hy_spread_3m_warning: float = 0.75
    hy_spread_3m_crisis: float = 1.00
    vix_percentile_warning: float = 0.80
    vix_percentile_crisis: float = 0.90
    move_level_crisis: float = 110.0
    move_percentile_warning: float = 0.85
    move_percentile_crisis: float = 0.90
    ten_year_3m_rate_shock: float = 0.25
    unemployment_gap_warning: float = 0.50
    unemployment_gap_crisis: float = 0.75
    nfci_warning: float = 0.20
    nfci_crisis: float = 0.50
    hyg_relative_1m_crisis: float = -0.03


RISK_CONFIG = RiskConfig()


def risk_config_summary(config: RiskConfig = RISK_CONFIG) -> str:
    return (
        f"Full defensive (cash) gate triggers when {config.entry_votes}+ of the 5 independent pillars "
        f"(Credit, Rates, Liquidity, Labor, Equity Vol) are active, or when a separate credit/liquidity "
        f"override (HY spread blowout, NFCI stress, or a MOVE-percentile + rate-shock combo) is met on its "
        f"own. Cash gate holds at least {config.minimum_hold_days} trading days and requires "
        f"{config.exit_clear_days} clear days before re-entering. Note: crisis_votes={config.crisis_votes} "
        f"is defined in RiskConfig but is not read by this gate — it drives a separate BOND-core-weight "
        f"escalation tier instead."
    )
