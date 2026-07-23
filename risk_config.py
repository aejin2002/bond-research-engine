from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    """Shared risk gate settings used by rule scoring, backtests, and UI copy."""

    entry_votes: int = 2
    crisis_votes: int = 3
    minimum_hold_days: int = 20
    exit_clear_days: int = 5
    warning_exposure: float = 0.50
    crisis_exposure: float = 0.00

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
        f"Risk gate uses independent pillars; warning starts at {config.entry_votes}+ pillars, "
        f"crisis at {config.crisis_votes}+ pillars. Cash gate holds at least "
        f"{config.minimum_hold_days} trading days and requires {config.exit_clear_days} clear days."
    )
