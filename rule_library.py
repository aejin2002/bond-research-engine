from __future__ import annotations

from signal_config import (
    CARRY,
    CREDIT_SPREAD,
    DEFENSIVE,
    FACTORS,
    MOMENTUM,
    REAL_YIELD,
    REGIME_OVERLAY,
    REGIME_PREFERENCES,
    VALUE,
    YIELD_CURVE,
)


def _bp(value: float) -> str:
    return f"{value * 100:.0f}bp"


def _pctile(value: float) -> str:
    return f"{value * 100:.0f}백분위"


def _momentum_gated_rule_names() -> list[str]:
    return [cfg.name for cfg in FACTORS.values() if cfg.momentum_gated]


def _carry_definition(cfg=CARRY) -> str:
    t = cfg.thresholds
    return (
        f"ETF 추정수익률−SHY 수익률이 {_bp(t['positive'])} 이상이고 Momentum≥0이면 +1, "
        f"{_bp(t['negative'])} 미만이면 -1. 국채는 roll-down을 추가 연구."
    )


def _carry_limits(cfg=CARRY) -> str:
    return (
        "금리 급등·스프레드 확대 시 carry가 자본손실에 압도된다. MBB는 단순 모기지금리로 정확한 옵션비용을 포착하기 어렵다. "
        f"{_bp(cfg.thresholds['positive'])}/{_bp(cfg.thresholds['negative'])} 임계값은 문헌에서 직접 도출된 수치가 아니라 "
        "개발자 초기값이다. 이 Rule의 +1 판정도 Momentum≥0을 함께 요구해 별도 Momentum 팩터의 신호를 재사용한다."
    )


def _momentum_definition(cfg=MOMENTUM) -> str:
    t = cfg.thresholds
    return (
        f"SHY 대비 {t['lookback_short_m']}개월·{t['lookback_long_m']}개월 총수익이 모두 양수면 +1, "
        "모두 음수면 -1, 그 외 0."
    )


def _momentum_limits() -> str:
    names = "/".join(_momentum_gated_rule_names())
    return (
        "급격한 반전에서 늦고 채권의 낮은 변동성 때문에 신호가 작다. 2022년처럼 롱온리 제약은 하락추세 수익화를 막는다. "
        f"또한 이 Rule은 단독 팩터로 그치지 않고 {names}의 +1 판정에도 Momentum≥0(HYG는 Momentum>0) 조건으로 다시 "
        "쓰이므로, 강한 모멘텀 하나가 여러 팩터 점수를 동시에 움직여 팩터 가중치(20%)가 시사하는 것보다 총점에 더 크게 "
        "기여할 수 있다. 6개월·12개월 조합 규칙 자체도 특정 논문에서 그대로 가져온 값이 아니라 개발자가 경험적으로 "
        "설정한 규칙이다."
    )


def _value_definition(cfg=VALUE) -> str:
    t = cfg.thresholds
    return (
        f"각 ETF 수익률 proxy가 과거 {t['lookback_years']}년 {_pctile(t['positive_pct'])} 이상이고 Momentum≥0, "
        f"물가 재가속이 아니면 +1; {_pctile(t['negative_pct'])} 이하면 -1."
    )


def _value_limits(cfg=VALUE) -> str:
    t = cfg.thresholds
    return (
        "높은 수익률이 구조적 부실·인플레이션 위험의 정당한 보상일 수 있다. yield proxy와 ETF 실제 YTM 사이 오차 존재. "
        f"{t['positive_pct'] * 100:.0f}/{t['negative_pct'] * 100:.0f} 백분위 임계값은 문헌 수치가 아니라 개발자가 채택한 "
        "관행적 quantile 컨벤션이다. 이 Rule의 +1 판정도 Momentum≥0을 함께 요구해 Momentum 팩터의 신호를 재사용한다."
    )


def _real_yield_definition(cfg=REAL_YIELD) -> str:
    t = cfg.thresholds
    return (
        f"10년 TIPS 실질금리≥{t['positive']:.0f}%이고 TIP Momentum≥0이면 TIP +1; "
        f"실질금리<{t['negative']:.0f}%이면 -1."
    )


def _real_yield_limits(cfg=REAL_YIELD) -> str:
    return (
        f"{cfg.thresholds['positive']:.0f}%는 표본의존 초기 임계값이다. breakeven에는 물가기대뿐 아니라 유동성과 "
        "inflation risk premium이 섞인다. 이 Rule의 +1 판정도 TIP Momentum≥0을 함께 요구해 Momentum 팩터의 신호를 "
        "재사용한다."
    )


def _yield_curve_definition(cfg=YIELD_CURVE) -> str:
    t = cfg.thresholds
    return (
        f"10Y−2Y>{_bp(t['steep'])}이면 IEF +1. {_bp(t['inverted'])} 미만이고 물가 둔화·실업 gap≥{_bp(t['unrate_gap'])}"
        f"이면 IEF/TLT 후보(Momentum 필터 적용); {_bp(t['inverted'])} 미만이면 SHY -1."
    )


def _yield_curve_limits(cfg=YIELD_CURVE) -> str:
    t = cfg.thresholds
    return (
        "곡선 역전은 기대단기금리와 기간프리미엄 원인이 다르다. 공급충격으로 장기물이 약해질 수 있다. "
        f"{_bp(t['steep'])}/{_bp(t['inverted'])}/실업 gap {_bp(t['unrate_gap'])} 임계값은 문헌에서 도출된 수치가 아니라 "
        "개발자 초기값이다. IEF/TLT 후보 판정도 Momentum≥0을 함께 요구해 Momentum 팩터의 신호를 재사용한다."
    )


def _credit_spread_definition(cfg=CREDIT_SPREAD) -> str:
    t = cfg.thresholds
    return (
        f"LQD: IG OAS<{_bp(t['lqd_negative'])}이면 -1; IG OAS>{_bp(t['lqd_positive'])}이고 Momentum≥0이면 +1"
        "(HYG와 달리 spread widening이 멈췄는지 조건은 없음). "
        f"HYG: HY OAS<{_bp(t['hyg_negative'])} 또는 HY OAS 3개월 변화≥{_bp(t['hyg_negative_3m_change'])}이면 -1; "
        f"HY OAS>{_bp(t['hyg_positive'])}이고 3개월 변화≤0(확대 멈춤)이며 Momentum>0이면 +1."
    )


def _credit_spread_limits(cfg=CREDIT_SPREAD) -> str:
    origin_values = "/".join(
        _bp(cfg.thresholds[key])
        for key in ["lqd_negative", "lqd_positive", "hyg_negative", "hyg_negative_3m_change", "hyg_positive"]
    )
    return (
        "OAS가 넓어도 부도손실이 더 크게 상승할 수 있다. 2026년 이후 ICE 일부 시계열의 공개 이력이 짧다. "
        f"{origin_values} 임계값은 문헌 수치가 아니라 개발자의 경험적 초기값이다. LQD와 HYG의 +1 판정도 각각 "
        "Momentum≥0/Momentum>0을 함께 요구해 Momentum 팩터의 신호를 재사용한다."
    )


def _defensive_definition(cfg=DEFENSIVE) -> str:
    t = cfg.thresholds
    return (
        f"HY OAS 3개월 +{_bp(t['hy_3m'])}, VIX 3년 {_pctile(t['vix_pct'])}, 실업률 12개월 저점 대비 +{_bp(t['unrate_gap'])} "
        f"중 {t['vote_threshold']:.0f}개 이상이면 Risk Off."
    )


def _defensive_limits(cfg=DEFENSIVE) -> str:
    t = cfg.thresholds
    origin = (
        f"{_bp(t['hy_3m'])}/{_pctile(t['vix_pct'])}/{_bp(t['unrate_gap'])} 임계값과 {t['vote_threshold']:.0f}표 문턱은 "
        "문헌 인용이 아니라 RISK_CONFIG의 개발자 초기값이다."
    )
    return (
        "신호가 빠르면 carry를 포기하고, 늦으면 drawdown을 이미 겪는다. 주식 VIX가 채권 스트레스를 완전히 대변하지 않는다. "
        f"{origin} IEF·TLT의 방어자산 인정도 Momentum>0을 요구해 Momentum 팩터의 신호를 재사용한다. 이 Rule은 "
        "signals.py의 Base Rule Engine(7개 ETF fallback 전략)에만 적용된다 — 대표 전략(Risk-First Bond Signal Engine)의 "
        "실제 risk gate는 backtest.py의 _severe_risk_snapshot에 별도로 구현되어 있으며, 5개 독립 pillar(Credit/Rates/"
        "Liquidity/Labor/Equity Vol, 이 Rule보다 높은 crisis급 임계값)와 별도의 신용·유동성 override, 최소 보유 20거래일·"
        "연속 clear 5거래일 규칙을 쓴다. 두 메커니즘은 pillar 구성과 임계값이 서로 다르다."
    )


def _regime_definition(cfg=REGIME_OVERLAY) -> str:
    t = cfg.thresholds
    return (
        f"산업생산 6개월 연율의 3개월 평균이 {t['growth']:.0%} 초과인지, 근원 CPI 3개월 연율의 3개월 평균이 "
        f"{t['inflation']:.0%} 초과인지로 네 국면을 구분."
    )


def _regime_limits(cfg=REGIME_OVERLAY) -> str:
    t = cfg.thresholds
    return (
        "거시 데이터는 수정되고 후행적이다. 현재 단순 nowcast는 최초 발표 vintage를 보존하지 않아 연구 편향 가능. "
        f"{t['growth']:.0%}/{t['inflation']:.0%} 임계값은 문헌에서 직접 도출된 수치가 아니라 개발자 초기값이다."
    )


def _regime_mapping(prefs=REGIME_PREFERENCES) -> str:
    label_map = {"Disinflation / Recession": "Disinflation"}
    parts = []
    for regime_name, sets in prefs.items():
        label = label_map.get(regime_name, regime_name)
        prefer = "/".join(sets["prefer"])
        avoid = "/".join(sets["avoid"])
        parts.append(f"{label}={prefer} +1, {avoid} -1")
    return ". ".join(parts) + "."


SOURCES = {
    "AQR styles": "https://www.aqr.com/insights/research/journal-article/style-investing-in-fixed-income-markets",
    "AQR timing": "https://www.aqr.com/Insights/Research/Journal-Article/How-Do-Factor-Premia-Vary-Over-Time-A-Century-of-Evidence",
    "PIMCO carry": "https://www.pimco.com/jp/ja/resources/education/bond-basic/fixed-income-1/what-is-carry-and-rolldown",
    "PIMCO income": "https://www.pimco.com/au/en/investment-strategies/income-strategies",
    "BlackRock systematic": "https://www.blackrock.com/uk/professionals/solutions/systematic-investing/systematic-fixed-income",
    "JPM active FI": "https://am.jpmorgan.com/sg/en/asset-management/institutional/insights/portfolio-insights/etf-perspectives/fixed-income-etfs/",
    "Vanguard curve": "https://advisors.vanguard.com/insights/article/series/portfolio-perspectives",
    "Research Affiliates": "https://www.researchaffiliates.com/insights/publications/articles/563-systematic-global-macro",
    "RA trend/carry": "https://www.researchaffiliates.com/publications/articles/1107-should-trend-follow-carry-lessons-from-bonds-gold-and-2022",
}


COMMON = {
    "투자 철학": "서로 다른 기대수익 원천을 규칙으로 분리하고 결합해 단일 전망과 재량 편향에 대한 의존을 줄인다. "
    f"단, Momentum은 별도 팩터(20% 가중)이면서 동시에 {'/'.join(_momentum_gated_rule_names())} "
    "규칙의 진입조건에도 공통 필터(Momentum≥0)로 재사용되므로, 8개 팩터가 통계적으로 완전히 독립은 아니다.",
    "투자 유니버스": "SHY, IEF, TLT, TIP, LQD, HYG, MBB의 미국 상장 롱온리 ETF.",
    "데이터": "ETF 배당재투자 총수익, FRED 금리·실질금리·OAS·물가·성장·실업·VIX. 발표시점 데이터 사용이 원칙.",
    "리밸런싱": "월말 신호 산출, 다음 거래기간 적용. 목표비중 변화 5%p 미만은 거래 생략 후보.",
    "롱/숏 구성": "숏 금지. +1은 비중 확대, 0은 중립, -1은 0% 또는 SHY 이동으로 표현.",
    "가중치": "팩터 가중합을 ETF 변동성으로 나눈 뒤 자산별 상한 적용. 잔여 위험예산은 SHY.",
    "거래비용": "기본 편도 회전율당 5bp. 2·5·10bp 민감도 분석 필요.",
    "리스크 관리": "Dollar neutral 또는 duration neutral이 아님. 롱온리, 완전투자, ETF 상한과 Defensive overlay 사용.",
    "성과지표": "CAGR, 변동성, Sharpe, Sortino, MDD, Calmar, t-stat, 회전율, PIMCO 대비 IR·상관·beta.",
    "Python 구현": "월별 point-in-time slice → Rule 점수 → 종합점수 → 위험조정 비중 → 익월수익 → 비용 차감.",
}


def rule(name, definition, limits, mapping, practice, pimco, sources, **overrides):
    item = {"Rule": name, **COMMON}
    item.update(
        {
            "팩터 정의": definition,
            "한계점": limits,
            "ETF 근사 가능성": mapping,
            "실무 활용도": practice,
            "PIMCO와의 연결": pimco,
            "근거 자료": sources,
        }
    )
    item.update(overrides)
    return item


RULES = [
    rule(
        "Carry",
        _carry_definition(),
        _carry_limits(),
        "IEF/TLT의 curve carry, LQD/HYG의 spread carry, MBB의 income carry. SHY는 funding proxy.",
        "매우 높음. 인컴·롤다운·섹터 로테이션은 전통적 채권운용의 핵심이지만 보통 개별채권 수준에서 구현된다.",
        "강하게 관찰. PIMCO는 carry와 roll-down을 명시하며 BOND의 일별 holdings에서 MBS·credit·Treasury 비중 변화를 확인할 수 있다.",
        ["PIMCO carry", "PIMCO income", "AQR styles", "Vanguard curve"],
    ),
    rule(
        "Momentum",
        _momentum_definition(),
        _momentum_limits(),
        "모든 ETF에 동일 산식. 음의 신호는 숏 대신 SHY 이동.",
        "중간~높음. 단독보다는 value·carry·quality와 결합하는 systematic process에서 활용.",
        "직접적인 PIMCO 공시 신호로는 제한적. 포지션 변화가 가격추세와 같은 방향인지 사후 검증 가능.",
        ["AQR styles", "AQR timing", "Research Affiliates", "RA trend/carry"],
    ),
    rule(
        "Value",
        _value_definition(),
        _value_limits(),
        "Treasury maturity yield, TIP real yield, LQD/HYG OAS 포함 수익률, MBB mortgage-rate proxy.",
        "높음. 액티브 운용사는 지수의 부채가중 왜곡을 피하고 저평가·고품질 채권을 선별한다고 설명한다.",
        "관찰 가능. PIMCO의 섹터별 overweight가 당시 spread/yield valuation과 일치하는지 비교.",
        ["AQR styles", "JPM active FI", "BlackRock systematic"],
    ),
    rule(
        "Real Yield",
        _real_yield_definition(),
        _real_yield_limits(),
        "TIP으로 직접 근사. 명목채 대비 TIP 상대배분으로 표현.",
        "높음. 실질수익률과 물가보호는 real-return 및 자산배분 운용의 표준 입력.",
        "부분 관찰. BOND의 물가연동채·effective duration 변화와 실질금리 수준을 연결해 검증.",
        ["PIMCO income", "Vanguard curve"],
    ),
    rule(
        "Yield Curve",
        _yield_curve_definition(),
        _yield_curve_limits(),
        "SHY·IEF·TLT 사이 duration rotation.",
        "매우 높음. duration과 curve positioning은 모든 글로벌 채권 운용사의 핵심 의사결정.",
        "강하게 관찰 가능. PIMCO 공시 effective duration과 만기구간 배분 변화를 규칙 신호와 비교.",
        ["PIMCO carry", "Vanguard curve", "PIMCO income"],
    ),
    rule(
        "Credit Spread",
        _credit_spread_definition(),
        _credit_spread_limits(),
        "LQD=IG beta, HYG=HY beta, MBB=securitized 대체. 부정 신호는 Treasury/SHY 이동.",
        "매우 높음. 섹터·품질·개별 신용선택은 액티브 채권의 대표적 alpha source.",
        "강하게 관찰. BOND의 회사채 품질·섹터 비중 변화와 OAS 국면을 비교.",
        ["PIMCO income", "JPM active FI", "BlackRock systematic"],
    ),
    rule(
        "Defensive",
        _defensive_definition(),
        _defensive_limits(),
        "HYG 0%, LQD 제한, SHY 우선. TLT는 Momentum>0일 때만 방어자산 인정.",
        "매우 높음. 체계적 운용과 재량 운용 모두 drawdown·quality·liquidity overlay를 사용.",
        "강하게 관찰. PIMCO가 위험국면에 고품질·유동성·agency MBS 비중을 높이는지 검증.",
        ["BlackRock systematic", "PIMCO income", "JPM active FI"],
    ),
    rule(
        "Regime Overlay",
        _regime_definition(),
        _regime_limits(),
        _regime_mapping(),
        "높음. 운용사들은 macro view와 cycle에 따라 duration·quality·sector를 동적으로 조정한다.",
        "가장 직접적인 비교 대상. BOND의 월별 섹터·duration 변화가 동일 레짐 방향과 맞는지 측정.",
        ["PIMCO income", "BlackRock systematic", "JPM active FI", "Research Affiliates"],
    ),
]


RULE_FIELDS = [
    "투자 철학", "투자 유니버스", "데이터", "팩터 정의", "리밸런싱", "롱/숏 구성", "가중치", "거래비용",
    "리스크 관리", "성과지표", "한계점", "ETF 근사 가능성", "Python 구현", "실무 활용도", "PIMCO와의 연결",
]
