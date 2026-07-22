from __future__ import annotations


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
    "투자 철학": "독립적인 기대수익 원천을 규칙으로 분리하고 결합해 단일 전망과 재량 편향에 대한 의존을 줄인다.",
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
        "ETF 추정수익률−SHY 수익률이 75bp 이상이고 Momentum≥0이면 +1, 25bp 미만이면 -1. 국채는 roll-down을 추가 연구.",
        "금리 급등·스프레드 확대 시 carry가 자본손실에 압도된다. MBB는 단순 모기지금리로 정확한 옵션비용을 포착하기 어렵다.",
        "IEF/TLT의 curve carry, LQD/HYG의 spread carry, MBB의 income carry. SHY는 funding proxy.",
        "매우 높음. 인컴·롤다운·섹터 로테이션은 전통적 채권운용의 핵심이지만 보통 개별채권 수준에서 구현된다.",
        "강하게 관찰. PIMCO는 carry와 roll-down을 명시하며 BOND의 일별 holdings에서 MBS·credit·Treasury 비중 변화를 확인할 수 있다.",
        ["PIMCO carry", "PIMCO income", "AQR styles", "Vanguard curve"],
    ),
    rule(
        "Momentum",
        "SHY 대비 6개월·12개월 총수익이 모두 양수면 +1, 모두 음수면 -1, 그 외 0.",
        "급격한 반전에서 늦고 채권의 낮은 변동성 때문에 신호가 작다. 2022년처럼 롱온리 제약은 하락추세 수익화를 막는다.",
        "모든 ETF에 동일 산식. 음의 신호는 숏 대신 SHY 이동.",
        "중간~높음. 단독보다는 value·carry·quality와 결합하는 systematic process에서 활용.",
        "직접적인 PIMCO 공시 신호로는 제한적. 포지션 변화가 가격추세와 같은 방향인지 사후 검증 가능.",
        ["AQR styles", "AQR timing", "Research Affiliates", "RA trend/carry"],
    ),
    rule(
        "Value",
        "각 ETF 수익률 proxy가 과거 5년 80백분위 이상이고 Momentum≥0, 물가 재가속이 아니면 +1; 20백분위 이하면 -1.",
        "높은 수익률이 구조적 부실·인플레이션 위험의 정당한 보상일 수 있다. yield proxy와 ETF 실제 YTM 사이 오차 존재.",
        "Treasury maturity yield, TIP real yield, LQD/HYG OAS 포함 수익률, MBB mortgage-rate proxy.",
        "높음. 액티브 운용사는 지수의 부채가중 왜곡을 피하고 저평가·고품질 채권을 선별한다고 설명한다.",
        "관찰 가능. PIMCO의 섹터별 overweight가 당시 spread/yield valuation과 일치하는지 비교.",
        ["AQR styles", "JPM active FI", "BlackRock systematic"],
    ),
    rule(
        "Real Yield",
        "10년 TIPS 실질금리≥2%이고 TIP Momentum≥0이면 TIP +1; 실질금리<0%이면 -1.",
        "2%는 표본의존 초기 임계값이다. breakeven에는 물가기대뿐 아니라 유동성과 inflation risk premium이 섞인다.",
        "TIP으로 직접 근사. 명목채 대비 TIP 상대배분으로 표현.",
        "높음. 실질수익률과 물가보호는 real-return 및 자산배분 운용의 표준 입력.",
        "부분 관찰. BOND의 물가연동채·effective duration 변화와 실질금리 수준을 연결해 검증.",
        ["PIMCO income", "Vanguard curve"],
    ),
    rule(
        "Yield Curve",
        "10Y−2Y>75bp이면 IEF +1. -25bp 미만이고 물가 둔화·실업 gap≥30bp이면 IEF/TLT 후보, 단 Momentum 필터 적용.",
        "곡선 역전은 기대단기금리와 기간프리미엄 원인이 다르다. 공급충격으로 장기물이 약해질 수 있다.",
        "SHY·IEF·TLT 사이 duration rotation.",
        "매우 높음. duration과 curve positioning은 모든 글로벌 채권 운용사의 핵심 의사결정.",
        "강하게 관찰 가능. PIMCO 공시 effective duration과 만기구간 배분 변화를 규칙 신호와 비교.",
        ["PIMCO carry", "Vanguard curve", "PIMCO income"],
    ),
    rule(
        "Credit Spread",
        "IG OAS<90bp 또는 HY OAS<300bp이면 -1. IG>150bp/HY>500bp이고 확대가 멈추며 Momentum≥0이면 +1.",
        "OAS가 넓어도 부도손실이 더 크게 상승할 수 있다. 2026년 이후 ICE 일부 시계열의 공개 이력이 짧다.",
        "LQD=IG beta, HYG=HY beta, MBB=securitized 대체. 부정 신호는 Treasury/SHY 이동.",
        "매우 높음. 섹터·품질·개별 신용선택은 액티브 채권의 대표적 alpha source.",
        "강하게 관찰. BOND의 회사채 품질·섹터 비중 변화와 OAS 국면을 비교.",
        ["PIMCO income", "JPM active FI", "BlackRock systematic"],
    ),
    rule(
        "Defensive",
        "HY OAS 3개월 +75bp, VIX 3년 80백분위, 실업률 12개월 저점 대비 +50bp 중 두 가지 이상이면 Risk Off.",
        "신호가 빠르면 carry를 포기하고, 늦으면 drawdown을 이미 겪는다. 주식 VIX가 채권 스트레스를 완전히 대변하지 않는다.",
        "HYG 0%, LQD 제한, SHY 우선. TLT는 Momentum>0일 때만 방어자산 인정.",
        "매우 높음. 체계적 운용과 재량 운용 모두 drawdown·quality·liquidity overlay를 사용.",
        "강하게 관찰. PIMCO가 위험국면에 고품질·유동성·agency MBS 비중을 높이는지 검증.",
        ["BlackRock systematic", "PIMCO income", "JPM active FI"],
    ),
    rule(
        "Regime Overlay",
        "산업생산 6개월 연율의 3개월 평균이 0% 초과인지, 근원 CPI 3개월 연율의 3개월 평균이 3% 초과인지로 네 국면을 구분.",
        "거시 데이터는 수정되고 후행적이다. 현재 단순 nowcast는 최초 발표 vintage를 보존하지 않아 연구 편향 가능.",
        "Goldilocks=LQD/HYG/MBB, Reflation=TIP/HYG/SHY, Stagflation=TIP/SHY, Disinflation=IEF/TLT/SHY.",
        "높음. 운용사들은 macro view와 cycle에 따라 duration·quality·sector를 동적으로 조정한다.",
        "가장 직접적인 비교 대상. BOND의 월별 섹터·duration 변화가 동일 레짐 방향과 맞는지 측정.",
        ["PIMCO income", "BlackRock systematic", "JPM active FI", "Research Affiliates"],
    ),
]


RULE_FIELDS = [
    "투자 철학", "투자 유니버스", "데이터", "팩터 정의", "리밸런싱", "롱/숏 구성", "가중치", "거래비용",
    "리스크 관리", "성과지표", "한계점", "ETF 근사 가능성", "Python 구현", "실무 활용도", "PIMCO와의 연결",
]
