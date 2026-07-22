from __future__ import annotations

import math

import numpy as np
import pandas as pd


FEATURE_META = {
    "10Y Treasury": ("미 10년 국채금리", "%", 2),
    "10Y Treasury 3M Change": ("미 10년 금리 3개월 변화", "%p", 2),
    "10Y Real Yield": ("10년 실질금리", "%", 2),
    "10Y Breakeven": ("10년 기대인플레이션", "%", 2),
    "10Y-2Y Curve": ("10년-2년 커브", "%p", 2),
    "30Y-5Y Curve": ("30년-5년 커브", "%p", 2),
    "IG OAS": ("IG OAS", "%p", 2),
    "IG OAS 3M Change": ("IG OAS 3개월 변화", "%p", 2),
    "HY OAS": ("HY OAS", "%p", 2),
    "HY OAS 3M Change": ("HY OAS 3개월 변화", "%p", 2),
    "Mortgage Spread": ("모기지금리-10년 국채 스프레드", "%p", 2),
    "Core CPI 3M Ann": ("근원 CPI 3개월 연율", "%", 1),
    "Industrial Production 6M Ann": ("산업생산 6개월 연율", "%", 1),
    "Unemployment Gap": ("실업률 12개월 저점 대비 상승폭", "%p", 2),
    "VIX": ("VIX", "", 1),
    "NFCI": ("Chicago Fed NFCI", "", 2),
}

TARGET_META = {
    "Effective Duration": ("Effective Duration", 0.15, "년", 8.0),
    "5Y KRD": ("5년 Key Rate Duration", 0.10, "년", 3.0),
    "10Y KRD": ("10년 Key Rate Duration", 0.10, "년", 3.0),
    "30Y KRD": ("30년 Key Rate Duration", 0.07, "년", 2.0),
    "Agency MBS": ("Agency MBS 비중", 0.015, "%p", 1.0),
    "Credit & Loans": ("Credit & Loans 비중", 0.015, "%p", 1.0),
    "Spread Risk Proxy": ("Spread Risk 비중", 0.015, "%p", 1.0),
    "Gross / Net": ("Gross/Net", 0.08, "x", 1.0),
    "Rate Derivative Notional / NAV": ("금리파생 명목금액/NAV", 0.08, "x", 2.0),
}


def build_rule_features(fred: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    monthly = fred.resample("ME").last().ffill()
    features = pd.DataFrame(index=monthly.index)
    features["10Y Treasury"] = monthly["DGS10"]
    features["10Y Treasury 3M Change"] = monthly["DGS10"].diff(3)
    features["10Y Real Yield"] = monthly["DFII10"]
    features["10Y Breakeven"] = monthly["T10YIE"]
    features["10Y-2Y Curve"] = monthly["DGS10"] - monthly["DGS2"]
    features["30Y-5Y Curve"] = monthly["DGS30"] - monthly["DGS5"]
    features["IG OAS"] = monthly["BAMLC0A0CM"]
    features["IG OAS 3M Change"] = monthly["BAMLC0A0CM"].diff(3)
    features["HY OAS"] = monthly["BAMLH0A0HYM2"]
    features["HY OAS 3M Change"] = monthly["BAMLH0A0HYM2"].diff(3)
    features["Mortgage Spread"] = monthly["MORTGAGE30US"] - monthly["DGS10"]
    features["Core CPI 3M Ann"] = ((monthly["CPILFESL"] / monthly["CPILFESL"].shift(3)) ** 4 - 1) * 100
    features["Industrial Production 6M Ann"] = ((monthly["INDPRO"] / monthly["INDPRO"].shift(6)) ** 2 - 1) * 100
    features["Unemployment Gap"] = monthly["UNRATE"] - monthly["UNRATE"].rolling(12).min()
    features["VIX"] = monthly["VIXCLS"]
    features["NFCI"] = monthly["NFCI"]
    return features.reindex(pd.DatetimeIndex(dates), method="ffill")


def _binomial_tail(successes: int, trials: int, probability: float) -> float:
    if trials <= 0 or not 0 <= probability <= 1:
        return np.nan
    return min(1.0, sum(math.comb(trials, k) * probability**k * (1 - probability) ** (trials - k) for k in range(successes, trials + 1)))


def _format_threshold(feature: str, value: float, operator: str) -> str:
    label, unit, digits = FEATURE_META[feature]
    symbol = "≥" if operator == ">=" else "≤"
    return f"{label} {symbol} {value:.{digits}f}{unit}"


def _format_change(target: str, value: float) -> str:
    _, _, unit, _ = TARGET_META[target]
    if unit == "%p":
        return f"{value * 100:+.1f}%p"
    return f"{value:+.2f}{unit}"


def _economic_logic(feature: str, target: str, direction: int) -> str:
    up = direction > 0
    if "Duration" in target or "KRD" in target:
        if "Real Yield" in feature or "Treasury" in feature:
            return "금리 수준은 채권의 시작수익률과 평균회귀 여유를 바꿔 듀레이션 위험의 기대보상을 변화시킨다."
        if "Curve" in feature:
            return "커브 기울기는 롤다운·캐리와 어느 만기구간에 금리위험을 배치할지 결정한다."
        return "물가·성장·금융여건 변화는 정책금리 경로와 듀레이션 위험예산을 바꾼다."
    if target == "Agency MBS":
        return "모기지 스프레드와 위험회피 정도는 Agency MBS의 추가 캐리와 조기상환·볼록성 위험 간 보상을 바꾼다."
    if target in {"Credit & Loans", "Spread Risk Proxy"}:
        return "신용스프레드는 기대손실·유동성 위험을 감수하고 받는 보상이므로 확대/축소 국면에서 크레딧 위험예산이 달라진다."
    if target in {"Gross / Net", "Rate Derivative Notional / NAV"}:
        stance = "확대" if up else "축소"
        return f"시장 기회와 유동성 위험의 균형에 따라 파생상품·Repo를 통한 총 익스포저를 {stance}하는 위험예산 논리다."
    return "거시 상태 변화가 포트폴리오 위험예산과 섹터 배분을 동시에 변화시키는 것으로 해석한다."


def discover_if_then_rules(
    bond_history: pd.DataFrame,
    fred: pd.DataFrame,
    max_per_target: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = bond_history.sort_index()
    features = build_rule_features(fred, history.index)
    panel = features.copy()
    for target in TARGET_META:
        if target in history:
            panel[f"NEXT::{target}"] = history[target].diff().shift(-1)

    candidates: list[dict] = []
    test_count = 0
    quantiles = [0.25, 1 / 3, 0.50, 2 / 3, 0.75]
    for target, (target_label, minimum_move, _, scale) in TARGET_META.items():
        delta_col = f"NEXT::{target}"
        if delta_col not in panel:
            continue
        target_delta = panel[delta_col]
        valid_target = target_delta.dropna()
        if len(valid_target) < 10:
            continue
        for feature in FEATURE_META:
            sample = pd.concat([panel[feature], target_delta], axis=1).dropna()
            if len(sample) < 10 or sample[feature].nunique() < 5:
                continue
            split = max(10, int(len(sample) * .70))
            train, test = sample.iloc[:split], sample.iloc[split:]
            for threshold in sorted(set(train[feature].quantile(quantiles).round(6))):
                for operator in [">=", "<="]:
                    train_condition = train[feature] >= threshold if operator == ">=" else train[feature] <= threshold
                    conditioned = train.loc[train_condition, delta_col]
                    if len(conditioned) < 4:
                        continue
                    test_count += 1
                    average_change = conditioned.mean()
                    direction = 1 if average_change >= 0 else -1
                    successes = int((conditioned * direction >= minimum_move).sum())
                    hit_rate = successes / len(conditioned)
                    baseline = float((train[delta_col] * direction >= minimum_move).mean())
                    lift = hit_rate - baseline
                    test_condition = test[feature] >= threshold if operator == ">=" else test[feature] <= threshold
                    oos = test.loc[test_condition, delta_col]
                    oos_successes = int((oos * direction >= minimum_move).sum())
                    oos_hit_rate = oos_successes / len(oos) if len(oos) else np.nan
                    stable = len(oos) >= 2 and np.sign(oos.mean()) == direction
                    p_value = _binomial_tail(successes, len(conditioned), baseline)
                    if hit_rate < 0.55 or lift < 0.08 or abs(average_change) < minimum_move * 0.75:
                        continue
                    score = lift + min(abs(average_change) / scale, 1.0) * .15 + (0.10 if stable and oos_hit_rate >= .50 else 0) - p_value * .03
                    confidence = "후보 A" if len(oos) >= 3 and oos_hit_rate >= .67 and stable else "후보 B" if len(oos) >= 2 and oos_hit_rate >= .50 and stable else "탐색적"
                    condition_text = f"IF {_format_threshold(feature, threshold, operator)}"
                    action = "확대" if direction > 0 else "축소"
                    candidates.append(
                        {
                            "Target": target_label,
                            "Target Key": target,
                            "Feature": feature,
                            "IF": condition_text,
                            "THEN": f"THEN {target_label} {action} (평균 {_format_change(target, average_change)})",
                            "관측 수": len(conditioned),
                            "적중률": hit_rate,
                            "기준 적중률": baseline,
                            "Lift": lift,
                            "OOS 관측 수": len(oos),
                            "OOS 적중률": oos_hit_rate,
                            "평균 변화": average_change,
                            "p-value": p_value,
                            "기간 안정성": bool(stable),
                            "신뢰도": confidence,
                            "경제적 논리": _economic_logic(feature, target, direction),
                            "Score": score,
                        }
                    )
    if not candidates:
        return pd.DataFrame(), panel
    all_rules = pd.DataFrame(candidates).sort_values(["Target", "Score"], ascending=[True, False])
    all_rules["탐색보정 p-value"] = (all_rules["p-value"] * max(test_count, 1)).clip(upper=1.0)
    selected = all_rules.groupby("Target", group_keys=False).head(max_per_target).sort_values("Score", ascending=False)
    selected["_grade"] = selected["신뢰도"].map({"후보 A": 3, "후보 B": 2, "탐색적": 1}).fillna(0)
    selected = selected.sort_values(["_grade", "OOS 적중률", "Score"], ascending=[False, False, False]).drop(columns="_grade")
    selected.index = range(1, len(selected) + 1)
    selected.index.name = "Rule"
    return selected, panel
