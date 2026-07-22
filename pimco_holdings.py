from __future__ import annotations

import io
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from data_sources import APP_DIR, CACHE_DIR, FetchStatus


BOND_CIK = "1450011"
BOND_SERIES_ID = "S000033233"
SEC_SEARCH = "https://efts.sec.gov/LATEST/search-index"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
SEED_PATH = APP_DIR / "data" / "pimco_bond_sec_history.csv"
LATEST_PATH = APP_DIR / "data" / "pimco_bond_latest_holdings.csv"
EXPOSURE_COLUMNS = [
    "Government Related",
    "Agency MBS",
    "Non-Agency MBS",
    "ABS / CLO",
    "Credit & Loans",
    "Municipal / Other",
    "Cash & Repo",
]


def _sec_request(url: str, timeout: int = 35) -> bytes:
    user_agent = os.getenv("SEC_USER_AGENT", "BondSignalResearch research@example.com")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json,application/xml,text/xml,*/*",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(parent: ET.Element, name: str, default: str = "") -> str:
    for item in parent.iter():
        if _local_name(item.tag) == name:
            return (item.text or default).strip()
    return default


def _number(value: str | None, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _classify_sector(row: dict) -> str:
    name = f"{row.get('Name', '')} {row.get('Title', '')}".upper()
    asset = row.get("Asset Category", "")
    issuer = row.get("Issuer Category", "")
    country = row.get("Country", "")
    if asset == "RA" or "REPURCHASE" in name or " TREASURY REPO" in name:
        return "Cash & Repo"
    if asset in {"DFE", "DCR", "DIR"}:
        return "Derivative"
    if issuer == "USGSE" or any(token in name for token in ["FANNIE", "FREDDIE", "GINNIE", "FNMA", "FHLMC", "GNMA"]):
        return "Agency MBS" if asset == "ABS-MBS" or "MORTGAGE" in name else "Government Related"
    if issuer in {"UST", "USGA", "NUSS"}:
        return "Government Related"
    if asset == "ABS-MBS":
        return "Non-Agency MBS"
    if asset in {"ABS-CBDO", "ABS-O"}:
        return "ABS / CLO"
    if asset == "LON" or "BANK LOAN" in name:
        return "Credit & Loans"
    if issuer == "CORP" and asset in {"DBT", "EP", "EC"}:
        return "Credit & Loans"
    if issuer == "MUN":
        return "Municipal / Other"
    if country and country != "US" and issuer in {"CORP", "RF"}:
        return "Credit & Loans"
    return "Municipal / Other"


def parse_nport_xml(payload: bytes) -> tuple[dict, pd.DataFrame]:
    root = ET.fromstring(payload)
    report_date = pd.Timestamp(_text(root, "repPdDate"))
    net_assets = _number(_text(root, "netAssets"))
    total_assets = _number(_text(root, "totAssets"))

    dv01 = 0.0
    key_rate_dv01 = {"3M KRD": 0.0, "1Y KRD": 0.0, "5Y KRD": 0.0, "10Y KRD": 0.0, "30Y KRD": 0.0}
    krd_map = {"period3Mon": "3M KRD", "period1Yr": "1Y KRD", "period5Yr": "5Y KRD", "period10Yr": "10Y KRD", "period30Yr": "30Y KRD"}
    for item in root.iter():
        if _local_name(item.tag) == "intrstRtRiskdv01":
            for period, value in item.attrib.items():
                amount = _number(value, 0.0)
                dv01 += amount
                if period in krd_map:
                    key_rate_dv01[krd_map[period]] += amount
    effective_duration = dv01 / net_assets * 10_000 if net_assets else np.nan

    rows: list[dict] = []
    for item in root.iter():
        if _local_name(item.tag) != "invstOrSec":
            continue
        flat = {_local_name(child.tag): (child.text or "").strip() for child in item.iter()}
        row = {
            "Report Date": report_date,
            "Name": flat.get("name", ""),
            "Title": flat.get("title", ""),
            "CUSIP": flat.get("cusip", ""),
            "Market Value": _number(flat.get("valUSD")),
            "Percent NAV": _number(flat.get("pctVal")) / 100,
            "Asset Category": flat.get("assetCat", ""),
            "Issuer Category": flat.get("issuerCat", ""),
            "Country": flat.get("invCountry", ""),
            "Maturity": pd.to_datetime(flat.get("maturityDt"), errors="coerce"),
            "Coupon": _number(flat.get("annualizedRt")),
            "Payoff": flat.get("payoffProfile", ""),
            "Notional": _number(flat.get("notionalAmt")),
            "USD FX Notional": (
                _number(flat.get("amtCurSold")) if flat.get("curSold") == "USD"
                else _number(flat.get("amtCurPur")) if flat.get("curPur") == "USD"
                else np.nan
            ),
        }
        row["Sector"] = _classify_sector(row)
        rows.append(row)
    holdings = pd.DataFrame(rows)

    funded = holdings[
        holdings["Sector"].isin(EXPOSURE_COLUMNS)
        & holdings["Market Value"].notna()
        & (holdings["Market Value"] > 0)
    ].copy()
    gross = funded["Market Value"].sum()
    sector = funded.groupby("Sector")["Market Value"].sum().div(gross) if gross else pd.Series(dtype=float)
    summary = {
        "Report Date": report_date,
        "Net Assets": net_assets,
        "Total Assets": total_assets,
        "Gross / Net": total_assets / net_assets if net_assets else np.nan,
        "Effective Duration": effective_duration,
        "Number of Positions": len(holdings),
        **{name: amount / net_assets * 10_000 if net_assets else np.nan for name, amount in key_rate_dv01.items()},
        **{column: float(sector.get(column, 0.0)) for column in EXPOSURE_COLUMNS},
    }
    rate_derivatives = holdings[holdings["Asset Category"] == "DIR"]
    fx_derivatives = holdings[holdings["Asset Category"] == "DFE"]
    summary["Rate Derivative Notional / NAV"] = rate_derivatives["Notional"].abs().sum() / net_assets if net_assets else np.nan
    summary["FX Forward USD Notional / NAV"] = fx_derivatives["USD FX Notional"].abs().sum() / net_assets if net_assets else np.nan
    summary["Derivative Count"] = int(holdings["Asset Category"].isin(["DIR", "DFE", "DCR"]).sum())
    summary["High Quality Proxy"] = summary["Government Related"] + summary["Agency MBS"]
    summary["Spread Risk Proxy"] = (
        summary["Non-Agency MBS"] + summary["ABS / CLO"] + summary["Credit & Loans"]
    )
    return summary, holdings


def discover_bond_filings(max_periods: int = 16) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "q": BOND_SERIES_ID,
            "forms": "NPORT-P",
            "startdt": "2019-01-01",
            "enddt": pd.Timestamp.today().date().isoformat(),
            "from": 0,
            "size": 100,
        }
    )
    raw = json.loads(_sec_request(f"{SEC_SEARCH}?{params}"))
    candidates: list[dict] = []
    for hit in raw.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        if BOND_CIK.zfill(10) not in source.get("ciks", []):
            continue
        candidates.append(
            {
                "accession": source["adsh"],
                "report_date": pd.Timestamp(source["period_ending"]),
                "file_date": pd.Timestamp(source["file_date"]),
                "form": source.get("form", "NPORT-P"),
            }
        )
    if not candidates:
        raise RuntimeError("SEC에서 BOND N-PORT 공시를 찾지 못했습니다.")
    frame = pd.DataFrame(candidates).sort_values(["report_date", "file_date"])
    frame = frame.drop_duplicates("report_date", keep="last").sort_values("report_date", ascending=False)
    return frame.head(max_periods).to_dict("records")


def _filing_xml(filing: dict) -> tuple[bytes, bool]:
    accession = filing["accession"]
    cache_path = CACHE_DIR / f"sec_bond_{accession}.xml"
    if cache_path.exists():
        return cache_path.read_bytes(), True
    accession_plain = accession.replace("-", "")
    url = f"{SEC_ARCHIVES}/{BOND_CIK}/{accession_plain}/primary_doc.xml"
    payload = _sec_request(url, timeout=60)
    cache_path.write_bytes(payload)
    return payload, False


def _read_seed() -> pd.DataFrame:
    if not SEED_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(SEED_PATH, parse_dates=["Report Date"]).set_index("Report Date").sort_index()
    return frame


def _read_latest() -> pd.DataFrame:
    if not LATEST_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(LATEST_PATH, parse_dates=["Report Date", "Maturity"])
    return frame


def fetch_bond_exposure_history(max_periods: int = 16) -> tuple[pd.DataFrame, pd.DataFrame, FetchStatus]:
    seed = _read_seed()
    if not seed.empty and os.getenv("PIMCO_FORCE_REFRESH") != "1":
        next_public_window = seed.index.max() + pd.offsets.QuarterEnd() + pd.Timedelta(days=65)
        if pd.Timestamp.today().normalize() <= next_public_window:
            latest = _read_latest()
            status = FetchStatus(
                "SEC BOND Holdings", True, True, seed.index.max().date().isoformat(), "bundled SEC N-PORT history"
            )
            return seed.tail(max_periods), latest, status
    summaries: dict[pd.Timestamp, dict] = {
        pd.Timestamp(index): {"Report Date": pd.Timestamp(index), **row.dropna().to_dict()}
        for index, row in seed.iterrows()
    }
    latest_holdings = pd.DataFrame()
    cached_any = False
    errors: list[str] = []
    filings: list[dict] = []
    force_refresh = os.getenv("PIMCO_FORCE_REFRESH") == "1"
    try:
        filings = discover_bond_filings(max_periods)
    except Exception as exc:
        errors.append(f"검색 {type(exc).__name__}")

    for position, filing in enumerate(filings):
        date = pd.Timestamp(filing["report_date"])
        need_parse = force_refresh or date not in summaries or position == 0
        if not need_parse:
            continue
        try:
            payload, cached = _filing_xml(filing)
            cached_any = cached_any or cached
            summary, holdings = parse_nport_xml(payload)
            summary["Accession"] = filing["accession"]
            summaries[date] = summary
            if position == 0:
                latest_holdings = holdings
        except Exception as exc:
            errors.append(f"{date.date()} {type(exc).__name__}")

    if not summaries:
        raise RuntimeError("BOND 공개 포지션 데이터를 불러오지 못했습니다: " + ", ".join(errors))
    history = pd.DataFrame(summaries.values()).set_index("Report Date").sort_index()
    history = history[~history.index.duplicated(keep="last")].tail(max_periods)
    updated = history.index.max().date().isoformat()
    message = "SEC N-PORT 분기말 공시"
    if errors:
        message += " · 일부 로컬 이력 사용"
    status = FetchStatus("SEC BOND Holdings", True, cached_any or bool(errors), updated, message)
    try:
        write_seed(history, latest_holdings)
    except OSError:
        pass
    return history, latest_holdings, status


ETF_DURATION = {
    "SHY": 1.8, "IEF": 7.3, "TLT": 16.0, "TIP": 6.5, "LQD": 8.0,
    "HYG": 3.2, "MBB": 5.5, "JAAA": 0.4, "BKLN": 0.3, "CMBS": 5.0,
}


def rule_exposures(weights: pd.DataFrame) -> pd.DataFrame:
    w = weights.reindex(columns=ETF_DURATION, fill_value=0.0).fillna(0.0)
    result = pd.DataFrame(index=w.index)
    result["Effective Duration"] = w.mul(pd.Series(ETF_DURATION)).sum(axis=1)
    result["Government Related"] = w[["SHY", "IEF", "TLT", "TIP"]].sum(axis=1)
    result["Agency MBS"] = w["MBB"]
    result["Non-Agency MBS"] = w["CMBS"]
    result["ABS / CLO"] = w[["JAAA", "BKLN"]].sum(axis=1)
    result["Credit & Loans"] = w[["LQD", "HYG"]].sum(axis=1)
    result["High Quality Proxy"] = w[["SHY", "IEF", "TLT", "TIP", "LQD", "MBB", "JAAA"]].sum(axis=1)
    result["Spread Risk Proxy"] = w[["HYG", "BKLN", "CMBS"]].sum(axis=1)
    return result


def _strategy_exposure_for_date(weight_row: pd.Series, bond_row: pd.Series) -> pd.Series:
    bond_weight = float(weight_row.get("BOND", 0.0))
    etf_weights = weight_row.drop(labels=["BOND"], errors="ignore")
    etf_exposure = rule_exposures(pd.DataFrame([etf_weights], index=[bond_row.name])).iloc[0]
    for column in etf_exposure.index:
        etf_exposure[column] = float(etf_exposure[column]) + bond_weight * float(bond_row.get(column, 0.0))
    return etf_exposure


def position_alignment(weights: pd.DataFrame, bond_history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if weights.empty or bond_history.empty:
        return pd.DataFrame(), {}
    common = [
        "Effective Duration", "Government Related", "Agency MBS", "Non-Agency MBS",
        "ABS / CLO", "Credit & Loans", "High Quality Proxy", "Spread Risk Proxy",
    ]
    rows = []
    for date, bond_row in bond_history.iterrows():
        available = weights.sort_index().loc[:date]
        if available.empty:
            continue
        rule_row = _strategy_exposure_for_date(available.iloc[-1], bond_row)
        row = {"Report Date": date}
        for column in common:
            row[f"BOND {column}"] = bond_row.get(column, np.nan)
            row[f"Rule {column}"] = rule_row.get(column, np.nan)
        rows.append(row)
    aligned = pd.DataFrame(rows).set_index("Report Date") if rows else pd.DataFrame()
    if len(aligned) < 2:
        return aligned, {}

    match_values = []
    errors = []
    for column in common:
        bond = aligned[f"BOND {column}"]
        strategy = aligned[f"Rule {column}"]
        valid = pd.concat([bond, strategy], axis=1).dropna()
        if len(valid) < 2:
            continue
        scale = 8.0 if column == "Effective Duration" else 1.0
        errors.extend(((valid.iloc[:, 0] - valid.iloc[:, 1]) / scale).abs().tolist())
        bond_change = valid.iloc[:, 0].diff()
        rule_change = valid.iloc[:, 1].diff()
        tolerance = 0.10 if column == "Effective Duration" else 0.01
        meaningful = (bond_change.abs() >= tolerance) | (rule_change.abs() >= tolerance)
        if meaningful.any():
            same = np.sign(bond_change[meaningful]) == np.sign(rule_change[meaningful])
            match_values.extend(same.dropna().astype(float).tolist())
    mae = float(np.mean(errors)) if errors else np.nan
    direction_match = float(np.mean(match_values)) if match_values else np.nan
    fit_score = float(max(0.0, 1.0 - mae) * 0.6 + (direction_match if np.isfinite(direction_match) else 0.0) * 0.4)
    return aligned, {"Exposure MAE": mae, "Direction Match": direction_match, "Position Fit": fit_score}


def mbs_gap_diagnostic(
    weights: pd.DataFrame,
    bond_history: pd.DataFrame,
    signals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Compare BOND's securitized book with the ETF-only MBB proxy."""
    if weights.empty or bond_history.empty:
        return pd.DataFrame(), {}

    rows: list[dict] = []
    for date, bond_row in bond_history.iterrows():
        available_weight = weights.sort_index().loc[:date]
        if available_weight.empty:
            continue
        weight_row = available_weight.iloc[-1]
        bond_weight = float(weight_row.get("BOND", 0.0))
        rule_row = _strategy_exposure_for_date(weight_row, bond_row)
        signal_row = pd.Series(dtype=object)
        if signals is not None and not signals.empty:
            available_signals = signals.loc[:date]
            if not available_signals.empty:
                signal_row = available_signals.iloc[-1]
        agency = float(bond_row.get("Agency MBS", np.nan))
        non_agency = float(bond_row.get("Non-Agency MBS", np.nan))
        abs_clo = float(bond_row.get("ABS / CLO", np.nan))
        structured = non_agency + abs_clo
        total_securitized = agency + structured
        rule_mbb = float(rule_row.get("Agency MBS", np.nan))
        rule_structured_proxy = float(
            bond_weight * structured + sum(weight_row.get(ticker, 0.0) for ticker in ["JAAA", "BKLN", "CMBS"])
        )
        credit_proxy = float(rule_row.get("Credit & Loans", np.nan))
        row = {
            "Report Date": date,
            "BOND Agency MBS": agency,
            "Rule MBB": rule_mbb,
            "Agency Gap": agency - rule_mbb,
            "BOND Non-Agency MBS": non_agency,
            "BOND ABS / CLO": abs_clo,
            "Missing Structured Sleeve": structured,
            "BOND Total Securitized": total_securitized,
            "Rule Explicit Securitized": rule_mbb,
            "Rule Structured Proxy": rule_structured_proxy,
            "Rule Total Securitized Proxy": rule_mbb + rule_structured_proxy,
            "Total Securitized Gap": total_securitized - rule_mbb,
            "Total Gap After Structured Proxy": total_securitized - rule_mbb - rule_structured_proxy,
            "Rule Credit Proxy": credit_proxy,
            "MBS RV Score": signal_row.get("MBS RV Score", np.nan),
            "MBS Spread Percentile": signal_row.get("MBS Spread Percentile", np.nan),
            "MBB vs IEF 3M": signal_row.get("MBB vs IEF 3M", np.nan),
            "Dominant Alpha": signal_row.get("Dominant Alpha", ""),
            "MOVE": signal_row.get("MOVE", np.nan),
            "MOVE Percentile": signal_row.get("MOVE Percentile", np.nan),
        }
        rows.append(row)
    diagnostic = pd.DataFrame(rows).set_index("Report Date") if rows else pd.DataFrame()
    if len(diagnostic) < 1:
        return diagnostic, {}

    metrics = {
        "Avg Agency Gap": diagnostic["Agency Gap"].mean(),
        "Avg Missing Structured Sleeve": diagnostic["Missing Structured Sleeve"].mean(),
        "Avg Total Securitized Gap": diagnostic["Total Securitized Gap"].mean(),
        "Avg Total Gap After Structured Proxy": diagnostic["Total Gap After Structured Proxy"].mean(),
        "Latest Agency Gap": diagnostic["Agency Gap"].iloc[-1],
        "Latest Missing Structured Sleeve": diagnostic["Missing Structured Sleeve"].iloc[-1],
        "Latest Total Securitized Gap": diagnostic["Total Securitized Gap"].iloc[-1],
        "Latest Total Gap After Structured Proxy": diagnostic["Total Gap After Structured Proxy"].iloc[-1],
    }
    if len(diagnostic) >= 3:
        metrics["Agency Direction Match"] = (
            np.sign(diagnostic["BOND Agency MBS"].diff()) == np.sign(diagnostic["Rule MBB"].diff())
        ).dropna().mean()
        metrics["MBS Score vs Next Agency Change Corr"] = diagnostic["MBS RV Score"].corr(
            diagnostic["BOND Agency MBS"].diff().shift(-1)
        )
        metrics["MBS Score vs Next Total Securitized Change Corr"] = diagnostic["MBS RV Score"].corr(
            diagnostic["BOND Total Securitized"].diff().shift(-1)
        )
    return diagnostic, metrics


def combination_position_scores(weight_sets: dict[str, pd.DataFrame], bond_history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, weights in weight_sets.items():
        _, metrics = position_alignment(weights, bond_history)
        rows.append({"Combination": name, **metrics})
    return pd.DataFrame(rows).set_index("Combination") if rows else pd.DataFrame()


def write_seed(history: pd.DataFrame, latest_holdings: pd.DataFrame | None = None) -> Path:
    SEED_PATH.parent.mkdir(exist_ok=True)
    history.reset_index().to_csv(SEED_PATH, index=False)
    if latest_holdings is not None and not latest_holdings.empty:
        latest_holdings.to_csv(LATEST_PATH, index=False)
    return SEED_PATH
