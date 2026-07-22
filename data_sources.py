from __future__ import annotations

import io
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR = APP_DIR / "data"
FRED_SEED_PATH = DATA_DIR / "fred_snapshot.csv"
ETF_SEED_PATH = DATA_DIR / "etf_total_return_snapshot.csv"

ETF_TICKERS = ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "MBB"]
STRUCTURED_PROXY_TICKERS = ["JAAA", "BKLN", "CMBS"]
CASH_PROXY_TICKERS = ["SGOV", "BIL", "USFR"]

FRED_SERIES = {
    "DGS3MO": "US 3M Treasury",
    "DGS2": "US 2Y Treasury",
    "DGS5": "US 5Y Treasury",
    "DGS10": "US 10Y Treasury",
    "DGS30": "US 30Y Treasury",
    "T10Y2Y": "10Y–2Y Curve",
    "DFII10": "10Y Real Yield",
    "T10YIE": "10Y Breakeven",
    "BAMLC0A0CM": "IG OAS",
    "BAMLH0A0HYM2": "HY OAS",
    "AAA10Y": "Moody's Aaa - 10Y Treasury spread",
    "BAA10Y": "Moody's Baa - 10Y Treasury spread",
    "CPILFESL": "Core CPI",
    "INDPRO": "Industrial Production",
    "UNRATE": "Unemployment Rate",
    "VIXCLS": "VIX",
    "NFCI": "Chicago Fed NFCI",
    "MORTGAGE30US": "30Y Mortgage Rate",
}


@dataclass
class FetchStatus:
    source: str
    ok: bool
    cached: bool
    updated: str
    message: str = ""


def _request(url: str, timeout: int = 12) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 BondSignalDashboard/1.0",
            "Accept": "application/json,text/csv,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _write_cache(name: str, payload: bytes) -> None:
    (CACHE_DIR / name).write_bytes(payload)


def _read_cache(name: str) -> bytes | None:
    path = CACHE_DIR / name
    return path.read_bytes() if path.exists() else None


def _fred_json_series(payload: bytes, series_id: str) -> pd.Series:
    raw = json.loads(payload)
    observations = raw.get("observations", [])
    if not observations:
        raise ValueError(f"{series_id}: empty API response")
    index = pd.to_datetime([row["date"] for row in observations])
    values = pd.to_numeric([row["value"] for row in observations], errors="coerce")
    return pd.Series(values, index=index, name=series_id, dtype="float64")


def _fred_csv_series(payload: bytes, series_id: str) -> pd.Series:
    frame = pd.read_csv(io.BytesIO(payload))
    date_col = "observation_date" if "observation_date" in frame.columns else frame.columns[0]
    value_col = series_id if series_id in frame.columns else frame.columns[-1]
    index = pd.to_datetime(frame[date_col])
    values = pd.to_numeric(frame[value_col], errors="coerce")
    return pd.Series(values.to_numpy(), index=index, name=series_id)


def _fetch_one_fred(series_id: str) -> tuple[str, pd.Series, bool, str]:
    api_key = os.getenv("FRED_API_KEY", "").strip()
    errors: list[str] = []
    if api_key:
        api_params = urllib.parse.urlencode(
            {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": "2000-01-01",
            }
        )
        api_url = f"https://api.stlouisfed.org/fred/series/observations?{api_params}"
        try:
            payload = _request(api_url, timeout=20)
            values = _fred_json_series(payload, series_id)
            _write_cache(f"fred_{series_id}.json", payload)
            return series_id, values, False, "FRED API"
        except Exception as exc:
            # Never include a request URL here: it contains the API key.
            errors.append(f"API/{type(exc).__name__}")

    params = urllib.parse.urlencode({"id": series_id, "cosd": "2000-01-01"})
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{params}"
    try:
        payload = _request(url, timeout=25)
        values = _fred_csv_series(payload, series_id)
        _write_cache(f"fred_{series_id}.csv", payload)
        return series_id, values, False, "FRED CSV"
    except Exception as exc:
        errors.append(f"CSV/{type(exc).__name__}")

    json_cache = _read_cache(f"fred_{series_id}.json")
    if json_cache is not None:
        return series_id, _fred_json_series(json_cache, series_id), True, "cached API"
    csv_cache = _read_cache(f"fred_{series_id}.csv")
    if csv_cache is not None:
        return series_id, _fred_csv_series(csv_cache, series_id), True, "cached CSV"
    raise RuntimeError(f"{series_id}: " + ", ".join(errors))


def fetch_fred(series: list[str] | None = None, force_live: bool = False) -> tuple[pd.DataFrame, FetchStatus]:
    ids = series or list(FRED_SERIES)
    seed = pd.DataFrame()
    if FRED_SEED_PATH.exists():
        try:
            seed = pd.read_csv(FRED_SEED_PATH, index_col=0, parse_dates=True).sort_index()
            seed = seed[[sid for sid in ids if sid in seed.columns]]
        except Exception:
            seed = pd.DataFrame()
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not force_live and set(ids).issubset(seed.columns):
        updated = seed.dropna(how="all").index.max().date().isoformat()
        return seed[ids], FetchStatus("FRED", True, True, updated, "번들 스냅샷 · 실시간 갱신 토글 OFF")
    pieces: dict[str, pd.Series] = {}
    cached_ids: list[str] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(_fetch_one_fred, sid): sid for sid in ids}
        for job in as_completed(jobs):
            sid = jobs[job]
            try:
                series_id, values, cached, _ = job.result()
                pieces[series_id] = values
                if cached:
                    cached_ids.append(series_id)
            except Exception as exc:
                failures.append(str(exc))
    if not pieces and set(ids).issubset(seed.columns):
        updated = seed.dropna(how="all").index.max().date().isoformat()
        return seed[ids], FetchStatus("FRED", True, True, updated, "네트워크 실패 · 번들 스냅샷")
    if not pieces:
        raise RuntimeError("FRED 데이터를 불러오지 못했습니다: " + "; ".join(failures))
    missing = [sid for sid in ids if sid not in pieces]
    for sid in list(missing):
        if sid in seed:
            pieces[sid] = seed[sid]
            cached_ids.append(sid)
            missing.remove(sid)
    if missing:
        raise RuntimeError("필수 FRED 시리즈 누락: " + ", ".join(missing))
    frame = pd.concat([pieces[sid] for sid in ids], axis=1).sort_index()
    if {"BAMLC0A0CM", "AAA10Y", "BAA10Y"}.issubset(frame.columns):
        ig_proxy = (frame["AAA10Y"] + frame["BAA10Y"]) / 2
        frame["BAMLC0A0CM"] = frame["BAMLC0A0CM"].combine_first(ig_proxy)
    if {"BAMLH0A0HYM2", "BAA10Y"}.issubset(frame.columns):
        hy_proxy = frame["BAA10Y"] * 1.75
        frame["BAMLH0A0HYM2"] = frame["BAMLH0A0HYM2"].combine_first(hy_proxy)
    updated = frame.dropna(how="all").index.max().date().isoformat()
    message = "모두 최신" if not cached_ids else "일부 캐시: " + ", ".join(cached_ids)
    return frame, FetchStatus("FRED", True, bool(cached_ids), updated, message)


def _parse_yahoo_chart(payload: bytes, ticker: str) -> tuple[pd.Series, dict]:
    raw = json.loads(payload)
    result = raw["chart"]["result"][0]
    timestamps = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None).normalize()
    indicators = result["indicators"]
    values = indicators.get("adjclose", [{}])[0].get("adjclose")
    if values is None:
        values = indicators["quote"][0]["close"]
    series = pd.Series(values, index=timestamps, name=ticker, dtype="float64").dropna()
    return series[~series.index.duplicated(keep="last")], result.get("meta", {})


def _fetch_one_etf(ticker: str) -> tuple[pd.Series | None, dict | None, FetchStatus]:
        period2 = int(time.time()) + 86400
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1=946684800&period2={period2}&interval=1d&events=div%2Csplits"
        cache_name = f"yahoo_{ticker}.json"
        cached = False
        try:
            payload = _request(url)
            _write_cache(cache_name, payload)
            message = "Yahoo Finance chart"
        except Exception as exc:
            payload = _read_cache(cache_name)
            if payload is None:
                return None, None, FetchStatus(ticker, False, False, "—", str(exc))
            cached = True
            message = f"캐시 사용: {type(exc).__name__}"

        try:
            series, meta = _parse_yahoo_chart(payload, ticker)
            last = float(series.iloc[-1])
            previous = float(series.iloc[-2]) if len(series) > 1 else last
            quote = {
                    "Ticker": ticker,
                    "Price": float(meta.get("regularMarketPrice") or last),
                    "Day %": (last / previous - 1.0) * 100,
                    "Currency": meta.get("currency", "USD"),
                    "As of": series.index[-1].date().isoformat(),
                }
            return series, quote, FetchStatus(ticker, True, cached, series.index[-1].date().isoformat(), message)
        except Exception as exc:
            return None, None, FetchStatus(ticker, False, cached, "—", f"파싱 실패: {exc}")


def fetch_etfs(tickers: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[FetchStatus]]:
    tickers = tickers or ETF_TICKERS
    seed = pd.DataFrame()
    if ETF_SEED_PATH.exists():
        try:
            seed = pd.read_csv(ETF_SEED_PATH, index_col=0, parse_dates=True).sort_index()
        except Exception:
            seed = pd.DataFrame()
    prices: list[pd.Series] = []
    quotes: list[dict] = []
    statuses: list[FetchStatus] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(_fetch_one_etf, ticker): ticker for ticker in tickers}
        results = {jobs[job]: job.result() for job in as_completed(jobs)}
    for ticker in tickers:
        series, quote, status = results[ticker]
        if series is None and ticker in seed:
            series = seed[ticker].dropna().rename(ticker)
            if len(series):
                last = float(series.iloc[-1])
                previous = float(series.iloc[-2]) if len(series) > 1 else last
                quote = {"Ticker": ticker, "Price": last, "Day %": (last / previous - 1) * 100,
                         "Currency": "USD", "As of": series.index[-1].date().isoformat()}
                status = FetchStatus(ticker, True, True, series.index[-1].date().isoformat(), "번들 총수익 스냅샷")
        statuses.append(status)
        if series is not None and quote is not None:
            prices.append(series)
            quotes.append(quote)

    if not prices:
        raise RuntimeError("ETF 가격 데이터를 불러오지 못했고 사용할 캐시도 없습니다.")
    return pd.concat(prices, axis=1).sort_index().ffill(), pd.DataFrame(quotes).set_index("Ticker"), statuses


def cache_age_minutes() -> dict[str, int]:
    now = time.time()
    return {p.name: int((now - p.stat().st_mtime) / 60) for p in CACHE_DIR.glob("*")}


def demo_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FetchStatus, list[FetchStatus]]:
    """Deterministic offline fixture used only for UI tests when BOND_DASHBOARD_DEMO=1."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=2600, freq="B")
    n = len(idx)
    trend = np.linspace(0, 1, n)
    fred = pd.DataFrame(index=idx)
    fred["DGS3MO"] = 1.0 + 3.0 * trend + np.sin(np.arange(n) / 70) * .18
    fred["DGS2"] = 1.2 + 3.1 * trend + np.sin(np.arange(n) / 80) * .25
    fred["DGS5"] = fred["DGS2"] + .12
    fred["DGS10"] = fred["DGS2"] + .28
    fred["DGS30"] = fred["DGS2"] + .55
    fred["T10Y2Y"] = fred["DGS10"] - fred["DGS2"]
    fred["DFII10"] = -0.2 + 2.1 * trend + np.sin(np.arange(n) / 110) * .18
    fred["T10YIE"] = fred["DGS10"] - fred["DFII10"]
    fred["BAMLC0A0CM"] = 1.05 + np.sin(np.arange(n) / 130) * .25
    fred["BAMLH0A0HYM2"] = 3.4 + np.sin(np.arange(n) / 100) * .8
    fred["VIXCLS"] = 17 + np.maximum(0, np.sin(np.arange(n) / 70)) * 7
    fred["NFCI"] = -.25 + np.sin(np.arange(n) / 180) * .2
    fred["MORTGAGE30US"] = fred["DGS10"] + 2.4
    monthly_idx = idx
    fred["CPILFESL"] = 260 * np.exp(np.linspace(0, .32, n))
    fred["INDPRO"] = 98 * np.exp(np.linspace(0, .09, n) + np.sin(np.arange(n) / 300) * .015)
    fred["UNRATE"] = 4.2 - .6 * trend + np.sin(np.arange(n) / 250) * .15
    base = rng.normal(.00012, .002, (n, len(ETF_TICKERS)))
    durations = np.array([1.8, 7.3, 16.0, 6.5, 8.0, 3.2, 5.5])
    rate_shock = rng.normal(0, .00025, n)
    base -= rate_shock[:, None] * durations[None, :]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(base, axis=0)), index=idx, columns=ETF_TICKERS)
    prices["MOVE"] = 90 + np.maximum(0, np.sin(np.arange(n) / 55)) * 45
    quotes_source = prices[ETF_TICKERS]
    quotes = pd.DataFrame({"Ticker": ETF_TICKERS, "Price": quotes_source.iloc[-1].values,
                           "Day %": quotes_source.pct_change().iloc[-1].values * 100,
                           "Currency": "USD", "As of": idx[-1].date().isoformat()}).set_index("Ticker")
    status = FetchStatus("DEMO", True, False, idx[-1].date().isoformat(), "offline UI fixture")
    return fred, prices, quotes, status, [status for _ in ETF_TICKERS]
