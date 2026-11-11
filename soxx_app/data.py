"""1) 최근 20년치 거래 데이터 다운로드 및 엑셀 저장."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import DEFAULT_TICKER

COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def period_bounds(years: int = 20, today: date | None = None) -> tuple[date, date]:
    """(시작일, 어제) 반환. 시작일 = 오늘로부터 `years`년 전."""
    today = today or date.today()
    start = today.replace(year=today.year - years) if not (today.month == 2 and today.day == 29) \
        else date(today.year - years, 2, 28)
    return start, today - timedelta(days=1)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """yfinance 결과(멀티인덱스 컬럼 가능)를 단일 컬럼 OHLCV 프레임으로 정리."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    df = df[[c for c in COLUMNS if c in df.columns]]
    df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df[~df.index.duplicated()].sort_index().dropna(subset=["Close"])
    return df


def download(ticker: str = DEFAULT_TICKER, years: int = 20, today: date | None = None,
             warmup_years: int = 0) -> pd.DataFrame:
    """Yahoo Finance에서 일별 OHLCV 데이터를 내려받는다 (오늘 제외, 어제까지).

    warmup_years: 이동평균 등 피처 계산용으로 추가로 받을 과거 기간(년).
    """
    import yfinance as yf

    start, yesterday = period_bounds(years + warmup_years, today)
    raw = yf.download(ticker, start=start.isoformat(), end=(yesterday + timedelta(days=1)).isoformat(),
                      auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"'{ticker}' 데이터를 받지 못했습니다. 인터넷 연결 또는 티커를 확인하세요.")
    return normalize(raw)


def trim(df: pd.DataFrame, years: int = 20, today: date | None = None) -> pd.DataFrame:
    """최근 `years`년(어제까지)만 남긴다."""
    start, yesterday = period_bounds(years, today)
    return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(yesterday))]


def yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df.index.year)
    out = pd.DataFrame({
        "시가(첫 거래일)": g["Open"].first(),
        "종가(마지막 거래일)": g["Close"].last(),
        "최고가": g["High"].max(),
        "최저가": g["Low"].min(),
        "평균 거래량": g["Volume"].mean().round(0),
        "거래일수": g["Close"].count(),
    })
    out["연간 수익률(%)"] = (g["Adj Close"].last() / g["Adj Close"].first() - 1).mul(100).round(2)
    out.index.name = "연도"
    return out


def save_excel(df: pd.DataFrame, path: str | Path, ticker: str = DEFAULT_TICKER) -> Path:
    """일별 데이터 + 연도별 요약을 엑셀로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    daily = df.copy()
    daily.index = daily.index.date
    daily.index.name = "Date"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        daily.round(4).to_excel(xw, sheet_name=f"{ticker}_Daily")
        yearly_summary(df).to_excel(xw, sheet_name="Yearly_Summary")
        for ws in xw.book.worksheets:
            ws.column_dimensions["A"].width = 12
            for col in "BCDEFGH":
                ws.column_dimensions[col].width = 16
            ws.freeze_panes = "B2"
    return path


def load_excel(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df
