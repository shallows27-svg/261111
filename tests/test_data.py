from datetime import date

import pandas as pd

from soxx_app import data


def test_period_bounds():
    s, e = data.period_bounds(20, date(2026, 11, 11))
    assert s == date(2006, 11, 11) and e == date(2026, 11, 10)


def test_normalize_multiindex():
    idx = pd.date_range("2020-01-01", periods=3)
    cols = pd.MultiIndex.from_product([["Close", "Open", "High", "Low", "Volume", "Adj Close"], ["SOXX"]])
    raw = pd.DataFrame(1.0, index=idx, columns=cols)
    out = data.normalize(raw)
    assert list(out.columns) == data.COLUMNS and out.index.name == "Date"


def test_trim_and_excel_roundtrip(synthetic_df, tmp_path):
    df = data.trim(synthetic_df, 20)
    assert df.index[0] >= pd.Timestamp(data.period_bounds(20)[0])
    assert df.index[-1] < pd.Timestamp(date.today())
    path = data.save_excel(df, tmp_path / "x.xlsx")
    back = data.load_excel(path)
    assert len(back) == len(df)
    assert abs(back["Close"].iloc[-1] - df["Close"].iloc[-1]) < 1e-3
    assert "Yearly_Summary" in pd.ExcelFile(path).sheet_names
