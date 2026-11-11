import pytest
import pandas as pd

from soxx_app import analysis


def test_drawdown():
    p = pd.Series([100, 120, 60, 130.0])
    assert analysis.drawdown(p).tolist() == [0, 0, -0.5, 0]


def test_top_drawdowns():
    p = pd.Series([100, 120, 60, 130.0, 52, 70], index=pd.date_range("2020-01-01", periods=6))
    t = analysis.top_drawdowns(p)
    assert t["최대낙폭"].iloc[0] == pytest.approx(-0.6)
    assert pd.isna(t["회복"].iloc[0])  # 미회복 구간


def test_analyze_and_figures(synthetic_df, tmp_path):
    ins = analysis.analyze(synthetic_df)
    assert ins.headline and len(ins.lines) == 5
    assert set(ins.regime.index) == {"200일선 위", "200일선 아래"}
    figs = analysis.make_figures(synthetic_df)
    assert len(figs) == 3
    paths = analysis.save_figures(figs, tmp_path)
    assert all(p.exists() and p.stat().st_size > 10_000 for p in paths)
