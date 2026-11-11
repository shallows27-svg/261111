import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def synthetic_df():
    """오늘 기준 21년치 가상 주가(GBM) — 네트워크 없이 테스트."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(date.today() - pd.DateOffset(years=21), date.today() - timedelta(days=1))
    r = rng.normal(0.0006, 0.02, len(idx))
    close = 10 * np.exp(np.cumsum(r))
    df = pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.003, len(idx))),
        "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Adj Close": close, "Volume": rng.integers(1e5, 1e6, len(idx)),
    }, index=pd.DatetimeIndex(idx, name="Date"))
    return df


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", help="실제 Yahoo Finance 데이터로 테스트")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live"):
        skip = pytest.mark.skip(reason="--live 옵션 필요")
        for it in items:
            if "live" in it.keywords:
                it.add_marker(skip)
