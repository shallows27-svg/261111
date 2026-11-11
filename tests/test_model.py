from datetime import date

import numpy as np
import pandas as pd
import pytest

from soxx_app import data, model


def test_features_no_lookahead(synthetic_df):
    c = synthetic_df["Close"]
    f1 = model.make_features(c)
    c2 = c.copy()
    c2.iloc[-50:] *= 3  # 미래 값 변경
    f2 = model.make_features(c2)
    pd.testing.assert_frame_equal(f1.iloc[:-50], f2.iloc[:-50])


def test_mape():
    assert model.mape([100, 200], [110, 180]) == pytest.approx(10.0)


def test_evaluate_split_and_mape(synthetic_df):
    res, _ = model.evaluate(synthetic_df)
    te_start, te_end = model.split_dates(synthetic_df.index)
    tf = res.test_frame
    assert tf.index.min() >= te_start and tf.index.max() <= te_end
    tr_end = pd.Timestamp(res.train_period.split(" ~ ")[1])
    assert tr_end < te_start  # TR/TE 겹침 없음
    assert 200 < res.n_test < 270  # 약 1년
    assert res.mape < model.MAPE_TARGET


def test_forecast_future(synthetic_df):
    fc = model.forecast_future(synthetic_df)
    assert len(fc) == model.HORIZON
    assert fc.index[0] > synthetic_df.index[-1]
    assert (fc["Lower95"] <= fc["Forecast"]).all() and (fc["Forecast"] <= fc["Upper95"]).all()
    assert np.isfinite(fc.values).all()


@pytest.mark.live
def test_live_soxx_mape_below_5():
    full = data.download("SOXX", years=20, warmup_years=1)
    assert data.trim(full, 20).index[0].year == date.today().year - 20
    res, _ = model.evaluate(full)
    print("\n" + res.summary())
    assert res.mape < 5.0
