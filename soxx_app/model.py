"""3) XGBRegressor 기반 미래 3개월 가격 예측.

- TR: 20년 전 ~ 1년 전 / TE: 1년 전 ~ 어제
- 타깃: 다음 거래일 로그수익률 → 예측가격 = 전일 종가 × exp(예측 수익률)
  (트리 모델은 학습 범위를 벗어난 가격 수준을 외삽하지 못하므로 수익률을 학습)
- 평가: TE 구간 1일 선행(one-step-ahead) 예측가격의 MAPE (목표 < 5%)
- 미래 3개월: 전체 데이터로 재학습 후 63거래일 재귀(recursive) 예측
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from xgboost import XGBRegressor

HORIZON = 63  # 약 3개월(거래일)
LAGS = 10
MAPE_TARGET = 5.0

XGB_PARAMS = dict(
    n_estimators=300, max_depth=2, learning_rate=0.01, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=50, reg_lambda=1.0,
    objective="reg:squarederror", random_state=42, n_jobs=-1,
)


def make_features(close: pd.Series) -> pd.DataFrame:
    """시점 t까지의 종가만 사용하는 피처 (미래 정보 누출 없음)."""
    c = close.astype(float)
    lr = np.log(c).diff()
    f = {f"ret_lag{i}": lr.shift(i - 1) for i in range(1, LAGS + 1)}
    for w in (5, 10, 20, 60, 120, 200):
        f[f"ma_gap{w}"] = c / c.rolling(w).mean() - 1
    for w in (5, 20, 60):
        f[f"vol{w}"] = lr.rolling(w).std()
    for w in (20, 60, 120):
        f[f"mom{w}"] = c / c.shift(w) - 1
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    f["dd252"] = c / c.rolling(252, min_periods=1).max() - 1
    return pd.DataFrame(f, index=c.index)


def make_dataset(close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    X = make_features(close)
    y = np.log(close.astype(float)).diff().shift(-1).rename("target")  # t+1 로그수익률
    return X, y


def split_dates(index: pd.DatetimeIndex, today: date | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(TE 시작=1년 전, TE 끝=어제)."""
    today = pd.Timestamp(today or date.today())
    return today - pd.DateOffset(years=1), today - pd.Timedelta(days=1)


def mape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


@dataclass
class EvalResult:
    mape: float
    rmse: float
    naive_mape: float
    direction_acc: float
    n_train: int
    n_test: int
    train_period: str
    test_period: str
    test_frame: pd.DataFrame  # Date, Actual, Predicted
    recursive_mape: float | None = None

    @property
    def passed(self) -> bool:
        return self.mape < MAPE_TARGET

    def summary(self) -> str:
        s = [
            f"TR 기간: {self.train_period} ({self.n_train:,}일)",
            f"TE 기간: {self.test_period} ({self.n_test:,}일)",
            f"모델: XGBRegressor (타깃: 다음날 로그수익률)",
            f"TE MAPE (1일 선행 가격예측): {self.mape:.2f}%  → 목표 5% 미만 {'달성 ✅' if self.passed else '미달 ❌'}",
            f"TE RMSE: ${self.rmse:,.2f}",
            f"참고) 단순 전일종가 모델 MAPE: {self.naive_mape:.2f}%",
            f"참고) 방향 적중률: {self.direction_acc:.1f}%",
        ]
        if self.recursive_mape is not None:
            s.append(f"참고) TE 첫 {HORIZON}거래일 재귀(3개월) 예측 MAPE: {self.recursive_mape:.2f}%")
        return "\n".join(s)


def _fit(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    m = XGBRegressor(**XGB_PARAMS)
    m.fit(X.values, y.values)
    return m


def recursive_forecast(model: XGBRegressor, close: pd.Series, steps: int = HORIZON) -> pd.Series:
    """예측 수익률로 가격 경로를 한 스텝씩 이어붙이며 `steps` 거래일 앞까지 예측."""
    hist = close.astype(float).iloc[-300:].copy()
    out_idx, out_val = [], []
    for _ in range(steps):
        x = make_features(hist).iloc[[-1]].values
        r = float(model.predict(x)[0])
        nxt = hist.index[-1] + BDay(1)
        val = hist.iloc[-1] * np.exp(r)
        hist.loc[nxt] = val
        out_idx.append(nxt)
        out_val.append(val)
    return pd.Series(out_val, index=pd.DatetimeIndex(out_idx, name="Date"), name="Forecast")


def evaluate(df: pd.DataFrame, today: date | None = None, years: int = 20) -> tuple[EvalResult, XGBRegressor]:
    close = df["Close"].astype(float)
    te_start, te_end = split_dates(close.index, today)
    tr_start = pd.Timestamp(today or date.today()) - pd.DateOffset(years=years)
    X, y = make_dataset(close)
    data = X.join(y).dropna()
    # 타깃(t+1)이 TE 구간에 들어가는 행은 학습에서 제외 → TR/TE 경계 누출 방지
    next_date = pd.Series(close.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=close.index)
    tr = data[(data.index >= tr_start) & (next_date.reindex(data.index) < te_start)]
    model = _fit(tr.drop(columns="target"), tr["target"])

    # TE: t+1 날짜가 [te_start, te_end] 인 예측
    feat = X.dropna()
    nd = next_date.reindex(feat.index)
    te_rows = feat[(nd >= te_start) & (nd <= te_end)]
    pred_r = model.predict(te_rows.values)
    base = close.loc[te_rows.index].values
    pred_price = base * np.exp(pred_r)
    tgt_dates = nd.loc[te_rows.index].values
    actual = close.loc[tgt_dates].values
    frame = pd.DataFrame({"Actual": actual, "Predicted": pred_price}, index=pd.DatetimeIndex(tgt_dates, name="Date"))

    # 참고: TE 시작 시점부터 3개월 재귀 예측의 MAPE
    hist = close[close.index < te_start]
    rec = recursive_forecast(model, hist, HORIZON)
    act_rec = close[close.index >= te_start].iloc[:HORIZON]
    rec_mape = mape(act_rec.values, rec.values[: len(act_rec)]) if len(act_rec) else None

    res = EvalResult(
        mape=mape(actual, pred_price),
        rmse=float(np.sqrt(np.mean((actual - pred_price) ** 2))),
        naive_mape=mape(actual, base),
        direction_acc=float(np.mean(np.sign(pred_r) == np.sign(np.log(actual / base))) * 100),
        n_train=len(tr), n_test=len(frame),
        train_period=f"{tr.index[0]:%Y-%m-%d} ~ {tr.index[-1]:%Y-%m-%d}",
        test_period=f"{frame.index[0]:%Y-%m-%d} ~ {frame.index[-1]:%Y-%m-%d}",
        test_frame=frame, recursive_mape=rec_mape,
    )
    return res, model


def forecast_future(df: pd.DataFrame, steps: int = HORIZON, today: date | None = None) -> pd.DataFrame:
    """전체 데이터(TR+TE)로 재학습 후 미래 `steps` 거래일 예측 + 과거 잔차 기반 밴드."""
    close = df["Close"].astype(float)
    _, te_end = split_dates(close.index, today)
    close = close[close.index <= te_end]
    X, y = make_dataset(close)
    data = X.join(y).dropna()
    model = _fit(data.drop(columns="target"), data["target"])
    path = recursive_forecast(model, close, steps)
    sigma = float(np.std(data["target"].iloc[-252:]))  # 최근 1년 일간 변동성
    h = np.arange(1, steps + 1)
    band = 1.96 * sigma * np.sqrt(h)
    return pd.DataFrame({
        "Forecast": path.values,
        "Lower95": path.values * np.exp(-band),
        "Upper95": path.values * np.exp(band),
    }, index=path.index)


# ---------------------------------------------------------------- 시각화
def fig_test(res: EvalResult):
    from matplotlib.figure import Figure
    from .analysis import BLUE, ORANGE, _style
    fig = Figure(figsize=(10, 4.2), dpi=100, facecolor="white")
    ax = fig.add_subplot()
    f = res.test_frame
    ax.plot(f.index, f["Actual"], color=BLUE, linewidth=1.6, label="Actual")
    ax.plot(f.index, f["Predicted"], color=ORANGE, linewidth=1.4, label="XGBR prediction (1-day ahead)")
    _style(ax, f"Test period: actual vs predicted  (MAPE {res.mape:.2f}%)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.margins(x=0)
    fig.tight_layout()
    return fig


def fig_forecast(df: pd.DataFrame, fc: pd.DataFrame, lookback_days: int = 252):
    from matplotlib.figure import Figure
    from .analysis import BLUE, ORANGE, _style
    close = df["Close"].astype(float).iloc[-lookback_days:]
    fig = Figure(figsize=(10, 4.2), dpi=100, facecolor="white")
    ax = fig.add_subplot()
    ax.plot(close.index, close, color=BLUE, linewidth=1.6, label="Actual (last 1y)")
    joined = pd.concat([close.iloc[-1:], fc["Forecast"]])
    ax.plot(joined.index, joined.values, color=ORANGE, linewidth=2, label="3-month forecast")
    ax.fill_between(fc.index, fc["Lower95"], fc["Upper95"], color=ORANGE, alpha=0.15, linewidth=0,
                    label="95% band (recent volatility)")
    last = fc.iloc[-1]
    ax.annotate(f"${last['Forecast']:,.2f}\n{fc.index[-1]:%Y-%m-%d}", xy=(fc.index[-1], last["Forecast"]),
                xytext=(-60, 15), textcoords="offset points", fontsize=9)
    _style(ax, "SOXX 3-month forecast (XGBR, recursive)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.margins(x=0)
    fig.tight_layout()
    return fig
