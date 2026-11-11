"""2) 데이터 분석 · 핵심 인사이트 도출 · 시각화 3종."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

TRADING_DAYS = 252
FWD_3M = 63  # 약 3개월 거래일

# 검증된 기본 팔레트(light)
BLUE, ORANGE, RED, AQUA = "#2a78d6", "#eb6834", "#e34948", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


@dataclass
class Insights:
    stats: dict = field(default_factory=dict)
    regime: pd.DataFrame | None = None
    drawdowns: pd.DataFrame | None = None
    headline: str = ""
    lines: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join([f"★ 핵심 인사이트: {self.headline}", ""] + [f"• {l}" for l in self.lines])


def price_series(df: pd.DataFrame) -> pd.Series:
    """수익률 계산은 배당 반영 수정종가 기준."""
    return (df["Adj Close"] if "Adj Close" in df else df["Close"]).astype(float)


def drawdown(p: pd.Series) -> pd.Series:
    return p / p.cummax() - 1


def top_drawdowns(p: pd.Series, n: int = 5) -> pd.DataFrame:
    dd = drawdown(p)
    rows, in_dd, start = [], False, None
    for t, v in dd.items():
        if v < 0 and not in_dd:
            in_dd, start = True, t
        elif v == 0 and in_dd:
            seg = dd[start:t]
            rows.append((start, seg.idxmin(), t, seg.min()))
            in_dd = False
    if in_dd:
        seg = dd[start:]
        rows.append((start, seg.idxmin(), pd.NaT, seg.min()))
    out = pd.DataFrame(rows, columns=["시작", "저점", "회복", "최대낙폭"])
    out = out.sort_values("최대낙폭").head(n).reset_index(drop=True)
    out["회복까지(거래일)"] = [
        (len(p[s:r]) - 1) if pd.notna(r) else np.nan for s, r in zip(out["시작"], out["회복"])
    ]
    return out


def regime_table(p: pd.Series, window: int = 200) -> pd.DataFrame:
    """200일 이동평균 위/아래 국면별 성과 비교 (다음날 수익률 기준 → 미래 정보 누출 없음)."""
    ma = p.rolling(window).mean()
    ret_next = p.pct_change().shift(-1)
    fwd3m = p.shift(-FWD_3M) / p - 1
    dd = drawdown(p)
    frame = pd.DataFrame({"above": p > ma, "r": ret_next, "f": fwd3m, "dd": dd})[ma.notna()]
    rows = {}
    for key, name in [(True, "200일선 위"), (False, "200일선 아래")]:
        s = frame[frame["above"] == key]
        r = s["r"].dropna()
        rows[name] = {
            "비중(%)": len(s) / len(frame) * 100,
            "연환산 수익률(%)": r.mean() * TRADING_DAYS * 100,
            "연환산 변동성(%)": r.std() * np.sqrt(TRADING_DAYS) * 100,
            "샤프(무위험0)": r.mean() / r.std() * np.sqrt(TRADING_DAYS),
            "3개월 후 손실확률(%)": (s["f"].dropna() < 0).mean() * 100,
            "-5% 이하 급락일 비중(%)": (r < -0.05).mean() * 100,
        }
    return pd.DataFrame(rows).T.round(2)


def analyze(df: pd.DataFrame) -> Insights:
    p = price_series(df)
    r = p.pct_change().dropna()
    years = (p.index[-1] - p.index[0]).days / 365.25
    cagr = (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    dd = drawdown(p)
    yearly = p.groupby(p.index.year).agg(["first", "last"])
    yr_ret = (yearly["last"] / yearly["first"] - 1)
    reg = regime_table(p)
    dds = top_drawdowns(p)

    stats = {
        "기간": f"{p.index[0]:%Y-%m-%d} ~ {p.index[-1]:%Y-%m-%d}",
        "총 수익률(%)": (p.iloc[-1] / p.iloc[0] - 1) * 100,
        "CAGR(%)": cagr * 100,
        "연환산 변동성(%)": vol * 100,
        "최대낙폭 MDD(%)": dd.min() * 100,
        "MDD 저점일": f"{dd.idxmin():%Y-%m-%d}",
        "상승 연도 수": f"{(yr_ret > 0).sum()} / {len(yr_ret)}",
    }
    up, dn = reg.loc["200일선 위"], reg.loc["200일선 아래"]
    vol_x = dn["연환산 변동성(%)"] / up["연환산 변동성(%)"]
    crash_x = dn["-5% 이하 급락일 비중(%)"] / max(up["-5% 이하 급락일 비중(%)"], 1e-9)
    better = "위" if up["샤프(무위험0)"] > dn["샤프(무위험0)"] else "아래"
    headline = (
        f"SOXX는 20년간 연 {cagr*100:.1f}% 성장했지만 MDD {dd.min()*100:.0f}%의 '고수익·고위험' 자산이며, "
        f"위험은 200일 이동평균 아래 국면에 집중된다 (변동성 {vol_x:.1f}배, -5% 급락일 {crash_x:.1f}배, "
        f"3개월 후 손실확률 {dn['3개월 후 손실확률(%)']:.0f}% vs {up['3개월 후 손실확률(%)']:.0f}%). "
        f"위험조정수익(샤프)은 200일선 {better} 국면이 우세 → 200일선이 리스크 관리의 핵심 신호"
    )
    lines = [
        f"{stats['기간']} 총 수익률 {stats['총 수익률(%)']:,.0f}%, CAGR {cagr*100:.1f}%, 연 변동성 {vol*100:.1f}%.",
        f"200일선 위(전체의 {up['비중(%)']:.0f}%): 연환산 {up['연환산 수익률(%)']:.1f}%, 변동성 {up['연환산 변동성(%)']:.1f}%, 샤프 {up['샤프(무위험0)']:.2f}.",
        f"200일선 아래({dn['비중(%)']:.0f}%): 연환산 {dn['연환산 수익률(%)']:.1f}%, 변동성 {dn['연환산 변동성(%)']:.1f}%, -5% 급락일 비중 {dn['-5% 이하 급락일 비중(%)']:.2f}% (위 국면 {up['-5% 이하 급락일 비중(%)']:.2f}%).",
        f"최대낙폭 {dd.min()*100:.1f}% ({dd.idxmin():%Y-%m}). 상위 5개 낙폭 중 -30% 이상 하락 {int((dds['최대낙폭'] <= -0.30).sum())}회 → 높은 장기수익의 대가는 깊은 하락.",
        f"상승 연도 {stats['상승 연도 수']}; 최고 {yr_ret.idxmax()}년 {yr_ret.max()*100:+.1f}%, 최저 {yr_ret.idxmin()}년 {yr_ret.min()*100:+.1f}%.",
    ]
    return Insights(stats=stats, regime=reg, drawdowns=dds, headline=headline, lines=lines)


# ---------------------------------------------------------------- 시각화
def _style(ax, title: str):
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_trend_regime(df: pd.DataFrame) -> Figure:
    """시각화 1: 가격(로그) + 200일 이동평균, 200일선 아래 국면 음영."""
    p = price_series(df)
    ma = p.rolling(200).mean()
    fig = Figure(figsize=(10, 4.6), dpi=100, facecolor="white")
    ax = fig.add_subplot()
    below = (p < ma) & ma.notna()
    ax.fill_between(p.index, p.min() * 0.9, p.max() * 1.1, where=below, color=RED, alpha=0.08,
                    linewidth=0, label="Below 200-day MA")
    ax.plot(p.index, p, color=BLUE, linewidth=1.4, label="SOXX (adj. close)")
    ax.plot(ma.index, ma, color=ORANGE, linewidth=2, label="200-day MA")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylim(p.min() * 0.9, p.max() * 1.1)
    ax.margins(x=0)
    _style(ax, "1. Trend regimes: risk concentrates below the 200-day MA (shaded, log scale)")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    return fig


def fig_drawdown(df: pd.DataFrame) -> Figure:
    """시각화 2: 고점 대비 낙폭(Underwater)."""
    p = price_series(df)
    dd = drawdown(p) * 100
    fig = Figure(figsize=(10, 4.2), dpi=100, facecolor="white")
    ax = fig.add_subplot()
    ax.fill_between(dd.index, dd, 0, color=RED, alpha=0.18, linewidth=0)
    ax.plot(dd.index, dd, color=RED, linewidth=1.2)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    i = dd.idxmin()
    ax.annotate(f"Max drawdown {dd.min():.1f}%\n{i:%Y-%m-%d}", xy=(i, dd.min()), xytext=(15, 10),
                textcoords="offset points", fontsize=9, color=INK)
    ax.margins(x=0)
    _style(ax, "2. Drawdown from previous peak — the price of high growth")
    fig.tight_layout()
    return fig


def fig_annual_returns(df: pd.DataFrame) -> Figure:
    """시각화 3: 연도별 수익률(배당 포함)."""
    p = price_series(df)
    y = p.groupby(p.index.year).agg(["first", "last"])
    ret = (y["last"] / y["first"] - 1) * 100
    fig = Figure(figsize=(10, 4.2), dpi=100, facecolor="white")
    ax = fig.add_subplot()
    colors = [BLUE if v >= 0 else RED for v in ret]
    ax.bar(ret.index.astype(str), ret.values, color=colors, width=0.7, edgecolor="white", linewidth=2)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    for k in (ret.idxmax(), ret.idxmin(), ret.index[-1]):
        v = ret[k]
        ax.annotate(f"{v:+.0f}%", xy=(str(k), v), xytext=(0, 4 if v >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.tick_params(axis="x", rotation=45)
    _style(ax, "3. Calendar-year returns (first/last year partial)")
    fig.tight_layout()
    return fig


FIGURES = {
    "1_trend_regime.png": fig_trend_regime,
    "2_drawdown.png": fig_drawdown,
    "3_annual_returns.png": fig_annual_returns,
}


def make_figures(df: pd.DataFrame) -> dict[str, Figure]:
    return {name: fn(df) for name, fn in FIGURES.items()}


def save_figures(figs: dict[str, Figure], out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figs.items():
        path = out_dir / name
        fig.savefig(path, dpi=150, facecolor="white")
        paths.append(path)
    return paths
