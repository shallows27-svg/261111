"""필라델피아 반도체 ETF(SOXX) 분석 · 예측 GUI.

실행:  python app.py
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import tkinter as tk  # noqa: E402
from tkinter import filedialog, font as tkfont, messagebox, ttk  # noqa: E402

import pandas as pd  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

from soxx_app import DEFAULT_TICKER, analysis, data, model  # noqa: E402

YEARS = 20
WARMUP_YEARS = 1  # 이동평균(200일) 등 피처 계산용 추가 데이터


KOREAN_FONTS = ["Malgun Gothic", "맑은 고딕", "Apple SD Gothic Neo", "AppleGothic", "NanumGothic",
                "Noto Sans CJK KR", "Noto Sans KR", "UnDotum", "WenQuanYi Zen Hei"]


def apply_korean_font(root: tk.Misc, size: int = 10) -> str | None:
    """설치된 한글 폰트를 Tk 기본 폰트로 지정 (한글 깨짐 방지)."""
    families = set(tkfont.families(root))
    for fam in KOREAN_FONTS:
        if fam in families:
            for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont", "TkFixedFont"):
                try:
                    tkfont.nametofont(name).configure(family=fam, size=size)
                except tk.TclError:
                    pass
            return fam
    return None


class App(tk.Tk):
    def __init__(self, out_dir: str | Path = "output"):
        super().__init__()
        self.title("필라델피아 반도체 ETF (SOXX) 분석 · 예측")
        self.geometry("1180x820")
        self.minsize(900, 640)
        apply_korean_font(self)

        self.full_df: pd.DataFrame | None = None  # 워밍업 포함
        self.df: pd.DataFrame | None = None       # 최근 20년
        self.eval_result = None
        self.forecast = None
        self._q: queue.Queue = queue.Queue()
        self._busy = False

        self.ticker = tk.StringVar(value=DEFAULT_TICKER)
        self.out_dir = tk.StringVar(value=str(Path(out_dir).resolve()))
        self.status = tk.StringVar(value="준비 완료 — ① 데이터 다운로드부터 시작하세요.")

        self._build()
        self.after(100, self._poll)

    # ------------------------------------------------------------ UI
    def _build(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Accent.TButton", padding=(10, 6))

        top = ttk.Frame(self, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="티커").pack(side="left")
        ttk.Entry(top, textvariable=self.ticker, width=8).pack(side="left", padx=(4, 12))
        ttk.Label(top, text="저장 폴더").pack(side="left")
        ttk.Entry(top, textvariable=self.out_dir, width=46).pack(side="left", padx=4)
        ttk.Button(top, text="찾기…", command=self._choose_dir).pack(side="left")

        btns = ttk.Frame(self, padding=(10, 0, 10, 8))
        btns.pack(fill="x")
        self.buttons = [
            ttk.Button(btns, text="① 20년 데이터 다운로드 → 엑셀 저장", style="Accent.TButton",
                       command=lambda: self._run(self._task_download)),
            ttk.Button(btns, text="엑셀 불러오기", command=self._load_excel),
            ttk.Button(btns, text="② 분석 · 인사이트 · 시각화 3종", style="Accent.TButton",
                       command=lambda: self._run(self._task_analyze)),
            ttk.Button(btns, text="③ XGBR 3개월 예측", style="Accent.TButton",
                       command=lambda: self._run(self._task_predict)),
            ttk.Button(btns, text="전체 실행 (①→②→③)", command=lambda: self._run(self._task_all)),
            ttk.Button(btns, text="저장 폴더 열기", command=self._open_dir),
        ]
        for b in self.buttons:
            b.pack(side="left", padx=(0, 6))

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10)

        # 데이터 탭
        self.tab_data = ttk.Frame(self.nb, padding=6)
        self.nb.add(self.tab_data, text="데이터")
        self.data_info = ttk.Label(self.tab_data, text="데이터 없음", anchor="w")
        self.data_info.pack(fill="x", pady=(0, 6))
        cols = ["Date"] + data.COLUMNS
        frame = ttk.Frame(self.tab_data)
        frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(frame, columns=cols, show="headings")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="e")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # 인사이트 탭
        self.tab_ins = ttk.Frame(self.nb, padding=6)
        self.nb.add(self.tab_ins, text="인사이트")
        self.ins_text = tk.Text(self.tab_ins, height=11, wrap="word", font=("TkDefaultFont", 11),
                                relief="flat", background="#fafaf8")
        self.ins_text.pack(fill="x")
        self.ins_nb = ttk.Notebook(self.tab_ins)
        self.ins_nb.pack(fill="both", expand=True, pady=(6, 0))

        # 예측 탭
        self.tab_pred = ttk.Frame(self.nb, padding=6)
        self.nb.add(self.tab_pred, text="예측")
        self.pred_text = tk.Text(self.tab_pred, height=10, wrap="word", font=("TkDefaultFont", 11),
                                 relief="flat", background="#fafaf8")
        self.pred_text.pack(fill="x")
        self.pred_nb = ttk.Notebook(self.tab_pred)
        self.pred_nb.pack(fill="both", expand=True, pady=(6, 0))

        bottom = ttk.Frame(self, padding=(10, 6))
        bottom.pack(fill="x")
        self.progress = ttk.Progressbar(bottom, mode="indeterminate", length=160)
        self.progress.pack(side="right")
        ttk.Label(bottom, textvariable=self.status, anchor="w").pack(side="left", fill="x", expand=True)

    def _choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.out_dir.get())
        if d:
            self.out_dir.set(d)

    def _open_dir(self):
        d = Path(self.out_dir.get())
        d.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(d)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{d}"')
            else:
                os.system(f'xdg-open "{d}" >/dev/null 2>&1 &')
        except Exception as e:  # pragma: no cover
            messagebox.showerror("오류", str(e))

    def _load_excel(self):
        path = filedialog.askopenfilename(initialdir=self.out_dir.get(),
                                          filetypes=[("Excel", "*.xlsx"), ("All", "*.*")])
        if not path:
            return
        try:
            self._set_data(data.load_excel(path))
            self.status.set(f"엑셀 불러옴: {path}")
        except Exception as e:
            messagebox.showerror("불러오기 실패", str(e))

    # ------------------------------------------------------------ 작업 실행 (백그라운드 스레드)
    def _run(self, task):
        if self._busy:
            return
        self._busy = True
        for b in self.buttons:
            b.state(["disabled"])
        self.progress.start(12)

        def worker():
            try:
                task()
            except Exception as e:
                self._q.put(("error", f"{e}\n\n{traceback.format_exc(limit=3)}"))
            finally:
                self._q.put(("done", None))

        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "status":
                    self.status.set(payload)
                elif kind == "ui":
                    payload()
                elif kind == "error":
                    self.status.set("오류 발생")
                    messagebox.showerror("오류", payload)
                elif kind == "done":
                    self._busy = False
                    self.progress.stop()
                    for b in self.buttons:
                        b.state(["!disabled"])
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _ui(self, fn):
        self._q.put(("ui", fn))

    def _say(self, msg):
        self._q.put(("status", msg))

    @property
    def _out(self) -> Path:
        p = Path(self.out_dir.get())
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _need_data(self):
        if self.full_df is None:
            self._task_download()

    # ① 다운로드 & 엑셀
    def _task_download(self):
        t = self.ticker.get().strip().upper() or DEFAULT_TICKER
        self._say(f"{t} 최근 {YEARS}년 데이터 다운로드 중…")
        full = data.download(t, years=YEARS, warmup_years=WARMUP_YEARS)
        df = data.trim(full, YEARS)
        path = data.save_excel(df, self._out / f"{t}_{YEARS}y_daily.xlsx", ticker=t)
        self.full_df = full
        self._ui(lambda: self._set_data(full, saved=path))
        self._say(f"① 완료: {len(df):,}거래일 저장 → {path}")

    def _set_data(self, full: pd.DataFrame, saved: Path | None = None):
        self.full_df = full
        self.df = data.trim(full, YEARS)
        df = self.df
        self.tree.delete(*self.tree.get_children())
        for d, row in df.iloc[::-1].iterrows():
            self.tree.insert("", "end", values=[f"{d:%Y-%m-%d}"] +
                             [f"{row[c]:,.0f}" if c == "Volume" else f"{row[c]:,.2f}" for c in data.COLUMNS if c in row])
        info = f"{self.ticker.get()} | {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d} | {len(df):,}거래일"
        if saved:
            info += f" | 저장: {saved}"
        self.data_info.config(text=info)
        self.nb.select(self.tab_data)

    # ② 분석
    def _task_analyze(self):
        self._need_data()
        self._say("② 데이터 분석 및 시각화 생성 중…")
        df = data.trim(self.full_df, YEARS)
        ins = analysis.analyze(df)
        figs = analysis.make_figures(df)
        paths = analysis.save_figures(figs, self._out)
        with pd.ExcelWriter(self._out / "analysis_insights.xlsx") as xw:
            pd.Series(ins.stats, name="값").to_frame().to_excel(xw, sheet_name="요약통계")
            ins.regime.to_excel(xw, sheet_name="200일선_국면비교")
            ins.drawdowns.to_excel(xw, sheet_name="상위낙폭")
            pd.DataFrame({"인사이트": [ins.headline] + ins.lines}).to_excel(xw, sheet_name="인사이트", index=False)
        (self._out / "insights.txt").write_text(ins.text(), encoding="utf-8")

        def show():
            self._set_text(self.ins_text, ins.text() + "\n\n[200일선 국면 비교]\n" + ins.regime.to_string())
            self._show_figs(self.ins_nb, {
                "① 추세 국면(200일선)": figs["1_trend_regime.png"],
                "② 낙폭(Drawdown)": figs["2_drawdown.png"],
                "③ 연도별 수익률": figs["3_annual_returns.png"],
            })
            self.nb.select(self.tab_ins)

        self._ui(show)
        self._say(f"② 완료: 시각화 {len(paths)}개 저장 → {self._out}")

    # ③ 예측
    def _task_predict(self):
        self._need_data()
        self._say("③ XGBR 학습(TR: 20년 전~1년 전) 및 TE 평가 중…")
        full = self.full_df
        res, _ = model.evaluate(full, years=YEARS)
        self._say("③ 전체 데이터 재학습 후 미래 3개월 예측 중…")
        fc = model.forecast_future(full)
        f_test, f_fc = model.fig_test(res), model.fig_forecast(full, fc)
        f_test.savefig(self._out / "4_test_actual_vs_pred.png", dpi=150, facecolor="white")
        f_fc.savefig(self._out / "5_forecast_3m.png", dpi=150, facecolor="white")
        with pd.ExcelWriter(self._out / "forecast_3m.xlsx") as xw:
            out = fc.copy()
            out.index = out.index.date
            out.index.name = "Date"
            out.round(2).to_excel(xw, sheet_name="Forecast_3M")
            tf = res.test_frame.copy()
            tf["APE(%)"] = (tf["Predicted"] / tf["Actual"] - 1).abs() * 100
            tf.index = tf.index.date
            tf.index.name = "Date"
            tf.round(3).to_excel(xw, sheet_name="Test_Eval")
            pd.Series({"MAPE(%)": res.mape, "RMSE": res.rmse, "Naive MAPE(%)": res.naive_mape,
                       "방향 적중률(%)": res.direction_acc, "TR": res.train_period, "TE": res.test_period},
                      name="값").to_frame().to_excel(xw, sheet_name="Metrics")
        self.eval_result, self.forecast = res, fc
        last = full["Close"].iloc[-1]
        end = fc.iloc[-1]
        msg = (res.summary() + "\n\n"
               f"[미래 3개월 예측] 기준 종가 {full.index[-1]:%Y-%m-%d} ${last:,.2f}\n"
               f"→ {fc.index[-1]:%Y-%m-%d} 예측 ${end['Forecast']:,.2f} ({(end['Forecast']/last-1)*100:+.1f}%), "
               f"95% 범위 ${end['Lower95']:,.2f} ~ ${end['Upper95']:,.2f}\n"
               "※ 일간 수익률은 예측력이 낮아 장기 재귀 예측은 추세 방향 참고용입니다. 투자 판단의 근거로 사용하지 마세요.")

        def show():
            self._set_text(self.pred_text, msg)
            self._show_figs(self.pred_nb, {"미래 3개월 예측": f_fc, "TE 실제 vs 예측": f_test})
            self.nb.select(self.tab_pred)

        self._ui(show)
        self._say(f"③ 완료: TE MAPE {res.mape:.2f}% ({'목표 달성' if res.passed else '목표 미달'}) — 결과 저장 → {self._out}")

    def _task_all(self):
        self._task_download()
        self._task_analyze()
        self._task_predict()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _set_text(widget: tk.Text, text: str):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.config(state="disabled")

    @staticmethod
    def _show_figs(nb: ttk.Notebook, figs: dict):
        for tab in nb.tabs():
            nb.forget(tab)
        for title, fig in figs.items():
            frame = ttk.Frame(nb)
            nb.add(frame, text=title)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
