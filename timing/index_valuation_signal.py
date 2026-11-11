# -*- coding: utf-8 -*-
"""指数估值 + 一致预期择时信号（db_wind/IndexValuation.pq 新源）。

链路位置：
    1. 读取 db_wind/IndexValuation.pq 主要宽基指数的 PE_TTM/DIVIDEND_YIELD/
       YOYPROFIT_Y1（一致预期净利润增速）
    2. 计算各指数估值指标的 5 年滚动分位
    3. 输出历史信号面板与最新截面 csv，供风格轮动/择时参考

输入数据：
    - IndexValuation.pq（Date/Ticker 索引，PE_TTM 倍、DIVIDEND_YIELD 百分数、
      YOYPROFIT_Y1 百分数），2026-09-09 新入库

关键口径：
    - 估值分位为滚动窗口内当日值的历史百分位（0~1），越高越贵
    - 一致预期增速直接取 YOYPROFIT_Y1，不另做变换
    - 信号仅描述状态，不含交易规则；接入策略前按 strategy_workflow 验证

输出：
    - Config().allocation_path/index_valuation/index_valuation_signal.csv（全历史）
    - 同目录 latest.csv（最新截面）
"""
import os

import pandas as pd

from config.config import Config
from utility.logout import LogOut

INDEX_CODES = {
    "000016.SH": "上证50",
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "000906.SH": "中证800",
}
ROLL_WINDOW = 5 * 244   # 5 年滚动窗口（交易日近似）


def build_signal():
    """计算指数估值分位与一致预期信号，落盘全历史与最新截面。"""

    logout = LogOut("IndexValuationSignal")
    logout.reset_log()
    path = os.path.join(Config().db_wind_path, "IndexValuation.pq")
    df = pd.read_parquet(path, columns=["PE_TTM", "PB_LF", "DIVIDEND_YIELD",
                                        "YOYPROFIT_Y1", "YOYGR_Y1"])
    df = df[df.index.get_level_values("Ticker").isin(INDEX_CODES)]

    out = {}
    for col in ["PE_TTM", "PB_LF"]:
        panel = df[col].unstack(level="Ticker")
        out[f"{col}_pct"] = panel.rolling(ROLL_WINDOW, min_periods=244).apply(
            lambda x: x.rank(pct=True).iloc[-1], raw=False)
    # 股息率分位越高越"便宜"，取 1-分位 与估值分位同向
    dy = df["DIVIDEND_YIELD"].unstack(level="Ticker")
    out["DY_pct"] = dy.rolling(ROLL_WINDOW, min_periods=244).apply(
        lambda x: x.rank(pct=True).iloc[-1], raw=False)
    out["DY_pct_cheap"] = 1 - out["DY_pct"]
    for col in ["YOYPROFIT_Y1", "YOYGR_Y1"]:
        out[col] = df[col].unstack(level="Ticker")

    sig = pd.concat(out, axis=1)
    sig.index.names = ["Date"]
    sig = sig.rename(columns=INDEX_CODES)
    sig = sig.dropna(how="all")

    out_dir = os.path.join(Config().allocation_path, "index_valuation")
    os.makedirs(out_dir, exist_ok=True)
    hist_file = os.path.join(out_dir, "index_valuation_signal.csv")
    latest_file = os.path.join(out_dir, "latest.csv")
    sig.to_csv(hist_file, encoding="utf-8-sig")
    latest = sig.tail(1).T
    latest.columns = [str(sig.index[-1].date())]
    latest.to_csv(latest_file, encoding="utf-8-sig")
    logout.logger.info(f"信号落盘：{hist_file}（{len(sig)} 行）与 {latest_file}")
    return sig


if __name__ == "__main__":

    build_signal()
