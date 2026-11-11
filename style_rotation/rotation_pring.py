# -*- coding: utf-8 -*-
"""
普林格周期轮动模型 —— 基于普林格经济周期六阶段（原名宏观轮动，2026-09-25 更名）

严格复现 MATLAB 原始代码:
    D:\Smart Beta rotation program\[月中调仓]Smart Beta rotation program\macrocycle.m
    D:\Smart Beta rotation program\[月中调仓]Smart Beta rotation program\macrorotate_doubleasset.m

数据来源（2026-09-22 去 WindPy 迁移，本地底表）:
    - 000016.SH  close   上证50收盘价（经济增长代理）<- db_wind/IndexEOD.pq
    - CRB.RB     close   CRB商品指数（通胀代理）    <- api_wind/global_index_eod.pq
    - S0059749   edb     中债国债到期收益率10年（利率代理）<- api_wind/macro_edb.pq
      （迁移前为 WindPy wsd/edb 在线调用；迁移后无 Wind 终端依赖，
        见 config/doc/db_data_missing.md §五）

风格指数（回测用，2026-09-25 对齐 style_model.py 四风格腿体系）:
    - h20269.CSI 红利（红利低波全收益；MATLAB 原用 H30269 价格版）
    - 980080.CNI 成长（成长100 价格版，同 style_model）
    - 480081.CNI 周期（价值100R 全收益，同 style_model 价值腿；MATLAB 原用 980081 价格版）
    - 932430CNY010.CSI 质量（中证质量全收益；MATLAB 原用 932430 价格版）
    （周期信号生成逻辑与周期→风格权重矩阵仍为 MATLAB 复现，仅腿指数口径
      与 style_model 统一；对齐前历史结果见 git 与模块研究记录）
"""
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from config.config import Config

try:
    from model_allocation.style_rotation.style_model import StyleModel
except ImportError:
    from style_model import StyleModel

warnings.filterwarnings("ignore")


# ======================================================================
# 普林格周期轮动模型
# ======================================================================

class RotationPring(object):
    """普林格周期轮动模型（普林格经济周期六阶段）— 严格复现 MATLAB"""

    # --- 数据配置（与 MATLAB macrocycle.m 一致）---
    GROWTH_CODE = "000016.SH"     # 上证50收盘价（经济增长代理）
    CRB_CODE = "CRB.RB"           # CRB商品指数（Wind WSD日度数据，通胀代理）
    BOND_CODE = "S0059749"        # 中债国债到期收益率10年（Wind EDB，利率代理）

    # --- 风格指数（2026-09-25 对齐 style_model.py 腿体系；红利/周期/质量换全收益）---
    # [Dividend, Growth, Cycle, Quality]
    STYLE_NAMES = ["红利", "成长", "周期", "质量"]
    STYLE_CODES = {
        "红利": "h20269.CSI",
        "成长": "980080.CNI",
        "周期": "480081.CNI",
        "质量": "932430CNY010.CSI",
    }

    # --- 周期 -> 风格权重 [红利, 成长, 周期, 质量]（严格复现 MATLAB 第29-55行）---
    CYCLE_WEIGHTS = {
        1:   [1, 0, 0, 1],
        1.5: [1, 0, 0, 1],
        2:   [1, 0, 1, 0],
        3:   [0, 1, 0, 1],
        4:   [0, 1, 0, 1],
        5:   [0, 0, 1, 1],
        5.5: [0, 0, 1, 1],
        6:   [1, 0, 0, 1],
    }

    CYCLE_NAMES = {
        1: "货币宽松期", 1.5: "货币宽松期(过渡)",
        2: "经济复苏期", 3: "经济上行期",
        4: "经济过热期", 5: "滞涨期", 5.5: "滞涨期(过渡)",
        6: "衰退期",
    }

    # --- 信号参数（与 MATLAB 一致）---
    MA_LONG = 60
    MA_SHORT = 20
    SMOOTH_WINDOW = 5
    BEG_DATE = "2002-01-01"

    # --- 回测参数 ---
    BACKTEST_START = "20050131"  # 2026-09-25 按回测时长原则自 2013 延至数据下限
                                 # （腿经补齐链回补到 2005-01，2005~2012 为代理腿口径）

    def __init__(self):
        self.class_name = self.__class__.__name__
        self.db_wind_path = Config().db_wind_path
        self.data_path = r"./data/output"
        self.raw_path = os.path.join(self.data_path, "raw_data")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.exists(self.raw_path):
            os.makedirs(self.raw_path)

    # ==================================================================
    # 数据加载（本地底表，2026-09-22 去 WindPy 迁移）
    # ==================================================================

    def load_data(self):
        """从本地底表加载全部数据，并保存原始数据到 raw_path

        迁移映射：上证50/风格指数 <- db_wind/IndexEOD.pq；
        CRB <- api_wind/global_index_eod.pq（1994-01 起）；
        中债10Y <- api_wind/macro_edb.pq S0059749_CNBond10Y（2002-01 起，
        EDB 下载时 Fill=Previous，与前视口径一致）。
        """
        print("=" * 60)
        print("加载数据（本地底表，无 WindPy）")
        print("=" * 60)

        import pyarrow.parquet as pq

        beg = pd.Timestamp(self.BEG_DATE)

        def _pick(close_wide, col):
            s = close_wide[col].dropna().astype(float).sort_index()
            s.index = pd.to_datetime(s.index.astype(str).str[:10])
            return s[s.index >= beg]

        # 1. 上证50 + 风格指数（IndexEOD.pq，filter 下推避免全量读 2400 万行）
        idx_codes = [self.GROWTH_CODE] + list(self.STYLE_CODES.values())
        # pyarrow filter 精确匹配且大小写敏感（库内 h 前缀代码为 h30269.CSI
        # 混合大小写）：先读 Ticker 列做大小写不敏感匹配，取库内精确代码过滤
        ticker_col = pq.read_table(
            self.db_wind_path / "IndexEOD.pq", columns=["Ticker"]).column("Ticker")
        lib_tickers = pd.unique(ticker_col.to_pandas())
        want = {c.lower() for c in idx_codes}
        filter_codes = sorted(t for t in lib_tickers if str(t).lower() in want)
        missing = want - {str(t).lower() for t in filter_codes}
        if missing:
            raise RuntimeError("IndexEOD 缺少指数: %s" % sorted(missing))
        table = pq.read_table(
            self.db_wind_path / "IndexEOD.pq", columns=["Date", "Ticker", "Close"],
            filters=[("Ticker", "in", filter_codes)])
        ie = table.to_pandas(ignore_metadata=True)
        ie["Ticker"] = ie["Ticker"].str.lower()
        close_w = ie.pivot_table(index="Date", columns="Ticker", values="Close")
        lower_map = {c.lower(): c for c in idx_codes}
        close_w = close_w.rename(
            columns={c: lower_map[c] for c in close_w.columns if c in lower_map})

        print("获取上证50(%s)收盘价..." % self.GROWTH_CODE)
        self.data1 = _pick(close_w, self.GROWTH_CODE).to_frame("SZ50")
        self._save_raw("000016_SH_close.csv", self.data1)
        print("  %s ~ %s, %d 条" % (
            self.data1.index[0].date(), self.data1.index[-1].date(), len(self.data1)))

        # 2. CRB商品指数（api_wind/global_index_eod.pq）
        print("获取CRB商品指数(%s)..." % self.CRB_CODE)
        gi = pq.read_table(
            Config().api_wind_path / "global_index_eod.pq",
            columns=["Date", "Ticker", "Close"],
            filters=[("Ticker", "=", self.CRB_CODE)]).to_pandas()
        crb = gi.set_index("Date")["Close"].astype(float).sort_index()
        crb.index = pd.to_datetime(crb.index.astype(str).str[:10])
        crb = crb[crb.index >= beg]
        self.data2 = crb.to_frame("Commodity")
        self._save_raw("CRB_RB_close.csv", self.data2)
        print("  %s ~ %s, %d 条" % (
            self.data2.index[0].date(), self.data2.index[-1].date(), len(self.data2)))

        # 3. 10年期国债收益率（api_wind/macro_edb.pq）
        print("获取10年期国债收益率(%s)..." % self.BOND_CODE)
        me = pd.read_parquet(Config().api_wind_path / "macro_edb.pq").set_index("Date")
        bond = me["S0059749_CNBond10Y"].dropna().astype(float).sort_index()
        bond.index = pd.to_datetime(bond.index.astype(str).str[:10])
        bond = bond[bond.index >= beg]
        self.data3 = bond.to_frame("BondRate")
        self._save_raw("S0059749_bond10y.csv", self.data3)
        print("  %s ~ %s, %d 条" % (
            self.data3.index[0].date(), self.data3.index[-1].date(), len(self.data3)))

        # 4. 风格指数收盘价（2026-09-25 按回测时长原则改走 StyleModel 补齐链，
        #    四腿回补到 2005-01-04 数据下限；2005~2012 段为代理腿口径）
        print("获取风格指数收盘价（StyleModel 加载+历史回补）...")
        sm = StyleModel()
        all_codes = list(self.STYLE_CODES.values()) + list(sm.fill_index.values())
        sm_data = sm.fill_price_data(sm.load_index_data(list(set(all_codes))))
        self.style_prices = {}
        style_df = pd.DataFrame()
        for name, code in self.STYLE_CODES.items():
            s = sm_data[code].dropna().astype(float)
            s.index = pd.to_datetime(s.index.astype(str).str[:10])
            s = s[s.index >= beg]
            self.style_prices[name] = s
            style_df[name] = s
            print("  %s(%s): %s ~ %s, %d 条" % (
                name, code, s.index[0].date(), s.index[-1].date(), len(s)))
        self._save_raw("style_indices_close.csv", style_df)

    def _save_raw(self, filename, df):
        """保存原始数据到 raw_path"""
        path = os.path.join(self.raw_path, filename)
        try:
            df.to_csv(path, encoding="gbk")
            print("  原始数据已保存: %s" % path)
        except PermissionError:
            print("  [警告] 文件被占用, 跳过保存: %s" % path)

    # ==================================================================
    # 信号生成（严格复现 MATLAB macrocycle.m）
    # ==================================================================

    def _compute_ma_signal(self, series, is_bond=False):
        """生成均线信号 + 5日平滑（严格复现 MATLAB 第18-54行）

        返回: (avg_signal_series, detail_df)
          detail_df 包含: Value, MA60, MA20, RawSignal, AvgSignal
        """
        n = len(series)
        values = series.values.copy()

        ma_long = np.full(n, np.nan)
        ma_short = np.full(n, np.nan)
        raw_signal = np.full(n, np.nan)
        avg_signal = np.full(n, np.nan)

        for nj in range(self.MA_LONG - 1, n):
            ma_long[nj] = np.nanmean(values[nj - self.MA_LONG + 1: nj + 1])
            ma_short[nj] = np.nanmean(values[nj - self.MA_SHORT + 1: nj + 1])

        for nj in range(self.MA_LONG - 1, n):
            if not is_bond:
                if values[nj] > ma_long[nj] and ma_short[nj] > ma_short[nj - 1]:
                    raw_signal[nj] = 1
                else:
                    raw_signal[nj] = -1
            else:
                if values[nj] < ma_long[nj] and ma_short[nj] < ma_short[nj - 1]:
                    raw_signal[nj] = 1
                else:
                    raw_signal[nj] = -1

            avg_signal[nj] = np.nanmean(raw_signal[nj - self.SMOOTH_WINDOW + 1: nj + 1])

        detail = pd.DataFrame({
            "Value": values,
            "MA60": ma_long,
            "MA20": ma_short,
            "RawSignal": raw_signal,
            "AvgSignal": avg_signal,
        }, index=series.index)

        return pd.Series(avg_signal, index=series.index, name=series.name + "_avg"), detail

    def _apply_threshold_conversion(self, data_subset):
        """三阈值平滑信号转换（严格复现 MATLAB 第77-93行）

        输入: 包含 Signal1/Signal2/Signal3 列的 DataFrame
        输出: 添加了 RealSignal1/2/3 的 DataFrame
        """
        for ns in [1, 2, 3]:
            col = "Signal%d" % ns
            temp_avg = data_subset[col].values.copy()
            temp_real = np.full(len(data_subset), np.nan)

            for nj in range(1, len(data_subset)):
                if temp_avg[nj] > 0.5:
                    temp_real[nj] = 1
                elif temp_avg[nj] > 0.2 and temp_avg[nj] < temp_avg[nj - 1] - 0.39:
                    temp_real[nj] = -1
                elif temp_avg[nj] <= 0.2 and temp_avg[nj] > temp_avg[nj - 1] + 0.49:
                    temp_real[nj] = 1
                else:
                    temp_real[nj] = np.sign(temp_avg[nj])

            data_subset["RealSignal%d" % ns] = temp_real

        for col in ["RealSignal1", "RealSignal2", "RealSignal3"]:
            data_subset[col] = data_subset[col].fillna(0).astype(int)

        return data_subset

    def generate_signals(self):
        """生成普林格信号（严格复现 MATLAB macrocycle.m）

        生成两组信号：
        - 月末信号（self.signals）: 每月最后一个交易日（排除未结束的当月）
        - 月中信号（self.signals_mid）: 每月14日（含）最后一个交易日（排除未到14日的当月）
        """
        print("\n" + "=" * 60)
        print("生成普林格信号（严格复现 MATLAB）")
        print("=" * 60)

        # Step 1: 计算三序列的均线信号
        sig1, det1 = self._compute_ma_signal(self.data1["SZ50"], is_bond=False)
        sig2, det2 = self._compute_ma_signal(self.data2["Commodity"], is_bond=False)
        sig3, det3 = self._compute_ma_signal(self.data3["BondRate"], is_bond=True)

        # Step 2: outer join on Date
        data1_df = pd.DataFrame({"SZ50": self.data1["SZ50"], "Signal1": sig1})
        data2_df = pd.DataFrame({"Commodity": self.data2["Commodity"], "Signal2": sig2})
        data3_df = pd.DataFrame({"BondRate": self.data3["BondRate"], "Signal3": sig3})
        data = data1_df.join(data2_df, how="outer").join(data3_df, how="outer")
        data = data.sort_index()

        # Step 3: NaN 向前填充
        for col in ["Signal1", "Signal2", "Signal3"]:
            data[col] = data[col].ffill()

        # Step 4: 按 SZ50 非空筛选
        data = data.dropna(subset=["SZ50"])
        print("合并后数据: %s ~ %s, %d 条" % (
            data.index[0].date(), data.index[-1].date(), len(data)))

        # 保存日度数据（用于计算今日信号 + 展示计算过程）
        det1.index = data1_df.index
        det2.index = data2_df.index
        det3.index = data3_df.index
        self._daily_data = data.copy()  # 完整的日度数据（含 Signal1/2/3 AvgSignal）
        self._detail1 = det1  # SZ50 计算细节
        self._detail2 = det2  # Commodity 计算细节
        self._detail3 = det3  # BondRate 计算细节

        # === 月末/月中日期选择 ===
        data["_ym"] = data.index.to_period("M")
        today_ym = pd.Timestamp.now().to_period("M")

        # Step 5a: 取月末数据（排除未结束的当月）
        data["_day"] = data.index.day
        month_end_mask = data["_ym"] != data["_ym"].shift(-1)
        # 排除当月（月份未结束）
        data_me = data[month_end_mask & (data["_ym"] != today_ym)].copy()
        data_me = data_me.drop(columns=["_ym", "_day"])
        print("月末数据(排除当月): %s ~ %s, %d 条" % (
            data_me.index[0].date(), data_me.index[-1].date(), len(data_me)))

        # Step 5b: 取月中数据（每月14日含之前的最后一个交易日，排除当月如果今天≤14）
        mid_mask = data["_day"] <= 14
        mid_data_full = data[mid_mask].copy()
        mid_dates = mid_data_full.groupby("_ym").apply(lambda g: g.index[-1])
        # 构建排除条件：如果当月今天≤14，说明当月月中数据还要等15日后才有
        today = pd.Timestamp.now()
        if today.day <= 14:
            mid_dates = [d for d in mid_dates.values if pd.Timestamp(d).to_period("M") != today_ym]
        data_mid = data.loc[mid_dates].copy()
        data_mid = data_mid.drop(columns=["_ym", "_day"])
        print("月中数据(<=14日, 排除未到的): %s ~ %s, %d 条" % (
            data_mid.index[0].date(), data_mid.index[-1].date(), len(data_mid)))

        # Step 6a: 月末三阈值转换
        data_me = self._apply_threshold_conversion(data_me)
        self.signals = pd.DataFrame({
            "SZ50": data_me["RealSignal1"],
            "Commodity": data_me["RealSignal2"],
            "BondRate": data_me["RealSignal3"],
        }, index=data_me.index)

        # Step 6b: 月中三阈值转换
        data_mid = self._apply_threshold_conversion(data_mid)
        self.signals_mid = pd.DataFrame({
            "SZ50": data_mid["RealSignal1"],
            "Commodity": data_mid["RealSignal2"],
            "BondRate": data_mid["RealSignal3"],
        }, index=data_mid.index)

        print("\n月末信号分布:")
        for col in ["BondRate", "SZ50", "Commodity"]:
            vc = self.signals[col].value_counts()
            print("  %s: %s" % (col, {k: v for k, v in vc.items()}))

        print("\n月中信号分布:")
        for col in ["BondRate", "SZ50", "Commodity"]:
            vc = self.signals_mid[col].value_counts()
            print("  %s: %s" % (col, {k: v for k, v in vc.items()}))

    # ==================================================================
    # 周期识别（严格复现 MATLAB 第100-127行）
    # ==================================================================

    def _identify_stage(self, bond, sz50, comm):
        """根据三信号识别周期阶段（严格复现 MATLAB 第100-127行）"""
        if bond == 1 and sz50 == -1 and comm == -1:
            return 1
        elif bond == 1 and sz50 == 1 and comm == -1:
            return 2
        elif bond == 1 and sz50 == 1 and comm == 1:
            return 3
        elif bond == -1 and sz50 == 1 and comm == 1:
            return 4
        elif bond == -1 and sz50 == -1 and comm == 1:
            return 5
        elif bond == -1 and sz50 == -1 and comm == -1:
            return 6
        elif bond == -1 and sz50 == 1 and comm == -1:
            return 1.5
        elif bond == 1 and sz50 == -1 and comm == 1:
            return 5.5
        else:
            return 6

    def _stage_to_allocation(self, stage):
        """根据阶段返回持仓风格字符串（50%+50%）"""
        weights = self.CYCLE_WEIGHTS[stage]
        active = [self.STYLE_NAMES[i] for i, w in enumerate(weights) if w > 0]
        return " + ".join(active)

    def _stage_to_allocation_detail(self, stage):
        """根据阶段返回四个风格的具体权重"""
        weights = self.CYCLE_WEIGHTS[stage]
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        return {name: w[i] for i, name in enumerate(self.STYLE_NAMES)}

    def identify_cycles(self):
        """识别每个月的普林格周期状态（月末 + 月中）"""
        print("\n" + "=" * 60)
        print("识别普林格周期")
        print("=" * 60)

        # 月末周期识别
        stages = []
        allocations = []
        for nj in range(len(self.signals)):
            row = self.signals.iloc[nj]
            stage = self._identify_stage(row["BondRate"], row["SZ50"], row["Commodity"])
            stages.append(stage)
            allocations.append(self._stage_to_allocation(stage))

        self.signals["Stage"] = stages
        self.signals["StageName"] = self.signals["Stage"].map(self.CYCLE_NAMES)
        self.signals["Allocation"] = allocations

        cycle_counts = self.signals["StageName"].value_counts()
        print("月末周期分布:")
        for cycle, count in cycle_counts.items():
            pct = count / len(self.signals) * 100
            print("  %s: %d 个月 (%.1f%%)" % (cycle, count, pct))

        signal_file = os.path.join(self.data_path, "macro_signals.csv")
        try:
            self.signals.to_csv(signal_file, encoding="gbk")
            print("月末信号已保存: %s" % signal_file)
        except PermissionError:
            print("[警告] 文件被占用, 跳过保存: %s" % signal_file)

        # 月中周期识别
        stages_mid = []
        allocations_mid = []
        for nj in range(len(self.signals_mid)):
            row = self.signals_mid.iloc[nj]
            stage = self._identify_stage(row["BondRate"], row["SZ50"], row["Commodity"])
            stages_mid.append(stage)
            allocations_mid.append(self._stage_to_allocation(stage))

        self.signals_mid["Stage"] = stages_mid
        self.signals_mid["StageName"] = self.signals_mid["Stage"].map(self.CYCLE_NAMES)
        self.signals_mid["Allocation"] = allocations_mid

        cycle_counts_mid = self.signals_mid["StageName"].value_counts()
        print("\n月中周期分布:")
        for cycle, count in cycle_counts_mid.items():
            pct = count / len(self.signals_mid) * 100
            print("  %s: %d 个月 (%.1f%%)" % (cycle, count, pct))

        signal_mid_file = os.path.join(self.data_path, "macro_signals_mid.csv")
        try:
            self.signals_mid.to_csv(signal_mid_file, encoding="gbk")
            print("月中信号已保存: %s" % signal_mid_file)
        except PermissionError:
            print("[警告] 文件被占用, 跳过保存: %s" % signal_mid_file)

    # ==================================================================
    # 回测
    # ==================================================================

    def backtest(self):
        """回测：月度调仓，根据普林格周期阶段轮动风格指数"""
        print("\n" + "=" * 60)
        print("回测")
        print("=" * 60)

        all_prices = pd.DataFrame(self.style_prices).dropna(how="all")

        # 月末日期
        all_prices["_ym"] = all_prices.index.to_period("M")
        month_end_mask = all_prices["_ym"] != all_prices["_ym"].shift(-1)
        month_end_dates = all_prices.index[month_end_mask]
        all_prices = all_prices.drop(columns=["_ym"])

        # 月度收益率
        monthly_prices = all_prices.reindex(month_end_dates)
        monthly_ret = monthly_prices.pct_change()

        # 对齐信号日期和月度收益率日期
        sig_dates = self.signals.index
        ret_dates = monthly_ret.index
        common_dates = sig_dates.intersection(ret_dates)
        common_dates = common_dates[common_dates >= pd.Timestamp(self.BACKTEST_START)]

        print("回测期间: %s ~ %s, %d 个月" % (
            common_dates[0].date(), common_dates[-1].date(), len(common_dates)))

        ret_records = []
        prev_w = None  # 2026-10-10 全族统一费后口径（单边千三）：跟踪换手扣费
        for i in range(len(common_dates) - 1):
            sig_date = common_dates[i]
            next_date = common_dates[i + 1]

            stage = self.signals.loc[sig_date, "Stage"]
            weights = np.array(self.CYCLE_WEIGHTS[stage], dtype=float)
            weights = weights / weights.sum()

            turn_m = float(np.abs(weights).sum()) if prev_w is None \
                else float(0.5 * np.abs(weights - prev_w).sum())
            prev_w = weights

            next_ret = monthly_ret.loc[next_date, self.STYLE_NAMES].values
            port_ret = np.nansum(weights * next_ret) - turn_m * 0.003  # 费后（全族统一单边千三）

            bench_weights = np.array([0.25, 0.25, 0.25, 0.25])
            bench_ret = np.nansum(bench_weights * next_ret)

            ret_records.append({
                "日期": next_date,
                "周期": self.CYCLE_NAMES[stage],
                "持仓风格": self._stage_to_allocation(stage),
                "权重红利": weights[0],
                "权重成长": weights[1],
                "权重周期": weights[2],
                "权重质量": weights[3],
                "组合月收益": port_ret,
                "基准月收益": bench_ret,
                "超额收益": port_ret - bench_ret,
                "换手": turn_m,
            })

        self.result = pd.DataFrame(ret_records).set_index("日期")

        self.result["组合净值"] = (self.result["组合月收益"] + 1).cumprod()
        self.result["基准净值"] = (self.result["基准月收益"] + 1).cumprod()
        self.result["组合累计收益"] = self.result["组合净值"] - 1
        self.result["基准累计收益"] = self.result["基准净值"] - 1
        self.result["组合回撤"] = (
            self.result["组合净值"] / self.result["组合净值"].expanding().max() - 1)
        self.result["基准回撤"] = (
            self.result["基准净值"] / self.result["基准净值"].expanding().max() - 1)

        self._print_summary()

    # ==================================================================
    # 绩效统计
    # ==================================================================

    def _print_summary(self):
        """打印回测绩效统计"""
        print("\n" + "-" * 60)
        print("回测绩效统计")
        print("-" * 60)

        port_ret = self.result["组合月收益"]
        bench_ret = self.result["基准月收益"]
        excess_ret = port_ret - bench_ret

        port_cum = (port_ret + 1).prod() - 1
        n_years = len(port_ret) / 12
        port_annual = (1 + port_cum) ** (1 / n_years) - 1
        port_vol = port_ret.std() * np.sqrt(12)
        port_mdd = self.result["组合回撤"].min()
        port_sharpe = port_annual / port_vol if port_vol > 0 else 0

        bench_cum = (bench_ret + 1).prod() - 1
        bench_annual = (1 + bench_cum) ** (1 / n_years) - 1
        bench_vol = bench_ret.std() * np.sqrt(12)
        bench_mdd = self.result["基准回撤"].min()
        bench_sharpe = bench_annual / bench_vol if bench_vol > 0 else 0

        tracking_error = excess_ret.std() * np.sqrt(12)
        info_ratio = (port_annual - bench_annual) / tracking_error if tracking_error > 0 else 0

        print("                    轮动组合        等权基准        差异")
        print("  累计收益率:    %8.2f%%      %8.2f%%      %+.2f%%" % (
            port_cum * 100, bench_cum * 100, (port_cum - bench_cum) * 100))
        print("  年化收益率:    %8.2f%%      %8.2f%%      %+.2f%%" % (
            port_annual * 100, bench_annual * 100, (port_annual - bench_annual) * 100))
        print("  年化波动率:    %8.2f%%      %8.2f%%      %+.2f%%" % (
            port_vol * 100, bench_vol * 100, (port_vol - bench_vol) * 100))
        print("  最大回撤:      %8.2f%%      %8.2f%%      %+.2f%%" % (
            port_mdd * 100, bench_mdd * 100, (port_mdd - bench_mdd) * 100))
        print("  夏普比率:      %8.2f        %8.2f        %+.2f" % (
            port_sharpe, bench_sharpe, port_sharpe - bench_sharpe))
        print("  跟踪误差:                   %8.2f%%" % (tracking_error * 100))
        print("  信息比率:                  %8.2f" % info_ratio)

        # 分年度统计
        print("\n" + "-" * 60)
        print("分年度统计")
        print("-" * 60)
        self.result["年份"] = self.result.index.year.astype(str)

        yearly_stats = []
        for year, group in self.result.groupby("年份"):
            port_yr_ret = (group["组合月收益"] + 1).prod() - 1
            bench_yr_ret = (group["基准月收益"] + 1).prod() - 1
            port_yr_vol = group["组合月收益"].std() * np.sqrt(12)
            bench_yr_vol = group["基准月收益"].std() * np.sqrt(12)

            port_nav = (group["组合月收益"] + 1).cumprod()
            bench_nav = (group["基准月收益"] + 1).cumprod()
            port_yr_mdd = (port_nav / port_nav.expanding().max() - 1).min()
            bench_yr_mdd = (bench_nav / bench_nav.expanding().max() - 1).min()

            main_cycle = group["周期"].value_counts().index[0]

            yearly_stats.append({
                "年份": year,
                "组合收益": port_yr_ret,
                "基准收益": bench_yr_ret,
                "超额收益": port_yr_ret - bench_yr_ret,
                "组合波动": port_yr_vol,
                "基准波动": bench_yr_vol,
                "组合回撤": port_yr_mdd,
                "基准回撤": bench_yr_mdd,
                "主要周期": main_cycle,
            })

        self.yearly_stats = pd.DataFrame(yearly_stats).set_index("年份")
        print(self.yearly_stats.to_string(
            float_format=lambda x: "%.2f%%" % (x * 100) if abs(x) < 10 else "%.2f" % x))

    # ==================================================================
    # 输出
    # ==================================================================

    def save_results(self):
        """保存回测结果"""
        # 1. 完整回测结果
        path = os.path.join(self.data_path, "backtest_result.csv")
        try:
            self.result.to_csv(path, encoding="gbk")
            print("回测结果已保存: %s" % path)
        except PermissionError:
            print("[警告] 文件被占用, 跳过保存: %s" % path)

        # 2. 分年度统计
        path = os.path.join(self.data_path, "yearly_stats.csv")
        try:
            self.yearly_stats.to_csv(path, encoding="gbk")
            print("分年度统计已保存: %s" % path)
        except PermissionError:
            print("[警告] 文件被占用, 跳过保存: %s" % path)

        # 3. 调仓记录（cycle_positions.csv）：持仓风格 + 各风格权重
        cols = ["周期", "持仓风格", "权重红利", "权重成长", "权重周期", "权重质量",
                "组合月收益", "基准月收益", "超额收益"]
        cycle_df = self.result[cols].rename(columns={
            "权重红利": "红利权重", "权重成长": "成长权重",
            "权重周期": "周期权重", "权重质量": "质量权重"})
        path = os.path.join(self.data_path, "cycle_positions.csv")
        try:
            cycle_df.to_csv(path, encoding="gbk")
            print("调仓记录已保存: %s" % path)
        except PermissionError:
            print("[警告] 文件被占用, 跳过保存: %s" % path)

    def generate_html_report(self):
        """生成 HTML 回测报告（净值曲线+超额收益+年度对比）"""
        import json, math

        result = self.result.reset_index()
        ys = self.yearly_stats.reset_index()

        dates = result["日期"].astype(str).tolist()
        nav_rot = result["组合净值"].round(6).tolist()
        nav_bench = result["基准净值"].round(6).tolist()
        excess = result["超额收益"].round(6).tolist()
        stages = result["周期"].tolist()
        dd_rot = result["组合回撤"].round(6).tolist()
        dd_bench = result["基准回撤"].round(6).tolist()

        cum_excess = []
        running = 0.0
        for e in excess:
            running += e
            cum_excess.append(round(running, 6))

        years_data = []
        for _, yr in ys.iterrows():
            years_data.append({
                "year": str(yr["年份"]),
                "rot": round(float(yr["组合收益"]) * 100, 2),
                "bench": round(float(yr["基准收益"]) * 100, 2),
                "excess": round(float(yr["超额收益"]) * 100, 2),
                "vol_rot": round(float(yr["组合波动"]) * 100, 2),
                "vol_bench": round(float(yr["基准波动"]) * 100, 2),
                "dd_rot": round(float(yr["组合回撤"]) * 100, 2),
                "dd_bench": round(float(yr["基准回撤"]) * 100, 2),
                "cycle": str(yr["主要周期"]),
            })

        total_rot = round((nav_rot[-1] - 1) * 100, 2)
        total_bench = round((nav_bench[-1] - 1) * 100, 2)
        total_excess = round(total_rot - total_bench, 2)
        n_months = len(dates)
        n_years = n_months / 12
        ann_rot = round(((nav_rot[-1]) ** (1 / n_years) - 1) * 100, 2)
        ann_bench = round(((nav_bench[-1]) ** (1 / n_years) - 1) * 100, 2)
        ann_excess = round(ann_rot - ann_bench, 2)

        mean_excess = sum(excess) / len(excess)
        std_excess = math.sqrt(sum((x - mean_excess) ** 2 for x in excess) / (len(excess) - 1))
        ir = round(mean_excess / std_excess * math.sqrt(12), 2) if std_excess > 0 else 0

        win_months = sum(1 for e in excess if e > 0)
        win_rate = round(win_months / len(excess) * 100, 1)

        max_dd_rot = round(min(dd_rot) * 100, 2)
        max_dd_bench = round(min(dd_bench) * 100, 2)

        win_years = sum(1 for y in years_data if y["excess"] > 0)
        year_win_rate = round(win_years / len(years_data) * 100, 1)

        year_labels = [y["year"] for y in years_data]
        year_excess_vals = [y["excess"] for y in years_data]

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>普林格周期轮动模型 — 回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; background: #0f1117; color: #e1e4e8; padding: 24px; }}
.header {{ text-align: center; padding: 32px 0 24px; }}
.header h1 {{ font-size: 26px; font-weight: 700; color: #f0f2f5; margin-bottom: 6px; }}
.header p {{ font-size: 13px; color: #8b949e; }}
.kpi-row {{ display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin-bottom: 24px; }}
.kpi-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px 22px; min-width: 140px; text-align: center; }}
.kpi-card .label {{ font-size: 11px; color: #8b949e; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
.kpi-card .value {{ font-size: 24px; font-weight: 700; }}
.kpi-card .sub {{ font-size: 11px; color: #8b949e; margin-top: 2px; }}
.pos {{ color: #ef4444; }}
.neg {{ color: #22c55e; }}
.neutral {{ color: #f0f2f5; }}
.chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px; margin-bottom: 24px; }}
.chart-container h2 {{ font-size: 16px; font-weight: 600; margin-bottom: 8px; color: #f0f2f5; }}
#nav_chart {{ width: 100%; height: 500px; }}
#year_chart {{ width: 100%; height: 380px; }}
.table-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px; margin-bottom: 24px; overflow-x: auto; }}
.table-container h2 {{ font-size: 16px; font-weight: 600; margin-bottom: 12px; color: #f0f2f5; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th {{ background: #21262d; color: #8b949e; font-weight: 600; padding: 8px 10px; text-align: right; border-bottom: 1px solid #30363d; white-space: nowrap; }}
th:first-child, th:last-child {{ text-align: left; }}
td {{ padding: 7px 10px; text-align: right; border-bottom: 1px solid #21262d; white-space: nowrap; }}
td:first-child, td:last-child {{ text-align: left; }}
tr:hover td {{ background: #1c2129; }}
.footer {{ text-align: center; font-size: 11px; color: #484f58; padding: 16px 0; }}
</style>
</head>
<body>

<div class="header">
  <h1>普林格周期轮动模型 — 普林格经济周期六阶段</h1>
  <p>回测区间：{dates[0]} ~ {dates[-1]} · 共 {n_months} 个月 ({n_years:.1f} 年)</p>
</div>

<div class="kpi-row">
  <div class="kpi-card">
    <div class="label">轮动累计收益</div>
    <div class="value pos">{total_rot:+.2f}%</div>
    <div class="sub">年化 {ann_rot:+.2f}%</div>
  </div>
  <div class="kpi-card">
    <div class="label">基准累计收益</div>
    <div class="value neutral">{total_bench:+.2f}%</div>
    <div class="sub">年化 {ann_bench:+.2f}%</div>
  </div>
  <div class="kpi-card">
    <div class="label">累计超额收益</div>
    <div class="value {'pos' if total_excess >= 0 else 'neg'}">{total_excess:+.2f}%</div>
    <div class="sub">年化 {ann_excess:+.2f}%</div>
  </div>
  <div class="kpi-card">
    <div class="label">信息比率</div>
    <div class="value neutral">{ir:.2f}</div>
    <div class="sub">月胜率 {win_rate:.1f}%</div>
  </div>
  <div class="kpi-card">
    <div class="label">轮动最大回撤</div>
    <div class="value neg">{max_dd_rot:.2f}%</div>
    <div class="sub">基准 {max_dd_bench:.2f}%</div>
  </div>
  <div class="kpi-card">
    <div class="label">年度胜率</div>
    <div class="value {'pos' if year_win_rate >= 50 else 'neg'}">{year_win_rate:.0f}%</div>
    <div class="sub">{win_years}/{len(years_data)} 年</div>
  </div>
</div>

<div class="chart-container">
  <h2>净值曲线 &amp; 超额收益</h2>
  <div id="nav_chart"></div>
</div>

<div class="chart-container">
  <h2>年度收益对比</h2>
  <div id="year_chart"></div>
</div>

<div class="table-container">
  <h2>分年度统计</h2>
  <table>
    <thead>
      <tr>
        <th>年份</th>
        <th>轮动收益</th>
        <th>基准收益</th>
        <th>超额收益</th>
        <th>轮动波动</th>
        <th>基准波动</th>
        <th>轮动回撤</th>
        <th>基准回撤</th>
        <th>主要周期</th>
      </tr>
    </thead>
    <tbody>
"""

        for y in years_data:
            ec = "pos" if y["excess"] >= 0 else "neg"
            html += f"""      <tr>
        <td>{y["year"]}</td>
        <td class="{'pos' if y['rot'] >= 0 else 'neg'}">{y['rot']:+.2f}%</td>
        <td class="{'pos' if y['bench'] >= 0 else 'neg'}">{y['bench']:+.2f}%</td>
        <td class="{ec}">{y['excess']:+.2f}%</td>
        <td>{y['vol_rot']:.1f}%</td>
        <td>{y['vol_bench']:.1f}%</td>
        <td class="neg">{y['dd_rot']:.2f}%</td>
        <td class="neg">{y['dd_bench']:.2f}%</td>
        <td>{y['cycle']}</td>
      </tr>
"""

        from datetime import datetime
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        html += f"""    </tbody>
  </table>
</div>

<div class="footer">
  数据来源：Wind 终端 · 模型：普林格六阶段周期轮动 · 生成时间：{gen_time}
</div>

<script>
var dates = {json.dumps(dates)};
var navRot = {json.dumps(nav_rot)};
var navBench = {json.dumps(nav_bench)};
var cumExcess = {json.dumps(cum_excess)};
var excessMonthly = {json.dumps(excess)};

var traceNavRot = {{
  x: dates, y: navRot, type: 'scatter', mode: 'lines',
  name: '轮动组合', line: {{ color: '#ef4444', width: 2 }}, yaxis: 'y',
  hovertemplate: '%{{x}}<br>轮动净值: %{{y:.4f}}<extra></extra>',
}};
var traceNavBench = {{
  x: dates, y: navBench, type: 'scatter', mode: 'lines',
  name: '等权基准', line: {{ color: '#8b949e', width: 2, dash: 'dash' }}, yaxis: 'y',
  hovertemplate: '%{{x}}<br>基准净值: %{{y:.4f}}<extra></extra>',
}};
var traceCumExcess = {{
  x: dates, y: cumExcess, type: 'scatter', mode: 'lines',
  name: '累计超额(右轴)', line: {{ color: '#d2991d', width: 1.5 }}, yaxis: 'y2',
  hovertemplate: '%{{x}}<br>累计超额: %{{y:.4f}}<extra></extra>',
}};

var navLayout = {{
  template: 'plotly_dark',
  paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
  font: {{ color: '#8b949e', size: 11 }},
  margin: {{ l: 60, r: 60, t: 20, b: 50 }},
  legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(22,27,34,0.9)', bordercolor: '#30363d' }},
  xaxis: {{ title: '', gridcolor: '#21262d', zerolinecolor: '#30363d', rangeslider: {{ visible: false }}, type: 'date' }},
  yaxis: {{ title: '净值', gridcolor: '#21262d', zerolinecolor: '#30363d', side: 'left' }},
  yaxis2: {{ title: '累计超额', gridcolor: 'rgba(0,0,0,0)', zerolinecolor: '#d2991d', side: 'right', overlaying: 'y' }},
  hovermode: 'x unified',
}};

Plotly.newPlot('nav_chart', [traceNavRot, traceNavBench, traceCumExcess], navLayout, {{ responsive: true }});

// ── 年度收益对比 ──
var yearLabels = {json.dumps(year_labels)};
var yearExcess = {json.dumps(year_excess_vals)};
var yearRotVals = {json.dumps([y["rot"] for y in years_data])};
var yearBenchVals = {json.dumps([y["bench"] for y in years_data])};

var barColorsRot = yearRotVals.map(function(v) {{ return v >= 0 ? '#ef4444' : '#22c55e'; }});
var barColorsBench = yearBenchVals.map(function(v) {{ return v >= 0 ? 'rgba(239,68,68,0.4)' : 'rgba(34,197,94,0.4)'; }});

var traceYearRot = {{
  x: yearLabels, y: yearRotVals, type: 'bar', name: '轮动组合',
  marker: {{ color: barColorsRot }},
  hovertemplate: '%{{x}}<br>轮动: %{{y:+.2f}}%<extra></extra>',
}};
var traceYearBench = {{
  x: yearLabels, y: yearBenchVals, type: 'bar', name: '等权基准',
  marker: {{ color: barColorsBench }},
  hovertemplate: '%{{x}}<br>基准: %{{y:+.2f}}%<extra></extra>',
}};
var traceYearExcess = {{
  x: yearLabels, y: yearExcess, type: 'scatter', mode: 'lines+markers',
  name: '超额收益', line: {{ color: '#d2991d', width: 2 }},
  marker: {{ size: 8, color: '#d2991d' }},
  hovertemplate: '%{{x}}<br>超额: %{{y:+.2f}}%<extra></extra>',
}};

var yearLayout = {{
  template: 'plotly_dark',
  paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
  font: {{ color: '#8b949e', size: 11 }},
  margin: {{ l: 60, r: 40, t: 20, b: 50 }},
  legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(22,27,34,0.9)', bordercolor: '#30363d' }},
  xaxis: {{ gridcolor: '#21262d', zerolinecolor: '#30363d' }},
  yaxis: {{ title: '收益率 (%)', gridcolor: '#21262d', zerolinecolor: '#30363d', ticksuffix: '%' }},
  hovermode: 'x unified',
  barmode: 'group',
}};

Plotly.newPlot('year_chart', [traceYearRot, traceYearBench, traceYearExcess], yearLayout, {{ responsive: true }});
</script>

</body>
</html>"""

        out = os.path.join(self.data_path, "回测报告.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        print("HTML 回测报告已生成: %s" % out)

    # ==================================================================
    # 最新信号
    # ==================================================================

    def _compute_today_signal(self):
        """计算今日日度信号：取最新数据点的 avg_signal，用上月末值做三阈值转换"""
        from datetime import datetime

        daily = self._daily_data
        today = pd.Timestamp.now().normalize()

        # 找到 ≤ 今天的最新日度数据
        mask = daily.index <= today
        if mask.sum() == 0:
            return None, None

        today_data = daily[mask].iloc[-1]
        today_date = daily.index[mask][-1]

        # 上月末数据的 avg_signal 作为三阈值转换的"前一值"
        prev_me = self.signals.iloc[-1] if len(self.signals) > 0 else None

        # 对上月末，需要找到它对应的原始 avg_signal（在 _daily_data 中）
        if prev_me is not None:
            prev_me_date = self.signals.index[-1]
            prev_mask = daily.index <= prev_me_date
            prev_daily = daily[prev_mask].iloc[-1] if prev_mask.sum() > 0 else None
        else:
            prev_daily = None

        # 三阈值转换
        results = {}
        for ns, col in [(1, "Signal1"), (2, "Signal2"), (3, "Signal3")]:
            cur_avg = today_data[col]
            prev_avg = prev_daily[col] if prev_daily is not None else cur_avg

            if cur_avg > 0.5:
                real = 1
            elif cur_avg > 0.2 and cur_avg < prev_avg - 0.39:
                real = -1
            elif cur_avg <= 0.2 and cur_avg > prev_avg + 0.49:
                real = 1
            else:
                real = np.sign(cur_avg)
                real = 0 if real == 0 else int(real)

            results[ns] = real

        # 映射到信号名
        signal_dict = {
            "BondRate": results[3],
            "SZ50": results[1],
            "Commodity": results[2],
        }
        return today_date, signal_dict

    def print_latest_signal(self):
        """打印今日日度信号 + 上月末信号，附详细计算过程"""
        print("\n" + "=" * 60)
        print("最新普林格信号与持仓建议")
        print("=" * 60)

        # 辅助：根据index名称获取detail DataFrame
        def _get_detail_at(name, target_date):
            if name == "SZ50":
                det = self._detail1
            elif name == "Commodity":
                det = self._detail2
            else:
                det = self._detail3
            mask = det.index <= target_date
            if mask.sum() == 0:
                return None
            return det[mask].iloc[-1]

        def _print_signal_detail(label, date, stage, stage_name, signals_dict):
            """打印一个信号的全部细节"""
            weights = self.CYCLE_WEIGHTS[stage]
            active_styles = [self.STYLE_NAMES[i] for i, w in enumerate(weights) if w > 0]
            is_bond_flag = {"SZ50": False, "Commodity": False, "BondRate": True}
            detail_map = {"SZ50": self._detail1, "Commodity": self._detail2, "BondRate": self._detail3}

            print("\n%s %s | Stage %.1f %s | 持仓: %s" % (
                label, date.date(), stage, stage_name, " + ".join(active_styles)))
            print("-" * 50)

            for idx_name, cn_name in [("BondRate", "利率 (10Y国债)"),
                                      ("SZ50", "经济 (上证50)"),
                                      ("Commodity", "通胀 (CRB.RB)")]:
                det_df = detail_map[idx_name]
                det = _get_detail_at(idx_name, date)
                if det is None:
                    print("  %s: 无数据" % cn_name)
                    continue

                final_signal = signals_dict[idx_name]
                is_bond = is_bond_flag[idx_name]

                print("  [%s]" % cn_name)
                print("    当日值:  %.4f" % det["Value"])
                print("    MA60:    %.4f" % det["MA60"])
                print("    MA20:    %.4f" % det["MA20"])
                if is_bond:
                    ma20_prev_val = self._prev_ma20(det_df, date)
                    print("    → 债券方向相反: 收益率下行=利好(+1)")
                    print("    判断: 值<MA60(%s) 且 MA20↓(%s) → RawSignal=%+.0f" % (
                        "是" if det["Value"] < det["MA60"] else "否",
                        "是" if not np.isnan(ma20_prev_val) and det["MA20"] < ma20_prev_val else "否",
                        det["RawSignal"]))
                else:
                    ma20_prev_val = self._prev_ma20(det_df, date)
                    print("    判断: 值>MA60(%s) 且 MA20↑(%s) → RawSignal=%+.0f" % (
                        "是" if det["Value"] > det["MA60"] else "否",
                        "是" if not np.isnan(ma20_prev_val) and det["MA20"] > ma20_prev_val else "否",
                        det["RawSignal"]))
                print("    5日平滑信号:  %+.3f" % det["AvgSignal"])
                print("    三阈值转换后:  %+d" % final_signal)

        # ================================================================
        # 今日信号
        # ================================================================
        print("\n" + "─" * 50)
        print("  【今日信号】")
        print("─" * 50)

        today_date, today_signals = self._compute_today_signal()
        if today_date is not None and today_signals is not None:
            today_stage = self._identify_stage(
                today_signals["BondRate"], today_signals["SZ50"], today_signals["Commodity"])
            today_stage_name = self.CYCLE_NAMES[today_stage]
            _print_signal_detail("[今日信号]", today_date, today_stage, today_stage_name, today_signals)
        else:
            print("  (今日无数据)")

        # ================================================================
        # 上月末信号
        # ================================================================
        print("\n" + "─" * 50)
        print("  【上月末信号】")
        print("─" * 50)

        if len(self.signals) >= 1:
            prev_date = self.signals.index[-1]
            prev = self.signals.loc[prev_date]
            prev_signals = {
                "BondRate": int(prev["BondRate"]),
                "SZ50": int(prev["SZ50"]),
                "Commodity": int(prev["Commodity"]),
            }
            prev_stage = prev["Stage"]
            prev_stage_name = self.CYCLE_NAMES[prev_stage]
            _print_signal_detail("[上月末信号]", prev_date, prev_stage, prev_stage_name, prev_signals)
        else:
            print("  (无上月末数据)")

    def _prev_ma20(self, det_df, date):
        """获取某日前一交易日的MA20值"""
        mask = det_df.index < date
        if mask.sum() < 1:
            return np.nan
        return det_df[mask].iloc[-1]["MA20"]

    # ==================================================================
    # 调度器入口：返回结构化当日信号（供 cal_data_daily 调用）
    # ==================================================================

    def run_daily(self, run_backtest=True, verbose=False):
        """每日完整流程：load_data → generate_signals → identify_cycles
        → (可选) backtest/save_results/generate_html_report，
        返回当日 + 上月末信号的结构化 dict，供 CalDataDaily 汇总成 Word 日报。

        返回字段：
            date            今日信号对应的数据日期 (pd.Timestamp)
            today_signals   dict(BondRate/SZ50/Commodity = ±1/0)
            today_stage     float 阶段编号（1/1.5/.../6）
            today_stage_name str 阶段中文名
            today_alloc     dict(红利/成长/周期/质量 = 权重，和=1)
            today_detail    dict(三大件各自 Value/MA60/MA20/RawSignal/AvgSignal/RealSignal)
            trend           近 N 日三大件平滑信号演变（DataFrame）
            prev_date       上月末日期
            prev_signals    dict 上月末三信号
            prev_stage      上月末阶段
            prev_stage_name 上月末阶段名
            prev_alloc      上月末配置权重 dict
            prev_detail     dict 上月末三大件明细（同 today_detail 结构）
            mid_date        最新月中信号日期
            mid_signals     dict 最新月中三信号
            mid_stage       月中阶段
            mid_stage_name  月中阶段名
            mid_alloc       月中配置权重 dict
            mid_detail      dict 月中三大件明细
            recent_monthly  近 n 期月末信号一览（DataFrame：三信号/阶段/阶段名/配置）
            recent_mid      近 n 期月中信号一览（DataFrame 同上）
            html_report     回测报告.html 路径（run_backtest=True 时生成）
        """
        import io
        import contextlib

        # 完整流程；print 走 stdout，用 verbose 控制是否屏蔽
        buf = io.StringIO()
        ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(buf)
        with ctx:
            self.load_data()
            self.generate_signals()
            self.identify_cycles()
            html_path = None
            if run_backtest:
                self.backtest()
                self.save_results()
                self.generate_html_report()
                html_path = os.path.join(self.data_path, "回测报告.html")

        # ===== 今日信号 =====
        today_date, today_signals = self._compute_today_signal()
        if today_date is None:
            raise RuntimeError("普林格轮动：今日信号计算失败（无日度数据）")
        today_stage = self._identify_stage(
            today_signals["BondRate"], today_signals["SZ50"], today_signals["Commodity"])
        today_stage_name = self.CYCLE_NAMES[today_stage]
        today_alloc = self._stage_to_allocation_detail(today_stage)

        # 今日信号对应的三大件计算细节
        def _det(name, target_date):
            det_df = {"SZ50": self._detail1,
                      "Commodity": self._detail2,
                      "BondRate": self._detail3}[name]
            mask = det_df.index <= target_date
            if mask.sum() == 0:
                return None
            row = det_df[mask].iloc[-1]
            return {
                "Value": float(row["Value"]),
                "MA60": float(row["MA60"]),
                "MA20": float(row["MA20"]),
                "RawSignal": float(row["RawSignal"]),
                "AvgSignal": float(row["AvgSignal"]),
            }
        today_detail = {"SZ50": _det("SZ50", today_date),
                        "Commodity": _det("Commodity", today_date),
                        "BondRate": _det("BondRate", today_date)}
        for k in today_detail:
            if today_detail[k] is not None:
                today_detail[k]["RealSignal"] = int(today_signals[k])

        # ===== 近 N 日平滑信号演变（观察信号是否在临界区间漂移）=====
        n_trend = 5
        det_map = {"SZ50": self._detail1, "Commodity": self._detail2,
                   "BondRate": self._detail3}
        trend = pd.DataFrame(index=["经济SZ50", "通胀CRB", "利率Bond"])
        for name, col_name in [("SZ50", "经济SZ50"), ("Commodity", "通胀CRB"),
                               ("BondRate", "利率Bond")]:
            d = det_map[name]
            tail = d[d.index <= today_date].tail(n_trend)
            for dt, row in tail.iterrows():
                trend.loc[col_name, dt.strftime("%m-%d")] = float(row["AvgSignal"])
        # 三大件交易日历不完全一致：按日期排序 + 去掉全 NaN 列，
        # 保证列数 = 各大件最近 n_trend 日的并集（可能 > n_trend）
        trend = trend[sorted(trend.columns)]
        trend = trend.dropna(axis=1, how="all")

        # ===== 上月末信号 + 明细 =====
        if len(self.signals) < 1:
            raise RuntimeError("普林格轮动：上月末信号为空")
        prev_date = self.signals.index[-1]
        prev_row = self.signals.iloc[-1]
        prev_signals = {"BondRate": int(prev_row["BondRate"]),
                        "SZ50": int(prev_row["SZ50"]),
                        "Commodity": int(prev_row["Commodity"])}
        prev_stage = float(prev_row["Stage"])
        prev_stage_name = str(prev_row["StageName"])
        prev_alloc = self._stage_to_allocation_detail(prev_stage)
        prev_detail = {"SZ50": _det("SZ50", prev_date),
                       "Commodity": _det("Commodity", prev_date),
                       "BondRate": _det("BondRate", prev_date)}
        for k in prev_detail:
            if prev_detail[k] is not None:
                prev_detail[k]["RealSignal"] = int(prev_signals[k])

        # ===== 最新月中信号 + 明细 =====
        mid_date = mid_signals = mid_stage = mid_stage_name = mid_alloc = mid_detail = None
        if len(self.signals_mid) >= 1:
            mid_date = self.signals_mid.index[-1]
            mid_row = self.signals_mid.iloc[-1]
            mid_signals = {"BondRate": int(mid_row["BondRate"]),
                           "SZ50": int(mid_row["SZ50"]),
                           "Commodity": int(mid_row["Commodity"])}
            mid_stage = float(mid_row["Stage"])
            mid_stage_name = str(mid_row["StageName"])
            mid_alloc = self._stage_to_allocation_detail(mid_stage)
            mid_detail = {"SZ50": _det("SZ50", mid_date),
                          "Commodity": _det("Commodity", mid_date),
                          "BondRate": _det("BondRate", mid_date)}
            for k in mid_detail:
                if mid_detail[k] is not None:
                    mid_detail[k]["RealSignal"] = int(mid_signals[k])

        # ===== 近 n 期月末/月中信号一览 =====
        def _recent_df(sig_df, n):
            cols = ["SZ50", "Commodity", "BondRate", "Stage", "StageName", "Allocation"]
            take = sig_df[cols].tail(n).iloc[::-1].copy()
            take.index = [d.strftime("%Y-%m-%d") for d in take.index]
            return take

        recent_monthly = _recent_df(self.signals, 6)
        recent_mid = _recent_df(self.signals_mid, 6)

        return {
            "date": today_date,
            "today_signals": today_signals,
            "today_stage": today_stage,
            "today_stage_name": today_stage_name,
            "today_alloc": today_alloc,
            "today_detail": today_detail,
            "trend": trend,
            "prev_date": prev_date,
            "prev_signals": prev_signals,
            "prev_stage": prev_stage,
            "prev_stage_name": prev_stage_name,
            "prev_alloc": prev_alloc,
            "prev_detail": prev_detail,
            "mid_date": mid_date,
            "mid_signals": mid_signals,
            "mid_stage": mid_stage,
            "mid_stage_name": mid_stage_name,
            "mid_alloc": mid_alloc,
            "mid_detail": mid_detail,
            "recent_monthly": recent_monthly,
            "recent_mid": recent_mid,
            "html_report": html_path,
        }


# ======================================================================
# 主函数
# ======================================================================

if __name__ == "__main__":

    model = RotationPring()

    # 1. 加载数据（本地底表）
    model.load_data()

    # 2. 生成信号（严格复现 MATLAB）
    model.generate_signals()

    # 3. 识别周期
    model.identify_cycles()

    # 4. 回测
    model.backtest()

    # 5. 保存结果
    model.save_results()

    # 6. 生成 HTML 回测报告
    model.generate_html_report()

    # 7. 最新信号
    model.print_latest_signal()
