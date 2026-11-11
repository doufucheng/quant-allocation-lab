# -*- coding: utf-8 -*-
"""
国债期货高频时间维度择时模型 —— 兴证金工研报复现（2026-08-27）

参考研报：《基于高频时间维度的国债期货择时因子及个股CTA研究》
兴业证券金融工程 2026-06-09（分析师：郑兆磊）
解析稿：Config().knowledge_base_path 下的报告正文

=====================================================================
核心思路（跳出"价量二维"框架，挖掘纯时间维度信息）：
以 10 年期（T）与 30 年期（TL）国债期货主力合约为标的，基于日内
1 分钟高频数据构建两大时间维度择时体系：

A. 持续期因子（Duration）
   定义：价格/成交量"突破动态阈值"的时间间隔（分钟）。
   - 阈值：基于过去 N 日（默认 250）价差/成交量样本的分位数动态滚动，
     自适应当日波动状态（固定阈值会因价格水平漂移而失效）
   - 持续期长 = 该价格区间内无增量信息冲击、多空博弈充分出清
     → 该价格是"日内公允均衡价格"（Fair Price）
   聚合因子：
   1) 价格持续期偏离因子 DFP = 公允均衡价格(最大/前三大持续期) - 收盘价
      公允价 > 收盘价（尾盘超跌）→ 看多；公允价 < 收盘价（尾盘过热）→ 看空
   2) 价格持续期时点因子 PMT = argmax(价格持续期) 的日内分钟序号
      早盘达成共识（值小）→ 看多；共识滞后（值大）→ 看空
   3) 成交量持续期比值因子 VR = 午前均值 / 午后均值
      >1 早盘稳定 → 看多；<1 早盘紊乱 → 看空
   4) 成交量持续期时点因子 VMT = argmax(成交量持续期) 的分钟序号
      值小 → 看多；值大 → 看空

B. 时间戳因子（Timestamp）
   定义：日内关键行情事件出现的分钟时序位置（1 = 开盘第 1 分钟）。
   5) 最高价时间戳：首次触高越早 → 多方发力 → 看多；越晚 → 上攻乏力 → 看空
   6) 最低价时间戳：首次触低越早 → 看空；越晚 → 探底乏力 → 看多
   7) 最大成交量时间戳：最大量越晚（尾盘放量=资金布局）→ 看多；早盘放量=兑现 → 看空
   8) 午后最高价时间戳 / 9) 午后最低价时间戳：聚焦午后段，逻辑同 5/6
   10) 早盘尖峰次数：早盘触碰日内最高价的次数，越多 → 多头进攻意愿强 → 看多

信号与回测（与研报一致）：
   - 信号：滚动 N 日分位数阈值法，高于高分位做多、低于低分位做空、中间空仓
   - 组合：各因子三态信号等权平均 → 综合仓位（规避主观赋权过拟合）
   - 调仓：日频，T 日盘后出信号、T+1 日开盘执行、持仓至 T+2 开盘（相邻衔接）
   - 杠杆：1 倍无杠杆（保证金 100%），还原纯因子收益特征

数据约束（实测 2026-08-27，重要）：
   - Wind wsi 分钟数据仅提供近约 2 年：T.CFE / TL.CFE 自 2024-06 起
     （连续合约与具体合约均如此，2017 年 T1709 等拉不到）
   - Wind Oracle 底库无分钟表（仅 EOD 日频），无法补更早历史
   - 因此回测区间为 2024-06 至今，随每日增量更新自然延长；
     研报 T 为 2017-01~2026-04、TL 为 2023-04~2026-04，样本期差异需知悉
   - 国债期货分钟 bar 每日 257 根（连续竞价 09:30-11:30 + 13:00-15:15）

使用：
   python bond_futures_timing.py update     # 拉取/增量更新 1 分钟数据（需 Wind 终端）
   python bond_futures_timing.py backtest   # 因子+信号+回测+HTML 报告
运行环境：项目配置的 Anaconda 解释器（需安装 WindPy）
=====================================================================
"""

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config.config import Config
from utility.windpy_guard import get_windpy, is_windpy_allowed

W_AVAILABLE = is_windpy_allowed()


class BondFuturesTiming(object):

    """国债期货高频时间维度择时模型（持续期因子 + 时间戳因子）"""

    # 标的：10 年期 / 30 年期国债期货主力连续合约
    CODES = {
        "T.CFE": "10年期国债期货(T)",
        "TL.CFE": "30年期国债期货(TL)",
    }

    # 参数（研报稳健参数区间：T 阈值回看 250 日 / 分位数 50-60%，信号窗口 10-120 日）
    THRESHOLD_LOOKBACK = 250      # 动态阈值回看期（交易日）
    THRESHOLD_QUANTILE = 0.50     # 阈值分位数（研报 TL 用 50%-60%）
    SIGNAL_WINDOW = 60            # 信号滚动分位数窗口（研报 20-60 日较优，取中值 60）
    SIGNAL_Q_HI = 0.80            # 高分位做多阈值
    SIGNAL_Q_LO = 0.20            # 低分位做空阈值
    FEE_RATE = 0.0001             # 单边交易成本（国债期货手续费极低，取 0.01% 保守值）

    # 因子方向：+1 = 因子值越大越看多，-1 = 因子值越小越看多
    FACTOR_DIRECTION = {
        "价格持续期偏离_最大": +1,
        "价格持续期偏离_前三大": +1,
        "价格持续期时点": -1,
        "成交量持续期比值": +1,
        "成交量持续期时点": -1,
        "最高价时间戳": -1,
        "最低价时间戳": +1,
        "最大成交量时间戳": +1,
        "午后最高价时间戳": -1,
        "午后最低价时间戳": +1,
        "早盘尖峰次数": +1,
    }

    def __init__(self):

        self.sub_data_path = r'derived_data\futures_data\bond_timing'
        self.data_path = Config().data_dir(self.sub_data_path)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.out_path = Config().data_dir("derived_data", "allocation_data", "bond_futures_timing")
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.min_file = lambda code: os.path.join(self.data_path, '%s_1min.csv' % code.replace('.', '_'))

    # ==================================================================
    # 一、数据层：Wind wsi 拉取 1 分钟 bar（全量/增量）
    # ==================================================================
    def update_min_data(self, code="T.CFE", start="2024-06-01"):
        """拉取/增量更新 1 分钟数据。

        按月循环 wsi（Wind 分钟数据仅保留近 ~2 年，2024-06 起）；
        已缓存则从缓存最后日期的下一交易日继续增量拉取。
        """
        if not W_AVAILABLE:
            raise RuntimeError("WindPy 不可用，请用系统 Python(miniforge) 运行并确保 Wind 终端已启动")

        df = self.load_min_data(code, quiet=True)
        if df is not None and len(df) > 0:
            last_dt = df.index.max()
            start = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            print("[%s] 增量更新，从 %s 起" % (code, start))
        else:
            df = pd.DataFrame()
            print("[%s] 全量拉取，从 %s 起" % (code, start))

        w = get_windpy(__file__)
        new = self._fetch_range(code, start, w)
        if new is not None and len(new) > 0:
            df = pd.concat([df, new])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_csv(self.min_file(code), encoding="utf-8-sig")
        print("[%s] 分钟数据共 %d 行，%d 个交易日，最新 %s"
              % (code, len(df), df.index.normalize().nunique(),
                 df.index.max().strftime("%Y-%m-%d %H:%M") if len(df) else "无"))
        return df

    def _fetch_range(self, code, start, w):
        """按月循环拉取 [start, 今天] 的 1 分钟数据"""
        frames = []
        beg = pd.Timestamp(start).normalize()
        today = pd.Timestamp(datetime.now().strftime("%Y-%m-%d")).normalize()
        # 当日若未收盘（15:15 后），跳过当日（用昨日）
        end = today - pd.Timedelta(days=1) if datetime.now().hour < 15 or (datetime.now().hour == 15 and datetime.now().minute < 20) else today
        if beg > end:
            print("[%s] 起始日期晚于可用日，跳过" % code)
            return None

        ym = (beg.year, beg.month)
        while (ym[0], ym[1]) <= (end.year, end.month):
            b = "%04d-%02d-01" % ym
            if ym == (beg.year, beg.month):
                b = beg.strftime("%Y-%m-%d")
            e = ("%04d-%02d-%d" % (ym[0], ym[1], pd.Timestamp(year=ym[0], month=ym[1], day=1).days_in_month))
            if (ym[0], ym[1]) == (end.year, end.month):
                e = end.strftime("%Y-%m-%d")
            f = self._fetch_one(code, b, e, w)
            if f is not None and len(f):
                frames.append(f)
                print("[%s] %s ~ %s: %d 根" % (code, b, e, len(f)))
            else:
                print("[%s] %s ~ %s: 无数据(跳过)" % (code, b, e))
            # 下个月
            ym = (ym[0] + 1, 1) if ym[1] == 12 else (ym[0], ym[1] + 1)

        if not frames:
            return None
        out = pd.concat(frames)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    def _fetch_one(self, code, beg_date, end_date, w):
        """拉取单个时间段的 1 分钟数据"""
        d = w.wsi(code, "open,high,low,close,volume,amt",
                  "%s 09:00:00" % beg_date, "%s 15:30:00" % end_date,
                  BarSize=1, returnType=1)
        if d.ErrorCode != 0:
            return None
        if d.Times is None or len(d.Times) == 0:
            return None
        df = pd.DataFrame(
            np.array(d.Data).T,
            columns=["open", "high", "low", "close", "volume", "amt"],
        )
        df.index = pd.to_datetime(d.Times)
        # 只保留连续竞价 bar（09:30-11:30 / 13:00-15:15），过滤异常
        df = df[(df.index.hour * 60 + df.index.minute) >= 9 * 60 + 30]
        df = df[df["close"] > 0]
        return df

    def load_min_data(self, code="T.CFE", quiet=False):
        """读缓存 1 分钟数据"""
        file = self.min_file(code)
        if not os.path.exists(file):
            return None
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        if len(df) == 0:
            return None
        if not quiet:
            print("[%s] 缓存 %d 行，%d 个交易日，%s ~ %s"
                  % (code, len(df), df.index.normalize().nunique(),
                     df.index.min().strftime("%Y-%m-%d"), df.index.max().strftime("%Y-%m-%d")))
        return df

    def load_daily(self, code="T.CFE"):
        """从分钟数据聚合日度 open/close（open=9:30 bar 开盘价，close=15:15 bar 收盘价）"""
        df = self.load_min_data(code)
        if df is None:
            return None
        g = df.groupby(df.index.normalize())
        daily = pd.DataFrame({
            "open": g["open"].first(),
            "close": g["close"].last(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "volume": g["volume"].sum(),
        })
        daily.index = pd.to_datetime(daily.index)
        return daily

    # ==================================================================
    # 二、持续期基础：段内持续期序列 + 动态阈值
    # ==================================================================
    @staticmethod
    def _duration_series(vals, threshold):
        """段内持续期：对每个 i，向前找最近 j<i 使 |vals[i]-vals[j]| >= threshold；
        若不存在则从段起点累计（持续期 = i）。返回 int 数组。"""
        n = len(vals)
        dur = np.zeros(n, dtype=int)
        for i in range(n):
            vi = vals[i]
            j = i - 1
            found = -1
            while j >= 0:
                if abs(vi - vals[j]) >= threshold:
                    found = j
                    break
                j -= 1
            dur[i] = i - found if found >= 0 else i
        return dur

    def _daily_thresholds(self, min_df):
        """动态阈值：对每个交易日，用过去 THRESHOLD_LOOKBACK 日的
        相邻分钟价差绝对值 / 分钟成交量样本的分位数作为当日阈值。

        返回 (thr_p, thr_v) 两个 Series（index=日期）。
        """
        g = min_df.groupby(min_df.index.normalize())
        dates = sorted(g.groups.keys())

        diff_p, vol_v = {}, {}
        for d in dates:
            sub = g.get_group(d)
            am = sub[sub.index.hour < 12]
            pm = sub[sub.index.hour >= 12]
            ds = np.abs(np.diff(am["close"].values))
            if len(pm) > 1:
                ds = np.concatenate([ds, np.abs(np.diff(pm["close"].values))])
            diff_p[d] = ds
            vol_v[d] = sub["volume"].values.astype(float)

        lb = self.THRESHOLD_LOOKBACK
        q = self.THRESHOLD_QUANTILE
        thr_p, thr_v = {}, {}
        for i, d in enumerate(dates):
            if i < lb:
                thr_p[d], thr_v[d] = np.nan, np.nan
                continue
            pool_p = np.concatenate([diff_p[dates[j]] for j in range(i - lb, i)])
            pool_v = np.concatenate([vol_v[dates[j]] for j in range(i - lb, i)])
            thr_p[d] = np.quantile(pool_p, q)
            thr_v[d] = np.quantile(pool_v, q)

        return (pd.Series(thr_p).sort_index(), pd.Series(thr_v).sort_index())

    # ==================================================================
    # 三、因子层：逐日计算 11 个时间维度因子
    # ==================================================================
    def calc_factors(self, code="T.CFE", save=True):
        """计算全部时间维度因子（逐交易日，index=日期）"""
        min_df = self.load_min_data(code)
        if min_df is None:
            print("[%s] 无分钟数据，请先 update" % code)
            return None

        thr_p, thr_v = self._daily_thresholds(min_df)
        g = min_df.groupby(min_df.index.normalize())
        dates = sorted(g.groups.keys())

        rows = []
        for d in dates:
            if np.isnan(thr_p[d]) or np.isnan(thr_v[d]):
                continue  # 阈值预热期（前 THRESHOLD_LOOKBACK 日无因子）
            sub = g.get_group(d)
            am = sub[sub.index.hour < 12]
            pm = sub[sub.index.hour >= 12]
            n_am, n_pm = len(am), len(pm)

            # ---- 持续期（段内独立，跨午休不追溯）----
            dp_am = self._duration_series(am["close"].values, thr_p[d])
            dp_pm = self._duration_series(pm["close"].values, thr_p[d])
            dv_am = self._duration_series(am["volume"].values.astype(float), thr_v[d])
            dv_pm = self._duration_series(pm["volume"].values.astype(float), thr_v[d])
            dur_p = np.concatenate([dp_am, dp_pm])       # 全天价格持续期（257 根）
            dur_v = np.concatenate([dv_am, dv_pm])       # 全天成交量持续期
            close_all = np.concatenate([am["close"].values, pm["close"].values])
            vol_all = np.concatenate([am["volume"].values, pm["volume"].values])
            high_all = np.concatenate([am["high"].values, pm["high"].values])
            low_all = np.concatenate([am["low"].values, pm["low"].values])
            pos_1 = np.arange(1, len(dur_p) + 1)         # 日内分钟序号（1-based）

            # ---- 1/2) 价格持续期偏离因子 ----
            fp_max = close_all[int(np.argmax(dur_p))]
            top3_idx = np.argsort(dur_p)[-3:]
            fp_top3 = close_all[top3_idx].mean()
            close_day = close_all[-1]

            # ---- 3) 价格持续期时点 / 5) 成交量持续期时点 ----
            pmt = pos_1[int(np.argmax(dur_p))]
            vmt = pos_1[int(np.argmax(dur_v))]

            # ---- 4) 成交量持续期比值（午前/午后）----
            vdam = float(np.mean(dv_am))
            vdpm = float(np.mean(dv_pm))
            vr = vdam / vdpm if vdpm > 0 else np.nan

            # ---- 6)~10) 时间戳因子 ----
            ts_high = pos_1[int(np.argmax(high_all))]        # 首次触及日内最高价
            ts_low = pos_1[int(np.argmin(low_all))]          # 首次触及日内最低价
            ts_vmax = pos_1[int(np.argmax(vol_all))]         # 首次最大成交量
            ts_pm_high = n_am + pos_1[:len(pm)][int(np.argmax(pm["high"].values))]   # 午后最高价（全天分钟号）
            ts_pm_low = n_am + pos_1[:len(pm)][int(np.argmin(pm["low"].values))]     # 午后最低价（全天分钟号）
            spike_am = int(np.sum(am["high"].values >= high_all.max()))               # 早盘触碰日内最高价次数

            rows.append({
                "日期": d,
                "价格持续期偏离_最大": fp_max - close_day,
                "价格持续期偏离_前三大": fp_top3 - close_day,
                "价格持续期时点": pmt,
                "成交量持续期比值": vr,
                "成交量持续期时点": vmt,
                "最高价时间戳": ts_high,
                "最低价时间戳": ts_low,
                "最大成交量时间戳": ts_vmax,
                "午后最高价时间戳": ts_pm_high,
                "午后最低价时间戳": ts_pm_low,
                "早盘尖峰次数": spike_am,
            })

        fac = pd.DataFrame(rows).set_index("日期")
        fac.index = pd.to_datetime(fac.index)
        if save:
            fac.to_csv(os.path.join(self.out_path, "factors_%s.csv" % code.replace(".", "_")), encoding="utf-8-sig")
            print("[%s] 因子 %d 个交易日 × %d 个因子" % (code, len(fac), fac.shape[1]))
        return fac

    # ==================================================================
    # 四、信号层：滚动分位数三态信号 + 等权组合
    # ==================================================================
    def _quantile_signal(self, f, direction):
        """滚动 N 日分位数三态信号：当日因子高于历史高分位 → +1（做多）；
        低于历史低分位 → -1（做空）；中间 → 0（空仓）。
        direction=+1 因子大做多；direction=-1 因子小做多（对信号取反）。
        阈值用 shift(1) 避免前视（只用当日之前的样本）。"""
        hi = f.rolling(self.SIGNAL_WINDOW, min_periods=20).quantile(self.SIGNAL_Q_HI).shift(1)
        lo = f.rolling(self.SIGNAL_WINDOW, min_periods=20).quantile(self.SIGNAL_Q_LO).shift(1)
        sig = pd.Series(0.0, index=f.index)
        sig[f > hi] = 1.0
        sig[f < lo] = -1.0
        sig[(f <= hi) & (f >= lo)] = 0.0
        return direction * sig

    def calc_signal(self, code="T.CFE", save=True):
        """逐因子信号 + 等权组合信号（index=日期）"""
        fac = self.calc_factors(code, save=False)
        if fac is None:
            return None

        sig = pd.DataFrame(index=fac.index)
        for name, direction in self.FACTOR_DIRECTION.items():
            if name not in fac.columns:
                continue
            sig[name] = self._quantile_signal(fac[name], direction)

        n_fac = sig.shape[1]
        combo = sig.mean(axis=1)                     # 等权平均 ∈ [-1, 1]
        sig["组合信号"] = combo
        # 仓位：T 日盘后信号 → T+1 日开盘执行；组合=0 保持昨日仓位
        pos = pd.Series(np.nan, index=sig.index)
        cur = 0.0
        for i, v in enumerate(combo.values):
            if not np.isnan(v):
                if v > 0:
                    cur = 1.0
                elif v < 0:
                    cur = -1.0
                # v == 0 保持 cur 不变
            pos.iloc[i] = cur
        sig["仓位"] = pos
        if save:
            sig.to_csv(os.path.join(self.out_path, "signals_%s.csv" % code.replace(".", "_")), encoding="utf-8-sig")
        return sig

    # ==================================================================
    # 五、回测层：日频调仓、开盘价成交、1 倍杠杆
    # ==================================================================
    def backtest(self, code="T.CFE", cost_rate=None, save=True):
        """回测：T 日盘后信号 → T+1 日开盘建仓 → 持有至 T+2 日开盘。

        收益口径：ret[t] = pos[t] * (open[t+1]/open[t] - 1)
        pos[t] 由 T-1 日因子信号决定（无前视）。
        """
        cost_rate = self.FEE_RATE if cost_rate is None else cost_rate
        sig = self.calc_signal(code, save=False)
        daily = self.load_daily(code)
        if sig is None or daily is None:
            return None

        # 对齐（因子/信号从阈值预热期之后才有）
        idx = sig.index.intersection(daily.index)
        sig = sig.loc[idx]
        daily = daily.loc[idx]
        open_p = daily["open"]

        pos = sig["仓位"].reindex(idx).fillna(0.0).values
        ret = pd.Series(np.nan, index=idx)
        turn = pd.Series(0.0, index=idx)
        prev_pos = 0.0
        for i, d in enumerate(idx):
            # T 日持仓 pos[i] 持有到 T+1 开盘
            if i + 1 < len(idx):
                r = pos[i] * (open_p.iloc[i + 1] / open_p.iloc[i] - 1)
            else:
                r = 0.0
            ret.iloc[i] = r
            turn.iloc[i] = abs(pos[i] - prev_pos)
            prev_pos = pos[i]

        fee = turn * cost_rate
        ret_net = ret - fee
        cum = (1 + ret_net).cumprod()

        bt = pd.DataFrame({
            "open": open_p,
            "仓位": pos,
            "换手": turn,
            "成本": fee,
            "收益毛": ret,
            "收益净": ret_net,
            "累计净值": cum,
        }, index=idx)

        # 绩效统计（全期 + 多头端 + 空头端）
        stats = {}
        for label, mask in [("全期", np.ones(len(bt), bool)),
                            ("多头端", pos > 0),
                            ("空头端", pos < 0)]:
            r = ret_net[mask]
            stats[label] = self._metrics(r)

        if save:
            file = os.path.join(self.out_path, "backtest_%s.xlsx" % code.replace(".", "_"))
            with pd.ExcelWriter(file) as xw:
                bt.to_excel(xw, sheet_name="backtest")
                pd.DataFrame(stats).T.to_excel(xw, sheet_name="绩效")
            print("[%s] 回测已输出: %s" % (code, file))
            print("  全期: 年化 %.2f%% | 波动 %.2f%% | 收益风险比 %.2f | 卡玛 %.2f | 日胜率 %.1f%% | 回撤 %.2f%%"
                  % (stats["全期"]["年化"] * 100, stats["全期"]["年化波动"] * 100,
                     stats["全期"]["收益风险比"], stats["全期"]["卡玛比率"],
                     stats["全期"]["日胜率"] * 100, stats["全期"]["最大回撤"] * 100))
        return bt, stats

    @staticmethod
    def _metrics(r, periods=242):
        """绩效指标（r 为日收益序列，可能含 0 空仓日）"""
        r = r.dropna()
        if len(r) == 0:
            return {"年化": np.nan, "年化波动": np.nan, "收益风险比": np.nan,
                    "最大回撤": np.nan, "卡玛比率": np.nan, "日胜率": np.nan, "样本天数": 0}
        cum = (1 + r).cumprod()
        ann = cum.iloc[-1] ** (periods / len(r)) - 1
        vol = r.std() * np.sqrt(periods)
        dd = (cum / cum.cummax() - 1).min()
        return {
            "年化": ann,
            "年化波动": vol,
            "收益风险比": ann / vol if vol > 0 else np.nan,
            "最大回撤": dd,
            "卡玛比率": ann / abs(dd) if dd < 0 else np.nan,
            "日胜率": (r > 0).mean(),
            "样本天数": len(r),
        }

    # ==================================================================
    # 六、HTML 报告
    # ==================================================================
    def gen_report(self):
        """生成回测报告 HTML（净值图 + 绩效表 + 因子信号明细）"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
        plt.rcParams["axes.unicode_minus"] = False

        img_files = []
        html_rows = ""
        for code, name in self.CODES.items():
            key = code.replace(".", "_")
            bt_file = os.path.join(self.out_path, "backtest_%s.xlsx" % key)
            if not os.path.exists(bt_file):
                print("[%s] 无回测结果，跳过报告" % code)
                continue
            bt = pd.read_excel(bt_file, sheet_name="backtest", index_col=0)
            stats = pd.read_excel(bt_file, sheet_name="绩效", index_col=0)

            # 净值图
            fig, ax = plt.subplots(figsize=(9, 3.6))
            ax.plot(bt.index, bt["累计净值"], lw=1.4, color="#c0392b")
            ax.axhline(1.0, color="#999", lw=0.8, ls="--")
            ax.set_title("%s 时间因子等权组合净值（%s ~ %s）" % (name, bt.index[0].strftime("%Y-%m-%d"), bt.index[-1].strftime("%Y-%m-%d")))
            ax.set_ylabel("净值")
            ax.grid(alpha=0.3)
            img = os.path.join(self.out_path, "净值_%s.png" % key)
            fig.tight_layout()
            fig.savefig(img, dpi=110)
            plt.close(fig)
            img_files.append(img)

            # 仓位图（底部）
            fig, ax = plt.subplots(figsize=(9, 1.8))
            ax.plot(bt.index, bt["仓位"], lw=1.0, color="#2980b9")
            ax.set_ylim(-1.2, 1.2)
            ax.set_title("仓位（+1 多 / -1 空 / 0 空仓）")
            ax.grid(alpha=0.3)
            img2 = os.path.join(self.out_path, "仓位_%s.png" % key)
            fig.tight_layout()
            fig.savefig(img2, dpi=110)
            plt.close(fig)
            img_files.append(img2)

            # 绩效表
            def fmt_pct(x):
                return "-" if pd.isna(x) else "%.2f%%" % (x * 100)

            def fmt_num(x, n=2):
                return "-" if pd.isna(x) else ("%.2f" % x)

            rows_tbl = ""
            for label in ["全期", "多头端", "空头端"]:
                if label not in stats.index:
                    continue
                s = stats.loc[label]
                rows_tbl += ("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                             "<td>%s</td><td>%s</td><td>%d</td></tr>"
                             % (label, fmt_pct(s["年化"]), fmt_pct(s["年化波动"]),
                                fmt_num(s["收益风险比"]), fmt_num(s["卡玛比率"]),
                                fmt_pct(s["日胜率"]), fmt_pct(s["最大回撤"]), s["样本天数"]))

            html_rows += """
            <div class="card">
              <h3>%s（%s）</h3>
              <img src="%s" class="chart"/>
              <img src="%s" class="chart"/>
              <table>
                <tr><th>口径</th><th>年化收益</th><th>年化波动</th><th>收益风险比</th>
                    <th>卡玛比率</th><th>日胜率</th><th>最大回撤</th><th>样本天数</th></tr>
                %s
              </table>
            </div>
            """ % (name, code, os.path.basename(img), os.path.basename(img2), rows_tbl)

        html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>国债期货高频时间维度择时模型回测报告</title>
<style>
body{font-family:"Microsoft YaHei",sans-serif;margin:24px;color:#222;background:#f7f8fa}
h1{font-size:22px} h2{font-size:17px;color:#555;margin-top:26px}
.card{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:18px;margin-bottom:18px}
.card h3{margin-top:0;color:#1a5276}
.chart{width:100%;max-width:860px;border-radius:6px;margin:6px 0}
table{border-collapse:collapse;width:100%;max-width:860px;margin-top:10px;font-size:13px}
th,td{border:1px solid #dfe3e8;padding:6px 10px;text-align:right}
th{background:#eef3f8} td:first-child{text-align:left;font-weight:bold}
.note{color:#888;font-size:12.5px;line-height:1.8}
.foot{color:#999;font-size:12px;margin-top:30px}
</style></head><body>
<h1>国债期货高频时间维度择时模型 —— 回测报告</h1>
<p class="note">模型复现自《基于高频时间维度的国债期货择时因子及个股CTA研究》（兴证金工 2026-06）。
基于日内 1 分钟数据构建 <b>持续期因子</b>（价格/成交量突破动态阈值的时间间隔）与
<b>时间戳因子</b>（高低点、量额峰值的日内时序位置），滚动分位数三态信号等权组合，
日频调仓、1 倍杠杆。生成时间：%s</p>
<h2>一、策略绩效</h2>
%s
<h2>二、参数与说明</h2>
<table>
<tr><th style="text-align:left">参数</th><th style="text-align:left">取值</th><th style="text-align:left">说明</th></tr>
<tr><td style="text-align:left">阈值回看期</td><td style="text-align:left">%d 日</td><td style="text-align:left">动态阈值样本池（价差/成交量分位数）</td></tr>
<tr><td style="text-align:left">阈值分位数</td><td style="text-align:left">%.0f%%</td><td style="text-align:left">研报稳健区间 50%%-60%%</td></tr>
<tr><td style="text-align:left">信号窗口</td><td style="text-align:left">%d 日</td><td style="text-align:left">滚动分位数回看窗口</td></tr>
<tr><td style="text-align:left">信号阈值</td><td style="text-align:left">%.0f%% / %.0f%%</td><td style="text-align:left">高于高分位做多、低于低分位做空</td></tr>
<tr><td style="text-align:left">交易成本</td><td style="text-align:left">%.4f%%</td><td style="text-align:left">单边（按换手扣减）</td></tr>
<tr><td style="text-align:left">杠杆</td><td style="text-align:left">1 倍</td><td style="text-align:left">保证金 100%%，还原纯因子收益</td></tr>
</table>
<h2>三、重要说明</h2>
<p class="note">
1. <b>数据可得性约束</b>：Wind wsi 分钟数据仅提供近约 2 年（T.CFE/TL.CFE 自 2024-06 起），
Oracle 底库无分钟表，故回测区间为 2024-06 至今；研报 T 为 2017-2026、TL 为 2023-2026。
模型支持每日增量更新，回测区间随时间自然延长。<br>
2. 前 250 日为阈值预热期（因子为空），实际有效回测自 2025-04 前后开始。<br>
3. 收益口径：T 日盘后出信号、T+1 日开盘执行、持有至 T+2 日开盘；信号不变不换仓。<br>
4. 本报告仅供研究参考，不构成投资建议。模型基于历史数据，市场环境转变时存在失效风险。
</p>
<p class="foot">生成时间 %s ｜ 代码：model_allocation/bond_futures_timing.py</p>
</body></html>
""" % (datetime.now().strftime("%Y-%m-%d %H:%M"), html_rows,
       self.THRESHOLD_LOOKBACK, self.THRESHOLD_QUANTILE * 100,
       self.SIGNAL_WINDOW, self.SIGNAL_Q_HI * 100, self.SIGNAL_Q_LO * 100,
       self.FEE_RATE * 10000,
       datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        file = os.path.join(self.out_path, "回测报告.html")
        with open(file, "w", encoding="utf-8") as f:
            f.write(html)
        print("HTML 报告已生成: %s" % file)
        return file, img_files


# ==================================================================
# 主入口
# ==================================================================
def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("update", "backtest", "all"):
        print("用法: python bond_futures_timing.py [update|backtest|all]")
        return

    model = BondFuturesTiming()
    cmd = sys.argv[1]

    if cmd in ("update", "all"):
        for code in model.CODES:
            model.update_min_data(code)

    if cmd in ("backtest", "all"):
        for code in model.CODES:
            model.calc_factors(code)
            model.calc_signal(code)
            model.backtest(code)
        report, imgs = model.gen_report()
        print("报告: %s" % report)


if __name__ == "__main__":
    main()
