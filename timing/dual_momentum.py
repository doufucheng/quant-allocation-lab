# -*- coding: utf-8 -*-
# 以脚本方式直接运行时，把项目根目录加入 sys.path
import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import datetime
import math
import pandas as pd
import numpy as np

from utility.write_excel import WriteExcel
from utility.date import Date
from config.config import Config
from utility.windpy_guard import get_windpy


class DualMomentum(object):

    """
    双动量模型
    双动量：时序动量和相对动量都是正
    只买一个或者资产

    当交易标的大幅度变化的时候，收益率会很不一样
    当持有标的为1或者2时候结果也会不一样

    ======================= 2026-08 优化过程记录 =======================
    本轮优化共四步，前两步落地（成交价修复+交易成本+涨跌停约束），后两步
    实验后回退/保留解耦结构。以下按时间顺序记录，避免重复踩坑。

    [1] VWAP 成交价问题（已修复，vwap 数据与代码已删除）
        现象：原 backtest() 第125行把 vwap 文件路径误写成 close（文件名 bug），
        名义上"按均价成交"，实际一直读的是收盘价文件。
        尝试修复时改读真实 vwapetf.csv / vwapindex.csv 后，回测出现
        2026-08-24 单日 +201% 的假收益。
        根因：Wind 的 vwap 字段在 PriceAdj=B 下【不做分红复权】——
          - 长期分红标的（纳指ETF 513100）vwap/close 比值从 1.0 漂移至 0.2
          - 货币ETF 511880 比值 0.83
          - 指数池 NDX.GI 等量纲错位（比值 ~0.01）
          - 期货主力合约（M.DCE / I.DCE）因展期比值大幅跳变（0.44~0.63）
        跨日收益混用 vwap 会在除息日产生假跳空；假收益即来自卖出腿落在
        vwap 缺失日回退 close 时跨越了 5 倍复权因子。
        最终方案：成交价统一用后复权 close（与信号同源、内部一致），
        "按均价成交"的执行假设差异并入 cost_rate；vwap 相关代码与
        vwapetf.csv / vwapindex.csv 数据文件全部删除。

    [2] 交易成本 + 可交易性约束（已加入）
        backtest(cost_rate=0.0005)：按持仓变化计算换手，双边扣减（2x单边），
        现金-现金切换不计成本，输出 turnover / cost / ret_gross 三列。
        可交易性：买入日当日涨幅>+9.5% 视为涨停买不进、卖出日<-9.5% 视为
        跌停卖不掉，成交日顺延（最多10日），两条持仓腿分别处理。
        结果（ETF池/Index池，2026-08-27 数据）：
          费前年化 20.58%/24.04% -> 费后 16.51%/18.83%
          年化成本侵蚀 -4.07 / -5.21 pct；涨跌停顺延 23 / 16 次
        意外收获：涨跌停约束反而提升年化约 +1.9 / +1.2 pct——2026 年
        原油LOF 溢价泡沫期模型想追买，涨停买不进强制顺延，等能买时
        泡沫已破，约束恰好躲过追高。

    [3] 信号计算与回测解耦（已保留）
        calc_signal() 每日计算 25 日回归动量得分，独立存 signal{sheet}.csv；
        backtest() 只读信号文件+收盘价。回测不再内部重复算信号。

    [4] 周频调仓实验（负结果，已回退日频）
        实验：信号仍每日计算，仅调仓频率 D->W（每周最后交易日出信号、
        T+1 收盘成交），成本/涨跌停约束不变。
        结果（费后年化/夏普/年均单边换手）：
          ETF 池 : 日频 16.51% / 0.82 / 34.3倍 -> 周频 11.95% / 0.64 / 22.0倍
          Index池: 日频 18.83% / 0.83 / 42.9倍 -> 周频 16.99% / 0.72 / 25.1倍
        费前年化损失 4~6pct，远大于省下的成本 1.5~2.2pct/年，净亏损。
        原因：25 日动量得分的日度排序极不稳定——调仓日与 lag-1 日的 Top2
        完全重合率仅 79%，lag-3 降至 67%，周度采样错过中途反转日。
        结论：降频必须改信号本身（更长回看期 60/120/250 日、加缓冲区
        规则），不能只改调仓频率。相关代码与 backtest2_*_W.xlsx 已删除。

    ======================= 当前基线（2026-08-27） =======================
    ETF 池（2013-12~2026-08，hold_num=2，日频）：费后年化 16.51%，夏普 0.82，
      最大回撤 -27.1%，总换手 419.5 倍单边，成本拖累 34.3%
    Index 池（2004-12~2026-08）：费后年化 18.83%，夏普 0.83，回撤 -35.1%，
      总换手 899.5 倍单边，成本拖累 59.3%
    下一步方向：60/120/250 日回看期对比、调仓缓冲区（top1/top3 不换）、
      等权基准对比。

    ======================= 输出说明（2026-08-27 双sheet） =======================
    backtest{hold_num}_{sheet}.xlsx 含两个 sheet：
      backtest_gross：费前口径。仅回测逻辑收益（ret/ret1/ret2）+ 持仓明细
        + 年度/总体 summary，不含任何成本列。
      backtest_net：费后口径。在费前明细基础上增加成本相关列——
        ret_gross(费前日收益) / turnover(当日单边换手0~1) / cost(当日双边扣费) /
        cost_drag(累计费率磨损) / cum_turnover(累计单边换手) /
        cumret_gross(费前累计) ；右侧附 summary 与成本统计块
        （单边费率/累计磨损/费前费后年化/年化侵蚀pct/总换手/年均换手/顺延次数）。
    """

    def __init__(self):

        self.etf_pool = []
        # 双动量模型专用数据目录（2026-08-27 从 allocation_data 迁出，与其他模型隔离）
        self.data_path = Config().data_dir("derived_data", "allocation_data", "asset_momentum")

    def get_info(self, sheet_name="etf", hold_num=1, period="D"):

        # 参数文件
        self.sheet_name = sheet_name
        self.hold_num = hold_num
        self.period = period

        # 标的（资产池清单见 etf_momentum.xlsx，最新信号输出已含全部标的名称）
        file = os.path.join(Config().param_path, "etf_momentum.xlsx")
        data = pd.read_excel(file, index_col=0, sheet_name=self.sheet_name)
        self.etf_pool = data.index
        self.etf_name = pd.DataFrame(data["名称"]).to_dict()["名称"]

        self.load_beg_date = "2003-12-01"
        self.period_num = Date().get_period_number_for_year(self.period)

        # 现金代码与回测起点（按池区分）
        if "etf" in self.sheet_name:
            self.cash_code = "511880.SH"
            self.backtest_date = "20131220"
        else:
            self.cash_code = "H11025.CSI"
            self.backtest_date = "20041220"


    def _load_local_prices(self, codes, beg):
        """本地底表价格（2026-09-22 迁移，db_data_missing.md §五）。

        覆盖映射（与 WindPy wsd close PriceAdj=B 同口径的部分）：
        - 6 位数字 .SH/.SZ 且非 399 开头（场内 ETF）-> ETFEOD.pq AdjClose（后复权）
        - 399xxx.SZ / .CSI / .CNI（A 股指数）      -> IndexEOD.pq Close（大小写不敏感）
        - AU9999.SGE（上金所现货）                   -> db_net/au9999_sge.csv
        - HSTECH.HI / HSAHP.HI（港股指数）           -> api_wind/ah_premium_index.pq
        不覆盖（保留 WindPy，PriceAdj=B 展期/汇率复权口径本地无等价）：
        期货（.DCE/.SHF/.INE/.CZC/.GFE/.DCF 等）、NDX.GI、CBA21801.CS。

        返回 (宽表 DataFrame[日期×代码], 未覆盖代码列表)。
        """
        import pyarrow.parquet as pq

        beg_ts = pd.Timestamp(beg)
        out, missing = {}, []
        fut_suffixes = {".DCE", ".SHF", ".INE", ".CZC", ".GFE", ".DCF"}

        etf_codes, idx_codes = [], []
        for c in codes:
            head, _, suf = c.partition(".")
            if c == "AU9999.SGE":
                p = os.path.join(str(Config().db_data_path), "db_net",
                                 "au9999_sge.csv")
                if os.path.exists(p):
                    s = pd.read_csv(p, index_col=0, parse_dates=True)["Close"]
                    out[c] = s[s.index >= beg_ts]
                else:
                    missing.append(c)
            elif c in ("HSTECH.HI", "HSAHP.HI"):
                p = os.path.join(str(Config().api_wind_path),
                                 "ah_premium_index.pq")
                if os.path.exists(p):
                    d = pd.read_parquet(p)
                    sub = d[d["Ticker"] == c].set_index("Date")["Close"]
                    sub.index = pd.to_datetime(sub.index.astype(str).str[:10])
                    out[c] = sub[sub.index >= beg_ts]
                else:
                    missing.append(c)
            elif suf in ("SH", "SZ") and len(head) == 6 and not head.startswith("399"):
                etf_codes.append(c)
            elif suf in ("CSI", "CNI") or (suf == "SZ" and head.startswith("399")):
                idx_codes.append(c)
            elif "." + suf in fut_suffixes or suf in ("GI", "CS"):
                missing.append(c)          # 期货/海外指数/中债指数 -> WindPy
            else:
                missing.append(c)

        # ETFEOD.pq AdjClose（后复权，与 wsd PriceAdj=B 同源）
        if etf_codes:
            t = pq.read_table(os.path.join(str(Config().db_wind_path), "ETFEOD.pq"),
                              columns=["Date", "Ticker", "AdjClose"],
                              filters=[("Ticker", "in", etf_codes)]).to_pandas()
            if "Date" not in t.columns:
                t = t.reset_index()
            for c, sub in t.dropna(subset=["AdjClose"]).groupby("Ticker"):
                s = sub.set_index("Date")["AdjClose"].astype(float).sort_index()
                s.index = pd.to_datetime(s.index.astype(str).str[:10])
                out[c] = s[s.index >= beg_ts]
            missing += [c for c in etf_codes if c not in out]

        # IndexEOD.pq Close（指数无分红调整；h 前缀全收益代码库内为小写）
        if idx_codes:
            want = {c.lower() for c in idx_codes}
            tcol = pq.read_table(os.path.join(str(Config().db_wind_path),
                                              "IndexEOD.pq"),
                                 columns=["Ticker"]).column("Ticker")
            lib = pd.unique(tcol.to_pandas())
            exact = sorted(x for x in lib if str(x).lower() in want)
            t = pq.read_table(os.path.join(str(Config().db_wind_path), "IndexEOD.pq"),
                              columns=["Date", "Ticker", "Close"],
                              filters=[("Ticker", "in", exact)]).to_pandas()
            if "Date" not in t.columns:
                t = t.reset_index()
            lower_back = {str(x).lower(): x for x in exact}
            t["Ticker"] = t["Ticker"].map(lambda x: lower_back.get(str(x).lower()))
            for c, sub in t.dropna(subset=["Close"]).groupby("Ticker"):
                s = sub.set_index("Date")["Close"].astype(float).sort_index()
                s.index = pd.to_datetime(s.index.astype(str).str[:10])
                out[c] = s[s.index >= beg_ts]
            missing += [c for c in idx_codes if c not in out]

        wide = pd.DataFrame(out)
        if len(wide.columns):
            wide = wide.sort_index()
        return wide, missing

    def update_data(self):

        # 日期（默认 cj 本地交易日历；原 source="wind" 需 Wind 终端，
        # aidou 主机被 windpy_guard 禁用，2026-09-22 改本地源）
        Date().load_trade_date_series()

        # 价格（后复权收盘价，信号与成交同源）
        # 注：Wind vwap 字段在 PriceAdj=B 下不做分红复权（长期分红标的 vwap/close
        #     比值从 1.0 漂移至 0.2，指数/期货池量纲错位），不可用于跨日收益，
        #     已弃用；成交统一用后复权 close，执行价差并入 cost_rate。
        # 2026-09-22 起：ETF/A股指数/AU9999/港股指数优先走本地底表
        # （ETFEOD AdjClose/IndexEOD/db_net/api_wind），仅期货/海外指数/
        # 中债指数保留 WindPy（PriceAdj=B 展期复权口径本地无等价）。
        today = datetime.datetime.today().strftime('%Y%m%d')
        wide, missing = self._load_local_prices(
            list(self.etf_pool), self.load_beg_date)
        print("[update_data] 本地底表覆盖 %d/%d：%s" % (
            len(wide.columns), len(self.etf_pool), list(wide.columns)))
        if missing:
            print("[update_data] 走 WindPy（本地无等价口径）：%s" % missing)
            code_str = ",".join(missing)
            w = get_windpy(__file__)
            data = w.wsd(code_str, "close", self.load_beg_date, today, "PriceAdj=B")
            wind_df = pd.DataFrame(index=data.Codes, data=data.Data,
                                   columns=data.Times).T
            wind_df.index = wind_df.index.map(Date().change_to_str)
            wide.index = wide.index.map(Date().change_to_str)
            df = wide.join(wind_df, how="outer").sort_index()
        else:
            wide.index = wide.index.map(Date().change_to_str)
            df = wide
        df = df[list(self.etf_pool)]
        file = os.path.join(self.data_path, 'close%s.csv' % self.sheet_name)
        df.to_csv(file, encoding='gbk')

    def calc_signal(self):

        """ 每日计算动量信号，与回测解耦
        动量得分 = 25日对数价格回归年化收益 x R^2（joinquant口径）
        输出 signal{sheet}.csv：每个交易日 x 池内全部标的
        """

        file = os.path.join(self.data_path, 'close%s.csv' % self.sheet_name)
        close = pd.read_csv(file, index_col=0)
        close.index = close.index.map(Date().change_to_str)
        close = close[self.etf_pool]
        close = close.loc[self.backtest_date:, ]

        mom_index = pd.DataFrame([], index=close.index, columns=close.columns)
        for i in range(25, len(close)):
            data = close.iloc[i-25:i, ]
            data = data.T.dropna(how="any").T
            date = close.index[i]

            for code in data.columns:
                df = pd.DataFrame(data[code])
                y = df['log'] = np.log(df[code])
                x = df['num'] = np.arange(df.log.size)
                slope, intercept = np.polyfit(x, y, 1)
                annualized_returns = math.pow(math.exp(slope), 250) - 1
                r_squared = 1 - (sum((y - (slope * x + intercept)) ** 2) / ((len(y) - 1) * np.var(y, ddof=1)))
                score = annualized_returns * r_squared
                mom_index.loc[date, code] = score

        file = os.path.join(self.data_path, 'signal%s.csv'% self.sheet_name)
        mom_index.to_csv(file, encoding='gbk')

        # 最新信号提示（输出全部标的，按动量得分降序；hold_num 个动量为正的为当前建议持仓）
        res = mom_index.iloc[-1].dropna().sort_values(ascending=False)
        res = pd.DataFrame(res, columns=[mom_index.index[-1]])
        res["名称"] = [self.etf_name.get(c, c) for c in res.index]
        print("信号区间: %s ~ %s" % (mom_index.index[0], mom_index.index[-1]))
        print("最新信号（全部标的，按动量得分降序）:")
        print(res.to_string())
        return mom_index


    def backtest(self, cost_rate=0.0005):

        self.cost_rate = cost_rate

        # 读取信号文件（由 calc_signal 生成）
        file = os.path.join(self.data_path, 'signal%s.csv'% self.sheet_name)
        mom_index = pd.read_csv(file, index_col=0)
        mom_index.index = mom_index.index.map(Date().change_to_str)

        # 读取收盘价（成交与可交易性判断）
        file = os.path.join(self.data_path, 'close%s.csv' % self.sheet_name)
        close = pd.read_csv(file, index_col=0)
        close.index = close.index.map(Date().change_to_str)
        close = close[self.etf_pool]
        close = close.loc[self.backtest_date:, ]

        # 根据指标回测
        # 成交价统一用后复权 close（与信号同源，vwap 已弃用，见类 docstring [1]）
        # 时序：T 日收盘出信号 -> T+1 收盘买入 -> T+2 收盘卖出（日频下相邻调仓日正好衔接）

        # 可交易性约束：买入日涨停（当日涨幅>+9.5%）的 ETF 买不进，卖出日跌停（<-9.5%）的卖不掉
        #   受限时成交日顺延到下一个可成交的交易日（最多顺延10日，仍未成交按当日价处理）
        ret_chg = close.pct_change()
        buyable = ret_chg <= 0.095
        sellable = ret_chg >= -0.095
        n_limit_delay = [0]

        def next_tradable(d, code, flags, max_step=10):
            for _ in range(max_step):
                if d not in flags.index or pd.isna(flags.loc[d, code]) or bool(flags.loc[d, code]):
                    return d
                n_limit_delay[0] += 1
                nxt = Date().get_trade_date_offset(d, 1)
                if nxt is None or nxt == d:
                    return d
                d = nxt
            return d

        mom_index = mom_index.dropna(how="all")
        # 调仓日序列（period 默认 D 日频调仓）
        change_date = Date().get_trade_date_series(mom_index.index[0], mom_index.index[-1],  self.period)
        ret = pd.DataFrame(index=change_date, columns=["ret", "code"])
        for i_date in range(0, len(change_date) - 1):
            date = change_date[i_date]
            hold = mom_index.loc[date]
            hold = hold.dropna().sort_values(ascending=False)
            hold = hold[hold > 0]
            buy_date = Date().get_trade_date_offset(date, 1)
            next_date = change_date[i_date + 1]
            sell_date = Date().get_trade_date_offset(next_date, 1)

            if self.hold_num == 1:
                if len(hold) == 0:
                    code = self.cash_code
                    ret.loc[date, "code"] = code
                    ret.loc[date, "name"] = self.etf_name[code]
                    ret.loc[date, "ret"] = 0
                else:
                    code = hold.index[0]
                    # 可交易性：买入日涨停顺延买入，卖出日跌停顺延卖出
                    buy_date = next_tradable(buy_date, code, buyable)
                    sell_date = next_tradable(sell_date, code, sellable)
                    ret.loc[date, "code"] = code
                    ret.loc[date, "name"] = self.etf_name[code]

                ret.loc[date, "buy"] = buy_date
                ret.loc[date, "sell"] = sell_date
                ret.loc[date, "ret"] = close.loc[sell_date, code] / close.loc[buy_date, code] - 1
            elif self.hold_num == 2:
                if len(hold) == 0:
                    code = self.cash_code
                    code1 = code
                    code2 = code
                    buy_date1 = buy_date
                    sell_date1 = sell_date
                    buy_date2 = buy_date
                    sell_date2 = sell_date
                    ret.loc[date, "code1"] = code1
                    ret.loc[date, "name1"] = self.etf_name[code1]
                    ret.loc[date, "code2"] = code2
                    ret.loc[date, "name2"] = self.etf_name[code2]
                    ret.loc[date, "ret1"] = 0
                    ret.loc[date, "ret2"] = 0
                    ret.loc[date, "ret"] = 0
                else:
                    code1 = hold.index[0]
                    # 可交易性：买入日涨停顺延买入，卖出日跌停顺延卖出（分腿处理）
                    buy_date1 = next_tradable(buy_date, code1, buyable)
                    sell_date1 = next_tradable(sell_date, code1, sellable)
                    ret.loc[date, "code1"] = code1
                    ret.loc[date, "name1"] = self.etf_name[code1]
                    if len(hold) == 1:
                        code2 = self.cash_code
                    else:
                        code2 = hold.index[1]
                    buy_date2 = next_tradable(buy_date, code2, buyable)
                    sell_date2 = next_tradable(sell_date, code2, sellable)
                    ret.loc[date, "code2"] = code2
                    ret.loc[date, "name2"] = self.etf_name[code2]

                ret.loc[date, "buy"] = buy_date1
                ret.loc[date, "sell"] = sell_date1
                ret.loc[date, "ret1"] = close.loc[sell_date1, code1] / close.loc[buy_date1, code1] - 1
                ret.loc[date, "ret2"] = close.loc[sell_date2, code2] / close.loc[buy_date2, code2] - 1
                ret.loc[date, "ret"] = ret.loc[date, "ret1"] * 0.5 + ret.loc[date, "ret2"] * 0.5

                for i in range(len(self.etf_pool)):

                    code = self.etf_pool[i]
                    name = self.etf_name[code]

                    if len(hold) == 1:
                        if code == hold.index[0]:
                            ret.loc[date, name] = 1.0
                        else:
                            ret.loc[date, name] = 0.0
                    else:
                        if code == hold.index[0]:
                            ret.loc[date, name] = 0.5
                        elif code == hold.index[1]:
                            ret.loc[date, name] = 0.5
                        else:
                            ret.loc[date, name] = 0


            else:
                print("不支持此参数")
                pass

        # 清理初始化占位列（hold_num=2 时 code 列全程为空）
        if "code" in ret.columns and ret["code"].isna().all():
            ret = ret.drop(columns=["code"])

        # ======================== 交易成本 ========================
        # 单边换手（0~1）：持仓腿中发生变化的占比；双边扣减 cost = turnover * 2 * cost_rate
        # 首期为建仓，非现金部分计为换手；现金-现金切换不计成本
        if "code1" in ret.columns:
            code_cols = ["code1", "code2"]
        elif "code" in ret.columns:
            code_cols = ["code"]
        else:
            code_cols = []

        if code_cols:
            # 剔除循环未填充的尾部空行
            ret = ret[ret[code_cols].notna().any(axis=1)]
            codes = ret[code_cols]
            prev = codes.shift(1)
            changed = (codes.values != prev.values)
            # 首期为建仓，非现金部分计为换手
            changed[0, :] = (codes.iloc[0].values != np.repeat(self.cash_code, len(code_cols)))
            ret["turnover"] = changed.mean(axis=1)
            ret["cost"] = ret["turnover"] * 2.0 * self.cost_rate

        # 费前/费后收益与成本磨损序列
        ret["ret_gross"] = ret["ret"].fillna(0.0)
        ret["ret"] = (ret["ret_gross"] + 1.0) * (1.0 - ret["cost"]) - 1.0
        ret["cost_drag"] = 1.0 - (1.0 - ret["cost"]).cumprod()          # 累计费率磨损
        ret["cum_turnover"] = ret["turnover"].cumsum()                    # 累计换手（单边）

        # ======================== 分年度/总体统计 ========================
        def calc_summary(ret_series):

            """ 输入日收益序列，输出年度+总体的收益/波动/回撤/夏普/卡玛 """

            df = pd.DataFrame({"ret": ret_series})
            df["year"] = df.index.map(lambda x: x[0:4])
            years = sorted(set(df["year"]))
            summary = pd.DataFrame(index=years, columns=["收益", "波动", "回撤", "夏普", "卡玛"])

            for year in years:
                ret_year = df[df["year"] == year]["ret"]
                yr_ret = (ret_year + 1.0).prod() - 1.0
                yr_vol = ret_year.std() * np.sqrt(self.period_num)
                nav = (ret_year + 1.0).cumprod()
                yr_mdd = (nav / nav.expanding().max() - 1.0).min()
                summary.loc[year, "收益"] = yr_ret
                summary.loc[year, "波动"] = yr_vol
                summary.loc[year, "回撤"] = yr_mdd
                summary.loc[year, "夏普"] = yr_ret / yr_vol
                summary.loc[year, "卡玛"] = - yr_ret / yr_mdd

            an_ret = (ret_series + 1.0).prod() ** (self.period_num / len(ret_series)) - 1
            an_vol = ret_series.std() * np.sqrt(self.period_num)
            nav_all = (ret_series + 1.0).cumprod()
            mdd = (nav_all / nav_all.expanding().max() - 1.0).min()
            summary.loc["summary", "收益"] = an_ret
            summary.loc["summary", "波动"] = an_vol
            summary.loc["summary", "回撤"] = mdd
            summary.loc["summary", "夏普"] = an_ret / an_vol
            summary.loc["summary", "卡玛"] = - an_ret / mdd
            return summary

        # ======================== 费前表（不含费率） ========================
        # 只含回测逻辑收益，无任何成本列，便于观察策略"原始"表现
        ret["year"] = ret.index.map(lambda x: x[0:4])
        gross = ret.drop(columns=["turnover", "cost", "cost_drag", "cum_turnover"]).copy()
        gross["ret"] = gross["ret_gross"]   # 恢复为费前收益
        gross = gross.drop(columns=["ret_gross"])
        gross["nav"] = (gross["ret"] + 1.0).cumprod()
        gross["max_dd"] = gross["nav"] / gross["nav"].expanding().max() - 1.0
        gross["cumret"] = (gross["ret"] + 1).cumprod() - 1.0
        summary_gross = calc_summary(gross["ret"])

        # ======================== 费后表（含费率） ========================
        # 在费前明细基础上加入：每日换手率、当日扣费、累计费率磨损、累计换手、费前收益对照
        net = ret.copy()
        net["nav"] = (net["ret"] + 1.0).cumprod()
        net["max_dd"] = net["nav"] / net["nav"].expanding().max() - 1.0
        net["cumret"] = (net["ret"] + 1).cumprod() - 1.0
        net["cumret_gross"] = (net["ret_gross"] + 1).cumprod() - 1.0
        # 列顺序：收益明细 -> 成本相关 -> 累计净值 -> 持仓信息与权重
        front_cols = [c for c in ["ret1", "ret2"] if c in net.columns]
        cost_cols = ["ret_gross", "turnover", "cost", "cost_drag", "cum_turnover"]
        nav_cols = ["ret", "cumret", "cumret_gross", "nav", "max_dd"]
        rest_cols = [c for c in net.columns if c not in set(front_cols + cost_cols + nav_cols)]
        net = net[front_cols + cost_cols + nav_cols + rest_cols]
        summary_net = calc_summary(net["ret"])

        # 成本统计块（费后 sheet 右侧）：百分数与数值分两块，避免数字格式冲突
        an_gross = summary_gross.loc["summary", "收益"]
        an_net = summary_net.loc["summary", "收益"]
        n_years = len(net) / self.period_num
        cost_stats_pct = pd.DataFrame(index=["单边费率", "累计费用磨损", "费前年化", "费后年化"],
                                      columns=["值"])
        cost_stats_pct.loc["单边费率", "值"] = self.cost_rate
        cost_stats_pct.loc["累计费用磨损", "值"] = net["cost_drag"].iloc[-1]
        cost_stats_pct.loc["费前年化", "值"] = an_gross
        cost_stats_pct.loc["费后年化", "值"] = an_net
        cost_stats_num = pd.DataFrame(index=["年化成本侵蚀(pct)", "总换手(单边,倍)", "年均换手(单边,倍)",
                                             "涨跌停顺延次数"],
                                      columns=["值"])
        cost_stats_num.loc["年化成本侵蚀(pct)", "值"] = (an_gross - an_net) * 100
        cost_stats_num.loc["总换手(单边,倍)", "值"] = net["turnover"].sum()
        cost_stats_num.loc["年均换手(单边,倍)", "值"] = net["turnover"].sum() / n_years
        cost_stats_num.loc["涨跌停顺延次数", "值"] = n_limit_delay[0]

        print("费前年化 %.2f%% -> 费后年化 %.2f%%，年化成本侵蚀 %.2f pct" % (an_gross * 100, an_net * 100, (an_gross - an_net) * 100))
        print("总换手 %.1f 倍（单边），累计费用磨损 %.2f%%" % (net["turnover"].sum(), net["cost_drag"].iloc[-1] * 100))
        print("涨跌停导致成交顺延次数: %d" % n_limit_delay[0])

        # ======================== 写入 Excel：两个 sheet ========================
        filename = os.path.join(self.data_path, 'backtest%s_%s.xlsx' % (self.hold_num, self.sheet_name))
        excel = WriteExcel(filename)

        # sheet1：不含费率
        ws_gross = excel.add_worksheet("backtest_gross")
        num_format_pd = pd.DataFrame([], columns=gross.columns, index=['format'])
        num_format_pd.loc['format', :] = '0.0%'
        excel.write_pandas(gross, ws_gross, begin_row_number=0, begin_col_number=1,
                           num_format_pd=num_format_pd, color="gray", fillna=True)
        col_number = 3 + len(gross.columns)
        excel.write_pandas(summary_gross, ws_gross, begin_row_number=0, begin_col_number=col_number,
                           num_format_pd=num_format_pd, color="gray", fillna=True)

        # sheet2：含费率（每日换手/扣费/磨损 + 成本统计块）
        ws_net = excel.add_worksheet("backtest_net")
        num_format_pd = pd.DataFrame([], columns=net.columns, index=['format'])
        num_format_pd.loc['format', :] = '0.0%'
        excel.write_pandas(net, ws_net, begin_row_number=0, begin_col_number=1,
                           num_format_pd=num_format_pd, color="gray", fillna=True)
        col_number = 3 + len(net.columns)
        excel.write_pandas(summary_net, ws_net, begin_row_number=0, begin_col_number=col_number,
                           num_format_pd=num_format_pd, color="gray", fillna=True)
        # 成本统计块：百分数块（与 summary 间隔2列）
        num_format_pd2 = pd.DataFrame([], columns=cost_stats_pct.columns, index=['format'])
        num_format_pd2.loc['format', :] = '0.00%'
        col_number2 = col_number + 8
        excel.write_pandas(cost_stats_pct, ws_net, begin_row_number=0, begin_col_number=col_number2,
                           num_format_pd=num_format_pd2, color="gray", fillna=True)
        # 数值块（百分数块下方）
        num_format_pd3 = pd.DataFrame([], columns=cost_stats_num.columns, index=['format'])
        num_format_pd3.loc['format', :] = '0.00'
        excel.write_pandas(cost_stats_num, ws_net, begin_row_number=len(cost_stats_pct) + 3, begin_col_number=col_number2,
                           num_format_pd=num_format_pd3, color="gray", fillna=True)

        excel.close()

    # ==================================================================
    # 每日信号输出（供调度器 cal_data_daily 调用）
    # ==================================================================

    def run_daily(self, update=True, hold_num=2, cost_rate=0.0005):
        """输出两个资产池（etf/index）当日最新信号与当前应持仓标的。

        参数
        ----------
        update : bool
            是否先通过 WindPy 更新价格（需要 Wind 终端）
        hold_num : int
            持仓数（1 或 2，与 backtest 一致）
        cost_rate : float
            单边费率（供回测口径参考，此处仅输出信号与持仓）

        返回
        -------
        字典，值为 DataFrame
            {sheet_name: 当日信号明细DataFrame}
            明细列：动量得分降序的全部标的 + 名称 + 是否在持仓中
        """

        output = {}
        for sheet in ["etf", "index"]:
            self.get_info(sheet_name=sheet, hold_num=hold_num, period="D")
            if update:
                self.update_data()
            mom_index = self.calc_signal()

            # 最新信号：动量得分降序，标记是否入选持仓（得分>0 的前 hold_num 个）
            last_date = mom_index.index[-1]
            last = mom_index.iloc[-1].dropna().sort_values(ascending=False)
            positive = last[last > 0]
            hold_codes = list(positive.index[:hold_num])

            rows = []
            for code, score in last.items():
                name = self.etf_name.get(code, code)
                rows.append({
                    "标的": code,
                    "名称": name,
                    "动量得分": round(float(score), 4),
                    "是否持仓": "是" if code in hold_codes else "",
                })
            df = pd.DataFrame(rows)
            df.insert(0, "日期", last_date)
            df.insert(1, "资产池", "ETF池" if sheet == "etf" else "Index池")
            df = df.set_index("日期")
            output[sheet] = df

            print("\n[双动量-%s] 信号日期: %s | 当前持仓 %d 个: %s" % (
                sheet, last_date, len(hold_codes),
                "、".join(self.etf_name.get(c, c) for c in hold_codes) or "现金"))
        return output


if __name__ == '__main__':

    # 屏蔽 pandas 警告
    import warnings
    warnings.filterwarnings('ignore')
    import pandas as pd
    pd.options.mode.chained_assignment = None  # 专门关闭 SettingWithCopyWarning

    """ 参数 """
    self = DualMomentum()

    # ETF 池与 Index 池：日频调仓，单边成本 0.05%
    for sheet in ["etf", "index"]:
        self.get_info(hold_num=2, sheet_name=sheet, period="D")
        self.update_data()
        self.calc_signal()
        self.backtest(cost_rate=0.0005)
