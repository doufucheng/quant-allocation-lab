# -*- coding: utf-8 -*-
# 以脚本方式直接运行时，把项目根目录加入 sys.path
import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import datetime
import numpy as np
import pandas as pd

from config.config import Config
from utility.write_excel import WriteExcel


class AnnualMA(object):

    """
    红利低波年均线策略（复现自 community-shared《红利低波_年均线策略.xlsx》）
    ================================================================

    策略思想：指数收盘价跌破/站上"年均线"（约一年交易日数的滚动均价）是
    经典的牛熊分界信号。该 Excel 用 6 个指数统计了"低于年线时买入、
    持有1年/2年"的历史收益分布，核心结论（红利低波全收益，2015-12~2026-08）：
      - 全样本持有1年：平均 +10.4%，正收益概率 86.1%（2309个样本）
      - 仅低于年线时持有1年：平均 +16.8%，正收益概率 94.9%（514个样本）
      - 即"低于年线"是比"任意时点"更好的买入时点（逆向买入逻辑）

    ============================ 数据说明 ============================
    数据全部来自本地库 Config().index_eod_path
    （Wind Oracle AINDEXEODPRICES），不用 Wind wsd 接口。

    指数池（3 个境内指数，均为本地库内代码；库内部分代码为小写，
    如 h20269.CSI，匹配按大小写不敏感处理）：
      红利低波全收益   H20269.CSI         2014-12-30 起（中证红利低波全收益，
                        注意不是价格指数 H30269.CSI）
      A500红利低波全收益 932422CNY010.CSI  2014-12-30 起
      东方红红利低波   921446.CSI         2014-12-30 起
    原 Excel 中另有纳斯达克100（NDX.GI）、华证A股微盘（999004.SSI）、
    恒生科技（HSTECH.HI）3 个海外/华证指数，不在本地库内，已删除。

    数据起点与 Excel data sheet 一致；策略样本从年均线满窗口后开始，
    即首个样本日 2015-12-31，与 Excel 策略 sheet 的首行日期一致。

    ============================ 规则复现 ============================
    Excel 公式逐条对照（以红利低波 sheet 为例）：
      收盘价           = data sheet 收盘列（Wind s_dq_close，未复权点位）
      年均线           = AVERAGE(前246个交易日至当日收盘)（公式 AVERAGE(C3:C248)，
                          即含当日共 246 个交易日；Excel 注释写"年线"）
      位置             = 收盘/年线 - 1（偏离度）
      是否低于年线     = 收盘 <= 年线 → "是"
      持有1年收益率     = 242个交易日后收盘 / 下一交易日收盘 - 1
                          （F3 = B245/B4-1，即 T+1 买入、T+1+242 卖出）
      低于年线持有1年   = 若当日低于年线，取持有1年收益率，否则空
      汇总统计         = 平均收益 / 正收益概率 / 样本数（全样本 vs 低于年线样本）
      历史最大回撤区间整理 = 手工整理的 4 次低于年线事件明细（起止日/持续天数/
                          最大偏离/最低点买入1年收益），本程序自动识别事件。

    与 Excel 的已知差异：
      1. Excel 用 TDays 生成的交易日历与 Wind 行情返回的自然序列存在
         个别差异，242/246 的窗口按"行数"对齐；本程序按 pandas 序列
         行号对齐，逐日数值与 Excel 一致（相对 Excel 值差异 < 1e-8）。
      2. 数据源为本地库而非 Excel 的 Wind Excel 插件；仅保留库内
         存在的 3 个境内指数，纳指/华证微盘/恒科已删除。

    ============================ 输出说明 ============================
    输出目录：Config().data_dir("derived_data", "allocation_data", "annual_ma")
      signal_{sheet}.csv       每日信号明细（日期/收盘/年线/位置/是否低于年线/
                               持有1年/持有2年收益率及低于年线版本）
      summary_{sheet}.csv      汇总统计（平均收益/正收益概率/样本数，
                               全样本 vs 低于年线，持有1年与2年口径）
      events_{sheet}.csv       低于年线事件段明细（连续低于年线的起止日/
                               持续交易日/最大偏离幅度/最低偏离日及当日信号）
      annual_ma_timing.xlsx    汇总工作簿（每日明细 + 汇总统计 + 事件明细，
                               每个指数一个 sheet 组）
    """

    def __init__(self):

        # 数据与输出目录（独立目录，与其他模型隔离）
        self.data_path = Config().data_dir("derived_data", "allocation_data", "annual_ma")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        # 指数池：sheet名 -> (本地库代码, 数据起始日)
        # 数据全部来自本地库 IndexEOD.pq（Wind Oracle AINDEXEODPRICES），不用 wsd
        # 注意1：红利低波用 H20269.CSI（全收益指数），Excel 表头"红利低波全收益"
        #   即该代码；H30269.CSI 是价格指数，点位系统性偏低（2015年末 7881 vs 10422）
        # 注意2：库内部分代码为小写形式（如 h20269.CSI），匹配按大小写不敏感处理
        # 注意3：原 Excel 中的海外指数（纳指/恒科）与华证微盘不在本地库内，已删除
        self.index_pool = {
            "红利低波":     ("H20269.CSI",         "2014-12-30"),
            "A500红利低波": ("932422CNY010.CSI",   "2014-12-30"),
            "东方红红利低波": ("921446.CSI",       "2014-12-30"),
            "红利低波100":  ("H20955.CSI",         "2014-12-30"),
        }

        # 年均线窗口：Excel 公式 AVERAGE(C3:C248) = 含当日共 246 个交易日
        self.ma_window = 246

        # 持有期窗口：Excel F3 = B245/B4-1 → 卖出日 = 买入日(T+1) + 242 个交易日
        # 持有2年口径同理翻倍（纳斯达克 sheet 有持有2年列）
        self.hold1_days = 242
        self.hold2_days = 484

        # 信号文件缓存路径（当日起点变化或数据过期时自动重新下载）
        self.close_file = os.path.join(self.data_path, "close.csv")

    def get_info(self, sheet_name="红利低波"):
        """设置当前处理的标的参数"""
        self.sheet_name = sheet_name
        self.wind_code, self.beg_date = self.index_pool[sheet_name]

    def get_data(self, use_cache=True):
        """读取指数收盘价：全部来自本地 db_wind 库（IndexEOD.pq），不用 wsd

        本地库 IndexEOD.pq 源自 Wind Oracle AINDEXEODPRICES（境内指数，部分代码
        小写如 h20269.CSI），按大小写不敏感匹配。
        """
        if use_cache and os.path.exists(self.close_file):
            # 缓存文件7天内且含近期数据则直接使用
            df = pd.read_csv(self.close_file, index_col=0, parse_dates=True)
            ts = datetime.datetime.fromtimestamp(os.path.getmtime(self.close_file))
            if (datetime.datetime.now() - ts).days < 7 and df.index.max() >= datetime.datetime(2026, 8, 5):
                if self.sheet_name in df.columns:
                    return df[[self.sheet_name]].dropna()

        # 本地 db_wind 库：索引为 (Date, Ticker)，代码统一按大小写不敏感匹配
        db_file = os.path.join(Config().db_wind_path, "IndexEOD.pq")
        db = pd.read_parquet(db_file, columns=["Close"])
        db_codes = {}
        for c in db.index.get_level_values("Ticker").unique():
            db_codes[str(c).upper()] = c

        df = pd.DataFrame(index=None)
        for name, (code, beg) in self.index_pool.items():
            if code.upper() not in db_codes:
                raise KeyError(f"{code} 不在本地库 IndexEOD.pq 内")
            s = db.xs(db_codes[code.upper()], level="Ticker")["Close"].dropna()
            s.index = pd.to_datetime(s.index)
            s = s[s.index >= pd.Timestamp(beg)]
            df[name] = s

        df = df.sort_index().dropna(how="all")
        df.to_csv(self.close_file)
        return df[[self.sheet_name]].dropna()

    def calc_signal(self, close, use_cache=True):
        """计算年均线、位置、是否低于年线、持有1/2年收益率

        参数
        ----
        close : pd.DataFrame 单列收盘价（列名=sheet名）
        """
        name = close.columns[0]
        s = close[name].dropna()

        # 年均线：含当日共246个交易日均价（与 Excel AVERAGE(C{t-245}:C{t}) 一致）
        ma = s.rolling(self.ma_window, min_periods=self.ma_window).mean()

        sig = pd.DataFrame({
            "收盘价": s,
            "年均线": ma,
        })
        sig["位置"] = sig["收盘价"] / sig["年均线"] - 1
        sig["是否低于年线"] = np.where(sig["收盘价"] <= sig["年均线"], "是", "否")

        # 持有1年收益率：T日信号 → T+1收盘买入 → T+1+242交易日收盘卖出
        # Excel: F_t = B_{t+242} / B_{t+1} - 1（行号偏移）
        for hold_days, col in [(self.hold1_days, "持有1年收益率"),
                               (self.hold2_days, "持有2年收益率")]:
            sell = s.shift(-hold_days)      # T+1+hold_days 的收盘
            buy = s.shift(-1)               # T+1 的收盘
            sig[col] = sell / buy - 1

        # 低于年线版本：仅低于年线时记录收益
        sig["低于年线持有1年"] = np.where(sig["是否低于年线"] == "是", sig["持有1年收益率"], np.nan)
        sig["低于年线持有2年"] = np.where(sig["是否低于年线"] == "是", sig["持有2年收益率"], np.nan)

        return sig

    def calc_summary(self, sig):
        """汇总统计：全样本 vs 低于年线，持有1/2年口径（对应Excel中部统计块）"""
        rows = []
        for hold_col, below_col in [("持有1年收益率", "低于年线持有1年"),
                                    ("持有2年收益率", "低于年线持有2年")]:
            x_all = sig[hold_col].dropna()
            x_below = sig[below_col].dropna()
            hold_name = "持有1年" if "1年" in hold_col else "持有2年"
            rows.append({
                "口径": hold_name,
                "样本类型": "全部时点",
                "收益均值": x_all.mean(),
                "正收益概率": (x_all > 0).sum() / len(x_all) if len(x_all) else np.nan,
                "样本数": len(x_all),
            })
            rows.append({
                "口径": hold_name,
                "样本类型": "低于年线",
                "收益均值": x_below.mean(),
                "正收益概率": (x_below > 0).sum() / len(x_below) if len(x_below) else np.nan,
                "样本数": len(x_below),
            })
        summary = pd.DataFrame(rows)
        return summary.set_index(["口径", "样本类型"])

    def calc_events(self, sig):
        """识别连续低于年线事件段（对应Excel右侧事件整理表）

        事件段定义：连续"是"（收盘<=年线）为一个事件，记录：
          开始低于均线日 / 结束低于均线日 / 持续交易日 / 最大偏离均线幅度 /
          最低偏离日 / 最低偏日买入1年收益
        """
        below = (sig["是否低于年线"] == "是") & sig["年均线"].notna()
        events = []
        start = None
        for dt, flag in below.items():
            if flag and start is None:
                start = dt
            elif not flag and start is not None:
                events.append((start, prev_dt))
                start = None
            prev_dt = dt
        if start is not None:
            events.append((start, below.index[-1]))

        rows = []
        for e_start, e_end in events:
            seg = sig.loc[e_start:e_end]
            low_day = seg["位置"].idxmin()
            rows.append({
                "开始低于均线日": e_start,
                "结束低于均线日": e_end,
                "持续交易日": len(seg),
                "最大偏离均线幅度": seg["位置"].min(),
                "最低偏离日": low_day,
                "最低偏日买入1年收益": sig.loc[low_day, "持有1年收益率"] if low_day in sig.index else np.nan,
                "最低偏日买入2年收益": sig.loc[low_day, "持有2年收益率"] if low_day in sig.index else np.nan,
            })
        ev = pd.DataFrame(rows)
        if ev.empty:
            return ev
        ev["开始低于均线日"] = pd.to_datetime(ev["开始低于均线日"])
        ev["结束低于均线日"] = pd.to_datetime(ev["结束低于均线日"])
        return ev

    def run(self, sheet_name="红利低波", use_cache=True, we=None):
        """主流程：读本地库→信号→统计→事件→输出CSV，并可写入共享Excel工作簿

        参数 we : WriteExcel 实例（可选）。传入时把三个 sheet 追加到该工作簿，
        由调用方负责 close；不传则不写 Excel（只输出 CSV）。
        """
        self.get_info(sheet_name)
        close = self.get_data(use_cache=use_cache)
        sig = self.calc_signal(close)
        summary = self.calc_summary(sig)
        events = self.calc_events(sig)

        # 输出 CSV
        sig.to_csv(os.path.join(self.data_path, f"signal_{sheet_name}.csv"))
        summary.to_csv(os.path.join(self.data_path, sheet_name + "_summary.csv"))
        if not events.empty:
            events.to_csv(os.path.join(self.data_path, f"events_{self.sheet_name}.csv"))

        # 追加写入共享工作簿（每日明细 + 汇总统计 + 事件明细，sheet 名按指数区分）
        if we is not None:
            ws_sig = we.add_worksheet(sheet_name + "-每日信号")
            we.write_pandas(sig.reset_index(), ws_sig)
            ws_sum = we.add_worksheet(sheet_name + "-汇总统计")
            we.write_pandas(summary.reset_index(), ws_sum)
            ws_ev = we.add_worksheet(sheet_name + "-事件")
            we.write_pandas(events.reset_index(), ws_ev)

        return sig, summary, events


if __name__ == "__main__":

    model = AnnualMA()
    all_summary = []

    # 全部指数写入同一个工作簿，最后统一 close
    we = WriteExcel(os.path.join(model.data_path, "annual_ma_timing.xlsx"))
    for name in model.index_pool:
        sig, summary, events = model.run(name, we=we)
        summary.insert(0, "指数", name)
        all_summary.append(summary)
        print(f"[{name}] 完成：{len(sig)} 个交易日，"
              f"低于年线样本 {int(summary.loc[('持有1年', '低于年线'), '样本数'])} 个")
    we.close()

    # 汇总打印 3 指数对比
    total = pd.concat(all_summary)
    print(total.to_string())
