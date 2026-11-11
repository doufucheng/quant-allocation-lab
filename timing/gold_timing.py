# -*- coding: utf-8 -*-
# 以脚本方式直接运行时，把项目根目录加入 sys.path
import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from config.config import Config


class GoldTiming(object):

    """
    黄金择时：三主体框架·国内可得版（2026-09-19）
    ================================================================

    一、来源
    ----------------------------------------------------------------
    平安证券《多资产量化配置系列之二：基于投资者持仓的黄金分析框架与
    择时策略》(2026-09-14)。原框架：金价由【ETF 投资者 / 央行 / 投机者】
    三主体持仓 + 美元、美债实际利率两个国际锚决定。
    回测金价 = AU9999.SGE 上金所现货（本地库无此表，用户 2026-09-19 授权
     网络源：东方财富公开 K 线接口 secid=118.AU9999，2004 年起，免费无 key；
     原始数据统一缓存在 E:\Data\db_data\db_net\au9999_sge.csv——网络下载的
     原始数据一律放 db_net 目录（用户定规），超 7 日自动刷新，
     `python gold_timing.py refresh` 可强制刷新）；信号指标仍由本地 AU.SHF
     （与现货日收益相关 0.946、基差约 0.2%）。仍缺：央行购金、CFTC 持仓、
     美元指数、美债利率、伦敦金现（登记 config/doc/db_data_missing.md §3.10），
     故为两主体降级可得版（央行主体缺失）。

    二、指标清单（= gold_panel.csv 的列；全部 T 日收盘可得，无前视）
    ----------------------------------------------------------------
    A. 金价走势 —— FutEOD.pq 的 AU.SHF（上期所黄金指数，2008-01 起）
       A1 au_mom250        金价 250 日动量（跨年趋势）
       A2 au_mom60         金价 60 日动量（季度趋势）
       A3 au_ma_bias       金价 / 250 日均线 - 1（趋势位置）
       A4 au_vol20         20 日收益年化波动
       A5 au_dd250         金价距 250 日高点回撤
    B. ETF 主体（国内版）—— FundDesc.pq 筛出 7 只商品型黄金 ETF，
       FundShares.pq 每日份额合计（2013-07 起，当前约 273 亿份）
       B1 etf_shares       黄金 ETF 份额合计（配置资金存量）
       B2 etf_share_chg60  份额 60 日净增速（配置资金流入速度）
       B3 etf_share_z250   份额增速 250 日 z 值（流入是否极端）
    C. 投机主体（国内代理，无 CFTC）—— FutEOD.pq 的 AU.SHF 总持仓 oi
       C1 au_oi_z250       总持仓 250 日 z 值（持仓拥挤度）
       C2 au_voi           成交量 / 总持仓（投机活跃度）
    D. 比价指标 —— AG.SHF 白银（2012-05 起）、CU.SHF 铜（1995 起）
       D1 gsr_z250         金银比 250 日 z 值（金相对银贵/便宜）
       D2 gcr_z250         金铜比 250 日 z 值（金相对铜贵/便宜）
    E. 国际锚 —— api_wind（2026-09-21 补，见 db_data_missing.md §3.10/§3.11）
       E1 usdx_z250        美元指数(USDX.FX) 250 日 z 值（美元强弱锚）
       E2 us10y_z250       美债 10Y 名义收益率(G0000888) 250 日 z 值
       E3 us10y_mom250     美债 10Y 收益率 250 日动量（利率趋势锚）
       注：TIPS 实际利率仍缺，用名义 10Y 代理（通胀预期缺位，口径差异已知）；
       美元/美债 T 日值在北京 T 日 05:00 前可得，A 股 T 日收盘信号使用无前视。
    回测价格：ETFEOD.pq 的 518880.SH（华安黄金 ETF）后复权收盘。

    三、信号（12 个；T 收盘生成、T+1 收盘成交、单边成本 5bp；
       各窗口为首次默认值，未做参数扫描）
    ----------------------------------------------------------------
       S1 sig_tsmom      趋势：A1 金价 250 日动量>0 持有
       S2 sig_etf_flow   资金：B2 份额 60 日净增(chg60>0) 持有
       S3 sig_gsr        比价：D1 金银比 z<0（金相对便宜）持有
       S4 sig_oi_crowd   拥挤过滤：C1 持仓 z≥2（极端拥挤）空仓，否则持有
       S5 sig_combo      三票多数决 = S1+S2+S3 中 ≥2 票同意持有。
                         注意：只合成 S1/S2/S3 三个"方向信号"；
                         S4 是风险过滤器（几乎总持有），不参与投票
       S6 sig_ma_band    年线带：A3 金价跌破年线 3% 以上(ma_bias<-3%)空仓
       S7 sig_mom_dd     趋势+回撤双条件：A1 动量>0 且 A5 回撤浅于 -15%
       S8 sig_etf_ext    份额极端流入退出：B3 份额流入 z≥2（散户蜂拥）
                         时空仓，其余时间持有（低频攻守型）
       S9 sig_major_crowd S5 多数决 ∧ S4 拥挤过滤（双重过滤版）
       S10 sig_usdx      美元锚：E1 美元指数 250 日 z<0（美元偏弱）持有
       S11 sig_us10y     利率锚：E3 美债 10Y 收益率 250 日动量<0（利率
                         下行）持有
       S12 sig_anchor    双锚版：S10 ∧ S11（美元弱且利率下行才持有）

    四、回测结果（单边 5bp 调仓成本已计入；各信号按各自数据就绪日
       独立起跑，样本起见表；au 类指标 2009 年初就绪）
    ----------------------------------------------------------------
    段 1：AU9999.SGE 现货金，2009-01 ~ 2026-09（17.1 年，含 2011-2015
           完整熊市，基准区间收益 -41.0%/回撤 -44.9% 的主战场）
        策略              年化    Sharpe  最大回撤   卡玛   调仓
        买入持有(基准)    10.2%    0.64    -44.9%   0.227    0
        S1 趋势            8.6%    0.62    -33.4%   0.257  101  ← 卡玛唯一超基准
        S3 金银比(2013起)  1.4%    0.13    -30.1%   0.047  105
        S4 持仓拥挤        9.2%    0.62    -45.7%   0.202   75
        S6 年线带          7.6%    0.53    -42.1%   0.179  115
        S7 趋势+回撤       8.1%    0.64    -37.8%   0.214  116
        熊市段(2011-09~2015-12)：基准 -41.0%/-44.9% → S1 -26.3%/-29.3%
        （S1 卡玛改善全部来自熊市躲跌；2009-2026 仅一个完整牛熊循环；
          与 AU.SHF 期货指数口径结论一致，现货口径下略强）

    段 2：518880.SH 黄金 ETF（可交易口径），2013-07 ~ 2026-09（12.8 年）
       策略              年化    Sharpe  最大回撤   卡玛   调仓
       买入持有(基准)    10.2%    0.67    -30.5%   0.334    0
       S8 份额极端退出   12.8%    0.87    -30.5%   0.420   31  ← 收益/夏普/卡玛
                                                              三超基准且最低频
       S7 趋势+回撤       8.3%    0.67    -22.8%   0.363   74  ← 回撤少 8pp，
                                                              卡玛超基准(防守型)
       S1/S2/S3/S4/S5/S6/S9 均不达标（收益与卡玛同降或大幅跑输）

       结论：达标信号 2 个——S8（攻守兼备，年调仓约 2.7 次）与
       S7（防守型，回撤 -30.5%→-22.8%）；S1 仅在含完整熊市的长样本上
       改善卡玛。S8 样本仅 11.5 年且只经历一次半熊市，z≥2 阈值为首次
       设定未扫描，属"继续验证"而非"已采用"。

    段 3：国际锚信号（2026-09-22 新增，api_wind 数据，详见 §3.11）
       S10 美元锚 / S11 利率锚 / S12 双锚（美元弱且利率下行才持有）：
       全周期段（2009-2026，基准年化 10.0%/回撤 -44.9%）：
       S10 年化 6.6%/回撤 -21.9%（卡玛 0.301）；S12 年化 5.2%/回撤 -20.6%
       熊市段（2011-09~2015-12，基准 -41.0%/-44.9%）：
       S10 +0.5%/-17.7%；S12 +3.4%/-6.9% —— 双锚几乎完全躲过熊市
       ETF 段（2013-2026，基准 10.0%/-30.5%）：
       S10 6.4%/-20.6%；S12 3.4%/-20.5%
       定位：强防守型（与 S1 同类），牛市段空仓偏多导致年化跑输；
       不作主策略，可作组合防守组件或与 S8 叠加的过滤器。
       注意：名义 10Y 代理实际利率（TIPS 缺），2021 后通胀高企期
       名义利率与实际利率走势背离，锚信号在该段解释力打折。

    五、输出（E:\\Data\\allocation_data\\gold_timing\\）
    ----------------------------------------------------------------
       gold_panel.csv            每日数据面板（上述 12 个指标）
       gold_summary_au.csv       全周期段绩效汇总（即段 1 表格数字）
       gold_summary_etf.csv      可交易段绩效汇总（即段 2 表格数字）
       gold_backtest_au.csv      全周期段信号 + 各策略净值明细
       gold_backtest_etf.csv     可交易段信号 + 各策略净值明细
       gold_yearly_au/etf.csv    逐年收益

    口径备注：份额合计已剔除"黄金股 ETF"（跟踪矿业股票指数，非金价）与
    联接基金（feeder，防重复计数）；AU.SHF 为全合约聚合指数，oi 即全市场
    总持仓；金银比 = au收盘/ag收盘（无量纲，只用相对水平）；无风险利率 0。
    """

    def __init__(self):

        self.db_wind = str(Config().db_wind_path)
        self.out_dir = Config().data_dir("derived_data", "allocation_data", "gold_timing")
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # 回测参数（预注册，未扫描）
        self.target = "518880.SH"          # 可交易回测标的：华安黄金 ETF
        self.cost_bp = 5.0                 # 单边成本 5bp
        self.warmup = 260                  # 预热：250 日 z-score 窗口 + 缓冲

        # AU9999.SGE 现货金价格缓存（用户 2026-09-19 授权网络源；
        # 来源=东方财富公开 K 线接口 secid=118.AU9999，即上金所 Au99.99 现货）。
        # 网络下载的原始数据统一放 E:\Data\db_data\db_net（用户 2026-09-19 定规）
        self.sge_csv = os.path.join(str(Config().db_data_path), "db_net", "au9999_sge.csv")

    # ------------------------------------------------------------------
    # 读取工具（db_wind parquet 统一读法：忽略 pandas 索引元数据）
    # ------------------------------------------------------------------

    def _rd(self, name, cols):
        df = pq.read_table(os.path.join(self.db_wind, name), columns=cols)\
            .to_pandas(ignore_metadata=True)
        if any(n is not None for n in df.index.names) and \
                df.index.name not in (None, "__index_level_0__"):
            df = df.reset_index()
        return df

    def update_sge_price(self, force=False):
        """AU9999.SGE 现货日价：本地缓存优先，无缓存（或 force）时从东方财富
        公开 K 线接口下载（secid=118.AU9999，上金所 Au99.99，2004 年起，免费
        无 key）。返回 Date 索引的 Close 序列（元/克）。"""

        need_dl = force or not os.path.exists(self.sge_csv)
        if not need_dl:
            px = pd.read_csv(self.sge_csv, index_col=0, parse_dates=True)["Close"]
            # 缓存超过 7 个自然日则刷新（非交易日接口返回不变，安全幂等）
            if (pd.Timestamp.today() - px.index.max()).days <= 7:
                return px.astype(float)

        url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
               "?secid=118.AU9999&fields1=f1,f2,f3,f4,f5,f6"
               "&fields2=f51,f52,f53,f54,f55,f56,f57"
               "&klt=101&fqt=0&beg=19900101&end=20500101")
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        kl = (r.json().get("data") or {}).get("klines") or []
        if not kl:
            raise RuntimeError(f"AU9999 下载失败: HTTP {r.status_code}, klines 为空")
        rows = [x.split(",") for x in kl]     # date,open,close,high,low,vol,amt
        df = pd.DataFrame(rows, columns=["Date", "Open", "Close", "High", "Low",
                                         "Volume", "Amount"])
        df["Date"] = pd.to_datetime(df["Date"])
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.drop_duplicates(subset="Date", keep="last")\
            .sort_values("Date").set_index("Date")
        df.to_csv(self.sge_csv, encoding="utf-8-sig")
        print(f"AU9999.SGE 现货价: {df.index.min().date()} ~ {df.index.max().date()} "
              f"rows={len(df)} -> {self.sge_csv}")
        return df["Close"].astype(float)

    # ------------------------------------------------------------------
    # 数据构建
    # ------------------------------------------------------------------

    def build_panel(self):
        """构建黄金择时面板：走势 + 两主体 + 比价。"""

        # --- 1) 期货指数：白银 / 铜（比价）+ AU.SHF（仅取持仓与成交） ---
        fut = self._rd("FutEOD.pq", ["Date", "Ticker", "Close", "Volume", "oi"])
        px = {}
        for code in ["AU.SHF", "AG.SHF", "CU.SHF"]:
            g = fut[fut["Ticker"] == code].sort_values("Date")
            g = g.drop_duplicates(subset="Date", keep="last").set_index("Date")
            px[code] = g[["Close", "Volume", "oi"]].astype(float)
            pre = code.split(".")[0].lower()
            close_col = "au_shf_close" if code == "AU.SHF" else f"{pre}_close"
            px[code].columns = [close_col, f"{pre}_volume", f"{pre}_oi"]
        panel = px["AU.SHF"].join(px["AG.SHF"][["ag_close"]], how="left")\
            .join(px["CU.SHF"][["cu_close"]], how="left")

        # --- 1b) AU9999.SGE 现货金（2004 年起，网络源 db_net）：走势指标基准 ---
        # 走势类指标（动量/均线/波动/回撤）用现货价计算，样本从 2005 年起就绪；
        # AU.SHF 期货指数仅提供持仓类指标（oi/volume，2008 年起）。
        sge = self.update_sge_price().rename("au_close")
        panel = panel.join(sge, how="outer").sort_index()
        panel["au_close"] = panel["au_close"].ffill()   # 现货停报日用前值（不用于成交价，仅指标）

        # --- 2) ETF 主体：场内金价商品型黄金 ETF 份额合计 ---
        desc = self._rd("FundDesc.pq", ["Ticker", "Name", "FirstIType", "IsETF", "IsFeeder"])
        is_gold_etf = (
            desc["Name"].str.contains("黄金", na=False)
            & ~desc["Name"].str.contains("黄金股|联接", na=False)   # 剔除矿业股票 ETF 与 feeder
            & (desc["FirstIType"] == "商品型")                        # 跟踪金价的商品型
            & (desc["IsETF"] == True)
            & (desc["IsFeeder"] != True)
        )
        gold_codes = sorted(set(desc.loc[is_gold_etf, "Ticker"].str[:6]))
        print("黄金 ETF 名单(代码前6位):", gold_codes)

        fsh = self._rd("FundShares.pq", ["Date", "Ticker", "Shares"])
        fsh = fsh[fsh["Ticker"].str[:6].isin(gold_codes)].dropna(subset=["Shares"])
        etf_shares = fsh.groupby("Date")["Shares"].sum().astype(float)\
            .rename("etf_shares").sort_index()
        # 面板按日重采样对齐：份额非交易日（申赎清单日）向前无效——只在交易日索引上
        # 用最近可得份额（披露频率为日频清单，滞后一日可得；保守按 T 日值参与 T 日信号）
        panel = panel.join(etf_shares, how="left")

        # --- 3) 走势指标 ---
        c = panel["au_close"]
        panel["au_mom20"] = c / c.shift(20) - 1
        panel["au_mom60"] = c / c.shift(60) - 1
        panel["au_mom120"] = c / c.shift(120) - 1
        panel["au_mom250"] = c / c.shift(250) - 1
        panel["au_ma60"] = c.rolling(60).mean()
        panel["au_ma250"] = c.rolling(250).mean()
        panel["au_ma_bias"] = c / panel["au_ma250"] - 1
        ret = c.pct_change(fill_method=None)
        panel["au_vol20"] = ret.rolling(20).std() * np.sqrt(250)
        panel["au_dd250"] = c / c.rolling(250).max() - 1

        # --- 4) ETF 主体指标 ---
        panel["etf_share_chg60"] = panel["etf_shares"] / panel["etf_shares"].shift(60) - 1
        mu = panel["etf_share_chg60"].rolling(250).mean()
        sd = panel["etf_share_chg60"].rolling(250).std()
        panel["etf_share_z250"] = (panel["etf_share_chg60"] - mu) / sd

        # --- 5) 投机者主体（国内代理）指标 ---
        oi = panel["au_oi"]
        panel["au_oi_chg20"] = oi / oi.shift(20) - 1
        panel["au_oi_z250"] = (oi - oi.rolling(250).mean()) / oi.rolling(250).std()
        panel["au_voi"] = panel["au_volume"] / oi            # 成交持仓比（投机活跃度）

        # --- 6) 比价指标 ---
        panel["gsr"] = panel["au_close"] / panel["ag_close"]             # 金银比
        panel["gsr_z250"] = (panel["gsr"] - panel["gsr"].rolling(250).mean())\
            / panel["gsr"].rolling(250).std()
        panel["gcr"] = panel["au_close"] / panel["cu_close"]             # 金铜比
        panel["gcr_z250"] = (panel["gcr"] - panel["gcr"].rolling(250).mean())\
            / panel["gcr"].rolling(250).std()

        # --- 7) 国际锚（api_wind，2026-09-21 补；详见 §3.11）---
        # USDX.FX 日频（1990-12 起）；G0000888 美债 10Y 日频（1990-01 起）。
        # reindex+ffill 对齐到本面板（A 股）日历：US 日期 D 的收盘值在北京
        # D 日 05:00 前可得，A 股 D 日收盘信号使用无前视。
        api = Config().api_wind_path
        gi = pq.read_table(os.path.join(str(api), "global_index_eod.pq"),
                           columns=["Date", "Ticker", "Close"]).to_pandas()
        usdx = gi[gi["Ticker"] == "USDX.FX"].set_index("Date")["Close"]\
            .astype(float).sort_index()
        usdx.index = pd.to_datetime(usdx.index.astype(str).str[:10])
        me = pd.read_parquet(os.path.join(str(api), "macro_edb.pq")).set_index("Date")
        us10y = me["G0000888_US10Y_D"].dropna().astype(float).sort_index()
        us10y.index = pd.to_datetime(us10y.index.astype(str).str[:10])
        # 对齐到面板日历（ffill 取 <=面板日的最近 US 值）
        panel["usdx"] = usdx.reindex(panel.index, method="ffill")
        panel["us10y"] = us10y.reindex(panel.index, method="ffill")
        panel["usdx_z250"] = (panel["usdx"] - panel["usdx"].rolling(250).mean())\
            / panel["usdx"].rolling(250).std()
        panel["us10y_z250"] = (panel["us10y"] - panel["us10y"].rolling(250).mean())\
            / panel["us10y"].rolling(250).std()
        panel["us10y_mom250"] = panel["us10y"] / panel["us10y"].shift(250) - 1

        panel.index.name = "Date"
        return panel

    # ------------------------------------------------------------------
    # 信号与回测
    # ------------------------------------------------------------------

    def run(self):
        panel = self.build_panel()
        panel.to_csv(os.path.join(self.out_dir, "gold_panel.csv"),
                     encoding="utf-8-sig")

        # ---- 信号（T 收盘生成；1=持有，0=空仓）----
        df = panel.copy()
        # 全周期信号（金价/持仓/比价指标 2008 年起可得）
        df["sig_tsmom"] = (df["au_mom250"] > 0).astype(int)               # S1 趋势
        df["sig_gsr"] = (df["gsr_z250"] < 0).astype(int)                  # S3 金银比
        df["sig_oi_crowd"] = (df["au_oi_z250"] < 2).astype(int)           # S4 拥挤过滤
        df["sig_ma_band"] = (df["au_ma_bias"] > -0.03).astype(int)        # S6 年线带
        df["sig_mom_dd"] = ((df["au_mom250"] > 0)
                            & (df["au_dd250"] > -0.15)).astype(int)       # S7 趋势+回撤双条件
        # ETF 类信号（份额 2013-07 起；早期 NaN 一律视为"不触发退出/不持有"，仅 etf 段使用）
        df["sig_etf_flow"] = (df["etf_share_chg60"] > 0).astype(int)      # S2 份额流入
        df["sig_combo"] = ((df["sig_tsmom"] + df["sig_etf_flow"]
                            + df["sig_gsr"]) / 3.0 >= 0.5).astype(int)    # S5 三票多数
        df["sig_etf_ext"] = (~(df["etf_share_z250"] > 2)).astype(int)     # S8 份额流入极端退出
        df["sig_major_crowd"] = ((df["sig_combo"] == 1)
                                 & (df["sig_oi_crowd"] == 1)).astype(int)  # S9 多数∧拥挤过滤
        # 国际锚信号（2026-09-22 新增，api_wind 数据；预注册阈值，未扫描）
        df["sig_usdx"] = (df["usdx_z250"] < 0).astype(int)                # S10 美元偏弱
        df["sig_us10y"] = (df["us10y_mom250"] < 0).astype(int)            # S11 利率下行
        df["sig_anchor"] = ((df["sig_usdx"] == 1)
                            & (df["sig_us10y"] == 1)).astype(int)         # S12 双锚

        au_sigs = ["sig_tsmom", "sig_gsr", "sig_oi_crowd", "sig_ma_band",
                   "sig_mom_dd", "sig_usdx", "sig_us10y", "sig_anchor"]
        etf_sigs = au_sigs + ["sig_etf_flow", "sig_combo", "sig_etf_ext",
                              "sig_major_crowd"]

        # 回测价格 2：518880.SH 后复权收盘（可交易版本）
        etf = self._rd("ETFEOD.pq", ["Date", "Ticker", "AdjClose"])
        tgt = etf[etf["Ticker"] == self.target].sort_values("Date")\
            .drop_duplicates(subset="Date", keep="last").set_index("Date")["AdjClose"]\
            .astype(float)

        # 回测价格 1：AU9999.SGE 现货（上金所 Au99.99，东财 118.AU9999，
        # 网络源已授权；信号仍由 AU.SHF 指标计算，两者同源高度相关）
        sge_close = self.update_sge_price()

        targets = {
            "AU9999.SGE现货_全周期": (sge_close, au_sigs, "gold_summary_au.csv",
                                      "gold_backtest_au.csv", "gold_yearly_au.csv", True),
            "518880.SH_2013起可交易": (tgt, etf_sigs, "gold_summary_etf.csv",
                                       "gold_backtest_etf.csv", "gold_yearly_etf.csv", False),
        }

        cost = self.cost_bp / 10000.0
        bear = ("2011-09-01", "2015-12-31")   # 黄金熊市段（择时主战场，仅全周期段报告）
        # 段 1 统一截止：本地持仓数据（au_oi）末日，保证信号共同样本公平；
        # AU9999 现货更新到 09-21，但 oi/gsr 等本地指标止 09-08，混算会引入假空仓
        oi_end = df["au_oi"].dropna().index.max()
        all_summaries = {}
        for tname, (px, sigs, f_sum, f_bt, f_yr, report_bear) in targets.items():
            d = df.join(px.rename("target"), how="inner").dropna(subset=["target"])
            if report_bear:
                d = d.loc[d.index <= oi_end]
            # 每个信号的依赖列（决定该信号自己的可回测起点）
            dep = {"sig_tsmom": "au_mom250", "sig_gsr": "gsr_z250",
                   "sig_oi_crowd": "au_oi_z250", "sig_ma_band": "au_ma_bias",
                   "sig_mom_dd": "au_dd250", "sig_etf_flow": "etf_share_chg60",
                   "sig_combo": "etf_share_chg60", "sig_etf_ext": "etf_share_z250",
                   "sig_major_crowd": "etf_share_z250",
                   "sig_usdx": "usdx_z250", "sig_us10y": "us10y_mom250",
                   "sig_anchor": "us10y_mom250"}
            ret = d["target"].pct_change(fill_method=None).fillna(0.0)

            # 基准：金价类指标（au_*）就绪后全段持有，从 2009 年初起，
            # 覆盖 2011-2015 完整熊市；各信号按各自依赖列就绪日独立起跑
            base_dep = sorted({dep[s] for s in sigs if dep[s].startswith("au_")})
            b_start = d.dropna(subset=sorted(set(base_dep))).index.min()
            bmask = d.index >= b_start
            bench_nav = (1 + ret[bmask]).cumprod()

            res = {"买入持有": bench_nav}
            positions = {}
            starts = {}
            for s in sigs:
                s0 = d.dropna(subset=[dep[s]]).index.min()
                starts[s] = s0
                sm = d.index >= s0
                pos = pd.Series(0.0, index=d.index)
                pos.loc[sm] = d.loc[sm, s].shift(1).fillna(0.0).values  # T+1 生效
                positions[s] = pos
                turnover = pos.diff().abs().fillna(pos.abs())
                strat_ret = pos * ret - turnover * cost   # 调仓成本已计入（单边 5bp）
                res[s] = (1 + strat_ret[sm]).cumprod()
            out = d[["target"] + sigs].copy()
            for k, v in res.items():
                out[f"nav_{k}"] = np.nan
                out.loc[v.index, f"nav_{k}"] = v.values
            out.to_csv(os.path.join(self.out_dir, f_bt), encoding="utf-8-sig")

            def perf(nav, pos=None):
                r = nav.pct_change(fill_method=None).fillna(0.0)
                ann = nav.iloc[-1] ** (250 / max(len(nav), 1)) - 1
                vol = r.std() * np.sqrt(250)
                sharpe = ann / vol if vol > 0 else np.nan
                mdd = (nav / nav.cummax() - 1).min()
                calmar = ann / abs(mdd) if mdd < 0 else np.nan
                n_tr = int((pos.diff().abs() > 1e-12).sum()) if pos is not None else 0
                yrs = len(nav) / 250.0
                return ann, vol, sharpe, mdd, calmar, n_tr, yrs

            cols = ["年化收益", "年化波动", "Sharpe", "最大回撤", "卡玛",
                    "调仓次数", "年数"]
            rows = [{"策略": "买入持有(基准)", "样本起": str(b_start.date()), **dict(
                zip(cols, perf(bench_nav)))}]
            for s in sigs:
                stats = dict(zip(cols, perf(res[s], positions[s])))
                rows.append({"策略": s, "样本起": str(starts[s].date()), **stats})
            summary = pd.DataFrame(rows)
            summary.to_csv(os.path.join(self.out_dir, f_sum),
                           index=False, encoding="utf-8-sig")
            all_summaries[tname] = summary

            # 逐年收益（基准 vs 各信号）
            yr = (1 + pd.DataFrame({f"nav_{s}": res[s].pct_change(fill_method=None)
                                    for s in ["买入持有"] + sigs}))\
                .groupby(pd.to_datetime(pd.Series(res["买入持有"].index)).dt.year)\
                .apply(lambda x: x.prod() - 1)
            yr.to_csv(os.path.join(self.out_dir, f_yr), encoding="utf-8-sig")

            print(f"\n===== {tname} =====")
            print(f"回测区间: {b_start.date()} ~ {d.index.max().date()}（各信号独立起点见表）"
                  f" 单边成本={self.cost_bp}bp（已计入）")
            print(summary.round(4).to_string(index=False))

            # 熊市段检验：区间收益 + 区间最大回撤（择时是否躲过 2011-2015 大跌；
            # 信号起点晚于熊市起点的，起点前视为空仓，已注明样本起）
            if report_bear:
                b0, b1 = pd.Timestamp(bear[0]), pd.Timestamp(bear[1])
                m = (d.index >= b0) & (d.index <= b1)
                print(f"\n熊市段 {bear[0]} ~ {bear[1]}（区间收益 / 区间最大回撤）:")
                bear_nav = (1 + ret[m]).cumprod()
                print(f"  买入持有: {bear_nav.iloc[-1]-1:+.1%} / "
                      f"{(bear_nav/bear_nav.cummax()-1).min():.1%}")
                for s in sigs:
                    r_m = positions[s][m] * ret[m]
                    bn = (1 + r_m).cumprod()
                    print(f"  {s} (起{starts[s].date()}): {bn.iloc[-1]-1:+.1%} / "
                          f"{(bn/bn.cummax()-1).min():.1%}")

        # ---- 最新信号快照（AU9999 现货最新日；走势类信号可用到当日，
        #      持仓/份额类受本地底表更新节奏限制，依赖列缺失显示 None）----
        last_dt = sge_close.index.max()
        snap_cols = ["au_close", "au_mom250", "au_ma_bias", "au_dd250",
                     "gsr_z250", "au_oi_z250", "etf_share_z250",
                     "usdx_z250", "us10y_mom250"]
        snap = df.loc[last_dt, snap_cols].to_dict()
        sig_dep2 = {"sig_tsmom": "au_mom250", "sig_ma_band": "au_ma_bias",
                    "sig_mom_dd": "au_dd250", "sig_gsr": "gsr_z250",
                    "sig_etf_ext": "etf_share_z250",
                    "sig_usdx": "usdx_z250", "sig_us10y": "us10y_mom250"}
        sig_snap = {}
        for s, depc in sig_dep2.items():
            if pd.isna(df.loc[last_dt, depc]):
                sig_snap[s] = "N/A(数据止" + str(df[depc].dropna().index.max().date()) + ")"
            else:
                sig_snap[s] = int(df.loc[last_dt, s])
        print(f"\n===== 最新信号快照（{last_dt.date()}，AU9999={snap['au_close']:.2f} 元/克）=====")
        print(f"  mom250={snap['au_mom250']:+.1%}  年线偏离={snap['au_ma_bias']:+.1%}  "
              f"250日回撤={snap['au_dd250']:.1%}")
        print(f"  金银比z={snap['gsr_z250']:+.2f}  持仓z={snap['au_oi_z250']:+.2f}  "
              f"份额流入z={snap['etf_share_z250']:+.2f}")
        print(f"  美元指数z={snap['usdx_z250']:+.2f}  美债10Y动量250={snap['us10y_mom250']:+.1%}")
        print(f"  信号(1=持有): {sig_snap}")

        print(f"\n产物目录: {self.out_dir}")
        return all_summaries


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "refresh":
        GoldTiming().update_sge_price(force=True)   # 仅强制刷新 SGE 现货价
    else:
        GoldTiming().run()
