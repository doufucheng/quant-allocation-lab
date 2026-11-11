# -*- coding: utf-8 -*-
"""
个股Alpha风格轮动模型（四因子合成打分 × 风格指数成分股 → 月度轮动，2026-08-31）

【2026-09-25 翻案重跑】原信号源 combo4 已于 2026-09-14 随"基本面合成/三因子
等权/四因子等权"整体下线（ensemble.py 及 combo_scores/combo4_scores 数据目录
已删除），原样重跑不可行。本轮按用户指令改用现役 7 个 RollComp 因子
（factor_exposure/*.h5：Growth/Lowvol/Momentum/Quality/Reversal/Smallcap/
Valuation RollComp）截面 rank 等权合成作为个股 alpha 信号源，在新腿体系
（style_model 2026-09-25 定版）下验证阶段一判负结论是否翻案。
2026-08-31 首轮判负归因（时序 IC 分裂）见 readme.md 研究阶段简表·阶段一，保留为对照。

一、策略概述
============================================================
用个股 alpha 打分，按四个风格指数的成分股聚合打分，在 style_model 的
四风格指数体系上做月度轮动：

  信号源（2026-09-25 起）：7 个 RollComp 因子各截面 rank(pct) 等权平均
    factor_exposure/*.h5，score ∈ [0,1] 全市场截面百分位
    （替代已下线的 combo4：GRU+CNN+分钟滚动+基本面四因子合成）

  1. 风格桶 = 风格指数成分股（每月末 point-in-time，取 IndexWeights.pq 中
     ≤信号日最近一期权重>0的成分，权重月度更新）。主指数当期成分不全
     （<80%×该指数历史最大期成分数）时，自动切换成分股全量的相似指数：
     - 红利股：中证红利低波成分（h20269.CSI，50只/期全量，2019-04起）
     - 成长股：国证成长成分（980080.CNI，2025年起100只/期全量；
       2021-2024年本地库仅有19~58只记录 → 用同族老国证成长399370.SZ，
       332只/期全量、2010年起）
     - 质量股：中证质量成分（932430.CSI，2025-03起100只/期；
       此前无权重 → 用盈利质量930860.CSI，100只/期全量、2016-07起）
     - 价值股：国证价值成分（980081.CNI同族替代价格腿480081，2025年起
       100只/期；2021-2024年不全 → 用老国证价值399371.SZ，332只/期全量）

  2. 风格alpha信号（时序分位数口径）：成分股内个股alpha打分的指数权重加权均值
     （最近3个打分日平均去噪；成分股∩有打分<10只视为无效），再取当月桶alpha在
     其自身过去12个月历史中的百分位（最少6个月历史，此前预热期等权）。
     相对自身历史取分位可消除各腿alpha水平差异（红利桶长期偏高、成长桶长期
     偏低）导致的静态风格暴露，捕捉各腿alpha的边际变化：
     分位>0.5 = 该风格内选股alpha相对自身过去一年走强，<0.5 走弱。

  3. 轮动规则：分位数映射 z = (分位-0.5)×2 ∈ [-1,1]，
     权重 = 等权 + tilt × z（tilt默认0.15，权重范围[10%,40%]），
     截断非负后归一化。信号退化规则：
     - 分位可得腿≥2：按上式倾斜轮动（分位缺失的腿不参与）
     - 0~1腿有分位：四腿等权（回测前6个月为历史预热期）

  4. 月度再平衡（费前），对照基准：四风格等权组合 + 中证全指全收益。

二、风格腿（与 style_model.py / rotation_rate.py 完全一致，直接复用其数据链路）
============================================================
  红利股：中证红利低波全收益（h20269.CSI，2005-12起按中证红利全收益回补）
  成长股：国证成长100（980080.CNI，按国证成长399370.SZ回补）
  质量股：中证质量全收益（932430CNY010.CSI，按全指消费000990.CSI回补，2010-12起）
  价值股：国证价值100全收益（480081.CNI，按国证价值R（CN2371.CNI）回补）
  指数基准：中证全指全收益（h00985.CSI）

三、数据链路
============================================================
  打分：Config().score_file(Config().combo4_scores_path)（三因子
        combo_scores 可作对照，参数 score_file 可切换）
  成分权重：IndexWeights.pq（ITicker/Date/Ticker/value，月末权重；
        pyarrow 按ITicker过滤读取。IndexMembers.pq成分进出表对四风格指数
        覆盖更差（两只缺失、最早2024-10起），不采用）
  风格指数：IndexEOD.pq（经 StyleModel 加载+历史回补）

四、关键参数
============================================================
  tilt           = 0.15  分位数z → 权重的倾斜系数（权重=等权±tilt）
  n_score_avg    = 3     桶alpha平均的打分日数（打分约每5个交易日一期）
  min_members    = 10    腿内有效成分股数下限（成分∩有打分）
  full_ratio     = 0.8   主指数成分完整率阈值（低于则切换替代指数）
  ts_lookback    = 12    时序分位数回看月数（过去1年）
  ts_min_periods = 6     时序分位数最少有效历史月数（此前预热等权）
  回测起点自动取打分首个可得月末（2021-01-29）

五、更新日志
============================================================
  2026-08-31 初始版本（特征分类口径）
  2026-08-31 改为风格指数成分股口径（IndexWeights聚合，价值腿用同族
             国证价值980081成分替代），并加入信号退化规则
  2026-08-31 成分权重直接asof取最近一期（权重月度更新，去掉45天过期判断）
  2026-08-31 主指数成分不全时自动切换相似指数成分（成长→老国证成长399370、
             价值→老国证价值399371、质量→盈利质量930860，成分全量月度）
  2026-08-31 信号改为桶alpha的过去1年时序分位数（消除静态风格暴露，
             捕捉各腿alpha边际变化；权重=等权+tilt×(分位-0.5)×2）

Author: Fucheng Dou
日期：2026-08-31
"""

import os
import pandas as pd
import numpy as np
import pyarrow.dataset as ds

from utility.factor_operate import FactorOperate
from utility.write_excel import WriteExcel
from utility.date import Date
from config.config import Config

# 复用 style_model 的指数数据链路（加载+历史回补），两种运行方式兼容
try:
    from model_allocation.style_rotation.style_model import StyleModel
except ImportError:
    from style_model import StyleModel


class RotationAlphaModel(object):
    """
    个股Alpha风格轮动模型
    四因子合成个股alpha打分 × 风格指数成分股加权聚合 → 截面z-score → 月度轮动
    """

    def __init__(self):
        cfg = Config()
        self.data_path = cfg.data_dir("derived_data", "allocation_data", "rotation_alpha")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.db_wind_path = cfg.db_wind_path
        self.index_weights_path = self.db_wind_path / "IndexWeights.pq"

        # 信号源（2026-09-25 翻案重跑）：现役 7 个 RollComp 因子（替代已下线 combo4）
        self.rollcomp_factors = [
            "GrowthRollComp", "LowvolRollComp", "MomentumRollComp",
            "QualityRollComp", "ReversalRollComp", "SmallcapRollComp",
            "ValuationRollComp",
        ]
        self.rollcomp_path = cfg.data_dir(
            "derived_data", "stock_data", "alpha_model", "factor_exposure")

        # 风格腿：与 style_model.py 完全一致（数据加载直接复用 StyleModel）
        self.style_index = {
            '红利股': 'h20269.CSI',      # 中证红利低波全收益
            '成长股': '980080.CNI',      # 国证成长100（价格版）
            '质量股': '932430CNY010.CSI',  # 中证质量全收益
            '价值股': '480081.CNI',      # 国证价值100全收益
        }

        # 成分股权重代码（IndexWeights.pq）：{腿: (主指数, 替代指数)}
        # 主指数某期权重记录不全（<full_ratio×该指数历史最大期成分数）时，
        # 用成分股全量的相似指数替代当期成分：
        # - 成长/价值主指数980080/980081在2021-2024年每期仅有19~58只权重记录
        #   （应有100只，2025年起才完整），替代用同族老代码国证成长/价值
        #   （399370.SZ/399371.SZ，巨潮风格系列，每期332只全量、2010年起月度）
        # - 质量主指数932430权重2025-03起才有，替代用盈利质量930860.CSI
        #   （每期100只全量、2016-07起月度）
        # - 红利低波h20269每期50只全量，无需替代
        self.member_index = {
            '红利股': ('h20269.CSI', None),          # 2019-04起，50只全量
            '成长股': ('980080.CNI', '399370.SZ'),    # 主2021-12起；替代=老国证成长
            '质量股': ('932430.CSI', '930860.CSI'),   # 主2025-03起；替代=盈利质量
            '价值股': ('980081.CNI', '399371.SZ'),    # 国证价值；替代=老国证价值
        }

        self.benchmark_code = 'h00985.CSI'      # 中证全指全收益

        # 信号与轮动参数
        self.tilt = 0.15            # 分位数z → 权重倾斜系数（z∈[-1,1]→权重±tilt）
        self.n_score_avg = 3        # alpha信号平均的打分日数
        self.min_members = 10       # 腿内有效成分股数下限
        self.full_ratio = 0.8       # 成分完整率阈值（当期成分数/该指数历史最大期成分数）
        self.ts_lookback = 12       # 时序分位数回看月数（过去1年）
        self.ts_min_periods = 6     # 时序分位数最少有效历史月数（此前预热等权）

        print("\n" + "=" * 80)
        print("个股Alpha风格轮动模型 - 参数配置")
        print("=" * 80)
        print(f"打分源：{len(self.rollcomp_factors)} 个 RollComp 因子等权合成（{self.rollcomp_path}）")
        print(f"成分股权重（主, 替代）：{self.member_index}")
        print(f"成分完整率阈值：{self.full_ratio:.0%}（主指数当期不全时用替代指数成分）")
        print(f"信号：桶alpha的过去{self.ts_lookback}个月时序分位数"
              f"（最少{self.ts_min_periods}个月历史，预热期等权）")
        print(f"倾斜系数 tilt：{self.tilt}（分位z∈[-1,1] → 权重=等权±tilt）")
        print(f"alpha信号平均打分日数：{self.n_score_avg}")
        print(f"再平衡频率：月度")
        print(f"风格腿：{self.style_index}")
        print(f"基准：{self.benchmark_code}")
        print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    def load_scores(self):
        """
        加载个股 alpha 打分 → 宽表（index=打分日Timestamp, columns=Ticker）

        2026-09-25 翻案重跑信号源：7 个 RollComp 因子（factor_exposure/*.h5，
        index=股票 × columns=日期，HdfMfc 读取）各截面 rank(pct) 等权平均
        → score ∈ [0,1] 全市场截面百分位。
        """
        print(f"加载 RollComp 因子: {self.rollcomp_factors}")
        fo = FactorOperate()
        comp_scores = []
        for name in self.rollcomp_factors:
            s = fo.read_factor_data(name, path=self.rollcomp_path, factor_type='h5')
            if s is None or s.empty:
                raise RuntimeError("RollComp 因子读取失败: %s" % name)
            r = s.rank(axis=0, pct=True)      # 截面 rank（每日期列内按股票排名）
            comp_scores.append(r.T)           # → index=日期, columns=股票
        aligned = comp_scores[0]
        for s in comp_scores[1:]:
            aligned = aligned.add(s, fill_value=np.nan)
        scores = aligned / len(comp_scores)
        scores.index = pd.to_datetime(scores.index.astype(str), format='%Y%m%d')
        scores = scores.sort_index()
        print(f"打分日数: {len(scores)}, "
              f"{scores.index.min().date()} ~ {scores.index.max().date()}, "
              f"日均覆盖 {int(scores.notna().sum(axis=1).median())} 只")
        return scores

    def load_index_weights(self):
        """
        读取各腿主/替代指数的成分股权重

        返回
        -------
        dict
            {腿名: {'main': (代码, 宽表), 'alt': (代码, 宽表) or None,
                    'full': {代码: 完整成分数阈值}}}
            宽表 index=Date, columns=Ticker, values=权重；
            完整成分数阈值 = 该指数历史最大期成分数 × full_ratio
        """
        codes = []
        for main, alt in self.member_index.values():
            codes.append(main)
            if alt:
                codes.append(alt)
        print(f"读取 IndexWeights 成分权重: {codes}")
        dataset = ds.dataset(str(self.index_weights_path))
        flt = ds.field('ITicker').isin(codes)
        iw = dataset.to_table(filter=flt).to_pandas()

        members = {}
        for style_name, (main, alt) in self.member_index.items():
            entry = {'full': {}}
            for tag, code in (('main', main), ('alt', alt)):
                if code is None:
                    entry[tag] = None
                    continue
                wide = iw.xs(code, level='ITicker')['value'].unstack('Ticker').sort_index()
                per = (wide > 0).sum(axis=1)          # 每期权重>0的成分股数
                entry[tag] = (code, wide)
                entry['full'][code] = self.full_ratio * per.max()
                print(f"  {style_name} {tag} ({code}): {wide.shape[0]}期 "
                      f"{wide.index.min().date()} ~ {wide.index.max().date()}, "
                      f"每期成分数 min/中位/max = {per.min()}/"
                      f"{int(per.median())}/{per.max()}")
            members[style_name] = entry
        return members

    # ------------------------------------------------------------------
    # 信号构建
    # ------------------------------------------------------------------
    def pick_members(self, entry, d):
        """
        信号日 d 的腿成分选择：主指数优先，当期不全时切换替代指数

        规则：asof 取≤d的最近一期权重；主指数当期成分数 ≥ 完整阈值则用主，
        否则替代指数当期可用则用替代，再否则退回主指数的部分成分。

        返回
        -------
        (weights Series, 来源代码 str) 或 (None, None)
        """
        best_fallback = None
        for tag in ('main', 'alt'):
            item = entry.get(tag)
            if item is None:
                continue
            code, wide = item
            avail = wide.index[wide.index <= d]
            if len(avail) == 0:
                continue
            w = wide.loc[avail[-1]]
            w = w[w > 0]
            if len(w) >= entry['full'][code]:
                return w, code
            if best_fallback is None or len(w) > len(best_fallback[0]):
                best_fallback = (w, code)
        return best_fallback if best_fallback is not None else (None, None)

    def compute_signals(self, scores, date_series):
        """
        逐月末计算风格alpha信号与轮动权重（时序分位数口径）

        两步计算：
        1. 桶alpha：各腿取 IndexWeights 中≤信号日的最近一期权重（>0成分，
           主指数不全时切换替代指数），与个股alpha打分取交集后做指数权重
           加权均值（近n_score_avg个打分日平均）。
        2. 时序分位数信号：当月桶alpha在其自身过去 ts_lookback 个月历史中的
           百分位（不含当月；有效历史<ts_min_periods视为无信号），再映射权重：
           z = (分位 - 0.5) × 2 ∈ [-1,1]，权重 = 等权 + tilt×z，截断非负归一。
           相对自身历史取分位可消除各腿alpha水平差异（红利桶alpha长期偏高、
           成长桶长期偏低）导致的静态风格暴露，捕捉各腿alpha的边际变化。

        参数
        ----------
        scores : DataFrame
            打分宽表（load_scores 输出）
        date_series : list of str
            回测月末交易日 YYYYMMDD

        返回
        -------
        signals : DataFrame
            index=月末YYYYMMDD，columns=各风格alpha/分位/成分数/来源/权重
        """
        print("构建风格alpha信号（成分股口径，时序分位数）...")
        members = self.load_index_weights()

        style_names = list(self.style_index.keys())

        # ---- 第一步：逐月桶alpha ----
        sig_rows = {}
        for date_str in date_series:
            d = pd.Timestamp(date_str)

            # alpha：最近 n_score_avg 个打分日的均值
            sdates = scores.index[scores.index <= d]
            if len(sdates) == 0:
                continue
            alpha_stock = scores.loc[sdates[-self.n_score_avg:]].mean()
            alpha_valid = alpha_stock.dropna()
            if len(alpha_valid) == 0:
                continue

            row = {}
            for s in style_names:
                w, src = self.pick_members(members[s], d)
                if w is None:
                    row[f'{s}alpha'] = np.nan
                    row[f'{s}数量'] = 0
                    row[f'{s}来源'] = '无'
                    continue
                inter = w.index.intersection(alpha_valid.index)
                if len(inter) < self.min_members:
                    row[f'{s}alpha'] = np.nan
                    row[f'{s}数量'] = len(inter)
                    row[f'{s}来源'] = src
                    continue
                # 指数权重加权alpha（权重归一到交集内）
                wi = w[inter] / w[inter].sum()
                row[f'{s}alpha'] = float((wi * alpha_valid[inter]).sum())
                row[f'{s}数量'] = len(inter)
                row[f'{s}来源'] = src

            sig_rows[date_str] = row

        if not sig_rows:
            print("警告: 未生成任何信号（打分与月末日历无交集）")
            return pd.DataFrame()

        # ---- 第二步：桶alpha → 过去1年时序分位数 → 权重 ----
        dates_order = sorted(sig_rows.keys())
        for s in style_names:
            series = pd.Series([sig_rows[d].get(f'{s}alpha', np.nan)
                                for d in dates_order], index=dates_order)
            for i, d in enumerate(dates_order):
                cur = series.iloc[i]
                if pd.isna(cur):
                    sig_rows[d][f'{s}分位'] = np.nan
                    continue
                hist = series.iloc[max(0, i - self.ts_lookback):i].dropna()
                if len(hist) < self.ts_min_periods:
                    sig_rows[d][f'{s}分位'] = np.nan
                    continue
                # 重复值按半票处理，历史全为同值时分位0.5
                pct = ((hist < cur).sum() + 0.5 * (hist == cur).sum()) / len(hist)
                sig_rows[d][f'{s}分位'] = float(pct)

        for d in dates_order:
            row = sig_rows[d]
            # 分位 → z → 权重（退化规则：≥2腿可得倾斜/可得腿等权；0~1腿全等权）
            pcts = pd.Series({s: row.get(f'{s}分位', np.nan) for s in style_names})
            valid = pcts.dropna()
            weights = pd.Series(0.0, index=style_names)
            if len(valid) <= 1:
                weights.loc[:] = 1.0 / len(style_names)
            elif len(valid) < len(style_names):
                weights.loc[valid.index] = 1.0 / len(valid)
            else:
                z = (valid - 0.5) * 2.0          # 分位数映射到[-1,1]
                w = (1.0 / len(style_names) + self.tilt * z).clip(lower=0.0)
                weights = w / w.sum()
            for s in style_names:
                row[f'{s}权重'] = weights[s]
            sig_rows[d] = row

        signals = pd.DataFrame(sig_rows).T
        n_valid = signals[[f'{s}分位' for s in style_names]].notna().sum(axis=1)
        print(f"信号覆盖 {len(signals)} 个月末: "
              f"{signals.index[0]} ~ {signals.index[-1]}；"
              f"4腿分位齐全{int((n_valid == 4).sum())}月 / "
              f"预热等权{int((n_valid < 2).sum())}月"
              f"（前{self.ts_min_periods}个月为历史预热期）")
        return signals

    # ------------------------------------------------------------------
    # 回测
    # ------------------------------------------------------------------
    def backtest(self):
        """
        回测：月末信号 → 下月持仓 → 月度收益（费前）

        收益归属：买入本月末、卖出下月末，计入下月末所在月份
        （与 style_model / rotation_rate 口径一致）
        """
        print("\n" + "=" * 80)
        print("开始回测个股Alpha风格轮动策略")
        print("=" * 80)

        # 风格指数数据（复用 StyleModel：加载 + 历史回补；含补齐指数代码）
        sm = StyleModel()
        all_codes = list(self.style_index.values()) + list(sm.fill_index.values()) \
            + [self.benchmark_code]
        style_data = sm.fill_price_data(sm.load_index_data(list(set(all_codes))))

        # 打分与回测日历（起点=首个有打分的月末）
        scores = self.load_scores()
        date_series = Date().get_trade_date_series(
            scores.index.min().strftime('%Y%m%d'),
            style_data.index[-1], 'M')
        print(f"回测期间: {date_series[0]} 至 {date_series[-1]}，"
              f"共 {len(date_series)} 个调仓时点")

        # 信号
        signals = self.compute_signals(scores, date_series)

        style_names = list(self.style_index.keys())
        ret_dict = {}
        prev_w = None

        def next_trade_date(date):
            """信号收盘后首个可成交交易日。"""
            pos = style_data.index.searchsorted(date, side='right')
            return style_data.index[pos] if pos < len(style_data.index) else None

        for i in range(len(date_series) - 1):
            date = date_series[i]
            next_date = date_series[i + 1]
            buy_date = next_trade_date(date)
            sell_date = next_trade_date(next_date)
            if (date not in style_data.index or next_date not in style_data.index
                    or buy_date is None or sell_date is None):
                continue

            # 月末 alpha 信号收盘后才可得，下一交易日成交。
            period_ret = {}
            for s in style_names:
                code = self.style_index[s]
                if code in style_data.columns:
                    buy = style_data.loc[buy_date, code]
                    sell = style_data.loc[sell_date, code]
                    period_ret[s + '收益'] = sell / buy - 1 \
                        if pd.notna(buy) and pd.notna(sell) else np.nan
                else:
                    period_ret[s + '收益'] = np.nan

            # 当月权重（信号缺失退化为可得腿等权）
            if date in signals.index:
                w = pd.Series({s: signals.loc[date, s + '权重'] for s in style_names},
                              dtype=float)
            else:
                w = None
            if w is None or w.isna().all():
                avail = [s for s in style_names if pd.notna(period_ret[s + '收益'])]
                w = pd.Series(0.0, index=style_names)
                w.loc[avail] = 1.0 / len(avail)

            port_gross = sum(w[s] * period_ret[s + '收益'] for s in style_names
                             if pd.notna(period_ret.get(s + '收益', np.nan))
                             and pd.notna(w[s]))
            turn_m = w.abs().sum() if prev_w is None else 0.5 * (w - prev_w).abs().sum()
            period_ret['组合收益'] = port_gross - turn_m * 0.003  # 费后（2026-10-10 全族统一单边千三）
            period_ret['等权收益'] = np.mean(
                [period_ret[s + '收益'] for s in style_names
                 if pd.notna(period_ret[s + '收益'])])

            # 基准
            if self.benchmark_code in style_data.columns:
                b = style_data[self.benchmark_code]
                if buy_date in b.index and sell_date in b.index:
                    period_ret['基准收益'] = b.loc[sell_date] / b.loc[buy_date] - 1

            # 权重与换手（首月从现金建仓，单边换手=Σ|w|）
            for s in style_names:
                period_ret[s + '权重'] = w[s]
            period_ret['换手'] = turn_m
            prev_w = w

            ret_dict[next_date] = period_ret

        port_ret = pd.DataFrame(ret_dict).T
        port_ret['轮动净值'] = (1 + port_ret['组合收益'].fillna(0)).cumprod()
        port_ret['等权净值'] = (1 + port_ret['等权收益'].fillna(0)).cumprod()
        if '基准收益' in port_ret.columns:
            port_ret['基准净值'] = (1 + port_ret['基准收益'].fillna(0)).cumprod()

        summary = self.calculate_performance(port_ret, signals, style_names)
        return port_ret, signals, summary

    # ------------------------------------------------------------------
    # 绩效与保存
    # ------------------------------------------------------------------
    def calculate_performance(self, ret, signals, style_names):
        """组合绩效指标 + 分年度汇总 + 保存Excel"""
        def perf(nav, r):
            total = nav.iloc[-1] - 1
            n = len(r) / 12.0
            an = (1 + total) ** (1 / n) - 1 if total > -1 else -1.0
            vol = r.std() * np.sqrt(12)
            mdd = (nav / nav.expanding().max() - 1).min()
            sharpe = an / vol if vol > 0 else np.nan
            calmar = -an / mdd if mdd < 0 else np.nan
            return total, an, vol, mdd, sharpe, calmar

        stats = {}
        for name, nav_col, ret_col in [('轮动', '轮动净值', '组合收益'),
                                       ('等权', '等权净值', '等权收益')]:
            stats[name] = perf(ret[nav_col], ret[ret_col])
        if '基准净值' in ret.columns:
            stats['基准'] = perf(ret['基准净值'], ret['基准收益'])

        print("\n====== 个股Alpha风格轮动绩效 ======")
        for name, (total, an, vol, mdd, sharpe, calmar) in stats.items():
            print(f"[{name}] 总收益{total:.2%} 年化{an:.2%} 波动{vol:.2%} "
                  f"回撤{mdd:.2%} 夏普{sharpe:.2f} 卡玛{calmar:.2f}")
        print(f"年化超额(vs等权): {stats['轮动'][1] - stats['等权'][1]:.2%}")
        if '基准' in stats:
            print(f"年化超额(vs基准): {stats['轮动'][1] - stats['基准'][1]:.2%}")
        print(f"月均单边换手: {ret['换手'].mean():.2%}")

        # 分年度汇总
        ret['年份'] = ret.index.map(lambda x: x[0:4])
        years = sorted(set(ret['年份']))
        cols = ['组合收益', '等权收益', '基准收益', '超额(vs等权)',
                '组合回撤', '夏普'] + [s + '收益' for s in style_names]
        summary = pd.DataFrame(index=years, columns=cols, dtype=float)
        for year in years:
            ry = ret[ret['年份'] == year]
            summary.loc[year, '组合收益'] = (1 + ry['组合收益'].fillna(0)).prod() - 1
            summary.loc[year, '等权收益'] = (1 + ry['等权收益'].fillna(0)).prod() - 1
            if '基准收益' in ry.columns:
                summary.loc[year, '基准收益'] = (1 + ry['基准收益'].fillna(0)).prod() - 1
            summary.loc[year, '超额(vs等权)'] = \
                summary.loc[year, '组合收益'] - summary.loc[year, '等权收益']
            nav_y = (1 + ry['组合收益'].fillna(0)).cumprod()
            summary.loc[year, '组合回撤'] = (nav_y / nav_y.cummax() - 1).min()
            vol_y = ry['组合收益'].std() * np.sqrt(12)
            summary.loc[year, '夏普'] = summary.loc[year, '组合收益'] / vol_y \
                if vol_y > 0 else np.nan
            for s in style_names:
                summary.loc[year, s + '收益'] = (1 + ry[s + '收益'].fillna(0)).prod() - 1
        summary.loc['全区间', '组合收益'] = stats['轮动'][1]
        summary.loc['全区间', '等权收益'] = stats['等权'][1]
        if '基准' in stats:
            summary.loc['全区间', '基准收益'] = stats['基准'][1]
        summary.loc['全区间', '超额(vs等权)'] = stats['轮动'][1] - stats['等权'][1]
        summary.loc['全区间', '组合回撤'] = stats['轮动'][3]
        summary.loc['全区间', '夏普'] = stats['轮动'][4]
        for s in style_names:
            summary.loc['全区间', s + '收益'] = \
                (1 + ret[s + '收益'].fillna(0)).prod() - 1

        print("\n====== 分年度绩效 ======")
        print(summary)

        self.save_results(ret, signals, summary, style_names)
        return summary

    def save_results(self, ret, signals, summary, style_names):
        """保存回测结果 Excel（回测明细 / 信号明细 / 绩效汇总）"""
        filename = os.path.join(self.data_path, 'rotation_alpha_backtest.xlsx')
        excel = WriteExcel(filename)

        # 回测明细
        ws = excel.add_worksheet('回测结果')
        fmt = pd.DataFrame([], columns=ret.columns, index=['format'])
        for c in ret.columns:
            fmt.loc['format', c] = '0.0%' if c not in ('年份',) else '@'
        excel.write_pandas(ret, ws, begin_row_number=0, begin_col_number=1,
                           num_format_pd=fmt, color='gray', fillna=True)

        # 信号明细
        ws2 = excel.add_worksheet('信号明细')
        fmt2 = pd.DataFrame([], columns=signals.columns, index=['format'])
        for c in signals.columns:
            if '来源' in c:
                fmt2.loc['format', c] = '@'
            elif 'alpha' in c or 'z' in c:
                fmt2.loc['format', c] = '0.0%'
            elif '数量' in c:
                fmt2.loc['format', c] = '0'
            else:
                fmt2.loc['format', c] = '0.0%'
        excel.write_pandas(signals, ws2, begin_row_number=0, begin_col_number=1,
                           num_format_pd=fmt2, color='gray', fillna=True)

        # 绩效汇总
        col_number = 3 + len(ret.columns)
        fmt3 = pd.DataFrame([], columns=summary.columns, index=['format'])
        fmt3.loc['format', :] = '0.0%'
        excel.write_pandas(summary, ws, begin_row_number=0, begin_col_number=col_number,
                           num_format_pd=fmt3, color='gray', fillna=True)

        excel.close()
        print(f"\n结果已保存至: {filename}")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    pd.options.mode.chained_assignment = None

    model = RotationAlphaModel()
    model.backtest()
