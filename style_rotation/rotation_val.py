# -*- coding: utf-8 -*-
"""
估值分位风格轮动模型（IndexValuation.pq 估值分位 → 风格指数月度轮动，2026-08-31）

一、策略概述
============================================================
按研究综述（style_rotation_research.md）优先级第一的"估值价差/估值分位"信号构建：
每个风格腿的指数 PE_TTM 在自身过去 60 个月（5年）历史中的分位数，
估值越便宜的风格超配——依据 Arnott et al. (2021, FAJ)：估值价差变化解释
风格收益的约2/3，极值分位处存在均值回复。

  信号：分位 pct ∈ [0,1]（越低越便宜）
  映射：z = (0.5 - pct) × 2 ∈ [-1,1]（便宜→正）
        权重 = 等权 + tilt × z（tilt默认0.15，权重范围约[10%,40%]）
        截断非负后归一化
  退化规则：滚动历史不足 min_periods 的腿不参与；≥2腿可得按可得腿轮动，
            0~1腿可得四腿等权

二、估值数据（IndexValuation.pq，价格腿与 style_model 一致，估值用同族指数官方 PE）
============================================================
  红利股：价格 h20269.CSI ｜ 估值 h30269.CSI（2013-12起，PE 4.6-14.3 合理，无补齐）
  成长股：价格 980080.CNI ｜ 估值 399370.SZ 国证成长整段（2010-01起）
  质量股：价格 932430CNY010.CSI ｜ 估值 932430.CSI（2024-12起）+ 000990.CSI 补齐（2011-08起）
  价值股：价格 480081.CNI ｜ 估值 399371.SZ 国证价值整段（2010-01起）
  基准：中证全指全收益（h00985.CSI）
  注1：全收益指数无估值数据，估值分位用同族指数 PE_TTM 代理。
  注2：数据质量修正（2026-09-25）——980080/980081 官方 PE_TTM 口径异常弃用
  （980081 2012-2024 虚高 3-8 倍、2025 才修正；980080 重叠期比率摆动 1.0-4.8），
  成长/价值腿整段改用 399370/399371 官方 PE；质量腿拼接比率 0.41-0.48 稳定保留。
  注3：估值补齐（回测时长原则）——质量腿主段前用代理官方 PE 拼接，重叠期
  PE 比率中位数水平校正，滚动分位跨拼接点连续。补齐后最早估值 2010-01
  （成长/价值），四腿齐点 2013-12（红利）。

三、关键参数
============================================================
  window_m    = 60   滚动分位窗口（月，5年）
  min_periods = 36   滚动分位最少历史（月，3年）→ 红利/成长/价值约2016-2017年起可用
  tilt        = 0.15 分位z → 权重倾斜系数
  回测起点自动取首个"≥2腿有分位"的月末；月度再平衡费前

四、更新日志
============================================================
  2026-08-31 初始版本（依据 style_rotation_research.md 第四节落地要素）

Author: Fucheng Dou
日期：2026-08-31
"""

import os
import pandas as pd
import numpy as np
import pyarrow.dataset as ds

from utility.write_excel import WriteExcel
from utility.date import Date
from config.config import Config

try:
    from model_allocation.style_rotation.style_model import StyleModel
except ImportError:
    from style_model import StyleModel


class RotationValModel(object):
    """
    估值分位风格轮动模型
    风格指数PE_TTM的5年滚动分位 → 便宜超配 → 月度轮动
    """

    def __init__(self):
        self.data_path = Config().data_dir("derived_data", "allocation_data", "rotation_val")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.db_wind_path = Config().db_wind_path
        self.index_valuation_path = self.db_wind_path / "IndexValuation.pq"

        # 风格腿（与 style_model.py 完全一致）
        self.style_index = {
            '红利股': 'h20269.CSI',      # 中证红利低波全收益
            '成长股': '980080.CNI',      # 国证成长100（价格版）
            '质量股': '932430CNY010.CSI',  # 中证质量全收益
            '价值股': '480081.CNI',      # 国证价值100全收益
        }

        # 估值代理指数（IndexValuation.pq；全收益指数无估值，用同族指数PE_TTM）
        # 2026-09-25 数据质量修正：980080/980081 官方 PE_TTM 口径异常弃用——
        # 980081 2012-2024 虚高 3-8 倍（PE 30-83，2025 起才修正为 ~9 的正常口径），
        # 980080 与 399370 重叠期比率 1.0-4.8 剧烈摆动；成长/价值腿估值整段改用
        # 官方可信的 399370.SZ / 399371.SZ（2010-01 起，与价格腿补齐代理同族同向）
        self.valuation_index = {
            '红利股': 'h30269.CSI',      # 中证红利低波族，2013-12起（PE 4.6-14.3 合理）
            '成长股': '399370.SZ',       # 国证成长，2010-01起（整段）
            '质量股': '932430.CSI',      # 2024-12起 + 000990 补齐（比率 0.41-0.48 稳定）
            '价值股': '399371.SZ',       # 国证价值，2010-01起（整段）
        }

        # 估值补齐指数（2026-09-25 按回测时长原则延长：主指数估值起点前用代理
        # 官方 PE 拼接；重叠期 PE 比率中位数水平校正保证滚动分位连续；
        # 仅质量腿需要补齐，红利腿无更早官方代理，成长/价值腿整段即代理）
        self.valuation_fill_index = {
            '质量股': '000990.CSI',     # 全指消费，官方 PE 2011-08 起
        }

        self.benchmark_code = 'h00985.CSI'      # 中证全指全收益

        # 参数
        self.window_m = 60         # 滚动分位窗口（月）
        self.min_periods = 36      # 滚动分位最少历史（月）
        self.tilt = 0.15           # 分位z → 权重倾斜系数

        print("\n" + "=" * 80)
        print("估值分位风格轮动模型 - 参数配置")
        print("=" * 80)
        print(f"估值代理指数：{self.valuation_index}")
        print(f"滚动分位窗口：{self.window_m}月（最少{self.min_periods}月）")
        print(f"倾斜系数 tilt：{self.tilt}（便宜超配，权重=等权±tilt）")
        print(f"再平衡频率：月度（费前）")
        print(f"风格腿：{self.style_index}")
        print(f"基准：{self.benchmark_code}")
        print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # 数据与信号
    # ------------------------------------------------------------------
    def load_index_valuation(self):
        """
        读取各腿估值代理指数的 PE_TTM → 月度宽表（含代理补齐段）

        补齐规则（2026-09-25）：主指数估值起点之前，用补齐指数（与价格腿
        fill_index 同构）的官方 PE_TTM，经重叠期 PE 比率中位数水平校正后拼接，
        保证滚动分位跨拼接点连续。

        返回
        -------
        字典，值为 DataFrame
            {腿名: Series(index=月末Timestamp, values=PE_TTM)}
        """
        codes = list(self.valuation_index.values()) + \
            list(self.valuation_fill_index.values())
        print(f"读取 IndexValuation 估值: {codes}")
        dataset = ds.dataset(str(self.index_valuation_path))
        # columns 须带 Date/Ticker 索引列，否则 to_pandas 无法还原 MultiIndex
        iv = dataset.to_table(
            filter=ds.field('Ticker').isin(codes),
            columns=['Date', 'Ticker', 'PE_TTM']).to_pandas()

        def _monthly(code):
            s = iv.xs(code, level='Ticker')['PE_TTM'].dropna().sort_index()
            if len(s) == 0:
                return pd.Series(dtype=float)
            return s.groupby(pd.PeriodIndex(s.index, freq='M')).last()

        val = {}
        for style_name, code in self.valuation_index.items():
            main = _monthly(code)
            fill_code = self.valuation_fill_index.get(style_name)
            fill = _monthly(fill_code) if fill_code else pd.Series(dtype=float)
            if len(main) == 0 and len(fill) == 0:
                val[style_name] = pd.Series(dtype=float)
                print(f"  {style_name} ({code}): 无估值数据")
                continue
            if len(main) > 0 and len(fill) > 0 and main.index.min() > fill.index.min():
                # 拼接：重叠期 PE 比率中位数做水平校正，主段前用校正后代理段
                overlap = main.index.intersection(fill.index)
                if len(overlap) >= 6:
                    ratio = (main.loc[overlap] / fill.loc[overlap]).median()
                    fill_adj = fill[fill.index < main.index.min()] * ratio
                    combined = pd.concat([fill_adj, main])
                    combined = combined[~combined.index.duplicated()]
                    print(f"  {style_name} ({code}): 主段{len(main)}月"
                          f"（{main.index.min()}起）+ 代理{fill_code}补齐"
                          f"{len(fill_adj)}月（水平校正×{ratio:.3f}）")
                    val[style_name] = combined
                    continue
            if len(main) == 0:
                val[style_name] = fill
                print(f"  {style_name}: 主指数无估值，整段用代理 {fill_code}")
                continue
            val[style_name] = main
            print(f"  {style_name} ({code}): {len(main)}个月末 "
                  f"{main.index.min()} ~ {main.index.max()}")
        return val

    def compute_signals(self, val, month_ends):
        """
        各腿PE_TTM滚动分位 → 权重

        参数
        ----------
        val : dict of Series
            load_index_valuation 输出（index=Period月）
        month_ends : list of str
            回测月末 YYYYMMDD

        返回
        -------
        signals : DataFrame
            index=月末YYYYMMDD，columns=各腿分位/权重
        """
        style_names = list(self.style_index.keys())
        sig_rows = {}
        for date_str in month_ends:
            d = pd.Period(date_str[:6], 'M')
            row = {}
            for s in style_names:
                series = val.get(s)
                if series is None or len(series) == 0:
                    row[f'{s}分位'] = np.nan
                    continue
                hist = series.loc[:d]
                cur = hist.iloc[-1] if len(hist) > 0 else np.nan
                hist = hist.iloc[:-1].iloc[-self.window_m:]     # 不含当月的窗口
                if pd.isna(cur) or len(hist) < self.min_periods:
                    row[f'{s}分位'] = np.nan
                    continue
                # PE越低越便宜：分位 = 历史中低于当前的比例
                row[f'{s}分位'] = float((hist < cur).mean())

            # 分位 → z → 权重（便宜超配；≥2腿可得轮动，否则全等权）
            pcts = pd.Series({s: row.get(f'{s}分位', np.nan) for s in style_names})
            valid = pcts.dropna()
            weights = pd.Series(0.0, index=style_names)
            if len(valid) <= 1:
                weights.loc[:] = 1.0 / len(style_names)
            elif len(valid) < len(style_names):
                # 可得腿间仍按"便宜超配"倾斜（分位映射与全腿一致，归一到可得腿）
                z = (0.5 - valid) * 2.0
                w = (1.0 / len(valid) + self.tilt * z).clip(lower=0.0)
                weights.loc[valid.index] = w / w.sum()
            else:
                z = (0.5 - valid) * 2.0
                w = (1.0 / len(style_names) + self.tilt * z).clip(lower=0.0)
                weights = w / w.sum()
            for s in style_names:
                row[f'{s}权重'] = weights[s]
            sig_rows[date_str] = row

        signals = pd.DataFrame(sig_rows).T
        n_valid = signals[[f'{s}分位' for s in style_names]].notna().sum(axis=1)
        if len(signals) > 0:
            print(f"信号覆盖 {len(signals)} 个月末: {signals.index[0]} ~ "
                  f"{signals.index[-1]}；4腿分位齐全{int((n_valid == 4).sum())}月 / "
                  f"部分{int((n_valid >= 2).sum() - (n_valid == 4).sum())}月 / "
                  f"退化等权{int((n_valid < 2).sum())}月")
        return signals

    # ------------------------------------------------------------------
    # 回测
    # ------------------------------------------------------------------
    def backtest(self):
        """回测：月末估值分位信号 → 下月首个交易日成交 → 月度收益（费前）"""
        print("\n" + "=" * 80)
        print("开始回测估值分位风格轮动策略")
        print("=" * 80)

        # 风格指数价格（复用 StyleModel：加载 + 历史回补）
        sm = StyleModel()
        all_codes = list(self.style_index.values()) + list(sm.fill_index.values()) \
            + [self.benchmark_code]
        style_data = sm.fill_price_data(sm.load_index_data(list(set(all_codes))))

        # 估值与分位（预计算全区间，再取回测月末）
        val = self.load_index_valuation()

        # 回测日历：估值数据首个月末起（分位预热在其后 min_periods 个月完成）
        first_period = min(v.index.min() for v in val.values() if len(v) > 0)
        beg = (first_period + self.min_periods + 1).to_timestamp(how='end')
        beg_str = beg.strftime('%Y%m%d')
        month_ends = Date().get_trade_date_series(beg_str, style_data.index[-1], 'M')
        print(f"回测期间: {month_ends[0]} 至 {month_ends[-1]}，共 {len(month_ends)} 个调仓时点")

        signals = self.compute_signals(val, month_ends)

        style_names = list(self.style_index.keys())
        ret_dict = {}
        prev_w = None

        def next_trade_date(date):
            """信号收盘后首个可成交交易日。"""
            pos = style_data.index.searchsorted(date, side='right')
            return style_data.index[pos] if pos < len(style_data.index) else None

        for i in range(len(month_ends) - 1):
            date, next_date = month_ends[i], month_ends[i + 1]
            buy_date = next_trade_date(date)
            sell_date = next_trade_date(next_date)
            if (date not in style_data.index or next_date not in style_data.index
                    or buy_date is None or sell_date is None):
                continue

            period_ret = {}
            for s in style_names:
                code = self.style_index[s]
                if code in style_data.columns:
                    buy, sell = style_data.loc[buy_date, code], style_data.loc[sell_date, code]
                    period_ret[s + '收益'] = sell / buy - 1 \
                        if pd.notna(buy) and pd.notna(sell) else np.nan
                else:
                    period_ret[s + '收益'] = np.nan

            # 当月权重
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
                             if pd.notna(period_ret.get(s + '收益', np.nan)) and pd.notna(w[s]))
            turn_m = w.abs().sum() if prev_w is None else 0.5 * (w - prev_w).abs().sum()
            period_ret['组合收益'] = port_gross - turn_m * 0.003  # 费后（2026-10-10 全族统一单边千三）
            period_ret['等权收益'] = np.mean(
                [period_ret[s + '收益'] for s in style_names
                 if pd.notna(period_ret[s + '收益'])])

            if self.benchmark_code in style_data.columns:
                b = style_data[self.benchmark_code]
                if buy_date in b.index and sell_date in b.index:
                    period_ret['基准收益'] = b.loc[sell_date] / b.loc[buy_date] - 1

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

        self.calculate_performance(port_ret, signals, style_names)
        return port_ret, signals

    # ------------------------------------------------------------------
    # 绩效与保存
    # ------------------------------------------------------------------
    def calculate_performance(self, ret, signals, style_names):
        """组合绩效指标 + 分年度汇总 + 保存Excel（口径与 rotation_alpha 一致）"""
        def perf(nav, r):
            total = nav.iloc[-1] - 1
            n = len(r) / 12.0
            an = (1 + total) ** (1 / n) - 1 if total > -1 else -1.0
            vol = r.std() * np.sqrt(12)
            mdd = (nav / nav.expanding().max() - 1).min()
            return total, an, vol, mdd, (an / vol if vol > 0 else np.nan), \
                (-an / mdd if mdd < 0 else np.nan)

        stats = {'轮动': perf(ret['轮动净值'], ret['组合收益']),
                 '等权': perf(ret['等权净值'], ret['等权收益'])}
        if '基准净值' in ret.columns:
            stats['基准'] = perf(ret['基准净值'], ret['基准收益'])

        print("\n====== 估值分位风格轮动绩效 ======")
        for name, (total, an, vol, mdd, sharpe, calmar) in stats.items():
            print(f"[{name}] 总收益{total:.2%} 年化{an:.2%} 波动{vol:.2%} "
                  f"回撤{mdd:.2%} 夏普{sharpe:.2f} 卡玛{calmar:.2f}")
        print(f"年化超额(vs等权): {stats['轮动'][1] - stats['等权'][1]:.2%}")
        if '基准' in stats:
            print(f"年化超额(vs基准): {stats['轮动'][1] - stats['基准'][1]:.2%}")
        print(f"月均单边换手: {ret['换手'].mean():.2%}")

        ret['年份'] = ret.index.map(lambda x: str(x)[:4])
        years = sorted(set(ret['年份']))
        cols = ['组合收益', '等权收益', '基准收益', '超额(vs等权)', '组合回撤', '夏普'] \
            + [s + '收益' for s in style_names]
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
        filename = os.path.join(self.data_path, 'rotation_val_backtest.xlsx')
        excel = WriteExcel(filename)

        ws = excel.add_worksheet('回测结果')
        fmt = pd.DataFrame([], columns=ret.columns, index=['format'])
        for c in ret.columns:
            fmt.loc['format', c] = '@' if c == '年份' else '0.0%'
        excel.write_pandas(ret, ws, begin_row_number=0, begin_col_number=1,
                           num_format_pd=fmt, color='gray', fillna=True)

        ws2 = excel.add_worksheet('信号明细')
        fmt2 = pd.DataFrame([], columns=signals.columns, index=['format'])
        for c in signals.columns:
            fmt2.loc['format', c] = '0.0%'
        excel.write_pandas(signals, ws2, begin_row_number=0, begin_col_number=1,
                           num_format_pd=fmt2, color='gray', fillna=True)

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

    model = RotationValModel()
    model.backtest()
