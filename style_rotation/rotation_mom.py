# -*- coding: utf-8 -*-
"""
风格动量轮动模型（IndexEOD 近12月超额收益 → 风格指数月度轮动，2026-08-31）

一、策略概述
============================================================
各风格腿指数近 lookback_m 个月相对基准（中证全指全收益）的超额收益做
截面 z-score，强者超配：权重 = 等权 + tilt × z，截断非负后归一化。

回看期敏感性（2026-08-31，2013-01~2026-08 月度费前，tilt=0.15，
年化超额 vs 四腿等权 / 月均单边换手）：

  lookback   1月      2月      3月      6月     12月
  正向     +0.57%  -1.34%   -1.35%   +1.73%  +4.49%
  反向     -1.05%  +0.82%   +0.83%   -2.24%  -4.86%
  换手      31.8%   24.8%    19.8%    14.2%    7.9%

期限结构清晰：短端（2-3月）为均值回复区（负超额），中长端（6-12月）趋势
延续，**12个月动量超额+4.49%、胜率54.6%、换手仅7.9%**（费前；千1单边成本
约0.95%/年，费后约+3.5%）。A股风格指数在月度频率上短端均值回复（与
rotation_alpha 归因中四腿月收益一阶自相关 -0.11~-0.36 一致）、12个月维度
趋势延续；12月动量为经典先验（Jegadeesh-Titman 1993；Asness 风格动量），
期限结构单调、非样本内挖掘形状。

二、风格腿（与 style_model.py 完全一致，直接复用其数据链路）
============================================================
  红利股：中证红利低波全收益（h20269.CSI）
  成长股：国证成长100（980080.CNI，价格版）
  质量股：中证质量全收益（932430CNY010.CSI）
  价值股：国证价值100全收益（480081.CNI）
  基准：中证全指全收益（h00985.CSI）
  回测起点 2006-01-31（2026-09-25 按回测时长原则自 2013 延长：腿经 StyleModel
  补齐链回补到 2005-01 数据下限，12月动量信号预热后首个调仓月 2006-01；
  2006-01~2012-12 段信号基于代理腿口径——国证成长/国证价值R/全指消费，
  2013-01 起为主指数真实数据）。

三、关键参数
============================================================
  lookback_m = 12   动量回看月数（敏感性见上表，短回看换手高且超额弱）
  tilt       = 0.15 z-score → 权重倾斜系数
  月度再平衡费前

四、更新日志
============================================================
  2026-08-31 初始版本（lag-1动量，依据 style_rotation_research.md；
             回测2013-01~2026-08：年化15.74%/等权15.17%，超额+0.57%、
             月换手31.8%，费后为负——A股四腿月收益均值回复所致，弃用）
  2026-08-31 回看期敏感性检验（1/2/3/6/12月×正/反向）后改为12个月：
             超额+0.57%→+4.49%、换手31.8%→7.9%（代码仅保留12月版，
             敏感性结论见上表）

Author: Fucheng Dou
日期：2026-08-31
"""

import os
import pandas as pd
import numpy as np

from utility.write_excel import WriteExcel
from utility.date import Date
from config.config import Config

try:
    from model_allocation.style_rotation.style_model import StyleModel
except ImportError:
    from style_model import StyleModel


class RotationMomModel(object):
    """
    风格动量轮动模型
    各腿近12个月相对基准超额收益截面z-score → 强者超配 → 月度轮动
    （回看期敏感性见模块docstring：2-3月为均值回复区，12月趋势延续）
    """

    def __init__(self):
        self.data_path = Config().data_dir("derived_data", "allocation_data", "rotation_mom")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        # 风格腿（与 style_model.py 完全一致）
        self.style_index = {
            '红利股': 'h20269.CSI',      # 中证红利低波全收益
            '成长股': '980080.CNI',      # 国证成长100（价格版）
            '质量股': '932430CNY010.CSI',  # 中证质量全收益
            '价值股': '480081.CNI',      # 国证价值100全收益
        }
        self.benchmark_code = 'h00985.CSI'      # 中证全指全收益

        # 参数
        self.lookback_m = 12       # 动量回看月数（敏感性：1月+0.57%/2月-1.34%/3月-1.35%/6月+1.73%/12月+4.49%）
        self.tilt = 0.15           # z-score → 权重倾斜系数
        self.backtest_date = '20060131'   # 2026-09-25 按回测时长原则自 2013 延长：
                                          # 腿经补齐链回补到 2005-01，12月动量信号
                                          # 预热后首个调仓月为 2006-01；
                                          # 2006~2012 段信号基于代理腿口径

        print("\n" + "=" * 80)
        print("风格动量轮动模型 - 参数配置")
        print("=" * 80)
        print(f"动量回看期：{self.lookback_m}个月（超额收益相对基准，强者超配）")
        print(f"倾斜系数 tilt：{self.tilt}（强者超配，权重=等权+tilt×z）")
        print(f"再平衡频率：月度（费前）")
        print(f"风格腿：{self.style_index}")
        print(f"基准：{self.benchmark_code}")
        print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # 信号与回测
    # ------------------------------------------------------------------
    def compute_signal_row(self, style_data, date_series, i):
        """
        第 i 个月末的动量信号与权重

        各腿近 lookback_m 个月相对基准超额收益 → 截面z-score → 权重
        """
        style_names = list(self.style_index.keys())
        d_now = date_series[i]
        d_past = date_series[max(0, i - self.lookback_m)]
        row = {}

        if self.benchmark_code in style_data.columns \
                and d_past in style_data.index and d_now in style_data.index:
            bench_ret = style_data.loc[d_now, self.benchmark_code] \
                / style_data.loc[d_past, self.benchmark_code] - 1
        else:
            bench_ret = np.nan

        for s in style_names:
            code = self.style_index[s]
            if code in style_data.columns and d_past in style_data.index \
                    and pd.notna(style_data.loc[d_past, code]) \
                    and pd.notna(style_data.loc[d_now, code]) \
                    and pd.notna(bench_ret):
                leg_ret = style_data.loc[d_now, code] / style_data.loc[d_past, code] - 1
                row[f'{s}动量'] = leg_ret - bench_ret
            else:
                row[f'{s}动量'] = np.nan

        moms = pd.Series({s: row.get(f'{s}动量', np.nan) for s in style_names})
        valid = moms.dropna()
        weights = pd.Series(0.0, index=style_names)
        if len(valid) <= 1:
            weights.loc[:] = 1.0 / len(style_names)
        else:
            sd = valid.std(ddof=0)
            z = (valid - valid.mean()) / sd if sd > 0 else valid * 0.0
            w = (1.0 / len(valid) + self.tilt * z).clip(lower=0.0)
            weights.loc[valid.index] = w / w.sum()
        for s in style_names:
            row[f'{s}权重'] = weights[s]
        return row

    def backtest(self):
        """回测：月末动量信号 → 下月首个交易日成交 → 月度收益（费前）"""
        print("\n" + "=" * 80)
        print("开始回测风格动量轮动策略")
        print("=" * 80)

        # 风格指数价格（复用 StyleModel：加载 + 历史回补）
        sm = StyleModel()
        all_codes = list(self.style_index.values()) + list(sm.fill_index.values()) \
            + [self.benchmark_code]
        style_data = sm.fill_price_data(sm.load_index_data(list(set(all_codes))))

        month_ends = Date().get_trade_date_series(
            self.backtest_date, style_data.index[-1], 'M')
        print(f"回测期间: {month_ends[0]} 至 {month_ends[-1]}，"
              f"共 {len(month_ends)} 个调仓时点")

        style_names = list(self.style_index.keys())
        sig_rows = {}
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

            # 动量信号（本月末，用过去lookback_m个月收益）
            row = self.compute_signal_row(style_data, month_ends, i)
            sig_rows[date] = row
            w = pd.Series({s: row.get(f'{s}权重', np.nan) for s in style_names},
                          dtype=float)
            if w.isna().all():
                w = pd.Series(1.0 / len(style_names), index=style_names)

            period_ret = {}
            for s in style_names:
                code = self.style_index[s]
                if code in style_data.columns:
                    buy, sell = style_data.loc[buy_date, code], style_data.loc[sell_date, code]
                    period_ret[s + '收益'] = sell / buy - 1 \
                        if pd.notna(buy) and pd.notna(sell) else np.nan
                else:
                    period_ret[s + '收益'] = np.nan

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

        signals = pd.DataFrame(sig_rows).T
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

        print("\n====== 风格动量轮动绩效 ======")
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
        filename = os.path.join(self.data_path, 'rotation_mom_backtest.xlsx')
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
            fmt2.loc['format', c] = '0.0%' if '权重' in c else '0.00%'
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

    model = RotationMomModel()
    model.backtest()
