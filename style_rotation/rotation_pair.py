# -*- coding: utf-8 -*-
"""两腿风格轮动（红利/成长对）：估值价差硬切换 + 拥挤度减半（2026-10-09 用户裁定豁免准入）

链路位置：
    1. 独立两腿体系（红利 vs 成长），与四腿主线 rotation_comb 互补
       （月度超额相关 -0.07，跨体系正交）
    2. 价差信号复用 rotation_crowd.calc_val_growth_spread（行业内 BP
       top20%/bottom20% 均值比 → 1200 交易日滚动 Z，缓存
       val_growth_spread_daily.csv，缺失自动重算）
    3. 拥挤度减半复用 rotation_comb.build_crowding（腿级三标尺 60 月 Z）

信号（月末，全先验无调参）：
    价差方向：score > 0 看好价值 → 超配红利；score < 0 看好成长
    （阶段九用户指令口径，硬切换 50/50 ± tilt 0.30——两腿截面 1 个自由度，
    tilt 为四腿 0.15 的两倍）
    拥挤度风控：score > 1.5 的腿权重 ×0.5 后归一（rotation_comb 定版规则）

回测（费后，单边千一）：
    2011-07~2026-09 共 183 月：超额 +3.21%/年、t=1.37、胜月 52%、
    月换手 8.8%、最大单年占正向和 23%；三段 IS13-20 +1.72%/胜月 53%｜
    验证 21-23 +6.76%/胜月 58%｜OOS 24+ +6.27%/胜月 48%
    （用户 2026-10-09 裁定豁免 OOS 胜率 48% 差 2 个点，接受为生产策略）

输出：
    ./data/output/rotation_pair_backtest.xlsx
    （月度明细 + 分年度两 sheet）+ 控制台最新信号
"""
import os

import numpy as np
import pandas as pd

from config.config import Config
from utility.write_excel import WriteExcel

try:
    from model_allocation.style_rotation.rotation_comb import (
        load_style_data, build_crowding)
    from model_allocation.style_rotation.rotation_crowd import (
        calc_val_growth_spread)
except ImportError:
    from rotation_comb import load_style_data, build_crowding
    from rotation_crowd import calc_val_growth_spread

TILT = 0.30
CROWD_TH = 1.5
LEGS = ['红利股', '成长股']
COST = 0.003         # 单边千三（2026-10-10 用户定规：style_rotation 全族统一费后口径，原千一）
DATA_DIR = Config().data_dir("derived_data", "allocation_data", "rotation_pair")


def month_prices(style_data):
    """两腿月末价格（StyleModel 补齐链，2005 起），index=月 Period"""
    p = style_data[LEGS].copy()
    p.index = pd.PeriodIndex(p.index.astype(str), freq='M')
    return p.groupby(level=0).last().dropna(how='all')


def build_weights(mp):
    """月末价差方向 + 拥挤度减半 → {持有月: 权重 Series}"""
    spread = calc_val_growth_spread(use_cache=True)['score'].dropna()
    spread.index = pd.PeriodIndex(spread.index.astype(str), freq='M')
    me = spread.groupby(level=0).last()          # 月末价差
    crowd = build_crowding()                     # 腿级拥挤度（月度 60 月 Z）
    weights = {}
    for p_sig in me.index:
        p_hold = p_sig + 1
        s = me.loc[p_sig]
        w = pd.Series({LEGS[0]: 0.5 + TILT * np.sign(s),
                       LEGS[1]: 0.5 - TILT * np.sign(s)})
        if p_sig in crowd.index:
            hot = crowd.loc[p_sig][LEGS].astype(float) > CROWD_TH
            for leg in LEGS:
                if hot.get(leg, False):
                    w[leg] *= 0.5
        if w.sum() > 0:
            weights[p_hold] = w / w.sum()
    return weights, me, crowd


def run_backtest(weights, mp, cost=COST):
    """持有月收益（费后）：组合 vs 两腿等权"""
    rows, prev_w = [], None
    for p in sorted(weights):
        if p not in mp.index:
            continue
        w = weights[p]
        r = mp.pct_change().loc[p, list(w.index)]
        if r.isna().any():
            continue
        port = float((w * r).sum())
        eqw = float(r.mean())
        turn = 0.5 * float(np.abs(w - prev_w).sum()) if prev_w is not None else 0.0
        prev_w = w
        rows.append({'ind': str(p), '组合收益': port - cost * turn * 2,
                     '等权收益': eqw, '超额': port - eqw - cost * turn * 2,
                     '换手': turn,
                     'w_红利': round(float(w[LEGS[0]]), 4),
                     'w_成长': round(float(w[LEGS[1]]), 4)})
    res = pd.DataFrame(rows).set_index('ind')
    res['年份'] = [d[:4] for d in res.index]
    return res


def report(res, name):
    ex = res['超额'].dropna()
    ann = (1 + res['组合收益']).prod() ** (12 / len(res)) - 1
    anne = (1 + res['等权收益']).prod() ** (12 / len(res)) - 1
    t = ex.mean() / ex.std(ddof=1) * np.sqrt(len(ex))
    print("%s(费后): %s~%s %d月 | 年化 %.2f%% vs 等权 %.2f%% | 超额 %.2f%%/年 | "
          "t=%.2f | 胜月 %.0f%% | 月换手 %.1f%%"
          % (name, res.index[0].replace('-', ''), res.index[-1].replace('-', ''),
             len(res), ann * 100, anne * 100, (ann - anne) * 100, t,
             (ex > 0).mean() * 100, res['换手'].mean() * 100))
    for tag, a, b in [('IS13-20', '2013', '2020'), ('VAL21-23', '2021', '2023'),
                      ('OOS24+', '2024', '2099')]:
        g = res[(res['年份'] >= a) & (res['年份'] <= b)]
        if not len(g):
            continue
        pa = (1 + g['组合收益']).prod() ** (12 / len(g)) - 1
        pe = (1 + g['等权收益']).prod() ** (12 / len(g)) - 1
        print("  %s: %+.2f%%/年 胜月%.0f%%" % (tag, (pa - pe) * 100,
                                               (g['超额'] > 0).mean() * 100))
    yearly = res.groupby('年份').apply(lambda g: pd.Series(
        {'超额': (1 + g['组合收益']).prod() - (1 + g['等权收益']).prod()}))
    print((yearly['超额'] * 100).round(2).to_string())
    return yearly


def main():
    sm, style_data = load_style_data()
    mp = month_prices(style_data)
    weights, me, crowd = build_weights(mp)

    res = run_backtest(weights, mp)
    yearly = report(res, '两腿轮动(价差+拥挤度减半)')

    # 最新信号
    p_last = me.index[-1]
    c_last = crowd.loc[p_last][LEGS] if p_last in crowd.index else None
    sig = '红利' if me.loc[p_last] > 0 else '成长'
    print("\n最新信号（%s 月末）：价差 score=%+.2f → 看好%s；"
          "拥挤度 %s；下月权重 %s"
          % (p_last, me.loc[p_last], sig,
             ('；'.join("%s %+.2f%s" % (l, c_last[l], '(过热×0.5)'
                                       if c_last[l] > CROWD_TH else '')
                        for l in LEGS) if c_last is not None else '缺失'),
             weights.get(p_last + 1).round(3).to_dict()
             if p_last + 1 in weights else '—'))

    os.makedirs(DATA_DIR, exist_ok=True)
    f = os.path.join(DATA_DIR, "rotation_pair_backtest.xlsx")
    excel = WriteExcel(f)
    ws = excel.add_worksheet('回测结果')
    excel.write_pandas(res, ws, begin_row_number=0, begin_col_number=0,
                       num_format_pd=None, color="gray", fillna=True)
    ws2 = excel.add_worksheet('分年度')
    excel.write_pandas(yearly, ws2, begin_row_number=0, begin_col_number=0,
                       num_format_pd=None, color="gray", fillna=True)
    excel.close()
    print("\n结果保存:", f)


if __name__ == "__main__":
    main()
