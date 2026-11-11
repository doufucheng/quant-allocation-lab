# -*- coding: utf-8 -*-
"""风格轮动最终策略：动量+估值综合 + 拥挤度减半剔除（2026-10-09 用户定版）

【定版】三变体检验（readme.md 研究阶段简表·阶段十）后，最终采用 +拥挤度减半
（score>1.5 权重×0.5）：152 月超额 +3.17%/年、t=3.12、胜月 58%、月换手 4.8%
（费前）；清零版收益更高（+4.12%）但 t 降至 2.40、换手翻倍，弃用。

信号（月末截面）：
    动量 z = 各腿 12 月收益 − 基准(h00985)12 月收益 → 截面 z（rotation_mom 同款）
    估值 z = 各腿 PE 60 月分位 → (0.5−pct)×2（rotation_val 同款，含代理补齐链）
    combo = (动量z + 估值z)/2 → w = 1/4 + 0.15×combo，clip≥0 后归一

拥挤度剔除（腿级三标尺，长历史独立计算，阶段四同口径）：
    成分链：红利 h30269 / 成长 399370 / 质量 930860 / 价值 399371
    腿成分个股 to/vol/beta（60 日）腿内均值 ÷ 全市场均值 → 60 月滚动 Z →
    三标尺均值 = score；score > 1.5 → 该腿权重 ×0.5 后归一（缺失月不剔除）

输出：allocation_data/rotation_comb/rotation_comb_backtest.xlsx
"""
import os

import numpy as np
import pandas as pd

from config.config import Config
from utility.date import Date
from utility.write_excel import WriteExcel

try:
    from model_allocation.style_rotation.style_model import StyleModel
    from model_allocation.style_rotation.rotation_val import RotationValModel
except ImportError:
    from style_model import StyleModel
    from rotation_val import RotationValModel

DATA_DIR = Config().data_dir("derived_data", "allocation_data", "rotation_comb")
TILT = 0.15
CROWD_WINDOW_M = 60
CROWD_MIN_M = 36
CROWD_TH = 1.5
FEE = 0.003  # 2026-10-10 用户定规：style_rotation 全族统一单边千三费后口径
LEG_NAMES = ['红利股', '成长股', '质量股', '价值股']
CROWD_MEMBER = {'红利股': 'h30269.CSI', '成长股': '399370.SZ',
                '质量股': '930860.CSI', '价值股': '399371.SZ'}


def load_style_data():
    """四腿+基准价格（StyleModel 补齐链），列名=腿名，index=YYYYMMDD str"""
    sm = StyleModel()
    all_codes = list(sm.style_index.values()) + list(sm.fill_index.values()) \
        + [sm.benchmark_code]
    data = sm.fill_price_data(sm.load_index_data(list(set(all_codes))))
    return sm, data.rename(columns={v: k for k, v in sm.style_index.items()})


def build_signals(month_ends, style_data, sm):
    """动量 z 与估值 z（月末截面）"""
    vr = RotationValModel()
    val = vr.load_index_valuation()          # {腿: Series(index=月Period, PE)}
    rows = {}
    for i in range(12, len(month_ends)):
        d_now, d_past = month_ends[i], month_ends[i - 12]
        row = {}
        bench12 = style_data.loc[d_now, sm.benchmark_code] \
            / style_data.loc[d_past, sm.benchmark_code] - 1
        mom = pd.Series({
            s: style_data.loc[d_now, s] / style_data.loc[d_past, s] - 1 - bench12
            for s in LEG_NAMES}).dropna()
        if len(mom) >= 2:
            z = (mom - mom.mean()) / mom.std(ddof=0)
            for s in z.index:
                row['momz_' + s] = z[s]
        for s in LEG_NAMES:
            series = val.get(s)
            p = pd.Period(d_now[:6], 'M')
            pct = np.nan
            if series is not None and len(series):
                hist = series.loc[:p]
                cur = hist.iloc[-1] if len(hist) else np.nan
                window = hist.iloc[:-1].iloc[-vr.window_m:]
                if pd.notna(cur) and len(window) >= vr.min_periods:
                    pct = float((window < cur).mean())
            row['valz_' + s] = (0.5 - pct) * 2.0 if pd.notna(pct) else np.nan
        rows[d_now] = row
    return pd.DataFrame(rows).T


def build_crowding():
    """腿级三标尺拥挤度 score（月末，60 月滚动 Z）"""
    import pyarrow.dataset as ds
    cfg = Config()
    beg = pd.Timestamp('2009-01-01')
    se = ds.dataset(str(cfg.db_wind_path / "StockEOD.pq")).to_table(
        filter=ds.field('Date') >= beg,
        columns=['Date', 'Ticker', 'FRTO', 'AdjClose']).to_pandas()

    def _w(col):
        x = se[col].unstack('Ticker')
        return x[x.index >= beg]
    frto, px = _w('FRTO'), _w('AdjClose')

    pct = px.pct_change(fill_method=None)
    to_m = frto.rolling(60, min_periods=40).mean()
    vol_m = pct.rolling(60, min_periods=40).std() * np.sqrt(250)
    mkt = pct.mean(axis=1)
    xm = pct.sub(mkt, axis=0)
    cov_m = xm.mul(mkt, axis=0).rolling(60, min_periods=40).mean() \
        - xm.rolling(60, min_periods=40).mean() * mkt.rolling(60, min_periods=40).mean()
    beta_m = cov_m / mkt.rolling(60, min_periods=40).var()

    panels = {k: p.groupby(p.index.to_period('M')).last()
              for k, p in [('to', to_m), ('vol', vol_m), ('beta', beta_m)]}

    # 腿成分（IndexWeights 月末权重>0）
    iw = ds.dataset(str(cfg.db_wind_path / "IndexWeights.pq")).to_table(
        filter=ds.field('ITicker').isin(list(CROWD_MEMBER.values()))).to_pandas()
    iw = iw[iw['value'] > 0]
    members = {}
    for leg, code in CROWD_MEMBER.items():
        sub = iw[iw.index.get_level_values('ITicker') == code]
        dates_p = sub.index.get_level_values('Date').to_period('M')
        tickers = sub.index.get_level_values('Ticker')
        members[leg] = {p: set(tickers[dates_p == p]) for p in sorted(set(dates_p))}

    # 腿内均值 ÷ 全市场均值
    ratio_rows = {}
    for p in panels['to'].index:
        row = {}
        for leg in LEG_NAMES:
            ms = members[leg].get(p)
            if not ms:
                continue
            for k, pan in panels.items():
                x = pan.loc[p].dropna()
                legs_x = x[x.index.intersection(list(ms))]
                if len(legs_x) >= 10 and x.mean() not in (0, np.nan):
                    row[f'{k}_{leg}'] = legs_x.mean() / x.mean()
        if row:
            ratio_rows[p] = row
    ratio = pd.DataFrame(ratio_rows).T.sort_index()

    # 60 月滚动 Z，三标尺均值
    score = pd.DataFrame(index=ratio.index)
    for leg in LEG_NAMES:
        zs = []
        for k in ['to', 'vol', 'beta']:
            x = ratio.get(f'{k}_{leg}')
            if x is None:
                continue
            mu = x.rolling(CROWD_WINDOW_M, min_periods=CROWD_MIN_M).mean()
            sd = x.rolling(CROWD_WINDOW_M, min_periods=CROWD_MIN_M).std()
            zs.append((x - mu) / sd)
        if zs:
            score[leg] = pd.concat(zs, axis=1).mean(axis=1)
    print("拥挤度 score 可用月：", {c: int(score[c].notna().sum()) for c in score})
    return score


def run_backtest(signals, style_data, score, overlay='B'):
    """overlay: 'B'减半（定版默认）/ 'C'对照 / 'A'清零"""
    idx = style_data.index
    month_ends = signals.index.tolist()
    rows = []
    prev_w = None
    for i in range(len(month_ends) - 1):
        me_prev, me_next = month_ends[i], month_ends[i + 1]
        sig = signals.loc[me_prev]
        combo = {s: (sig['momz_' + s] + sig['valz_' + s]) / 2.0
                 for s in LEG_NAMES
                 if pd.notna(sig.get('momz_' + s)) and pd.notna(sig.get('valz_' + s))}
        if not combo:
            continue
        w = pd.Series({s: 1.0 / len(combo) + TILT * c for s, c in combo.items()}
                      ).clip(lower=0.0)
        # 拥挤度剔除（缺失月视为未过热）
        p_prev = pd.Timestamp(me_prev).to_period('M')
        if score is not None and p_prev in score.index:
            hot = score.loc[p_prev].astype(float) > CROWD_TH
            hot = hot[hot].index.intersection(w.index)
            if overlay == 'A':
                w[hot] = 0.0
            elif overlay == 'B':
                w[hot] = w[hot] * 0.5
        if w.sum() <= 0:
            w = pd.Series(1.0, index=list(combo))
        w = w / w.sum()
        turn = 0.5 * float(np.abs(w - prev_w).sum()) if prev_w is not None else 0.0
        prev_w = w
        # 持有期：上月末次日~本月末（index 为 YYYYMMDD str，字典序即时间序）
        seg = style_data.loc[(idx > str(me_prev)) & (idx <= str(me_next))]
        if not len(seg):
            continue
        port_gross = sum(w[s] * (seg[s].iloc[-1] / seg[s].iloc[0] - 1) for s in w.index)
        port = port_gross - turn * FEE  # 费后（2026-10-10 全族统一口径）
        eqw = np.mean([seg[s].iloc[-1] / seg[s].iloc[0] - 1 for s in LEG_NAMES])
        rows.append({'ind': me_next, '组合收益': port, '等权收益': eqw, '换手': turn,
                     **{'w_' + s: round(w.get(s, 0.0), 4) for s in LEG_NAMES}})
    res = pd.DataFrame(rows).set_index('ind')
    res['超额'] = res['组合收益'] - res['等权收益']
    res['年份'] = [d[:4] for d in res.index]
    return res


def report(res, name):
    ex = res['超额'].dropna()
    ann = (1 + res['组合收益']).prod() ** (12 / len(res)) - 1
    anne = (1 + res['等权收益']).prod() ** (12 / len(res)) - 1
    t = ex.mean() / ex.std(ddof=1) * np.sqrt(len(ex))
    print("%s: %s~%s %d月 | 年化 %.2f%% vs 等权 %.2f%% | 超额 %.2f%%/年 | "
          "t=%.2f | 胜月 %.0f%% | 月换手 %.1f%%"
          % (name, res.index[0], res.index[-1], len(res), ann * 100, anne * 100,
             (ann - anne) * 100, t, (ex > 0).mean() * 100, res['换手'].mean() * 100))
    yearly = res.groupby('年份').apply(lambda g: pd.Series({
        '超额': (1 + g['组合收益']).prod() - (1 + g['等权收益']).prod()}))
    print((yearly['超额'] * 100).round(2).to_string())
    return yearly


if __name__ == "__main__":
    sm, style_data = load_style_data()
    month_ends = Date().get_trade_date_series(
        '20130131', style_data.index[-1], 'M')
    signals = build_signals(month_ends, style_data, sm)
    print("信号面板：", signals.shape)
    score = build_crowding()

    os.makedirs(DATA_DIR, exist_ok=True)
    f = os.path.join(DATA_DIR, "rotation_comb_backtest.xlsx")
    excel = WriteExcel(f)
    for ov, name in [('B', '最终策略'), ('C', '对照')]:
        res = run_backtest(signals, style_data, score, ov)
        yearly = report(res, name + ('(+拥挤度减半)' if ov == 'B' else '(动量+估值)'))
        ws = excel.add_worksheet(name)
        excel.write_pandas(res, ws, begin_row_number=0, begin_col_number=0,
                           num_format_pd=None, color="gray", fillna=True)
        ws2 = excel.add_worksheet(name + '年')
        excel.write_pandas(yearly, ws2, begin_row_number=0, begin_col_number=0,
                           num_format_pd=None, color="gray", fillna=True)
    excel.close()
    print("\n结果保存:", f)
