# -*- coding: utf-8 -*-
"""
美债利率风格轮动模型（原 rotation_rate.md 说明已并入本注释，2026-08-29）

一、策略概述
============================================================
美国10年期国债利率作为全球资产定价的锚，其变化对A股风格轮动有显著影响。
利率指标 Z = -(当月利率 - 过去18月均值) / 过去18月标准差：

  利率下行趋势（Z > 0.7）：风险偏好提升，超配成长股+质量股（各50%）
  利率上行趋势（Z < -0.7）：风险偏好下降，超配红利股+价值股（各50%）
  利率无趋势（-0.7 < Z < 0.7）：等权配置各25%

风格指数（对齐 style_model.py 四风格腿体系）：
  红利股：中证红利低波全收益（h20269.CSI，2005-12起）
  成长股：国证成长100（980080.CNI，价格版，2012-12起）
  质量股：中证质量全收益（932430CNY010.CSI，2010-12起）
  价值股：国证价值100全收益（480081.CNI，2012-12起）
  基准：中证全指全收益（h00985.CSI）

二、利率指标计算（对照the reference spreadsheet (equity-bond valuation sheet)）
============================================================
Excel公式（第20行起，G列=美债10Y月均值）：
  =-(G20-AVERAGE(G3:G20))/STDEV.P(G3:G20)   ← 18个月滚动窗口
负号含义：Z值越大表示利率相对越低（越应超配成长）。
Python实现见 calculate_rate_indicator()（rolling 18月均值/标准差）。

三、轮动信号（对照Excel「利率-风格」工作表）
============================================================
  Q列(红利): =IF(Z<-0.7, 0.5, IF(Z>0.7, 0, 0.25))
  R列(价值): =IF(Z> 0.7, 0.5, IF(Z<-0.7, 0, 0.25))
  质量跟随价值、成长跟随红利（Excel实现），本模型按PPT语义实现：
  下行→成长+质量各50%，上行→红利+价值各50%，中性→等权25%×4。
  （Excel实际实现与PPT描述略有差异，此处按PPT逻辑）

四、数据链路
============================================================
美债利率（load_us_treasury_rate 多源自动降级，成功后落盘
E:\\Data\\allocation_data\\rotation_rate\\us_10y_treasury_rate.csv）：
  本地CSV → 代码目录CSV → Excel(Wind EDB Y列) → Fred API(GS10) → WindPy(G0000888)
每日调度 _refresh_rate_data：只用 Wind API（WindPy G0000888），
  需启动 Wind 终端；失败回退本地 CSV。
  （db_wind 离线源 G0000891 已删：海外指标同步 2025-06 停止，无最新值）
风格指数：StyleModel 四腿补齐链（IndexEOD.pq，与全族统一；回测起点 2014
不涉及补齐段）。
注意：db_wind/TreasuryYield.pq 是中国国债数据，不可用于美债！

五、关键参数
============================================================
  lookback_period = 18   回看期（月），Excel 18个月滚动窗口
  threshold       = 0.7  Z值阈值（Excel P2单元格）
  rebalance_freq  = 'M'  月度再平衡
敏感性（PPT第12页）：月度换仓下阈值0.7最优；阈值越小超额波动越大；
不论参数组合模型始终有效，失效年份一致。

六、历史回测失效年份（风险提示）
============================================================
  （2014 前历史，仅作参考，已不在正式回测区间）
  2006-2007：股改牛市，国内价值红利领涨，美债传导链失效（超额-26%/-57%）
  2009：四万亿刺激，价值股V型反转（超额-34%）
  正式区间（2014 起）失效年份：
  2015：杠杆牛风格极端（-3.6%）
  2025：美联储降息周期（-7.6%）
数据风险：美债月度数据需及时更新；汇率波动影响外资流向判断；
政策变化可能改变利率传导机制。
回测基准（2026-09-25 腿体系定版后，2014-02~2026-09 共152月，费前月度）：
轮动年化 10.83%/夏普 0.53 vs 等权 10.44%/0.54，超额 +0.40%/年，t=0.20。
（2026-09-02 旧腿体系数字 16.16% vs 10.02%、超额 +6.14% 已作废；
 2005-2026 全区间旧结果：轮动 7.42% vs 等权 11.29%，超额 -3.87%，
 负超额全部来自 2006/2007/2009 三年）

七、参考文献 / 更新日志
============================================================
1. internal rate-rotation research deck (2026-03)
2. the reference spreadsheet
3. Fred: https://fred.stlouisfed.org/series/GS10
日志：
  2026-03-18 初始版本（Excel+PPT梳理计算逻辑）
  2026-08-29 prepare_rate_data.py 融合为多源自动加载；
            风格腿对齐 style_model.py 腿体系（2026-09-25 定版）；
            说明文档并入本注释，总览见模块 readme.md
  2026-09-02 弃用 WindPy 在线源（G0000888），_refresh_rate_data 改走
            db_wind 离线库（ZSJJEDB G0000891），无需启动 Wind 终端；
            同日发现离线库海外指标同步 2025-06-14 停止（美债滞后~15月），
            最终决策：只用 Wind API（WindPy G0000888），删除 db_wind
            源与 rate_source 开关，load_us_treasury_yield 已从 db_wind.py 删除
            回测起点 2005 → 2014（负超额集中在前沪港通时代，
            美债→A股风格传导 2014 年起才稳定成立）

Author: Fucheng Dou
日期：2026年3月 / 2026-08-29重构
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from utility.write_excel import WriteExcel
from utility.date import Date
from config.config import Config

try:
    from model_allocation.style_rotation.style_model import StyleModel
except ImportError:
    from style_model import StyleModel


class RotationRateModel(object):

    """
    美债利率风格轮动模型
    """

    @staticmethod
    def _drop_incomplete_current_month(df):
        """正式信号只保留已结束月份，避免把月内值标成月末值。"""
        if df is None or df.empty:
            return df
        today = pd.Timestamp.today().normalize()
        current_ym = today.strftime('%Y%m')
        idx = pd.Index(df.index.map(str))
        return df.loc[idx.str[:6] != current_ym].copy()

    def __init__(self):
        self.data_path = r"./data/output"
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
        # db_wind 数据库路径
        self.db_wind_path = Config().db_wind_path
        self.index_eod_path = self.db_wind_path / "IndexEOD.pq"
        
        # 风格指数（对齐 style_model.py 四风格腿体系）
        # 红利：红利低波全收益 h20269；成长：成长100 价格版 980080
        # 质量：中证质量全收益 932430CNY010（价格版股息差 3-4%/年）
        # 价值：国证价值100全收益 480081
        self.style_index = {
            '红利股': 'h20269.CSI',      # 中证红利低波全收益
            '成长股': '980080.CNI',      # 国证成长100（价格版）
            '质量股': '932430CNY010.CSI',  # 中证质量全收益
            '价值股': '480081.CNI',      # 国证价值100全收益
        }
        self.benchmark_code = 'h00985.CSI'  # 基准：中证全指全收益
        
        # 轮动参数
        self.lookback_period = 18  # 利率指标计算回看期（月）
        self.threshold = 0.7       # Z值阈值
        self.rebalance_freq = 'M'  # 月度再平衡
        
        # 数据起始日期
        # 2026-09-02 用户决策：回测起点从 2005 改为 2014。
        # 原因：美债利率→A股风格传导链在 2014 沪港通（外资定价权增强）后
        # 才稳定成立；2006/2007/2009 三年负超额（-26%/-57%/-34%）由国内
        # 因素主导的行情造成（股改牛市/四万亿），不代表模型当前有效性。
        # 2014-2026 轮动年化约9.9% vs 等权9.3%，波动18% vs 25%，回撤-33% vs -42%。
        self.load_beg_date = '20131231'
        self.backtest_date = '20140131'
        
        print("\n" + "=" * 80)
        print("美债利率风格轮动模型 - 参数配置")
        print("=" * 80)
        print(f"利率指标回看期：{self.lookback_period}月")
        print(f"Z值阈值：{self.threshold}")
        print(f"再平衡频率：{self.rebalance_freq}")
        print(f"回测起始日期：{self.backtest_date}")
        print("=" * 80 + "\n")
    
    def _load_api_wind_rate(self):
        """从 api_wind/macro_edb.pq 读美债10Y（G0000888_US10Y_D）月度序列。

        2026-09-22 新增（api_wind 补数体系，见 db_data_missing.md §3.11）：
        db_wind_api_windpy.py 全量刷新后的本地源，与策略内 WindPy 同指标；
        返回 YYYYMMDD 月末索引的 DataFrame(['利率'])，不可得返回 None。
        """
        try:
            me = pd.read_parquet(
                Config().api_wind_path / "macro_edb.pq").set_index("Date")
            s = me["G0000888_US10Y_D"].dropna()
        except Exception as e:
            print(f"api_wind 美债利率读取失败: {e}")
            return None
        if len(s) == 0:
            return None
        df = s.to_frame("利率")
        df.index = pd.to_datetime(df.index.astype(str).str[:10])
        current_ym = pd.Timestamp.today().strftime("%Y-%m")
        df = df.loc[df.index.strftime("%Y-%m") != current_ym]
        df = df.resample("ME").last().dropna()
        df.index = df.index.strftime("%Y%m%d")
        return df

    def load_us_treasury_rate(self):
        """
        加载美国10年期国债利率数据（月度，%）

        数据源优先级（自动降级，成功后落盘供下次直读）：
        1. 本地CSV (./data/output/us_10y_treasury_rate.csv)
           + api_wind 增鲜（macro_edb.pq G0000888_US10Y_D 更新则合并落盘）
        1b. api_wind macro_edb.pq（CSV 缺失时；db_wind_api_windpy.py 全量刷新）
        2. 代码目录CSV (style_rotation/us_10y_treasury_rate.csv，历史备份)
        3. Excel Wind EDB（the reference spreadsheet）
        4. Fred API (GS10，需 pip install pandas-datareader)
        5. WindPy (G0000888，需Wind终端)

        正式刷新走 _refresh_rate_data（api_wind 优先，WindPy G0000888 兜底）。

        Returns
        -------
        DataFrame
            利率数据，index为日期(YYYYMMDD月末)，columns为['利率']
        """

        def _save_rate(df):
            """在线源数据落盘（gbk，index=YYYYMMDD月末）"""
            df.to_csv(os.path.join(self.data_path, 'us_10y_treasury_rate.csv'), encoding='gbk')
            print(f"利率数据已落盘: {os.path.join(self.data_path, 'us_10y_treasury_rate.csv')}")

        def _to_monthly(df):
            """日/月度index → YYYYMMDD月末"""
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            # 当前月未结束时不写入正式月度序列；监控值不能冒充月末值。
            current_ym = pd.Timestamp.today().strftime('%Y-%m')
            df = df.loc[df.index.strftime('%Y-%m') != current_ym]
            df = df.resample('ME').last().dropna()
            df.index = df.index.strftime('%Y%m%d')
            return df

        # 方法1：本地CSV（优先）+ api_wind 增鲜
        rate_file = os.path.join(self.data_path, 'us_10y_treasury_rate.csv')
        if os.path.exists(rate_file):
            print(f"从本地CSV文件加载美债利率数据: {rate_file}")
            df = pd.read_csv(rate_file, index_col=0, encoding='gbk')
            df.index = df.index.map(str)
            df = self._drop_incomplete_current_month(df)
            print(f"成功加载 {len(df)} 条月度利率数据")
            print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
            # api_wind 增鲜：本地源更新则合并落盘（月末值口径一致，取并集）
            api_df = self._load_api_wind_rate()
            if api_df is not None and api_df.index[-1] > df.index[-1]:
                merged = pd.concat([df, api_df.loc[api_df.index > df.index[-1]]])
                merged.index = merged.index.map(str)
                _save_rate(merged)
                print(f"api_wind 增鲜至 {merged.index[-1]}（新增 {len(merged)-len(df)} 期）")
                return merged
            return df

        # 方法1b：api_wind macro_edb.pq（CSV 缺失时的本地源）
        api_df = self._load_api_wind_rate()
        if api_df is not None and len(api_df) > 0:
            print(f"从 api_wind 加载美债利率数据 (G0000888_US10Y_D)")
            print(f"成功加载 {len(api_df)} 条月度利率数据")
            print(f"时间范围: {api_df.index[0]} 至 {api_df.index[-1]}")
            _save_rate(api_df)
            return api_df

        # 方法2：代码目录CSV（历史备份）
        local_copy = Path(__file__).parent / 'us_10y_treasury_rate.csv'
        if local_copy.exists():
            print(f"从代码目录CSV加载美债利率数据: {local_copy}")
            df = pd.read_csv(local_copy, index_col=0, encoding='gbk')
            df.index = df.index.map(str)
            df = self._drop_incomplete_current_month(df)
            print(f"成功加载 {len(df)} 条月度利率数据")
            print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
            return df

        # 方法3：Excel Wind EDB
        excel_file = r'./data/reference/'
        if os.path.exists(excel_file):
            print(f"从Excel文件提取美债利率数据: {excel_file}")
            try:
                import openpyxl

                wb = openpyxl.load_workbook(excel_file, data_only=True)
                ws = wb['股债估值data']

                data_list = []
                for row_idx in range(4, 300):
                    date_val = ws.cell(row=row_idx, column=1).value
                    us_rate = ws.cell(row=row_idx, column=25).value
                    if date_val is not None and us_rate is not None:
                        data_list.append({'日期': date_val, '利率': us_rate})
                wb.close()

                if len(data_list) > 0:
                    df = pd.DataFrame(data_list).set_index('日期')
                    df = _to_monthly(df)
                    print(f"成功提取 {len(df)} 条月度利率数据")
                    print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
                    _save_rate(df)
                    return df
            except Exception as e:
                print(f"从Excel提取数据失败: {e}")

        # 方法5：Fred API（免费）
        try:
            from pandas_datareader import data as pdr

            print("从Fred API下载美债利率数据 (GS10)...")
            df = pdr.DataReader('GS10', 'fred', '2000-01-01',
                                datetime.now().strftime('%Y-%m-%d'))
            df.columns = ['利率']
            df = _to_monthly(df)
            print(f"成功下载 {len(df)} 条月度利率数据")
            print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
            _save_rate(df)
            return df
        except ImportError:
            print("Fred API不可用（未安装 pandas-datareader）")
        except Exception as e:
            print(f"Fred下载失败: {e}")

        # 方法6：WindPy
        try:
            from WindPy import w

            print("从Wind数据库加载美债利率数据 (G0000888)...")
            w.start()
            data = w.edb("G0000888", '2000-01-01',
                         datetime.now().strftime('%Y-%m-%d'), "Fill=Previous")
            if data.ErrorCode == 0:
                df = pd.DataFrame({'利率': data.Data[0]}, index=pd.to_datetime(data.Times))
                df = _to_monthly(df)
                print(f"成功加载 {len(df)} 条月度利率数据")
                print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
                _save_rate(df)
                return df
            print(f"Wind数据获取失败: ErrorCode={data.ErrorCode}")
        except ImportError:
            print("WindPy不可用（未安装Wind终端Python接口）")
        except Exception as e:
            print(f"Wind数据加载失败: {e}")

        # 未找到数据
        print("未找到美债利率数据（本地CSV/目录CSV/db_wind/Excel/Fred/Wind均失败）")
        print("数据来源：Wind EDB（美国:国债收益率:10年:月:中位数）或 Fred GS10")
        return None
    
    def calculate_rate_indicator(self, rate_data):
        """
        计算利率指标（Z-score）
        
        Z = -(当月利率 - 过去18月平均数) / 过去18月标准差
        
        Parameters
        ----------
        rate_data : DataFrame
            利率数据，index为日期（月末），columns包含'利率'
            
        Returns
        -------
        Series
            利率指标，index为日期
        """
        print("计算利率指标...")
        
        # 确保利率数据按日期排序
        rate_series = rate_data['利率'].sort_index()
        
        # 计算滚动平均和标准差（过去18个月）
        rolling_mean = rate_series.rolling(window=self.lookback_period, min_periods=self.lookback_period).mean()
        rolling_std = rate_series.rolling(window=self.lookback_period, min_periods=self.lookback_period).std()
        
        # 计算Z-score（注意是负号）
        z_score = -(rate_series - rolling_mean) / rolling_std
        
        return z_score
    
    def generate_signals(self, z_score):
        """三态权重：Z>0.7 利率下行→成长+质量；Z<-0.7 上行→红利+价值；否则等权"""
        cols = ['红利股', '价值股', '质量股', '成长股']
        weights = pd.DataFrame(0.25, index=z_score.index, columns=cols)
        weights.loc[z_score > self.threshold] = [0.0, 0.0, 0.5, 0.5]
        weights.loc[z_score < -self.threshold] = [0.5, 0.5, 0.0, 0.0]
        return weights
    
    def load_style_index_data(self):
        """四腿+基准价格（StyleModel 补齐链，与全族统一），列名=腿名+基准代码，
        index=YYYYMMDD str。回测起点 2014 > 全部主指数发布线，不涉及补齐段。"""
        sm = StyleModel()
        all_codes = list(self.style_index.values()) \
            + list(sm.fill_index.values()) + [self.benchmark_code]
        data = sm.fill_price_data(sm.load_index_data(list(set(all_codes))))
        data = data.rename(columns={v: k for k, v in self.style_index.items()})
        data = data.loc[self.load_beg_date:]
        print("风格指数（StyleModel 补齐链）: %s ~ %s, %d 条"
              % (data.index[0], data.index[-1], len(data)))
        return data
    
    def backtest(self, rate_data=None, style_data=None):
        """
        回测利率轮动策略
        
        Parameters
        ----------
        rate_data : DataFrame, optional
            利率数据
        style_data : DataFrame, optional
            风格指数价格数据
        """
        print("\n" + "=" * 80)
        print("开始回测美债利率风格轮动策略")
        print("=" * 80)
        
        # 加载数据
        if rate_data is None:
            rate_data = self.load_us_treasury_rate()
        
        if style_data is None:
            style_data = self.load_style_index_data()
        
        if rate_data is None or style_data is None:
            print("数据加载失败，无法进行回测")
            return None
        
        # 计算利率指标
        z_score = self.calculate_rate_indicator(rate_data)
        
        # 生成轮动信号
        weights = self.generate_signals(z_score)
        
        # 获取月末交易日序列
        date_series = Date().get_trade_date_series(self.backtest_date, style_data.index[-1], 'M')
        
        print(f"回测期间: {date_series[0]} 至 {date_series[-1]}")
        print(f"共 {len(date_series)} 个调仓时点")
        
        # 计算组合收益
        ret_dict = {}
        prev_w = None  # 2026-10-10 全族统一费后口径（单边千三）：跟踪换手扣费
        
        def next_trade_date(date):
            """信号收盘后首个可成交交易日。"""
            pos = style_data.index.searchsorted(date, side='right')
            return style_data.index[pos] if pos < len(style_data.index) else None

        for i in range(len(date_series) - 1):
            date = date_series[i]
            next_date = date_series[i + 1]
            buy_date = next_trade_date(date)
            sell_date = next_trade_date(next_date)
            
            # 检查日期是否存在
            if (date not in style_data.index or next_date not in style_data.index
                    or buy_date is None or sell_date is None):
                continue
            
            # 检查利率数据是否存在
            if date not in weights.index:
                continue

            # date 收盘后形成的权重，必须到下一交易日才成交。
            
            # 获取当前权重
            current_weights = weights.loc[date]
            
            # 计算各指数收益率
            period_ret = {}
            
            for style_name in ['红利股', '价值股', '质量股', '成长股']:
                if style_name in style_data.columns:
                    buy_price = style_data.loc[buy_date, style_name]
                    sell_price = style_data.loc[sell_date, style_name]
                    if pd.notna(buy_price) and pd.notna(sell_price):
                        period_ret[style_name + '收益'] = sell_price / buy_price - 1
                    else:
                        period_ret[style_name + '收益'] = np.nan
                else:
                    period_ret[style_name + '收益'] = np.nan
            
            # 计算组合收益率
            port_ret = 0.0
            for style_name in ['红利股', '价值股', '质量股', '成长股']:
                weight = current_weights[style_name]
                ret = period_ret.get(style_name + '收益', np.nan)
                if pd.notna(weight) and pd.notna(ret):
                    port_ret += weight * ret

            w_s = current_weights[['红利股', '价值股', '质量股', '成长股']].astype(float)
            turn_m = float(w_s.abs().sum()) if prev_w is None \
                else float(0.5 * (w_s - prev_w).abs().sum())
            period_ret['换手'] = turn_m
            prev_w = w_s
            period_ret['组合收益'] = port_ret - turn_m * 0.003  # 费后（2026-10-10 全族统一单边千三）
            
            # 计算等权组合收益（基准）
            equal_ret = np.mean([period_ret[sn + '收益'] for sn in ['红利股', '价值股', '质量股', '成长股'] 
                                if pd.notna(period_ret.get(sn + '收益', np.nan))])
            period_ret['等权收益'] = equal_ret
            
            # 存储结果
            ret_dict[next_date] = period_ret
        
        # 创建收益DataFrame
        port_ret = pd.DataFrame(ret_dict).T
        
        # 计算净值
        port_ret['轮动净值'] = (1 + port_ret['组合收益']).cumprod()
        port_ret['等权净值'] = (1 + port_ret['等权收益'].fillna(0)).cumprod()
        
        # 计算绩效指标
        self.perf = self.calculate_performance(port_ret)

        return port_ret
    
    def calculate_performance(self, ret):
        """
        计算组合绩效指标
        
        Parameters
        ----------
        ret : DataFrame
            收益率数据（月度）
        """
        # 年化收益率
        total_ret_rotation = ret['轮动净值'].iloc[-1] - 1
        total_ret_equal = ret['等权净值'].iloc[-1] - 1
        num_years = len(ret) / 12.0
        
        an_ret_rotation = (1 + total_ret_rotation) ** (1 / num_years) - 1
        an_ret_equal = (1 + total_ret_equal) ** (1 / num_years) - 1
        excess_ret = an_ret_rotation - an_ret_equal
        
        # 年化波动率
        an_vol_rotation = ret['组合收益'].std() * np.sqrt(12)
        an_vol_equal = ret['等权收益'].std() * np.sqrt(12)
        
        # 最大回撤
        ret['轮动回撤'] = ret['轮动净值'] / ret['轮动净值'].expanding().max() - 1
        ret['等权回撤'] = ret['等权净值'] / ret['等权净值'].expanding().max() - 1
        mdd_rotation = ret['轮动回撤'].min()
        mdd_equal = ret['等权回撤'].min()
        
        # 夏普比率
        sharpe_rotation = an_ret_rotation / an_vol_rotation if an_vol_rotation != 0 else np.nan
        sharpe_equal = an_ret_equal / an_vol_equal if an_vol_equal != 0 else np.nan
        
        print("\n====== 轮动策略绩效 ======")
        print(f"总收益率: {total_ret_rotation:.2%}")
        print(f"年化收益率: {an_ret_rotation:.2%}")
        print(f"年化波动率: {an_vol_rotation:.2%}")
        print(f"最大回撤: {mdd_rotation:.2%}")
        print(f"夏普比率: {sharpe_rotation:.2f}")
        
        print("\n====== 等权基准绩效 ======")
        print(f"总收益率: {total_ret_equal:.2%}")
        print(f"年化收益率: {an_ret_equal:.2%}")
        print(f"年化波动率: {an_vol_equal:.2%}")
        print(f"最大回撤: {mdd_equal:.2%}")
        print(f"夏普比率: {sharpe_equal:.2f}")
        
        print("\n====== 超额收益 ======")
        print(f"年化超额收益: {excess_ret:.2%}")
        
        # 保存结果
        self.save_results(ret)
        
        return {
            '轮动年化收益': an_ret_rotation,
            '等权年化收益': an_ret_equal,
            '超额收益': excess_ret,
            '轮动夏普': sharpe_rotation,
            '等权夏普': sharpe_equal,
            '轮动回撤': mdd_rotation,
            '等权回撤': mdd_equal
        }
    
    def save_results(self, ret):
        """
        保存回测结果。文件被占用（如Excel打开）时仅告警不中断，
        回测绩效已在内存中，不影响 run_daily 返回信号。
        """
        filename = os.path.join(self.data_path, 'rotation_rate_backtest.xlsx')
        try:
            excel = WriteExcel(filename)

            # 写入回测明细
            worksheet = excel.add_worksheet('回测结果')
            num_format_pd = pd.DataFrame([], columns=ret.columns, index=['format'])
            num_format_pd.loc['format', :] = '0.0%'
            excel.write_pandas(ret, worksheet, begin_row_number=0, begin_col_number=1,
                               num_format_pd=num_format_pd, color='gray', fillna=True)

            excel.close()
            print(f"\n结果已保存至: {filename}")
        except PermissionError:
            print(f"\n结果落盘跳过（{filename} 被占用，请关闭 Excel 后重跑刷新）")
        except Exception as e:
            # xlsxwriter 把 PermissionError 包装成 FileCreateError
            if "Errno 13" in str(e) or "Permission" in str(e):
                print(f"\n结果落盘跳过（{filename} 被占用，请关闭 Excel 后重跑刷新）")
            else:
                raise

    # ==================================================================
    # 调度器入口：每日最新信号（供 cal_data_daily 调用）
    # ==================================================================

    def _refresh_rate_data(self):
        """刷新美债利率月度数据并落盘（2026-09-22 起双源）。

        优先级：api_wind macro_edb.pq（G0000888_US10Y_D，本地，由
        db_wind_api_windpy.py 全量刷新）→ WindPy EDB G0000888（需终端）。
        历史背景：2026-09-02 用户决策只用 Wind API（db_wind 离线库海外指标
        2025-06 停更）；2026-09-21 api_wind 补数体系建立后作为本地优先源。
        日度 EDB 序列 resample 到月末，覆盖写 us_10y_treasury_rate.csv。
        当月未走完时不写入正式月度序列，避免将月内最新值标成月末值。
        日常信号只使用最近一个已结束月份。
        两源均不可用时返回 None，由调用方回退本地 CSV。
        """
        # 源1：api_wind（本地 parquet，无需 Wind 终端）
        api_df = self._load_api_wind_rate()
        if api_df is not None and len(api_df) > 0:
            print("[run_daily] api_wind 刷新美债10Y（G0000888_US10Y_D）")
            self._save_rate_csv(api_df)
            return api_df
        print("[run_daily] api_wind 不可用，回退 WindPy 在线源")

        # 源2：WindPy（需 Wind 终端）
        try:
            from WindPy import w
        except ImportError:
            print("[run_daily] WindPy 不可用（未安装或无终端），跳过在线刷新")
            return None

        print("[run_daily] Wind EDB 拉取美债10Y (G0000888)...")
        try:
            w.start()
            data = w.edb("G0000888", "2000-01-01",
                         datetime.now().strftime("%Y-%m-%d"), "Fill=Previous")
        except Exception as e:
            print("[run_daily] Wind 拉取失败: %s" % e)
            return None
        if data.ErrorCode != 0:
            print("[run_daily] Wind 拉取失败 ErrorCode=%s" % data.ErrorCode)
            return None

        df = pd.DataFrame({'利率': data.Data[0]}, index=pd.to_datetime(data.Times))
        df = df.resample('ME').last().dropna()
        df.index = df.index.strftime('%Y%m%d')
        df = self._drop_incomplete_current_month(df)
        self._save_rate_csv(df)
        return df

    def _save_rate_csv(self, df):
        """利率月度序列落盘。文件被占用（如Excel打开）时仅告警不中断，
        内存中的新数据照常供本次回测使用。"""
        out = os.path.join(self.data_path, "us_10y_treasury_rate.csv")
        try:
            df.to_csv(out, encoding="gbk")
            print("[run_daily] 利率数据已落盘 %s（%s ~ %s, %d 期）" % (
                out, df.index[0], df.index[-1], len(df)))
        except PermissionError:
            print("[run_daily] 落盘跳过（%s 被占用，请关闭 Excel）" % out)

    def run_daily(self, update=True, verbose=False):
        """每日完整流程：刷新利率数据 → 回测 → 返回当日最新信号结构化 dict。

        供 CalDataDaily 调度器使用：
        - update=True 时先用 Wind API (WindPy G0000888) 强制刷新利率 CSV
          （需 Wind 终端；失败自动回退本地 CSV，本地也没有则报错）；
        - 随后完整回测（月度调仓）并落盘 rotation_rate_backtest.xlsx；
        - 最后计算"截至今日"的最新 Z 值与对应四风格权重。

        返回字段：
            rate_date    最新已结束月度利率期（YYYYMMDD）
            rate         当期美债10Y利率（%）
            z_score      当期利率指标 Z（None 表示窗口不足）
            regime       '下行' / '上行' / '中性'
            weights      dict(红利股/价值股/质量股/成长股 = 权重)
            holding      建议持仓描述，如 '成长股 + 质量股（各50%）'
            perf         回测绩效 dict（轮动/等权 年化、夏普、回撤、超额）
            last_rebalance  上一个正式月度调仓时点（YYYYMMDD）
            last_weights    该时点的权重 dict
            xlsx         回测明细 xlsx 路径
        """
        import io
        import contextlib

        buf = io.StringIO()
        ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(buf)
        with ctx:
            # 1) 数据：在线刷新（可选）→ 回退本地
            rate_data = None
            if update:
                rate_data = self._refresh_rate_data()
            if rate_data is None:
                rate_data = self.load_us_treasury_rate()
            if rate_data is None:
                raise RuntimeError("利率轮动：美债利率数据不可用（本地与在线均失败）")

            # 2) 完整回测（含落盘 + self.perf）
            port_ret = self.backtest(rate_data=rate_data)
            if port_ret is None:
                raise RuntimeError("利率轮动：回测失败")

            # 3) 当日最新 Z 值与权重
            z_score = self.calculate_rate_indicator(rate_data)
            weights = self.generate_signals(z_score)

            z_now = z_score.iloc[-1]
            w_now = weights.iloc[-1]
            rate_date = str(rate_data.index[-1])
            rate_now = float(rate_data['利率'].iloc[-1])

            if pd.isna(z_now):
                regime, weights_now, holding = '中性', \
                    {c: 0.25 for c in ['红利股', '价值股', '质量股', '成长股']}, \
                    '等权配置（各25%，Z值窗口不足）'
            else:
                if z_now > self.threshold:
                    regime = '下行'
                    holding = '成长股 + 质量股（各50%）'
                elif z_now < -self.threshold:
                    regime = '上行'
                    holding = '红利股 + 价值股（各50%）'
                else:
                    regime = '中性'
                    holding = '等权配置（各25%）'
                weights_now = {c: float(w_now[c]) for c in
                               ['红利股', '价值股', '质量股', '成长股']}

            # 4) 上一个正式月度调仓时点（回测调仓序列的最后一期）
            last_date = str(port_ret.index[-2]) if len(port_ret) >= 2 else None
            last_weights = None
            if last_date:
                # weights.index 是月末自然日(YYYYMMDD)，调仓日是交易日，
                # 用 <= last_date 的最近一期月末
                w_idx = weights.index[weights.index <= last_date]
                if len(w_idx) > 0:
                    wl = weights.loc[w_idx[-1]]
                    last_weights = {c: float(wl[c]) for c in
                                    ['红利股', '价值股', '质量股', '成长股']}

        return {
            'rate_date': rate_date,
            'rate': rate_now,
            'z_score': float(z_now) if not pd.isna(z_now) else None,
            'regime': regime,
            'weights': weights_now,
            'holding': holding,
            'perf': getattr(self, 'perf', None),
            'last_rebalance': last_date,
            'last_weights': last_weights,
            'xlsx': os.path.join(self.data_path, 'rotation_rate_backtest.xlsx'),
        }


def main():
    """入口：加载利率数据（多源降级链）→ 加载风格腿（StyleModel）→ 回测落盘"""
    import warnings
    warnings.filterwarnings('ignore')
    pd.options.mode.chained_assignment = None

    model = RotationRateModel()
    try:
        model.backtest()
    except Exception as e:
        print(f"\n回测失败: {e}")
        print("请确保美债利率数据可得（本地 CSV / api_wind / Fred / WindPy）")


if __name__ == '__main__':
    main()
