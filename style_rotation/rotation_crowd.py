# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from config.config import Config
from utility.code_format import CodeFormat
from utility.factor_operate import FactorOperate
from utility.factor_standard import FactorStandard
from utility.factor_remove_value import FactorRemoveValue
from datetime import datetime
from utility.date import Date
from utility.write_excel import WriteExcel
from loguru import logger


class RotationCrowd(object):

    """ 国盛证券 风格轮动的三个标尺
    1、Factor of Factor
    2、估值-趋势-拥挤度
    3、多空换手率比值 多空波动率比值 多空beta比值
    4、都是最近3个月数值，每个行业分5组，头/尾
    5、做2月平滑（这里做1月平滑）
    6、比率进行平滑之后求时序上的Zscore数值（报告是6年，这里采用5年）
    7 报告中的离散度没有采用 采用的是估值价差 - 拥挤度 - 趋势 三个指标
    8 数值大于1.5的时候较为危险

    2026-08-31 口径修复（详见 readme.md 研究阶段简表·阶段四拥挤度行）：
    a) to/vol 的 rolling(60) 加 min_periods=40——原全窗口口径下个股停牌/缺数据
       一日整窗即NaN，早年换手分量大片断续（拥挤度score缺失的主因）；
    b) 指数拥挤度/估值价差改为四风格腿固定单一指数成分全程（2026-09-25 对齐
       style_model 腿体系：红利h20269/成长980080/质量932430CNY010/价值980081
       价格母指数），成分权重改从 db_wind/IndexWeights.pq 读取（原
       index_weight.pkl 不存在，该部分从未运行过）；固定单一代码源
       避免主/替切换造成比值断点。
    拥挤度定位（本项目回测结论）：稳定性/风控信号（Z>1.5设权重上限），
    反向单信号无超额，不宜作收益型轮动信号。
    """

    def __int__(self):
        pass

    def cal_factor_crowdeness(self, factor, beta, vol, to, ind_data, csv_file):

        file = os.path.join(Config().style_signals_path, "%s_拥挤度.xlsx" % (csv_file,))
        if os.path.exists(file):
            res = pd.read_excel(file, index_col=[0], usecols="A:D")
            res.index = res.index.map(str)
            res = res.replace(0.0, np.nan)
        else:
            res = pd.DataFrame(index=beta.index, columns=["to", "beta", "vol"])
        file = os.path.join(Config().data_path, r"derived_data\stock_data\\stock_basic_data", "stock_wind_info.csv")
        info = self._load_stock_info()
        beta = beta.T

        for date in beta.columns:

            mv_date = pd.DataFrame(factor[date]).dropna()
            mv_date = mv_date.sort_values(by=[date], ascending=False)
            data_date = pd.concat([mv_date, info[["ListDate", "DelistDate", "Name"]]], axis=1)
            data_date = data_date.dropna(subset=[date, "ListDate"], how="any")
            data_date["ListDate"] = data_date["ListDate"].map(Date().change_to_str)
            data_date["DelistDate"] = data_date["DelistDate"].fillna("20990101")
            data_date["DelistDate"] = data_date["DelistDate"].map(Date().change_to_str)
            data_date = data_date[data_date["ListDate"] <= Date().get_normal_date_offset(date, -90)]
            data_date = data_date[data_date["DelistDate"] > date]

            beta_date = pd.DataFrame(beta[date]).dropna()
            beta_date.columns = ["beta"]

            vol_date = pd.DataFrame(vol[date]).dropna()
            vol_date.columns = ["vol"]

            to_date = pd.DataFrame(to[date]).dropna()
            to_date.columns = ["to"]

            ind_date = Date().get_factor_recent_date(ind_data, date, "ind")

            data_date = pd.concat([data_date, beta_date, vol_date, to_date, ind_date], axis=1)
            data_date = data_date.dropna(subset=["beta", "vol", "to", date, "ind"], how="any")
            data_date["ind"] = data_date["ind"].map(str)

            ind_list = list(set(data_date.ind))

            res_ind = pd.DataFrame([], index=ind_list, columns=["beta", "vol", "to"])

            for ind in ind_list:
                data_date_ind = data_date[data_date["ind"] == ind]
                data_date_ind = data_date_ind.sort_values(by=[date], ascending=False)
                data_date_ind_top = data_date_ind.iloc[0:int(len(data_date_ind) / 5 + 1), :]
                data_date_ind_bot = data_date_ind.iloc[-int(len(data_date_ind) / 5 + 1):, :]
                res_ind.loc[ind, "beta"] = data_date_ind_top["beta"].mean() / data_date_ind_bot["beta"].mean()
                res_ind.loc[ind, "vol"] = data_date_ind_top["vol"].mean() / data_date_ind_bot["vol"].mean()
                res_ind.loc[ind, "to"] = data_date_ind_top["to"].mean() / data_date_ind_bot["to"].mean()

            res_ind = res_ind.mean()
            res.loc[date, :] = res_ind

            logger.info("%s cal FactorCrowdeness %s" % (date, csv_file))

        res["beta_m"] = res["beta"].rolling(window=20 * 1).mean()
        res["vol_m"] = res["vol"].rolling(window=20 * 1).mean()
        res["to_m"] = res["to"].rolling(window=20 * 1).mean()

        res["beta_std"] = res["beta_m"].rolling(window=240 * 5).std()
        res["beta_mean"] = res["beta_m"].rolling(window=240 * 5).mean()
        res["beta_score"] = (res["beta_m"] - res["beta_mean"]) / res["beta_std"]

        res["vol_std"] = res["vol_m"].rolling(window=240 * 5).std()
        res["vol_mean"] = res["vol_m"].rolling(window=240 * 5).mean()
        res["vol_score"] = (res["vol_m"] - res["vol_mean"]) / res["vol_std"]

        res["to_std"] = res["to_m"].rolling(window=240 * 5).std()
        res["to_mean"] = res["to_m"].rolling(window=240 * 5).mean()
        res["to_score"] = (res["to_m"] - res["to_mean"]) / res["to_std"]

        res["score"] = res[["beta_score", "to_score", "vol_score"]].mean(axis=1)
        file = os.path.join(Config().style_signals_path, "%s_拥挤度.xlsx" % (csv_file,))
        excel = WriteExcel(file)
        worksheet = excel.add_worksheet("拥挤度")
        excel.write_pandas(res, worksheet, begin_row_number=0, begin_col_number=0,
                           num_format_pd=None, color="gray", fillna=True)

        data_plot = res[["score"]]
        col = len(res.columns) + 2
        excel.write_pandas(data_plot, worksheet, begin_row_number=0, begin_col_number=col,
                           num_format_pd=None, color="gray", fillna=True)
        excel.line_chart_time_series_plot(worksheet, row_number=0,
                                          col_number=col, data_pd=data_plot,
                                          series_name=data_plot.columns, chart_name="%s拥挤度" % csv_file,
                                          insert_pos="E2", sheet_name="拥挤度")

        excel.close()

    def cal_factor_valuation(self, factor, val, ind_data, csv_file):

        file = os.path.join(Config().style_signals_path, "%s_估值价差.xlsx" % (csv_file,))
        if os.path.exists(file):
            res = pd.read_excel(file, index_col=[0], usecols="A:B")
            res.index = res.index.map(str)
            res = res.replace(0.0, np.nan)
        else:
            res = pd.DataFrame(index=val.columns, columns=["val"])

        info = self._load_stock_info()

        for date in val.columns:

            mv_date = pd.DataFrame(factor[date]).dropna()
            mv_date = mv_date.sort_values(by=[date], ascending=False)
            data_date = pd.concat([mv_date, info[["ListDate", "DelistDate", "Name"]]], axis=1)
            data_date = data_date.dropna(subset=[date, "ListDate"], how="any")
            data_date["ListDate"] = data_date["ListDate"].map(Date().change_to_str)
            data_date["DelistDate"] = data_date["DelistDate"].fillna("20990101")
            data_date["DelistDate"] = data_date["DelistDate"].map(Date().change_to_str)
            data_date = data_date[data_date["ListDate"] <= Date().get_normal_date_offset(date, -90)]
            data_date = data_date[data_date["DelistDate"] > date]

            val_date = pd.DataFrame(val[date]).dropna()
            val_date.columns = ["val"]

            ind_date = Date().get_factor_recent_date(ind_data, date, "ind")

            data_date = pd.concat([data_date, val_date, ind_date], axis=1)
            data_date = data_date.dropna(subset=["val", date, "ind"], how="any")
            data_date["ind"] = data_date["ind"].map(str)

            ind_list = list(set(data_date.ind))

            res_ind = pd.DataFrame([], index=ind_list, columns=["val"])

            for ind in ind_list:
                data_date_ind = data_date[data_date["ind"] == ind]
                data_date_ind = data_date_ind.sort_values(by=[date], ascending=False)
                data_date_ind_top = data_date_ind.iloc[0:int(len(data_date_ind) / 5 + 1), :]
                data_date_ind_bot = data_date_ind.iloc[-int(len(data_date_ind) / 5 + 1):, :]
                res_ind.loc[ind, "val"] = data_date_ind_top["val"].mean() / data_date_ind_bot["val"].mean()

            res_ind = res_ind.mean()
            res.loc[date, :] = res_ind

            logger.info("%s cal FactorValuation %s" % (date, csv_file))

        res["val_m"] = res["val"].rolling(window=20 * 1).mean()
        res["val_std"] = res["val_m"].rolling(window=240 * 5).std()
        res["val_mean"] = res["val_m"].rolling(window=240 * 5).mean()
        res["val_score"] = (res["val_m"] - res["val_mean"]) / res["val_std"]

        res["score"] = res[["val_score"]].mean(axis=1)
        file = os.path.join(Config().style_signals_path, "%s_估值价差.xlsx" % (csv_file,))
        excel = WriteExcel(file)
        worksheet = excel.add_worksheet("拥挤度")
        excel.write_pandas(res, worksheet, begin_row_number=0, begin_col_number=0,
                           num_format_pd=None, color="gray", fillna=True)

        data_plot = res[["score"]]
        col = len(res.columns) + 2
        excel.write_pandas(data_plot, worksheet, begin_row_number=0, begin_col_number=col,
                           num_format_pd=None, color="gray", fillna=True)
        excel.line_chart_time_series_plot(worksheet, row_number=0,
                                          col_number=col, data_pd=data_plot,
                                          series_name=data_plot.columns, chart_name="%s估值价差" % csv_file,
                                          insert_pos="E2", sheet_name="估值价差")

        excel.close()

    def get_port(self, file="excess_port_20250620.mat"):

        file = os.path.join(Config().style_signals_path, file)
        import scipy.io as sio

        # 读取.mat文件
        mat_data = sio.loadmat(file)

        num = mat_data["D1"].shape[0]
        data = pd.DataFrame()
        for i in range(0, num):
            date = str(mat_data["D1"][i][0][0][0][0])
            val = mat_data["Port"][i][0]
            stock_list = [subarray[0][0] for subarray in val]
            python_str_list = [str(code) for code in stock_list]

            weight = mat_data["Port_wt"][i][0]
            stock_weight = [subarray[0] for subarray in weight]
            python_weight = [float(code) for code in stock_weight]

            data_date = pd.DataFrame(python_weight, index=python_str_list, columns=[date])
            data = pd.concat([data, data_date], axis=1)

        data.columns = data.columns.map(lambda x: datetime.strftime(datetime.strptime(x, "%Y/%m/%d"), "%Y%m%d"))
        return data

    def get_leg_weights(self, codes):
        """
        四风格腿成分权重：从 db_wind/IndexWeights.pq 读取（月末权重）

        2026-09-25 口径对齐 style_model 腿体系：拥挤度/估值价差固定用
        （红利h20269/成长980080/质量932430CNY010/价值980081）成分全程，
        不再读 index_data/index_factor/index_weight.pkl（该文件不存在）。
        其中价值腿 480081（价值100R）在 IndexWeights 无任何权重，用同族
        价格母指数 980081；其余三腿为主指数本身（监测窗口 since 起 100% 覆盖，
        早年权重起点：h20269 2019-04 / 980080 2021-12 / 932430CNY010 2025-03）。

        返回
        -------
        字典，值为 DataFrame
            {代码: 宽表(index=Ticker, columns=YYYYMMDD, values=权重)}
            与 cal_port_crowdeness/cal_port_valuation 的 port 参数格式一致
        """
        import pyarrow.dataset as ds
        path = os.path.join(Config().db_wind_path, "IndexWeights.pq")
        iw = ds.dataset(path).to_table(
            filter=ds.field('ITicker').isin(list(codes))).to_pandas()

        legs = {}
        for c in codes:
            try:
                wide = iw.xs(c, level='ITicker')['value'].unstack('Ticker').sort_index()
            except KeyError:
                print("警告: IndexWeights 中无 %s" % c)
                continue
            w = wide.T                                    # index为Ticker，columns为日期
            w.columns = w.columns.map(lambda x: datetime.strftime(x, "%Y%m%d"))
            legs[c] = w
        return legs

    def cal_port_crowdeness(self, port, beta, vol, to, ind_data, csv_file):

        file = os.path.join(Config().style_signals_path, "%s_拥挤度.xlsx" % (csv_file,))
        if os.path.exists(file):
            res = pd.read_excel(file, index_col=[0], usecols="A:D")
            res.index = res.index.map(str)
            res = res.replace(0.0, np.nan)
        else:
            res = pd.DataFrame(index=beta.index, columns=["to", "beta", "vol"])

        info = self._load_stock_info()
        beta = beta.T

        for date in beta.columns:

            mv_date = Date().get_factor_recent_date(port, date, "port")
            mv_date = mv_date.sort_values(by=["port"], ascending=False)
            data_date = pd.concat([mv_date, info[["ListDate", "DelistDate", "Name"]]], axis=1)
            data_date = data_date.dropna(subset=["ListDate"], how="any")
            data_date["ListDate"] = data_date["ListDate"].map(Date().change_to_str)
            data_date["DelistDate"] = data_date["DelistDate"].fillna("20990101")
            data_date["DelistDate"] = data_date["DelistDate"].map(Date().change_to_str)
            data_date = data_date[data_date["ListDate"] <= Date().get_normal_date_offset(date, -90)]
            data_date = data_date[data_date["DelistDate"] > date]

            beta_date = pd.DataFrame(beta[date]).dropna()
            beta_date.columns = ["beta"]

            vol_date = pd.DataFrame(vol[date]).dropna()
            vol_date.columns = ["vol"]

            to_date = pd.DataFrame(to[date]).dropna()
            to_date.columns = ["to"]

            ind_date = Date().get_factor_recent_date(ind_data, date, "ind")

            data_date = pd.concat([data_date, beta_date, vol_date, to_date, ind_date], axis=1)
            data_date = data_date.dropna(subset=["beta", "vol", "to", "ind"], how="any")
            data_date["ind"] = data_date["ind"].map(str)

            ind_list = list(set(data_date.ind))

            data_port = data_date.dropna(subset=["beta", "vol", "to", "port", "ind"], how="any")
            data_port = data_port[data_port["port"] > 0]
            res_ind = pd.DataFrame([], index=ind_list, columns=["beta", "vol", "to"])

            for ind in ind_list:
                data_date_port_ind = data_port[data_port["ind"] == ind]
                data_date_ind = data_date[data_date["ind"] == ind]

                res_ind.loc[ind, "beta"] = data_date_port_ind["beta"].mean() / data_date_ind["beta"].mean()
                res_ind.loc[ind, "vol"] = data_date_port_ind["vol"].mean() / data_date_ind["vol"].mean()
                res_ind.loc[ind, "to"] = data_date_port_ind["to"].mean() / data_date_ind["to"].mean()

            res_ind = res_ind.mean()
            res.loc[date, :] = res_ind

            logger.info("%s cal PortfolioCrowdeness %s" % (date, csv_file))

        res["beta_m"] = res["beta"].rolling(window=20 * 1).mean()
        res["vol_m"] = res["vol"].rolling(window=20 * 1).mean()
        res["to_m"] = res["to"].rolling(window=20 * 1).mean()

        res["beta_std"] = res["beta_m"].rolling(window=240 * 5).std()
        res["beta_mean"] = res["beta_m"].rolling(window=240 * 5).mean()
        res["beta_score"] = (res["beta_m"] - res["beta_mean"]) / res["beta_std"]

        res["vol_std"] = res["vol_m"].rolling(window=240 * 5).std()
        res["vol_mean"] = res["vol_m"].rolling(window=240 * 5).mean()
        res["vol_score"] = (res["vol_m"] - res["vol_mean"]) / res["vol_std"]

        res["to_std"] = res["to_m"].rolling(window=240 * 5).std()
        res["to_mean"] = res["to_m"].rolling(window=240 * 5).mean()
        res["to_score"] = (res["to_m"] - res["to_mean"]) / res["to_std"]

        res["score"] = res[["beta_score", "to_score", "vol_score"]].mean(axis=1)
        file = os.path.join(Config().style_signals_path, "%s_拥挤度.xlsx" % (csv_file,))
        excel = WriteExcel(file)
        worksheet = excel.add_worksheet("拥挤度")
        excel.write_pandas(res, worksheet, begin_row_number=0, begin_col_number=0,
                           num_format_pd=None, color="gray", fillna=True)

        data_plot = res[["score"]]
        col = len(res.columns) + 2
        excel.write_pandas(data_plot, worksheet, begin_row_number=0, begin_col_number=col,
                           num_format_pd=None, color="gray", fillna=True)
        excel.line_chart_time_series_plot(worksheet, row_number=0,
                                          col_number=col, data_pd=data_plot,
                                          series_name=data_plot.columns, chart_name="%s拥挤度" % csv_file,
                                          insert_pos="E2", sheet_name="拥挤度")

        excel.close()

    def cal_port_valuation(self, port, val, ind_data, csv_file):

        # 计算组合估值价差
        file = os.path.join(Config().style_signals_path, "%s_估值价差.xlsx" % (csv_file,))
        if os.path.exists(file):
            res = pd.read_excel(file, index_col=[0], usecols="A:B")
            res.index = res.index.map(str)
            res = res.replace(0.0, np.nan)
        else:
            res = pd.DataFrame(index=val.columns, columns=["val"])
        info = self._load_stock_info()

        for date in val.columns:

            mv_date = Date().get_factor_recent_date(port, date, "port")
            mv_date = mv_date.sort_values(by=["port"], ascending=False)
            data_date = pd.concat([mv_date, info[["ListDate", "DelistDate", "Name"]]], axis=1)
            data_date = data_date.dropna(subset=["ListDate"], how="any")
            data_date["ListDate"] = data_date["ListDate"].map(Date().change_to_str)
            data_date["DelistDate"] = data_date["DelistDate"].fillna("20990101")
            data_date["DelistDate"] = data_date["DelistDate"].map(Date().change_to_str)
            data_date = data_date[data_date["ListDate"] <= Date().get_normal_date_offset(date, -90)]
            data_date = data_date[data_date["DelistDate"] > date]

            val_date = pd.DataFrame(val[date]).dropna()
            val_date.columns = ["val"]

            ind_date = Date().get_factor_recent_date(ind_data, date, "ind")

            data_date = pd.concat([data_date, val_date, ind_date], axis=1)
            data_date = data_date.dropna(subset=["val", "ind"], how="any")
            data_date["ind"] = data_date["ind"].map(str)

            ind_list = list(set(data_date.ind))

            data_port = data_date.dropna(subset=["val", "port", "ind"], how="any")
            data_port = data_port[data_port["port"] > 0]
            res_ind = pd.DataFrame([], index=ind_list, columns=["val"])

            for ind in ind_list:
                data_date_port_ind = data_port[data_port["ind"] == ind]
                data_date_ind = data_date[data_date["ind"] == ind]
                res_ind.loc[ind, "val"] = data_date_port_ind["val"].mean() / data_date_ind["val"].mean()

            res_ind = res_ind.mean()
            res.loc[date, :] = res_ind

            logger.info("%s cal PortfolioValuation %s" % (date, csv_file))

        res["val_m"] = res["val"].rolling(window=20 * 1).mean()

        res["val_std"] = res["val_m"].rolling(window=240 * 5).std()
        res["val_mean"] = res["val_m"].rolling(window=240 * 5).mean()
        res["val_score"] = (res["val_m"] - res["val_mean"]) / res["val_std"]

        res["score"] = res[["val_score"]].mean(axis=1)
        file = os.path.join(Config().style_signals_path, "%s_估值价差.xlsx" % (csv_file,))
        excel = WriteExcel(file)
        worksheet = excel.add_worksheet("估值价差")
        excel.write_pandas(res, worksheet, begin_row_number=0, begin_col_number=0,
                           num_format_pd=None, color="gray", fillna=True)

        data_plot = res[["score"]]
        col = len(res.columns) + 2
        excel.write_pandas(data_plot, worksheet, begin_row_number=0, begin_col_number=col,
                           num_format_pd=None, color="gray", fillna=True)
        excel.line_chart_time_series_plot(worksheet, row_number=0,
                                          col_number=col, data_pd=data_plot,
                                          series_name=data_plot.columns, chart_name="%s估值价差" % csv_file,
                                          insert_pos="E2", sheet_name="估值价差")

        excel.close()

    def _load_local_panels(self, beg):
        """2026-09-25 数据链迁移：旧 stock_factor h5（load_factor_path）已随
        stock_data 存量清理退役，5 个量价因子改读 db_wind/StockEOD.pq，
        中信一级行业改由 SectorInOut.pq（Level==1）point-in-time 构建（月末列）。

        返回 dict：mv/pe/pb/to/close/ind
            值为宽表 index=股票代码(XXXXXX.SZ/SH), columns=YYYYMMDD字符串
            （与旧 h5 因子面板同构，下游 beta/vol/to/分位逻辑零改动）
        """
        import pyarrow.dataset as ds
        beg_ts = pd.Timestamp(beg)
        se = ds.dataset(str(Config().db_wind_path / "StockEOD.pq")).to_table(
            filter=ds.field('Date') >= beg_ts,
            columns=['Date', 'Ticker', 'MV', 'PE_TTM', 'PB', 'FRTO',
                     'AdjClose']).to_pandas()
        panels = {}
        for key, col in [('mv', 'MV'), ('pe', 'PE_TTM'), ('pb', 'PB'),
                         ('to', 'FRTO'), ('close', 'AdjClose')]:
            w = se[col].unstack('Ticker')
            w.index = w.index.strftime('%Y%m%d')
            w = w.T
            panels[key] = w[w.index.str.endswith("SZ") | w.index.str.endswith("SH")]
        # 中信一级行业 point-in-time 面板（月末列，行业代码=Iticker 字符串）
        so = pd.read_parquet(Config().db_wind_path / "SectorInOut.pq")
        so = so[so['Level'] == 1].copy()
        so['Outdate'] = so['Outdate'].fillna(pd.Timestamp('2099-12-31'))
        month_ends = pd.date_range(beg_ts, pd.Timestamp.now(), freq='ME')
        rows = {}
        for d in month_ends:
            m = so[(so['Indate'] <= d) & (so['Outdate'] > d)]
            rows[d.strftime('%Y%m%d')] = m.set_index('Ticker')['Iticker']
        panels['ind'] = pd.DataFrame(rows)
        return panels

    def _load_stock_info(self):
        """股票基础信息（2026-09-25 迁移）：stock_wind_info.csv 已退役，
        改读 db_wind/StockDesc.pq，输出与旧 info 同构
        （index=Ticker，ListDate/DelistDate=YYYYMMDD字符串，DelistDate 填 20990101）"""
        info = pd.read_parquet(
            Config().db_wind_path / "StockDesc.pq",
            columns=['Ticker', 'ListDate', 'DelistDate', 'Name']).set_index('Ticker')
        info['ListDate'] = info['ListDate'].map(Date().change_to_str)
        info['DelistDate'] = info['DelistDate'].map(Date().change_to_str)
        info['DelistDate'] = info['DelistDate'].fillna('20990101')
        return info

    def cal_data(self, since):

        """ 计算拥挤度指标
        """
        os.makedirs(Config().style_signals_path, exist_ok=True)
        # 读取全部数据（2026-09-25 迁移：旧 h5 因子改读 db_wind 本地面板；
        # 预留 1900 自然日（≈1300 交易日）给 vol/to rolling(60) 与拥挤度/估值价差
        # rolling(1200) 预热——首版 1600 自然日≈1090 交易日不足 1200 窗口，
        # 导致 score 全 NaN，2026-10-09 修正）
        beg = (pd.Timestamp(since) - pd.Timedelta(days=1900)).strftime('%Y-%m-%d')
        panels = self._load_local_panels(beg)
        mv = panels['mv']
        pe = panels['pe']
        pb = panels['pb']
        ind_data = panels['ind']
        to = panels['to']
        close = panels['close']

        # 计算波动率和beta
        # 2026-08-31: rolling加min_periods=40（原全窗口口径下停牌/缺数据一日整窗NaN）
        vol = close.T.pct_change(fill_method=None).rolling(window=60, min_periods=40).std().T * np.sqrt(250)
        pct = close.T.pct_change(fill_method=None)
        mean_pct = pct.mean(axis=1)
        # 2026-09-25 迁移提速：逐股 concat 为 O(n²)，改为 var 循环外一次计算 +
        # dict 收集后一次成表（口径不变：cov(个股, 等权均值) / var(等权均值)，60日窗）
        var = mean_pct.rolling(window=60, min_periods=30).var()
        beta_parts = {}
        for stock in pct.columns:
            cov = pct[stock].rolling(window=60, min_periods=30).cov()
            beta_parts[stock] = cov / var
        beta = pd.DataFrame(beta_parts)

        beta = beta.replace(0.0, np.nan)
        vol = vol.replace(0.0, np.nan)
        to = to.replace(0.0, np.nan)
        # 2026-08-31: rolling加min_periods=40（同上）
        to = to.T.rolling(window=60, min_periods=40).mean().T
        beta = beta.loc[since:, :]

        # 计算估值因子
        bp = 1 / pb
        ep = 1 / pe
        bp = FactorStandard().standardization(bp)
        ep = FactorStandard().standardization(ep)
        [bp, ep] = FactorOperate().make_same_index_columns([bp, ep])
        val = bp.add(ep)
        val = FactorRemoveValue().remove_extreme_value_mad(val)

        # 计算因子拥挤度
        self.cal_factor_crowdeness(mv, beta, vol, to, ind_data, "大小盘")
        self.cal_factor_crowdeness(val, beta, vol, to, ind_data, "价值成长")

        # 计算多头组合拥挤度（2026-09-25 迁移：.mat 组合文件随 zs_factor 目录退役，
        # 无当前对应物时跳过并登记；恢复条件=指定 beta_model 生产权重映射）
        for csv_file in ["excess_port_20250620.mat", "quality_port_20250620.mat",
                         "smallcap_port_20250620.mat"]:
            try:
                port = self.get_port(csv_file)
                self.cal_port_crowdeness(port, beta, vol, to, ind_data, csv_file)
            except FileNotFoundError:
                print("[登记] 组合 %s 不存在（旧 zs_factor 目录已退役），跳过组合拥挤度" % csv_file)

        # 计算指数的拥挤度（2026-09-25 对齐 style_model.py 腿体系：主指数有权重
        # 的直接用主指数，权重缺失的用同族价格母指数；监测窗口 since 起 100% 覆盖）
        leg_index = {
            '红利股': 'h20269.CSI',        # style_model 主指数，IndexWeights 2019-04 起（50只/月）
            '成长股': '980080.CNI',        # style_model 主指数，权重 2021-12 起、2025 起全量 100只/月
            '质量股': '932430CNY010.CSI',  # style_model 主指数（全收益），权重 2025-03 起 100只/月
            '价值股': '980081.CNI',        # 480081（价值100R）IndexWeights 无权重，用同族价格母指数
        }
        legs = self.get_leg_weights(list(leg_index.values()))
        for leg_name, leg_code in leg_index.items():
            if leg_code not in legs:
                continue
            self.cal_port_crowdeness(legs[leg_code], beta, vol, to, ind_data, leg_name)

        # 计算估值因子
        pb[pb <= 0.0] = np.nan
        pe[pe <= 0.0] = np.nan
        bp = 1 / pb
        # 2026-10-09 修正：不再截断到 since——估值价差 score 需 1200 交易日
        # 滚动窗口，截断后窗口永不满足、score 全 NaN；保留全历史让预热在
        # since 之前完成（_load_local_panels 已回退 1900 自然日）

        # 计算因子估值价差
        self.cal_factor_valuation(mv, bp, ind_data, "大小盘")
        self.cal_factor_valuation(val, bp, ind_data, "价值成长")

        # 计算指数的估值价差（2026-08-31与拥挤度同口径：四腿固定替代指数成分）
        for leg_name, leg_code in leg_index.items():
            if leg_code not in legs:
                continue
            self.cal_port_valuation(legs[leg_code], bp, ind_data, leg_name)

        # 计算多头组合估值价差（同上：.mat 退役，无对应物跳过）
        for csv_file in ["excess_port_20250620.mat", "quality_port_20250620.mat",
                         "smallcap_port_20250620.mat"]:
            try:
                port = self.get_port(csv_file)
                self.cal_port_valuation(port, bp, ind_data, csv_file)
            except FileNotFoundError:
                print("[登记] 组合 %s 不存在（旧 zs_factor 目录已退役），跳过组合估值价差" % csv_file)


# ======================================================================
# 价值成长估值价差信号 → 双风格对轮动回测（2026-10-09 融入；原独立脚本已删）
# 判负留证（阶段九）：2011-07~2026-09 超额 +1.80%/年、t=0.91 不显著
# ======================================================================
def calc_val_growth_spread(beg="2005-01-01", use_cache=True):
    """长历史重算价值成长估值价差（与 cal_factor_valuation 同口径，日度）。

    返回 DataFrame(index=YYYYMMDD str)：val / val_m / score
    """
    import pyarrow.dataset as ds
    data_dir = Config().data_dir("derived_data", "allocation_data", "rotation_crowd")
    spread_csv = os.path.join(data_dir, "val_growth_spread_daily.csv")
    if use_cache and os.path.exists(spread_csv):
        out = pd.read_csv(spread_csv, index_col=0, dtype={0: str})
        out.index = out.index.map(str)
        if len(out) > 1000:
            print("使用缓存估值价差: %s（%d 日）" % (spread_csv, len(out)))
            return out

    cfg = Config()
    se = ds.dataset(str(cfg.db_wind_path / "StockEOD.pq")).to_table(
        filter=ds.field('Date') >= pd.Timestamp(beg),
        columns=['Date', 'Ticker', 'PB', 'PE_TTM', 'MV']).to_pandas()

    def _w(col):
        x = se[col].unstack('Ticker')
        x.index = x.index.strftime('%Y%m%d')
        x = x.T
        return x[x.index.str.endswith("SZ") | x.index.str.endswith("SH")]

    mv, pe, pb = _w('MV'), _w('PE_TTM'), _w('PB')

    bp_raw = 1 / pb
    ep_raw = 1 / pe
    bp_raw = FactorStandard().standardization(bp_raw)
    ep_raw = FactorStandard().standardization(ep_raw)
    [bp_raw, ep_raw] = FactorOperate().make_same_index_columns([bp_raw, ep_raw])
    val = bp_raw.add(ep_raw)
    val = FactorRemoveValue().remove_extreme_value_mad(val)

    # 中信一级行业 point-in-time 月末面板
    so = pd.read_parquet(cfg.db_wind_path / "SectorInOut.pq")
    so = so[so['Level'] == 1].copy()
    so['Outdate'] = so['Outdate'].fillna(pd.Timestamp('2099-12-31'))
    ind_panels = {}
    for d in pd.date_range(pd.Timestamp(beg), pd.Timestamp.now(), freq='ME'):
        m = so[(so['Indate'] <= d) & (so['Outdate'] > d)]
        ind_panels[d.strftime('%Y%m%d')] = m.set_index('Ticker')['Iticker']
    ind_cols = sorted(ind_panels)

    info = pd.read_parquet(
        cfg.db_wind_path / "StockDesc.pq",
        columns=['Ticker', 'ListDate', 'DelistDate']).set_index('Ticker')
    info['ListDate'] = info['ListDate'].map(Date().change_to_str)
    info['DelistDate'] = info['DelistDate'].fillna('20990101')
    info['DelistDate'] = info['DelistDate'].map(Date().change_to_str)

    dates = [c for c in val.columns if c >= beg.replace('-', '')]
    res = {}
    for i, date in enumerate(dates):
        ic = [c for c in ind_cols if c <= date]
        if not ic:
            continue
        mv_col = mv[date].dropna()
        frame = pd.concat([mv_col.rename('mv'),
                           val[date].rename('val'),
                           bp[date].rename('bp'),
                           ind_panels[ic[-1]].rename('ind'),
                           info[['ListDate', 'DelistDate']]], axis=1).dropna()
        frame = frame[frame['ListDate'] <= Date().get_normal_date_offset(date, -90)]
        frame = frame[frame['DelistDate'] > date]
        ratios = []
        for ind, g in frame.groupby('ind'):
            g = g.sort_values('val', ascending=False)
            k = int(len(g) / 5 + 1)
            top, bot = g.iloc[:k], g.iloc[-k:]
            if top['bp'].mean() > 0 and bot['bp'].mean() > 0:
                ratios.append(top['bp'].mean() / bot['bp'].mean())
        if ratios:
            res[date] = float(np.mean(ratios))
        if i % 500 == 0:
            print("  估值价差进度 %d/%d (%s)" % (i, len(dates), date))

    out = pd.DataFrame.from_dict(res, orient='index', columns=['val']).sort_index()
    out["val_m"] = out["val"].rolling(window=20).mean()
    out["val_std"] = out["val_m"].rolling(window=1200, min_periods=600).std()
    out["val_mean"] = out["val_m"].rolling(window=1200, min_periods=600).mean()
    out["score"] = (out["val_m"] - out["val_mean"]) / out["val_std"]

    os.makedirs(data_dir, exist_ok=True)
    out.to_csv(spread_csv, encoding="utf-8-sig")
    print("估值价差落盘: %s（%d 日，score 可用 %d 日）"
          % (spread_csv, len(out), out['score'].notna().sum()))
    return out


def run_val_growth_backtest(spread):
    """月末 score → 下月双风格对 50/50（看好价值→红利+价值 / 看好成长→成长+质量）。"""
    from model_allocation.style_rotation.style_model import StyleModel
    sm = StyleModel()
    all_codes = list(sm.style_index.values()) + list(sm.fill_index.values()) \
        + [sm.benchmark_code]
    style_data = sm.fill_price_data(sm.load_index_data(list(set(all_codes))))
    style_data = style_data.rename(columns={v: k for k, v in sm.style_index.items()})
    monthly_ret = style_data.pct_change()

    score = spread["score"].dropna()
    score.index = pd.to_datetime(score.index, format='%Y%m%d')
    m = score.groupby(score.index.to_period('M')).last()
    m = m[m.index < pd.Timestamp.now().to_period('M')]

    legs = list(sm.style_index.keys())
    month_ends = Date().get_trade_date_series(
        m.index[0].to_timestamp('M').strftime('%Y%m%d'),
        style_data.index[-1], 'M')
    idx = monthly_ret.index

    rows = []
    prev_state = None
    for i in range(len(month_ends) - 1):
        me_prev, me_next = month_ends[i], month_ends[i + 1]
        sig_period = pd.Timestamp(me_prev).to_period('M')
        if sig_period not in m.index:
            continue
        s = m.loc[sig_period]
        state = 'value' if s > 0 else 'growth'
        w = {'红利股': 0.0, '成长股': 0.0, '质量股': 0.0, '价值股': 0.0}
        if state == 'value':
            w['红利股'], w['价值股'] = 0.5, 0.5
        else:
            w['成长股'], w['质量股'] = 0.5, 0.5
        turnover = 1.0 if (prev_state is not None and state != prev_state) else 0.0
        prev_state = state
        seg = monthly_ret.loc[(idx > me_prev) & (idx <= me_next)]
        if not len(seg):
            continue
        port = sum(w[k] * (1 + seg[k]).prod() for k in legs if w[k] > 0)
        eqw = np.mean([(1 + seg[k]).prod() for k in legs])
        bench = (1 + seg[sm.benchmark_code]).prod()
        rows.append({'ind': str(me_next), 'state': state, 'score': round(float(s), 3),
                     '组合收益': port - 1 - turnover * 0.003,  # 费后（2026-10-10 全族统一单边千三）
                     '等权收益': eqw - 1,
                     '基准收益': bench - 1, '换手': turnover})

    res = pd.DataFrame(rows).set_index('ind')
    res['超额(vs等权)'] = res['组合收益'] - res['等权收益']
    res['年份'] = [d[:4] for d in res.index]

    def _ann(x):
        x = x.dropna()
        return (1 + x).prod() ** (12 / len(x)) - 1 if len(x) else np.nan

    print("\n[判负留证] 回测: %s ~ %s（%d 月），价值态 %d 月 / 成长态 %d 月，换手 %d 次"
          % (res.index[0], res.index[-1], len(res),
             (res['state'] == 'value').sum(), (res['state'] == 'growth').sum(),
             int(res['换手'].sum())))
    print("轮动年化 %.2f%% | 等权年化 %.2f%% | 超额(vs等权) %.2f%%"
          % (_ann(res['组合收益']) * 100, _ann(res['等权收益']) * 100,
             _ann(res['超额(vs等权)']) * 100))
    yearly = res.groupby('年份').apply(
        lambda g: pd.Series({
            '组合': (1 + g['组合收益']).prod() - 1,
            '等权': (1 + g['等权收益']).prod() - 1,
            '超额': (1 + g['组合收益']).prod() - (1 + g['等权收益']).prod()}))
    print((yearly['超额'] * 100).round(2).to_string())

    data_dir = Config().data_dir("derived_data", "allocation_data", "rotation_crowd")
    os.makedirs(data_dir, exist_ok=True)
    f = os.path.join(data_dir, "crowd_val_growth_backtest.xlsx")
    excel = WriteExcel(f)
    ws = excel.add_worksheet("回测结果")
    excel.write_pandas(res, ws, begin_row_number=0, begin_col_number=0,
                       num_format_pd=None, color="gray", fillna=True)
    ws2 = excel.add_worksheet("分年度")
    excel.write_pandas(yearly, ws2, begin_row_number=0, begin_col_number=0,
                       num_format_pd=None, color="gray", fillna=True)
    excel.close()
    print("结果保存: %s" % f)
    return res


if __name__ == "__main__":

    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "backtest":
        # 价值成长估值价差双风格对回测（判负留证，阶段九）
        spread = calc_val_growth_spread()
        run_val_growth_backtest(spread)
    else:
        # 默认：拥挤度/估值价差监测（国盛三标尺）
        since = "20250701"
        self = RotationCrowd()
        self.cal_data(since)
