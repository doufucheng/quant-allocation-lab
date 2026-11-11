
# -*- coding: utf-8 -*-
import datetime
import os
import pandas as pd
import numpy as np

from config.config import Config
from utility.write_excel import WriteExcel
from utility.date import Date


class AllocationRp(object):

    """
    资产配置模型
    """

    def __init__(self):
        self.etf_pool = []
        self.data_path = Config().allocation_path
        data = pd.read_excel(os.path.join(self.data_path, "etf_momentum.xlsx"), index_col=0)
        self.load_beg_date = "2003-12-01"
        self.backtest_date = "20131220"

        # 指数代码
        self.equity_china_code = "H00922.CSI"  # 红利全收益
        self.equity_china_code = "H20269.CSI"  # CN low-vol total-return index
        self.bond_china_code = "CBA21801.CS"  # 30年国债总财富
        self.bond_china_code = "CBA04501.CS"  # 10年国债总财富
        self.merchant_code = "CICSF102.WI"  # 中信商品期货指数
        self.gold_code = "AU9999.SGE"  # 黄金

    def update_data(self):

        # 资产月度收益
        excel_file = os.path.join(Config().wps_my_research_path, "资产配置(6资产)model.xlsx")
        data = pd.read_excel(excel_file, index_col=0, sheet_name="data")









if __name__ == '__main__':

    """ 参数 """
    date = '20071228'
