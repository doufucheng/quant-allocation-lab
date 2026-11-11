# -*- coding: utf-8 -*-
"""
lppls_model.py（原 timing_lppls.py）
基于 LPPLS（Log-Periodic Power Law Singularity）模型的 A 股指数泡沫/拐点择时信号。

核心模型
--------
LPPLS 假设资产价格在重大拐点（泡沫破裂或恐慌底部）附近服从：

    E[ln p(t)] = A + B*(tc - t)^m + C*(tc - t)^m * cos(w*ln(tc - t) - phi)

其中 tc 为临界时间，m 为幂律指数（0<m<1），w 为对数周期频率。参数 B 的符号决定泡沫方向：
    B < 0  -> 正向泡沫（价格超指数上涨，可能见顶）
    B > 0  -> 负向泡沫（价格超指数下跌，可能见底）

学术来源
--------
- Sornette (2003), "Critical Market Crashes".
  系统阐述 LPPLS 在金融泡沫识别与重大回撤预测中的应用。
- Johansen, Ledoit & Sornette (2000), "Crashes as critical points".
  提出价格接近崩溃点时会呈现幂律 + 对数周期震荡的理论框架。

实现与参数来源
--------------
- lppls Python 库: https://github.com/Boulder-Investment-Technologies/lppls
  提供 nested fits、compute_indicators、filter_conditions_config 的工程实现。
- 《择时的艺术——可识别拐点与趋势在多资产中的应用》:
  采用 LPPLS pos/neg 置信度，提出“泡沫消散后反向开仓”的择时信号规则。
- 兴业证券 郑兆磊、宫民 (2024.08), 《如何捕捉市场趋势与极端拐点？》:
  仅将 LPPLS 负向泡沫置信度作为极端下跌后底部反弹的提示指标，触发阈值 0.3，
  触发后持有两个月；并指出正向泡沫出现后价格仍可能继续上涨，对顶部提示不可靠。

本版本仅使用 LPPLS 负向泡沫置信度作为择时信号，参数与触发/持有规则
对齐兴业证券报告《如何捕捉市场趋势与极端拐点？》：
    - 拟合回看窗口 250 个交易日，以 5 日为步长递减；
    - neg_conf > 0.3 时发出 +1（看多反弹）信号并持有 40 个交易日；
    - 不使用正向泡沫信号做空（报告认为正向泡沫对顶部提示不可靠）。

数据
----
从 Config().db_wind_path 读取日线，标的覆盖（对齐 chan_theory_analysis.py）：
    - 国内指数：IndexEOD.pq / IndexEODWind.pq（Close）
    - ETF/LOF ：ETFEOD.pq（AdjClose 复权价）
    - 商品期货：FUTEOD.pq 主力连续合约（Close，需前复权消除换月跳升；
      用 Close/Presettle-1 收益率合成连续价格，与缠论程序口径一致）

输出
----
Config().data_dir("derived_data", "allocation_data", "timing_lppls")/
    - {name}_lppls.csv        : 单资产的收盘价、pos_conf、neg_conf、signal
    - lppls_confidence_panel.csv : 所有资产的 pos_conf / neg_conf 面板
    - lppls_signal_panel.csv     : 所有资产的信号面板
    - lppls_latest_signal.csv    : 最新一期信号汇总（date/asset/signal 长表）
    - lppls_backtest_summary.csv : 信号策略回测评估（收益/夏普/回撤/胜率 vs 一直持有）
    - lppls_resonance.csv        : 多资产共振指标（当日触发负向泡沫的资产数与占比）
    - {name}_lppls_fits.parquet  : 每次 LPPLS 拟合的原始参数（tc/m/w/a/b/c/O/D 及是否合格）
    - {name}_lppls_res.pkl       : 增量缓存（按窗口结束日期保存拟合结果，日常更新
                                   只拟合新增窗口，参数变化自动失效全量重算）
    - png/{name}_lppls.png       : 单资产价格 + 置信度图（嵌入 Word 报告）
    - LPPLS择时报告_YYYYMMDD_HHMMSS.docx : Word 每日报告（结构参考缠论报告）

性能
----
    嵌套拟合计算量大（每个外层窗口位置 47 次内层拟合 × 25 次随机搜索），
    2000 天回看约 41 万次非线性优化。两项加速措施：
    1. 多进程并行：--workers N（默认 -1 = 全部 CPU 核心），24 核约 10-20 倍加速；
    2. 增量缓存：外层窗口位置相互独立，按窗口结束日期缓存拟合结果；
       日常更新只拟合新增日期（如无新数据则跳过全部拟合），配合固定
       --start-date 使用效果最佳（--max-history 会滑动起始日导致网格
       锚点移动，增量命中率下降但仍正确）。

用法
----
    cd <项目根目录>
    python model_allocation\timing\lppls_model.py
    python model_allocation\timing\lppls_model.py --assets 中证1000 创业板指
    python model_allocation\timing\lppls_model.py --max-history 1500
    python model_allocation\timing\lppls_model.py --start-date 20180101 --end-date 20190430
    python model_allocation\timing\lppls_model.py --workers 16 --no-word
"""

import os
import sys
import json
import pickle
import hashlib
import traceback
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------------
# 项目路径与配置
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import Config

_CFG = Config()

# ------------------------------------------------------------------
# matplotlib 中文字体配置（必须在 import matplotlib.pyplot 之前）
# 缓存目录放在 allocation_data 下，避免污染 code 目录
# ------------------------------------------------------------------
_MPL_CACHE_DIR = Path(_CFG.data_dir("derived_data", "allocation_data", ".matplotlib"))
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans SC",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

# Word 报告依赖
try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    warnings.warn(
        "python-docx 未安装，将跳过 Word 报告生成；请执行：pip install python-docx"
    )

try:
    from lppls import lppls
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "缺少 lppls 包，请先执行：pip install lppls"
    ) from e


# ------------------------------------------------------------------
# LPPLS 多进程 worker（模块级函数，保证 Windows spawn 模式下可 pickle）
# 每个 worker 进程初始化一份 LPPLS 模型，任务只传数据切片，避免重复序列化
# ------------------------------------------------------------------
_LPPLS_WORKER_MODEL: Any = None


def _lppls_worker_init(observations: np.ndarray, filter_config: Optional[dict]):
    """Pool initializer：在每个 worker 进程内构建 LPPLS 模型。"""
    global _LPPLS_WORKER_MODEL
    _LPPLS_WORKER_MODEL = lppls.LPPLS(observations=observations)
    _LPPLS_WORKER_MODEL.filter_conditions_config = (
        _LPPLS_WORKER_MODEL._resolve_filter_conditions_config(filter_config)
    )


def _lppls_worker_task(task: tuple) -> Dict[str, Any]:
    """执行一个外层窗口位置的嵌套拟合任务。"""
    return _LPPLS_WORKER_MODEL._func_compute_nested_fits(task)


class TimingLppls(object):
    """
    LPPLS 泡沫择时模型（仅使用 LPPLS，不引入宏观/趋势/风控指标）。

    对每类资产计算滚动嵌套窗口的 LPPLS 拟合，得到：
        - pos_conf：正向（超涨）泡沫置信度
        - neg_conf：负向（超跌）泡沫置信度

    信号规则（对齐兴业证券《如何捕捉市场趋势与极端拐点？》）：
        - 仅使用负向泡沫置信度 neg_conf；
        - 当 neg_conf > bubble_threshold（默认 0.3）时，发出 +1（看多反弹）信号；
        - 触发后连续持有 hold_days（默认 40 个交易日），期间不再重复发信号；
        - 不使用正向泡沫信号做空（报告认为正向泡沫出现后价格仍可能继续上涨，
          对顶部提示不可靠）。
    """

    # 默认计算资产：名称、parquet 表名、ticker、价格字段
    # 标的清单对齐 chan_theory_analysis.py 的 DEFAULT_TARGETS（ETF/指数/商品期货）
    # 价格字段：指数用 Close；ETF/LOF 用 AdjClose（复权价，消除分红除权跳变）；
    # 商品期货用 Close（读取时做前复权，消除主力换月跳升）
    DEFAULT_ASSETS = [
        # ================================================================
        # 一、国内指数（IndexEOD.pq / IndexEODWind.pq）
        # ================================================================
        # --- 宽基指数 ---
        ("沪深300", "IndexEOD.pq", "000300.SH", "Close"),
        ("中证500", "IndexEOD.pq", "000905.SH", "Close"),
        ("中证1000", "IndexEOD.pq", "000852.SH", "Close"),
        ("中证2000", "IndexEOD.pq", "932000.CSI", "Close"),
        ("中证800", "IndexEOD.pq", "000906.SH", "Close"),
        ("中证全指", "IndexEOD.pq", "000985.CSI", "Close"),
        ("创业板指", "IndexEOD.pq", "399006.SZ", "Close"),
        ("科创50", "IndexEOD.pq", "000688.SH", "Close"),
        ("万得全A", "IndexEODWind.pq", "881001.WI", "Close"),
        # --- 策略指数（红利/成长/价值/质量/低波） ---
        ("中证红利", "IndexEOD.pq", "000922.CSI", "Close"),
        ("中证红利低波", "IndexEOD.pq", "h30269.CSI", "Close"),
        ("中证红利质量", "IndexEOD.pq", "932315.CSI", "Close"),
        ("消费红利", "IndexEOD.pq", "930955.CSI", "Close"),
        ("国证成长100", "IndexEOD.pq", "980080.CNI", "Close"),
        ("国证价值100", "IndexEOD.pq", "980081.CNI", "Close"),
        # --- 行业指数 ---
        ("中证银行", "IndexEOD.pq", "399986.SZ", "Close"),
        ("煤炭等权", "IndexEOD.pq", "399990.SZ", "Close"),
        ("中证白酒", "IndexEOD.pq", "399997.SZ", "Close"),
        # ================================================================
        # 二、ETF/LOF（ETFEOD.pq，AdjClose 复权价）
        # ================================================================
        ("基本面50ETF", "ETFEOD.pq", "512750.SH", "AdjClose"),
        ("金选300LOF", "ETFEOD.pq", "501060.SH", "AdjClose"),
        ("标普500ETF", "ETFEOD.pq", "513500.SH", "AdjClose"),
        ("道琼斯ETF", "ETFEOD.pq", "513400.SH", "AdjClose"),
        ("纳指100ETF", "ETFEOD.pq", "513100.SH", "AdjClose"),
        ("恒生科技ETF", "ETFEOD.pq", "513180.SH", "AdjClose"),
        ("法国ETF", "ETFEOD.pq", "513080.SH", "AdjClose"),
        ("日经ETF", "ETFEOD.pq", "513880.SH", "AdjClose"),
        ("印度基金LOF", "ETFEOD.pq", "164824.SZ", "AdjClose"),
        ("沙特ETF", "ETFEOD.pq", "520830.SH", "AdjClose"),
        ("巴西ETF", "ETFEOD.pq", "520870.SH", "AdjClose"),
        ("港股红利低波ETF", "ETFEOD.pq", "520550.SH", "AdjClose"),
        ("黄金ETF", "ETFEOD.pq", "518880.SH", "AdjClose"),
        ("南方原油LOF", "ETFEOD.pq", "501018.SH", "AdjClose"),
        ("豆粕ETF", "ETFEOD.pq", "159985.SZ", "AdjClose"),
        ("有色金属ETF", "ETFEOD.pq", "159980.SZ", "AdjClose"),
        ("能源化工ETF", "ETFEOD.pq", "159981.SZ", "AdjClose"),
        ("酒ETF", "ETFEOD.pq", "512690.SH", "AdjClose"),
        ("十年国债ETF", "ETFEOD.pq", "511260.SH", "AdjClose"),
        ("30年国债ETF", "ETFEOD.pq", "511090.SH", "AdjClose"),
        # ================================================================
        # 三、商品期货主力连续合约（FUTEOD.pq，读取时前复权）
        # ================================================================
        # --- 黑色系 ---
        ("螺纹钢主力", "FUTEOD.pq", "RB.SHF", "Close"),
        ("热卷主力", "FUTEOD.pq", "HC.SHF", "Close"),
        ("铁矿石主力", "FUTEOD.pq", "I.DCE", "Close"),
        ("焦炭主力", "FUTEOD.pq", "J.DCE", "Close"),
        ("焦煤主力", "FUTEOD.pq", "JM.DCE", "Close"),
        # --- 有色金属 ---
        ("铜主力", "FUTEOD.pq", "CU.SHF", "Close"),
        ("铝主力", "FUTEOD.pq", "AL.SHF", "Close"),
        ("锌主力", "FUTEOD.pq", "ZN.SHF", "Close"),
        ("镍主力", "FUTEOD.pq", "NI.SHF", "Close"),
        ("铅主力", "FUTEOD.pq", "PB.SHF", "Close"),
        ("锡主力", "FUTEOD.pq", "SN.SHF", "Close"),
        # --- 贵金属 ---
        ("黄金主力", "FUTEOD.pq", "AU.SHF", "Close"),
        ("白银主力", "FUTEOD.pq", "AG.SHF", "Close"),
        # --- 能源化工 ---
        ("原油主力", "FUTEOD.pq", "SC.INE", "Close"),
        ("燃油主力", "FUTEOD.pq", "FU.SHF", "Close"),
        ("橡胶主力", "FUTEOD.pq", "RU.SHF", "Close"),
        ("PTA主力", "FUTEOD.pq", "TA.CZC", "Close"),
        ("甲醇主力", "FUTEOD.pq", "MA.CZC", "Close"),
        ("PP主力", "FUTEOD.pq", "PP.DCE", "Close"),
        ("PVC主力", "FUTEOD.pq", "V.DCE", "Close"),
        ("塑料主力", "FUTEOD.pq", "L.DCE", "Close"),
        ("乙二醇主力", "FUTEOD.pq", "EG.DCE", "Close"),
        ("纯碱主力", "FUTEOD.pq", "SA.CZC", "Close"),
        ("玻璃主力", "FUTEOD.pq", "FG.CZC", "Close"),
        # --- 农产品 ---
        ("豆粕主力", "FUTEOD.pq", "M.DCE", "Close"),
        ("豆油主力", "FUTEOD.pq", "Y.DCE", "Close"),
        ("棕榈油主力", "FUTEOD.pq", "P.DCE", "Close"),
        ("豆一主力", "FUTEOD.pq", "A.DCE", "Close"),
        ("玉米主力", "FUTEOD.pq", "C.DCE", "Close"),
        ("鸡蛋主力", "FUTEOD.pq", "JD.DCE", "Close"),
        ("白糖主力", "FUTEOD.pq", "SR.CZC", "Close"),
        ("棉花主力", "FUTEOD.pq", "CF.CZC", "Close"),
        ("菜油主力", "FUTEOD.pq", "OI.CZC", "Close"),
        ("菜粕主力", "FUTEOD.pq", "RM.CZC", "Close"),
    ]

    # 表名 -> 类别（用于报告分组）
    TABLE_CATEGORY = {
        "IndexEOD.pq": "指数",
        "IndexEODWind.pq": "指数",
        "ETFEOD.pq": "ETF",
        "FUTEOD.pq": "商品期货",
    }

    # LPPLS 滚动拟合参数（对齐兴业证券报告《如何捕捉市场趋势与极端拐点？》）
    # 报告描述：“在每个时点回看 250 天，然后以步长 5 日减少回看窗口，得到多条
    # 回看价格序列，然后分别拟合 LPPLS 模型，统计满足泡沫识别条件的数量占比”。
    # 本代码用 lppls 库的 nested-fit 框架复现该过程：外层窗口滚动，每个位置再用
    # 一组从 250 日向下收缩的内层窗口分别拟合；汇总后统计满足过滤条件且 B<0/B>0
    # 的占比，得到 pos_conf / neg_conf。理论基础见 Sornette (2003)。
    LPPLS_PARAMS = {
        # 外层窗口长度（交易日）。兴业报告明确使用 250 日回看窗口。
        "window_size": 250,
        # 内层最小窗口长度。报告未明确下限，保持 lppls 库默认 20 日，
        # 保证每个子窗口仍有足够自由度估计非线性参数 tc/m/w。
        "smallest_window_size": 20,
        # 外层窗口滑动步长（交易日）。报告未明确滚动步长；为平衡时间分辨率与
        # 计算量，这里沿用 lppls 库默认 5 日。
        "outer_increment": 5,
        # 内层窗口收缩步长（交易日）。兴业报告明确“以步长 5 日减少回看窗口”。
        "inner_increment": 5,
        # 每次拟合的最大随机搜索次数。lppls 库默认 / Sornette 文献建议值 = 25。
        # 注：lppls 库 fit() 用 Python random 模块随机选初始点，结果有微小随机性；
        # 置信度是 47 次嵌套拟合的投票占比，对该随机性鲁棒。
        "max_searches": 25,
        # 拟合质量过滤条件。字段含义与默认值来自 lppls 库；数值针对 A 股做宽松化：
        #   - O_min 从库默认 2.5 降至 1.0：A 股泡沫震荡次数往往低于成熟市场。
        #   - D_min 从库默认 0.5 降至 0.0：避免过滤掉较多有效信号。
        #   - m_max 放宽到 1.0，w_max 放宽到 15.0，扩大可接受参数空间。
        # tc_* 条件限定临界时间 tc 的合理范围，防止拟合收敛到遥远未来。
        # 注意：上述宽松化参数为人工经验设置、暂无回测依据；可通过输出文件
        # lppls_backtest_summary.csv 对比库默认值（O_min=2.5, D_min=0.5）下的
        # 信号触发频率与策略表现，再决定是否收紧。
        "filter_conditions_config": {
            "m_min": 0.0,
            "m_max": 1.0,
            "w_min": 2.0,
            "w_max": 15.0,
            "O_min": 1.0,
            "D_min": 0.0,
            "tc_min_days": 0.0,
            "tc_min_frac": 0.0,
            "tc_max_days": 252.0,
            "tc_max_frac": 0.9,
        },
    }

    def __init__(
        self,
        assets=None,
        max_history=2000,
        bubble_threshold=0.30,
        hold_days=40,
        start_date=None,
        end_date=None,
        output_subdir="timing_lppls",
        output_dir=None,
        workers=-1,
        generate_word=True,
    ):
        """
        参数
        ----------
        assets : list[tuple]
            资产列表，每项为 (名称, parquet文件名, ticker, 价格字段)。
        max_history : int
            每个资产只取最近多少个交易日用于拟合，减少计算量。
            该参数仅限制输入序列长度，不改变 LPPLS 窗口参数。
        bubble_threshold : float
            负向泡沫置信度触发阈值。兴业报告将 neg_conf > 0.3 视为超跌信号。
        hold_days : int
            触发后持有交易日数。兴业报告表述为“延续两个月”，按约 20 个交易日/月
            折算为 40 个交易日。
        start_date : str, optional
            数据起始日期（YYYYMMDD 或 YYYY-MM-DD）。先按日期范围过滤再截取
            max_history，便于回测特定时段（如 2008、2015、2018 极端底部）。
        end_date : str, optional
            数据结束日期（YYYYMMDD 或 YYYY-MM-DD）。
        output_subdir : str
            在 allocation_data 下新建的子目录名。
        output_dir : str, optional
            完整输出目录路径（优先于 output_subdir），主要用于测试时
            重定向输出，避免污染正式数据目录。
        workers : int
            LPPLS 嵌套拟合的并行进程数。-1 表示使用全部 CPU 核心（默认）；
            1 表示单进程顺序执行。拟合为 CPU 密集型，24 核机器并行约 10-20 倍加速。
        generate_word : bool
            是否生成 Word 报告（需要 python-docx；结构参考缠论每日报告）。
        """
        self.cfg = Config()
        self.assets = assets if assets is not None else self.DEFAULT_ASSETS
        self.max_history = max_history
        self.bubble_threshold = bubble_threshold
        self.hold_days = hold_days
        self.start_date = start_date
        self.end_date = end_date
        self.workers = workers
        self.generate_word = generate_word and DOCX_AVAILABLE

        if output_dir is not None:
            self.output_path = Path(output_dir)
        else:
            self.output_path = Path(self.cfg.allocation_path) / output_subdir
        self.png_dir = self.output_path / "png"
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.png_dir.mkdir(parents=True, exist_ok=True)

        self.db_wind_path = Path(self.cfg.db_wind_path)
        self._pool = None  # 懒创建的持久拟合进程池

    # ------------------------------------------------------------------
    # 数据读取
    # ------------------------------------------------------------------
    def _slice_and_validate(self, s: pd.Series) -> pd.Series:
        """日期范围过滤 + max_history 截取 + 价格正值校验。"""
        # 先按日期范围过滤，再截取 max_history，保证 --start-date/--end-date
        # 与 --max-history 组合时取到的是范围内的最近数据
        if self.start_date is not None:
            s = s[s.index >= pd.Timestamp(self.start_date)]
        if self.end_date is not None:
            s = s[s.index <= pd.Timestamp(self.end_date)]

        if self.max_history and len(s) > self.max_history:
            s = s.iloc[-self.max_history :]

        # 价格异常值保护：LPPLS 需要对数价格，含 0/负值会产生 -inf 污染全部拟合
        if (s <= 0).any():
            n_bad = int((s <= 0).sum())
            bad_dates = s.index[s <= 0][:5].strftime("%Y-%m-%d").tolist()
            raise ValueError(
                f"价格序列含非正值（共 {n_bad} 个，如 {bad_dates}），"
                f"无法取对数，请先检查数据质量"
            )
        return s

    def _load_price(self, table, ticker, price_col):
        """从 parquet 读取指定 ticker 的价格序列（期货自动前复权）。"""
        file_path = self.db_wind_path / table
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        if table == "FUTEOD.pq":
            return self._load_futures_price(file_path, ticker)

        # 只读价格列，索引 (Date, Ticker) 会自动保留
        df = pd.read_parquet(file_path, columns=[price_col])

        try:
            s = df.xs(ticker, level="Ticker")[price_col]
        except KeyError:
            # fallback：兼容 level 名不是 "Ticker" 的情况
            s = df[df.index.get_level_values("Ticker") == ticker][price_col]

        s = s.dropna().sort_index()
        return self._slice_and_validate(s)

    def _load_futures_price(self, file_path: Path, ticker: str) -> pd.Series:
        """
        读取商品期货主力连续合约收盘价并前复权。

        复权原理（与 chan_theory_analysis.py 的 _futures_forward_adjust 一致）：
        主力连续合约在换月时价格会跳升/跳降，直接用原始 Close 会把换月价差
        误认为趋势。用同合约口径的日收益率 r_t = Close_t / Presettle_t - 1
        （Presettle 是当日开盘时同一合约的前结算价）递推合成连续价格序列，
        消除换月跳升；再归一化到最新原始 Close，保持最新价与实际一致。

        返回
        -------
        pd.Series
            前复权后的连续收盘价。
        """
        # 读取复权所需列（Presettle/Settle 可能不存在，逐个探测）；
        # 用 pyarrow 只读 schema，避免为拿列名而全量加载 parquet
        import pyarrow.parquet as pq

        all_cols = pq.read_schema(file_path).names
        need = [c for c in ["Close", "Presettle", "Settle"] if c in all_cols]
        df = pd.read_parquet(file_path, columns=need)

        try:
            df = df.xs(ticker, level="Ticker")
        except KeyError:
            df = df[df.index.get_level_values("Ticker") == ticker].droplevel("Ticker")

        df = df.dropna(subset=["Close"]).sort_index()

        # 前结算价：优先 Presettle，缺失/非正时用前一日 Settle 补
        if "Presettle" in df.columns:
            presettle = df["Presettle"].copy()
        else:
            presettle = pd.Series(np.nan, index=df.index)
        if "Settle" in df.columns:
            presettle = presettle.where(
                presettle.notna() & (presettle > 0), df["Settle"].shift(1)
            )
        presettle = presettle.ffill()

        # 同合约口径日收益率（换月日 Close 与 Presettle 属同一新合约，无换月价差）
        valid = presettle.notna() & (presettle > 0)
        ret = (df["Close"] / presettle - 1.0).where(valid, 0.0).fillna(0.0)

        # 递推合成连续价格，并归一化到最新原始 Close
        synth = df["Close"].iloc[0] * (1.0 + ret).cumprod()
        if len(synth) >= 2 and synth.iloc[-1] > 0:
            s = synth * (df["Close"].iloc[-1] / synth.iloc[-1])
        else:
            s = df["Close"].copy()

        return self._slice_and_validate(s)

    # ------------------------------------------------------------------
    # LPPLS 核心计算
    # ------------------------------------------------------------------
    def _compute_lppls(self, price, name=""):
        """
        对单个价格序列计算 pos_conf / neg_conf。

        参数
        ----------
        price : pd.Series
            收盘价序列（index 为交易日）。
        name : str
            资产名称，仅用于进度条展示。

        实现说明
        --------
        - 并行：复用 lppls 库的 worker 函数 (_func_compute_nested_fits)，
          通过 multiprocessing.Pool 把外层窗口位置分发到多进程（Windows spawn
          模式，worker 见模块级 _lppls_worker_init/_lppls_worker_task）。
          workers=1 时退化为单进程顺序执行。
        - 增量缓存：外层每个窗口位置的嵌套拟合相互独立（pos/neg_conf 只由该
          位置的 47 次内层拟合投票决定），因此把每次拟合结果按“窗口结束日期”
          缓存到 {name}_lppls_res.pkl；后续运行只对新增窗口结束日期做拟合，
          历史部分直接复用。价格序列末端新增数据对历史窗口的影响是等比例
          缩放（ETF 复权 / 期货前复权均为末端锚定），LPPLS 对常数缩放不变，
          因此复用历史拟合是安全的；参数（窗口/步长/过滤条件）变化时缓存
          自动失效，全量重算。
        - 网格锚定：已完结周的窗口结束日（t2）取“每周最后一个交易日”
          （日历锚定），而非按切片起点等步长推进。这样数据更新导致
          max_history 切片整体滑动时，历史 t2 日期保持不变，增量缓存才能
          跨数据更新日复用（按切片起点步进时，切片滑动 1 天会让全部 t2
          偏移、缓存全失效）。
        - 当前周尚未完结，其每个交易日也纳入网格（保证周中运行也能给出
          当日最新信号）。当前周每日仅新增 1 个拟合窗口，代价很小；下周
          该周完结后，网格回到周锚定点，周中每日窗口自然从缓存淘汰。
        """
        time_ord = [pd.Timestamp.toordinal(t) for t in price.index]
        log_price = np.log(price.values)
        observations = np.array([time_ord, log_price])

        params = self.LPPLS_PARAMS.copy()
        filter_config = params.pop("filter_conditions_config", None)

        window_size = params.get("window_size", 80)
        smallest_window_size = params.get("smallest_window_size", 20)
        outer_increment = params.get("outer_increment", 5)
        inner_increment = params.get("inner_increment", 2)
        max_searches = params.get("max_searches", 25)

        if len(observations[0]) < window_size:
            raise ValueError(
                f"序列长度 {len(observations[0])} 不足窗口 {window_size}，无法拟合"
            )

        # 稳定网格：已完结周取每周最后一个交易日为 t2（跨数据更新不变）
        last_iso = price.index[-1].isocalendar()
        cur_week = (last_iso[0], last_iso[1])
        week_last_idx = {}
        for i, d in enumerate(price.index):
            wk = (d.isocalendar()[0], d.isocalendar()[1])
            if wk != cur_week:  # 只对已完结周取周内最后交易日
                week_last_idx[wk] = i
        anchor_idx = {
            v for v in week_last_idx.values() if v >= window_size - 1
        }
        # 当前周：周内每个已发生交易日均作为 t2（周中也能出当日信号）
        cur_week_idx = [
            i
            for i, d in enumerate(price.index)
            if (d.isocalendar()[0], d.isocalendar()[1]) == cur_week
            and i >= window_size - 1
        ]
        grid_idx = sorted(anchor_idx | set(cur_week_idx))

        tasks = []
        task_end_ords = []
        for i in grid_idx:
            obs = observations[:, i - window_size + 1 : i + 1]
            tasks.append(
                (
                    obs,
                    window_size,
                    i,
                    smallest_window_size,
                    outer_increment,
                    inner_increment,
                    max_searches,
                )
            )
            task_end_ords.append(int(observations[0, i]))

        # ---- 增量缓存：按窗口结束日期复用历史拟合 ----
        cached_by_ord = self._load_res_cache(name, params, filter_config)
        new_idx = [
            k
            for k, end_ord in enumerate(task_end_ords)
            if end_ord not in cached_by_ord
        ]
        if cached_by_ord and new_idx:
            print(
                f"[{name}] 增量计算: 复用 {len(task_end_ords) - len(new_idx)} 个"
                f"历史窗口，新拟合 {len(new_idx)} 个"
            )
        elif cached_by_ord:
            print(f"[{name}] 全部命中增量缓存（{len(cached_by_ord)} 个窗口），跳过拟合")

        new_res = self._run_fit_tasks(
            [tasks[k] for k in new_idx], name, observations, filter_config
        )

        # 按网格日期顺序合并缓存与新结果
        res = []
        new_by_ord = {int(r["t2"]): r for r in new_res}
        for end_ord in task_end_ords:
            r = cached_by_ord.get(end_ord)
            if r is None:
                r = new_by_ord.get(end_ord)
            if r is not None:
                res.append(r)

        # 保存合并后的拟合结果缓存（参数未变才会走到这里）
        self._save_res_cache(name, params, filter_config, price, res)

        # ---- 指标计算（与 lppls 库行为一致）----
        model = lppls.LPPLS(observations=observations)
        model.filter_conditions_config = model._resolve_filter_conditions_config(
            filter_config
        )
        indicators = model.compute_indicators(res)

        # compute_indicators 会在 res 上原地标注 is_qualified，这里把每次拟合的
        # 原始参数（tc/m/w/a/b/c/O/D 等）展开成 DataFrame，方便后续排查过滤条件。
        fits_df = self._flatten_fits(res)

        # time 列为 ordinal，转回真实日期
        indicators["Date"] = (
            indicators["time"]
            .astype(int)
            .map(lambda x: pd.Timestamp(datetime.fromordinal(x)))
        )
        indicators = indicators.set_index("Date")[["pos_conf", "neg_conf"]]

        # 对齐到原始日线，向前填充
        indicators = indicators.reindex(price.index, method="ffill")
        return indicators, fits_df

    # ------------------------------------------------------------------
    # 拟合任务执行（单进程 / 多进程）
    # ------------------------------------------------------------------
    def _get_pool(self, filter_config):
        """
        懒创建并复用全局唯一的拟合进程池。

        Windows spawn 模式下每个 worker 需重新 import numpy/scipy/lppls
        （约 1-2 秒/进程），若每个资产单独建池，启动开销会远超计算本身。
        因此整个 run() 生命周期只建一个池：_func_compute_nested_fits 只使用
        任务自带的数据切片，不依赖 model.observations，故 worker 用任意
        合法形状的占位 observations 初始化即可服务所有资产。
        """
        if self._pool is None:
            n_workers = (
                self.workers
                if self.workers and self.workers > 0
                else (os.cpu_count() or 1)
            )
            window_size = self.LPPLS_PARAMS["window_size"]
            dummy_obs = np.zeros((2, window_size))
            print(
                f"初始化 LPPLS 拟合进程池: {n_workers} workers "
                f"(Windows spawn 启动约需 10-30 秒，仅此一次)"
            )
            self._pool = Pool(
                processes=n_workers,
                initializer=_lppls_worker_init,
                initargs=(dummy_obs, filter_config),
            )
        return self._pool

    def _close_pool(self):
        """关闭并回收进程池（run() 结束时调用）。"""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def _run_fit_tasks(self, tasks, name, observations, filter_config):
        """执行外层窗口拟合任务列表，返回 res dict 列表（顺序与 tasks 一致）。"""
        if not tasks:
            return []

        n_workers = (
            self.workers
            if self.workers and self.workers > 0
            else (os.cpu_count() or 1)
        )
        desc = f"[{name}] LPPLS 嵌套拟合" if name else "LPPLS 嵌套拟合"

        if n_workers <= 1 or len(tasks) == 1:
            # 单进程：主线程顺序执行（单任务时多进程无收益）
            model = lppls.LPPLS(observations=observations)
            model.filter_conditions_config = (
                model._resolve_filter_conditions_config(filter_config)
            )
            out = []
            for t in tqdm(tasks, desc=desc, leave=False):
                out.append(model._func_compute_nested_fits(t))
            return out

        # 多进程：复用持久进程池（_get_pool 懒创建）
        pool = self._get_pool(filter_config)
        async_results = [pool.apply_async(_lppls_worker_task, (t,)) for t in tasks]
        out = []
        for a in tqdm(async_results, desc=desc, leave=False):
            out.append(a.get())
        return out

    # ------------------------------------------------------------------
    # 增量结果缓存（{name}_lppls_res.pkl）
    # ------------------------------------------------------------------
    @staticmethod
    def _params_fingerprint(params, filter_config):
        """拟合参数指纹：窗口/步长/搜索次数 + 过滤条件哈希。"""
        fp = {
            k: params[k]
            for k in (
                "window_size",
                "smallest_window_size",
                "outer_increment",
                "inner_increment",
                "max_searches",
            )
            if k in params
        }
        fp["filter_hash"] = hashlib.md5(
            json.dumps(filter_config or {}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return fp

    def _res_cache_path(self, name):
        return self.output_path / f"{self._safe_name(name)}_lppls_res.pkl"

    def _load_res_cache(self, name, params, filter_config):
        """
        读取增量缓存，返回 {窗口结束ordinal: res dict}。

        缓存失效条件（返回空 dict 触发全量重算）：
            - 缓存文件不存在 / 损坏；
            - 拟合参数或过滤条件指纹不一致。
        """
        cache_path = self._res_cache_path(name)
        if not cache_path.exists():
            return {}
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            return {}
        if not isinstance(cache, dict) or "res" not in cache:
            return {}
        if cache.get("fingerprint") != self._params_fingerprint(
            params, filter_config
        ):
            return {}
        out = {}
        for r in cache["res"]:
            try:
                out[int(r["t2"])] = r
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def _save_res_cache(self, name, params, filter_config, price, res):
        """保存合并后的拟合结果缓存。"""
        cache = {
            "fingerprint": self._params_fingerprint(params, filter_config),
            "price_first": str(price.index[0].date()),
            "price_last": str(price.index[-1].date()),
            "price_len": len(price),
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "res": res,
        }
        try:
            with open(self._res_cache_path(name), "wb") as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            warnings.warn(f"[{name}] 保存增量缓存失败（不影响本次结果）: {e}")

    @staticmethod
    def _ordinal_to_date(x):
        """把 ordinal 日期转为 datetime，无效值返回 NaT。"""
        try:
            return pd.Timestamp(datetime.fromordinal(int(x)))
        except Exception:
            return pd.NaT

    def _flatten_fits(self, res):
        """
        把嵌套拟合（nested fits）的原始结果展开为平铺 DataFrame。

        每行对应一次 LPPLS 拟合（一个外层窗口位置 + 一个内层子窗口），包含：
            - date          : 外层窗口结束日（即该组拟合对应的“当前时点”）
            - outer_idx     : 外层窗口序号
            - inner_idx     : 内层子窗口序号
            - outer_t1/t2   : 外层窗口起止 ordinal 日期
            - tc/m/w/a/b/c  : LPPLS 拟合参数（b 即公式中的 B，决定泡沫方向）
            - c1/c2         : 三角函数分解后的系数
            - t1/t2         : 内层子窗口起止 ordinal 日期
            - O/D           : 振荡次数 / 阻尼系数
            - is_qualified  : 是否通过 filter_conditions_config 过滤条件
        """
        rows = []
        for outer_idx, r in enumerate(res):
            outer_t2 = r["t2"]
            date = self._ordinal_to_date(outer_t2)
            for inner_idx, fit in enumerate(r["res"]):
                rows.append(
                    {
                        "date": date,
                        "outer_idx": outer_idx,
                        "inner_idx": inner_idx,
                        "outer_t1": self._ordinal_to_date(r["t1"]),
                        "outer_t2": self._ordinal_to_date(r["t2"]),
                        "tc": self._ordinal_to_date(fit["tc"]),
                        "m": fit["m"],
                        "w": fit["w"],
                        "a": fit["a"],
                        "b": fit["b"],
                        "c": fit["c"],
                        "c1": fit["c1"],
                        "c2": fit["c2"],
                        "t1": self._ordinal_to_date(fit["t1"]),
                        "t2": self._ordinal_to_date(fit["t2"]),
                        "O": fit["O"],
                        "D": fit["D"],
                        "is_qualified": fit.get("is_qualified", False),
                    }
                )
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df


    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------
    def _threshold_signal(self, conf: pd.Series) -> pd.Series:
        """
        阈值触发信号（neg/pos 共用）：conf > bubble_threshold 当日触发 +1，
        随后连续持有 hold_days 个交易日；期间再次触发刷新持有期。
        """
        conf = conf.fillna(0.0)
        signal = pd.Series(0, index=conf.index)
        hold_remaining = 0
        for i in range(len(conf)):
            if conf.iloc[i] > self.bubble_threshold:
                hold_remaining = self.hold_days
            if hold_remaining > 0:
                signal.iloc[i] = 1
                hold_remaining -= 1
        return signal

    def _compute_signal(self, indicators):
        """
        根据兴业证券报告的负向泡沫规则生成择时信号，返回 -1 / 0 / 1 的 Series。

        规则来源
        --------
        兴业证券《如何捕捉市场趋势与极端拐点？》：
            “我们将负向泡沫置信指标作为极端下跌后市场底部反弹的提示指标。
             具体来说我们每日观察，若指标大于 0.3 则给出做多信号并延续两个月。”

        实现要点
        --------
        - 只看 neg_conf，不根据 pos_conf 做空（报告认为正向泡沫对顶部提示不可靠）。
        - neg_conf > bubble_threshold 当日触发 +1，随后连续持有 hold_days 个交易日。
        - 持有期间若再次触发，重新刷新持有期；未触发且无剩余持有时信号为 0。
        """
        return self._threshold_signal(indicators["neg_conf"])

    def _compute_pos_signal(self, indicators):
        """
        正向泡沫警示信号：pos_conf > bubble_threshold 触发 +1，观察
        hold_days 个交易日（与负向对称）。仅作超涨警示提示，策略不做空。
        """
        return self._threshold_signal(indicators["pos_conf"])

    # ------------------------------------------------------------------
    # 策略回测评估
    # ------------------------------------------------------------------
    @staticmethod
    def _perf_stats(daily_ret, n):
        """由日收益序列计算 (累计, 年化, 年化波动, Sharpe, 最大回撤)。"""
        cum = float((1 + daily_ret).prod() - 1)
        ann = (1 + cum) ** (252 / n) - 1 if n > 0 else np.nan
        vol = float(daily_ret.std() * np.sqrt(252))
        sharpe = ann / vol if vol > 0 else np.nan
        nav = (1 + daily_ret).cumprod()
        mdd = float((nav / nav.cummax() - 1).min())
        return cum, ann, vol, sharpe, mdd

    @staticmethod
    def _seg_stats(close: pd.Series, signal: pd.Series,
                   direction: str = "long") -> Dict[str, float]:
        """
        统计信号历史表现：发生次数、胜率、赔率（盈亏比）。

        口径
        ----
        - 每段连续信号区间收益 = 次日生效后的持有区间收益；
          触发日收盘只生成信号，不计入可成交收益；
        - direction='long'（做多信号，如 neg_conf 反弹）：段收益 > 0 记为胜；
        - direction='warn'（警示信号，如 pos_conf 超涨）：段收益 < 0 记为胜
          （即警示后确实回落）；
        - 赔率（盈亏比）= 平均盈利段收益绝对值 / 平均亏损段收益绝对值。
        """
        n = len(signal)
        prev = signal.shift(1).fillna(0)
        trigger_starts = np.where((signal.values == 1) & (prev.values != 1))[0]
        sig_vals = signal.values
        seg_rets = []
        for st in trigger_starts:
            en = st
            while en + 1 < n and sig_vals[en + 1] == 1:
                en += 1
            exec_st = st + 1
            if exec_st >= n:
                continue
            # 信号翻转也在收盘后可见，因此退出最早发生在下一交易日。
            exec_en = min(en + 1, n - 1)
            seg_rets.append(close.iloc[exec_en] / close.iloc[exec_st] - 1)
        seg_rets = np.asarray(seg_rets, dtype=float)

        if len(seg_rets) == 0:
            return {"n_triggers": 0, "win_rate": np.nan, "odds": np.nan}

        wins = seg_rets < 0 if direction == "warn" else seg_rets > 0
        win_rate = float(wins.mean())
        win_rets = np.abs(seg_rets[wins])
        loss_rets = np.abs(seg_rets[~wins])
        if len(win_rets) > 0 and len(loss_rets) > 0 and loss_rets.mean() > 0:
            odds = float(win_rets.mean() / loss_rets.mean())
        else:
            odds = np.nan
        return {
            "n_triggers": int(len(trigger_starts)),
            "win_rate": win_rate,
            "odds": odds,
        }

    def _evaluate_asset(self, name, close, signal):
        """
        评估单资产信号质量：策略（信号次日生效）vs 一直持有基准。

        口径说明
        --------
        - 信号在 T 日收盘后计算得到，T+1 日起生效（signal.shift(1)），
          持有段收益从 T+1 可成交价格开始，避免把触发日收盘当作成交价；
        - 年化收益按 (1+累计)^(252/n)-1 复利折算；波动/Sharpe 按 252 交易日年化；
        - 触发次数 = signal 从 0 变 1 的段数；持仓占比 = signal==1 的天数比例；
        - 胜率 = 每段连续持仓区间收益（T+1 成交 -> 信号结束后的首个可成交日）> 0 的占比。

        返回
        -------
        dict 或 None（样本不足 2 天时返回 None）
        """
        ret = close.pct_change()
        strat_ret = (ret * signal.shift(1)).fillna(0.0)
        n = len(strat_ret)
        if n < 2:
            return None

        _, strat_ann, strat_vol, strat_sharpe, strat_mdd = self._perf_stats(
            strat_ret, n
        )
        _, bench_ann, _, _, bench_mdd = self._perf_stats(ret.fillna(0.0), n)

        # 触发与持仓统计（历史发生次数 / 胜率 / 赔率）
        seg = self._seg_stats(close, signal, direction="long")
        holding_ratio = float((signal == 1).mean())

        return {
            "asset": name,
            "n_days": n,
            "n_triggers": seg["n_triggers"],
            "holding_ratio": holding_ratio,
            "win_rate": seg["win_rate"],
            "odds": seg["odds"],
            "ann_ret": strat_ann,
            "ann_vol": strat_vol,
            "sharpe": strat_sharpe,
            "max_dd": strat_mdd,
            "bench_ann_ret": bench_ann,
            "bench_max_dd": bench_mdd,
            "excess_ann": strat_ann - bench_ann,
        }

    # ------------------------------------------------------------------
    # 缓存
    # ------------------------------------------------------------------
    def _safe_name(self, name):
        return name.replace("/", "_").replace("\\", "_")

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self):
        """计算所有资产的 LPPLS 指标并输出。"""
        conf_list = []
        signal_list = []
        eval_list = []
        results = []  # 报告用：每个资产的完整数据

        for name, table, ticker, price_col in tqdm(
            self.assets, desc="LPPLS 资产计算"
        ):
            print(f"\n[{name}] {ticker} 开始计算 ...")

            try:
                price = self._load_price(table, ticker, price_col)
            except Exception as e:
                warnings.warn(f"[{name}] 读取数据失败: {e}")
                continue

            if len(price) < self.LPPLS_PARAMS["window_size"]:
                warnings.warn(
                    f"[{name}] 历史长度不足 {self.LPPLS_PARAMS['window_size']}，跳过"
                )
                continue

            # 拟合阶段单独捕获异常：lppls 库版本不兼容、矩阵奇异等错误
            # 不应静默跳过，打印完整堆栈便于排查
            try:
                indicators, fits_df = self._compute_lppls(price, name=name)
                signal = self._compute_signal(indicators)
                pos_signal = self._compute_pos_signal(indicators)
            except Exception as e:
                warnings.warn(f"[{name}] LPPLS 计算失败: {e}")
                traceback.print_exc()
                continue

            # 单个资产输出（每次运行都重写；增量缓存见 {name}_lppls_res.pkl）
            single = pd.DataFrame(
                {
                    "close": price,
                    "pos_conf": indicators["pos_conf"],
                    "neg_conf": indicators["neg_conf"],
                    "signal": signal,
                    "pos_signal": pos_signal,
                }
            )
            safe_name = self._safe_name(name)
            single.to_csv(
                self.output_path / f"{safe_name}_lppls.csv", encoding="gbk"
            )

            # 保留每次 LPPLS 拟合的原始参数，便于核对过滤条件与置信度计算
            if fits_df is not None and not fits_df.empty:
                fits_df.to_parquet(
                    self.output_path / f"{safe_name}_lppls_fits.parquet",
                    index=False,
                )

            # 回测评估（基于最新信号重算，支持阈值/持有期参数变更）
            ev = self._evaluate_asset(name, price, signal)
            pos_seg = self._seg_stats(price, pos_signal, direction="warn")
            if ev is not None:
                ev.update(
                    {
                        "pos_n_triggers": pos_seg["n_triggers"],
                        "pos_win_rate": pos_seg["win_rate"],
                        "pos_odds": pos_seg["odds"],
                    }
                )
                eval_list.append(ev)

            # 报告用结果
            results.append(
                {
                    "name": name,
                    "ticker": ticker,
                    "table": table,
                    "category": self.TABLE_CATEGORY.get(table, "其他"),
                    "close": price,
                    "indicators": indicators,
                    "signal": signal,
                    "pos_signal": pos_signal,
                    "eval": ev,
                }
            )

            # 汇总面板
            conf = indicators.rename(
                columns={"pos_conf": f"{name}_pos_conf", "neg_conf": f"{name}_neg_conf"}
            )
            conf_list.append(conf)
            sig = signal.rename(f"{name}_signal")
            signal_list.append(sig)
            signal_list.append(pos_signal.rename(f"{name}_pos_signal"))

            print(f"[{name}] 最新neg信号: {signal.iloc[-1]} | pos信号: {pos_signal.iloc[-1]} | 最新 pos_conf: {indicators['pos_conf'].iloc[-1]:.3f} | neg_conf: {indicators['neg_conf'].iloc[-1]:.3f}")

        # 循环结束，回收拟合进程池
        self._close_pool()

        if not conf_list:
            print("没有成功计算任何资产，请检查 ticker 配置。")
            return

        conf_panel = pd.concat(conf_list, axis=1)
        signal_panel = pd.concat(signal_list, axis=1)

        conf_panel.to_csv(
            self.output_path / "lppls_confidence_panel.csv", encoding="gbk"
        )
        signal_panel.to_csv(
            self.output_path / "lppls_signal_panel.csv", encoding="gbk"
        )

        # 策略回测评估汇总：信号次日生效 vs 一直持有
        if eval_list:
            eval_df = pd.DataFrame(eval_list)
            eval_df.to_csv(
                self.output_path / "lppls_backtest_summary.csv",
                encoding="gbk",
                index=False,
            )
            print("\n策略回测评估（信号次日生效，基准=一直持有）:")
            print(eval_df.to_string(index=False, float_format="{:.4f}".format))

        # 多资产共振指标：当日 neg_conf > 阈值的资产数与占比
        # （兴业报告强调多指数同时触发负向泡沫是更强的底部信号）
        neg_cols = [c for c in conf_panel.columns if c.endswith("_neg_conf")]
        neg_panel = conf_panel[neg_cols]
        n_valid = neg_panel.notna().sum(axis=1)
        n_triggered = (neg_panel > self.bubble_threshold).sum(axis=1)
        resonance = pd.DataFrame(
            {
                "n_valid": n_valid,
                "n_triggered": n_triggered,
                "trigger_ratio": (n_triggered / n_valid.replace(0, np.nan)).fillna(
                    0.0
                ),
            }
        )
        resonance.index.name = "Date"
        resonance.to_csv(
            self.output_path / "lppls_resonance.csv", encoding="gbk"
        )

        # 最新一期信号（长表：date / asset / signal_type / signal）
        latest = signal_panel.iloc[[-1]].T.reset_index()
        latest.columns = ["asset", "signal"]
        latest["signal_type"] = np.where(
            latest["asset"].str.endswith("_pos_signal"), "pos", "neg"
        )
        latest["asset"] = (
            latest["asset"]
            .str.replace("_pos_signal", "", regex=False)
            .str.replace("_signal", "", regex=False)
        )
        latest = latest[["asset", "signal_type", "signal"]]
        latest.insert(0, "date", signal_panel.index[-1])
        latest.to_csv(
            self.output_path / "lppls_latest_signal.csv",
            encoding="gbk",
            index=False,
        )

        last_res = resonance.iloc[-1]
        print(
            f"\n最新共振: {int(last_res['n_triggered'])}/{int(last_res['n_valid'])} "
            f"资产 neg_conf > {self.bubble_threshold}"
        )

        # 生成图表 + Word 每日报告（结构参考缠论报告）
        if self.generate_word and results:
            try:
                self._generate_report(results, resonance)
            except Exception as e:
                warnings.warn(f"Word 报告生成失败: {e}")
                traceback.print_exc()

        print(f"\n全部计算完成，结果保存在: {self.output_path}")

    # ------------------------------------------------------------------
    # 图表
    # ------------------------------------------------------------------
    def _plot_asset(self, r: Dict[str, Any]) -> Optional[Path]:
        """
        绘制单资产图：上图为收盘价（信号持有区间红色阴影），下图为
        pos_conf / neg_conf 置信度（阈值虚线）。

        返回
        -------
        Path 或 None（绘图失败返回 None）
        """
        name = r["name"]
        close = r["close"]
        ind = r["indicators"]
        signal = r["signal"]
        try:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 1]
            )
            # 价格 + 信号持有区间阴影（红=超跌反弹持有，橙=超涨警示）
            ax1.plot(close.index, close.values, color="#333333", lw=0.9)
            held = signal.values == 1
            ax1.fill_between(
                close.index,
                close.values.min(),
                close.values.max(),
                where=held,
                color="#d94a4a",
                alpha=0.15,
                label="超跌反弹持有区间",
            )
            pos_signal = r.get("pos_signal")
            if pos_signal is not None:
                pos_held = pos_signal.values == 1
                ax1.fill_between(
                    close.index,
                    close.values.min(),
                    close.values.max(),
                    where=pos_held,
                    color="#e69138",
                    alpha=0.15,
                    label="超涨警示区间",
                )
            ax1.set_ylabel("收盘价")
            ax1.legend(loc="upper left", fontsize=8)
            sig_txt = "反弹持有" if signal.iloc[-1] == 1 else "无反弹"
            if pos_signal is not None and pos_signal.iloc[-1] == 1:
                sig_txt += "|超涨警示"
            ax1.set_title(
                f"{name}({r['ticker']})  "
                f"neg_conf={ind['neg_conf'].iloc[-1]:.2f}  "
                f"pos_conf={ind['pos_conf'].iloc[-1]:.2f}  "
                f"当前信号={sig_txt}",
                fontsize=11,
            )
            ax1.grid(alpha=0.3)

            # 置信度
            ax2.plot(ind.index, ind["neg_conf"], color="#d94a4a", lw=1.0,
                     label="neg_conf(超跌)")
            ax2.plot(ind.index, ind["pos_conf"], color="#4a7dd9", lw=1.0,
                     label="pos_conf(超涨)")
            ax2.axhline(self.bubble_threshold, color="#888888", ls="--", lw=0.8)
            ax2.text(
                ind.index[0], self.bubble_threshold + 0.02,
                f"触发阈值 {self.bubble_threshold}", fontsize=8, color="#888888",
            )
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("置信度")
            ax2.legend(loc="upper left", fontsize=8)
            ax2.grid(alpha=0.3)

            fig.tight_layout()
            png_path = self.png_dir / f"{self._safe_name(name)}_lppls.png"
            fig.savefig(png_path, dpi=110)
            plt.close(fig)
            return png_path
        except Exception as e:
            warnings.warn(f"[{name}] 绘图失败: {e}")
            plt.close("all")
            return None

    def _plot_resonance(self, resonance: pd.DataFrame) -> Optional[Path]:
        """绘制多资产共振时序图（触发占比 + 触发数）。"""
        try:
            fig, ax = plt.subplots(figsize=(10, 3.2))
            ax.plot(
                resonance.index, resonance["trigger_ratio"] * 100,
                color="#d94a4a", lw=1.0, label="触发占比(%)",
            )
            ax.axhline(50, color="#888888", ls="--", lw=0.8)
            ax.set_ylabel("触发占比(%)")
            ax.set_ylim(0, 100)
            ax.set_title("多资产负向泡沫共振（当日 neg_conf>阈值 的资产占比）", fontsize=11)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)
            fig.tight_layout()
            png_path = self.png_dir / "lppls_resonance.png"
            fig.savefig(png_path, dpi=110)
            plt.close(fig)
            return png_path
        except Exception as e:
            warnings.warn(f"共振图绘制失败: {e}")
            plt.close("all")
            return None

    # ------------------------------------------------------------------
    # Word 报告
    # ------------------------------------------------------------------
    @staticmethod
    def _set_font(run, font_name="微软雅黑", font_size=None, bold=False, color=None):
        run.font.name = font_name
        run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
        if font_size is not None:
            run.font.size = Pt(font_size)
        run.font.bold = bold
        if color is not None:
            run.font.color.rgb = color

    @classmethod
    def _set_cell_font(cls, cell, font_name="微软雅黑", **kwargs):
        for p in cell.paragraphs:
            for run in p.runs:
                cls._set_font(run, font_name, **kwargs)

    @classmethod
    def _set_para_font(cls, paragraph, font_name="微软雅黑", **kwargs):
        for run in paragraph.runs:
            cls._set_font(run, font_name, **kwargs)

    @classmethod
    def _add_heading_with_font(cls, doc, text, level, font_size):
        h = doc.add_heading(text, level=level)
        for run in h.runs:
            cls._set_font(run, "微软雅黑", font_size=font_size, bold=True)
        return h

    @staticmethod
    def _fmt(v, fmt="{:.2f}", na="-"):
        """数值格式化，NaN/None 显示为 na。"""
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return na
        return fmt.format(v)

    @staticmethod
    def _seg_status(signal: pd.Series, close: pd.Series) -> Dict[str, Any]:
        """
        提取信号序列末尾连续持有段的状态：
        cur / held_days / new_trigger / trigger_date / seg_ret。
        """
        cur = int(signal.iloc[-1])
        held_days = 0
        for v in signal.values[::-1]:
            if v == 1:
                held_days += 1
            else:
                break
        new_trigger = bool(cur == 1 and len(signal) >= 2 and signal.iloc[-2] == 0)
        trigger_date = pd.NaT
        if held_days > 0:
            trigger_date = signal.index[len(signal) - held_days]
        seg_ret = np.nan
        if held_days > 0:
            # 信号在触发日收盘后产生，次一交易日才生效；
            # 触发日当天没有可实现的持有收益。
            st = max(len(signal) - held_days, 0)
            exec_st = st + 1
            if exec_st < len(close):
                seg_ret = float(close.iloc[-1] / close.iloc[exec_st] - 1)
        return {
            "cur": cur,
            "held_days": held_days,
            "new_trigger": new_trigger,
            "trigger_date": trigger_date,
            "seg_ret": seg_ret,
        }

    def _asset_status(self, r: Dict[str, Any]) -> Dict[str, Any]:
        """提取单资产的最新状态（用于核心信号表与汇总表）。"""
        signal = r["signal"]
        pos_signal = r["pos_signal"]
        ind = r["indicators"]
        close = r["close"]
        neg = float(ind["neg_conf"].iloc[-1])
        pos = float(ind["pos_conf"].iloc[-1])

        neg_seg = self._seg_status(signal, close)
        pos_seg = self._seg_status(pos_signal, close)

        # 近 20 日涨跌幅
        ret20 = np.nan
        if len(close) > 20:
            ret20 = float(close.iloc[-1] / close.iloc[-21] - 1)

        ev = r.get("eval") or {}
        return {
            "name": r["name"],
            "ticker": r["ticker"],
            "category": r["category"],
            "last_date": signal.index[-1],
            "last_close": float(close.iloc[-1]),
            "neg_conf": neg,
            "pos_conf": pos,
            "cur_signal": neg_seg["cur"],
            "held_days": neg_seg["held_days"],
            "new_trigger": neg_seg["new_trigger"],
            "trigger_date": neg_seg["trigger_date"],
            "seg_ret": neg_seg["seg_ret"],
            "pos_cur_signal": pos_seg["cur"],
            "pos_held_days": pos_seg["held_days"],
            "pos_new_trigger": pos_seg["new_trigger"],
            "pos_trigger_date": pos_seg["trigger_date"],
            "pos_seg_ret": pos_seg["seg_ret"],
            "ret20": ret20,
            "n_triggers": ev.get("n_triggers", np.nan),
            "win_rate": ev.get("win_rate", np.nan),
            "odds": ev.get("odds", np.nan),
            "pos_n_triggers": ev.get("pos_n_triggers", np.nan),
            "pos_win_rate": ev.get("pos_win_rate", np.nan),
            "pos_odds": ev.get("pos_odds", np.nan),
            "ann_ret": ev.get("ann_ret", np.nan),
            "bench_ann_ret": ev.get("bench_ann_ret", np.nan),
            "excess_ann": ev.get("excess_ann", np.nan),
        }

    def _generate_report(
        self, results: List[Dict[str, Any]], resonance: pd.DataFrame
    ) -> Optional[Path]:
        """生成 Word 每日报告（结构参考缠论每日分析报告）。"""
        if not DOCX_AVAILABLE:
            return None

        stats = [self._asset_status(r) for r in results]

        doc = Document()
        style = doc.styles["Normal"]
        style.font.name = "微软雅黑"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        style.font.size = Pt(10.5)

        # ======== 标题 ========
        title = doc.add_heading("LPPLS 泡沫拐点择时报告", 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in title.runs:
            self._set_font(run, "微软雅黑", font_size=18, bold=True)

        last_date = max(s["last_date"] for s in stats)
        p = doc.add_paragraph(f"数据日期：{pd.Timestamp(last_date).strftime('%Y-%m-%d')}")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self._set_para_font(p, font_name="微软雅黑", font_size=10.5)

        p = doc.add_paragraph(
            f"数据来源：本地 Wind parquet（{self.db_wind_path}）；标的 {len(stats)} 个"
            f"（指数/ETF/商品期货，清单对齐缠论分析）；每个标的回看 "
            f"{self.max_history} 个交易日；商品期货已前复权，ETF 使用复权价。"
        )
        self._set_para_font(p, font_name="微软雅黑", font_size=10.5)

        # ======== 0. 今日核心信号 ========
        self._add_heading_with_font(doc, "今日核心信号", level=1, font_size=14)
        p = doc.add_paragraph(
            "LPPLS 负向泡沫置信度 neg_conf > 0.3 视为超跌反弹信号，触发后持有 40 个"
            "交易日（规则对齐兴业证券《如何捕捉市场趋势与极端拐点？》）。"
            "正向泡沫置信度 pos_conf > 0.3 视为超涨警示信号，同样观察 40 个交易日"
            "（仅作顶部风险提示，策略不做空）。"
            f"本报告只展示当前有信号且持有 {self.hold_days} 天内的标的"
            "（均标注信号触发日），无信号或持有超期的标的不在报告中展示。"
        )
        self._set_para_font(p, font_name="微软雅黑", font_size=10.5)

        # 持有超期（> hold_days）的信号视为失效，从关注列表中剔除；
        # 重点关注今日新触发与持有期（40 天）内的信号，新信号在前、按持有天数升序
        holding_all = [s for s in stats if s["cur_signal"] == 1]
        holding = sorted(
            [s for s in holding_all if s["held_days"] <= self.hold_days],
            key=lambda x: (not x["new_trigger"], x["held_days"]),
        )
        pos_holding_all = [s for s in stats if s["pos_cur_signal"] == 1]
        pos_holding = sorted(
            [s for s in pos_holding_all if s["pos_held_days"] <= self.hold_days],
            key=lambda x: (not x["pos_new_trigger"], x["pos_held_days"]),
        )
        # 有效信号标的（neg 或 pos 任一在持有期内），报告其余部分只展示这些
        active_names = {s["name"] for s in holding} | {s["name"] for s in pos_holding}
        active_stats = [s for s in stats if s["name"] in active_names]

        # --- 表1 当前持有信号 ---
        self._add_heading_with_font(
            doc,
            f"超跌反弹持有中（{self.hold_days} 天内，新信号在前；超期已剔除）",
            level=2, font_size=12,
        )
        if holding:
            self._render_status_table(
                doc, holding,
                headers=["标的", "类别", "信号触发日", "已持有(天)", "neg_conf",
                         "触发以来涨跌", "近20日涨跌",
                         "历史触发", "历史胜率", "历史赔率"],
                row=lambda s: [
                    f"{s['name']}({s['ticker']})"
                    + ("【今日新触发】" if s["new_trigger"] else ""),
                    s["category"],
                    pd.Timestamp(s["trigger_date"]).strftime("%Y-%m-%d"),
                    str(s["held_days"]),
                    self._fmt(s["neg_conf"]),
                    self._fmt(s["seg_ret"], "{:+.2%}"),
                    self._fmt(s["ret20"], "{:+.2%}"),
                    self._fmt(s["n_triggers"], "{:.0f}次"),
                    self._fmt(s["win_rate"], "{:.0%}"),
                    self._fmt(s["odds"], "{:.2f}"),
                ],
                color_col=None,
                bold_rows=lambda s: s["new_trigger"],
            )
            n_expired = len(holding_all) - len(holding)
            if n_expired > 0:
                p = doc.add_paragraph(
                    f"另有 {n_expired} 个标的持有已超 {self.hold_days} 天（信号失效），"
                    "不再列入关注。"
                )
                self._set_para_font(p, font_name="微软雅黑", font_size=10.5)
        else:
            p = doc.add_paragraph("当前无持有中的反弹信号。")
            self._set_para_font(p, font_name="微软雅黑", font_size=10.5)

        # --- 表2 今日新触发（neg 反弹 + pos 超涨警示合并） ---
        new_triggers = sorted(
            [s for s in stats if s["new_trigger"] or s["pos_new_trigger"]],
            key=lambda x: -(x["neg_conf"] + x["pos_conf"]),
        )
        self._add_heading_with_font(doc, "今日新触发", level=2, font_size=12)
        if new_triggers:
            self._render_status_table(
                doc, new_triggers,
                headers=["标的", "类别", "方向", "触发置信度", "另一侧置信度", "最新收盘"],
                row=lambda s: [
                    f"{s['name']}({s['ticker']})",
                    s["category"],
                    ("超跌反弹" if s["new_trigger"] else "")
                    + ("超涨警示" if s["pos_new_trigger"] else ""),
                    self._fmt(
                        s["neg_conf"] if s["new_trigger"] else s["pos_conf"]
                    ),
                    self._fmt(
                        s["pos_conf"] if s["new_trigger"] else s["neg_conf"]
                    ),
                    self._fmt(s["last_close"]),
                ],
            )
        else:
            p = doc.add_paragraph("今日无新触发信号。")
            self._set_para_font(p, font_name="微软雅黑", font_size=10.5)

        # --- 表3 正向泡沫警示中（pos_conf 信号，40 天内；仅警示不做空） ---
        self._add_heading_with_font(
            doc,
            f"正向泡沫警示中（pos_conf 超涨，{self.hold_days} 天内，新信号在前；"
            "仅顶部风险提示，不做空）",
            level=2, font_size=12,
        )
        if pos_holding:
            self._render_status_table(
                doc, pos_holding,
                headers=["标的", "类别", "信号触发日", "已持有(天)", "pos_conf",
                         "触发以来涨跌", "近20日涨跌",
                         "历史触发", "警示胜率", "警示赔率"],
                row=lambda s: [
                    f"{s['name']}({s['ticker']})"
                    + ("【今日新触发】" if s["pos_new_trigger"] else ""),
                    s["category"],
                    pd.Timestamp(s["pos_trigger_date"]).strftime("%Y-%m-%d"),
                    str(s["pos_held_days"]),
                    self._fmt(s["pos_conf"]),
                    self._fmt(s["pos_seg_ret"], "{:+.2%}"),
                    self._fmt(s["ret20"], "{:+.2%}"),
                    self._fmt(s["pos_n_triggers"], "{:.0f}次"),
                    self._fmt(s["pos_win_rate"], "{:.0%}"),
                    self._fmt(s["pos_odds"], "{:.2f}"),
                ],
                bold_rows=lambda s: s["pos_new_trigger"],
            )
            n_pos_expired = len(pos_holding_all) - len(pos_holding)
            if n_pos_expired > 0:
                p = doc.add_paragraph(
                    f"另有 {n_pos_expired} 个标的超涨警示已超 {self.hold_days} 天"
                    "（信号失效），不再列入关注。"
                )
                self._set_para_font(p, font_name="微软雅黑", font_size=10.5)
        else:
            p = doc.add_paragraph("当前无持有中的超涨警示信号。")
            self._set_para_font(p, font_name="微软雅黑", font_size=10.5)

        # --- 共振解读 ---
        self._add_heading_with_font(doc, "多资产共振解读", level=2, font_size=12)
        last_res = resonance.iloc[-1]
        max_ratio = float(resonance["trigger_ratio"].max())
        max_date = resonance["trigger_ratio"].idxmax()
        lines = [
            f"最新共振：{int(last_res['n_triggered'])}/{int(last_res['n_valid'])} "
            f"个资产 neg_conf > {self.bubble_threshold}"
            f"（占比 {last_res['trigger_ratio']:.1%}）。",
            f"回看期内共振峰值出现在 {pd.Timestamp(max_date).strftime('%Y-%m-%d')}，"
            f"当日触发占比 {max_ratio:.1%}（多指数同时触发负向泡沫是更强的底部信号）。",
        ]
        for line in lines:
            p = doc.add_paragraph(line)
            self._set_para_font(p, font_name="微软雅黑", font_size=10.5)

        # ======== 一、信号标的汇总（仅有有效信号的标的） ========
        self._add_heading_with_font(
            doc,
            f"一、信号标的汇总（{self.hold_days} 天内有信号，共 {len(active_stats)} 个）",
            level=1, font_size=14,
        )
        ordered_stats = sorted(
            active_stats, key=lambda x: (x["category"], -x["neg_conf"])
        )
        if ordered_stats:
            self._render_status_table(
                doc, ordered_stats,
                headers=["标的", "类别", "最新收盘", "neg_conf", "pos_conf",
                         "反弹信号(触发日)", "超涨警示(触发日)",
                         "反弹历史(次·胜率·赔率)", "警示历史(次·胜率·赔率)"],
                row=lambda s: [
                    f"{s['name']}({s['ticker']})",
                    s["category"],
                    self._fmt(s["last_close"]),
                    self._fmt(s["neg_conf"]),
                    self._fmt(s["pos_conf"]),
                    f"第{s['held_days']}天({pd.Timestamp(s['trigger_date']).strftime('%m-%d')})"
                    if s["cur_signal"] == 1 else "-",
                    f"第{s['pos_held_days']}天({pd.Timestamp(s['pos_trigger_date']).strftime('%m-%d')})"
                    if s["pos_cur_signal"] == 1 else "-",
                    "{:.0f}次·{}·{}".format(
                        s["n_triggers"] if pd.notna(s["n_triggers"]) else 0,
                        self._fmt(s["win_rate"], "{:.0%}"),
                        self._fmt(s["odds"], "{:.2f}"),
                    ),
                    "{:.0f}次·{}·{}".format(
                        s["pos_n_triggers"] if pd.notna(s["pos_n_triggers"]) else 0,
                        self._fmt(s["pos_win_rate"], "{:.0%}"),
                        self._fmt(s["pos_odds"], "{:.2f}"),
                    ),
                ],
                bold_rows=lambda s: s["new_trigger"] or s["pos_new_trigger"],
            )
        else:
            p = doc.add_paragraph("当前无有效信号标的。")
            self._set_para_font(p, font_name="微软雅黑", font_size=10.5)

        # ======== 二、信号规则与模型说明 ========
        self._add_heading_with_font(doc, "二、信号规则与模型说明", level=1, font_size=14)
        explanations = [
            ("项目", "说明"),
            ("模型", "LPPLS（对数周期幂律奇点）：价格在拐点附近服从 "
                     "A + B·(tc-t)^m + C·(tc-t)^m·cos(ω·ln(tc-t)-φ)"),
            ("pos_conf", "正向泡沫置信度（超涨，B<0 的合格拟合占比）；"
                         f"> {self.bubble_threshold} 触发超涨警示，观察 "
                         f"{self.hold_days} 个交易日，仅作顶部风险提示、不做空"),
            ("neg_conf", "负向泡沫置信度（超跌，B>0 的合格拟合占比）"),
            ("反弹信号", f"neg_conf > {self.bubble_threshold} 触发 +1（看多反弹），"
                     f"持有 {self.hold_days} 个交易日，期间再次触发则刷新持有期"),
            ("警示信号", f"pos_conf > {self.bubble_threshold} 触发（超涨警示），"
                     f"观察 {self.hold_days} 个交易日，期间再次触发则刷新"),
            ("拟合", "回看 250 日窗口，以 5 日步长递减做嵌套拟合，统计合格拟合占比"),
            ("方向", "只做多反弹，不做空（对齐兴业证券报告结论）"),
        ]
        table = doc.add_table(rows=len(explanations), cols=2)
        table.style = "Table Grid"
        for i, (k, v) in enumerate(explanations):
            cells = table.rows[i].cells
            cells[0].text = k
            cells[1].text = v
            for j in range(2):
                self._set_cell_font(
                    cells[j], "微软雅黑", font_size=10, bold=(i == 0)
                )

        # ======== 三、图表与标的分析（仅有有效信号的标的） ========
        self._add_heading_with_font(doc, "三、图表与标的分析", level=1, font_size=14)
        p = doc.add_paragraph(
            "每张图包含：上图为收盘价（红色阴影=超跌反弹持有区间，橙色阴影=超涨"
            "警示区间）、下图为正/负向泡沫置信度（虚线为触发阈值）。"
            "仅展示当前有信号且未超期的标的；超跌反弹组在前（新信号在前），"
            "其后为仅触发超涨警示的标的。"
        )
        self._set_para_font(p, font_name="微软雅黑", font_size=10.5)

        # 组织标的顺序：反弹组沿用表1顺序；仅超涨警示组沿用表3顺序
        neg_order = {s["name"]: i for i, s in enumerate(holding)}
        pos_order = {
            s["name"]: i
            for i, s in enumerate(pos_holding)
            if s["name"] not in neg_order
        }
        hold_group = sorted(
            [r for r in results if r["name"] in neg_order],
            key=lambda r: neg_order[r["name"]],
        )
        pos_only_group = sorted(
            [r for r in results if r["name"] in pos_order],
            key=lambda r: pos_order[r["name"]],
        )
        groups = [
            (f"超跌反弹持有中（{self.hold_days} 天内，新信号在前）",
             hold_group, RGBColor(192, 0, 0)),
            (f"正向泡沫警示中（{self.hold_days} 天内，新信号在前；仅警示）",
             pos_only_group, RGBColor(191, 96, 0)),
        ]

        for section_title, group, color in groups:
            if not group:
                continue
            self._add_heading_with_font(doc, section_title, level=2, font_size=12)
            for r in group:
                s = next(x for x in stats if x["name"] == r["name"])
                png_path = self._plot_asset(r)
                if png_path is None or not png_path.exists():
                    continue

                # 标题（neg 反弹 / pos 超涨警示分别标注）
                title_text = f"{r['name']}({r['ticker']})"
                if s["cur_signal"] == 1:
                    td = pd.Timestamp(s["trigger_date"]).strftime("%m-%d")
                    title_text += f" [反弹自{td}·第{s['held_days']}天]"
                    if s["new_trigger"]:
                        title_text += "【今日新触发】"
                if s["pos_cur_signal"] == 1:
                    td = pd.Timestamp(s["pos_trigger_date"]).strftime("%m-%d")
                    title_text += f" [超涨警示自{td}·第{s['pos_held_days']}天]"
                    if s["pos_new_trigger"]:
                        title_text += "【今日新触发】"
                h = self._add_heading_with_font(doc, title_text, level=3, font_size=11)
                if color is not None:
                    for run in h.runs:
                        self._set_font(
                            run, "微软雅黑", font_size=11, bold=True, color=color
                        )

                doc.add_picture(str(png_path), width=Inches(6))

                # 数据区间 + 简析
                close = r["close"]
                range_str = (
                    f"数据区间: {close.index[0].strftime('%Y-%m-%d')} ~ "
                    f"{close.index[-1].strftime('%Y-%m-%d')}（{len(close)} 个交易日）"
                )
                analysis = self._generate_analysis_text(s)
                for text in (range_str, analysis):
                    para = doc.add_paragraph()
                    run = para.add_run(text)
                    self._set_font(run, "微软雅黑", font_size=10)

                doc.add_page_break()

        # ======== 四、历史回测评估 ========
        self._add_heading_with_font(doc, "四、历史回测评估", level=1, font_size=14)
        p = doc.add_paragraph(
            "口径：信号次日生效（避免前视偏差），基准=一直持有。"
            "反弹胜率=每次触发后持有区间收益为正的比例，反弹赔率=平均盈利段"
            "收益/平均亏损段收益（盈亏比）；警示胜率=超涨警示后区间收益为负的"
            "比例（即正确提示回落），警示赔率=平均回落幅度/平均上涨幅度。"
        )
        self._set_para_font(p, font_name="微软雅黑", font_size=10.5)
        eval_stats = sorted(
            active_stats,
            key=lambda x: -(x["excess_ann"] if not np.isnan(x["excess_ann"]) else -9),
        )
        self._render_status_table(
            doc, eval_stats,
            headers=["标的", "反弹触发", "反弹胜率", "反弹赔率",
                     "警示触发", "警示胜率", "警示赔率",
                     "策略年化", "基准年化", "超额年化"],
            row=lambda s: [
                f"{s['name']}({s['ticker']})",
                self._fmt(s["n_triggers"], "{:.0f}次"),
                self._fmt(s["win_rate"], "{:.0%}"),
                self._fmt(s["odds"], "{:.2f}"),
                self._fmt(s["pos_n_triggers"], "{:.0f}次"),
                self._fmt(s["pos_win_rate"], "{:.0%}"),
                self._fmt(s["pos_odds"], "{:.2f}"),
                self._fmt(s["ann_ret"], "{:+.2%}"),
                self._fmt(s["bench_ann_ret"], "{:+.2%}"),
                self._fmt(s["excess_ann"], "{:+.2%}"),
            ],
        )

        # ======== 五、多资产共振 ========
        self._add_heading_with_font(doc, "五、多资产共振", level=1, font_size=14)
        res_png = self._plot_resonance(resonance)
        if res_png is not None and res_png.exists():
            doc.add_picture(str(res_png), width=Inches(6))
        p = doc.add_paragraph(
            "当日 neg_conf > 阈值的资产占比。占比越高，说明越多标的同时处于"
            "超跌泡沫状态，市场整体底部的信号越强（兴业报告强调多指数共振）。"
        )
        self._set_para_font(p, font_name="微软雅黑", font_size=10.5)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = self.output_path / f"LPPLS择时报告_{timestamp}.docx"
        doc.save(str(out_file))
        print(f"Word 报告已生成: {out_file}")
        return out_file

    def _generate_analysis_text(self, s: Dict[str, Any]) -> str:
        """生成单标的简析文字（neg 反弹 + pos 超涨警示）。"""
        parts = []
        if s["cur_signal"] == 1:
            td = (
                pd.Timestamp(s["trigger_date"]).strftime("%Y-%m-%d")
                if pd.notna(s["trigger_date"]) else "-"
            )
            parts.append(
                f"超跌反弹信号于 {td} 触发，当前处于持有期第 {s['held_days']} 天，"
                f"最新 neg_conf={s['neg_conf']:.2f}"
                f"（触发以来涨跌 {self._fmt(s['seg_ret'], '{:+.2%}')}）。"
            )
        else:
            parts.append(f"当前无反弹信号，最新 neg_conf={s['neg_conf']:.2f}。")
        if s["pos_cur_signal"] == 1:
            td = (
                pd.Timestamp(s["pos_trigger_date"]).strftime("%Y-%m-%d")
                if pd.notna(s["pos_trigger_date"]) else "-"
            )
            pos_hist = ""
            if pd.notna(s["pos_win_rate"]):
                pos_hist = (
                    f"历史警示 {s['pos_n_triggers']:.0f} 次、"
                    f"警示后回落胜率 {s['pos_win_rate']:.0%}、"
                    f"警示赔率 {self._fmt(s['pos_odds'], '{:.2f}')}，"
                )
            parts.append(
                f"超涨警示信号于 {td} 触发，当前第 {s['pos_held_days']} 天，"
                f"最新 pos_conf={s['pos_conf']:.2f}"
                f"（触发以来涨跌 {self._fmt(s['pos_seg_ret'], '{:+.2%}')}），"
                f"{pos_hist}注意顶部回落风险（仅提示，不做空）。"
            )
        elif pd.notna(s["pos_conf"]) and s["pos_conf"] > 0.3:
            parts.append(
                f"注意 pos_conf={s['pos_conf']:.2f} 偏高，存在超涨泡沫特征"
                "（顶部提示仅供参考）。"
            )
        if pd.notna(s["win_rate"]):
            parts.append(
                f"回看期内反弹信号触发 {s['n_triggers']:.0f} 次、胜率 "
                f"{s['win_rate']:.0%}、赔率（盈亏比）"
                f"{self._fmt(s['odds'], '{:.2f}')}，策略年化 "
                f"{self._fmt(s['ann_ret'], '{:+.2%}')} "
                f"vs 一直持有 {self._fmt(s['bench_ann_ret'], '{:+.2%}')}。"
            )
        return "".join(parts)

    def _render_status_table(
        self,
        doc,
        items: List[Dict[str, Any]],
        headers: List[str],
        row,
        bold_rows=None,
        color_col: Optional[int] = None,
    ):
        """通用状态表渲染（参考缠论报告的表格样式）。"""
        if not items:
            return
        table = doc.add_table(rows=len(items) + 1, cols=len(headers))
        table.style = "Table Grid"
        hdr_cells = table.rows[0].cells
        for i, text in enumerate(headers):
            hdr_cells[i].text = text
            self._set_cell_font(hdr_cells[i], "微软雅黑", font_size=10, bold=True)

        for i, s in enumerate(items):
            cells = table.rows[i + 1].cells
            values = row(s)
            for j, v in enumerate(values):
                cells[j].text = str(v)
                is_bold = bold_rows is not None and bold_rows(s)
                self._set_cell_font(
                    cells[j], "微软雅黑", font_size=10, bold=is_bold
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LPPLS 择时模型")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=None,
        help="指定要计算的资产名称（如：中证1000 创业板指），默认计算全部",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=2000,
        help="每个资产使用的最近交易日数量（默认 2000）",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="起始日期（YYYYMMDD 或 YYYY-MM-DD），默认不限制；先过滤日期再截取 max-history",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="结束日期（YYYYMMDD 或 YYYY-MM-DD），默认不限制",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="LPPLS 嵌套拟合并行进程数（默认 -1 = 全部 CPU 核心；1 = 单进程）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="完整输出目录路径（默认 allocation_data/timing_lppls），测试用",
    )
    parser.add_argument(
        "--no-word",
        action="store_true",
        help="不生成 Word 报告（仅输出 CSV/图表）",
    )
    args = parser.parse_args()

    assets = None
    if args.assets:
        asset_map = {asset[0]: asset for asset in TimingLppls.DEFAULT_ASSETS}
        assets = []
        for name in args.assets:
            if name not in asset_map:
                raise ValueError(f"未知资产: {name}，可用: {list(asset_map.keys())}")
            assets.append(asset_map[name])

    # Windows spawn 多进程要求入口保护；日常增量更新建议固定 --start-date
    model = TimingLppls(
        assets=assets,
        max_history=args.max_history,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        workers=args.workers,
        generate_word=not args.no_word,
    )
    model.run()
