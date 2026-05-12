"""
sensitivity_sobol.py
全局敏感性分析（Sobol 方差分解）

对应论文 4.5 节 Eqs.(16)-(17)：
    S1_i = Var(E[Y|X_i]) / Var(Y)        （一阶指数）
    ST_i = 1 - Var(E[Y|X_{~i}]) / Var(Y) （全阶指数）

通过 Saltelli 采样生成多变量扰动组合，输入两层预测模型，
计算 2025–2050 年累计建筑碳排放的 Sobol 指数。

多进程策略：每个子进程独立从磁盘加载模型，避免序列化大对象。
"""

import multiprocessing as mp
import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol as sobol_analyze
    _SALIB_AVAILABLE = True
except ImportError:
    _SALIB_AVAILABLE = False
    print("[警告] SALib 未安装，请运行: pip install SALib")

from .sensitivity_ovat import (
    DEFAULT_VAR_CONFIG, CARBON_FACTORS, TARGET_YEARS
)

# --------------------------------------------------------------------------
# 子进程全局变量（每个进程初始化一次）
# --------------------------------------------------------------------------
_w_models       = None
_w_scalers      = None
_w_feature_cols = None
_w_df_base      = None
_w_years        = None
_w_ssp          = None
_w_model_dir    = None
_w_data_path    = None


def _worker_init(model_dir: str, data_path: str, ssp: str):
    """子进程初始化：加载模型和基准数据（整个进程生命周期只执行一次）。"""
    global _w_models, _w_scalers, _w_feature_cols, _w_df_base, _w_years
    global _w_ssp, _w_model_dir, _w_data_path

    from .two_layer_predictor import load_models
    _w_models, _w_scalers, _w_feature_cols = load_models(model_dir)
    _w_df_base       = pd.read_csv(data_path)
    _w_df_base["time"] = _w_df_base["time"].astype("float32")
    _w_years         = _w_df_base["time"].values.astype("float32")
    _w_ssp           = ssp
    _w_model_dir     = model_dir
    _w_data_path     = data_path


def _worker_eval_chunk(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    子进程评估函数：对一批 Saltelli 样本计算累计碳排放。

    args = (chunk_param_matrix, var_names, chunk_id, var_config)
    """
    from .two_layer_predictor import predict_new_area, calc_carbon

    chunk, var_names, chunk_id, var_config = args
    n       = chunk.shape[0]
    results = np.zeros(n)
    cf      = CARBON_FACTORS[_w_ssp]

    for i in range(n):
        if i % 500 == 0 and i > 0:
            print(f"  [pid {os.getpid()}] chunk {chunk_id}: {i}/{n}")

        # 将 Saltelli 采样的绝对扰动值转换为相对扰动率
        for j, var in enumerate(var_names):
            raw     = _w_df_base[var].values.copy()
            new_val = raw + chunk[i, j]                 # 绝对扰动
            lb      = var_config[var]["lb"]
            ub      = var_config[var]["ub"]
            if lb is not None:
                new_val = np.maximum(new_val, lb)
            if ub is not None:
                new_val = np.minimum(new_val, ub)
            _w_df_base[var] = new_val

        new_area = predict_new_area(_w_df_base, _w_models, _w_scalers, _w_feature_cols)
        total    = sum(calc_carbon(new_area, _w_years, cf, TARGET_YEARS).values())
        results[i] = total

        # 还原扰动（复用 df_base）
        for j, var in enumerate(var_names):
            _w_df_base[var] = _w_df_base[var].values / (1.0 + 0)  # 实际已覆盖，直接重读
        _w_df_base = pd.read_csv(_w_data_path)
        _w_df_base["time"] = _w_df_base["time"].astype("float32")

    return chunk_id, results


def run_sobol(
    ssp: str,
    data_path: str,
    model_dir: str,
    output_path: str,
    var_config: dict = None,
    sobol_n: int = 1000,
    n_workers: int = None,
):
    """
    对单个 SSP 情景运行 Sobol 全局敏感性分析。

    Args:
        ssp:         情景名称（ssp1~ssp5）
        data_path:   2024–2050 预测数据 CSV
        model_dir:   模型 pkl 目录
        output_path: 输出 xlsx 路径
        var_config:  变量扰动配置；None 时使用默认值
        sobol_n:     Saltelli 基础样本数（总样本数 = N × (K+2)）
        n_workers:   并行进程数；None 时使用 cpu_count - 1
    """
    if not _SALIB_AVAILABLE:
        raise ImportError("请先安装 SALib: pip install SALib")

    if var_config is None:
        var_config = DEFAULT_VAR_CONFIG
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    # 确定参与分析的变量（须在某个模型特征列中）
    from .two_layer_predictor import load_models
    _, _, feature_cols = load_models(model_dir)
    all_feat  = {c for cols in feature_cols.values() for c in cols}
    var_names = [v for v in var_config if v in all_feat]
    skipped   = [v for v in var_config if v not in all_feat]
    if skipped:
        print(f"  跳过（不在任何模型特征列中）: {skipped}")
    print(f"  参与 Sobol 分析变量数: {len(var_names)}")

    # SALib 问题定义（绝对扰动范围）
    problem = {
        "num_vars": len(var_names),
        "names":    var_names,
        "bounds":   [[-var_config[v]["range"], var_config[v]["range"]]
                     for v in var_names],
    }

    total_n = sobol_n * (len(var_names) + 2)
    print(f"  Saltelli 采样 N={sobol_n}，总样本数={total_n}")
    param_values = saltelli.sample(problem, sobol_n, calc_second_order=False)

    # 分块并行
    chunks     = np.array_split(param_values, n_workers)
    chunk_args = [
        (c, var_names, idx, var_config)
        for idx, c in enumerate(chunks)
    ]

    print(f"  启动 {n_workers} 个子进程...")
    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(model_dir, data_path, ssp),
    ) as pool:
        raw = pool.map(_worker_eval_chunk, chunk_args)

    raw.sort(key=lambda x: x[0])
    Y = np.concatenate([r for _, r in raw])
    print(f"  评估完成，共 {len(Y)} 个结果")

    # 计算 Sobol 指数
    si = sobol_analyze.analyze(
        problem, Y, calc_second_order=False, print_to_console=False
    )

    df_out = pd.DataFrame({
        "variable": var_names,
        "S1":       si["S1"],
        "S1_conf":  si["S1_conf"],
        "ST":       si["ST"],
        "ST_conf":  si["ST_conf"],
    }).sort_values("ST", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as w:
        df_out.to_excel(w, index=False, sheet_name="Sobol")
    print(f"  已保存: {output_path}")
    print(df_out.head(10).to_string(index=False))
    return df_out


def run_sobol_all_ssps(
    ssp_data: dict[str, str],
    model_dir: str,
    output_dir: str,
    var_config: dict = None,
    sobol_n: int = 1000,
    n_workers: int = None,
):
    """对所有 SSP 情景逐一运行 Sobol 分析（各情景串行，内部并行）。"""
    os.makedirs(output_dir, exist_ok=True)
    for ssp, data_path in ssp_data.items():
        print(f"\n{'='*50}\nSobol 分析: {ssp}\n{'='*50}")
        run_sobol(
            ssp=ssp,
            data_path=data_path,
            model_dir=model_dir,
            output_path=os.path.join(output_dir, f"sobol_{ssp}.xlsx"),
            var_config=var_config,
            sobol_n=sobol_n,
            n_workers=n_workers,
        )