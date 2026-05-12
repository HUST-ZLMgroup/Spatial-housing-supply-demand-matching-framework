"""
sensitivity_ovat.py
单变量敏感性分析（One-At-a-Time, OAT）

对应论文 4.5 节 Eq.(15)：
    ΔC_k = (C_perturbed - C_baseline) / C_baseline

策略：一次扰动一个变量，其余保持不变。
扰动区间由历史与 SSP 预测序列的变异系数分布确定，并结合物理边界修正。
扰动步长：在 (0, range] 内均匀取 N_RATE_HALF 个正值，取反得负值，
          共 2×N_RATE_HALF 个严格对称值（不含 0）。

输出：每个变量一个 xlsx 文件（宽表：行=扰动幅度，列=ssp1~ssp5）
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# VAR_CONFIG 默认值（与 Sobol 共用，来源：各变量历史+SSP预测序列的变异系数）
DEFAULT_VAR_CONFIG = {
    "RdDens":     {"range": 0.245254, "lb": 0,    "ub": None},
    "SchDist":    {"range": 0.456214, "lb": 0,    "ub": None},
    "HospDist":   {"range": 0.392249, "lb": 0,    "ub": None},
    "BusDist":    {"range": 0.364719, "lb": 0,    "ub": None},
    "RailDist":   {"range": 0.309251, "lb": 0,    "ub": None},
    "MallDist":   {"range": 0.364730, "lb": 0,    "ub": None},
    "WaterCov":   {"range": 0.025724, "lb": 0,    "ub": 1   },
    "SewTreat":   {"range": 0.048732, "lb": 0,    "ub": 1   },
    "GasCov":     {"range": 0.067410, "lb": 0,    "ub": 1   },
    "WasteTreat": {"range": 0.040413, "lb": 0,    "ub": 1   },
    "CleanArea":  {"range": 0.180290, "lb": 0,    "ub": None},
    "PenIns":     {"range": 0.174015, "lb": 0,    "ub": None},
    "UnempIns":   {"range": 0.152495, "lb": 0,    "ub": None},
    "MedIns":     {"range": 0.667420, "lb": 0,    "ub": None},
    "IndFirm":    {"range": 1.121991, "lb": 0,    "ub": None},
    "WTD":        {"range": 0.086762, "lb": None, "ub": None},
    "HDS":        {"range": 0.172306, "lb": None, "ub": None},
    "LAI":        {"range": 0.139450, "lb": 0,    "ub": None},
    "Gini":       {"range": 0.040237, "lb": 0,    "ub": 1   },
    "PriInd":     {"range": 0.348962, "lb": 0,    "ub": None},
    "SecInd":     {"range": 0.470375, "lb": 0,    "ub": None},
    "TerInd":     {"range": 0.527141, "lb": 0,    "ub": None},
}

# 碳排放强度表（kgCO₂/m²），与 carbon_intensity.py 保持一致
CARBON_FACTORS = {
    "ssp1": {2020:502.1037375,2025:458.5658774,2030:412.4512823,2035:400.9534834,
             2040:378.7082488,2045:368.7131579,2050:357.7070138},
    "ssp2": {2020:502.1037375,2025:482.7798522,2030:469.5382931,2035:460.9387871,
             2040:450.765133, 2045:443.1567752,2050:434.3226372},
    "ssp3": {2020:502.1037375,2025:482.998699, 2030:469.9591812,2035:461.5458794,
             2040:451.5435112,2045:444.0923924,2050:435.402273 },
    "ssp4": {2020:502.1037375,2025:482.0541773,2030:468.1517746,2035:458.9518954,
             2040:448.2342395,2045:440.1343992,2050:430.8576787},
    "ssp5": {2020:502.1037375,2025:482.6342483,2030:469.2589712,2035:460.5369055,
             2040:450.2511617,2045:442.5405336,2050:433.6133296},
}

TARGET_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]


def _make_rates(r_range: float, n_half: int) -> np.ndarray:
    """生成 2×n_half 个对称扰动率（不含 0）。"""
    pos = np.linspace(r_range / n_half, r_range, n_half)
    return np.sort(np.concatenate([-pos, pos]))


def _process_one_ssp(args: tuple) -> tuple[str, dict]:
    """
    子进程入口：处理单个 SSP 的全部变量 × rate 计算。
    模型在子进程内自行加载，避免大对象跨进程序列化。
    """
    ssp, data_path, model_dir, var_config, n_rate_half = args

    # 延迟导入（子进程）
    import pandas as pd
    import numpy as np
    from src.utils.arcpy_env import ArcpyEnvManager  # noqa: 仅占位，实际不用
    from two_layer_predictor import (
        load_models, predict_new_area, calc_carbon,
    )

    if not os.path.exists(data_path):
        print(f"  [{ssp}] 文件不存在: {data_path}")
        return ssp, {}

    models, scalers, feature_cols = load_models(model_dir)
    df      = pd.read_csv(data_path)
    df["time"] = df["time"].astype("float32")
    years   = df["time"].values.astype("float32")
    cf      = CARBON_FACTORS[ssp]

    # baseline
    na_base     = predict_new_area(df, models, scalers, feature_cols)
    carbon_base = sum(calc_carbon(na_base, years, cf, TARGET_YEARS).values())
    print(f"  [{ssp}] baseline = {carbon_base:.2f}")

    all_feat = {c for cols in feature_cols.values() for c in cols}
    results  = {}

    for var, vcfg in var_config.items():
        if var not in all_feat:
            continue
        rates     = _make_rates(vcfg["range"], n_rate_half)
        var_res   = {0.0: 0.0}
        for rate in rates:
            na_pert = predict_new_area(
                df, models, scalers, feature_cols,
                perturb_var=var, perturb_rate=float(rate),
                lb=vcfg["lb"], ub=vcfg["ub"],
            )
            c_pert = sum(calc_carbon(na_pert, years, cf, TARGET_YEARS).values())
            var_res[float(rate)] = (
                (c_pert - carbon_base) / carbon_base if carbon_base else float("nan")
            )
        results[var] = var_res
        print(f"  [{ssp}] {var} 完成")

    return ssp, results


def run_sensitivity_ovat(
    ssp_data: dict[str, str],
    model_dir: str,
    output_dir: str,
    var_config: dict = None,
    n_rate_half: int = 10,
    n_jobs: int = 5,
):
    """
    批量单变量敏感性分析（OAT）。

    Args:
        ssp_data:    {ssp: predict_csv_path}
        model_dir:   模型 pkl 所在目录
        output_dir:  输出目录（每变量一个 xlsx）
        var_config:  变量扰动配置；None 时使用默认值
        n_rate_half: 单侧扰动步数（总共 2×n_rate_half 个 rate）
        n_jobs:      并行进程数
    """
    if var_config is None:
        var_config = DEFAULT_VAR_CONFIG
    os.makedirs(output_dir, exist_ok=True)

    tasks = [
        (ssp, path, model_dir, var_config, n_rate_half)
        for ssp, path in ssp_data.items()
    ]

    var_data: dict[str, dict] = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(tasks))) as pool:
        futures = {pool.submit(_process_one_ssp, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            ssp, ssp_res = fut.result()
            for var, rate_dict in ssp_res.items():
                var_data[var][ssp] = rate_dict
            print(f"✓ {ssp} 汇总完成")

    # 输出宽表 xlsx
    ssp_list = sorted(ssp_data.keys())
    for var in var_config:
        if var not in var_data:
            continue
        all_rates = sorted({r for d in var_data[var].values() for r in d})
        rows = [
            {"rate": r, **{ssp: var_data[var].get(ssp, {}).get(r, float("nan"))
                           for ssp in ssp_list}}
            for r in all_rates
        ]
        path = os.path.join(output_dir, f"sensitivity_{var}.xlsx")
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            df_out = pd.DataFrame(rows)
            df_out.to_excel(w, index=False, sheet_name=var)
        print(f"  已保存: {path}")

    print(f"\nOAT 敏感性分析完成，共 {len(var_data)} 个变量")