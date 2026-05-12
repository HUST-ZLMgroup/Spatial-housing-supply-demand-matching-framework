"""
two_layer_predictor.py
两层级联 XGBoost 预测器

因果分析（4.5）和敏感性分析（4.5）共用的预测逻辑：

  第一层：supply / old_supply / demand 各自独立预测
  中间层：new_supply = max(supply - old_supply, 0)
  第二层：new_area（特征中包含 new_supply / old_supply / demand，
           用第一层预测值替换 df 原始值后缩放）

模型文件命名规范（存于 model_dir）：
  xgb_model_{target}.pkl
  scalers_{target}.pkl
  feature_cols_{target}.pkl
"""

import os
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

_TARGETS = ("supply", "old_supply", "demand", "new_area")

# new_area 特征矩阵中的中间变量列名
_INTER_KEYS = {"new_supply": "new_supply",
               "old_supply": "old_supply",
               "demand":     "demand"}


# --------------------------------------------------------------------------
# 模型加载
# --------------------------------------------------------------------------

def load_models(model_dir: str, fallback_csv: str = None, exclude_cols: list = None):
    """
    从 model_dir 加载四个目标的模型、scaler、feature_cols。

    Returns:
        models       {target: XGBRegressor}
        scalers      {target: {col: MinMaxScaler}}
        feature_cols {target: [col, ...]}
    """
    print(f"[loader] 从 {model_dir} 加载模型...")
    models, scalers, feature_cols = {}, {}, {}

    for target in _TARGETS:
        model_path = os.path.join(model_dir, f"xgb_model_{target}.pkl")
        sc_path    = os.path.join(model_dir, f"scalers_{target}.pkl")
        feat_path  = os.path.join(model_dir, f"feature_cols_{target}.pkl")

        # 兼容不带后缀命名
        if not os.path.exists(sc_path):
            sc_path = os.path.join(model_dir, "scalers.pkl")
        if not os.path.exists(feat_path):
            feat_path = os.path.join(model_dir, "feature_cols.pkl")

        models[target] = _load_pkl(model_path)

        sc_obj = _load_pkl(sc_path)
        scalers[target] = sc_obj if isinstance(sc_obj, dict) else {target: sc_obj}

        if os.path.exists(feat_path):
            feature_cols[target] = _load_pkl(feat_path)
        elif fallback_csv and os.path.exists(fallback_csv):
            print(f"  [警告] feature_cols_{target}.pkl 缺失，从 CSV 推断")
            df_tmp = pd.read_csv(fallback_csv)
            feature_cols[target] = [
                c for c in df_tmp.columns if c not in (exclude_cols or [])
            ]
        else:
            raise FileNotFoundError(f"找不到 feature_cols_{target}.pkl 且无 fallback_csv")

        print(f"  {target}: {len(feature_cols[target])} 特征")

    _check_scalers(models, scalers, feature_cols)
    print("[loader] 加载完毕\n")
    return models, scalers, feature_cols


def _check_scalers(models, scalers, feature_cols):
    """验证每个模型的 scaler 覆盖了对应的全部特征列。"""
    errors = []
    for target in _TARGETS:
        if target not in models:
            continue
        for col in feature_cols.get(target, []):
            if col not in scalers.get(target, {}):
                errors.append(f"scalers['{target}'] 缺少列 '{col}'")
    if errors:
        raise KeyError("scaler 完整性检查失败:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------
# 特征矩阵构造
# --------------------------------------------------------------------------

def build_X(
    df: pd.DataFrame,
    feature_cols: list[str],
    scalers_for_target: dict,
    use_log: bool = True,
    log_eps: float = 1e-6,
    perturb_var: str = None,
    perturb_rate: float = 0.0,
    lb=None,
    ub=None,
) -> np.ndarray:
    """
    为第一层模型构造特征矩阵。

    1. 从 df 取 feature_cols 对应的原始值
    2. 可选对 perturb_var 施加扰动并截断到物理边界
    3. 用 scalers_for_target[col] 逐列 transform（不 fit）
    4. 可选 log(x + eps) 变换

    Returns:
        float32 矩阵，形状 (n_samples, len(feature_cols))
    """
    raw = df[feature_cols].values.astype(np.float64)

    if perturb_var and perturb_var in feature_cols and perturb_rate != 0.0:
        vi = feature_cols.index(perturb_var)
        raw[:, vi] *= 1.0 + perturb_rate
        if lb is not None:
            raw[:, vi] = np.maximum(raw[:, vi], lb)
        if ub is not None:
            raw[:, vi] = np.minimum(raw[:, vi], ub)

    X = np.column_stack([
        scalers_for_target[col].transform(raw[:, [i]])
        for i, col in enumerate(feature_cols)
    ])
    if use_log:
        X = np.log(X + log_eps)
    return X.astype(np.float32)


# --------------------------------------------------------------------------
# 两层级联预测（论文 4.4/4.5 核心）
# --------------------------------------------------------------------------

def predict_new_area(
    df: pd.DataFrame,
    models: dict,
    scalers: dict,
    feature_cols: dict,
    use_log: bool = True,
    log_eps: float = 1e-6,
    perturb_var: str = None,
    perturb_rate: float = 0.0,
    lb=None,
    ub=None,
) -> np.ndarray:
    """
    两层级联预测，返回 new_area 预测值（非负）。

    第一层各模型独立构造输入，扰动自动传播到各层；
    new_area 中的中间变量列用第一层预测值替换，其余普通列含扰动。

    Args:
        df:           包含所有特征列的 DataFrame（未缩放原始值）
        perturb_var:  施加扰动的变量名；None 表示不扰动（baseline）
        perturb_rate: 扰动幅度（相对值，如 0.1 = +10%）
        lb / ub:      物理边界（截断用）

    Returns:
        new_area 预测值，shape (n_samples,)，已 clip 至非负
    """
    pk = dict(perturb_var=perturb_var, perturb_rate=perturb_rate, lb=lb, ub=ub)

    # 第一层
    supply_pred     = np.maximum(models["supply"].predict(
        build_X(df, feature_cols["supply"],     scalers["supply"],     use_log, log_eps, **pk)), 0)
    old_supply_pred = np.maximum(models["old_supply"].predict(
        build_X(df, feature_cols["old_supply"], scalers["old_supply"], use_log, log_eps, **pk)), 0)
    demand_pred     = np.maximum(models["demand"].predict(
        build_X(df, feature_cols["demand"],     scalers["demand"],     use_log, log_eps, **pk)), 0)
    new_supply_pred = np.maximum(supply_pred - old_supply_pred, 0.0)

    # 第二层：逐列构造 X
    sc_na = scalers["new_area"]
    inter_map = {
        _INTER_KEYS["new_supply"]: new_supply_pred,
        _INTER_KEYS["old_supply"]: old_supply_pred,
        _INTER_KEYS["demand"]:     demand_pred,
    }

    X_cols = []
    for col in feature_cols["new_area"]:
        if col in inter_map:
            normed = sc_na[col].transform(inter_map[col].reshape(-1, 1)).flatten()
        else:
            raw = df[col].values.astype(np.float64)
            if perturb_var == col and perturb_rate != 0.0:
                raw = raw * (1.0 + perturb_rate)
                if lb is not None:
                    raw = np.maximum(raw, lb)
                if ub is not None:
                    raw = np.minimum(raw, ub)
            normed = sc_na[col].transform(raw.reshape(-1, 1)).flatten()
        if use_log:
            normed = np.log(normed + log_eps)
        X_cols.append(normed)

    X_area = np.column_stack(X_cols).astype(np.float32)
    return np.maximum(models["new_area"].predict(X_area), 0.0)


# --------------------------------------------------------------------------
# 碳排放计算（论文 4.4 Eq.16，5年滚动窗口在调用方处理）
# --------------------------------------------------------------------------

def calc_carbon(
    new_area: np.ndarray,
    years: np.ndarray,
    carbon_factors: dict[int, float],
    target_years: list[int],
) -> dict[int, float]:
    """
    按年份计算碳排放。

    carbon_t = new_area_t × CI_t

    Args:
        new_area:       new_area 预测值数组（与 years 等长）
        years:          每行对应的年份数组
        carbon_factors: {year: CI (kgCO₂/m²)}，来自 carbon_intensity 模块
        target_years:   统计的目标年份列表

    Returns:
        {year: carbon_emission}
    """
    return {
        yr: float(new_area[years == yr].sum()) * carbon_factors.get(yr, float("nan"))
        for yr in target_years
    }


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------

def _load_pkl(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到: {path}")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)