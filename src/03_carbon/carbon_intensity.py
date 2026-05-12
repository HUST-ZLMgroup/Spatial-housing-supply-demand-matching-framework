"""
carbon_intensity.py
建筑建造碳排放强度查找与插值

对应论文 4.4 节 Eq.(9)：
    CI_k = 由材料生产排放因子和施工终端能源碳强度共同决定，
           已按 SSP 情景预计算为节点年份强度值（单位：kgCO₂/m²）

非节点年份（如 2026、2031 等）通过线性插值获得。
2013–2019 年历史期使用 2020 年强度值（假设与基准年一致）。

强度表来源：ssp_config.yaml → carbon_intensity 字段
"""

import numpy as np


# 节点年份（与配置表对齐）
_ANCHOR_YEARS = [2020, 2025, 2030, 2035, 2040, 2045, 2050]

# 内置强度表（kgCO₂/m²），与 ssp_config.yaml 保持一致
_INTENSITY_TABLE = {
    "ssp1": [502.1037375, 458.5658774, 412.4512823, 400.9534834, 378.7082488, 368.7131579, 357.7070138],
    "ssp2": [502.1037375, 482.7798522, 469.5382931, 460.9387871, 450.765133,  443.1567752, 434.3226372],
    "ssp3": [502.1037375, 482.998699,  469.9591812, 461.5458794, 451.5435112, 444.0923924, 435.402273 ],
    "ssp4": [502.1037375, 482.0541773, 468.1517746, 458.9518954, 448.2342395, 440.1343992, 430.8576787],
    "ssp5": [502.1037375, 482.6342483, 469.2589712, 460.5369055, 450.2511617, 442.5405336, 433.6133296],
}


def get_intensity(year: int, scenario: str) -> float:
    """
    获取指定年份和情景的单位建筑面积碳排放强度（kgCO₂/m²）。

    节点年份直接查表，非节点年份线性插值。
    历史期（year < 2020）使用 2020 年基准值。

    Args:
        year:     目标年份
        scenario: SSP 情景名称（ssp1 / ssp2 / ssp3 / ssp4 / ssp5）

    Returns:
        碳排放强度（kgCO₂/m²）
    """
    scenario = scenario.lower()
    if scenario not in _INTENSITY_TABLE:
        raise ValueError(f"未知情景: {scenario}，可选: {list(_INTENSITY_TABLE.keys())}")

    intensities = _INTENSITY_TABLE[scenario]

    # 历史期用 2020 基准值
    if year <= 2020:
        return intensities[0]

    # 超出预测期末用末尾值
    if year >= 2050:
        return intensities[-1]

    # 线性插值
    return float(np.interp(year, _ANCHOR_YEARS, intensities))


def get_intensity_series(years: list[int], scenario: str) -> dict[int, float]:
    """
    批量获取年份序列的碳排放强度。

    Returns:
        {year: intensity} 字典
    """
    return {y: get_intensity(y, scenario) for y in years}


def load_from_config(cfg: dict) -> dict[str, dict[int, float]]:
    """
    从配置字典（ssp_config.yaml 加载结果）读取强度表，
    返回 {scenario: {year: intensity}}。

    Args:
        cfg: 包含 carbon_intensity.years 和 carbon_intensity.<ssp> 的配置字典
    """
    ci = cfg.get("carbon_intensity", {})
    anchor_years = ci.get("years", _ANCHOR_YEARS)
    result: dict[str, dict[int, float]] = {}

    for key, vals in ci.items():
        if key == "years":
            continue
        result[key] = dict(zip(anchor_years, vals))

    return result