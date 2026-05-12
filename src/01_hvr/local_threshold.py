"""
local_threshold.py
县级局地活动阈值计算


县级阈值用于校准不同地区城市规模、照明水平、
道路发育和 POI 完备性差异，避免全国统一阈值
在高/低等级城市间引入系统性偏差。

依赖：ArcPy（矢量处理）、numpy
"""

import numpy as np
import arcpy

from utils.arcpy_env import ArcpyEnvManager


# 候选县区字段（按优先级排列）
_COUNTY_FIELD_CANDIDATES = ["adcode", "gb"]


def _find_county_field(feature_class: str) -> str | None:
    """从候选字段列表中找到第一个存在于要素类的字段名。"""
    existing = {f.name for f in arcpy.ListFields(feature_class)}
    for field in _COUNTY_FIELD_CANDIDATES:
        if field in existing:
            return field
    return None


def compute_county_thresholds(
    building_with_ai: str,
    ai_field: str = "AI_VALUE",
    county_field: str | None = None,
) -> dict[str, float]:
    """
    按县级单元计算活动阈值（mean + 2σ）。

    Args:
        building_with_ai: 已连接 AI 值的建筑要素类路径
        ai_field:         AI 值字段名
        county_field:     县区编码字段名；为 None 时自动识别

    Returns:
        {county_code: threshold} 字典；
        若无有效县区字段则返回 {"__global__": global_threshold}
    """
    if county_field is None:
        county_field = _find_county_field(building_with_ai)

    # ---- 无县区字段：退回全局阈值 ----
    if county_field is None:
        print("  [threshold] 未找到县区字段，使用全局阈值")
        values = [
            row[0]
            for row in arcpy.da.SearchCursor(building_with_ai, [ai_field])
            if row[0] is not None and row[0] > 0
        ]
        if not values:
            return {"__global__": 0.0}
        arr = np.array(values)
        threshold = float(arr.mean() + 2 * arr.std())
        print(f"  [threshold] 全局阈值 = {threshold:.4f}")
        return {"__global__": threshold}

    # ---- 按县区计算 ----
    county_codes = sorted({
        row[0]
        for row in arcpy.da.SearchCursor(building_with_ai, [county_field])
        if row[0] is not None
    })
    print(f"  [threshold] 共 {len(county_codes)} 个县级单元")

    thresholds: dict[str, float] = {}
    for code in county_codes:
        where = f"{county_field} = '{code}'"
        values = [
            row[0]
            for row in arcpy.da.SearchCursor(building_with_ai, [ai_field], where)
            if row[0] is not None and row[0] > 0
        ]
        if values:
            arr = np.array(values)
            thresholds[str(code)] = float(arr.mean() + 2 * arr.std())
        else:
            thresholds[str(code)] = 0.0

    return thresholds