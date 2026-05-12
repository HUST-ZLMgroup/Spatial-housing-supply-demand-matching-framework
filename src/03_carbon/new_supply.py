"""
new_supply.py
新增住房供应面积计算

对应论文 4.4 节：
    new_supply_t = max(0, supply_t - old_supply_t)

supply_t   = 当年潜在住房总供应（XGBoost 预测）
old_supply = 存量闲置住房供应（XGBoost 预测）
new_supply = 当年新增建设产生的供应增量（不含存量）

负差值（supply < old_supply）置 0 而非 NoData，
保持栅格连续性，便于后续碳排放计算。

输入：supply{year}.tif / old_supply{year}.tif（同一目录）
输出：new_supply{year}.tif（同一目录）
"""

import os
import arcpy
from arcpy.sa import Con, Raster


def compute_new_supply(
    supply_dir: str,
    years: list[int],
    output_dir: str | None = None,
) -> dict[int, str]:
    """
    批量计算各年新增供应面积栅格。

    new_supply = max(0, supply - old_supply)

    Args:
        supply_dir: supply{year}.tif 和 old_supply{year}.tif 所在目录
        years:      目标年份列表
        output_dir: 输出目录；None 时与 supply_dir 相同

    Returns:
        {year: new_supply_tif_path}
    """
    arcpy.CheckOutExtension("Spatial")
    arcpy.env.overwriteOutput = True

    if output_dir is None:
        output_dir = supply_dir
    os.makedirs(output_dir, exist_ok=True)

    results: dict[int, str] = {}

    for year in years:
        old_path    = os.path.join(supply_dir, f"old_supply{year}.tif")
        supply_path = os.path.join(supply_dir, f"supply{year}.tif")
        out_path    = os.path.join(output_dir,  f"new_supply{year}.tif")

        if not arcpy.Exists(old_path):
            print(f"  [{year}] 缺少 {old_path}，跳过")
            continue
        if not arcpy.Exists(supply_path):
            print(f"  [{year}] 缺少 {supply_path}，跳过")
            continue

        try:
            diff   = Raster(supply_path) - Raster(old_path)
            result = Con(diff >= 0, diff, 0)   # 负值置 0（论文：安置扩增系数=0）
            result.save(out_path)
            print(f"  [{year}] new_supply → {out_path}")
            results[year] = out_path
        except Exception as exc:
            print(f"  [{year}] 计算失败: {exc}")

    arcpy.CheckInExtension("Spatial")
    print(f"\n[new_supply] 完成：{len(results)}/{len(years)} 年")
    return results