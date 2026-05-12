"""
demand_rigid.py
刚性住房需求计算

对应论文 4.3 节 Eq.(5) 中刚性需求部分：
    Rigid_Dem_g = max(0, Pop_t - Pop_{t-1}) × PerArea_t

人口增长带来的刚性需求：只有人口净增加的格子产生需求，
人口减少格子置 0。

输入：各年 PopCount{year}.tif（10km 网格人口栅格）
输出：rigid{year}.tif
"""

import os
import arcpy
from arcpy.sa import Con, Raster


# 各年人均住房面积（m²/人），来源：省级官方统计加权平均
PER_AREA = {
    2012: 31.81966667,
    2013: 33.433,
    2014: 33.46607143,
    2015: 33.73903226,
    2016: 35.77193548,
    2017: 36.00580645,
    2018: 37.79096774,
    2019: 38.89774194,
    2020: 38.92322581,
    2021: 40.87227273,
    2022: 41.11136364,
    2023: 40.07409091,
}


def calculate_rigid_demand(
    year: int,
    pop_dir: str,
    output_dir: str,
    per_area: dict = None,
) -> str | None:
    """
    计算单年刚性需求。

    Rigid_Dem = max(0, Pop_t - Pop_{t-1}) × PerArea_t

    Args:
        year:       目标年份（需要 year 和 year-1 的人口栅格）
        pop_dir:    人口 tif 所在目录，文件名格式 PopCount{year}.tif
        output_dir: 输出目录
        per_area:   {year: 人均住房面积} 字典；None 时使用内置默认值

    Returns:
        输出栅格路径；失败返回 None
    """
    if per_area is None:
        per_area = PER_AREA

    print(f"\n[rigid] 计算刚性需求 - {year}")

    if year not in per_area:
        print(f"  错误：per_area 中缺少 {year} 年数据")
        return None

    pa = per_area[year]

    pop_curr_path = os.path.join(pop_dir, f"PopCount{year}.tif")
    pop_prev_path = os.path.join(pop_dir, f"PopCount{year - 1}.tif")

    for p in (pop_curr_path, pop_prev_path):
        if not arcpy.Exists(p):
            print(f"  错误：找不到人口栅格 {p}")
            return None

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"rigid{year}.tif")

    if arcpy.Exists(output_path):
        arcpy.Delete_management(output_path)

    print(f"  人均住房面积: {pa} m²/人")

    pop_diff     = Raster(pop_curr_path) - Raster(pop_prev_path)
    pop_diff_pos = Con(pop_diff > 0, pop_diff, 0)   # 负值（人口减少）置 0
    result       = pop_diff_pos * pa
    result.save(output_path)

    print(f"  已保存: {output_path}")
    return output_path


def run(pop_dir: str, output_dir: str, years: list[int], per_area: dict = None):
    """批量计算刚性需求。"""
    arcpy.env.overwriteOutput = True
    arcpy.CheckOutExtension("Spatial")

    print("\n" + "=" * 50 + "\n开始计算刚性需求\n" + "=" * 50)
    for year in years:
        result = calculate_rigid_demand(year, pop_dir, output_dir, per_area)
        if not result:
            print(f"  失败: {year}")
    print("\n刚性需求计算完成")