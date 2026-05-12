"""
demand_merge.py
三类住房需求叠加合并

对应论文 4.3 节 Eq.(5) 最终汇总：
    Total_Dem_g = Rigid_Dem + Impr_Dem + Renw_Dem

将刚性需求、改进需求、更新需求三个栅格逐像元叠加求和，
输出 10km 网格尺度的年度总住房需求面积。

输入文件命名规则：
    rigid{year}.tif               （刚性需求）
    improvement_demand{year}.tif  （改进需求，已转为 tif）
    renewal_demand{year}.tif      （更新需求，已转为 tif）

输出：
    demand{year}.tif
"""

import os
import re
import arcpy
from arcpy.sa import Raster


def _scan_years(folder: str, pattern: str) -> set[int]:
    """从目录文件名中按正则提取年份集合。"""
    years = set()
    for fname in os.listdir(folder):
        m = re.match(pattern, fname, re.IGNORECASE)
        if m:
            years.add(int(m.group(1)))
    return years


def merge_demand(
    rigid_dir: str,
    improvement_dir: str,
    renewal_dir: str,
    output_dir: str,
    years: list[int] | None = None,
) -> dict[int, str]:
    """
    叠加三类需求栅格，输出年度总需求。

    Args:
        rigid_dir:       刚性需求 tif 目录
        improvement_dir: 改进需求 tif 目录
        renewal_dir:     更新需求 tif 目录
        output_dir:      输出目录
        years:           指定年份列表；None 时取三目录共有年份

    Returns:
        {year: output_tif_path} 字典
    """
    arcpy.env.overwriteOutput = True
    os.makedirs(output_dir, exist_ok=True)

    if years is None:
        y_rigid  = _scan_years(rigid_dir,       r"rigid(\d{4})\.tif$")
        y_impr   = _scan_years(improvement_dir, r"improvement_demand(\d{4})\.tif$")
        y_renew  = _scan_years(renewal_dir,     r"renewal_demand(\d{4})\.tif$")
        years    = sorted(y_rigid & y_impr & y_renew)

        if not years:
            print("未找到三目录共有年份，请检查文件命名。")
            print(f"  rigid:       {sorted(y_rigid)}")
            print(f"  improvement: {sorted(y_impr)}")
            print(f"  renewal:     {sorted(y_renew)}")
            return {}

    print(f"\n[merge] 共有年份: {years}")
    results: dict[int, str] = {}

    for year in years:
        rigid_path = os.path.join(rigid_dir,       f"rigid{year}.tif")
        impr_path  = os.path.join(improvement_dir, f"improvement_demand{year}.tif")
        renw_path  = os.path.join(renewal_dir,     f"renewal_demand{year}.tif")
        out_path   = os.path.join(output_dir,      f"demand{year}.tif")

        missing = [p for p in (rigid_path, impr_path, renw_path) if not os.path.exists(p)]
        if missing:
            print(f"  [{year}] 缺少文件，跳过:")
            for p in missing:
                print(f"    {p}")
            continue

        try:
            result = Raster(rigid_path) + Raster(impr_path) + Raster(renw_path)
            result.save(out_path)
            print(f"  [{year}] 叠加完成 → {out_path}")
            results[year] = out_path
        except Exception as exc:
            print(f"  [{year}] 叠加失败: {exc}")

    print(f"\n[merge] 完成：{len(results)}/{len(years)} 年")
    return results