"""
housing_supply.py
潜在住房供应面积计算

对应论文 4.3 节 Eq.(6)：
    Supply_b = Bldg_area_corrected × HVR_b

其中 Bldg_area_corrected 为经省级官方人均住房面积校正后的建筑楼层面积：
    Bldg_area_corrected = Bldg_area_raw × (PerArea_official × Pop_province) / Bldg_area_sum_province

校正系数（Correction_Factor）从外部 Excel 表读取，格式：
    national_correction_{year}.xlsx  → 列：Province, Correction_Factor

supply_area = Bldg_area × HVR × Correction_Factor

输出字段 supply_area 写入建筑要素类（省级 GDB），后续汇总至 10km 网格 tif。
"""

import os
import arcpy
import pandas as pd
from multiprocessing import Pool, cpu_count


# --------------------------------------------------------------------------
# 单 GDB 处理（支持多进程）
# --------------------------------------------------------------------------

def _process_single_gdb(args: tuple) -> str:
    """
    计算单个省份 GDB 中的 supply_area 字段。

    supply_area = Bldg_area × HVR × Correction_Factor  (Eq. 6)

    Args (tuple):
        gdb_path, gdb_name, full_year, corr_dict

    Returns:
        状态字符串
    """
    gdb_path, gdb_name, full_year, corr_dict = args

    try:
        if not arcpy.Exists(gdb_path):
            return f"GDB 不存在: {gdb_path}"

        province = gdb_name[:-4]  # 去掉 .gdb

        arcpy.env.workspace = gdb_path
        fcs = arcpy.ListFeatureClasses()
        if not fcs:
            return f"警告: {gdb_path} 无要素类"

        fc_path = os.path.join(gdb_path, fcs[0])

        if province not in corr_dict:
            return f"警告: 修正系数中未找到省份 {province}"

        correction = corr_dict[province]
        if pd.isna(correction):
            return f"警告: {province} {full_year} 年修正系数为 NaN"

        # 确保 supply_area 字段存在
        if "supply_area" not in {f.name for f in arcpy.ListFields(fc_path)}:
            arcpy.AddField_management(fc_path, "supply_area", "DOUBLE")

        # supply_area = Bldg_area × HVR × correction
        arcpy.CalculateField_management(
            fc_path,
            "supply_area",
            f"!Bldg_area! * !HVR! * {correction}",
            "PYTHON3",
        )
        return f"✓ {province} {full_year}: supply_area 计算完成 (修正系数={correction:.4f})"

    except Exception as exc:
        return f"✗ {gdb_path}: {exc}"


# --------------------------------------------------------------------------
# 按年批量处理
# --------------------------------------------------------------------------

def calculate_supply_by_year(
    hvr_root: str,
    correction_dir: str,
    year: int,
    n_workers: int | None = None,
) -> list[str]:
    """
    计算某年所有省份的 supply_area。

    Args:
        hvr_root:        HVR 结果根目录，结构：<root>/<year>/<province>.gdb
        correction_dir:  修正系数 Excel 所在目录
        year:            目标年份（两位或四位均可，内部统一处理为四位）
        n_workers:       并行进程数；None 时取 min(cpu//2, 4)

    Returns:
        每个 GDB 的处理结果字符串列表
    """
    # 支持两位年份（13→2013）
    full_year  = year if year > 2000 else 2000 + year
    short_year = full_year - 2000

    year_folder = os.path.join(hvr_root, str(short_year))
    if not os.path.exists(year_folder):
        print(f"[supply] 找不到年份目录: {year_folder}")
        return []

    # 读取修正系数表
    corr_file = os.path.join(correction_dir, f"national_correction_{full_year}.xlsx")
    if not os.path.exists(corr_file):
        print(f"[supply] 修正系数文件缺失: {corr_file}")
        return []

    df = pd.read_excel(corr_file)
    df.columns = df.columns.str.strip()
    if "Province" not in df.columns or "Correction_Factor" not in df.columns:
        print(f"[supply] {corr_file} 缺少 Province 或 Correction_Factor 列")
        return []

    corr_dict = dict(zip(df["Province"], df["Correction_Factor"]))

    # 收集 GDB 任务
    tasks = [
        (os.path.join(year_folder, f), f, full_year, corr_dict)
        for f in os.listdir(year_folder)
        if f.endswith(".gdb")
    ]
    if not tasks:
        print(f"[supply] {year_folder} 无 GDB 文件")
        return []

    print(f"\n[supply] {full_year} 年：{len(tasks)} 个省份，并行处理...")

    workers = n_workers or min(cpu_count() // 2, 4)
    with Pool(processes=workers) as pool:
        results = pool.map(_process_single_gdb, tasks)

    for r in results:
        print(f"  {r}")

    return results


def run(
    hvr_root: str,
    correction_dir: str,
    years: list[int],
    n_workers: int | None = None,
):
    """批量计算所有年份的潜在住房供应。"""
    print("\n" + "=" * 50 + "\n开始计算潜在住房供应\n" + "=" * 50)
    for year in years:
        calculate_supply_by_year(hvr_root, correction_dir, year, n_workers)
    print("\n潜在住房供应计算完成")