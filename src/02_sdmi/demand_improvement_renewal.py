"""
demand_improvement_renewal.py
改进需求与更新需求计算

对应论文 4.3 节 Eq.(5)：

  改进需求（Improvement Demand）：
    Impr_Dem = Pop_{t-1} × ΔPerArea
    人均住房面积提高带来的改善性需求，ΔPerArea 固定为 0.75 m²/人/年。

  更新需求（Renewal Demand）：
    Renw_Dem = Bldg_area × demolition_rate
    既有住房拆除重建产生的更新需求，拆除率默认 0.59%。

两类需求的底版数据均来自 national_{year}_demand.shp，
该 shp 包含 10km 网格的 Pop_{year-1} 和 Bldg_area 字段。

输出：
  improvement/national_{year}_improvement_demand.shp
  renewal/national_{year}_renewal_demand.shp
"""

import os
import time
import arcpy


# --------------------------------------------------------------------------
# 默认参数（可通过函数参数或配置文件覆盖）
# --------------------------------------------------------------------------
DEFAULT_PER_AREA_GROWTH  = 0.75    # 人均住房面积年增长量（m²/人），论文 4.3
DEFAULT_DEMOLITION_RATE  = 0.0059  # 建筑年拆除率，论文 4.3


# --------------------------------------------------------------------------
# 改进需求
# --------------------------------------------------------------------------

def calculate_improvement_demand(
    year: int,
    demand_dir: str,
    output_dir: str,
    per_area_growth: float = DEFAULT_PER_AREA_GROWTH,
) -> str | None:
    """
    改进需求 = Pop_{t-1} × ΔPerArea

    Args:
        year:            目标年份
        demand_dir:      national_{year}_demand.shp 所在目录
        output_dir:      输出目录
        per_area_growth: 人均住房面积年增长量（m²/人），默认 0.75

    Returns:
        输出 shp 路径；失败返回 None
    """
    print(f"\n[impr] 计算改进需求 - {year}")

    src_shp       = os.path.join(demand_dir, f"national_{year}_demand.shp")
    pop_prev_field = f"Pop_{year - 1}"

    if not arcpy.Exists(src_shp):
        print(f"  错误：缺少 {src_shp}")
        return None
    if pop_prev_field not in {f.name for f in arcpy.ListFields(src_shp)}:
        print(f"  错误：{src_shp} 缺少 {pop_prev_field} 字段")
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_shp = os.path.join(output_dir, f"national_{year}_improvement_demand.shp")

    if arcpy.Exists(output_shp):
        arcpy.Delete_management(output_shp)

    arcpy.Copy_management(src_shp, output_shp)
    arcpy.AddField_management(output_shp, "Impr_Dem", "DOUBLE")

    total, n = 0.0, 0
    with arcpy.da.UpdateCursor(output_shp, [pop_prev_field, "Impr_Dem"]) as cur:
        for row in cur:
            pop_prev = row[0] or 0.0
            impr     = pop_prev * per_area_growth
            row[1]   = impr
            cur.updateRow(row)
            total += impr
            n     += 1

    print(f"  ΔPerArea={per_area_growth} m²/人  |  格子数={n:,}  |  全国合计={total:,.0f} m²")
    print(f"  已保存: {output_shp}")
    _print_province_summary(output_shp, year, "Impr_Dem", "改进需求")
    return output_shp


# --------------------------------------------------------------------------
# 更新需求
# --------------------------------------------------------------------------

def calculate_renewal_demand(
    year: int,
    demand_dir: str,
    output_dir: str,
    demolition_rate: float = DEFAULT_DEMOLITION_RATE,
) -> str | None:
    """
    更新需求 = Bldg_area × demolition_rate

    Args:
        year:             目标年份
        demand_dir:       national_{year}_demand.shp 所在目录
        output_dir:       输出目录
        demolition_rate:  年拆除率，默认 0.0059（0.59%）

    Returns:
        输出 shp 路径；失败返回 None
    """
    print(f"\n[renw] 计算更新需求 - {year}")

    src_shp = os.path.join(demand_dir, f"national_{year}_demand.shp")

    if not arcpy.Exists(src_shp):
        print(f"  错误：缺少 {src_shp}")
        return None
    if "Bldg_area" not in {f.name for f in arcpy.ListFields(src_shp)}:
        print(f"  错误：{src_shp} 缺少 Bldg_area 字段")
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_shp = os.path.join(output_dir, f"national_{year}_renewal_demand.shp")

    if arcpy.Exists(output_shp):
        arcpy.Delete_management(output_shp)

    arcpy.Copy_management(src_shp, output_shp)
    arcpy.AddField_management(output_shp, "Renw_Dem", "DOUBLE")

    total, n = 0.0, 0
    with arcpy.da.UpdateCursor(output_shp, ["Bldg_area", "Renw_Dem"]) as cur:
        for row in cur:
            bldg   = row[0] or 0.0
            renw   = bldg * demolition_rate
            row[1] = renw
            cur.updateRow(row)
            total += renw
            n     += 1

    print(f"  拆除率={demolition_rate}  |  格子数={n:,}  |  全国合计={total:,.0f} m²")
    print(f"  已保存: {output_shp}")
    _print_province_summary(output_shp, year, "Renw_Dem", "更新需求")
    return output_shp


# --------------------------------------------------------------------------
# 辅助：按省汇总打印
# --------------------------------------------------------------------------

def _print_province_summary(shp: str, year: int, field: str, label: str):
    stats: dict[str, float] = {}
    with arcpy.da.SearchCursor(shp, ["省", field]) as cur:
        for row in cur:
            prov = row[0] or "未匹配"
            val  = row[1] or 0.0
            stats[prov] = stats.get(prov, 0.0) + val

    total = sum(stats.values())
    print(f"\n  [{year}] {label}按省统计:")
    print(f"  {'省份':<12} {'需求(m²)':>20}")
    print("  " + "-" * 34)
    for prov, val in sorted(stats.items(), key=lambda x: -x[1])[:10]:  # 只显示前10
        print(f"  {prov:<12} {val:>20,.0f}")
    print("  " + "-" * 34)
    print(f"  {'合计':<12} {total:>20,.0f}\n")


# --------------------------------------------------------------------------
# 批量运行
# --------------------------------------------------------------------------

def run_improvement(demand_dir, output_dir, years, per_area_growth=DEFAULT_PER_AREA_GROWTH):
    """批量计算改进需求。"""
    arcpy.env.overwriteOutput = True
    print("\n" + "=" * 50 + "\n开始计算改进需求\n" + "=" * 50)
    t0 = time.time()
    for year in years:
        if not calculate_improvement_demand(year, demand_dir, output_dir, per_area_growth):
            print(f"  失败: {year}")
    print(f"\n改进需求计算完成  耗时: {time.time()-t0:.1f}s")


def run_renewal(demand_dir, output_dir, years, demolition_rate=DEFAULT_DEMOLITION_RATE):
    """批量计算更新需求。"""
    arcpy.env.overwriteOutput = True
    print("\n" + "=" * 50 + "\n开始计算更新需求\n" + "=" * 50)
    t0 = time.time()
    for year in years:
        if not calculate_renewal_demand(year, demand_dir, output_dir, demolition_rate):
            print(f"  失败: {year}")
    print(f"\n更新需求计算完成  耗时: {time.time()-t0:.1f}s")