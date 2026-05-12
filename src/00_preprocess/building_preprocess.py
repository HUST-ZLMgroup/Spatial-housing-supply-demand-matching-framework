"""
building_preprocess.py
建筑数据预处理：合并、投影、筛选住宅、面积计算、3σ异常值剔除

在 01_hvr（HVR计算）之前运行，输出清洗后的省级建筑 GDB，
供后续所有年份复用（建筑数据本身不随年份变化）。

处理流程：
  1. 合并省内所有 shapefile
  2. 投影至 WGS 84 / UTM Zone 49N（EPSG:32649）
  3. 筛选 Function = 'Residence' 住宅建筑
  4. 计算建筑底面积 Foot_area（㎡）
  5. 计算建筑总面积 Bldg_area = Foot_area × Height / 3
  6. 3σ 原则剔除 Bldg_area 极端值

输出：
  output_dir/<province_name>/<province_name>.gdb/<province_name>
"""

import os
import numpy as np
import arcpy


# --------------------------------------------------------------------------
# 坐标系（与全流程统一）
# --------------------------------------------------------------------------
CRS_UTM49N = arcpy.SpatialReference(32649)


# --------------------------------------------------------------------------
# 单省处理
# --------------------------------------------------------------------------

def preprocess_province(
    province_name: str,
    building_raw_dir: str,
    output_dir: str,
    sigma: float = 3.0,
) -> dict | None:
    """
    完整预处理单个省份的建筑数据。

    Args:
        province_name:    省份名称（与子目录名一致）
        building_raw_dir: 原始建筑数据根目录，结构为 <root>/<province>/*.shp
        output_dir:       输出根目录，每省生成 <output_dir>/<province>/<province>.gdb
        sigma:            异常值剔除的 σ 倍数，默认 3

    Returns:
        统计结果字典；失败时返回 None
    """
    print(f"\n{'='*50}\n处理省份: {province_name}\n{'='*50}")

    province_folder = os.path.join(building_raw_dir, province_name)

    # 1. 收集 shapefile
    shp_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(province_folder)
        for f in files if f.endswith(".shp")
    ]
    if not shp_files:
        print(f"  警告: {province_name} 无 shapefile，跳过")
        return None
    print(f"  找到 {len(shp_files)} 个 shapefile")

    try:
        # 2. 合并
        temp_merge = os.path.join(arcpy.env.scratchGDB, f"merge_{province_name}")
        arcpy.management.Merge(shp_files, temp_merge)

        # 3. 投影
        temp_proj = os.path.join(arcpy.env.scratchGDB, f"projected_{province_name}")
        arcpy.management.Project(temp_merge, temp_proj, CRS_UTM49N)
        arcpy.management.Delete(temp_merge)

        # 4. 筛选住宅
        temp_selected = os.path.join(arcpy.env.scratchGDB, f"selected_{province_name}")
        arcpy.analysis.Select(temp_proj, temp_selected, "Function = 'Residence'")
        arcpy.management.Delete(temp_proj)

        count = int(arcpy.management.GetCount(temp_selected)[0])
        if count == 0:
            print(f"  警告: {province_name} 筛选后无数据，跳过")
            arcpy.management.Delete(temp_selected)
            return None
        print(f"  筛选后住宅建筑: {count} 个")

        # 5. 创建输出 GDB 并保存
        gdb_dir  = os.path.join(output_dir, province_name)
        gdb_path = os.path.join(gdb_dir, f"{province_name}.gdb")
        os.makedirs(gdb_dir, exist_ok=True)
        if not arcpy.Exists(gdb_path):
            arcpy.management.CreateFileGDB(gdb_dir, f"{province_name}.gdb")

        output_fc = os.path.join(gdb_path, province_name)
        arcpy.management.CopyFeatures(temp_selected, output_fc)
        arcpy.management.Delete(temp_selected)

        # 6. 计算 Foot_area（建筑底面积，㎡）
        _ensure_field(output_fc, "Foot_area", "DOUBLE")
        arcpy.management.CalculateGeometryAttributes(
            output_fc, [["Foot_area", "AREA"]], area_unit="SQUARE_METERS"
        )

        # 7. 计算 Bldg_area = Foot_area × Height / 3（楼层面积估算）
        field_names = {f.name for f in arcpy.ListFields(output_fc)}
        if "Height" not in field_names:
            print(f"  警告: {province_name} 无 Height 字段，跳过")
            return None

        _ensure_field(output_fc, "Bldg_area", "DOUBLE")
        with arcpy.da.UpdateCursor(output_fc, ["Foot_area", "Height", "Bldg_area"]) as cur:
            for row in cur:
                if row[0] is not None and row[1] is not None:
                    row[2] = row[0] * row[1] / 3
                    cur.updateRow(row)

        # 8. 3σ 剔除 Bldg_area 极端值
        removed = _remove_outliers(output_fc, field="Bldg_area", sigma=sigma)
        print(f"  3σ 剔除异常建筑: {removed} 个")

        # 9. 统计
        total_area, valid_count = 0.0, 0
        with arcpy.da.SearchCursor(output_fc, ["Bldg_area"]) as cur:
            for row in cur:
                if row[0] is not None:
                    total_area += row[0]
                    valid_count += 1

        print(f"  有效建筑: {valid_count} 个  |  Bldg_area 总和: {total_area:,.0f} ㎡")
        print(f"  输出: {output_fc}")

        return {
            "province":    province_name,
            "valid_count": valid_count,
            "total_area":  total_area,
            "removed":     removed,
            "output_fc":   output_fc,
        }

    except Exception as exc:
        import traceback
        print(f"  错误: {exc}")
        traceback.print_exc()
        return None


# --------------------------------------------------------------------------
# 辅助函数
# --------------------------------------------------------------------------

def _ensure_field(fc: str, field_name: str, field_type: str):
    """若字段不存在则新建。"""
    if field_name not in {f.name for f in arcpy.ListFields(fc)}:
        arcpy.management.AddField(fc, field_name, field_type)


def _remove_outliers(fc: str, field: str, sigma: float = 3.0) -> int:
    """
    按 mean ± sigma×std 剔除极端值要素，返回删除数量。

    论文建筑面积修正步骤：利用 3σ 原则去除 Bldg_area 异常建筑。
    """
    values = [
        row[0]
        for row in arcpy.da.SearchCursor(fc, [field])
        if row[0] is not None
    ]
    if not values:
        return 0

    arr = np.array(values)
    mean, std = arr.mean(), arr.std()
    lo, hi = mean - sigma * std, mean + sigma * std
    print(f"  {field} 均值={mean:.1f}  标准差={std:.1f}  剔除范围: <{lo:.1f} 或 >{hi:.1f}")

    removed = 0
    with arcpy.da.UpdateCursor(fc, [field]) as cur:
        for row in cur:
            if row[0] is not None and (row[0] < lo or row[0] > hi):
                cur.deleteRow()
                removed += 1
    return removed


# --------------------------------------------------------------------------
# 批量处理
# --------------------------------------------------------------------------

def preprocess_all_provinces(
    building_raw_dir: str,
    output_dir: str,
    province_list: list[str] | None = None,
    sigma: float = 3.0,
) -> dict:
    """
    批量预处理所有省份建筑数据。

    Args:
        building_raw_dir: 原始建筑数据根目录
        output_dir:       输出根目录
        province_list:    指定省份列表；None 时自动扫描
        sigma:            3σ 剔除倍数

    Returns:
        {province_name: result_dict} 汇总
    """
    arcpy.env.overwriteOutput = True

    if province_list is None:
        province_list = [
            d for d in os.listdir(building_raw_dir)
            if os.path.isdir(os.path.join(building_raw_dir, d))
        ]
    print(f"共 {len(province_list)} 个省份待处理")

    results = {}
    for prov in province_list:
        result = preprocess_province(prov, building_raw_dir, output_dir, sigma)
        results[prov] = result

    # 汇总
    ok   = [p for p, r in results.items() if r is not None]
    fail = [p for p, r in results.items() if r is None]
    print(f"\n{'='*50}")
    print(f"完成: {len(ok)} 成功  {len(fail)} 失败")
    if fail:
        print(f"失败省份: {', '.join(fail)}")

    return results