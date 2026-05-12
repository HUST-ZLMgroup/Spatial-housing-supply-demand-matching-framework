"""
hvr_estimate.py
建筑尺度住房空置率（HVR）代理值计算

当建筑 AI 接近或超过县域阈值时 HVR → 0（潜在空置程度低）；
AI 较低时 HVR → 1（潜在空置程度高）。
无有效 AI 或无法计算阈值的建筑 HVR 设为 1。

依赖：ArcPy、numpy
"""

import os
import numpy as np
import arcpy
from arcpy.sa import ExtractMultiValuesToPoints

from utils.arcpy_env import ArcpyEnvManager
from .local_threshold import compute_county_thresholds

_COUNTY_FIELD_CANDIDATES = ["adcode", "gb"]


# --------------------------------------------------------------------------
# 建筑数据预处理
# --------------------------------------------------------------------------

def merge_buildings(
    province_name: str,
    building_base: str,
    workspace: str,
    env: ArcpyEnvManager,
    batch_size: int = 30,
) -> str | None:
    """
    合并省内所有建筑 shapefile，大省分批处理后再合并。

    Returns:
        合并后的要素类路径，找不到数据时返回 None
    """
    print(f"  [building] 合并 {province_name} 建筑数据...")
    building_folder = os.path.join(building_base, province_name)

    old_ws = arcpy.env.workspace
    arcpy.env.workspace = building_folder
    shp_list = arcpy.ListFeatureClasses("*.shp")
    arcpy.env.workspace = old_ws

    if not shp_list:
        print(f"  [building] 警告：{province_name} 无建筑数据")
        return None

    paths = [os.path.join(building_folder, f) for f in shp_list]
    merged = os.path.join(workspace, "buildings_merged")
    env.track(merged)

    if len(paths) > batch_size:
        print(f"  [building] {len(paths)} 个文件，分批合并...")
        batches = []
        for i in range(0, len(paths), batch_size):
            batch_out = os.path.join(workspace, f"temp_merge_{i}")
            env.track(batch_out)
            arcpy.Merge_management(paths[i : i + batch_size], batch_out)
            batches.append(batch_out)
            print(f"  [building] 批次 {i // batch_size + 1} 完成")
        arcpy.Merge_management(batches, merged)
        for b in batches:
            env.safe_delete(b)
    else:
        arcpy.Merge_management(paths, merged)

    count = arcpy.GetCount_management(merged)[0]
    print(f"  [building] 合并完成，共 {count} 个建筑")
    return merged


def filter_residential(
    merged_building: str,
    workspace: str,
    env: ArcpyEnvManager,
) -> str:
    """
    投影至 UTM49N，筛选住宅建筑（Function = 'Residence'）。

    Returns:
        住宅建筑要素类路径
    """
    projected = env.track(os.path.join(workspace, "building_projected"))
    arcpy.Project_management(merged_building, projected, env.CRS_UTM49N)
    env.safe_delete(merged_building)

    residential = env.track(os.path.join(workspace, "building_residential"))
    try:
        arcpy.Select_analysis(projected, residential, "\"Function\" = 'Residence'")
        count = arcpy.GetCount_management(residential)[0]
        print(f"  [building] 筛选住宅建筑：{count} 个")
    except Exception:
        print("  [building] 筛选失败，使用全部建筑")
        arcpy.CopyFeatures_management(projected, residential)

    env.safe_delete(projected)
    return residential


# --------------------------------------------------------------------------
# HVR 核心计算（Eq. 4）
# --------------------------------------------------------------------------

def compute_hvr(
    building_path: str,
    ai_raster_path: str,
    province_boundary: str,
    county_boundary: str,
    workspace: str,
    province_name: str,
    env: ArcpyEnvManager,
) -> tuple[str, dict]:
    """
    计算建筑尺度 HVR 代理值。

    流程：
      1. 建筑面 → 质心点
      2. 从 AI 栅格提取点值
      3. 空间连接县区编码
      4. 按县区计算阈值（Eq. 3）
      5. HVR = max(0, min(1, 1 - AI / threshold))  (Eq. 4)

    Args:
        building_path:     住宅建筑要素类
        ai_raster_path:    AI 栅格路径
        province_boundary: 省级边界（用于空间参考投影）
        county_boundary:   含县区编码的边界要素类
        workspace:         GDB 工作空间路径
        province_name:     省份名（最终输出要素类名）
        env:               ArcPy 环境管理器

    Returns:
        (output_feature_class_path, hvr_stats_dict)
    """
    print("  [hvr] 计算建筑尺度 HVR...")

    # 1. 建筑面 → 质心点
    print("  [hvr] 建筑面转质心点...")
    building_pts = env.track(os.path.join(workspace, "building_points"))
    try:
        arcpy.RepairGeometry_management(building_path)
    except Exception:
        pass
    arcpy.FeatureToPoint_management(building_path, building_pts, "INSIDE")

    # 2. 提取 AI 栅格值
    print("  [hvr] 提取 AI 值...")
    ExtractMultiValuesToPoints(building_pts, [[ai_raster_path, "AI_VALUE"]], "NONE")

    # 填补无效值为 0
    with arcpy.da.UpdateCursor(building_pts, ["AI_VALUE"]) as cur:
        for row in cur:
            if row[0] is None or row[0] == -9999:
                row[0] = 0
            cur.updateRow(row)

    # 3. 空间连接县区编码
    print("  [hvr] 空间连接县区编码...")
    county_proj = env.track(os.path.join(workspace, "county_projected"))
    arcpy.Project_management(county_boundary, county_proj, env.CRS_UTM49N)

    pts_with_county = env.track(os.path.join(workspace, "pts_with_county"))
    arcpy.SpatialJoin_analysis(
        building_pts, county_proj, pts_with_county,
        "JOIN_ONE_TO_ONE", "KEEP_ALL", match_option="WITHIN",
    )
    env.safe_delete(building_pts)

    building_with_ai = env.track(os.path.join(workspace, "building_with_ai"))
    arcpy.SpatialJoin_analysis(
        building_path, pts_with_county, building_with_ai,
        "JOIN_ONE_TO_ONE", "KEEP_ALL", match_option="CONTAINS",
    )
    env.safe_delete(pts_with_county)
    env.safe_delete(county_proj)

    # 4. 按县区计算阈值（Eq. 3）
    thresholds = compute_county_thresholds(building_with_ai, ai_field="AI_VALUE")
    use_global = "__global__" in thresholds

    # 确定县区字段
    county_field = None
    if not use_global:
        for f in _COUNTY_FIELD_CANDIDATES:
            if f in {fld.name for fld in arcpy.ListFields(building_with_ai)}:
                county_field = f
                break

    # 5. 写入 HVR 和阈值字段
    arcpy.AddField_management(building_with_ai, "HVR", "DOUBLE")
    arcpy.AddField_management(building_with_ai, "AI_THRESH", "DOUBLE")

    fields = ["AI_VALUE", "HVR", "AI_THRESH"]
    if county_field:
        fields.append(county_field)

    with arcpy.da.UpdateCursor(building_with_ai, fields) as cur:
        for row in cur:
            ai_val   = row[0] if (row[0] is not None) else 0.0
            county   = str(row[3]) if county_field else "__global__"
            thresh   = thresholds.get(county, thresholds.get("__global__", 0.0))

            if ai_val > 0 and thresh > 0:
                row[1] = max(0.0, min(1.0, 1.0 - ai_val / thresh))
            else:
                row[1] = 1.0   # 无有效 AI 时默认最高空置代理值

            row[2] = thresh
            cur.updateRow(row)

    # 6. 输出最终要素类
    final_output = os.path.join(workspace, province_name)
    arcpy.CopyFeatures_management(building_with_ai, final_output)
    env.safe_delete(building_with_ai)

    # 统计 HVR
    sample_limit = 100_000
    hvr_values, n = [], 0
    with arcpy.da.SearchCursor(final_output, ["HVR"]) as cur:
        for row in cur:
            if row[0] is not None and n < sample_limit:
                hvr_values.append(row[0])
                n += 1

    hvr_stats: dict = {}
    if hvr_values:
        arr = np.array(hvr_values)
        hvr_stats = {
            "count":  len(arr),
            "mean":   float(arr.mean()),
            "median": float(np.median(arr)),
            "std":    float(arr.std()),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
        }
        print(
            f"  [hvr] HVR 统计: mean={hvr_stats['mean']:.4f}, "
            f"median={hvr_stats['median']:.4f}, "
            f"range=[{hvr_stats['min']:.4f}, {hvr_stats['max']:.4f}]"
        )

    return final_output, hvr_stats