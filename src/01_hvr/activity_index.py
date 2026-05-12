"""
activity_index.py
综合人类活动指数（Active Index, AI）计算

依赖：ArcPy + Spatial Analyst
"""

import os
import shutil

import arcpy
from arcpy.sa import (
    ExtractByMask, KernelDensity, LineDensity, Ln, Power, Raster
)

from utils.arcpy_env import ArcpyEnvManager


# --------------------------------------------------------------------------
# 各数据层预处理
# --------------------------------------------------------------------------

def prepare_light(
    light_tif: str,
    province_boundary: str,
    output_folder: str,
    env: ArcpyEnvManager,
) -> str:
    """
    裁剪、重投影、重采样夜间灯光栅格至 10 m 分辨率。

    Args:
        light_tif:         全国夜间灯光 GeoTIFF 路径
        province_boundary: 省级边界要素路径
        output_folder:     输出文件夹
        env:               ArcPy 环境管理器

    Returns:
        处理后的灯光栅格路径
    """
    print("  [light] 处理夜间灯光数据...")
    temp_dir = os.path.join(env.temp_base, "light_temp")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        clipped_path = os.path.join(temp_dir, "light_clipped.tif")
        clipped = ExtractByMask(light_tif, province_boundary)
        clipped.save(clipped_path)
        del clipped

        proj_path = os.path.join(temp_dir, "light_projected.tif")
        arcpy.ProjectRaster_management(
            clipped_path, proj_path, env.CRS_UTM49N, "BILINEAR"
        )

        light_final = os.path.join(output_folder, "light_final.tif")
        arcpy.Resample_management(proj_path, light_final, "10", "BILINEAR")
        env.track(light_final)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"  [light] 完成 → {light_final}")
    return light_final


def prepare_poi(
    poi_shp: str,
    province_boundary: str,
    workspace: str,
    output_folder: str,
    env: ArcpyEnvManager,
    search_radius: float = 650,
    cell_size: float = 10,
) -> str:
    """
    裁剪 POI 矢量、重投影，执行核密度分析（bandwidth=650 m, cell=10 m）。

    Args:
        poi_shp:        全国 POI shapefile 路径
        search_radius:  核密度搜索半径（米），默认 650
        cell_size:      输出栅格像元大小（米），默认 10

    Returns:
        POI 核密度栅格路径
    """
    print("  [poi] 处理 POI 数据...")
    temp_poi = env.track(os.path.join(workspace, "temp_poi"))
    arcpy.Clip_analysis(poi_shp, province_boundary, temp_poi)

    poi_proj = env.track(os.path.join(workspace, "poi_projected"))
    arcpy.Project_management(temp_poi, poi_proj, env.CRS_UTM49N)
    env.safe_delete(temp_poi)

    print("  [poi] 执行核密度分析...")
    temp_dir = os.path.join(env.temp_base, "poi_density")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        density_temp = os.path.join(temp_dir, "density.tif")
        density = KernelDensity(poi_proj, "NONE", cell_size, search_radius, "SQUARE_MAP_UNITS")
        density.save(density_temp)
        del density

        masked = ExtractByMask(density_temp, province_boundary)
        poi_final = os.path.join(output_folder, "poi_density_final.tif")
        masked.save(poi_final)
        del masked
        env.track(poi_final)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    env.safe_delete(poi_proj)
    print(f"  [poi] 完成 → {poi_final}")
    return poi_final


def prepare_road(
    road_shp: str,
    province_boundary: str,
    workspace: str,
    output_folder: str,
    env: ArcpyEnvManager,
    search_radius: float = 100,
    cell_size: float = 10,
) -> str:
    """
    裁剪道路矢量、重投影，执行线密度分析（radius=100 m, cell=10 m）。

    Args:
        road_shp:       全国道路 shapefile 路径
        search_radius:  线密度搜索半径（米），默认 100
        cell_size:      输出栅格像元大小（米），默认 10

    Returns:
        道路线密度栅格路径
    """
    print("  [road] 处理道路数据...")
    temp_road = env.track(os.path.join(workspace, "temp_road"))
    arcpy.Clip_analysis(road_shp, province_boundary, temp_road)

    road_proj = env.track(os.path.join(workspace, "road_projected"))
    arcpy.Project_management(temp_road, road_proj, env.CRS_UTM49N)
    env.safe_delete(temp_road)

    road_density = LineDensity(road_proj, "NONE", cell_size, search_radius, "SQUARE_METERS")
    road_final = os.path.join(output_folder, "road_density_final.tif")
    road_density.save(road_final)
    del road_density
    env.track(road_final)

    env.safe_delete(road_proj)
    print(f"  [road] 完成 → {road_final}")
    return road_final


# --------------------------------------------------------------------------
# 综合人类活动指数 AI（Eq. 2）
# --------------------------------------------------------------------------

def _normalize(raster: Raster, lo: float = 0.1, hi: float = 1.0) -> Raster:
    """
    对数变换后线性归一化至 [lo, hi]。

    论文要求归一化结果限制在 0.1–1.0，避免零值完全抵消几何平均。
    """
    log_r = Ln(raster + 1)
    min_val = float(arcpy.GetRasterProperties_management(log_r, "MINIMUM").getOutput(0))
    max_val = float(arcpy.GetRasterProperties_management(log_r, "MAXIMUM").getOutput(0))
    if max_val > min_val:
        return lo + (hi - lo) * (log_r - min_val) / (max_val - min_val)
    return lo + (hi - lo) / 2  # 常数栅格时返回中间值


def compute_activity_index(
    light_path: str,
    poi_path: str,
    road_path: str,
    output_folder: str,
    env: ArcpyEnvManager,
) -> tuple[str, dict]:
    """
    计算综合人类活动指数 AI（Eq. 2）。

    AI = (light_norm × poi_norm × road_norm)^(1/3)

    几何平均保证三类信号同时较强时 AI 才显著升高，
    单一高值不会主导结果（论文 4.1）。

    Returns:
        (ai_raster_path, stats_dict)
    """
    print("  [ai] 计算综合人类活动指数 AI...")

    light_norm = _normalize(Raster(light_path))
    poi_norm   = _normalize(Raster(poi_path))
    road_norm  = _normalize(Raster(road_path))

    ai = Power(light_norm * poi_norm * road_norm, 1.0 / 3.0)

    ai_path = os.path.join(output_folder, "activity_index.tif")
    ai.save(ai_path)
    env.track(ai_path)

    del light_norm, poi_norm, road_norm, ai

    stats = {
        "min":  float(arcpy.GetRasterProperties_management(ai_path, "MINIMUM").getOutput(0)),
        "max":  float(arcpy.GetRasterProperties_management(ai_path, "MAXIMUM").getOutput(0)),
        "mean": float(arcpy.GetRasterProperties_management(ai_path, "MEAN").getOutput(0)),
        "std":  float(arcpy.GetRasterProperties_management(ai_path, "STD").getOutput(0)),
    }
    print(
        f"  [ai] AI 统计: min={stats['min']:.4f}, max={stats['max']:.4f}, "
        f"mean={stats['mean']:.4f}, std={stats['std']:.4f}"
    )
    return ai_path, stats