"""
province_processor.py
省级 HVR 批处理器

负责调度 activity_index / local_threshold / hvr_estimate 三个模块，
完成单省或多省的端到端处理，并生成 JSON 结果和文字报告。
"""

import os
import gc
import json
import time
import datetime
import traceback

import arcpy
import numpy as np

from utils.arcpy_env import ArcpyEnvManager
from .activity_index import prepare_light, prepare_poi, prepare_road, compute_activity_index
from .hvr_estimate import merge_buildings, filter_residential, compute_hvr


class ProvinceHVRProcessor:
    """省级住房空置率（HVR）批处理器。"""

    def __init__(self, config: dict):
        """
        Args:
            config: 路径配置字典，键见 configs/base_config.yaml
        """
        self.config = config
        self.env = ArcpyEnvManager(temp_base=config.get("temp_base", "D:\\arcpy_temp"))
        self.results: dict = {}

    # ------------------------------------------------------------------
    # 单省处理
    # ------------------------------------------------------------------

    def process_province(self, province_name: str) -> dict | None:
        """处理单个省份，返回结果字典；失败返回 None。"""
        print(f"\n{'='*60}\n开始处理: {province_name}\n{'='*60}")
        t0 = time.time()

        output_folder = workspace = None
        try:
            self.env.release_locks()

            # 检查省级边界
            boundary = os.path.join(self.config["boundary_base"], f"{province_name}.shp")
            if not arcpy.Exists(boundary):
                raise FileNotFoundError(f"未找到边界文件: {boundary}")

            arcpy.env.extent = boundary
            arcpy.env.mask   = boundary

            output_folder, workspace = ArcpyEnvManager.create_province_workspace(
                province_name, self.config["output_base"]
            )
            arcpy.env.scratchWorkspace = workspace

            # 1. 合并 + 筛选住宅建筑
            merged   = merge_buildings(province_name, self.config["building_base"], workspace, self.env)
            if not merged:
                raise RuntimeError("无建筑数据，跳过")
            building = filter_residential(merged, workspace, self.env)

            # 2. 三类数据预处理
            light = prepare_light(self.config["light_tif"],  boundary, output_folder, self.env)
            poi   = prepare_poi(  self.config["poi_shp"],    boundary, workspace, output_folder, self.env)
            road  = prepare_road( self.config["road_shp"],   boundary, workspace, output_folder, self.env)

            # 3. 计算 AI（Eq. 2）
            ai_path, ai_stats = compute_activity_index(light, poi, road, output_folder, self.env)
            for r in (light, poi, road):
                self.env.safe_delete(r)

            # 4. 计算 HVR（Eq. 3 & 4）
            # county_boundary 与 province_boundary 相同（含县区编码字段）
            county_boundary = self.config.get("county_boundary", boundary)
            hvr_output, hvr_stats = compute_hvr(
                building, ai_path, boundary, county_boundary,
                workspace, province_name, self.env,
            )
            self.env.safe_delete(building)
            self.env.safe_delete(ai_path)

            # 5. 清理中间文件
            self._clean_workspace(workspace, keep=province_name)

            elapsed = time.time() - t0
            result = {
                "province":    province_name,
                "success":     True,
                "output_path": hvr_output,
                "ai_stats":    ai_stats,
                "hvr_stats":   hvr_stats,
                "elapsed_sec": elapsed,
                "elapsed_str": f"{int(elapsed // 60)}分{elapsed % 60:.1f}秒",
            }
            print(f"\n✓ {province_name} 处理完成  耗时: {result['elapsed_str']}")
            self.results[province_name] = result
            return result

        except Exception as exc:
            print(f"\n✗ {province_name} 处理失败: {exc}")
            traceback.print_exc()
            try:
                if workspace and output_folder:
                    self._clean_workspace(workspace, keep=province_name)
            except Exception:
                pass
            self.results[province_name] = {
                "province": province_name,
                "success":  False,
                "error":    str(exc),
                "elapsed_sec": time.time() - t0,
            }
            return None

        finally:
            self.env.release_locks()
            gc.collect()

    # ------------------------------------------------------------------
    # 批量处理
    # ------------------------------------------------------------------

    def process_all(self, province_list: list[str] | None = None):
        """批量处理省份列表；为 None 时自动扫描 building_base 目录。"""
        if province_list is None:
            base = self.config["building_base"]
            province_list = [
                d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))
            ]
        print(f"\n待处理省份（{len(province_list)}个）: {', '.join(province_list)}")

        ok, fail = 0, 0
        for i, prov in enumerate(province_list, 1):
            print(f"\n[进度 {i}/{len(province_list)}]")
            self.env.release_locks()
            result = self.process_province(prov)
            (ok if (result and result["success"]) else fail).__class__  # 仅触发表达式
            if result and result["success"]:
                ok += 1
            else:
                fail += 1
            self.env.release_locks()
            gc.collect()
            time.sleep(2)

        self._print_summary(ok, fail)
        self._save_json()
        self._save_report()
        self.env.cleanup_all_temp()

    # ------------------------------------------------------------------
    # 工作空间清理
    # ------------------------------------------------------------------

    def _clean_workspace(self, workspace: str, keep: str):
        """删除 GDB 内除最终输出以外的所有中间要素/栅格。"""
        self.env.release_locks()
        old_ws = arcpy.env.workspace
        arcpy.env.workspace = workspace

        for list_fn in (arcpy.ListFeatureClasses, arcpy.ListRasters, arcpy.ListTables):
            try:
                for item in (list_fn() or []):
                    if item != keep:
                        self.env.safe_delete(os.path.join(workspace, item))
            except Exception:
                pass

        arcpy.env.workspace = old_ws
        try:
            arcpy.Compact_management(workspace)
        except Exception:
            pass
        self.env.cleanup_tracked(keep=[os.path.join(workspace, keep)])

    # ------------------------------------------------------------------
    # 报告输出
    # ------------------------------------------------------------------

    def _print_summary(self, ok: int, fail: int):
        print(f"\n{'='*60}\n批处理完成  成功: {ok}  失败: {fail}\n{'='*60}")
        for prov, r in self.results.items():
            if r["success"]:
                mean_hvr = r.get("hvr_stats", {}).get("mean", float("nan"))
                print(f"  ✓ {prov}  HVR均值={mean_hvr:.4f}  耗时={r.get('elapsed_str','')}")
            else:
                print(f"  ✗ {prov}  {r.get('error','')}")

    def _save_json(self):
        path = os.path.join(self.config["output_base"], "processing_results.json")
        out = {
            prov: {k: v for k, v in r.items() if k not in ("output_path",)}
            for prov, r in self.results.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[报告] JSON 结果已保存: {path}")

    def _save_report(self):
        path = os.path.join(self.config["output_base"], "summary_report.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("住房空置率(HVR)批处理报告\n")
            f.write(f"生成时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write("=" * 60 + "\n\n")
            ok_provs = [p for p, r in self.results.items() if r.get("success")]
            means = []
            for p in ok_provs:
                s = self.results[p].get("hvr_stats", {})
                if s:
                    means.append(s["mean"])
                    f.write(
                        f"{p:10s}  mean={s['mean']:.4f}  median={s['median']:.4f}"
                        f"  range=[{s['min']:.4f},{s['max']:.4f}]\n"
                    )
            if means:
                f.write(f"\n全国平均 HVR: {np.mean(means):.4f}\n")
            fail_provs = [p for p, r in self.results.items() if not r.get("success")]
            if fail_provs:
                f.write("\n失败省份:\n")
                for p in fail_provs:
                    f.write(f"  {p}: {self.results[p].get('error','')}\n")
        print(f"[报告] 文字报告已保存: {path}")