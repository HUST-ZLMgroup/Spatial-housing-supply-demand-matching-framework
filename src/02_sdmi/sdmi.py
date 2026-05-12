"""
sdmi.py
住房供需空间失衡指数（SDMI）计算

对应论文 4.3 节 Eq.(8)：
    SDMI_g = (Supply_g - Demand_g) / (0.5 × (Supply_max + Demand_max))

其中：
    Supply_g = old_supply_g + new_supply_g   （存量闲置供应 + 新增供应）
    Supply_max, Demand_max 为当年全国 10km 网格最大值

SDMI > 0：供大于求（住房过剩区域）
SDMI < 0：供不应求（住房短缺区域）
SDMI = 0：供需平衡

输入：各年 demand{year}.tif / old_supply{year}.tif / new_supply{year}.tif
输出：supply_demand_ratio{year}.tif
"""

import os
from datetime import datetime

import arcpy
from arcpy.sa import Raster


class SDMICalculator:
    """住房供需空间失衡指数（SDMI）计算器。"""

    def __init__(
        self,
        demand_dir: str,
        old_supply_dir: str,
        new_supply_dir: str,
        output_dir: str,
        years: list[int] = None,
    ):
        self.demand_dir     = demand_dir
        self.old_supply_dir = old_supply_dir
        self.new_supply_dir = new_supply_dir
        self.output_dir     = output_dir
        self.years          = years or list(range(2013, 2024))

    # ------------------------------------------------------------------
    # 单年计算
    # ------------------------------------------------------------------

    def calculate_year(self, year: int) -> str | None:
        """
        计算单年 SDMI。

        SDMI = (Supply - Demand) / (0.5 × (Supply_max + Demand_max))

        Returns:
            输出栅格路径；失败返回 None
        """
        self._log(f"\n计算 SDMI - {year}")

        demand_path     = os.path.join(self.demand_dir,     f"demand{year}.tif")
        old_supply_path = os.path.join(self.old_supply_dir, f"old_supply{year}.tif")
        new_supply_path = os.path.join(self.new_supply_dir, f"new_supply{year}.tif")

        missing = [p for p in (demand_path, old_supply_path, new_supply_path)
                   if not os.path.exists(p)]
        if missing:
            for p in missing:
                self._log(f"  缺少文件: {p}")
            return None

        try:
            demand     = Raster(demand_path)
            old_supply = Raster(old_supply_path)
            new_supply = Raster(new_supply_path)

            supply = old_supply + new_supply

            supply_max = float(
                arcpy.GetRasterProperties_management(supply, "MAXIMUM").getOutput(0)
            )
            demand_max = float(
                arcpy.GetRasterProperties_management(demand, "MAXIMUM").getOutput(0)
            )
            self._log(f"  Supply_max={supply_max:.2f}  Demand_max={demand_max:.2f}")

            denominator = 0.5 * (supply_max + demand_max)
            if denominator == 0:
                self._log("  警告：分母为 0，跳过")
                return None

            sdmi = (supply - demand) / denominator

            # 统计
            s_min  = float(arcpy.GetRasterProperties_management(sdmi, "MINIMUM").getOutput(0))
            s_max  = float(arcpy.GetRasterProperties_management(sdmi, "MAXIMUM").getOutput(0))
            s_mean = float(arcpy.GetRasterProperties_management(sdmi, "MEAN").getOutput(0))
            self._log(f"  SDMI: min={s_min:.4f}  max={s_max:.4f}  mean={s_mean:.4f}")

            output_path = os.path.join(self.output_dir, f"sdmi{year}.tif")
            sdmi.save(output_path)
            self._log(f"  已保存: {output_path}")
            return output_path

        except Exception as exc:
            self._log(f"  计算失败: {exc}")
            import traceback
            self._log(traceback.format_exc())
            return None

    # ------------------------------------------------------------------
    # 批量运行
    # ------------------------------------------------------------------

    def run(self) -> dict[int, str]:
        """批量计算所有年份 SDMI。"""
        self._setup()
        t0 = datetime.now()

        self._log("\n" + "=" * 60)
        self._log("SDMI 计算  公式: (Supply - Demand) / (0.5×(S_max+D_max))")
        self._log(f"年份: {self.years}")
        self._log("=" * 60)

        results: dict[int, str] = {}
        for year in self.years:
            path = self.calculate_year(year)
            if path:
                results[year] = path

        arcpy.CheckInExtension("Spatial")
        elapsed = datetime.now() - t0
        self._log(f"\n完成：{len(results)}/{len(self.years)} 年  耗时: {elapsed}")
        return results

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _setup(self):
        if arcpy.CheckExtension("Spatial") != "Available":
            raise RuntimeError("Spatial Analyst 扩展不可用")
        arcpy.CheckOutExtension("Spatial")
        arcpy.env.overwriteOutput = True
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")