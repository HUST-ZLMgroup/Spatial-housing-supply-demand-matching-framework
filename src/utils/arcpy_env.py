"""
arcpy_env.py
ArcPy 环境初始化与临时文件管理工具
"""

import os
import gc
import time
import shutil

import arcpy


class ArcpyEnvManager:
    """管理 ArcPy 运行环境、临时目录和文件锁释放。"""

    # 坐标系：WGS 84 / UTM Zone 49N（统一空间参考，见论文 4.2）
    CRS_UTM49N = arcpy.SpatialReference(32649)

    def __init__(self, temp_base: str):
        """
        Args:
            temp_base: 临时文件根目录（建议使用空间充裕的磁盘）
        """
        self.temp_base = temp_base
        self._temp_files: list[str] = []
        self._setup()

    def _setup(self):
        """初始化临时目录与 ArcPy 全局环境。"""
        os.makedirs(self.temp_base, exist_ok=True)

        # 强制系统临时目录指向指定路径，避免系统盘空间不足
        for var in ("TEMP", "TMP", "TMPDIR"):
            os.environ[var] = self.temp_base

        arcpy.env.overwriteOutput = True
        arcpy.env.parallelProcessingFactor = "100%"
        arcpy.CheckOutExtension("Spatial")

        scratch_dir = os.path.join(self.temp_base, "scratch")
        os.makedirs(scratch_dir, exist_ok=True)
        arcpy.env.scratchWorkspace = scratch_dir
        arcpy.env.workspace = scratch_dir

        # 创建临时 GDB
        temp_gdb = os.path.join(scratch_dir, "temp_scratch.gdb")
        if not arcpy.Exists(temp_gdb):
            arcpy.CreateFileGDB_management(scratch_dir, "temp_scratch.gdb")

        print(f"[env] 临时工作空间: {scratch_dir}")

    # ------------------------------------------------------------------
    # 临时文件追踪
    # ------------------------------------------------------------------

    def track(self, path: str) -> str:
        """登记临时文件路径，返回原路径方便链式调用。"""
        if path and path not in self._temp_files:
            self._temp_files.append(path)
        return path

    def release_locks(self):
        """释放 ArcPy 文件锁并回收内存。"""
        arcpy.env.workspace = None
        arcpy.env.scratchWorkspace = None
        arcpy.env.extent = None
        arcpy.env.mask = None
        try:
            arcpy.ClearWorkspaceCache_management()
        except Exception:
            pass
        arcpy.Delete_management("in_memory")
        gc.collect()

    def safe_delete(self, path: str, max_attempts: int = 3) -> bool:
        """尝试删除 ArcGIS 数据集，失败时重试。"""
        for attempt in range(max_attempts):
            try:
                if arcpy.Exists(path):
                    if path.endswith(".gdb"):
                        try:
                            arcpy.Compact_management(path)
                        except Exception:
                            pass
                    arcpy.Delete_management(path)
                return True
            except Exception:
                if attempt < max_attempts - 1:
                    time.sleep(1)
                    self.release_locks()
                    gc.collect()
        return False

    def cleanup_tracked(self, keep: list[str] | None = None):
        """删除所有已追踪的临时文件（keep 列表中的除外）。"""
        keep = keep or []
        for path in self._temp_files:
            if path not in keep:
                self.safe_delete(path)
        self._temp_files = []

    def cleanup_all_temp(self):
        """清除整个临时目录（程序结束时调用）。"""
        if os.path.exists(self.temp_base):
            try:
                shutil.rmtree(self.temp_base, ignore_errors=True)
                print(f"[env] 已清除临时目录: {self.temp_base}")
            except Exception as e:
                print(f"[env] 清除临时目录失败: {e}")

    # ------------------------------------------------------------------
    # 工作空间管理
    # ------------------------------------------------------------------

    @staticmethod
    def create_province_workspace(province_name: str, output_base: str) -> tuple[str, str]:
        """
        为省份创建输出目录和 GDB 工作空间。

        Returns:
            (output_folder, gdb_path)
        """
        output_folder = os.path.join(output_base, province_name)
        os.makedirs(output_folder, exist_ok=True)

        gdb_name = f"{province_name}.gdb"
        gdb_path = os.path.join(output_folder, gdb_name)
        if arcpy.Exists(gdb_path):
            arcpy.Delete_management(gdb_path)
        arcpy.CreateFileGDB_management(output_folder, gdb_name)

        print(f"[env] 创建工作空间: {gdb_path}")
        return output_folder, gdb_path