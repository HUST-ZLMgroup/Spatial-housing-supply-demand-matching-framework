"""
run_01_hvr.py
4.1 建筑尺度住房空置率（HVR）计算入口

用法:
    python scripts/run_01_hvr.py                          # 处理所有省份、所有年份
    python scripts/run_01_hvr.py --provinces 湖北 湖南    # 指定省份
    python scripts/run_01_hvr.py --years 2020 2021        # 指定年份
    python scripts/run_01_hvr.py --provinces 广东 --years 2023
"""

import argparse
import sys
import os
import yaml

# 把项目根目录和 src 都加入路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# 01_hvr 以数字开头不能直接 import，用 importlib 加载
import importlib.util

def _load_module(name, rel_path):
    path = os.path.join(ROOT, "src", rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

province_processor = _load_module(
    "province_processor",
    os.path.join("01_hvr", "province_processor.py"),
)
ProvinceHVRProcessor = province_processor.ProvinceHVRProcessor


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(ROOT, "configs", "base_config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="建筑尺度 HVR 批处理")
    parser.add_argument("--provinces", nargs="*", default=None,
                        help="省份名称列表；不填则处理所有省份")
    parser.add_argument("--years", nargs="*", type=int, default=None,
                        help="年份列表；不填则处理配置文件中 year_start~year_end")
    parser.add_argument("--config", default=None, help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)

    # 确定处理年份
    if args.years:
        years = args.years
    else:
        years = list(range(config["year_start"], config["year_end"] + 1))

    processor = ProvinceHVRProcessor(config)
    processor.process_all(province_list=args.provinces, years=years)


if __name__ == "__main__":
    main()