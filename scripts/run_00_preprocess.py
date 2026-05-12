"""
run_00_preprocess.py
建筑数据预处理入口（在 run_01_hvr.py 之前运行）

用法:
    python scripts/run_00_preprocess.py                        # 处理所有省份
    python scripts/run_00_preprocess.py --provinces 湖北 湖南  # 指定省份
"""

import argparse
import os
import sys
import importlib.util
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def _load_module(name, rel_path):
    path = os.path.join(ROOT, "src", rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bp = _load_module("building_preprocess",
                  os.path.join("00_preprocess", "building_preprocess.py"))


def load_config(path=None):
    if path is None:
        path = os.path.join(ROOT, "configs", "base_config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="建筑数据预处理")
    parser.add_argument("--provinces", nargs="*", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    bp.preprocess_all_provinces(
        building_raw_dir=config["building_raw_dir"],
        output_dir=config["building_base"],   # 预处理输出 = HVR输入
        province_list=args.provinces,
        sigma=config.get("outlier_sigma", 3.0),
    )


if __name__ == "__main__":
    main()