"""
run_02_sdmi.py
4.3 住房供需失衡计算入口

运行顺序：
  1. 刚性需求   (demand_rigid)
  2. 改进需求   (demand_improvement_renewal)
  3. 更新需求   (demand_improvement_renewal)
  4. 需求合并   (demand_merge)
  5. 潜在供应   (housing_supply)
  6. SDMI 计算  (sdmi)

用法:
    python scripts/run_02_sdmi.py                    # 全部步骤，全部年份
    python scripts/run_02_sdmi.py --steps 1 2 3      # 仅跑前三步
    python scripts/run_02_sdmi.py --years 2020 2021  # 指定年份
"""

import argparse
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml


def _load(name, rel):
    path = os.path.join(ROOT, "src", rel)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dr   = _load("demand_rigid",               os.path.join("02_sdmi", "demand_rigid.py"))
dir_ = _load("demand_improvement_renewal", os.path.join("02_sdmi", "demand_improvement_renewal.py"))
dm   = _load("demand_merge",               os.path.join("02_sdmi", "demand_merge.py"))
hs   = _load("housing_supply",             os.path.join("02_sdmi", "housing_supply.py"))
sd   = _load("sdmi",                       os.path.join("02_sdmi", "sdmi.py"))


def load_config(path=None):
    if path is None:
        path = os.path.join(ROOT, "configs", "base_config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="住房供需失衡计算")
    parser.add_argument("--steps", nargs="*", type=int, default=[1,2,3,4,5,6],
                        help="执行步骤编号列表（1-6），默认全部")
    parser.add_argument("--years", nargs="*", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg   = load_config(args.config)
    years = args.years or list(range(cfg["year_start"], cfg["year_end"] + 1))
    steps = set(args.steps)

    if 1 in steps:
        dr.run(
            pop_dir    = cfg["pop_dir"],
            output_dir = cfg["demand_rigid_dir"],
            years      = years,
        )

    if 2 in steps:
        dir_.run_improvement(
            demand_dir     = cfg["demand_base_dir"],
            output_dir     = cfg["demand_improvement_dir"],
            years          = years,
            per_area_growth= cfg.get("per_area_growth", 0.75),
        )

    if 3 in steps:
        dir_.run_renewal(
            demand_dir      = cfg["demand_base_dir"],
            output_dir      = cfg["demand_renewal_dir"],
            years           = years,
            demolition_rate = cfg.get("demolition_rate", 0.0059),
        )

    if 4 in steps:
        dm.merge_demand(
            rigid_dir       = cfg["demand_rigid_dir"],
            improvement_dir = cfg["demand_improvement_dir"],
            renewal_dir     = cfg["demand_renewal_dir"],
            output_dir      = cfg["demand_total_dir"],
            years           = years,
        )

    if 5 in steps:
        hs.run(
            hvr_root       = cfg["hvr_output_base"],
            correction_dir = cfg["correction_dir"],
            years          = years,
        )

    if 6 in steps:
        calculator = sd.SDMICalculator(
            demand_dir     = cfg["demand_total_dir"],
            old_supply_dir = cfg["old_supply_dir"],
            new_supply_dir = cfg["new_supply_dir"],
            output_dir     = cfg["sdmi_output_dir"],
            years          = years,
        )
        calculator.run()


if __name__ == "__main__":
    main()