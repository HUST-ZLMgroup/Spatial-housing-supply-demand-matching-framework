"""
run_04_causal.py
4.5 驱动机制识别与敏感性分析入口

子任务：
  lingam  → DirectLiNGAM 因果分析（所有 SSP × 年份节点）
  ovat    → 单变量 OAT 敏感性分析（Eq.15）
  sobol   → Sobol 全局敏感性分析（Eqs.16-17）

用法：
  python scripts/run_04_causal.py --task lingam
  python scripts/run_04_causal.py --task ovat
  python scripts/run_04_causal.py --task sobol --ssp ssp2
  python scripts/run_04_causal.py --task sobol          # 所有 SSP
"""

import argparse
import importlib.util
import os
import sys
import yaml
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "04_causal"))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _load(name, rel):
    path = os.path.join(ROOT, "src", rel)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lingam_mod = _load("lingam_analysis",    os.path.join("04_causal", "lingam_analysis.py"))
ovat_mod   = _load("sensitivity_ovat",   os.path.join("04_causal", "sensitivity_ovat.py"))
sobol_mod  = _load("sensitivity_sobol",  os.path.join("04_causal", "sensitivity_sobol.py"))


def load_config(path=None):
    if path is None:
        path = os.path.join(ROOT, "configs", "ssp_config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="因果与敏感性分析")
    parser.add_argument("--task",   choices=["lingam", "ovat", "sobol"], required=True)
    parser.add_argument("--ssp",    default=None, help="指定情景（sobol任务可用）")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg      = load_config(args.config)
    ssps     = ["ssp1", "ssp2", "ssp3", "ssp4", "ssp5"]
    ssp_data = {
        ssp: os.path.join(cfg["scenarios"][ssp]["csv_dir"], "data_2.csv")
        for ssp in ssps
        if ssp in cfg.get("scenarios", {})
    }
    model_dir  = cfg.get("supply_model_dir", "output/models/supply")
    causal_dir = cfg.get("causal_dir",       "output/causal")
    sens_dir   = cfg.get("sensitivity_dir",  "output/sensitivity")

    if args.task == "lingam":
        base_dir = cfg.get("causal_base_dir", "data/processed/causal")
        lingam_mod.run_all(
            base_dir          = base_dir,
            bootstrap_n       = 100,
            bootstrap_workers = -1,
        )

    elif args.task == "ovat":
        ovat_mod.run_sensitivity_ovat(
            ssp_data   = ssp_data,
            model_dir  = model_dir,
            output_dir = os.path.join(sens_dir, "ovat"),
            n_jobs     = cfg.get("n_jobs", 5),
        )

    elif args.task == "sobol":
        if args.ssp:
            sobol_mod.run_sobol(
                ssp         = args.ssp,
                data_path   = ssp_data[args.ssp],
                model_dir   = model_dir,
                output_path = os.path.join(sens_dir, "sobol", f"sobol_{args.ssp}.xlsx"),
                sobol_n     = 1000,
            )
        else:
            sobol_mod.run_sobol_all_ssps(
                ssp_data   = ssp_data,
                model_dir  = model_dir,
                output_dir = os.path.join(sens_dir, "sobol"),
                sobol_n    = 1000,
            )


if __name__ == "__main__":
    mp.freeze_support()   # Windows 多进程必须
    main()