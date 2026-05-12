"""
run_03_carbon.py
4.4 碳排放与情景预测入口

子任务：
  demand    → 预测 2024–2050 年住房需求
  supply    → 预测 2024–2050 年住房供应（supply + old_supply）
  new_area  → 预测 2024–2050 年新增建设面积
  new_supply→ 计算 new_supply = max(0, supply - old_supply)（ArcPy 栅格运算）

用法：
  python scripts/run_03_carbon.py --task demand --scenario ssp2
  python scripts/run_03_carbon.py --task supply --scenario ssp5
  python scripts/run_03_carbon.py --task new_area --scenario ssp1
  python scripts/run_03_carbon.py --task new_supply
  python scripts/run_03_carbon.py --task demand --scenario ssp2 --replay
"""

import argparse
import importlib.util
import os
import sys
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load(name, rel):
    path = os.path.join(ROOT, "src", rel)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


xgb_mod = _load("xgboost_forecast", os.path.join("03_carbon", "xgboost_forecast.py"))
ns_mod  = _load("new_supply",       os.path.join("03_carbon", "new_supply.py"))
ce_mod  = _load("carbon_emission",  os.path.join("03_carbon", "carbon_emission.py"))

XGBoostForecaster  = xgb_mod.XGBoostForecaster
compute_new_supply = ns_mod.compute_new_supply
calculate_carbon   = ce_mod.calculate_carbon


def load_config(path=None):
    if path is None:
        path = os.path.join(ROOT, "configs", "ssp_config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# 各任务的目标变量和排除列
TASK_CONFIG = {
    "demand": {
        "target_cols":  ["demand"],
        "exclude_cols": ["flag", "demand"],
        "model_key":    "demand_model_dir",
        "result_key":   "demand_result_dir",
    },
    "supply": {
        "target_cols":  ["supply", "old_supply"],
        "exclude_cols": ["flag", "supply", "old_supply", "new_supply"],
        "model_key":    "supply_model_dir",
        "result_key":   "supply_result_dir",
    },
    "new_area": {
        "target_cols":  ["new_area"],
        "exclude_cols": ["flag", "new_area"],
        "model_key":    "new_area_model_dir",
        "result_key":   "new_area_result_dir",
    },
}


def main():
    parser = argparse.ArgumentParser(description="碳排放情景预测")
    parser.add_argument("--task",
                        choices=["demand", "supply", "new_area", "new_supply", "carbon"],
                        required=True)
    parser.add_argument("--scenario",
                        choices=["ssp1", "ssp2", "ssp3", "ssp4", "ssp5"],
                        default="ssp2")
    parser.add_argument("--replay", action="store_true",
                        help="仅加载 pkl 复现预测，不重新训练")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── new_supply（ArcPy 栅格运算）─────────────────────────────────────────
    if args.task == "new_supply":
        years = list(range(cfg["predict_years"][0], cfg["predict_years"][1] + 1))
        compute_new_supply(
            supply_dir = cfg["supply_result_dir"],
            years      = years,
            output_dir = cfg["new_supply_dir"],
        )
        return

    # ── carbon（碳排放计算，Eq.16 5年滚动窗口）──────────────────────────────
    if args.task == "carbon":
        ssp        = args.scenario
        input_csv  = os.path.join(
            cfg["new_area_result_dir"], ssp, f"predicted_new_area_{ssp}.csv"
        )
        output_csv = os.path.join(
            cfg.get("carbon_output_dir", "output/carbon"), ssp, f"carbon_{ssp}.csv"
        )
        calculate_carbon(
            new_area_csv   = input_csv,
            scenario       = ssp,
            output_csv     = output_csv,
            rolling_window = cfg.get("carbon_rolling_window", 5),
        )
        return

    # ── XGBoost 预测任务 ─────────────────────────────────────────────────────
    tc      = TASK_CONFIG[args.task]
    sc_cfg  = cfg["scenarios"][args.scenario]
    csv_dir = sc_cfg["csv_dir"]

    train_csv   = os.path.join(csv_dir, "data_1.csv")
    predict_csv = os.path.join(csv_dir, "data_2.csv")

    result_dir  = os.path.join(cfg[tc["result_key"]], args.scenario)
    model_dir   = os.path.join(cfg[tc["model_key"]],  args.scenario)
    output_csv  = os.path.join(result_dir, f"predicted_{args.task}_{args.scenario}.csv")
    report_path = os.path.join(result_dir, f"r2_report_{args.task}_{args.scenario}.txt")

    xgb_params = {k: cfg[k] for k in
                  ("n_folds", "use_hyperopt", "n_iter", "use_log_x", "log_eps",
                   "n_jobs", "random_state")}

    forecaster = XGBoostForecaster(
        target_cols  = tc["target_cols"],
        exclude_cols = tc["exclude_cols"],
        model_dir    = model_dir,
        **xgb_params,
    )

    print(f"\n情景: {sc_cfg['label']}  任务: {args.task}"
          + ("  [replay]" if args.replay else ""))

    if args.replay:
        forecaster.replay(predict_csv, output_csv)
    else:
        forecaster.run(train_csv, predict_csv, output_csv, report_path)


if __name__ == "__main__":
    main()