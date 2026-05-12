"""
run_05_optimization.py
4.6 最优策略模拟入口

论文 4.6 节设置五类治理策略：
  infrastructure  基础设施（RD / 距离类变量）
  social          社会保障（CleanArea / 保险类变量）
  economic        经济发展（IndFirm / Gini / 三产变量）
  environment     自然环境（WTD / HDS / LAI）
  cooperative     协同治理（全部 18 个变量）

用法：
  python scripts/run_05_optimization.py --strategy cooperative --scenario ssp2
  python scripts/run_05_optimization.py --strategy infrastructure --scenario ssp1
  python scripts/run_05_optimization.py --strategy all --scenario ssp2  # 运行全部策略
  python scripts/run_05_optimization.py --strategy cooperative --scenario ssp2 --ramp_start 2026
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


opt_mod = _load("nsga2_optimize", os.path.join("05_optimization", "nsga2_optimize.py"))
NSGA2Optimizer = opt_mod.NSGA2Optimizer


def load_config(path=None):
    if path is None:
        path = os.path.join(ROOT, "configs", "ssp_config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 完整变量配置（18 个可调控变量，来自论文 4.5 节筛选结果）──────────────────
ALL_VAR_CONFIG = {
    # 基础设施
    "RdDens":     {"range": 0.25, "lb": 0,    "ub": None},
    "SchDist":    {"range": 0.45, "lb": 0,    "ub": None},
    "HospDist":   {"range": 0.40, "lb": 0,    "ub": None},
    "BusDist":    {"range": 0.35, "lb": 0,    "ub": None},
    "RailDist":   {"range": 0.30, "lb": 0,    "ub": None},
    "MallDist":   {"range": 0.35, "lb": 0,    "ub": None},
    # 社会保障
    "CleanArea":  {"range": 0.20, "lb": 0,    "ub": None},
    "PenIns":     {"range": 0.20, "lb": 0,    "ub": None},
    "UnempIns":   {"range": 0.15, "lb": 0,    "ub": None},
    "MedIns":     {"range": 0.65, "lb": 0,    "ub": None},
    # 经济发展
    "IndFirm":    {"range": 1.10, "lb": 0,    "ub": None},
    "Gini":       {"range": 0.05, "lb": 0,    "ub": 1   },
    "PriInd":     {"range": 0.35, "lb": 0,    "ub": None},
    "SecInd":     {"range": 0.45, "lb": 0,    "ub": None},
    "TerInd":     {"range": 0.50, "lb": 0,    "ub": None},
    # 自然环境
    "WTD":        {"range": 0.10, "lb": None, "ub": None},
    "HDS":        {"range": 0.15, "lb": None, "ub": None},
    "LAI":        {"range": 0.15, "lb": 0,    "ub": None},
}

# ── 四类独立策略的变量子集 ───────────────────────────────────────────────────
STRATEGY_VARS = {
    "infrastructure": {k: v for k, v in ALL_VAR_CONFIG.items()
                       if k in ("RdDens","SchDist","HospDist","BusDist","RailDist","MallDist")},
    "social":         {k: v for k, v in ALL_VAR_CONFIG.items()
                       if k in ("CleanArea","PenIns","UnempIns","MedIns")},
    "economic":       {k: v for k, v in ALL_VAR_CONFIG.items()
                       if k in ("IndFirm","Gini","PriInd","SecInd","TerInd")},
    "environment":    {k: v for k, v in ALL_VAR_CONFIG.items()
                       if k in ("WTD","HDS","LAI")},
    "cooperative":    ALL_VAR_CONFIG,   # 全部 18 个变量
}

# 独立策略种群 500，协同策略种群 1000（论文 4.6）
STRATEGY_POP = {
    "infrastructure": 500,
    "social":         500,
    "economic":       500,
    "environment":    500,
    "cooperative":    1000,
}


def run_strategy(strategy: str, scenario: str, cfg: dict, args):
    """运行单个策略。"""
    if strategy not in STRATEGY_VARS:
        raise ValueError(f"未知策略: {strategy}，可选: {list(STRATEGY_VARS.keys())}")

    sc_cfg      = cfg["scenarios"][scenario]
    predict_csv = os.path.join(sc_cfg["csv_dir"], "data_2.csv")
    model_dir   = cfg.get("supply_model_dir", "output/models/supply")
    output_base = cfg.get("optimization_output_dir", "output/optimization")
    city_csv    = cfg.get("city_name_csv", "data/raw/city_name.csv")

    optimizer = NSGA2Optimizer(
        scenario       = scenario,
        strategy_name  = strategy,
        predict_csv    = predict_csv,
        city_csv       = city_csv,
        model_dir      = model_dir,
        output_base    = output_base,
        var_config     = STRATEGY_VARS[strategy],
        pop_size       = STRATEGY_POP[strategy],
        n_generations  = args.n_generations,
        ramp_start     = args.ramp_start,
        ramp_end       = 2050,
        opt_start      = args.ramp_start,
        opt_end        = 2050,
        random_state   = cfg.get("random_state", 42),
    )
    optimizer.run()


def main():
    parser = argparse.ArgumentParser(description="NSGA-II 多目标住房优化")
    parser.add_argument("--strategy",
                        choices=list(STRATEGY_VARS.keys()) + ["all"],
                        default="cooperative")
    parser.add_argument("--scenario",
                        choices=["ssp1","ssp2","ssp3","ssp4","ssp5"],
                        default="ssp2")
    parser.add_argument("--ramp_start",   type=int, default=2031,
                        help="政策干预启动年份（Eq.18）")
    parser.add_argument("--n_generations",type=int, default=100)
    parser.add_argument("--config",       default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    strategies = list(STRATEGY_VARS.keys()) if args.strategy == "all" else [args.strategy]
    for strat in strategies:
        print(f"\n{'='*60}\n策略: {strat}  情景: {args.scenario}\n{'='*60}")
        run_strategy(strat, args.scenario, cfg, args)


if __name__ == "__main__":
    main()