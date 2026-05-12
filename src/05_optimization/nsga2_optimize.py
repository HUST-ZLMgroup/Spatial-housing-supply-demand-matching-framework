"""
nsga2_optimize.py
NSGA-II 多目标优化

对应论文 4.6 节：
  - 以 SSP 情景下 2025–2050 年社会经济预测为基线
  - 18 个可调控社会变量作为政策干预对象（线性斜坡演化）
  - 四个优化目标：
      f1: 最小化新增建设需求（new_supply，城市尺度）
      f2: 最大化存量住房利用（old_supply，取负转最小化）
      f3: 最小化建筑碳排放（carbon）
      f4: 最小化住房供需结构偏离（balance）
  - 使用 pymoo NSGA-II，支持独立/协同治理策略
  - 全程 float64 预处理，与训练时 MinMaxScaler 数值等价

政策干预变量演化（Eq.18）：
    x_{c,k,t} = x_{c,k,baseline} × (1 + α_{c,k} × time_coeff(t))
    启动前 time_coeff=0，启动后线性增加至 1（到2050年）
"""

import gc
import os
import warnings
from datetime import datetime
from typing import Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# 碳排放强度表（kgCO₂/m²），与 carbon_intensity.py 一致
CARBON_FACTORS = {
    "ssp1": {2020:502.1037375,2025:458.5658774,2030:412.4512823,2035:400.9534834,
             2040:378.7082488,2045:368.7131579,2050:357.7070138},
    "ssp2": {2020:502.1037375,2025:482.7798522,2030:469.5382931,2035:460.9387871,
             2040:450.765133, 2045:443.1567752,2050:434.3226372},
    "ssp3": {2020:502.1037375,2025:482.998699, 2030:469.9591812,2035:461.5458794,
             2040:451.5435112,2045:444.0923924,2050:435.402273 },
    "ssp4": {2020:502.1037375,2025:482.0541773,2030:468.1517746,2035:458.9518954,
             2040:448.2342395,2045:440.1343992,2050:430.8576787},
    "ssp5": {2020:502.1037375,2025:482.6342483,2030:469.2589712,2035:460.5369055,
             2040:450.2511617,2045:442.5405336,2050:433.6133296},
}

TARGET_YEARS = [2035, 2040, 2045, 2050]


# --------------------------------------------------------------------------
# 性能指标追踪
# --------------------------------------------------------------------------

class PerformanceMetrics:
    """追踪 Hypervolume 和 Spacing 随代数的演化。"""

    def __init__(self):
        self.ref_point    = None
        self.hv_indicator = None
        self.history = {"generation": [], "hv": [], "spacing": [], "n_solutions": []}

    def update(self, generation: int, F: np.ndarray) -> Tuple[float, float]:
        if self.ref_point is None:
            self.ref_point    = F.max(axis=0) * 1.1
            self.hv_indicator = HV(ref_point=self.ref_point)
        try:
            hv = float(self.hv_indicator(F))
        except Exception:
            hv = 0.0
        spacing = float(cdist(F, F).copy())  # 下面正确计算
        spacing = self._calc_spacing(F)
        self.history["generation"].append(generation)
        self.history["hv"].append(hv)
        self.history["spacing"].append(spacing)
        self.history["n_solutions"].append(len(F))
        return hv, spacing

    @staticmethod
    def _calc_spacing(F: np.ndarray) -> float:
        if F is None or len(F) < 2:
            return 0.0
        dists = cdist(F, F)
        np.fill_diagonal(dists, np.inf)
        return float(dists.min(axis=1).std())

    def save_plot(self, path: str):
        df = pd.DataFrame(self.history)
        if df.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(df["generation"], df["hv"], "b-", linewidth=2)
        axes[0].set(xlabel="Generation", ylabel="Hypervolume",
                    title="Hypervolume Evolution")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(df["generation"], df["spacing"], "r-", linewidth=2)
        axes[1].set(xlabel="Generation", ylabel="Spacing",
                    title="Spacing Evolution (Lower is Better)")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  性能指标图已保存: {path}")


# --------------------------------------------------------------------------
# pymoo 问题定义
# --------------------------------------------------------------------------

class HousingOptimizationProblem(Problem):
    """
    住房供需优化问题（论文 4.6）。

    决策变量：每个城市 × 每个可调控变量的扰动率
              shape (n_cities × n_vars,)，范围 [-range, +range]

    四目标（均取最小化）：
      f1: new_supply 总量（↓ 降低新建需求）
      f2: -old_supply 总量（↓ 提升存量利用，取负转最小）
      f3: carbon 排放总量（↓ 减少建造碳排放）
      f4: 供需偏离平衡度（↓ 降低结构错配）
    """

    def __init__(
        self,
        scenario: str,
        df_base: pd.DataFrame,
        models: dict,
        scalers: dict,
        feature_cols: dict,
        city_names: list[str],
        var_config: dict,
        ramp_start: int = 2031,
        ramp_end: int   = 2050,
        opt_start: int  = 2031,
        opt_end: int    = 2050,
        eval_n_workers: int = 0,
        batch_size: int     = 0,
        use_log: bool       = True,
        log_eps: float      = 1e-6,
    ):
        self.scenario      = scenario
        self.df_base       = df_base
        self.models        = models
        self.scalers       = scalers
        self.feature_cols  = feature_cols
        self.city_names    = city_names
        self.var_config    = var_config
        self.ramp_start    = ramp_start
        self.ramp_end      = ramp_end
        self.opt_start     = opt_start
        self.opt_end       = opt_end
        self.use_log       = use_log
        self.log_eps       = log_eps
        self.carbon_map    = CARBON_FACTORS[scenario]

        # 确定参与优化的变量（须在某个模型特征列中）
        all_feat = {c for cols in feature_cols.values() for c in cols}
        self.opt_vars = [v for v in var_config if v in all_feat]
        print(f"  参与优化变量数: {len(self.opt_vars)}")
        print(f"  城市数:         {len(city_names)}")

        n_cities   = len(city_names)
        n_vars     = len(self.opt_vars)
        n_decision = n_cities * n_vars

        lowers = [-var_config[v]["range"] for _ in city_names for v in self.opt_vars]
        uppers = [+var_config[v]["range"] for _ in city_names for v in self.opt_vars]

        # 预计算
        self._precompute(df_base)
        self._setup_parallel(eval_n_workers, batch_size)

        # baseline
        bp = self._predict_batch(np.zeros((1, n_decision)))[0]
        self.baseline_ns     = float(bp[0])
        self.baseline_os     = float(bp[1])
        self.baseline_carbon = float(bp[2])
        self.baseline_na     = float(bp[3])
        print(f"  基线 new_supply:  {self.baseline_ns:.4f}")
        print(f"  基线 old_supply:  {self.baseline_os:.4f}")
        print(f"  基线 carbon:      {self.baseline_carbon:.4f}")

        super().__init__(
            n_var=n_decision, n_obj=4, n_constr=0,
            xl=np.array(lowers, dtype=np.float64),
            xu=np.array(uppers, dtype=np.float64),
        )

    def _precompute(self, df: pd.DataFrame):
        """预计算固定部分（非扰动列的缩放值），加速批量评估。"""
        N     = len(df)
        years = df["time"].values.astype(np.float64)

        # 城市索引映射
        city_to_idx     = {c: i for i, c in enumerate(self.city_names)}
        cities          = df["city_name"].values
        self.city_idx   = np.array([city_to_idx.get(c, -1) for c in cities], dtype=np.int32)
        self.safe_cidx  = np.where(self.city_idx >= 0, self.city_idx, 0).astype(np.int32)

        # 线性斜坡系数（Eq.18）
        ramp_span = float(self.ramp_end - self.ramp_start)
        ramp      = np.clip((years - self.ramp_start) / ramp_span, 0.0, 1.0)
        in_pert   = (years >= self.ramp_start) & (years <= self.ramp_end)
        ramp[~in_pert]           = 0.0
        ramp[self.city_idx < 0] = 0.0
        self.ramp      = ramp.astype(np.float64)
        self.ramp_mask = self.ramp > 0

        # 优化期 mask 和碳排放权重
        opt_mask    = (years >= self.opt_start) & (years <= self.opt_end)
        self.opt_w  = opt_mask.astype(np.float64)
        self.carb_w = np.array(
            [self.carbon_map.get(int(y), 0.0) for y in years], dtype=np.float64
        )
        na_mask    = np.isin(years.astype(np.int32), TARGET_YEARS)
        self.na_w  = na_mask.astype(np.float64)

        # 各目标可调变量的原始值
        self.base_pert = np.column_stack(
            [df[v].values.astype(np.float64) for v in self.opt_vars]
        )
        # lb / ub 向量
        self.pert_lb = np.array(
            [self.var_config[v]["lb"] if self.var_config[v]["lb"] is not None else -np.inf
             for v in self.opt_vars], dtype=np.float64
        )
        self.pert_ub = np.array(
            [self.var_config[v]["ub"] if self.var_config[v]["ub"] is not None else np.inf
             for v in self.opt_vars], dtype=np.float64
        )

        # 预缓存非扰动列
        opt_var_set   = set(self.opt_vars)
        inter_set     = {"new_supply", "old_supply", "demand"}
        self._caches  = {}
        for target in ("supply", "old_supply", "demand", "new_area"):
            cols = self.feature_cols[target]
            sc   = self.scalers[target]
            self._caches[target] = _build_cache(
                df, cols, sc, opt_var_set, inter_set,
                self.use_log, self.log_eps, N,
            )

        self.N = N

    def _setup_parallel(self, n_workers: int, batch_size: int):
        import multiprocessing as mp
        cores = mp.cpu_count()
        self.n_workers  = n_workers  if n_workers  > 0 else max(1, cores // 2)
        max_cols = max(c["n_cols"] for c in self._caches.values())
        self.batch_size = batch_size if batch_size > 0 else max(
            1, min(64, int(512 * 1024 * 1024 / max(self.N * max_cols * 8, 1)))
        )
        print(f"  并行: {self.n_workers} workers  批量: {self.batch_size}")

    def _predict_batch(self, X_chunk: np.ndarray) -> np.ndarray:
        """评估一批决策变量，返回 (B, 4) 目标值矩阵。"""
        B      = len(X_chunk)
        N      = self.N
        n_vars = len(self.opt_vars)
        n_c    = len(self.city_names)

        rates   = X_chunk.astype(np.float64).reshape(B, n_c, n_vars)
        rr      = rates[:, self.safe_cidx, :]           # (B, N, V)
        mult    = 1.0 + self.ramp[None, :, None] * rr
        pert    = self.base_pert[None, :, :] * mult     # (B, N, V) float64

        # lb/ub 截断（仅扰动行）
        if self.ramp_mask.any():
            sub = pert[:, self.ramp_mask, :]
            np.clip(sub, self.pert_lb[None, None, :],
                    self.pert_ub[None, None, :], out=sub)
            pert[:, self.ramp_mask, :] = sub

        # 第一层
        inter = {}
        for t in ("supply", "old_supply", "demand"):
            X = _assemble_X(B, N, self._caches[t], pert,
                            self.use_log, self.log_eps)
            inter[t] = np.maximum(
                self.models[t].predict(X).astype(np.float64).reshape(B, N), 0.0
            )
        inter["new_supply"] = np.maximum(inter["supply"] - inter["old_supply"], 0.0)

        # 第二层
        X_area = _assemble_X(B, N, self._caches["new_area"], pert,
                             self.use_log, self.log_eps, inter)
        new_area = np.maximum(
            self.models["new_area"].predict(X_area).astype(np.float64).reshape(B, N), 0.0
        )

        f1 = (inter["new_supply"] @ self.opt_w)  / 1e9
        f2 = -(inter["old_supply"] @ self.opt_w) / 1e9   # 最大化转最小化
        f3 = (new_area @ self.carb_w)            / 1e9
        na = (new_area @ self.na_w)              / 1e9
        f4 = np.abs(na - f1 - (-f2))

        return np.column_stack([f1, f2, f3, f4])

    def _evaluate(self, X_pop, out, *args, **kwargs):
        chunks  = [X_pop[i:i+self.batch_size]
                   for i in range(0, len(X_pop), self.batch_size)]
        if self.n_workers <= 1 or len(chunks) == 1:
            res = [self._predict_batch(c) for c in chunks]
        else:
            res = Parallel(n_jobs=self.n_workers, backend="threading")(
                delayed(self._predict_batch)(c) for c in chunks
            )
        out["F"] = np.vstack(res)

    def decode(self, x: np.ndarray) -> dict[str, dict[str, float]]:
        """将决策向量解码为 {city: {var: rate}} 字典。"""
        n_v = len(self.opt_vars)
        return {
            city: {var: float(x[ci * n_v + vi]) for vi, var in enumerate(self.opt_vars)}
            for ci, city in enumerate(self.city_names)
        }


# --------------------------------------------------------------------------
# 缓存与矩阵构建（内部工具函数）
# --------------------------------------------------------------------------

def _scaler_stats(scaler) -> Tuple[float, float]:
    return float(scaler.data_min_[0]), float(scaler.data_range_[0])


def _build_cache(df, cols, sc, opt_var_set, inter_set, use_log, eps, N):
    """预计算非扰动列的缩放值，减少热路径中的重复计算。"""
    cache = {"n_cols": len(cols), "cols": cols}
    pert_pos, pert_var_idx   = [], []
    non_pos                  = []
    inter_pos, inter_names   = [], []

    for i, col in enumerate(cols):
        if col in inter_set:
            inter_pos.append(i)
            inter_names.append(col)
        elif col in opt_var_set:
            pert_pos.append(i)
            pert_var_idx.append(list(opt_var_set).index(col)
                                if not hasattr(cache, "_var_list") else 0)
        else:
            non_pos.append(i)

    # 非扰动列预先缩放
    if non_pos:
        ns = np.empty((N, len(non_pos)), dtype=np.float64)
        for j, p in enumerate(non_pos):
            col   = cols[p]
            mn, rg = _scaler_stats(sc[col])
            raw   = df[col].values.astype(np.float64)
            raw   = _fill_nan(raw)
            ns[:, j] = _scale(raw, mn, rg, use_log, eps)
        cache["non_pert_scaled"] = ns
    else:
        cache["non_pert_scaled"] = np.empty((N, 0), dtype=np.float64)

    if pert_pos:
        cache["pert_mn"]  = np.array([_scaler_stats(sc[cols[p]])[0] for p in pert_pos], np.float64)
        cache["pert_rg"]  = np.array([_scaler_stats(sc[cols[p]])[1] for p in pert_pos], np.float64)
    if inter_pos:
        cache["inter_mn"] = np.array([_scaler_stats(sc[cols[p]])[0] for p in inter_pos], np.float64)
        cache["inter_rg"] = np.array([_scaler_stats(sc[cols[p]])[1] for p in inter_pos], np.float64)

    cache["pert_pos"]    = np.array(pert_pos,    np.int32)
    cache["pert_vi"]     = np.array(pert_var_idx,np.int32)
    cache["non_pos"]     = np.array(non_pos,     np.int32)
    cache["inter_pos"]   = np.array(inter_pos,   np.int32)
    cache["inter_names"] = inter_names
    return cache


def _fill_nan(v: np.ndarray) -> np.ndarray:
    if np.isnan(v).any():
        m = np.nanmean(v)
        v = np.where(np.isnan(v), 0.0 if np.isnan(m) else m, v)
    return v


def _scale(raw, mn, rg, use_log, eps):
    x = (raw - mn) / (rg + 1e-30)
    return np.log(x + eps) if use_log else x


def _assemble_X(B, N, cache, pert, use_log, eps, inter_map=None):
    """拼接特征矩阵，返回 float32 供 XGBoost 预测。"""
    n_cols = cache["n_cols"]
    X = np.empty((B * N, n_cols), dtype=np.float64)

    non_pos = cache["non_pos"]
    if len(non_pos):
        tile = np.broadcast_to(cache["non_pert_scaled"], (B, N, len(non_pos)))
        X[:, non_pos] = tile.reshape(B * N, -1)

    pp = cache["pert_pos"]
    if len(pp):
        vals  = pert[:, :, cache["pert_vi"]].reshape(B * N, -1)
        X[:, pp] = _scale(vals, cache["pert_mn"], cache["pert_rg"], use_log, eps)

    ip = cache["inter_pos"]
    if len(ip) and inter_map is not None:
        stacked = np.stack([inter_map[n] for n in cache["inter_names"]], axis=-1)
        X[:, ip] = _scale(stacked.reshape(B * N, -1),
                          cache["inter_mn"], cache["inter_rg"], use_log, eps)

    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# --------------------------------------------------------------------------
# 主优化器
# --------------------------------------------------------------------------

class NSGA2Optimizer:
    """
    NSGA-II 多目标优化器（论文 4.6）。

    Args:
        scenario:       SSP 情景（ssp1~ssp5）
        strategy_name:  治理策略名称（用于输出目录命名）
        var_config:     可调控变量配置 {var: {range, lb, ub}}
        pop_size:       种群大小
        n_generations:  迭代代数
        ramp_start:     政策干预斜坡启动年份（Eq.18）
    """

    def __init__(
        self,
        scenario: str,
        strategy_name: str,
        predict_csv: str,
        city_csv: str,
        model_dir: str,
        output_base: str,
        var_config: dict,
        pop_size: int      = 500,
        n_generations: int = 100,
        ramp_start: int    = 2031,
        ramp_end: int      = 2050,
        opt_start: int     = 2031,
        opt_end: int       = 2050,
        n_offsprings: int  = 500,
        crossover_prob: float = 0.9,
        crossover_eta: float  = 15,
        mutation_eta: float   = 20,
        metrics_interval: int = 2,
        random_state: int     = 42,
        eval_n_workers: int   = 0,
        batch_size: int       = 0,
        use_log: bool         = True,
        log_eps: float        = 1e-6,
    ):
        self.scenario       = scenario
        self.strategy_name  = strategy_name
        self.predict_csv    = predict_csv
        self.city_csv       = city_csv
        self.model_dir      = model_dir
        self.output_dir     = os.path.join(output_base, scenario, strategy_name)
        self.var_config     = var_config
        self.pop_size       = pop_size
        self.n_generations  = n_generations
        self.ramp_start     = ramp_start
        self.ramp_end       = ramp_end
        self.opt_start      = opt_start
        self.opt_end        = opt_end
        self.n_offsprings   = n_offsprings
        self.crossover_prob = crossover_prob
        self.crossover_eta  = crossover_eta
        self.mutation_eta   = mutation_eta
        self.metrics_intv   = metrics_interval
        self.random_state   = random_state
        self.eval_n_workers = eval_n_workers
        self.batch_size     = batch_size
        self.use_log        = use_log
        self.log_eps        = log_eps

    def run(self):
        t0 = datetime.now()
        os.makedirs(self.output_dir, exist_ok=True)

        # 加载模型（04_causal 以数字开头，不能用相对导入，改用 importlib）
        import importlib.util as _ilu
        _tlp_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "04_causal", "two_layer_predictor.py",
        )
        _spec = _ilu.spec_from_file_location("two_layer_predictor", _tlp_path)
        _tlp  = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_tlp)
        models, scalers, feature_cols = _tlp.load_models(self.model_dir)

        # 加载数据
        df = pd.read_csv(self.predict_csv)
        df["time"] = df["time"].astype(np.float64)

        city_df    = pd.read_csv(self.city_csv)
        city_names = city_df.iloc[:, 0].dropna().unique().tolist()

        # 构建问题
        print(f"\n{'='*60}\n情景={self.scenario}  策略={self.strategy_name}\n{'='*60}")
        problem = HousingOptimizationProblem(
            scenario       = self.scenario,
            df_base        = df,
            models         = models,
            scalers        = scalers,
            feature_cols   = feature_cols,
            city_names     = city_names,
            var_config     = self.var_config,
            ramp_start     = self.ramp_start,
            ramp_end       = self.ramp_end,
            opt_start      = self.opt_start,
            opt_end        = self.opt_end,
            eval_n_workers = self.eval_n_workers,
            batch_size     = self.batch_size,
            use_log        = self.use_log,
            log_eps        = self.log_eps,
        )

        # NSGA-II
        metrics  = PerformanceMetrics()
        last_t   = [datetime.now()]
        intv     = self.metrics_intv

        class _Callback:
            def __call__(self_, alg):
                gen = alg.n_gen
                if gen % intv == 0 or gen == 1:
                    F = alg.pop.get("F")
                    hv, sp = metrics.update(gen, F)
                    el = (datetime.now() - last_t[0]).total_seconds()
                    last_t[0] = datetime.now()
                    print(f"  代 {gen:3d}/{alg.termination.n_max_gen}: "
                          f"HV={hv:.4f}  Spacing={sp:.4f}  "
                          f"Pareto={len(F)}  {el:.1f}s")
                    if gen % (intv * 2) == 0:
                        gc.collect()

        algorithm = NSGA2(
            pop_size     = self.pop_size,
            n_offsprings = self.n_offsprings,
            sampling     = FloatRandomSampling(),
            crossover    = SBX(prob=self.crossover_prob, eta=self.crossover_eta),
            mutation     = PM(eta=self.mutation_eta),
            eliminate_duplicates=True,
        )

        print(f"\n开始优化  种群={self.pop_size}  代数={self.n_generations}  "
              f"决策变量数={problem.n_var}")
        result = minimize(
            problem, algorithm,
            get_termination("n_gen", self.n_generations),
            seed=self.random_state,
            callback=_Callback(),
            verbose=False,
            save_history=False,
        )
        print(f"优化完成  Pareto={len(result.F)}  耗时={datetime.now()-t0}")

        # 保存结果
        self._save(result, problem, metrics, df)
        print(f"\n总耗时: {datetime.now()-t0}")

    def _save(self, result, problem, metrics, df_base):
        F, X = result.F, result.X
        out  = self.output_dir

        # Pareto 前沿 CSV
        rows = []
        n_v  = len(problem.opt_vars)
        for i in range(len(F)):
            row = {"solution_id": i+1,
                   "f1_new_supply": F[i,0], "f2_old_supply": F[i,1],
                   "f3_carbon": F[i,2],     "f4_balance": F[i,3]}
            for ci, city in enumerate(problem.city_names):
                for vi, var in enumerate(problem.opt_vars):
                    row[f"rate_{city}_{var}"] = X[i, ci*n_v+vi]
            rows.append(row)
        pareto_path = os.path.join(out, "pareto_all.csv")
        pd.DataFrame(rows).to_csv(pareto_path, index=False, encoding="utf-8-sig")
        print(f"  Pareto 前沿: {pareto_path}")

        # 综合最优解（归一化后各目标之和最小）
        F_norm   = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-10)
        best_idx = int(np.argmin(F_norm.sum(1)))
        best_x   = X[best_idx]
        best_rates = problem.decode(best_x)
        print(f"  综合最优解 #{best_idx+1}: "
              f"f1={F[best_idx,0]:.4f} f2={F[best_idx,1]:.4f} "
              f"f3={F[best_idx,2]:.4f} f4={F[best_idx,3]:.4f}")

        # 最优解预测结果 CSV
        pred_path = os.path.join(out, "optimized_best.csv")
        df_out = df_base.copy()
        # （此处仅保存决策变量，完整预测可调用 two_layer_predictor.predict_new_area）
        for ci, city in enumerate(problem.city_names):
            for vi, var in enumerate(problem.opt_vars):
                df_out.loc[df_out["city_name"] == city, f"rate_{var}"] = \
                    best_rates[city][var]
        df_out.to_csv(pred_path, index=False, encoding="utf-8-sig")

        # 性能指标
        if metrics.history["generation"]:
            pd.DataFrame(metrics.history).to_csv(
                os.path.join(out, "metrics.csv"), index=False
            )
            metrics.save_plot(os.path.join(out, "metrics_plot.png"))

        # 文字报告
        rpt = os.path.join(out, "report.txt")
        with open(rpt, "w", encoding="utf-8") as f:
            f.write(f"NSGA-II 优化报告  {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"情景={self.scenario}  策略={self.strategy_name}\n")
            f.write(f"种群={self.pop_size}  代数={self.n_generations}\n")
            f.write(f"城市数={len(problem.city_names)}  "
                    f"变量数={len(problem.opt_vars)}  "
                    f"决策维度={problem.n_var}\n\n")
            f.write("Pareto 前沿（前20）:\n")
            for i in range(min(20, len(F))):
                f.write(f"  #{i+1}: f1={F[i,0]:.6f} f2={F[i,1]:.6f} "
                        f"f3={F[i,2]:.6f} f4={F[i,3]:.6f}\n")
        print(f"  报告: {rpt}")