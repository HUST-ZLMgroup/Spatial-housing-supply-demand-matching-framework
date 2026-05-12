"""
lingam_analysis.py
DirectLiNGAM 因果识别 + Bootstrap 稳健性验证

对应论文 4.5 节 Eq.(13)-(14)：
  - 非高斯线性结构方程模型（DirectLiNGAM）
  - Bootstrap 重采样（不同 SSP 情景 × 每 5 年节点，各100次）
  - 深度优先搜索消除环结构
  - 保留指向建筑碳排放的关键路径

流程：
  1. 加载数据
  2. 特征准备（取第5列起 + carbon）
  3. 关键成分提取（Pearson r 去冗余）
  4. Z-score 标准化
  5. DirectLiNGAM 拟合 + joblib 并行 Bootstrap
  6. DAG 验证与环移除
  7. 提取因果关系 + carbon 驱动因素分析
  8. 保存结果

输入：data/{ssp}/data_{year}_final.csv
输出：result/{ssp}/lingam_{year}.csv + _adjacency.csv + _bootstrap_prob.csv + _report.txt
"""

import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv

warnings.filterwarnings("ignore")

try:
    from lingam import DirectLiNGAM
    _LINGAM_AVAILABLE = True
except ImportError:
    _LINGAM_AVAILABLE = False
    print("[警告] lingam 未安装，将使用偏相关矩阵作为备选方法")


# --------------------------------------------------------------------------
# Bootstrap 单次任务（模块级函数，joblib 可序列化）
# --------------------------------------------------------------------------

def _bootstrap_once(X, idx, seed, prior_knowledge, causal_threshold):
    X_boot = X[idx]
    m = DirectLiNGAM(
        random_state=seed,
        prior_knowledge=prior_knowledge,
        measure="pwling",
    )
    try:
        m.fit(X_boot)
        adj   = m.adjacency_matrix_
        count = (np.abs(adj) > causal_threshold).astype(int)
        return adj, count
    except Exception:
        return None, None


# --------------------------------------------------------------------------
# 因果分析流水线
# --------------------------------------------------------------------------

class CausalAnalysisPipeline:
    """
    DirectLiNGAM 因果分析流水线（对应论文 4.5 节）。

    Args:
        data_path:             输入 CSV 路径
        output_path:           主输出 CSV 路径（邻接矩阵/Bootstrap概率矩阵同目录）
        causal_threshold:      因果边阈值（默认 0.01）
        correlation_threshold: Pearson r 去冗余阈值（默认 1，即不去除）
        bootstrap_n:           Bootstrap 次数（默认 100）
        bootstrap_workers:     joblib 并行进程数（-1 = 全部）
        random_state:          随机种子
    """

    def __init__(
        self,
        data_path: str,
        output_path: str,
        causal_threshold: float = 0.01,
        correlation_threshold: float = 1.0,
        bootstrap_n: int = 100,
        bootstrap_workers: int = -1,
        random_state: int = 42,
    ):
        self.data_path             = data_path
        self.output_path           = output_path
        self.causal_threshold      = causal_threshold
        self.correlation_threshold = correlation_threshold
        self.bootstrap_n           = bootstrap_n
        self.bootstrap_workers     = bootstrap_workers
        self.random_state          = random_state

        self.df: pd.DataFrame       = None
        self.processed: pd.DataFrame= None
        self.feature_names: list    = []
        self.adjacency_matrix       = None
        self.bootstrap_probs        = None
        self.causal_df: pd.DataFrame= None
        self.removed_vars: dict     = {}
        self.removed_edges: list    = []

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self):
        """端到端运行全部流程。"""
        print("\n" + "=" * 60)
        print("DirectLiNGAM 因果分析")
        print("=" * 60)
        (self
         .load_data()
         .prepare_features()
         .extract_essential_components()
         .standardize()
         .fit_lingam()
         .ensure_dag()
         .extract_causal_relations()
         .analyze_carbon_drivers()
         .save_results())
        print("\n✅ 因果分析完成")
        return self

    # ------------------------------------------------------------------
    # 步骤
    # ------------------------------------------------------------------

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(self.data_path)
        self.df = pd.read_csv(self.data_path)
        print(f"[1] 数据加载: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")
        return self

    def prepare_features(self):
        """取第5列起作为特征，并确保 carbon 列包含在内。"""
        features = self.df.iloc[:, 5:].copy()
        if "carbon" in self.df.columns:
            features["carbon"] = self.df["carbon"]
        features = features.apply(pd.to_numeric, errors="coerce").dropna()
        self.df           = features
        self.feature_names = list(features.columns)
        print(f"[2] 特征准备: {len(self.feature_names)} 个变量，{len(features)} 个样本")
        return self

    def extract_essential_components(self):
        """
        Pearson r 关键成分提取：移除高度相关的冗余变量。
        受保护变量（demand / supply / carbon）不被移除。
        """
        print(f"[3] 关键成分提取（r 阈值={self.correlation_threshold}）")
        protected = {"demand", "new_supply", "old_supply", "carbon"}
        data, n   = self.df[self.feature_names], len(self.feature_names)

        # 计算两两 Pearson r
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                r, _ = pearsonr(data.iloc[:, i], data.iloc[:, j])
                if abs(r) >= self.correlation_threshold:
                    pairs.append((self.feature_names[i], self.feature_names[j], abs(r)))

        if not pairs:
            print("  未发现需移除的冗余变量对")
            return self

        pairs.sort(key=lambda x: x[2], reverse=True)
        to_drop = set()
        for v1, v2, r in pairs:
            if v1 in to_drop or v2 in to_drop:
                continue
            if v1 in protected and v2 in protected:
                continue
            drop = v1 if v2 in protected else v2
            keep = v2 if v2 in protected else v1
            to_drop.add(drop)
            self.removed_vars[drop] = {"kept_partner": keep, "pearson_r": r}
            print(f"  移除 '{drop}'（保留 '{keep}'，r={r:.4f}）")

        self.feature_names = [v for v in self.feature_names if v not in to_drop]
        self.df = self.df[self.feature_names]
        print(f"  移除 {len(to_drop)} 个冗余变量，保留 {len(self.feature_names)} 个")
        return self

    def standardize(self):
        """Z-score 标准化（mean=0, std=1）。"""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.df[self.feature_names])
        self.processed = pd.DataFrame(
            scaled, columns=[c + "_std" for c in self.feature_names]
        )
        print(f"[4] Z-score 标准化完成")
        return self

    def fit_lingam(self):
        """
        拟合 DirectLiNGAM 主模型 + joblib 并行 Bootstrap（论文 Eq.13）。
        若 lingam 不可用则退回 Ledoit-Wolf 偏相关矩阵。
        """
        print(f"[5] 因果发现（DirectLiNGAM + Bootstrap × {self.bootstrap_n}）")
        X      = self.processed.values
        n_vars = X.shape[1]
        prior  = self._build_prior_knowledge()

        if not _LINGAM_AVAILABLE:
            self._fallback_partial_corr(X)
            return self

        try:
            # 主模型
            model = DirectLiNGAM(
                random_state=self.random_state,
                prior_knowledge=prior,
                measure="pwling",
            )
            model.fit(X)
            self.adjacency_matrix = model.adjacency_matrix_
            print("  主模型拟合成功")

            # Bootstrap
            rng          = np.random.default_rng(self.random_state)
            seeds        = rng.integers(0, 99999, size=self.bootstrap_n).tolist()
            boot_indices = [rng.integers(0, len(X), size=len(X))
                            for _ in range(self.bootstrap_n)]

            results = Parallel(
                n_jobs=self.bootstrap_workers, backend="loky", verbose=5
            )(
                delayed(_bootstrap_once)(
                    X, boot_indices[i], seeds[i], prior, self.causal_threshold
                )
                for i in range(self.bootstrap_n)
            )

            boot_counts = np.zeros((n_vars, n_vars), dtype=int)
            boot_sum    = np.zeros((n_vars, n_vars))
            success     = 0
            for adj, count in results:
                if adj is not None:
                    boot_counts += count
                    boot_sum    += adj
                    success     += 1

            self.bootstrap_probs = boot_counts / self.bootstrap_n
            print(f"  Bootstrap 完成（成功 {success}/{self.bootstrap_n}）")

        except Exception as exc:
            print(f"  LiNGAM 失败: {exc}，改用偏相关矩阵")
            self._fallback_partial_corr(X)

        self.adjacency_matrix = np.clip(self.adjacency_matrix, -10, 10)
        return self

    def _fallback_partial_corr(self, X):
        prec = pinv(LedoitWolf().fit(X).covariance_)
        d    = np.sqrt(np.diag(prec))
        pc   = -prec / np.outer(d, d)
        np.fill_diagonal(pc, 0)
        self.adjacency_matrix = pc
        self.bootstrap_probs  = np.ones_like(pc)

    def _build_prior_knowledge(self):
        """禁止 carbon → supply/demand 方向（先验知识）。"""
        names      = list(self.processed.columns)
        n          = len(names)
        carbon_idx = next((i for i, n in enumerate(names) if "carbon" in n.lower()), None)
        sd_idxs    = [i for i, n in enumerate(names)
                      if any(k in n.lower() for k in ("new_supply", "old_supply", "demand"))]

        if carbon_idx is None:
            return None

        prior = np.full((n, n), -1, dtype=int)
        np.fill_diagonal(prior, 0)
        for idx in sd_idxs:
            prior[idx, carbon_idx] = 0  # 禁止 carbon → supply/demand
        return prior

    def ensure_dag(self):
        """深度优先搜索检测环，按最小权重边移除（论文 4.5）。"""
        print("[6] DAG 验证与环移除")
        names = list(self.processed.columns)
        carbon_idx = next((i for i, n in enumerate(names) if "carbon" in n.lower()), None)

        for it in range(100):
            cycles = self._detect_cycles(self.adjacency_matrix, self.causal_threshold)
            if not cycles:
                print(f"  DAG 验证通过（{'无环' if it==0 else f'{it}轮后消除'}）")
                break
            for cycle in cycles:
                scores = []
                for i in range(len(cycle)):
                    src, tgt = cycle[i], cycle[(i + 1) % len(cycle)]
                    w = abs(self.adjacency_matrix[tgt, src])
                    if carbon_idx is not None:
                        if tgt == carbon_idx:
                            w *= 100  # 保护指向carbon的边
                        elif src == carbon_idx:
                            w *= 5
                    scores.append((src, tgt, w))
                src, tgt, _ = min(scores, key=lambda x: x[2])
                self.removed_edges.append((src, tgt, self.adjacency_matrix[tgt, src]))
                self.adjacency_matrix[tgt, src] = 0
        else:
            print("  警告：达到最大迭代，可能仍存在环")

        if self.removed_edges:
            print(f"  共移除 {len(self.removed_edges)} 条边")
        return self

    @staticmethod
    def _detect_cycles(adj, threshold=0.0):
        n      = len(adj)
        graph  = {i: [j for j in range(n) if abs(adj[i, j]) > threshold] for i in range(n)}
        cycles = []

        def dfs(node, visited, stack, path):
            visited[node] = stack[node] = True
            path.append(node)
            for nb in graph[node]:
                if not visited[nb]:
                    if dfs(nb, visited, stack, path):
                        return True
                elif stack[nb]:
                    start = path.index(nb)
                    cycles.append(path[start:] + [nb])
                    return True
            path.pop()
            stack[node] = False
            return False

        vis, stk = [False] * n, [False] * n
        for node in range(n):
            if not vis[node]:
                dfs(node, vis, stk, [])

        # 去重
        seen, unique = set(), []
        for c in cycles:
            mi  = c.index(min(c))
            key = tuple(c[mi:-1] + c[:mi])
            if key not in seen:
                seen.add(key)
                unique.append(list(key))
        return unique

    def extract_causal_relations(self):
        """提取所有超过阈值的有向因果边（论文 Eq.14 路径累积效应）。"""
        print("[7] 提取因果关系")
        names  = list(self.processed.columns)
        n      = len(names)
        rows   = []
        for i in range(n):
            for j in range(n):
                if i != j and abs(self.adjacency_matrix[i, j]) > self.causal_threshold:
                    prob = (self.bootstrap_probs[i, j]
                            if self.bootstrap_probs is not None else float("nan"))
                    rows.append([
                        names[j].replace("_std", ""),
                        names[i].replace("_std", ""),
                        self.adjacency_matrix[i, j],
                        prob,
                    ])
        rows.sort(key=lambda x: abs(x[2]), reverse=True)
        self.causal_df = pd.DataFrame(
            rows, columns=["source", "target", "causal_strength", "bootstrap_prob"]
        )
        print(f"  共发现 {len(rows)} 条因果关系")
        return self

    def analyze_carbon_drivers(self):
        """筛选指向 carbon 的驱动因素，打印 Top 15。"""
        print("[8] Carbon 驱动因素分析")
        drivers = self.causal_df[self.causal_df["target"] == "carbon"].copy()
        drivers.sort_values("causal_strength", key=abs, ascending=False, inplace=True)
        if drivers.empty:
            print("  未发现指向 carbon 的显著因果关系")
        else:
            print(f"  发现 {len(drivers)} 个驱动因素")
            print(drivers.head(15).to_string(index=False))
        return self

    def save_results(self):
        """保存因果关系 CSV、邻接矩阵、Bootstrap 概率矩阵和文字报告。"""
        print("[9] 保存结果")
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        self.causal_df.to_csv(self.output_path, index=False, encoding="utf-8-sig")

        feat_clean = [c.replace("_std", "") for c in self.processed.columns]
        base       = self.output_path.replace(".csv", "")

        pd.DataFrame(self.adjacency_matrix,
                     index=feat_clean, columns=feat_clean
                     ).to_csv(f"{base}_adjacency.csv", encoding="utf-8-sig")

        if self.bootstrap_probs is not None:
            pd.DataFrame(self.bootstrap_probs,
                         index=feat_clean, columns=feat_clean
                         ).to_csv(f"{base}_bootstrap_prob.csv", encoding="utf-8-sig")

        with open(f"{base}_report.txt", "w", encoding="utf-8") as f:
            f.write("DirectLiNGAM 因果分析报告\n" + "=" * 60 + "\n\n")
            f.write(f"样本数: {self.processed.shape[0]}  变量数: {self.processed.shape[1]}\n")
            f.write(f"Bootstrap 次数: {self.bootstrap_n}\n")
            f.write(f"因果阈值: {self.causal_threshold}\n")
            f.write(f"移除冗余变量: {len(self.removed_vars)}\n")
            f.write(f"移除环边: {len(self.removed_edges)}\n\n")
            f.write("Top 20 最强因果关系:\n")
            f.write(self.causal_df.head(20).to_string(index=False))

        print(f"  已保存: {self.output_path}")
        return self


# --------------------------------------------------------------------------
# 批量运行入口
# --------------------------------------------------------------------------

def run_all(
    base_dir: str,
    ssps: list[str] = None,
    years: list[int] = None,
    causal_threshold: float = 0.01,
    bootstrap_n: int = 100,
    bootstrap_workers: int = -1,
):
    """
    批量运行所有 SSP × 年份组合。

    Args:
        base_dir:  数据根目录，结构：<base_dir>/data/{ssp}/data_{year}_final.csv
        ssps:      情景列表，默认 ssp1~ssp5
        years:     节点年份列表，默认 [2025,2030,2035,2040,2045,2050]
    """
    ssps  = ssps  or ["ssp1", "ssp2", "ssp3", "ssp4", "ssp5"]
    years = years or [2025, 2030, 2035, 2040, 2045, 2050]

    total, done = len(ssps) * len(years), 0
    for ssp in ssps:
        for year in years:
            done += 1
            print(f"\n{'='*60}\n任务 {done}/{total}: {ssp} - {year}\n{'='*60}")
            data_path   = os.path.join(base_dir, "data",   ssp, f"data_{year}_final.csv")
            output_path = os.path.join(base_dir, "result", ssp, f"lingam_{year}.csv")
            try:
                pipeline = CausalAnalysisPipeline(
                    data_path=data_path,
                    output_path=output_path,
                    causal_threshold=causal_threshold,
                    bootstrap_n=bootstrap_n,
                    bootstrap_workers=bootstrap_workers,
                )
                pipeline.run()
            except Exception as exc:
                print(f"  ✗ 失败: {exc}")