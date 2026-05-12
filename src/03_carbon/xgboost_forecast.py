"""
xgboost_forecast.py
XGBoost 住房供需与新增建设面积预测

对应论文 4.4 节：
  - 以 10km 网格为基本单元，基于 2013–2023 年历史数据训练
  - 在不同 SSP 情景下预测 2024–2050 年住房需求、潜在住房供应和新增建设面积
  - XGBoost 通过最小化平方误差损失 + 正则化项进行优化（Eqs.11-12）

预处理流程（三个任务统一）：
  1. NaN → 列均值填充
  2. MinMaxScaler 归一化（scaler 仅在训练集上 fit）
  3. log(x + ε) 变换（USE_LOG_TRANSFORM_X=True）
  4. nan_to_num 兜底

支持两种模式：
  A. 直接训练并预测（完整流程）
  B. 加载已保存的 pkl 进行复现预测（replay 模式）
"""

import os
import pickle
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# 默认超参数（论文附录 Supplementary Table）
# --------------------------------------------------------------------------

DEFAULT_PARAMS = {
    # 4.4 住房需求预测
    "demand": {
        "subsample": 0.7, "reg_lambda": 1.0, "reg_alpha": 0.1,
        "n_estimators": 500, "min_child_weight": 5, "max_depth": 13,
        "learning_rate": 0.05, "gamma": 0.5,
        "colsample_bytree": 0.7, "colsample_bylevel": 0.6,
    },
    # 4.4 住房供应预测（存量供应）
    "supply": {
        "subsample": 0.7, "reg_lambda": 1.0, "reg_alpha": 0.1,
        "n_estimators": 500, "min_child_weight": 5, "max_depth": 13,
        "learning_rate": 0.05, "gamma": 0.5,
        "colsample_bytree": 0.7, "colsample_bylevel": 0.6,
    },
    # 4.4 旧屋供应预测
    "old_supply": {
        "subsample": 0.9, "reg_lambda": 5.0, "reg_alpha": 0.1,
        "n_estimators": 400, "min_child_weight": 1, "max_depth": 9,
        "learning_rate": 0.1, "gamma": 0,
        "colsample_bytree": 0.7, "colsample_bylevel": 1.0,
    },
    # 4.4 新增建设面积预测
    "new_area": {
        "subsample": 0.9, "reg_lambda": 5.0, "reg_alpha": 0.1,
        "n_estimators": 500, "min_child_weight": 7, "max_depth": 7,
        "learning_rate": 0.1, "gamma": 0.1,
        "colsample_bytree": 0.9, "colsample_bylevel": 0.5,
    },
}

# 超参数搜索空间
PARAM_GRID = {
    "n_estimators":      [50, 100, 150, 200, 300, 400, 500],
    "max_depth":         [3, 5, 7, 9, 11, 13, 15],
    "learning_rate":     [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
    "min_child_weight":  [1, 3, 5, 7, 10],
    "subsample":         [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma":             [0, 0.1, 0.2, 0.3, 0.5, 1.0],
    "reg_alpha":         [0, 0.001, 0.01, 0.1, 1.0],
    "reg_lambda":        [0.1, 0.5, 1.0, 2.0, 5.0],
}


# --------------------------------------------------------------------------
# 核心预测器
# --------------------------------------------------------------------------

class XGBoostForecaster:
    """
    通用 XGBoost 预测器，支持 demand / supply+old_supply / new_area 三种任务。

    Args:
        target_cols:     目标变量列表，如 ['demand'] 或 ['supply', 'old_supply']
        exclude_cols:    从特征中排除的列
        model_dir:       模型 pkl 保存/加载目录
        n_folds:         K 折交叉验证折数
        use_hyperopt:    是否使用 RandomizedSearchCV 搜索超参数
        n_iter:          超参数搜索迭代次数
        use_log_x:       是否对归一化特征做 log(x + ε) 变换
        log_eps:         log 变换的 ε
        n_jobs:          并行进程数
        random_state:    随机种子
    """

    def __init__(
        self,
        target_cols: list[str],
        exclude_cols: list[str],
        model_dir: str,
        n_folds: int = 5,
        use_hyperopt: bool = False,
        n_iter: int = 300,
        use_log_x: bool = True,
        log_eps: float = 1e-6,
        n_jobs: int = 8,
        random_state: int = 42,
        default_params: dict = None,
    ):
        self.target_cols   = target_cols
        self.exclude_cols  = exclude_cols
        self.model_dir     = model_dir
        self.n_folds       = n_folds
        self.use_hyperopt  = use_hyperopt
        self.n_iter        = n_iter
        self.use_log_x     = use_log_x
        self.log_eps       = log_eps
        self.n_jobs        = n_jobs
        self.random_state  = random_state
        self.default_params = default_params or DEFAULT_PARAMS

        # 运行时状态
        self.df_train: pd.DataFrame = None
        self.df_predict: pd.DataFrame = None
        self.feature_cols: list[str] = []
        self.scalers: dict = {}
        self.models: dict = {}
        self.best_params: dict = {}
        self.fold_scores: dict = {}
        self.r2_train: dict = {}
        self.mae_scores: dict = {}
        self.rmse_scores: dict = {}

    # ------------------------------------------------------------------
    # 完整训练+预测流程
    # ------------------------------------------------------------------

    def run(
        self,
        train_csv: str,
        predict_csv: str,
        output_csv: str,
        report_path: str,
    ):
        """端到端：加载 → 预处理 → 训练 → 预测 → 保存。"""
        t0 = datetime.now()
        self._load_data(train_csv, predict_csv)
        self._normalize()
        self._train()
        self._predict()
        self._save_results(output_csv)
        self._save_report(report_path)
        self._save_models()
        print(f"\n总耗时: {datetime.now() - t0}")

    # ------------------------------------------------------------------
    # 复现预测（replay）：只加载 pkl，不重新训练
    # ------------------------------------------------------------------

    def replay(
        self,
        predict_csv: str,
        output_csv: str,
        compare_col: str | None = None,
    ):
        """
        从 model_dir 加载已保存的 pkl，对新数据直接预测。

        用于在不同 SSP 情景下快速复现或切换数据集，无需重新训练。

        Args:
            predict_csv:  待预测 CSV（格式与训练时相同）
            output_csv:   输出 CSV 路径
            compare_col:  如果 CSV 中已有该预测列，则输出对比统计
        """
        print("=" * 60 + "\n[replay] 加载 pkl 复现预测\n" + "=" * 60)

        # 加载 pkl
        model_path  = os.path.join(self.model_dir, f"xgb_model_{self.target_cols[0]}.pkl")
        scaler_path = os.path.join(self.model_dir, "scalers.pkl")
        feat_path   = os.path.join(self.model_dir, "feature_cols.pkl")

        # 兼容带/不带 target 后缀的命名
        for suffix in ("", f"_{self.target_cols[0]}"):
            alt = os.path.join(self.model_dir, f"scalers{suffix}.pkl")
            if os.path.exists(alt):
                scaler_path = alt
            alt = os.path.join(self.model_dir, f"feature_cols{suffix}.pkl")
            if os.path.exists(alt):
                feat_path = alt

        self.models[self.target_cols[0]] = _load_pkl(model_path)
        sc_obj = _load_pkl(scaler_path)
        self.scalers = sc_obj if isinstance(sc_obj, dict) else {self.target_cols[0]: sc_obj}
        self.feature_cols = _load_pkl(feat_path)

        print(f"  特征数: {len(self.feature_cols)}")

        # 读 CSV
        df = pd.read_csv(predict_csv)
        df["time"] = df["time"].astype(np.float32)

        # 预处理（严格对齐训练时逻辑）
        n, m = len(df), len(self.feature_cols)
        X = np.empty((n, m), dtype=np.float32)
        for j, col in enumerate(self.feature_cols):
            vals = df[col].values.astype(np.float64).reshape(-1, 1)
            if np.any(np.isnan(vals)):
                mean = np.nanmean(vals) or 0.0
                vals = np.where(np.isnan(vals), mean, vals)
            normed = self.scalers[col].transform(vals).flatten()
            if self.use_log_x:
                normed = np.log(normed + self.log_eps)
            X[:, j] = normed.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 预测
        target = self.target_cols[0]
        y_pred = np.maximum(self.models[target].predict(X), 0.0)
        print(f"  {target}: min={y_pred.min():.4f}  max={y_pred.max():.4f}  mean={y_pred.mean():.4f}")

        # 可选对比
        if compare_col and compare_col in df.columns:
            a = df[compare_col].values
            valid = ~(np.isnan(a) | np.isnan(y_pred))
            if valid.any():
                d = np.abs(a[valid] - y_pred[valid])
                print(f"\n  [对比] max_diff={d.max():.6f}  mean_diff={d.mean():.6f}"
                      f"  >1e-3比例={( d > 1e-3).mean():.2%}")

        # 输出
        df[target] = y_pred
        out_cols = ["lon", "lat", "time", "flag", target] + [
            c for c in df.columns if c not in ("lon", "lat", "time", "flag", target)
        ]
        out_cols = [c for c in out_cols if c in df.columns]
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df[out_cols].to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n  已保存: {output_csv}")

    # ------------------------------------------------------------------
    # 内部步骤
    # ------------------------------------------------------------------

    def _load_data(self, train_csv: str, predict_csv: str):
        print("\n[1/6] 加载数据")
        self.df_train   = pd.read_csv(train_csv)
        self.df_predict = pd.read_csv(predict_csv)
        for df in (self.df_train, self.df_predict):
            df["time"] = df["time"].astype(np.float32)
        self.feature_cols = [
            c for c in self.df_train.columns if c not in self.exclude_cols
        ]
        print(f"  训练: {self.df_train.shape}  预测: {self.df_predict.shape}"
              f"  特征数: {len(self.feature_cols)}")

    def _normalize(self):
        """MinMax 归一化 + log 变换，scaler 仅在训练集上 fit。"""
        print("\n[2/6] 数据预处理")
        for col in self.feature_cols:
            train_vals = self.df_train[col].values.reshape(-1, 1)
            if np.any(np.isnan(train_vals)):
                mean = np.nanmean(train_vals) or 0.0
                train_vals = np.where(np.isnan(train_vals), mean, train_vals)
                self.df_train[col] = train_vals.flatten()

            scaler = MinMaxScaler()
            scaler.fit(train_vals)
            self.scalers[col] = scaler

            for df in (self.df_train, self.df_predict):
                vals = df[col].values.reshape(-1, 1)
                if np.any(np.isnan(vals)):
                    vals = np.nan_to_num(vals, nan=float(np.nanmean(vals)))
                    df[col] = vals.flatten()
                normed = scaler.transform(vals).flatten()
                if self.use_log_x:
                    normed = np.log(normed + self.log_eps)
                df[col + "_norm"] = normed
        print("  完成")

    def _train(self):
        """K 折交叉验证训练（支持超参数搜索）。"""
        print("\n[3/6] 训练模型")
        norm_cols = [c + "_norm" for c in self.feature_cols]
        X_full = np.nan_to_num(self.df_train[norm_cols].values, nan=0.0)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for target in self.target_cols:
            print(f"\n  目标: {target}")
            y_full  = self.df_train[target].values
            mask    = ~np.isnan(y_full)
            X_train = X_full[mask]
            y_train = y_full[mask]
            print(f"  样本数: {len(y_train):,}")

            if self.use_hyperopt:
                model = self._hyperopt_train(target, X_train, y_train, kf)
            else:
                model = self._default_train(target, X_train, y_train, kf)

            self.models[target] = model

            # 全量训练集评估
            y_hat = model.predict(X_train)
            self.r2_train[target]   = r2_score(y_train, y_hat)
            self.mae_scores[target] = mean_absolute_error(y_train, y_hat)
            self.rmse_scores[target] = np.sqrt(mean_squared_error(y_train, y_hat))
            print(f"  全量 R²={self.r2_train[target]:.4f}"
                  f"  MAE={self.mae_scores[target]:.4f}"
                  f"  RMSE={self.rmse_scores[target]:.4f}")

            # 特征重要性 Top 10
            imp = model.feature_importances_
            top = np.argsort(imp)[::-1][:10]
            print("  特征重要性 Top10:")
            for rank, idx in enumerate(top, 1):
                print(f"    {rank:2d}. {norm_cols[idx]}: {imp[idx]:.4f}")

    def _hyperopt_train(self, target, X, y, kf):
        base = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=1,
                                 random_state=self.random_state, verbosity=0)
        rs = RandomizedSearchCV(base, PARAM_GRID, n_iter=self.n_iter, cv=kf,
                                scoring="r2", n_jobs=self.n_jobs,
                                random_state=self.random_state, verbose=1,
                                return_train_score=True)
        rs.fit(X, y)
        self.best_params[target] = rs.best_params_
        self._run_cv_detail(rs.best_estimator_, target, X, y, kf)
        return rs.best_estimator_

    def _default_train(self, target, X, y, kf):
        params = self.default_params.get(target, {}).copy()
        self.best_params[target] = params
        model = xgb.XGBRegressor(**params, objective="reg:squarederror",
                                  n_jobs=self.n_jobs,
                                  random_state=self.random_state, verbosity=0)
        self._run_cv_detail(model, target, X, y, kf)
        model.fit(X, y, verbose=False)
        return model

    def _run_cv_detail(self, model, target, X, y, kf):
        """运行详细 K 折交叉验证并打印每折得分。"""
        cv = cross_validate(
            model, X, y, cv=kf,
            scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
            return_train_score=True, n_jobs=self.n_jobs,
        )
        self.fold_scores[target] = {
            "train_r2":  cv["train_r2"],
            "test_r2":   cv["test_r2"],
            "test_mae":  -cv["test_neg_mean_absolute_error"],
            "test_rmse": -cv["test_neg_root_mean_squared_error"],
        }
        fs = self.fold_scores[target]
        print(f"\n  {self.n_folds} 折交叉验证:")
        print(f"  {'折':>4}  {'训练R²':>8}  {'验证R²':>8}  {'MAE':>10}  {'RMSE':>10}")
        print("  " + "-" * 48)
        for i in range(self.n_folds):
            print(f"  {i+1:>4}  {fs['train_r2'][i]:>8.4f}  {fs['test_r2'][i]:>8.4f}"
                  f"  {fs['test_mae'][i]:>10.4f}  {fs['test_rmse'][i]:>10.4f}")
        print("  " + "-" * 48)
        print(f"  {'均值':>4}  {fs['train_r2'].mean():>8.4f}  {fs['test_r2'].mean():>8.4f}"
              f"  {fs['test_mae'].mean():>10.4f}  {fs['test_rmse'].mean():>10.4f}")
        print(f"  {'标准差':>4}  {fs['train_r2'].std():>8.4f}  {fs['test_r2'].std():>8.4f}"
              f"  {fs['test_mae'].std():>10.4f}  {fs['test_rmse'].std():>10.4f}")

    def _predict(self):
        print("\n[4/6] 预测")
        norm_cols = [c + "_norm" for c in self.feature_cols]
        X_pred = np.nan_to_num(self.df_predict[norm_cols].values, nan=0.0)
        for target in self.target_cols:
            y_pred = np.maximum(self.models[target].predict(X_pred), 0.0)
            self.df_predict[target + "_predicted"] = y_pred
            print(f"  {target}: min={y_pred.min():.2f}  max={y_pred.max():.2f}"
                  f"  mean={y_pred.mean():.2f}")

    def _save_results(self, output_csv: str):
        print("\n[5/6] 保存预测结果")
        base_cols = ["lon", "lat", "time", "flag"]
        pred_cols = [t + "_predicted" for t in self.target_cols]
        out_cols  = base_cols + pred_cols + [
            c for c in self.feature_cols if c not in base_cols
        ]
        out_cols = [c for c in out_cols if c in self.df_predict.columns]
        df_out   = self.df_predict[out_cols].rename(
            columns={t + "_predicted": t for t in self.target_cols}
        )
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"  已保存: {output_csv}")

    def _save_report(self, report_path: str):
        print("\n[6/6] 保存评估报告")
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("XGBoost 评估报告\n")
            f.write(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"K 折: {self.n_folds}  超参搜索: {self.use_hyperopt}\n\n")
            for target in self.target_cols:
                f.write(f"[{target}]\n")
                f.write(f"  参数: {self.best_params.get(target)}\n")
                if target in self.fold_scores:
                    fs = self.fold_scores[target]
                    f.write(f"  CV 验证 R²: {fs['test_r2'].mean():.4f} ± {fs['test_r2'].std():.4f}\n")
                    for i in range(self.n_folds):
                        f.write(f"  Fold {i+1}: R²={fs['test_r2'][i]:.4f}"
                                f"  MAE={fs['test_mae'][i]:.4f}"
                                f"  RMSE={fs['test_rmse'][i]:.4f}\n")
                f.write(f"  全量 R²={self.r2_train.get(target, float('nan')):.4f}"
                        f"  MAE={self.mae_scores.get(target, float('nan')):.4f}"
                        f"  RMSE={self.rmse_scores.get(target, float('nan')):.4f}\n\n")
        print(f"  已保存: {report_path}")

    def _save_models(self):
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.feature_cols, os.path.join(self.model_dir, "feature_cols.pkl"))
        joblib.dump(self.scalers,      os.path.join(self.model_dir, "scalers.pkl"))
        for target in self.target_cols:
            joblib.dump(self.models[target],
                        os.path.join(self.model_dir, f"xgb_model_{target}.pkl"))
        print(f"\n  模型已保存: {self.model_dir}")


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------

def _load_pkl(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到: {path}")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)