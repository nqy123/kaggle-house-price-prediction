import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 10
TARGET = "SalePrice"
ID_COL = "Id"


def rmsle_log(y_true_log, y_pred_log):
    """比赛指标 RMSLE 等价于 log 目标上的 RMSE。"""
    return mean_squared_error(y_true_log, y_pred_log) ** 0.5


def load_data():
    """读取数据，并移除 House Prices 经典高影响异常点。"""
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")

    # 经典异常点：超大居住面积但售价异常低，会明显干扰回归边界。
    outlier_mask = (train["GrLivArea"] > 4000) & (train[TARGET] < 300000)
    train = train.loc[~outlier_mask].reset_index(drop=True)

    y = np.log1p(train[TARGET].values)
    train_id = train[ID_COL].copy()
    test_id = test[ID_COL].copy()

    train_features = train.drop(columns=[TARGET])
    return train_features, test, y, train_id, test_id


def preprocess(train, test):
    """统一处理 train/test，避免编码不一致。"""
    n_train = len(train)
    all_data = pd.concat([train, test], axis=0, ignore_index=True)
    all_data = all_data.drop(columns=[ID_COL])

    # 这些缺失值在数据说明里通常表示“没有该设施”。
    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType",
        "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond",
        "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType",
    ]
    for col in none_cols:
        if col in all_data:
            all_data[col] = all_data[col].fillna("None")

    zero_cols = [
        "GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2",
        "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
        "MasVnrArea",
    ]
    for col in zero_cols:
        if col in all_data:
            all_data[col] = all_data[col].fillna(0)

    # LotFrontage 与街区强相关，按 Neighborhood 中位数填充。
    if "LotFrontage" in all_data:
        all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )

    # 其余类别缺失用众数，数值缺失用中位数。
    cat_cols = all_data.select_dtypes(include=["object"]).columns.tolist()
    num_cols = all_data.select_dtypes(exclude=["object"]).columns.tolist()
    for col in cat_cols:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    for col in num_cols:
        all_data[col] = all_data[col].fillna(all_data[col].median())

    # 一些数值列本质是类别。
    for col in ["MSSubClass", "YrSold", "MoSold"]:
        if col in all_data:
            all_data[col] = all_data[col].astype(str)

    # 有序质量特征映射，保留自然顺序。
    quality_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    for col in [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
        "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
    ]:
        if col in all_data:
            all_data[col] = all_data[col].map(quality_map).fillna(0).astype(int)

    exposure_map = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
    if "BsmtExposure" in all_data:
        all_data["BsmtExposure"] = all_data["BsmtExposure"].map(exposure_map).fillna(0).astype(int)

    finish_map = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    for col in ["BsmtFinType1", "BsmtFinType2"]:
        if col in all_data:
            all_data[col] = all_data[col].map(finish_map).fillna(0).astype(int)

    # 房价比赛常用强特征。
    all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
    all_data["TotalBath"] = (
        all_data["FullBath"] + 0.5 * all_data["HalfBath"]
        + all_data["BsmtFullBath"] + 0.5 * all_data["BsmtHalfBath"]
    )
    all_data["TotalPorchSF"] = (
        all_data["OpenPorchSF"] + all_data["3SsnPorch"]
        + all_data["EnclosedPorch"] + all_data["ScreenPorch"]
        + all_data["WoodDeckSF"]
    )
    all_data["HouseAge"] = all_data["YrSold"].astype(int) - all_data["YearBuilt"]
    all_data["RemodAge"] = all_data["YrSold"].astype(int) - all_data["YearRemodAdd"]
    all_data["GarageAge"] = all_data["YrSold"].astype(int) - all_data["GarageYrBlt"]
    all_data["GarageAge"] = all_data["GarageAge"].clip(lower=0, upper=120)
    all_data["IsRemodeled"] = (all_data["YearBuilt"] != all_data["YearRemodAdd"]).astype(int)
    all_data["IsNew"] = (all_data["YrSold"].astype(int) == all_data["YearBuilt"]).astype(int)
    all_data["HasPool"] = (all_data["PoolArea"] > 0).astype(int)
    all_data["HasGarage"] = (all_data["GarageArea"] > 0).astype(int)
    all_data["HasBsmt"] = (all_data["TotalBsmtSF"] > 0).astype(int)
    all_data["HasFireplace"] = (all_data["Fireplaces"] > 0).astype(int)
    all_data["OverallScore"] = all_data["OverallQual"] * all_data["OverallCond"]
    all_data["QualitySF"] = all_data["OverallQual"] * all_data["TotalSF"]

    # 对偏态数值特征做 Box-Cox，线性模型会明显受益。
    numeric_feats = all_data.select_dtypes(exclude=["object"]).columns
    skewness = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewed_feats = skewness[abs(skewness) > 0.75].index
    for feat in skewed_feats:
        min_value = all_data[feat].min()
        if min_value < 0:
            all_data[feat] = all_data[feat] - min_value
        all_data[feat] = boxcox1p(all_data[feat], 0.15)

    all_data = pd.get_dummies(all_data, drop_first=False)
    train_x = all_data.iloc[:n_train].copy()
    test_x = all_data.iloc[n_train:].copy()
    return train_x, test_x


def get_models():
    """回归赛小数据强基线模型集合。"""
    models = {
        "ridge": make_pipeline(RobustScaler(), RidgeCV(alphas=[5, 10, 15, 20, 30, 50])),
        "lasso": make_pipeline(RobustScaler(), Lasso(alpha=0.00045, random_state=RANDOM_STATE, max_iter=50000)),
        "elastic": make_pipeline(
            RobustScaler(),
            ElasticNet(alpha=0.00055, l1_ratio=0.75, random_state=RANDOM_STATE, max_iter=50000),
        ),
        "krr": make_pipeline(RobustScaler(), KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)),
        "gbr": GradientBoostingRegressor(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=4,
            max_features="sqrt",
            min_samples_leaf=15,
            min_samples_split=10,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["xgb"] = XGBRegressor(
            n_estimators=2600,
            learning_rate=0.018,
            max_depth=3,
            min_child_weight=2,
            subsample=0.72,
            colsample_bytree=0.68,
            reg_alpha=0.0006,
            reg_lambda=0.9,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except Exception as exc:
        print(f"[WARN] XGBoost 不可用: {exc}")

    try:
        from lightgbm import LGBMRegressor

        models["lgbm"] = LGBMRegressor(
            objective="regression",
            n_estimators=2600,
            learning_rate=0.018,
            num_leaves=5,
            max_depth=3,
            min_child_samples=18,
            subsample=0.75,
            colsample_bytree=0.65,
            reg_alpha=0.0005,
            reg_lambda=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception as exc:
        print(f"[WARN] LightGBM 不可用: {exc}")

    try:
        from catboost import CatBoostRegressor

        models["cat"] = CatBoostRegressor(
            iterations=2200,
            learning_rate=0.018,
            depth=4,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
    except Exception as exc:
        print(f"[WARN] CatBoost 不可用: {exc}")

    return models


def train_oof(models, x, y, test_x):
    """训练每个模型的 10 折 OOF，并保存基础预测。"""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = {}
    test_preds = {}
    scores = {}

    for name, model in models.items():
        print(f"\n========== {name} ==========")
        oof = np.zeros(len(x))
        pred = np.zeros(len(test_x))
        fold_scores = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(x, y), 1):
            est = clone(model)
            est.fit(x.iloc[tr_idx], y[tr_idx])
            va_pred = est.predict(x.iloc[va_idx])
            te_pred = est.predict(test_x)
            oof[va_idx] = va_pred
            pred += te_pred / N_SPLITS

            score = rmsle_log(y[va_idx], va_pred)
            fold_scores.append(score)
            print(f"{name} fold {fold}: RMSE={score:.6f}")

        oof_score = rmsle_log(y, oof)
        print(f"{name} mean={np.mean(fold_scores):.6f}, std={np.std(fold_scores):.6f}, OOF={oof_score:.6f}")

        oof_preds[name] = oof
        test_preds[name] = pred
        scores[name] = {
            "cv_rmse": float(np.mean(fold_scores)),
            "std_rmse": float(np.std(fold_scores)),
            "oof_rmse": float(oof_score),
        }

        pd.DataFrame({"SalePrice_log_oof": oof}).to_csv(OUTPUT_DIR / f"oof_{name}.csv", index=False)
        pd.DataFrame({"SalePrice_log_pred": pred}).to_csv(OUTPUT_DIR / f"pred_{name}.csv", index=False)

    return oof_preds, test_preds, scores


def optimize_blend(oof_preds, y):
    """基于 OOF 搜索非负且和为 1 的融合权重。"""
    names = list(oof_preds)
    matrix = np.column_stack([oof_preds[n] for n in names])

    def objective(w):
        blend = matrix @ w
        return rmsle_log(y, blend)

    n = len(names)
    x0 = np.ones(n) / n
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = [(0, 1)] * n
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    if not result.success:
        print(f"[WARN] 权重优化失败，使用简单平均: {result.message}")
        weights = x0
    else:
        weights = result.x

    blend_score = objective(weights)
    return names, weights, float(blend_score)


def save_submission(test_id, pred_log, filename):
    """保存 Kaggle 提交文件，预测值从 log 空间还原。"""
    pred = np.expm1(pred_log)
    pred = np.clip(pred, 1, None)
    submission = pd.DataFrame({ID_COL: test_id, TARGET: pred})
    path = OUTPUT_DIR / filename
    submission.to_csv(path, index=False)
    print(f"保存提交文件: {path}")
    print(submission[TARGET].describe())
    return path


def main():
    train, test, y, train_id, test_id = load_data()
    x, test_x = preprocess(train, test)
    print(f"训练集: {x.shape}, 测试集: {test_x.shape}")

    models = get_models()
    oof_preds, test_preds, scores = train_oof(models, x, y, test_x)

    names, weights, blend_oof = optimize_blend(oof_preds, y)
    print("\n========== Blend ==========")
    for name, weight in zip(names, weights):
        print(f"{name}: weight={weight:.6f}, oof_rmse={scores[name]['oof_rmse']:.6f}")
    print(f"blend OOF RMSE={blend_oof:.6f}")

    blend_oof_pred = np.column_stack([oof_preds[n] for n in names]) @ weights
    blend_test_pred = np.column_stack([test_preds[n] for n in names]) @ weights

    pd.DataFrame({ID_COL: train_id, "SalePrice_log_oof": blend_oof_pred}).to_csv(
        OUTPUT_DIR / "oof_best_blend.csv", index=False
    )
    pd.DataFrame({ID_COL: test_id, "SalePrice_log_pred": blend_test_pred}).to_csv(
        OUTPUT_DIR / "pred_best_blend.csv", index=False
    )
    submission_path = save_submission(test_id, blend_test_pred, "submission_best_blend.csv")

    experiment = {
        "competition": "house-prices-advanced-regression-techniques",
        "target": TARGET,
        "metric": "RMSLE",
        "n_splits": N_SPLITS,
        "removed_outliers": 2,
        "models": scores,
        "blend_models": names,
        "blend_weights": {n: float(w) for n, w in zip(names, weights)},
        "blend_oof_rmse": blend_oof,
        "submission_file": str(submission_path.relative_to(ROOT)),
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "experiment_id": "house_price_best_blend",
                "model": "OOF optimized blend",
                "features": "manual FE + skew correction + one-hot",
                "cv_rmse": "",
                "oof_rmse": blend_oof,
                "std_rmse": "",
                "submission_file": "outputs/submission_best_blend.csv",
                "notes": "Ridge/Lasso/ElasticNet/KRR/GBR/XGB/LGBM/CatBoost OOF 权重融合",
            }
        ]
    ).to_csv(OUTPUT_DIR / "experiment_log.csv", index=False)


if __name__ == "__main__":
    main()
