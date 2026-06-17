# Kaggle 房价预测项目

这是 Kaggle 入门赛 **House Prices - Advanced Regression Techniques** 的高分整理版项目。

当前版本已完成本地训练、Kaggle 线上提交和结果归档。最终保留的最佳提交文件为：

`outputs/submission_best_blend.csv`

## 当前最佳结果

| 项目 | 内容 |
|---|---|
| 比赛 | House Prices - Advanced Regression Techniques |
| Kaggle slug | `house-prices-advanced-regression-techniques` |
| 任务类型 | 回归 |
| 目标列 | `SalePrice` |
| ID 列 | `Id` |
| 评价指标 | RMSLE |
| 最佳方案 | 多模型 OOF 权重融合 |
| 本地 OOF RMSE | `0.106410` |
| Kaggle Public Score | `0.12304` |
| 线上分位 | 约前 `20.0%` |
| 是否达到前 40% | 是 |

## 方法概览

核心脚本：

`src/train_best.py`

主要步骤：

1. 移除 House Prices 经典高影响异常点。
2. 对 `SalePrice` 使用 `log1p`，使训练目标与 RMSLE 对齐。
3. 统一处理 train/test 缺失值。
4. 构造房价领域特征，包括总面积、总浴室数、门廊面积、房龄、翻新状态、车库/地下室/泳池标记等。
5. 对偏态数值特征做 Box-Cox 修正。
6. 对类别特征做 one-hot 编码。
7. 使用 10 折 KFold 训练多个模型。
8. 基于 OOF 预测搜索非负且和为 1 的融合权重。
9. 生成最终提交文件。

参与融合的模型：

- Ridge
- Lasso
- ElasticNet
- Kernel Ridge
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

## 项目结构

```text
kaggle房价预测/
├── data/
│   ├── train.csv
│   └── test.csv
├── outputs/
│   ├── submission_best_blend.csv
│   ├── oof_best_blend.csv
│   ├── pred_best_blend.csv
│   ├── experiment_log.csv
│   ├── experiment_summary.json
│   └── best_result_summary.csv
├── src/
│   └── train_best.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 复现方式

安装依赖：

```bash
pip install -r requirements.txt
```

运行训练：

```bash
python src/train_best.py
```

提交到 Kaggle：

```bash
kaggle competitions submit \
  -c house-prices-advanced-regression-techniques \
  -f outputs/submission_best_blend.csv \
  -m "best blend oof rmse 0.106410"
```

查看提交成绩：

```bash
kaggle competitions submissions -c house-prices-advanced-regression-techniques
```

## 最佳提交文件

`outputs/submission_best_blend.csv`

线上 public score：

`0.12304`

