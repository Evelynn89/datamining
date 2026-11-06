# H&M 个性化时尚推荐

## 目标

为每位顾客推荐 12 件商品，以 MAP@12 为主指标，同时报告 Recall@12 和 HitRate@12。保留 [CHANGELOG](optimization/code/CHANGELOG.md) 对应的原方案：五路召回（时间衰减复购、同天关联、隐式协同过滤、年龄分群热门、全局热门）、每人最多 100 个候选、31 个特征（26 个数值、5 个类别）和 LightGBM LambdaRank。

## 数据集

使用 Kaggle [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) 数据。原始文件置于根目录 `data/`：`transactions_train.csv`、`customers.csv`、`articles.csv`、`sample_submission.csv`。共 31,788,324 笔交易、1,371,980 位顾客、105,542 件商品，交易日期为 2018-09-20 至 2020-09-22。

历史与特征统计截止 2020-09-15，目标周为 2020-09-16 至 2020-09-22，含 68,984 位购买用户和 213,728 个去重用户商品对。顾客键使用 ID 后 16 位十六进制转成的 `uint64`；按 `customer_key % 10 < 8` 划为训练用户，其余为验证用户：55,265 / 13,719，约 80% / 20%。验证用户也用于早停，因此表中结果是本地验证成绩，不能直接作为独立测试或 Kaggle 榜单成绩。

## 运行命令

在项目根目录执行；使用 Python 3.10 和 CPU，无需 NPU、CUDA 或 RAPIDS。已有兼容环境时可直接安装依赖。

```bash
conda create -n hm-recsys python=3.10 -y
conda activate hm-recsys
python -m pip install -r optimization/requirements.txt

# 从原始数据执行完整流程
python optimization/run_notebook.py optimization/code/01_data_cleaning_eda.ipynb
python optimization/run_notebook.py optimization/code/02_data_analysis_visualization.ipynb
python optimization/run_notebook.py optimization/code/03_pattern_mining_features.ipynb
python optimization/run_notebook.py optimization/code/04_final_candidate_feature_model.ipynb

# 检查 MAP、排序方向、热门补齐、去重和无符号用户键
python -m unittest discover -s optimization -p test_evaluation.py -v
```

已有 `optimization/output/clean/` 和 `optimization/output/mining/` 输入时，仅执行 `04` 即可重建候选、特征、两组离线基线和最终模型。本次在现有 `npu` 环境（Python 3.10.20，依赖版本见 `optimization/requirements.txt`）中用 CPU 完整重跑 `04`，复用清洗与挖掘输入，未重跑 `01`—`03`。两组基线在 `04` 中按相同目标周和用户划分评估；根目录两个 baseline 文件夹保留公开方案参考 Notebook。

## 关键实验结果

以下为 2026-11-02 17:36:17 +0800 的重跑输出，均在同一组 13,719 名验证用户上计算。

| 方案 | MAP@12 | Recall@12 | HitRate@12 |
| --- | ---: | ---: | ---: |
| Baseline 1：近期复购 + 热门补齐 | 0.027734496 | 0.053060961 | 0.102631387 |
| Baseline 2：近期购买 + 同天伴侣 + 热门补齐 | 0.024913533 | 0.050712393 | 0.097747649 |
| LightGBM LambdaRank（31 特征） | 0.038757945 | 0.075663756 | 0.144981413 |

最终模型 MAP@12 为 **0.038757945492994**，相对 Baseline 1 / Baseline 2 提升 **39.75% / 55.57%**。候选池共 6,047,766 行，全体目标用户的平均 Recall@100 为 0.124243583；最佳迭代为 226，候选级 AUC 为 0.822038716（本次重新计算）。

MAP 使用累计命中公式：`AP@12 = sum(label_r * cumulative_hits_r / r) / min(12, truth_count)`，再对全部验证用户取平均；预测先按用户商品去重，无命中用户计 0。排名截断升序、分数排序降序；Baseline 2 对不足 12 个候选的用户去重后补热门。7 项回归测试通过，另从保存的 Top12 独立复算了三项指标，并核对保存模型的预测分数。

模型、候选、31 特征数据、Top12 预测、对比 CSV、图表和 JSON 摘要均位于 `optimization/output/modeling/final_model/`；运行日志为 `optimization/output/modeling/rerun_20261102.log`，独立核验及文件哈希见该模型目录的 `rerun_verification.json`。旧实验结果、未重算的 03 挖掘指标和图表均已删除。当前仅保留最终模型产物、必要的数据与挖掘输入，以及当前有效的 EDA 表格和图表。
