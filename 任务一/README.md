# 任务一：电影评分预测

## 项目概述

预测 `data/test.txt` 中未知 `(user, item)` 对的评分。当前实现以 RMSE 和稳健复现为目标，保留 Baseline、FunkSVD、SVD++、Item-KNN 四类模型，并新增验证集加权融合。

**数据集**: 598 个训练用户、9,077 个训练物品、90,854 条训练评分；测试集 610 个用户、9,982 个待预测 `(u, i)` 对；稀疏度约 98.47%。

## 当前结果

所有结果使用固定随机种子 `42`，按用户分层留出 10% 训练评分作为验证集。模型选参后会用完整 `train.txt` 重新训练，再预测 `test.txt`。

| 模型 | Valid RMSE | 最优参数 | 完整训练策略 | 预测文件 |
|------|-----------:|----------|--------------|----------|
| **Weighted Ensemble** | **16.5188** | Baseline 0.20 + FunkSVD 0.65 + SVD++ 0.15 | 各成员完整训练后融合 | `results/ensemble_result.txt` |
| FunkSVD | 16.6316 | factors=100, lr=0.005, reg=0.12, epochs=5 | 完整训练 5 epoch | `results/funk_svd_result.txt` |
| SVD++ | 16.9001 | factors=50, lr=0.002, reg=0.10, samples=20 | 完整训练上限 6 epoch，避免后期发散 | `results/svdpp_result.txt` |
| Baseline | 17.2272 | reg_user=10, reg_item=10, epochs=18 | 完整训练 18 epoch | `results/baseline_result.txt` |
| Item-KNN | 19.3244 | k=80, min_common=3 | 完整训练相似度缓存 | `results/knn_result.txt` |

**推荐提交文件**: `results/ensemble_result.txt`。它在当前验证集上优于单独的 FunkSVD。

## 代码结构

```text
任务一/
├── common.py              # 数据读取、划分、指标、结果格式化、结果校验
├── baseline.py            # global mean + user/item bias 基线模型
├── funk_svd.py            # 带偏置的 FunkSVD
├── svdpp.py               # 采样版 SVD++
├── knn_cf.py              # Item-KNN 协同过滤
├── ensemble.py            # 验证集权重搜索 + 完整训练融合预测
├── validate_results.py    # 检查预测文件覆盖、重复、范围和格式
├── data_exploration.py    # 数据统计脚本
├── data/
│   ├── train.txt
│   ├── test.txt
│   ├── DataFormatExplanation.txt
│   └── ResultForm.txt
└── results/
    ├── baseline_result.txt
    ├── funk_svd_result.txt
    ├── svdpp_result.txt
    ├── knn_result.txt
    ├── ensemble_result.txt
    └── rmse_summary.txt
```

## 运行方式

```bash
# 数据探索
python data_exploration.py

# 单模型训练 + 完整训练预测
python baseline.py
python funk_svd.py
python svdpp.py
python knn_cf.py

# 默认融合 Baseline + FunkSVD + SVD++
python ensemble.py

# 如需把较慢且较弱的 KNN 也纳入融合
python ensemble.py --include-knn

# 校验所有结果文件
python validate_results.py
```

## 实现要点

- `common.py` 统一数据协议，所有模型使用同一个解析、留出验证、RMSE/MAE、评分裁剪和结果输出逻辑。
- 网格搜索只用于验证集选参；最终结果文件统一来自完整 `train.txt` 重训后的模型。
- 冷启动预测统一回退到用户均值、物品均值、Baseline/全局均值，所有输出评分裁剪到 `10-100`。
- SVD++ 在验证集上第 8 轮最好，但完整训练后期容易发散，因此最终完整训练限制到 6 轮，保留更稳的预测文件。
- `validate_results.py` 会检查输出是否覆盖 `test.txt` 的全部 9,982 个 `(u, i)` 对，并确认评分范围合法。

## 待完成

- [ ] 实验报告：数据统计、算法描述、RMSE/训练时间/空间消耗、理论和实验分析。
- [ ] 最终提交：源码、可执行文件、实验报告、推荐预测结果 `ensemble_result.txt`。
- [ ] 截止时间：2026年6月25日 24:00。
