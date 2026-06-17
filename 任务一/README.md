# 任务一：电影评分预测

## 项目概述

预测 `test.txt` 中 (用户, 物品) 对的评分分数。实现了三种推荐算法并进行对比分析。

**数据集**: 598 用户 × 9,077 物品 × 90,854 条评分，稀疏度 98.47%

---

## 已完成工作

### 1. 数据探索与统计 ✅

| 指标 | 数值 |
|------|------|
| 训练集用户数 | 598 |
| 训练集物品数 | 9,077 |
| 训练集评分数 | 90,854 |
| 测试集 (u,i) 对数 | 9,982 |
| 全部物品数 | 9,724 |
| 稀疏度 | 98.47% |
| 评分均值 | 69.88 |
| 评分范围 | 10 ~ 100 |
| 评论频率最低的物品 | 仅评 5 次（<5个）的占 65.6% |

**相关文件**: `data_exploration.py`, `stats_output.txt`

### 2. 算法实现与对比 ✅

| 算法 | Valid RMSE | 最优参数 | 训练时间 | 预测文件 |
|------|-----------|---------|---------|---------|
| **FunkSVD** 🥇 | **16.63** | k=100, lr=0.005, reg=0.1 | 37s | `results/funk_svd_result.txt` |
| SVD++ 🥈 | 16.91 | k=50, lr=0.005, reg=0.1 | 198s | `results/svdpp_result.txt` |
| Item-KNN 🥉 | 19.32 | k=80 | 207s | `results/knn_result.txt` |

#### 算法简介

- **FunkSVD**: 矩阵分解，学习用户/物品隐向量 + 偏置项。最简洁，效果最好
- **SVD++**: 在 FunkSVD 基础上加入隐式反馈（采样 SGD 加速）。参数多易过拟合
- **Item-KNN**: 基于物品相似度的 K 近邻。可解释性强，但在稀疏数据上效果较差

#### 排名分析

矩阵分解 > 矩阵分解+隐式反馈 > 邻域方法，原因：
- 98.47% 稀疏度下，物品间共现太少，KNN 相似度不可靠
- SVD++ 的隐式反馈向量增加了 45 万参数，稀疏数据下过拟合
- FunkSVD 隐因子泛化能力最强，是这个数据集上的最优选择

---

## 文件结构

```
任务一/
├── README.md                    ← 本文件
├── data/
│   ├── train.txt                # 训练数据
│   ├── test.txt                 # 测试数据
│   ├── DataFormatExplanation.txt # 数据格式说明
│   └── ResultForm.txt           # 结果提交模板
├── results/
│   ├── rmse_summary.txt         # 所有 RMSE 网格搜索记录
│   ├── funk_svd_result.txt      # FunkSVD 预测结果
│   ├── svdpp_result.txt         # SVD++ 预测结果
│   └── knn_result.txt           # KNN-CF 预测结果
├── data_exploration.py          # 数据探索脚本
├── stats_output.txt             # 数据统计输出
├── funk_svd.py                  # FunkSVD 模型
├── svdpp.py                     # SVD++ 模型
└── knn_cf.py                    # Item-KNN 模型
```

---

## 运行方式

```bash
# 数据探索
python data_exploration.py

# 各模型训练 + 预测（均包含网格搜索）
python funk_svd.py
python svdpp.py
python knn_cf.py
```

---

## 待完成

- [ ] 实验报告（统计信息 + 算法描述 + 实验结果 + 理论/实验分析）
- [ ] 最终提交: 源码 + 可执行文件 + 报告 + 结果 打包发送至 bigdatacomputing@163.com
- [ ] 截止时间: 2026年6月25日 24:00
