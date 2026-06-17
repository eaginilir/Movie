"""
FunkSVD — 基于矩阵分解的协同过滤（带偏置项）
在评分残差上学习:  r - μ = b_u + b_i + p_u · q_i
SGD 最小化正则化平方误差
"""
import numpy as np
import time
import os
from collections import defaultdict


class FunkSVD:
    """
    FunkSVD with global mean, user/item biases, and latent factors.

    预测公式: r̂ = μ + b_u + b_i + p_u · q_i
    在残差 r - μ 上学习，所有参数初始化为 0/小随机数，数值稳定。
    """

    def __init__(self, n_factors=50, lr=0.005, reg=0.02, n_epochs=20,
                 early_stopping=True, patience=3, lr_decay=0.95, verbose=True):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg              # L2 正则化系数
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.lr_decay = lr_decay    # 每轮学习率衰减因子
        self.verbose = verbose

        # 可学习参数
        self.global_mean = 0.0            # μ
        self.bu = defaultdict(float)      # 用户偏置
        self.bi = defaultdict(float)      # 物品偏置
        self.P = {}                       # 用户隐向量  p_u
        self.Q = {}                       # 物品隐向量  q_i

    def _init_params(self, ratings):
        """初始化参数 — 残差学习，参数初始化为 0/小随机数"""
        self.global_mean = sum(r for _, _, r in ratings) / len(ratings)

        users = set(r[0] for r in ratings)
        items = set(r[1] for r in ratings)

        scale = 0.1 / np.sqrt(self.n_factors)  # 小方差初始化
        for u in users:
            self.bu[u] = 0.0
            self.P[u] = np.random.normal(0, scale, self.n_factors)
        for i in items:
            self.bi[i] = 0.0
            self.Q[i] = np.random.normal(0, scale, self.n_factors)

    def fit(self, train_ratings, valid_ratings=None):
        """
        训练模型

        train_ratings: list of (user, item, rating)
        valid_ratings: list of (user, item, rating) for early stopping
        """
        t0 = time.time()
        self._init_params(train_ratings)

        best_rmse = float("inf")
        best_state = None
        stall = 0

        for epoch in range(1, self.n_epochs + 1):
            np.random.shuffle(train_ratings)

            for u, i, r in train_ratings:
                # 残差 = 真实评分 - 全局均值
                residual = r - self.global_mean
                # 误差 = 残差 - (b_u + b_i + p_u·q_i)
                pred_residual = self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])
                err = residual - pred_residual

                # 梯度裁剪，防止单步更新过大
                err = np.clip(err, -50, 50)

                pu = self.P[u]
                qi = self.Q[i]

                # SGD 更新（带梯度裁剪）
                grad_bu = err - self.reg * self.bu[u]
                grad_bi = err - self.reg * self.bi[i]
                grad_pu = err * qi - self.reg * pu
                grad_qi = err * pu - self.reg * qi

                # 裁剪隐因子梯度
                grad_pu = np.clip(grad_pu, -10, 10)
                grad_qi = np.clip(grad_qi, -10, 10)

                self.bu[u] += self.lr * grad_bu
                self.bi[i] += self.lr * grad_bi
                self.P[u] += self.lr * grad_pu
                self.Q[i] += self.lr * grad_qi

            # 学习率衰减
            self.lr *= self.lr_decay

            # 评估
            if valid_ratings:
                rmse = self.compute_rmse(valid_ratings)
                if self.verbose:
                    elapsed = time.time() - t0
                    print(f"  Epoch {epoch:>3}/{self.n_epochs}  "
                          f"valid RMSE: {rmse:.4f}  "
                          f"({elapsed:.1f}s)")

                if self.early_stopping:
                    if rmse < best_rmse - 1e-5:
                        best_rmse = rmse
                        best_state = self._save_state()
                        stall = 0
                    else:
                        stall += 1
                        if stall >= self.patience:
                            if self.verbose:
                                print(f"  -> early stopped at epoch {epoch}")
                            break
            elif self.verbose:
                elapsed = time.time() - t0
                train_rmse = self.compute_rmse(train_ratings)
                print(f"  Epoch {epoch:>3}/{self.n_epochs}  "
                      f"train RMSE: {train_rmse:.4f}  "
                      f"({elapsed:.1f}s)")

        # 恢复最优参数
        if best_state is not None:
            self._load_state(best_state)

        self.train_time = time.time() - t0
        return self

    def _predict_residual(self, u, i):
        """预测残差 b_u + b_i + p_u·q_i"""
        if u not in self.P or i not in self.Q:
            return self.bu.get(u, 0.0) + self.bi.get(i, 0.0)
        return self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])

    def _predict_one(self, u, i):
        """预测评分（无裁剪）"""
        return self.global_mean + self._predict_residual(u, i)

    def predict(self, u, i, clip=True):
        """预测单个 (u, i) 的评分"""
        pred = self._predict_one(u, i)
        if clip:
            pred = max(10, min(100, pred))
        return pred

    def predict_batch(self, pairs):
        """批量预测 [(u,i), ...] -> [score, ...]"""
        return [self.predict(u, i) for u, i in pairs]

    def compute_rmse(self, ratings):
        """计算 RMSE"""
        if not ratings:
            return float("inf")
        squared = 0.0
        for u, i, r in ratings:
            err = r - self._predict_one(u, i)
            squared += err * err
        return np.sqrt(squared / len(ratings))

    def _save_state(self):
        import copy
        return {
            "global_mean": self.global_mean,
            "bu": copy.deepcopy(dict(self.bu)),
            "bi": copy.deepcopy(dict(self.bi)),
            "P": copy.deepcopy(dict(self.P)),
            "Q": copy.deepcopy(dict(self.Q)),
        }

    def _load_state(self, state):
        self.global_mean = state["global_mean"]
        self.bu = defaultdict(float, state["bu"])
        self.bi = defaultdict(float, state["bi"])
        self.P = state["P"]
        self.Q = state["Q"]


# ── 工具函数 ──────────────────────────────────────────────

def load_ratings(filename, has_score=True):
    """从文件中加载评分数据"""
    filepath = os.path.join(os.path.dirname(__file__), "data", filename)
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        header = lines[i]; i += 1
        uid, cnt = header.split("|"); cnt = int(cnt)
        for _ in range(cnt):
            parts = lines[i].split(); i += 1
            if has_score:
                entries.append((int(uid), int(parts[0]), int(parts[1])))
            else:
                entries.append((int(uid), int(parts[0])))
    return entries


def train_test_split(ratings, test_ratio=0.2, seed=42):
    """按用户分层划分训练/验证集，保证每个用户在训练集中至少有 1 条"""
    rng = np.random.RandomState(seed)
    by_user = defaultdict(list)
    for u, i, r in ratings:
        by_user[u].append((u, i, r))
    train, valid = [], []
    for u, items in by_user.items():
        rng.shuffle(items)
        n_valid = max(1, int(len(items) * test_ratio))
        valid.extend(items[:n_valid])
        train.extend(items[n_valid:])
    return train, valid


def format_predictions(pairs, predictions):
    """按照 ResultForm.txt 格式输出"""
    by_user = defaultdict(list)
    for (u, i), s in zip(pairs, predictions):
        by_user[u].append((i, s))

    lines = []
    for u in sorted(by_user):
        items = by_user[u]
        lines.append(f"{u}|{len(items)}")
        for item, score in items:
            lines.append(f"{item}  {int(round(score)):>3}")
    return "\n".join(lines)


# ── 主训练脚本 ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("FunkSVD 模型训练")
    print("=" * 50)

    # 1. 加载数据
    ratings = load_ratings("train.txt", has_score=True)
    test_pairs = load_ratings("test.txt", has_score=False)
    print(f"\n训练评分数: {len(ratings)}")
    print(f"测试 (u,i) 对数: {len(test_pairs)}")

    # 2. 划分训练/验证集
    train, valid = train_test_split(ratings, test_ratio=0.1)
    print(f"留出验证: {len(valid)} 条")

    # 3. 网格搜索
    print("\n" + "=" * 50)
    print("网格搜索超参数")
    print("=" * 50)

    param_grid = [
        # (n_factors, lr, reg)
        (50,  0.005, 0.05),
        (50,  0.005, 0.10),
        (50,  0.003, 0.10),
        (100, 0.005, 0.05),
        (100, 0.005, 0.10),
        (100, 0.003, 0.10),
    ]

    best_model = None
    best_rmse = float("inf")
    best_params = None

    for n_factors, lr, reg in param_grid:
        print(f"\n-- n_factors={n_factors}, lr={lr}, reg={reg} --")
        model = FunkSVD(
            n_factors=n_factors,
            lr=lr,
            reg=reg,
            n_epochs=25,
            early_stopping=True,
            patience=4,
            lr_decay=0.97,
            verbose=True,
        )
        model.fit(train, valid)

        rmse = model.compute_rmse(valid)
        print(f"  -> valid RMSE: {rmse:.4f}  time: {model.train_time:.1f}s")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = (n_factors, lr, reg)

    # 4. 最优结果
    print(f"\n{'=' * 50}")
    print(f"最优参数: n_factors={best_params[0]}, lr={best_params[1]}, reg={best_params[2]}")
    print(f"验证集 RMSE: {best_rmse:.4f}")
    train_rmse = best_model.compute_rmse(train)
    print(f"训练集 RMSE: {train_rmse:.4f}")
    print(f"训练用时:    {best_model.train_time:.2f} 秒")

    # 5. 预测测试集
    print(f"\n[预测测试集...]")
    predictions = best_model.predict_batch(test_pairs)

    # 6. 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    result_text = format_predictions(test_pairs, predictions)
    result_path = os.path.join(output_dir, "funk_svd_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"结果已保存到 {result_path}")
