"""
SVD++ — 带隐式反馈的矩阵分解 (采样版)
预测: r̂ = μ + b_u + b_i + q_i^T · (p_u + |R(u)|^{-1/2} · Σ_{j∈R(u)} y_j)
SGD 时每次只采样 n_samples 个 y_j 更新，速度接近 FunkSVD
"""
import numpy as np
import time
import os
from collections import defaultdict

from funk_svd import load_ratings, train_test_split, format_predictions


class SVDpp:
    """
    SVD++ with implicit feedback — 采样 SGD 优化。

    预测时使用完整 sum_Y，SGD 更新时只随机采样 n_samples 个 y_j。
    每 epoch 结束后全量重建 sum_Y 修正漂移。
    """

    def __init__(self, n_factors=50, lr=0.005, reg=0.05, n_epochs=20,
                 n_samples=20, early_stopping=True, patience=3,
                 lr_decay=0.97, verbose=True):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_samples = n_samples       # 每次 SGD 采样的 y_j 个数
        self.early_stopping = early_stopping
        self.patience = patience
        self.lr_decay = lr_decay
        self.verbose = verbose

        # 参数
        self.global_mean = 0.0
        self.bu = None          # (n_users,)
        self.bi = None          # (n_items,)
        self.P = None           # (n_users, k)
        self.Q = None           # (n_items, k)
        self.Y = None           # (n_items, k)

        # ID 映射
        self.uid2idx = {}
        self.iid2idx = {}
        self.n_users = 0
        self.n_items = 0

        # 用户缓存
        self.Ru_indices = []    # list of 1D int np arrays
        self.inv_sqrt_Ru = []   # |R(u)|^{-1/2}

    def _build_mappings(self, ratings):
        users = sorted(set(r[0] for r in ratings))
        items = sorted(set(r[1] for r in ratings))
        self.n_users = len(users)
        self.n_items = len(items)
        self.uid2idx = {uid: i for i, uid in enumerate(users)}
        self.iid2idx = {iid: i for i, iid in enumerate(items)}

    def _init_params(self, ratings, train_only=None):
        if train_only is None:
            train_only = ratings
        self._build_mappings(ratings)

        self.global_mean = sum(r for _, _, r in train_only) / len(train_only)
        k = self.n_factors
        scale = 0.1 / np.sqrt(k)

        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)
        self.P  = np.random.normal(0, scale, (self.n_users, k))
        self.Q  = np.random.normal(0, scale, (self.n_items, k))
        self.Y  = np.random.normal(0, scale, (self.n_items, k))

        ru_lists = defaultdict(list)
        for u, i, _ in train_only:
            ru_lists[self.uid2idx[u]].append(self.iid2idx[i])

        self.Ru_indices = [None] * self.n_users
        self.inv_sqrt_Ru = [0.0] * self.n_users
        for u_idx in range(self.n_users):
            arr = np.array(ru_lists[u_idx], dtype=np.int32)
            self.Ru_indices[u_idx] = arr
            self.inv_sqrt_Ru[u_idx] = 1.0 / np.sqrt(len(arr))

        self._rebuild_sum_Y()

    def _rebuild_sum_Y(self):
        """全量重建 sum_Y[u] = Σ Y[j] for j in R(u)"""
        self.sum_Y = np.zeros((self.n_users, self.n_factors))
        for u_idx in range(self.n_users):
            idxs = self.Ru_indices[u_idx]
            if len(idxs) > 0:
                self.sum_Y[u_idx] = self.Y[idxs].sum(axis=0)

    def fit(self, train_ratings, valid_ratings=None):
        t0 = time.time()

        all_ratings = train_ratings + (valid_ratings or [])
        self._init_params(all_ratings, train_only=train_ratings)

        train_idx = [(self.uid2idx[u], self.iid2idx[i], r) for u, i, r in train_ratings]
        valid_idx = None
        if valid_ratings:
            valid_idx = [(self.uid2idx[u], self.iid2idx[i], r) for u, i, r in valid_ratings]

        best_rmse = float("inf")
        best_state = None
        stall = 0

        for epoch in range(1, self.n_epochs + 1):
            np.random.shuffle(train_idx)

            for u, i, r in train_idx:
                # ---- 前向 ----
                inv_sqrt = self.inv_sqrt_Ru[u]
                s_u = self.P[u] + inv_sqrt * self.sum_Y[u]
                qi = self.Q[i]
                pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(qi, s_u)
                err = r - pred
                err = np.clip(err, -50, 50)

                # ---- 更新 b_u, b_i, p_u, q_i ----
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                grad_pu = np.clip(err * qi - self.reg * self.P[u], -10, 10)
                self.P[u] += self.lr * grad_pu

                grad_qi = np.clip(err * s_u - self.reg * qi, -10, 10)
                self.Q[i] += self.lr * grad_qi

                # ---- 采样 y_j 更新（核心改动）----
                ru = self.Ru_indices[u]
                if len(ru) == 0:
                    continue

                n_sample = min(self.n_samples, len(ru))
                sampled = np.random.choice(ru, size=n_sample, replace=False)

                old_sampled = self.Y[sampled].copy()
                grad_Y = err * inv_sqrt * qi - self.reg * old_sampled
                grad_Y = np.clip(grad_Y, -10, 10)
                self.Y[sampled] += self.lr * grad_Y

                delta_sum = self.Y[sampled].sum(axis=0) - old_sampled.sum(axis=0)
                self.sum_Y[u] += delta_sum

            # ---- epoch end ----
            self.lr *= self.lr_decay
            self._rebuild_sum_Y()  # 每个 epoch 全量修正一次，消除采样累积误差

            if valid_idx:
                rmse = self._compute_rmse_idx(valid_idx)
                if self.verbose:
                    elapsed = time.time() - t0
                    print(f"  Epoch {epoch:>3}/{self.n_epochs}  "
                          f"valid RMSE: {rmse:.4f}  ({elapsed:.1f}s)")

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
                train_rmse = self._compute_rmse_idx(train_idx)
                print(f"  Epoch {epoch:>3}/{self.n_epochs}  "
                      f"train RMSE: {train_rmse:.4f}  ({elapsed:.1f}s)")

        if best_state is not None:
            self._load_state(best_state)
        self.train_time = time.time() - t0
        return self

    def _predict_one(self, u_idx, i_idx):
        if u_idx >= self.n_users or i_idx >= self.n_items:
            bu = self.bu[u_idx] if u_idx < self.n_users else 0.0
            bi = self.bi[i_idx] if i_idx < self.n_items else 0.0
            return self.global_mean + bu + bi
        s_u = self.P[u_idx] + self.inv_sqrt_Ru[u_idx] * self.sum_Y[u_idx]
        return self.global_mean + self.bu[u_idx] + self.bi[i_idx] + np.dot(self.Q[i_idx], s_u)

    def predict(self, uid, iid, clip=True):
        u = self.uid2idx.get(uid, self.n_users)
        i = self.iid2idx.get(iid, self.n_items)
        pred = self._predict_one(u, i)
        if clip:
            pred = max(10, min(100, pred))
        return pred

    def predict_batch(self, pairs):
        return [self.predict(u, i) for u, i in pairs]

    def _compute_rmse_idx(self, ratings_idx):
        if not ratings_idx:
            return float("inf")
        sq = 0.0
        for u, i, r in ratings_idx:
            err = r - self._predict_one(u, i)
            sq += err * err
        return np.sqrt(sq / len(ratings_idx))

    def compute_rmse(self, ratings):
        if not ratings:
            return float("inf")
        idx = [(self.uid2idx[u], self.iid2idx[i], r) for u, i, r in ratings]
        return self._compute_rmse_idx(idx)

    def _save_state(self):
        return {
            "global_mean": self.global_mean,
            "bu": self.bu.copy(), "bi": self.bi.copy(),
            "P": self.P.copy(), "Q": self.Q.copy(), "Y": self.Y.copy(),
        }

    def _load_state(self, state):
        self.global_mean = state["global_mean"]
        self.bu = state["bu"]; self.bi = state["bi"]
        self.P = state["P"]; self.Q = state["Q"]; self.Y = state["Y"]
        self._rebuild_sum_Y()


# ── 主训练脚本 ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("SVD++ 模型训练 (NumPy 向量化)")
    print("=" * 50)

    ratings = load_ratings("train.txt", has_score=True)
    test_pairs = load_ratings("test.txt", has_score=False)
    print(f"\n训练评分数: {len(ratings)}")
    print(f"测试 (u,i) 对数: {len(test_pairs)}")

    train, valid = train_test_split(ratings, test_ratio=0.1)
    print(f"留出验证: {len(valid)} 条")

    print("\n" + "=" * 50)
    print("网格搜索超参数")
    print("=" * 50)

    param_grid = [
        (50,  0.005, 0.05),
        (50,  0.005, 0.10),
        (100, 0.005, 0.05),
        (100, 0.005, 0.10),
    ]

    best_model = None
    best_rmse = float("inf")
    best_params = None

    for n_factors, lr, reg in param_grid:
        print(f"\n-- n_factors={n_factors}, lr={lr}, reg={reg} --")
        model = SVDpp(
            n_factors=n_factors, lr=lr, reg=reg, n_samples=20,
            n_epochs=25, early_stopping=True, patience=4,
            lr_decay=0.97, verbose=True,
        )
        model.fit(train, valid)
        rmse = model.compute_rmse(valid)
        print(f"  -> valid RMSE: {rmse:.4f}  time: {model.train_time:.1f}s")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = (n_factors, lr, reg)

    print(f"\n{'=' * 50}")
    print(f"最优参数: n_factors={best_params[0]}, lr={best_params[1]}, reg={best_params[2]}")
    print(f"验证集 RMSE: {best_rmse:.4f}")
    print(f"训练用时:    {best_model.train_time:.2f} 秒")

    print(f"\n[预测测试集...]")
    predictions = best_model.predict_batch(test_pairs)

    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    result_text = format_predictions(test_pairs, predictions)
    result_path = os.path.join(output_dir, "svdpp_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"结果已保存到 {result_path}")
