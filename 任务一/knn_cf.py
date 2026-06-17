"""
KNN-CF — 基于物品的 K 近邻协同过滤 (Item-based CF)
使用修正余弦相似度，均值中心化，加权平均预测
"""
import numpy as np
import time
import os
from collections import defaultdict

from funk_svd import load_ratings, train_test_split, format_predictions


class ItemKNN:
    """
    Item-based KNN Collaborative Filtering.

    预测: r̂_ui = r̄_u + Σ_{j∈N(i,k)} sim(i,j)·(r_uj - r̄_u) / Σ|sim(i,j)|
    其中 N(i,k) 是物品 i 的 k 个最相似物品中用户 u 评价过的
    """

    def __init__(self, k=30, min_common=3, verbose=True):
        self.k = k                     # 近邻数
        self.min_common = min_common   # 最少共同评分用户数
        self.verbose = verbose

        # 数据
        self.user_ratings = {}         # uid -> {iid: rating}
        self.item_users = {}           # iid -> {uid: rating}
        self.user_mean = {}            # uid -> mean rating

        # 预计算
        self.item_sims = {}            # iid -> [(j, sim), ...]  top-k neighbors

    def fit(self, ratings):
        """构建数据结构并计算物品相似度"""
        t0 = time.time()

        # 1. 构建 user_ratings 和 item_users
        for u, i, r in ratings:
            if u not in self.user_ratings:
                self.user_ratings[u] = {}
            self.user_ratings[u][i] = r
            if i not in self.item_users:
                self.item_users[i] = {}
            self.item_users[i][u] = r

        # 2. 计算用户均值
        for u, items in self.user_ratings.items():
            self.user_mean[u] = np.mean(list(items.values()))

        # 3. 计算物品-物品相似度
        items = list(self.item_users.keys())
        if self.verbose:
            print(f"  计算 {len(items)} 个物品的相似度...")

        for idx, i in enumerate(items):
            if self.verbose and (idx + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(f"    {idx+1}/{len(items)} items ({elapsed:.1f}s)")

            sims = []
            users_i = self.item_users[i]

            # 只计算有共同评分用户的物品对
            candidates = defaultdict(int)
            for u in users_i:
                for j in self.user_ratings.get(u, {}):
                    if j != i:
                        candidates[j] += 1

            mean_r_i = np.mean(list(users_i.values()))

            for j, common_count in candidates.items():
                if common_count < self.min_common:
                    continue

                # 修正余弦相似度
                users_j = self.item_users[j]
                common_users = set(users_i.keys()) & set(users_j.keys())

                if len(common_users) < self.min_common:
                    continue

                # 中心化评分
                vec_i = []
                vec_j = []
                for u in common_users:
                    vec_i.append(users_i[u] - self.user_mean[u])
                    vec_j.append(users_j[u] - self.user_mean[u])

                vec_i = np.array(vec_i)
                vec_j = np.array(vec_j)

                dot = np.dot(vec_i, vec_j)
                norm = np.linalg.norm(vec_i) * np.linalg.norm(vec_j)
                if norm > 0:
                    sim = dot / norm
                    sims.append((j, sim))

            # 保留 top-k
            sims.sort(key=lambda x: x[1], reverse=True)
            self.item_sims[i] = sims[:self.k]

        self.train_time = time.time() - t0
        if self.verbose:
            print(f"  完成! 用时 {self.train_time:.1f}s")
        return self

    def predict(self, uid, iid, clip=True):
        """预测单个 (u, i) 评分"""
        user_mean = self.user_mean.get(uid, self.global_mean)

        # 冷启动物品
        if iid not in self.item_sims or not self.item_sims[iid]:
            # 用用户历史均值
            if uid in self.user_ratings:
                return np.mean(list(self.user_ratings[uid].values()))
            return user_mean

        # 收集有效邻居
        user_rated = self.user_ratings.get(uid, {})
        weighted_sum = 0.0
        weight_total = 0.0

        for j, sim in self.item_sims[iid]:
            if j in user_rated and sim > 0:  # 只用正相似度
                r_uj = user_rated[j]
                j_mean = self.user_mean.get(uid, user_mean)  # same user
                weighted_sum += sim * (r_uj - user_mean)
                weight_total += abs(sim)

        if weight_total > 0:
            pred = user_mean + weighted_sum / weight_total
        else:
            pred = user_mean

        if clip:
            pred = max(10, min(100, pred))
        return pred

    def predict_batch(self, pairs):
        return [self.predict(u, i) for u, i in pairs]

    def compute_rmse(self, ratings):
        if not ratings:
            return float("inf")
        sq = 0.0
        for u, i, r in ratings:
            err = r - self.predict(u, i, clip=False)
            sq += err * err
        return np.sqrt(sq / len(ratings))


# ── 主训练脚本 ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Item-KNN 模型训练")
    print("=" * 50)

    ratings = load_ratings("train.txt", has_score=True)
    test_pairs = load_ratings("test.txt", has_score=False)
    print(f"\n训练评分数: {len(ratings)}")

    # 全局均值（冷启动回退用）
    global_mean = sum(r for _, _, r in ratings) / len(ratings)

    train, valid = train_test_split(ratings, test_ratio=0.1)
    print(f"留出验证: {len(valid)} 条")

    print("\n" + "=" * 50)
    print("网格搜索 k 值")
    print("=" * 50)

    k_values = [10, 20, 30, 50, 80]

    best_model = None
    best_rmse = float("inf")
    best_k = None

    for k in k_values:
        print(f"\n-- k={k} --")
        model = ItemKNN(k=k, min_common=3, verbose=True)
        model.global_mean = global_mean
        model.fit(train)

        rmse = model.compute_rmse(valid)
        print(f"  -> valid RMSE: {rmse:.4f}  time: {model.train_time:.1f}s")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_k = k

    print(f"\n{'=' * 50}")
    print(f"最优 k: {best_k}")
    print(f"验证集 RMSE: {best_rmse:.4f}")

    # 预测测试集
    print(f"\n[预测测试集...]")
    predictions = best_model.predict_batch(test_pairs)

    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    result_text = format_predictions(test_pairs, predictions)
    result_path = os.path.join(output_dir, "knn_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"结果已保存到 {result_path}")
