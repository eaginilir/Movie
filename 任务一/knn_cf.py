"""
Item-based KNN collaborative filtering with adjusted cosine similarity.

This model is kept as an interpretable neighborhood baseline. Matrix
factorization is expected to perform better on this sparse dataset.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from common import (
    DEFAULT_SEED,
    build_rating_stats,
    clip_score,
    cold_start_fallback,
    compute_mae,
    compute_rmse,
    load_ratings,
    set_random_seed,
    train_test_split,
    write_predictions,
)


class ItemKNN:
    def __init__(self, k: int = 30, min_common: int = 3, verbose: bool = True):
        self.k = k
        self.min_common = min_common
        self.verbose = verbose

        self.user_ratings = {}
        self.item_users = {}
        self.user_mean = {}
        self.item_mean = {}
        self.centered_item_users = {}
        self.item_sims = {}
        self.stats = None
        self.global_mean = 0.0
        self.train_time = 0.0

    def fit(self, ratings):
        t0 = time.time()
        ratings = list(ratings)
        self.stats = build_rating_stats(ratings)
        self.global_mean = self.stats.global_mean

        user_ratings = defaultdict(dict)
        item_users = defaultdict(dict)
        for u, i, r in ratings:
            user_ratings[u][i] = r
            item_users[i][u] = r

        self.user_ratings = dict(user_ratings)
        self.item_users = dict(item_users)
        self.user_mean = self.stats.user_mean
        self.item_mean = self.stats.item_mean
        self.centered_item_users = {
            i: {u: r - self.user_mean[u] for u, r in users.items()}
            for i, users in self.item_users.items()
        }

        items = sorted(self.item_users)
        if self.verbose:
            print(f"  Computing similarities for {len(items)} items...")

        for idx, item_i in enumerate(items, start=1):
            if self.verbose and idx % 1000 == 0:
                elapsed = time.time() - t0
                print(f"    {idx}/{len(items)} items ({elapsed:.1f}s)")

            users_i = self.centered_item_users[item_i]
            candidates = defaultdict(int)
            for u in users_i:
                for item_j in self.user_ratings[u]:
                    if item_j != item_i:
                        candidates[item_j] += 1

            sims = []
            for item_j, common_count in candidates.items():
                if common_count < self.min_common:
                    continue
                sim = self._adjusted_cosine(item_i, item_j)
                if sim > 0:
                    sims.append((item_j, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            self.item_sims[item_i] = sims[: self.k]

        self.train_time = time.time() - t0
        if self.verbose:
            print(f"  Done in {self.train_time:.1f}s")
        return self

    def _adjusted_cosine(self, item_i, item_j):
        users_i = self.centered_item_users[item_i]
        users_j = self.centered_item_users[item_j]
        if len(users_i) > len(users_j):
            users_i, users_j = users_j, users_i

        dot = 0.0
        norm_i = 0.0
        norm_j = 0.0
        common = 0
        for u, val_i in users_i.items():
            val_j = users_j.get(u)
            if val_j is None:
                continue
            common += 1
            dot += val_i * val_j
            norm_i += val_i * val_i
            norm_j += val_j * val_j

        if common < self.min_common or norm_i <= 0.0 or norm_j <= 0.0:
            return 0.0
        return dot / np.sqrt(norm_i * norm_j)

    def predict(self, uid, iid, clip=True):
        if self.stats is None:
            raise RuntimeError("model is not fitted")

        if uid not in self.user_ratings or iid not in self.item_users:
            pred = cold_start_fallback(uid, iid, self.stats)
            return clip_score(pred) if clip else float(pred)

        neighbors = self.item_sims.get(iid, [])
        if not neighbors:
            pred = self.user_mean.get(uid, self.global_mean)
            return clip_score(pred) if clip else float(pred)

        user_rated = self.user_ratings.get(uid, {})
        user_mean = self.user_mean.get(uid, self.global_mean)
        weighted_sum = 0.0
        weight_total = 0.0

        for item_j, sim in neighbors:
            if item_j in user_rated:
                weighted_sum += sim * (user_rated[item_j] - user_mean)
                weight_total += abs(sim)

        if weight_total > 0:
            pred = user_mean + weighted_sum / weight_total
        else:
            pred = user_mean

        return clip_score(pred) if clip else float(pred)

    def predict_batch(self, pairs):
        return [self.predict(u, i) for u, i in pairs]

    def compute_rmse(self, ratings):
        return compute_rmse(self, ratings, clip=False)

    def memory_mb(self):
        pair_count = sum(len(v) for v in self.item_sims.values())
        # two Python numbers per pair; this is only a coarse report estimate.
        return pair_count * 16 / (1024 * 1024)


def run_experiment(seed: int = DEFAULT_SEED):
    print("=" * 50)
    print("Item-KNN model training")
    print("=" * 50)

    set_random_seed(seed)
    ratings = load_ratings("train.txt", has_score=True)
    test_pairs = load_ratings("test.txt", has_score=False)
    train, valid = train_test_split(ratings, test_ratio=0.1, seed=seed)
    print(f"\nTrain ratings: {len(ratings)}")
    print(f"Validation ratings: {len(valid)}")
    print(f"Test pairs: {len(test_pairs)}")

    k_values = [30, 50, 80]
    best_model = None
    best_rmse = float("inf")
    best_k = None

    print("\n" + "=" * 50)
    print("Grid search")
    print("=" * 50)
    for k in k_values:
        print(f"\n-- k={k} --")
        model = ItemKNN(k=k, min_common=3, verbose=True)
        model.fit(train)
        rmse = model.compute_rmse(valid)
        mae = compute_mae(model, valid, clip=False)
        print(f"  -> valid RMSE: {rmse:.4f}  MAE: {mae:.4f}  time: {model.train_time:.1f}s")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_k = k

    print(f"\nBest k: {best_k}")
    print(f"Validation RMSE: {best_rmse:.4f}")
    print(f"Similarity cache estimate: {best_model.memory_mb():.2f} MB")

    print("\n[Retraining on full train.txt...]")
    final_model = ItemKNN(k=best_k, min_common=3, verbose=True)
    final_model.fit(ratings)

    predictions = final_model.predict_batch(test_pairs)
    result_path = write_predictions("knn_result.txt", test_pairs, predictions)
    print(f"Result saved to {result_path}")
    return {
        "model": final_model,
        "best_rmse": best_rmse,
        "best_params": (best_k,),
        "result_path": result_path,
    }


if __name__ == "__main__":
    run_experiment()
