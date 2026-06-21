"""
Regularized baseline recommender.

The model predicts:
    r_hat = global_mean + user_bias + item_bias

It is fast, deterministic, useful as an experiment baseline, and provides a
stable fallback for cold-start cases and ensembles.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from common import (
    DEFAULT_SEED,
    build_rating_stats,
    clip_score,
    compute_mae,
    compute_rmse,
    load_ratings,
    memory_mb_for_mapping,
    set_random_seed,
    train_test_split,
    write_predictions,
)


class BaselineRecommender:
    def __init__(
        self,
        reg_user: float = 15.0,
        reg_item: float = 10.0,
        n_epochs: int = 20,
        early_stopping: bool = True,
        patience: int = 3,
        verbose: bool = True,
    ):
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose

        self.global_mean = 0.0
        self.bu = defaultdict(float)
        self.bi = defaultdict(float)
        self.stats = None
        self.best_epoch = n_epochs
        self.history = []
        self.train_time = 0.0

    def fit(self, ratings, valid_ratings=None):
        t0 = time.time()
        self.stats = build_rating_stats(ratings)
        self.global_mean = self.stats.global_mean
        self.bu = defaultdict(float)
        self.bi = defaultdict(float)

        by_user = defaultdict(list)
        by_item = defaultdict(list)
        for u, i, r in ratings:
            by_user[u].append((i, r))
            by_item[i].append((u, r))

        best_rmse = float("inf")
        best_state = None
        stall = 0
        self.history = []

        for epoch in range(1, self.n_epochs + 1):
            for u in sorted(by_user):
                vals = by_user[u]
                num = sum(r - self.global_mean - self.bi[i] for i, r in vals)
                self.bu[u] = num / (self.reg_user + len(vals))

            for i in sorted(by_item):
                vals = by_item[i]
                num = sum(r - self.global_mean - self.bu[u] for u, r in vals)
                self.bi[i] = num / (self.reg_item + len(vals))

            if valid_ratings:
                rmse = compute_rmse(self, valid_ratings, clip=False)
                mae = compute_mae(self, valid_ratings, clip=False)
                self.history.append((epoch, rmse, mae))
                if self.verbose:
                    elapsed = time.time() - t0
                    print(
                        f"  Epoch {epoch:>2}/{self.n_epochs}  "
                        f"valid RMSE: {rmse:.4f}  MAE: {mae:.4f}  ({elapsed:.1f}s)"
                    )

                if self.early_stopping:
                    if rmse < best_rmse - 1e-5:
                        best_rmse = rmse
                        self.best_epoch = epoch
                        best_state = self._save_state()
                        stall = 0
                    else:
                        stall += 1
                        if stall >= self.patience:
                            if self.verbose:
                                print(f"  -> early stopped at epoch {epoch}")
                            break
            elif self.verbose:
                train_rmse = compute_rmse(self, ratings, clip=False)
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:>2}/{self.n_epochs}  train RMSE: {train_rmse:.4f}  ({elapsed:.1f}s)")

        if best_state is not None:
            self._load_state(best_state)

        self.train_time = time.time() - t0
        return self

    def predict(self, uid, iid, clip=True):
        if self.stats is None:
            raise RuntimeError("model is not fitted")

        if uid in self.stats.user_mean and iid not in self.stats.item_mean:
            pred = self.stats.user_mean[uid]
        elif iid in self.stats.item_mean and uid not in self.stats.user_mean:
            pred = self.stats.item_mean[iid]
        else:
            pred = self.global_mean + self.bu.get(uid, 0.0) + self.bi.get(iid, 0.0)

        return clip_score(pred) if clip else float(pred)

    def predict_batch(self, pairs):
        return [self.predict(u, i) for u, i in pairs]

    def compute_rmse(self, ratings):
        return compute_rmse(self, ratings, clip=False)

    def memory_mb(self):
        return memory_mb_for_mapping(self.bu) + memory_mb_for_mapping(self.bi)

    def _save_state(self):
        return {
            "global_mean": self.global_mean,
            "bu": dict(self.bu),
            "bi": dict(self.bi),
            "best_epoch": self.best_epoch,
        }

    def _load_state(self, state):
        self.global_mean = state["global_mean"]
        self.bu = defaultdict(float, state["bu"])
        self.bi = defaultdict(float, state["bi"])
        self.best_epoch = state["best_epoch"]


def run_experiment(seed: int = DEFAULT_SEED):
    print("=" * 50)
    print("Baseline model training")
    print("=" * 50)

    set_random_seed(seed)
    ratings = load_ratings("train.txt", has_score=True)
    test_pairs = load_ratings("test.txt", has_score=False)
    train, valid = train_test_split(ratings, test_ratio=0.1, seed=seed)
    print(f"\nTrain ratings: {len(ratings)}")
    print(f"Validation ratings: {len(valid)}")
    print(f"Test pairs: {len(test_pairs)}")

    param_grid = [
        (10.0, 10.0),
        (15.0, 10.0),
        (20.0, 10.0),
        (20.0, 15.0),
        (30.0, 20.0),
    ]

    best_model = None
    best_rmse = float("inf")
    best_params = None

    print("\n" + "=" * 50)
    print("Grid search")
    print("=" * 50)
    for reg_user, reg_item in param_grid:
        print(f"\n-- reg_user={reg_user:g}, reg_item={reg_item:g} --")
        model = BaselineRecommender(
            reg_user=reg_user,
            reg_item=reg_item,
            n_epochs=25,
            early_stopping=True,
            patience=4,
            verbose=True,
        )
        model.fit(train, valid)
        rmse = model.compute_rmse(valid)
        print(f"  -> valid RMSE: {rmse:.4f}  best_epoch: {model.best_epoch}  time: {model.train_time:.1f}s")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = (reg_user, reg_item, model.best_epoch)

    reg_user, reg_item, best_epoch = best_params
    print(f"\nBest params: reg_user={reg_user:g}, reg_item={reg_item:g}, epochs={best_epoch}")
    print(f"Validation RMSE: {best_rmse:.4f}")

    print("\n[Retraining on full train.txt...]")
    final_model = BaselineRecommender(
        reg_user=reg_user,
        reg_item=reg_item,
        n_epochs=best_epoch,
        early_stopping=False,
        verbose=True,
    )
    final_model.fit(ratings)

    predictions = final_model.predict_batch(test_pairs)
    result_path = write_predictions("baseline_result.txt", test_pairs, predictions)
    print(f"Final train time: {final_model.train_time:.1f}s")
    print(f"Final model memory: {final_model.memory_mb():.2f} MB")
    print(f"Result saved to {result_path}")
    return {
        "model": final_model,
        "best_rmse": best_rmse,
        "best_params": best_params,
        "train_time": final_model.train_time,
        "memory_mb": final_model.memory_mb(),
        "result_path": result_path,
    }


if __name__ == "__main__":
    run_experiment()
