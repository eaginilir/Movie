"""
FunkSVD with user/item biases.

The experiment flow is:
1. Split train.txt into train/validation with a fixed seed.
2. Use validation RMSE only to select hyperparameters and best epoch.
3. Retrain the selected configuration on the full train.txt.
4. Predict test.txt and write results/funk_svd_result.txt.
"""
from __future__ import annotations

import copy
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
    memory_mb_for_arrays,
    set_random_seed,
    train_test_split,
    write_predictions,
)


class FunkSVD:
    def __init__(
        self,
        n_factors: int = 50,
        lr: float = 0.005,
        reg: float = 0.02,
        n_epochs: int = 20,
        early_stopping: bool = True,
        patience: int = 3,
        lr_decay: float = 0.95,
        seed: int = DEFAULT_SEED,
        verbose: bool = True,
    ):
        self.n_factors = n_factors
        self.initial_lr = lr
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.lr_decay = lr_decay
        self.seed = seed
        self.verbose = verbose

        self.global_mean = 0.0
        self.bu = defaultdict(float)
        self.bi = defaultdict(float)
        self.P = {}
        self.Q = {}
        self.stats = None
        self.best_epoch = n_epochs
        self.history = []
        self.train_time = 0.0

    def _init_params(self, ratings):
        self.lr = self.initial_lr
        self.stats = build_rating_stats(ratings)
        self.global_mean = self.stats.global_mean

        users = sorted({u for u, _, _ in ratings})
        items = sorted({i for _, i, _ in ratings})
        rng = np.random.RandomState(self.seed)
        scale = 0.05 / np.sqrt(self.n_factors)

        self.bu = defaultdict(float)
        self.bi = defaultdict(float)
        self.P = {}
        self.Q = {}
        for u in users:
            self.bu[u] = 0.0
            self.P[u] = rng.normal(0, scale, self.n_factors)
        for i in items:
            self.bi[i] = 0.0
            self.Q[i] = rng.normal(0, scale, self.n_factors)

    def fit(self, train_ratings, valid_ratings=None):
        t0 = time.time()
        train_ratings = list(train_ratings)
        valid_ratings = list(valid_ratings or [])
        self._init_params(train_ratings)

        rng = np.random.RandomState(self.seed + 17)
        best_rmse = float("inf")
        best_state = None
        stall = 0
        self.history = []
        self.best_epoch = self.n_epochs

        for epoch in range(1, self.n_epochs + 1):
            rng.shuffle(train_ratings)

            for u, i, r in train_ratings:
                pu = self.P[u]
                qi = self.Q[i]
                pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(pu, qi)
                err = np.clip(r - pred, -50.0, 50.0)

                grad_bu = err - self.reg * self.bu[u]
                grad_bi = err - self.reg * self.bi[i]
                grad_pu = np.clip(err * qi - self.reg * pu, -10.0, 10.0)
                grad_qi = np.clip(err * pu - self.reg * qi, -10.0, 10.0)

                self.bu[u] += self.lr * grad_bu
                self.bi[i] += self.lr * grad_bi
                self.P[u] += self.lr * grad_pu
                self.Q[i] += self.lr * grad_qi

            self.lr *= self.lr_decay

            if valid_ratings:
                rmse = compute_rmse(self, valid_ratings, clip=False)
                mae = compute_mae(self, valid_ratings, clip=False)
                self.history.append((epoch, rmse, mae))
                if self.verbose:
                    elapsed = time.time() - t0
                    print(
                        f"  Epoch {epoch:>3}/{self.n_epochs}  "
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
                train_rmse = compute_rmse(self, train_ratings, clip=False)
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:>3}/{self.n_epochs}  train RMSE: {train_rmse:.4f}  ({elapsed:.1f}s)")

        if best_state is not None:
            self._load_state(best_state)

        self.train_time = time.time() - t0
        return self

    def _predict_one(self, uid, iid):
        if self.stats is None:
            raise RuntimeError("model is not fitted")
        if uid not in self.P or iid not in self.Q:
            return cold_start_fallback(uid, iid, self.stats)
        return self.global_mean + self.bu[uid] + self.bi[iid] + np.dot(self.P[uid], self.Q[iid])

    def predict(self, uid, iid, clip=True):
        pred = self._predict_one(uid, iid)
        return clip_score(pred) if clip else float(pred)

    def predict_batch(self, pairs):
        return [self.predict(u, i) for u, i in pairs]

    def compute_rmse(self, ratings):
        return compute_rmse(self, ratings, clip=False)

    def memory_mb(self):
        arrays = list(self.P.values()) + list(self.Q.values())
        return memory_mb_for_arrays(arrays)

    def _save_state(self):
        return {
            "global_mean": self.global_mean,
            "bu": copy.deepcopy(dict(self.bu)),
            "bi": copy.deepcopy(dict(self.bi)),
            "P": copy.deepcopy(self.P),
            "Q": copy.deepcopy(self.Q),
            "lr": self.lr,
            "best_epoch": self.best_epoch,
        }

    def _load_state(self, state):
        self.global_mean = state["global_mean"]
        self.bu = defaultdict(float, state["bu"])
        self.bi = defaultdict(float, state["bi"])
        self.P = state["P"]
        self.Q = state["Q"]
        self.lr = state["lr"]
        self.best_epoch = state["best_epoch"]


def run_experiment(seed: int = DEFAULT_SEED):
    print("=" * 50)
    print("FunkSVD model training")
    print("=" * 50)

    set_random_seed(seed)
    ratings = load_ratings("train.txt", has_score=True)
    test_pairs = load_ratings("test.txt", has_score=False)
    train, valid = train_test_split(ratings, test_ratio=0.1, seed=seed)
    print(f"\nTrain ratings: {len(ratings)}")
    print(f"Validation ratings: {len(valid)}")
    print(f"Test pairs: {len(test_pairs)}")

    param_grid = [
        (50, 0.005, 0.08),
        (50, 0.005, 0.10),
        (80, 0.005, 0.10),
        (100, 0.005, 0.10),
        (100, 0.004, 0.10),
        (100, 0.005, 0.12),
    ]

    best_model = None
    best_rmse = float("inf")
    best_params = None

    print("\n" + "=" * 50)
    print("Grid search")
    print("=" * 50)
    for n_factors, lr, reg in param_grid:
        print(f"\n-- n_factors={n_factors}, lr={lr}, reg={reg} --")
        model = FunkSVD(
            n_factors=n_factors,
            lr=lr,
            reg=reg,
            n_epochs=30,
            early_stopping=True,
            patience=4,
            lr_decay=0.97,
            seed=seed,
            verbose=True,
        )
        model.fit(train, valid)
        rmse = model.compute_rmse(valid)
        print(
            f"  -> valid RMSE: {rmse:.4f}  best_epoch: {model.best_epoch}  "
            f"time: {model.train_time:.1f}s"
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = (n_factors, lr, reg, model.best_epoch)

    n_factors, lr, reg, best_epoch = best_params
    print(f"\nBest params: n_factors={n_factors}, lr={lr}, reg={reg}, epochs={best_epoch}")
    print(f"Validation RMSE: {best_rmse:.4f}")
    print(f"Validation train RMSE: {best_model.compute_rmse(train):.4f}")
    print(f"Validation model memory: {best_model.memory_mb():.2f} MB")

    print("\n[Retraining on full train.txt...]")
    final_model = FunkSVD(
        n_factors=n_factors,
        lr=lr,
        reg=reg,
        n_epochs=best_epoch,
        early_stopping=False,
        lr_decay=0.97,
        seed=seed,
        verbose=True,
    )
    final_model.fit(ratings)

    predictions = final_model.predict_batch(test_pairs)
    result_path = write_predictions("funk_svd_result.txt", test_pairs, predictions)
    print(f"Result saved to {result_path}")
    return {
        "model": final_model,
        "best_rmse": best_rmse,
        "best_params": best_params,
        "result_path": result_path,
    }


if __name__ == "__main__":
    run_experiment()
