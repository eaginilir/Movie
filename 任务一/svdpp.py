"""
SVD++ with sampled implicit-feedback updates.

The implementation is deterministic and maps IDs from the training split only,
so validation items unseen in the training split are evaluated through the same
cold-start fallback used for test-time predictions.
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
    memory_mb_for_arrays,
    set_random_seed,
    train_test_split,
    write_predictions,
)


class SVDpp:
    def __init__(
        self,
        n_factors: int = 50,
        lr: float = 0.005,
        reg: float = 0.05,
        n_epochs: int = 20,
        n_samples: int = 20,
        early_stopping: bool = True,
        patience: int = 3,
        lr_decay: float = 0.97,
        seed: int = DEFAULT_SEED,
        verbose: bool = True,
    ):
        self.n_factors = n_factors
        self.initial_lr = lr
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_samples = n_samples
        self.early_stopping = early_stopping
        self.patience = patience
        self.lr_decay = lr_decay
        self.seed = seed
        self.verbose = verbose

        self.global_mean = 0.0
        self.bu = None
        self.bi = None
        self.P = None
        self.Q = None
        self.Y = None
        self.sum_Y = None

        self.uid2idx = {}
        self.iid2idx = {}
        self.n_users = 0
        self.n_items = 0
        self.Ru_indices = []
        self.inv_sqrt_Ru = []
        self.stats = None
        self.best_epoch = n_epochs
        self.history = []
        self.train_time = 0.0

    def _build_mappings(self, ratings):
        users = sorted({u for u, _, _ in ratings})
        items = sorted({i for _, i, _ in ratings})
        self.uid2idx = {uid: idx for idx, uid in enumerate(users)}
        self.iid2idx = {iid: idx for idx, iid in enumerate(items)}
        self.n_users = len(users)
        self.n_items = len(items)

    def _init_params(self, ratings):
        self.lr = self.initial_lr
        self.stats = build_rating_stats(ratings)
        self.global_mean = self.stats.global_mean
        self._build_mappings(ratings)

        rng = np.random.RandomState(self.seed)
        scale = 0.05 / np.sqrt(self.n_factors)
        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)
        self.P = rng.normal(0, scale, (self.n_users, self.n_factors))
        self.Q = rng.normal(0, scale, (self.n_items, self.n_factors))
        self.Y = rng.normal(0, scale, (self.n_items, self.n_factors))

        ru_lists = defaultdict(list)
        for u, i, _ in ratings:
            ru_lists[self.uid2idx[u]].append(self.iid2idx[i])

        self.Ru_indices = []
        self.inv_sqrt_Ru = []
        for u_idx in range(self.n_users):
            arr = np.array(sorted(ru_lists[u_idx]), dtype=np.int32)
            self.Ru_indices.append(arr)
            self.inv_sqrt_Ru.append(1.0 / np.sqrt(len(arr)) if len(arr) else 0.0)
        self._rebuild_sum_Y()

    def _rebuild_sum_Y(self):
        self.sum_Y = np.zeros((self.n_users, self.n_factors))
        for u_idx, idxs in enumerate(self.Ru_indices):
            if len(idxs):
                self.sum_Y[u_idx] = self.Y[idxs].sum(axis=0)

    def fit(self, train_ratings, valid_ratings=None):
        t0 = time.time()
        train_ratings = list(train_ratings)
        valid_ratings = list(valid_ratings or [])
        self._init_params(train_ratings)

        train_idx = [(self.uid2idx[u], self.iid2idx[i], r) for u, i, r in train_ratings]
        rng = np.random.RandomState(self.seed + 31)
        best_rmse = float("inf")
        best_state = None
        stall = 0
        self.history = []
        self.best_epoch = self.n_epochs

        for epoch in range(1, self.n_epochs + 1):
            rng.shuffle(train_idx)

            for u, i, r in train_idx:
                inv_sqrt = self.inv_sqrt_Ru[u]
                implicit = inv_sqrt * self.sum_Y[u]
                user_vec = self.P[u] + implicit
                qi = self.Q[i]
                pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(qi, user_vec)
                err = np.clip(r - pred, -50.0, 50.0)

                old_qi = qi.copy()
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                self.P[u] += self.lr * np.clip(err * old_qi - self.reg * self.P[u], -10.0, 10.0)
                self.Q[i] += self.lr * np.clip(err * user_vec - self.reg * qi, -10.0, 10.0)

                ru = self.Ru_indices[u]
                if len(ru):
                    n_sample = min(self.n_samples, len(ru))
                    sampled = rng.choice(ru, size=n_sample, replace=False)
                    old_sampled = self.Y[sampled].copy()
                    grad_y = np.clip(err * inv_sqrt * old_qi - self.reg * old_sampled, -10.0, 10.0)
                    self.Y[sampled] += self.lr * grad_y
                    self.sum_Y[u] += self.Y[sampled].sum(axis=0) - old_sampled.sum(axis=0)

            self.lr *= self.lr_decay
            self._rebuild_sum_Y()

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

    def _predict_idx(self, u_idx, i_idx):
        user_vec = self.P[u_idx] + self.inv_sqrt_Ru[u_idx] * self.sum_Y[u_idx]
        return self.global_mean + self.bu[u_idx] + self.bi[i_idx] + np.dot(self.Q[i_idx], user_vec)

    def predict(self, uid, iid, clip=True):
        if self.stats is None:
            raise RuntimeError("model is not fitted")
        u_idx = self.uid2idx.get(uid)
        i_idx = self.iid2idx.get(iid)
        if u_idx is None or i_idx is None:
            pred = cold_start_fallback(uid, iid, self.stats)
        else:
            pred = self._predict_idx(u_idx, i_idx)
        return clip_score(pred) if clip else float(pred)

    def predict_batch(self, pairs):
        return [self.predict(u, i) for u, i in pairs]

    def compute_rmse(self, ratings):
        return compute_rmse(self, ratings, clip=False)

    def memory_mb(self):
        return memory_mb_for_arrays([self.bu, self.bi, self.P, self.Q, self.Y, self.sum_Y])

    def _save_state(self):
        return {
            "global_mean": self.global_mean,
            "bu": self.bu.copy(),
            "bi": self.bi.copy(),
            "P": self.P.copy(),
            "Q": self.Q.copy(),
            "Y": self.Y.copy(),
            "lr": self.lr,
            "best_epoch": self.best_epoch,
        }

    def _load_state(self, state):
        self.global_mean = state["global_mean"]
        self.bu = state["bu"]
        self.bi = state["bi"]
        self.P = state["P"]
        self.Q = state["Q"]
        self.Y = state["Y"]
        self.lr = state["lr"]
        self.best_epoch = state["best_epoch"]
        self._rebuild_sum_Y()


def run_experiment(seed: int = DEFAULT_SEED):
    print("=" * 50)
    print("SVD++ model training")
    print("=" * 50)

    set_random_seed(seed)
    ratings = load_ratings("train.txt", has_score=True)
    test_pairs = load_ratings("test.txt", has_score=False)
    train, valid = train_test_split(ratings, test_ratio=0.1, seed=seed)
    print(f"\nTrain ratings: {len(ratings)}")
    print(f"Validation ratings: {len(valid)}")
    print(f"Test pairs: {len(test_pairs)}")

    param_grid = [
        (50, 0.002, 0.10, 20),
        (50, 0.003, 0.10, 20),
        (50, 0.003, 0.12, 20),
    ]

    best_model = None
    best_rmse = float("inf")
    best_params = None

    print("\n" + "=" * 50)
    print("Grid search")
    print("=" * 50)
    for n_factors, lr, reg, n_samples in param_grid:
        print(f"\n-- n_factors={n_factors}, lr={lr}, reg={reg}, n_samples={n_samples} --")
        model = SVDpp(
            n_factors=n_factors,
            lr=lr,
            reg=reg,
            n_samples=n_samples,
            n_epochs=12,
            early_stopping=True,
            patience=3,
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
            best_params = (n_factors, lr, reg, n_samples, model.best_epoch)

    n_factors, lr, reg, n_samples, best_epoch = best_params
    print(
        f"\nBest params: n_factors={n_factors}, lr={lr}, reg={reg}, "
        f"n_samples={n_samples}, epochs={best_epoch}"
    )
    print(f"Validation RMSE: {best_rmse:.4f}")
    print(f"Validation model memory: {best_model.memory_mb():.2f} MB")

    final_epochs = min(best_epoch, 6)
    if final_epochs != best_epoch:
        print(f"\nSVD++ stability guard: final full-data epochs capped at {final_epochs} (best_epoch={best_epoch})")

    print("\n[Retraining on full train.txt...]")
    final_model = SVDpp(
        n_factors=n_factors,
        lr=lr,
        reg=reg,
        n_samples=n_samples,
        n_epochs=final_epochs,
        early_stopping=False,
        lr_decay=0.97,
        seed=seed,
        verbose=True,
    )
    final_model.fit(ratings)

    predictions = final_model.predict_batch(test_pairs)
    result_path = write_predictions("svdpp_result.txt", test_pairs, predictions)
    print(f"Result saved to {result_path}")
    return {
        "model": final_model,
        "best_rmse": best_rmse,
        "best_params": best_params + (final_epochs,),
        "result_path": result_path,
    }


if __name__ == "__main__":
    run_experiment()
