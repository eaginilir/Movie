"""
Weighted ensemble for task 1.

By default the ensemble trains Baseline, FunkSVD, and SVD++ on the fixed
validation split, searches convex weights, retrains selected models on the full
train.txt, and writes results/ensemble_result.txt. Item-KNN can be included
with --include-knn, but it is slower and mainly useful as a comparison model.
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

from baseline import BaselineRecommender
from common import (
    DEFAULT_SEED,
    MAX_SCORE,
    MIN_SCORE,
    load_ratings,
    rmse_from_predictions,
    set_random_seed,
    train_test_split,
    write_predictions,
)
from funk_svd import FunkSVD
from svdpp import SVDpp


def build_model_specs(seed: int, include_knn: bool = False):
    specs = [
        {
            "name": "baseline",
            "factory": lambda epochs=None, verbose=True: BaselineRecommender(
                reg_user=10.0,
                reg_item=10.0,
                n_epochs=epochs or 25,
                early_stopping=epochs is None,
                patience=4,
                verbose=verbose,
            ),
        },
        {
            "name": "funk_svd",
            "factory": lambda epochs=None, verbose=True: FunkSVD(
                n_factors=100,
                lr=0.005,
                reg=0.12,
                n_epochs=epochs or 30,
                early_stopping=epochs is None,
                patience=4,
                lr_decay=0.97,
                seed=seed,
                verbose=verbose,
            ),
        },
        {
            "name": "svdpp",
            "max_final_epochs": 6,
            "factory": lambda epochs=None, verbose=True: SVDpp(
                n_factors=50,
                lr=0.002,
                reg=0.10,
                n_samples=20,
                n_epochs=epochs or 12,
                early_stopping=epochs is None,
                patience=3,
                lr_decay=0.97,
                seed=seed,
                verbose=verbose,
            ),
        },
    ]

    if include_knn:
        from knn_cf import ItemKNN

        specs.append(
            {
                "name": "knn",
                "factory": lambda epochs=None, verbose=True: ItemKNN(
                    k=80,
                    min_common=3,
                    verbose=verbose,
                ),
            }
        )
    return specs


def simplex_weights(n_models: int, step: float):
    units = int(round(1.0 / step))
    if units <= 0:
        raise ValueError("step must be positive")

    def rec(prefix, remaining, slots):
        if slots == 1:
            yield prefix + [remaining]
            return
        for value in range(remaining + 1):
            yield from rec(prefix + [value], remaining - value, slots - 1)

    for counts in rec([], units, n_models):
        yield np.array(counts, dtype=np.float64) / units


def search_weights(valid_predictions, actual, step: float = 0.05):
    matrix = np.asarray(valid_predictions, dtype=np.float64)
    best_rmse = float("inf")
    best_weights = None

    for weights in simplex_weights(matrix.shape[0], step):
        pred = np.dot(weights, matrix)
        pred = np.clip(pred, MIN_SCORE, MAX_SCORE)
        rmse = rmse_from_predictions(actual, pred)
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = weights

    return best_weights, best_rmse


def predict_raw(model, pairs):
    return [model.predict(u, i, clip=False) for u, i in pairs]


def run_experiment(seed: int = DEFAULT_SEED, include_knn: bool = False, weight_step: float = 0.05):
    print("=" * 50)
    print("Weighted ensemble training")
    print("=" * 50)

    set_random_seed(seed)
    t0 = time.time()
    ratings = load_ratings("train.txt", has_score=True)
    test_pairs = load_ratings("test.txt", has_score=False)
    train, valid = train_test_split(ratings, test_ratio=0.1, seed=seed)
    actual = [r for _, _, r in valid]
    specs = build_model_specs(seed, include_knn=include_knn)

    valid_predictions = []
    trained_info = []

    for spec in specs:
        print(f"\n[{spec['name']}] validation training")
        model = spec["factory"](epochs=None, verbose=True)
        model.fit(train, valid)
        preds = predict_raw(model, [(u, i) for u, i, _ in valid])
        clipped_preds = np.clip(preds, MIN_SCORE, MAX_SCORE)
        rmse = rmse_from_predictions(actual, clipped_preds)
        best_epoch = getattr(model, "best_epoch", None)
        print(f"  validation RMSE for ensemble input: {rmse:.4f}")
        if best_epoch is not None:
            print(f"  selected epoch: {best_epoch}")
        valid_predictions.append(preds)
        trained_info.append({"name": spec["name"], "spec": spec, "rmse": rmse, "best_epoch": best_epoch})

    weights, ensemble_rmse = search_weights(valid_predictions, actual, step=weight_step)
    print("\n" + "=" * 50)
    print("Best ensemble weights")
    print("=" * 50)
    for info, weight in zip(trained_info, weights):
        print(f"  {info['name']:<10} weight={weight:.2f}  individual RMSE={info['rmse']:.4f}")
    print(f"Ensemble validation RMSE: {ensemble_rmse:.4f}")

    print("\n[Retraining ensemble members on full train.txt...]")
    final_predictions = []
    for info, weight in zip(trained_info, weights):
        if weight <= 0:
            continue
        spec = info["spec"]
        best_epoch = info["best_epoch"]
        if best_epoch is not None and spec.get("max_final_epochs") is not None:
            best_epoch = min(best_epoch, spec["max_final_epochs"])
        print(f"\n[{info['name']}] final training, weight={weight:.2f}")
        model = spec["factory"](epochs=best_epoch, verbose=True)
        model.fit(ratings)
        final_predictions.append((weight, predict_raw(model, test_pairs)))

    if not final_predictions:
        raise RuntimeError("all ensemble weights are zero")

    combined = np.zeros(len(test_pairs), dtype=np.float64)
    weight_total = 0.0
    for weight, preds in final_predictions:
        combined += weight * np.asarray(preds, dtype=np.float64)
        weight_total += weight
    combined /= weight_total
    combined = np.clip(combined, MIN_SCORE, MAX_SCORE)

    result_path = write_predictions("ensemble_result.txt", test_pairs, combined)
    print(f"\nResult saved to {result_path}")
    print(f"Total ensemble time: {time.time() - t0:.1f}s")
    return {
        "weights": {info["name"]: float(weight) for info, weight in zip(trained_info, weights)},
        "best_rmse": ensemble_rmse,
        "result_path": result_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train weighted ensemble for task 1.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--include-knn", action="store_true", help="Include the slower Item-KNN model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(seed=args.seed, include_knn=args.include_knn, weight_step=args.weight_step)
