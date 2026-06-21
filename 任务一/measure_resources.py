"""
Measure final training time and model memory footprint for task 1.

This script avoids expensive grid search by reusing the chosen final
hyperparameters already reported in the project README/report.
"""
from __future__ import annotations

import os
import time

from baseline import BaselineRecommender
from common import DEFAULT_SEED, format_memory_mb, format_seconds, load_ratings, set_random_seed
from funk_svd import FunkSVD
from knn_cf import ItemKNN
from svdpp import SVDpp


def train_baseline(ratings):
    model = BaselineRecommender(
        reg_user=10.0,
        reg_item=10.0,
        n_epochs=18,
        early_stopping=False,
        verbose=False,
    )
    model.fit(ratings)
    return model


def train_funk_svd(ratings):
    model = FunkSVD(
        n_factors=100,
        lr=0.005,
        reg=0.12,
        n_epochs=5,
        early_stopping=False,
        lr_decay=0.97,
        seed=DEFAULT_SEED,
        verbose=False,
    )
    model.fit(ratings)
    return model


def train_svdpp(ratings):
    model = SVDpp(
        n_factors=50,
        lr=0.002,
        reg=0.10,
        n_samples=20,
        n_epochs=6,
        early_stopping=False,
        lr_decay=0.97,
        seed=DEFAULT_SEED,
        verbose=False,
    )
    model.fit(ratings)
    return model


def train_knn(ratings):
    model = ItemKNN(k=80, min_common=3, verbose=False)
    model.fit(ratings)
    return model


def measure_models():
    set_random_seed(DEFAULT_SEED)
    ratings = load_ratings("train.txt", has_score=True)

    trainers = [
        ("Baseline", train_baseline),
        ("FunkSVD", train_funk_svd),
        ("SVD++", train_svdpp),
        ("Item-KNN", train_knn),
    ]

    records = []
    trained_models = {}
    for name, trainer in trainers:
        t0 = time.time()
        model = trainer(ratings)
        wall_time = time.time() - t0
        memory_mb = model.memory_mb() if hasattr(model, "memory_mb") else 0.0
        record = {
            "name": name,
            "train_time": getattr(model, "train_time", wall_time),
            "wall_time": wall_time,
            "memory_mb": memory_mb,
        }
        records.append(record)
        trained_models[name] = model

    ensemble_weights = {"Baseline": 0.20, "FunkSVD": 0.65, "SVD++": 0.15}
    ensemble_time = sum(record["train_time"] for record in records if record["name"] in ensemble_weights)
    ensemble_wall_time = sum(record["wall_time"] for record in records if record["name"] in ensemble_weights)
    ensemble_memory = sum(record["memory_mb"] for record in records if record["name"] in ensemble_weights)
    records.insert(
        0,
        {
            "name": "Weighted Ensemble",
            "train_time": ensemble_time,
            "wall_time": ensemble_wall_time,
            "memory_mb": ensemble_memory,
        },
    )
    return records


def render_table(records):
    header = f"{'Model':<18}{'Train Time':>14}{'Wall Time':>14}{'Memory':>14}"
    sep = "-" * len(header)
    lines = [header, sep]
    for record in records:
        lines.append(
            f"{record['name']:<18}"
            f"{format_seconds(record['train_time']):>14}"
            f"{format_seconds(record['wall_time']):>14}"
            f"{format_memory_mb(record['memory_mb']):>14}"
        )
    return "\n".join(lines)


def write_summary(records):
    summary = render_table(records)
    print(summary)
    out_path = os.path.join(os.path.dirname(__file__), "results", "resource_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\nResource summary saved to {out_path}")


if __name__ == "__main__":
    write_summary(measure_models())
