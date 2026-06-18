"""
Shared utilities for task 1 movie rating prediction.

The project data uses grouped records:
    <user id>|<number of items>
    <item id> [score]

This module keeps parsing, splitting, metrics, result formatting, and
validation in one place so every model follows the same experiment protocol.
"""
from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DEFAULT_SEED = 42
MIN_SCORE = 10
MAX_SCORE = 100

Rating = Tuple[int, int, int]
Pair = Tuple[int, int]


def set_random_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python and NumPy RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def clip_score(score: float, min_score: int = MIN_SCORE, max_score: int = MAX_SCORE) -> float:
    return float(max(min_score, min(max_score, score)))


def rounded_score(score: float) -> int:
    return int(round(clip_score(score)))


def load_ratings(filename: str, has_score: bool = True, data_dir: str = DATA_DIR):
    """Load train/test data while preserving the original grouped order."""
    filepath = os.path.join(data_dir, filename)
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    idx = 0
    while idx < len(lines):
        header = lines[idx]
        idx += 1
        if "|" not in header:
            raise ValueError(f"Bad header in {filename}: {header!r}")
        uid_text, count_text = header.split("|", 1)
        uid = int(uid_text)
        count = int(count_text)

        for _ in range(count):
            if idx >= len(lines):
                raise ValueError(f"Unexpected EOF after header {header!r} in {filename}")
            parts = lines[idx].split()
            idx += 1
            if has_score:
                if len(parts) != 2:
                    raise ValueError(f"Expected item and score in {filename}: {parts!r}")
                entries.append((uid, int(parts[0]), int(parts[1])))
            else:
                if len(parts) != 1:
                    raise ValueError(f"Expected item only in {filename}: {parts!r}")
                entries.append((uid, int(parts[0])))
    return entries


def train_test_split(
    ratings: Sequence[Rating],
    test_ratio: float = 0.1,
    seed: int = DEFAULT_SEED,
    min_train_per_user: int = 1,
) -> Tuple[List[Rating], List[Rating]]:
    """Stratified split by user, keeping at least one train rating per user."""
    rng = np.random.RandomState(seed)
    by_user: Dict[int, List[Rating]] = defaultdict(list)
    for rating in ratings:
        by_user[rating[0]].append(rating)

    train: List[Rating] = []
    valid: List[Rating] = []
    for uid in sorted(by_user):
        items = list(by_user[uid])
        rng.shuffle(items)
        if len(items) <= min_train_per_user:
            train.extend(items)
            continue
        n_valid = max(1, int(len(items) * test_ratio))
        n_valid = min(n_valid, len(items) - min_train_per_user)
        valid.extend(items[:n_valid])
        train.extend(items[n_valid:])
    return train, valid


def rmse_from_predictions(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if not actual:
        return float("inf")
    err = np.asarray(actual, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
    return float(np.sqrt(np.mean(err * err)))


def mae_from_predictions(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if not actual:
        return float("inf")
    err = np.asarray(actual, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
    return float(np.mean(np.abs(err)))


def compute_rmse(model, ratings: Sequence[Rating], clip: bool = False) -> float:
    if not ratings:
        return float("inf")
    actual = [r for _, _, r in ratings]
    predicted = [model.predict(u, i, clip=clip) for u, i, _ in ratings]
    return rmse_from_predictions(actual, predicted)


def compute_mae(model, ratings: Sequence[Rating], clip: bool = False) -> float:
    if not ratings:
        return float("inf")
    actual = [r for _, _, r in ratings]
    predicted = [model.predict(u, i, clip=clip) for u, i, _ in ratings]
    return mae_from_predictions(actual, predicted)


@dataclass
class RatingStats:
    global_mean: float
    user_mean: Dict[int, float]
    item_mean: Dict[int, float]
    user_count: Dict[int, int]
    item_count: Dict[int, int]


def build_rating_stats(ratings: Sequence[Rating]) -> RatingStats:
    if not ratings:
        raise ValueError("ratings must not be empty")

    user_values: Dict[int, List[int]] = defaultdict(list)
    item_values: Dict[int, List[int]] = defaultdict(list)
    all_scores = []
    for u, i, r in ratings:
        user_values[u].append(r)
        item_values[i].append(r)
        all_scores.append(r)

    return RatingStats(
        global_mean=float(np.mean(all_scores)),
        user_mean={u: float(np.mean(values)) for u, values in user_values.items()},
        item_mean={i: float(np.mean(values)) for i, values in item_values.items()},
        user_count={u: len(values) for u, values in user_values.items()},
        item_count={i: len(values) for i, values in item_values.items()},
    )


def cold_start_fallback(
    uid: int,
    iid: int,
    stats: RatingStats,
    baseline_model=None,
) -> float:
    """
    Layered fallback for missing model parameters.

    Priority follows the task plan: user mean, item mean, baseline model, global
    mean. A known user or item receives the corresponding empirical mean.
    """
    if uid in stats.user_mean and iid not in stats.item_mean:
        return stats.user_mean[uid]
    if iid in stats.item_mean and uid not in stats.user_mean:
        return stats.item_mean[iid]
    if baseline_model is not None:
        return baseline_model.predict(uid, iid, clip=False)
    if uid in stats.user_mean:
        return stats.user_mean[uid]
    if iid in stats.item_mean:
        return stats.item_mean[iid]
    return stats.global_mean


def format_predictions(pairs: Sequence[Pair], predictions: Sequence[float]) -> str:
    """Format predictions in ResultForm.txt style, preserving test order."""
    if len(pairs) != len(predictions):
        raise ValueError(f"pairs/predictions length mismatch: {len(pairs)} != {len(predictions)}")

    grouped: Dict[int, List[Tuple[int, float]]] = {}
    for (uid, iid), score in zip(pairs, predictions):
        grouped.setdefault(uid, []).append((iid, score))

    lines = []
    for uid, items in grouped.items():
        lines.append(f"{uid}|{len(items)}")
        for iid, score in items:
            lines.append(f"{iid}  {rounded_score(score):>3}")
    return "\n".join(lines) + "\n"


def write_predictions(filename: str, pairs: Sequence[Pair], predictions: Sequence[float]) -> str:
    ensure_results_dir()
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_predictions(pairs, predictions))
    return path


def parse_result_file(path: str) -> List[Rating]:
    """Parse a result file as (user, item, predicted_score)."""
    entries: List[Rating] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    idx = 0
    while idx < len(lines):
        header = lines[idx]
        idx += 1
        if "|" not in header:
            raise ValueError(f"Bad result header: {header!r}")
        uid_text, count_text = header.split("|", 1)
        uid = int(uid_text)
        count = int(count_text)
        for _ in range(count):
            if idx >= len(lines):
                raise ValueError(f"Unexpected EOF after result header {header!r}")
            parts = lines[idx].split()
            idx += 1
            if len(parts) != 2:
                raise ValueError(f"Bad result row: {parts!r}")
            entries.append((uid, int(parts[0]), int(parts[1])))
    return entries


def validate_result_file(result_path: str, test_pairs: Optional[Sequence[Pair]] = None) -> Dict[str, object]:
    """Validate coverage, order, duplicate count, and score range."""
    if test_pairs is None:
        test_pairs = load_ratings("test.txt", has_score=False)

    predictions = parse_result_file(result_path)
    result_pairs = [(u, i) for u, i, _ in predictions]
    scores = [r for _, _, r in predictions]

    missing = set(test_pairs) - set(result_pairs)
    extra = set(result_pairs) - set(test_pairs)
    duplicates = len(result_pairs) - len(set(result_pairs))
    order_matches = list(test_pairs) == result_pairs
    out_of_range = [score for score in scores if score < MIN_SCORE or score > MAX_SCORE]

    return {
        "path": result_path,
        "expected_pairs": len(test_pairs),
        "actual_pairs": len(result_pairs),
        "missing_pairs": len(missing),
        "extra_pairs": len(extra),
        "duplicates": duplicates,
        "order_matches": order_matches,
        "out_of_range": len(out_of_range),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "ok": (
            len(result_pairs) == len(test_pairs)
            and not missing
            and not extra
            and duplicates == 0
            and not out_of_range
        ),
    }


def memory_mb_for_arrays(arrays: Iterable[np.ndarray]) -> float:
    return sum(arr.nbytes for arr in arrays if arr is not None) / (1024 * 1024)


def append_summary(section: str) -> None:
    ensure_results_dir()
    path = os.path.join(RESULTS_DIR, "rmse_summary.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(section.rstrip() + "\n\n")


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes}m{rest:.0f}s"
