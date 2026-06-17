#!/usr/bin/env python3
"""Split all.json into train/test files under data/split."""

import argparse
import json
import random
import shutil
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def same_path(left, right):
    return left.resolve() == right.resolve()


def copy_or_dump_movies_info(source_path, output_path, movies):
    if same_path(source_path, output_path):
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.exists():
        shutil.copyfile(source_path, output_path)
    else:
        dump_json(output_path, movies)


def build_target_movie(item, movies_info):
    movie_id = item.get("movie_id", "")
    info = movies_info.get(movie_id, {})
    return {
        "movie_id": movie_id,
        "movie_name": item.get("movie_name") or info.get("name", ""),
        "director": item.get("director") or info.get("director", ""),
        "tags": item.get("tags") or info.get("tags", ""),
        "summary": info.get("summary", ""),
        "year": info.get("year", ""),
        "country": info.get("country", ""),
        "language": info.get("language", ""),
    }


def strip_comments(history):
    stripped = []
    for item in history:
        clean_item = dict(item)
        clean_item.pop("comment", None)
        stripped.append(clean_item)
    return stripped


def usable_history_count(history):
    return sum(1 for item in history if item.get("movie_id", ""))


def split_user_history(history, holdout_count):
    holdout_indices = []

    for idx in range(len(history) - 1, -1, -1):
        if not history[idx].get("movie_id", ""):
            continue

        holdout_indices.append(idx)
        if len(holdout_indices) >= holdout_count:
            break

    holdout_indices = set(holdout_indices)
    holdout_items = [
        item
        for idx, item in enumerate(history)
        if idx in holdout_indices
    ]
    holdout_comment_keys = {
        (
            item.get("movie_id", ""),
            item.get("rating", ""),
            item.get("comment", ""),
        )
        for item in holdout_items
    }
    train_history = [
        item
        for idx, item in enumerate(history)
        if idx not in holdout_indices
        and (
            item.get("movie_id", ""),
            item.get("rating", ""),
            item.get("comment", ""),
        ) not in holdout_comment_keys
    ]
    return train_history, holdout_items


def validate_args(args):
    if args.test_users <= 0:
        raise ValueError("--test-users must be > 0")
    if args.holdout <= 0:
        raise ValueError("--holdout must be > 0")
    if args.cold_users is not None and args.cold_users < 0:
        raise ValueError("--cold-users must be >= 0")
    if not 0 <= args.cold_ratio <= 1:
        raise ValueError("--cold-ratio must be between 0 and 1")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read data-dir/all.json and create a split directory containing "
            "movies_info.json, train.json, test_input.json, and test_answer.json."
        )
    )
    parser.add_argument("--data-dir", default="student/data")
    parser.add_argument("--input-file", default=None, help="Default: <data-dir>/all.json")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <data-dir>/split.",
    )
    parser.add_argument("--test-users", type=int, default=50)
    parser.add_argument("--holdout", type=int, default=2)
    parser.add_argument(
        "--cold-users",
        type=int,
        default=None,
        help="Number of test users to mark as cold-start. Default: round(test-users * cold-ratio).",
    )
    parser.add_argument("--cold-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    validate_args(args)

    data_dir = Path(args.data_dir)
    input_path = Path(args.input_file) if args.input_file else data_dir / "all.json"
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "split"
    all_data = load_json(input_path)
    movies_info_path = data_dir / "movies_info.json"
    movies_info = load_json(movies_info_path) if movies_info_path.exists() else all_data.get("movies", {})

    rng = random.Random(args.seed)
    users = list(all_data.get("users", []))
    eligible_user_ids = [
        user.get("user_id", "")
        for user in users
        if user.get("user_id", "") and usable_history_count(user.get("history", [])) > args.holdout
    ]
    if len(eligible_user_ids) < args.test_users:
        raise ValueError(
            f"Not enough users with more than {args.holdout} usable comments: "
            f"{len(eligible_user_ids)} available, {args.test_users} requested"
        )

    rng.shuffle(eligible_user_ids)
    test_user_ids = set(eligible_user_ids[:args.test_users])
    cold_user_count = args.cold_users
    if cold_user_count is None:
        cold_user_count = round(args.test_users * args.cold_ratio)
    cold_user_count = min(cold_user_count, args.test_users)
    cold_user_ids = set(eligible_user_ids[:cold_user_count])

    split_train_users = []
    test_input = []
    test_answer = []
    split_user_count = 0

    for user in users:
        user_id = user.get("user_id", "")
        history = list(user.get("history", []))
        if not user_id or user_id not in test_user_ids:
            split_train_users.append(user)
            continue

        context, holdout_items = split_user_history(history, args.holdout)
        split_user = dict(user)
        split_user["history"] = context
        split_train_users.append(split_user)
        split_user_count += 1

        for item in holdout_items:
            movie_id = item.get("movie_id", "")
            is_cold = user_id in cold_user_ids
            test_input.append({
                "user_id": user_id,
                "context_history": [] if is_cold else strip_comments(context),
                "target_movie": build_target_movie(item, movies_info),
                "is_cold_start": is_cold,
            })
            test_answer.append({
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": item.get("rating", 3),
            })

    split_train = dict(all_data)
    split_train["users"] = split_train_users

    copy_or_dump_movies_info(movies_info_path, output_dir / "movies_info.json", all_data.get("movies", {}))
    dump_json(output_dir / "train.json", split_train)
    dump_json(output_dir / "test_input.json", test_input)
    dump_json(output_dir / "test_answer.json", test_answer)

    print(f"Read {len(users)} users from {input_path}")
    print(f"Split users: {split_user_count}")
    print(f"Cold-start users: {len(cold_user_ids)}")
    print(f"Wrote split train users: {len(split_train_users)}")
    print(f"Wrote test samples: {len(test_input)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
