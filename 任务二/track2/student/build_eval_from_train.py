#!/usr/bin/env python3
"""Create eval_input.json/eval_answer.json from the visible training data."""

import argparse
import json
import random
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="student/data")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--holdout", type=int, default=2)
    parser.add_argument("--cold-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train = load_json(data_dir / "train.json")
    movies_info = load_json(data_dir / "movies_info.json")

    rng = random.Random(args.seed)
    users = list(train.get("users", []))
    rng.shuffle(users)
    users = users[: min(args.users, len(users))]

    eval_input = []
    eval_answer = []

    for user in users:
        user_id = user.get("user_id", "")
        history = list(user.get("history", []))
        if not user_id or len(history) <= args.holdout:
            continue

        holdout_items = []
        holdout_movie_ids = []
        for item in reversed(history):
            movie_id = item.get("movie_id", "")
            if not movie_id or movie_id in holdout_movie_ids:
                continue
            holdout_items.append(item)
            holdout_movie_ids.append(movie_id)
            if len(holdout_items) >= args.holdout:
                break
        holdout_items.reverse()
        context = [item for item in history if item.get("movie_id", "") not in holdout_movie_ids]

        for item in holdout_items:
            movie_id = item.get("movie_id", "")
            is_cold = rng.random() < args.cold_ratio
            eval_input.append({
                "user_id": user_id,
                "context_history": [] if is_cold else context,
                "target_movie": build_target_movie(item, movies_info),
                "is_cold_start": is_cold,
            })
            eval_answer.append({
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": item.get("rating", 3),
            })

    dump_json(data_dir / "eval_input.json", eval_input)
    dump_json(data_dir / "eval_answer.json", eval_answer)
    print(f"Wrote {len(eval_input)} eval samples to {data_dir}")


if __name__ == "__main__":
    main()
