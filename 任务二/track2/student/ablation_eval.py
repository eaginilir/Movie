#!/usr/bin/env python3
"""Offline ablations for the Track 2 prompt.

This script does not call GLM.  It evaluates the deterministic rating anchor
used by claude98prompt.py under several component removals and parameter
settings.  The goal is fast, repeatable evidence for the report:

  1. full warm leave-one-out over all 2,000 ratings;
  2. full cold leave-one-out with empty user history;
  3. official-like multi-seed splits: 50 users, 2 hold-outs/user, 25% cold.

The released test/eval GLM score can differ because the cold branch allows GLM
to adjust the public-score reference by at most one star.  The warm branch is a
deterministic printer, so these local warm ablations are directly meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Variant:
    name: str
    group: str
    description: str
    warm_mode: str = "neighbor"
    cold_mode: str = "bucket"
    use_tags: bool = True
    use_director: bool = True
    use_public: bool = True
    use_range_clamp: bool = True
    public_weight: float = 0.27
    director_weight: float = 2.0
    confidence_divisor: float = 6.0
    confidence_cap: float = 0.6
    rounding: str = "calibrated"


@dataclass(frozen=True)
class Sample:
    user_id: str
    movie_id: str
    history: list
    target: dict
    truth: int
    is_cold: bool = False


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def text_value(value, default="", max_len=120):
    if value is None:
        return default
    value = str(value).strip()
    if not value:
        return default
    return value[:max_len]


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def clamp_rating(value):
    try:
        value = int(round(float(value)))
    except Exception:
        value = 3
    return max(1, min(5, value))


def calibrated_round(value):
    value = to_float(value, 3.0)
    if value >= 4.55:
        return 5
    if value >= 3.55:
        return 4
    if value >= 2.55:
        return 3
    if value >= 1.70:
        return 2
    return 1


def normal_round(value):
    return clamp_rating(value)


def round_rating(value, mode):
    if mode == "normal":
        return normal_round(value)
    return calibrated_round(value)


def split_tags(tag_text, limit=16):
    tag_text = text_value(tag_text, "", 600)
    for ch in ["[", "]", "{", "}", "'", '"', "，", "/", "|", ";", "；", ":", "：", "(", ")", "（", "）"]:
        tag_text = tag_text.replace(ch, ",")
    tags = []
    for piece in tag_text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if piece.replace(".", "", 1).isdigit():
            continue
        if len(piece) > 20:
            piece = piece[:20]
        if piece not in tags:
            tags.append(piece)
        if len(tags) >= limit:
            break
    return tags


def movie_info_for(item, movies_info):
    movie_id = item.get("movie_id", "")
    if movie_id:
        return movies_info.get(movie_id, {})
    return {}


def tags_list(item, info, limit=14):
    return split_tags(item.get("tags") or info.get("tags"), limit)


def clean_director(item, info):
    raw = text_value(item.get("director") or info.get("director"), "", 200)
    if not raw:
        return ""
    if "{" in raw and ":" in raw:
        names = []
        for p in raw.replace("{", "").replace("}", "").split(","):
            val = p.split(":", 1)[1] if ":" in p else p
            val = val.strip().strip("'").strip('"').strip()
            if val and not val.replace(".", "", 1).isdigit():
                names.append(val)
            if len(names) >= 2:
                break
        if names:
            return "/".join(names)
    if "(" in raw:
        raw = raw.split("(")[0].strip()
    return raw[:20]


def parse_public(item, info):
    text = text_value(item.get("rating") or info.get("rating"), "", 220)
    if not text:
        return 0.0
    if "%" in text:
        nums = []
        buf = ""
        for ch in text:
            if ("0" <= ch <= "9") or ch == ".":
                buf += ch
            elif ch == "%":
                if buf:
                    nums.append(to_float(buf, 0.0))
                buf = ""
            else:
                buf = ""
        if nums:
            return (sum(nums) / len(nums)) / 10.0
        return 0.0
    buf = ""
    for ch in text:
        if ("0" <= ch <= "9") or ch == ".":
            buf += ch
        elif buf:
            break
    score = to_float(buf, 0.0)
    if score > 10 and score <= 100:
        score /= 10.0
    return score


def overlap_count(left, right):
    right_set = set(right)
    return sum(1 for item in left if item in right_set)


def user_stats(history):
    ratings = [clamp_rating(item.get("rating", 3)) for item in history]
    if not ratings:
        return {"count": 0, "avg": 0.0, "min": 0, "max": 0}
    return {
        "count": len(ratings),
        "avg": sum(ratings) / len(ratings),
        "min": min(ratings),
        "max": max(ratings),
    }


def cold_reference(public_score):
    if public_score <= 0:
        return 3.7
    if public_score < 6.7:
        return 3.0
    if public_score < 7.4:
        return 3.5
    if public_score < 8.0:
        return 3.9
    if public_score < 8.6:
        return 4.1
    if public_score < 9.0:
        return 4.3
    return 4.6


def warm_anchor(history, target, target_info, movies_info, stats, public_score, variant):
    if variant.warm_mode == "constant_4":
        return 4.0
    if variant.warm_mode == "user_mean":
        anchor = stats["avg"]
    elif variant.warm_mode == "public_only":
        anchor = public_score / 2.0 if public_score > 0 else 3.7
    else:
        target_tags = tags_list(target, target_info, 14) if variant.use_tags else []
        target_director = clean_director(target, target_info) if variant.use_director else ""
        wsum = 0.0
        vsum = 0.0
        for item in history:
            info = movie_info_for(item, movies_info)
            weight = overlap_count(target_tags, tags_list(item, info, 14)) if variant.use_tags else 0
            item_director = clean_director(item, info) if variant.use_director else ""
            if target_director and item_director and target_director == item_director:
                weight += variant.director_weight
            if weight > 0:
                rating = clamp_rating(item.get("rating", 3))
                wsum += weight
                vsum += weight * rating
        if wsum > 0:
            confidence = min(wsum / variant.confidence_divisor, variant.confidence_cap)
            anchor = (vsum / wsum) * confidence + stats["avg"] * (1 - confidence)
        else:
            anchor = stats["avg"]

    if variant.use_public and public_score > 0 and variant.warm_mode != "public_only":
        anchor = anchor * (1 - variant.public_weight) + (public_score / 2.0) * variant.public_weight

    if variant.use_range_clamp and stats["count"] > 0:
        if stats["max"] <= 4 and anchor > 4.5:
            anchor = 4.4
        if stats["min"] >= 3 and anchor < 2.5:
            anchor = 2.6
    return anchor


def cold_anchor(public_score, variant):
    if variant.cold_mode == "constant_4":
        return 4.0
    if variant.cold_mode == "global_mean":
        return 3.65
    if variant.cold_mode == "public_half":
        return public_score / 2.0 if public_score > 0 else 3.7
    return cold_reference(public_score)


def predict(sample, movies_info, variant):
    target_info = movie_info_for(sample.target, movies_info)
    public_score = parse_public(sample.target, target_info)
    stats = user_stats(sample.history)
    if not sample.history:
        anchor = cold_anchor(public_score, variant)
    else:
        anchor = warm_anchor(sample.history, sample.target, target_info, movies_info, stats, public_score, variant)
    return round_rating(anchor, variant.rounding)


def build_target_movie(item, movies_info):
    movie_id = item.get("movie_id", "")
    info = movies_info.get(movie_id, {})
    return {
        "movie_id": movie_id,
        "movie_name": item.get("movie_name") or info.get("name", ""),
        "name": item.get("movie_name") or info.get("name", ""),
        "director": item.get("director") or info.get("director", ""),
        "tags": item.get("tags") or info.get("tags", ""),
        "summary": info.get("summary", ""),
        "year": info.get("year", ""),
        "country": info.get("country", ""),
        "language": info.get("language", ""),
        "rating": item.get("rating_public") or info.get("rating", ""),
    }


def usable_history(history):
    return [item for item in history if item.get("movie_id", "")]


def strip_comments(history):
    stripped = []
    for item in history:
        clean_item = dict(item)
        clean_item.pop("comment", None)
        stripped.append(clean_item)
    return stripped


def split_user_history(history, holdout_count):
    holdout_indices = []
    for idx in range(len(history) - 1, -1, -1):
        if not history[idx].get("movie_id", ""):
            continue
        holdout_indices.append(idx)
        if len(holdout_indices) >= holdout_count:
            break
    holdout_indices = set(holdout_indices)
    holdouts = [item for idx, item in enumerate(history) if idx in holdout_indices]
    context = [item for idx, item in enumerate(history) if idx not in holdout_indices]
    return context, holdouts


def build_warm_loo_samples(users, movies_info):
    samples = []
    for user in users:
        user_id = user.get("user_id", "")
        history = usable_history(user.get("history", []))
        for idx, item in enumerate(history):
            context = [x for j, x in enumerate(history) if j != idx]
            samples.append(Sample(
                user_id=user_id,
                movie_id=item.get("movie_id", ""),
                history=strip_comments(context),
                target=build_target_movie(item, movies_info),
                truth=clamp_rating(item.get("rating", 3)),
                is_cold=False,
            ))
    return samples


def build_cold_loo_samples(users, movies_info):
    samples = []
    for user in users:
        user_id = user.get("user_id", "")
        for item in usable_history(user.get("history", [])):
            samples.append(Sample(
                user_id=user_id,
                movie_id=item.get("movie_id", ""),
                history=[],
                target=build_target_movie(item, movies_info),
                truth=clamp_rating(item.get("rating", 3)),
                is_cold=True,
            ))
    return samples


def build_official_like_samples(users, movies_info, seed, test_users=50, holdout=2, cold_ratio=0.25):
    rng = random.Random(seed)
    eligible = [
        user for user in users
        if user.get("user_id") and len(usable_history(user.get("history", []))) > holdout
    ]
    rng.shuffle(eligible)
    selected = eligible[:test_users]
    cold_count = min(test_users, round(test_users * cold_ratio))
    cold_ids = {user.get("user_id") for user in selected[:cold_count]}
    samples = []
    for user in selected:
        user_id = user.get("user_id", "")
        history = usable_history(user.get("history", []))
        context, holdouts = split_user_history(history, holdout)
        is_cold_user = user_id in cold_ids
        for item in holdouts:
            samples.append(Sample(
                user_id=user_id,
                movie_id=item.get("movie_id", ""),
                history=[] if is_cold_user else strip_comments(context),
                target=build_target_movie(item, movies_info),
                truth=clamp_rating(item.get("rating", 3)),
                is_cold=is_cold_user,
            ))
    return samples


def build_released_samples(data_dir):
    input_path = data_dir / "test_input.json"
    answer_path = data_dir / "test_answer.json"
    if not input_path.exists() or not answer_path.exists():
        return []
    test_input = load_json(input_path)
    answers = load_json(answer_path)
    by_key = {(a.get("user_id", ""), a.get("movie_id", "")): a for a in answers}
    samples = []
    for idx, item in enumerate(test_input):
        target = item.get("target_movie", {})
        key = (item.get("user_id", ""), target.get("movie_id", ""))
        answer = by_key.get(key) or (answers[idx] if idx < len(answers) else {})
        samples.append(Sample(
            user_id=item.get("user_id", ""),
            movie_id=target.get("movie_id", ""),
            history=item.get("context_history", []) or [],
            target=target,
            truth=clamp_rating(answer.get("rating", 3)),
            is_cold=not bool(item.get("context_history", [])),
        ))
    return samples


def metrics_for(samples, movies_info, variant):
    preds = [predict(sample, movies_info, variant) for sample in samples]
    truths = [sample.truth for sample in samples]
    errors = [abs(p - t) for p, t in zip(preds, truths)]
    if not errors:
        return {
            "n": 0, "mae": 0.0, "rmse": 0.0, "exact": 0.0, "within1": 0.0,
            "revenue": 0.0, "pred_avg": 0.0,
        }
    revenue = 0.0
    for error in errors:
        if error < 0.5:
            revenue += 10.0
        elif error <= 1.0:
            revenue += 2.0
        else:
            revenue += 0.1
    return {
        "n": len(errors),
        "mae": sum(errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "exact": sum(1 for error in errors if error == 0) / len(errors) * 100,
        "within1": sum(1 for error in errors if error <= 1) / len(errors) * 100,
        "revenue": revenue,
        "pred_avg": sum(preds) / len(preds),
    }


def mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def format_table(rows, columns):
    widths = []
    for key, title, _fmt in columns:
        widths.append(max(len(title), *(len(format_cell(row.get(key), _fmt)) for row in rows)))
    header = "  ".join(title.ljust(width) for width, (_key, title, _fmt) in zip(widths, columns))
    sep = "  ".join("-" * width for width in widths)
    lines = [header, sep]
    for row in rows:
        cells = [
            format_cell(row.get(key), fmt).ljust(width)
            for width, (key, _title, fmt) in zip(widths, columns)
        ]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def format_cell(value, fmt):
    if value is None:
        return ""
    if callable(fmt):
        return fmt(value)
    if isinstance(value, float):
        return fmt.format(value)
    return str(value)


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(title for _key, title, _fmt in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(key), fmt) for key, _title, fmt in columns) + " |")
    return "\n".join(lines)


def default_variants():
    final = Variant("final", "warm_component", "full tag+director+public+range clamp")
    return [
        replace(final, name="constant_4", description="always predict 4", warm_mode="constant_4", cold_mode="constant_4"),
        replace(final, name="user_mean", description="user mean only", warm_mode="user_mean", use_public=False, use_range_clamp=False),
        replace(final, name="user_mean+public", description="user mean + public score", warm_mode="user_mean", use_public=True, use_range_clamp=False),
        replace(final, name="tag_neighbor", description="tag neighbor + user mean", use_director=False, use_public=False, use_range_clamp=False),
        replace(final, name="tag+director", description="tag neighbor + same director", use_public=False, use_range_clamp=False),
        replace(final, name="tag+director+public", description="add public-score blend", use_range_clamp=False),
        final,
        replace(final, name="no_director", group="sensitivity", description="final without same-director bonus", use_director=False),
        replace(final, name="no_public", group="sensitivity", description="final without public-score blend", use_public=False),
        replace(final, name="no_range_clamp", group="sensitivity", description="final without user-range guardrails", use_range_clamp=False),
        replace(final, name="normal_round", group="sensitivity", description="final with ordinary rounding", rounding="normal"),
        replace(final, name="public_w0.10", group="sensitivity", description="public weight 0.10", public_weight=0.10),
        replace(final, name="public_w0.40", group="sensitivity", description="public weight 0.40", public_weight=0.40),
        replace(final, name="conf_cap0.40", group="sensitivity", description="neighbor confidence cap 0.40", confidence_cap=0.40),
        replace(final, name="conf_cap0.80", group="sensitivity", description="neighbor confidence cap 0.80", confidence_cap=0.80),
        replace(final, name="conf_div4", group="sensitivity", description="neighbor confidence divisor 4", confidence_divisor=4.0),
        replace(final, name="conf_div8", group="sensitivity", description="neighbor confidence divisor 8", confidence_divisor=8.0),
    ]


def cold_variants():
    final = Variant("cold_bucket", "cold", "final cold public-score buckets")
    return [
        replace(final, name="cold_constant_4", description="always predict 4", cold_mode="constant_4"),
        replace(final, name="cold_global_mean", description="global mean prior 3.65", cold_mode="global_mean"),
        replace(final, name="cold_public_half", description="round public score / 2", cold_mode="public_half"),
        final,
        replace(final, name="cold_bucket_normal_round", description="bucket map with ordinary rounding", rounding="normal"),
    ]


def evaluate_variant_set(variants, samples, movies_info, dataset_name):
    rows = []
    for variant in variants:
        row = metrics_for(samples, movies_info, variant)
        row.update({
            "dataset": dataset_name,
            "variant": variant.name,
            "group": variant.group,
            "description": variant.description,
        })
        rows.append(row)
    return rows


def summarize_cv(variants, users, movies_info, seeds, args):
    rows = []
    for variant in variants:
        per_seed = []
        per_seed_warm = []
        per_seed_cold = []
        for seed in seeds:
            samples = build_official_like_samples(
                users, movies_info, seed,
                test_users=args.test_users,
                holdout=args.holdout,
                cold_ratio=args.cold_ratio,
            )
            warm_samples = [sample for sample in samples if not sample.is_cold]
            cold_samples = [sample for sample in samples if sample.is_cold]
            per_seed.append(metrics_for(samples, movies_info, variant))
            per_seed_warm.append(metrics_for(warm_samples, movies_info, variant))
            per_seed_cold.append(metrics_for(cold_samples, movies_info, variant))
        row = {
            "dataset": "cv_official_like",
            "variant": variant.name,
            "group": variant.group,
            "description": variant.description,
            "n": per_seed[0]["n"] if per_seed else 0,
        }
        for key in ["mae", "rmse", "exact", "within1", "revenue"]:
            mean, std = mean_std([item[key] for item in per_seed])
            row[key] = mean
            row[f"{key}_std"] = std
        row["warm_within1"], row["warm_within1_std"] = mean_std([item["within1"] for item in per_seed_warm])
        row["cold_within1"], row["cold_within1_std"] = mean_std([item["within1"] for item in per_seed_cold])
        rows.append(row)
    return rows


def write_outputs(out_dir, all_rows, report_sections):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_results.csv"
    fieldnames = sorted({key for row in all_rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    md_path = out_dir / "ablation_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Track 2 Offline Ablation Summary\n\n")
        for title, rows, columns in report_sections:
            f.write(f"## {title}\n\n")
            f.write(markdown_table(rows, columns))
            f.write("\n\n")
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Run local deterministic ablations for claude98prompt.py")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="../../report/figs")
    parser.add_argument("--seeds", default="2026,7,99,1234,5555")
    parser.add_argument("--test-users", type=int, default=50)
    parser.add_argument("--holdout", type=int, default=2)
    parser.add_argument("--cold-ratio", type=float, default=0.25)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = script_dir / data_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir

    all_data = load_json(data_dir / "all.json")
    movies_info = load_json(data_dir / "movies_info.json")
    users = list(all_data.get("users", []))
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    variants = default_variants()
    cold_only_variants = cold_variants()
    warm_samples = build_warm_loo_samples(users, movies_info)
    cold_samples = build_cold_loo_samples(users, movies_info)
    released_samples = build_released_samples(data_dir)

    warm_component = [variant for variant in variants if variant.group == "warm_component"]
    sensitivity = [variant for variant in variants if variant.group == "sensitivity" or variant.name == "final"]

    warm_rows = evaluate_variant_set(warm_component, warm_samples, movies_info, "warm_loo")
    sensitivity_rows = evaluate_variant_set(sensitivity, warm_samples, movies_info, "warm_loo_sensitivity")
    cold_rows = evaluate_variant_set(cold_only_variants, cold_samples, movies_info, "cold_loo")
    release_rows = evaluate_variant_set([variant for variant in variants if variant.name in {
        "user_mean", "tag_neighbor", "tag+director", "tag+director+public", "final",
        "no_public", "normal_round",
    }], released_samples, movies_info, "released_test_proxy")
    cv_rows = summarize_cv([variant for variant in variants if variant.name in {
        "constant_4", "user_mean", "tag_neighbor", "tag+director", "tag+director+public",
        "final", "no_public", "normal_round", "public_w0.40",
    }], users, movies_info, seeds, args)

    all_rows = warm_rows + sensitivity_rows + cold_rows + release_rows + cv_rows

    base_columns = [
        ("variant", "variant", "{}"),
        ("mae", "MAE", "{:.3f}"),
        ("rmse", "RMSE", "{:.3f}"),
        ("exact", "Exact%", "{:.2f}"),
        ("within1", "<=1%", "{:.2f}"),
        ("revenue", "Revenue", "{:.1f}"),
    ]
    cv_columns = [
        ("variant", "variant", "{}"),
        ("mae", "MAE", "{:.3f}"),
        ("mae_std", "MAE_std", "{:.3f}"),
        ("rmse", "RMSE", "{:.3f}"),
        ("exact", "Exact%", "{:.2f}"),
        ("within1", "<=1%", "{:.2f}"),
        ("within1_std", "<=1_std", "{:.2f}"),
        ("warm_within1", "Warm<=1%", "{:.2f}"),
        ("cold_within1", "Cold<=1%", "{:.2f}"),
    ]
    report_sections = [
        ("Warm leave-one-out component ablation", warm_rows, base_columns),
        ("Warm leave-one-out sensitivity", sensitivity_rows, base_columns),
        ("Cold leave-one-out ablation", cold_rows, base_columns),
        ("Released test deterministic proxy", release_rows, base_columns),
        ("Official-like multi-seed split summary", cv_rows, cv_columns),
    ]

    csv_path, md_path = write_outputs(out_dir, all_rows, report_sections)

    print("\n== Warm leave-one-out component ablation ==")
    print(format_table(warm_rows, base_columns))
    print("\n== Warm leave-one-out sensitivity ==")
    print(format_table(sensitivity_rows, base_columns))
    print("\n== Cold leave-one-out ablation ==")
    print(format_table(cold_rows, base_columns))
    print("\n== Released test deterministic proxy ==")
    print(format_table(release_rows, base_columns))
    print("\n== Official-like multi-seed split summary ==")
    print(format_table(cv_rows, cv_columns))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
