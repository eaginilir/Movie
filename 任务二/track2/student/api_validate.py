#!/usr/bin/env python3
"""Small real-API validation harness for prompt variants.

Unlike ablation_eval.py, this script actually calls GLM-4.5-Air through the
OpenAI-compatible HTTP endpoint.  It is intentionally sample-limited: the goal is
to check whether GLM follows the short prompts and whether a few promising
anchor variants are worth a full online submission.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path

from ablation_eval import (
    Variant,
    clean_director,
    movie_info_for,
    parse_public,
    predict,
    replace,
    tags_list,
    text_value,
)
from self_test import PromptContext, execute_prompt_function, parse_llm_response


FINAL_VARIANT = Variant("final", "api", "exact submitted anchor")
NORMAL_ROUND = replace(FINAL_VARIANT, name="normal_round", description="ordinary rounding", rounding="normal")
PUBLIC_W040 = replace(FINAL_VARIANT, name="public_w0.40", description="public-score weight 0.40", public_weight=0.40)
VARIANTS = {
    "final": FINAL_VARIANT,
    "normal_round": NORMAL_ROUND,
    "public_w0.40": PUBLIC_W040,
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def answer_map(answers):
    return {(item.get("user_id", ""), item.get("movie_id", "")): item for item in answers}


def sample_rows(test_input, answers, warm_count, cold_count):
    by_key = answer_map(answers)
    warm = []
    cold = []
    for idx, item in enumerate(test_input):
        target = item.get("target_movie", {})
        key = (item.get("user_id", ""), target.get("movie_id", ""))
        answer = by_key.get(key)
        if answer is None and idx < len(answers):
            answer = answers[idx]
        if answer is None:
            continue
        row = {
            "idx": idx,
            "input": item,
            "truth": int(round(float(answer.get("rating", 3)))),
            "is_cold": not bool(item.get("context_history", [])),
        }
        if row["is_cold"]:
            cold.append(row)
        else:
            warm.append(row)
    return warm[:warm_count] + cold[:cold_count]


def movie_name(item, info):
    return text_value(item.get("movie_name") or item.get("name") or info.get("name"), "未知", 28)


def tags_str(item, info, limit=6):
    tags = tags_list(item, info, limit)
    return "/".join(tags) if tags else ""


def prompt_for_variant(row, movies_info, variant, final_code):
    item = row["input"]
    target = item.get("target_movie", {})
    history = item.get("context_history", []) or []
    ctx = PromptContext(history, target, movies_info, [])

    if variant.name == "final":
        return execute_prompt_function(final_code, ctx)

    class Sample:
        pass

    sample = Sample()
    sample.history = history
    sample.target = target
    sample.truth = row["truth"]
    anchor = predict(sample, movies_info, variant)

    target_info = movie_info_for(target, movies_info)
    public_score = parse_public(target, target_info)
    if not history:
        target_name = movie_name(target, target_info)
        target_director = clean_director(target, target_info) or "未知"
        target_tags = tags_str(target, target_info, 6) or "未知"
        public_text = ("%.1f/10" % public_score) if public_score > 0 else "未知"
        system_prompt = (
            "你预测普通豆瓣观众给电影打几星(1-5整数)。多数观众打3-4星，"
            "公认佳作4-5星，明显差片1-2星。基准是Python给的参考星级；"
            "仅当你确知该片口碑明显更好或更差时才调整1星。只输出[Result:X]。"
        )
        user_prompt = (
            "参考:%d星\n电影:%s 导演:%s 类型:%s 公开评分:%s\n[Result:"
            % (anchor, target_name, target_director, target_tags, public_text)
        )
        return system_prompt, user_prompt

    system_prompt = "输出给定评分。只回复[Result:X]，X为给定数字。"
    user_prompt = "评分=%d\n[Result:" % anchor
    return system_prompt, user_prompt


def call_glm(system_prompt, user_prompt, api_key):
    payload = {
        "model": "glm-4.5-air",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "thinking": {"type": "disabled"},
    }
    request = urllib.request.Request(
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    start = time.time()
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    elapsed = time.time() - start
    text = body["choices"][0]["message"]["content"]
    usage = body.get("usage") or {}
    return {
        "response": text,
        "input_tokens": usage.get("prompt_tokens", len(system_prompt) + len(user_prompt)),
        "output_tokens": usage.get("completion_tokens", len(text)),
        "elapsed": elapsed,
    }


def is_format_ok(response, user_prompt):
    """Check the low-token output format used by the submitted prompt.

    Warm prompts end with the prefix ``[Result:`` to save tokens, so a completion
    like ``4]`` is intentionally valid after concatenation with the prompt.
    Cold-start prompts ask for the full ``[Result:X]`` string.
    """
    text = response or ""
    if re.fullmatch(r"\s*\[Result:\s*[1-5]\s*\]\s*", text):
        return True
    if user_prompt.rstrip().endswith("[Result:"):
        return bool(re.fullmatch(r"\s*[1-5]\s*\]\s*", text))
    return False


def summarize(rows):
    errors = [abs(row["prediction"] - row["truth"]) for row in rows]
    if not errors:
        return {}
    return {
        "n": len(rows),
        "mae": sum(errors) / len(errors),
        "rmse": math.sqrt(sum(err * err for err in errors) / len(errors)),
        "exact": sum(1 for err in errors if err == 0) / len(errors) * 100,
        "within1": sum(1 for err in errors if err <= 1) / len(errors) * 100,
        "avg_input_tokens": sum(row["input_tokens"] for row in rows) / len(rows),
        "avg_output_tokens": sum(row["output_tokens"] for row in rows) / len(rows),
        "format_ok": sum(1 for row in rows if row["format_ok"]) / len(rows) * 100,
        "cold_n": sum(1 for row in rows if row["is_cold"]),
    }


def markdown_summary(summary_rows):
    lines = [
        "| variant | n | cold | MAE | RMSE | Exact% | <=1% | in_tok | out_tok | fmt_ok% |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {variant} | {n} | {cold_n} | {mae:.3f} | {rmse:.3f} | {exact:.2f} | "
            "{within1:.2f} | {avg_input_tokens:.1f} | {avg_output_tokens:.1f} | {format_ok:.1f} |".format(**row)
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run real GLM API validation for a few prompt variants")
    parser.add_argument("--data-dir", default="data/api_seed2026")
    parser.add_argument("--code", default="claude98prompt.py")
    parser.add_argument("--variants", default="final,normal_round,public_w0.40")
    parser.add_argument("--warm", type=int, default=12)
    parser.add_argument("--cold", type=int, default=6)
    parser.add_argument("--api-key", default=os.environ.get("ZHIPUAI_API_KEY", ""))
    parser.add_argument("--out-dir", default="../../report/figs")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = script_dir / data_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.api_key:
        raise SystemExit("Missing API key. Set ZHIPUAI_API_KEY or pass --api-key.")

    movies_info = load_json(data_dir / "movies_info.json")
    test_input = load_json(data_dir / "test_input.json")
    answers = load_json(data_dir / "test_answer.json")
    final_code = (script_dir / args.code).read_text(encoding="utf-8")
    selected = sample_rows(test_input, answers, args.warm, args.cold)
    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]

    detail_rows = []
    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        print(f"\n== {variant_name} ==")
        for row in selected:
            system_prompt, user_prompt = prompt_for_variant(row, movies_info, variant, final_code)
            result = call_glm(system_prompt, user_prompt, args.api_key)
            prediction = parse_llm_response(result["response"])
            detail = {
                "variant": variant_name,
                "idx": row["idx"],
                "is_cold": row["is_cold"],
                "truth": row["truth"],
                "prediction": prediction,
                "error": abs(prediction - row["truth"]),
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "elapsed": result["elapsed"],
                "format_ok": is_format_ok(result["response"], user_prompt),
                "response": result["response"].replace("\n", "\\n")[:500],
            }
            detail_rows.append(detail)
            print(
                "idx={idx:03d} cold={cold} truth={truth} pred={pred} "
                "tok={it}/{ot} fmt={fmt}".format(
                    idx=row["idx"],
                    cold=int(row["is_cold"]),
                    truth=row["truth"],
                    pred=prediction,
                    it=result["input_tokens"],
                    ot=result["output_tokens"],
                    fmt=int(detail["format_ok"]),
                )
            )

    csv_path = out_dir / "api_validation_details.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "variant", "idx", "is_cold", "truth", "prediction", "error",
            "input_tokens", "output_tokens", "elapsed", "format_ok", "response",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_rows = []
    for variant_name in variant_names:
        rows = [row for row in detail_rows if row["variant"] == variant_name]
        summary = summarize(rows)
        summary["variant"] = variant_name
        summary_rows.append(summary)

    md_path = out_dir / "api_validation_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Real GLM API Validation\n\n")
        f.write(markdown_summary(summary_rows))
        f.write("\n")

    print("\n== summary ==")
    print(markdown_summary(summary_rows))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
