#!/usr/bin/env python3
"""
多套随机验证驱动 (cross-validation driver)

目的：杜绝再对单一 100 条 test_input/test_answer 反复调参（那正是
best_6172102_100.py 在评分集上掉分的根因）。本脚本用多个不同 seed 各
生成一套独立切分，逐套调用 self_test.py 评分，最后汇总每项指标的
均值 ± 标准差。跨套方差大 = 仍在过拟合。

依赖同目录的 build_eval_from_train.py 与 self_test.py，二者均不修改。

用法：
    # 先用 mock 跑通流程（不需要 API Key，分数是随机的，只验证管线）
    python cv_eval.py --data-dir data --code best_6172102_general.py --mock

    # 真实评分（需要 GLM API Key，按 token 计费）
    # 多个 seed 并行跑，墙钟时间约缩短 jobs 倍：
    python cv_eval.py --data-dir data --code best_6172102_general.py \
        --api-key YOUR_KEY --jobs 5

    # 想先小样本控本可加 --limit；想看每套完整 self_test 报告默认就会打印。
    # 想安静些只看汇总，加 --quiet。

注意：data-dir 必须包含 all.json（build_eval_from_train.py 的输入）。
"""

import argparse
import math
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_SEEDS = [2026, 7, 99, 1234, 5555]

# 从 self_test.py 的 stdout 中抓取的全部数字（覆盖最终得分公式的所有分项）
PATTERNS = {
    "num_samples": re.compile(r"样本数量:\s*([0-9]+)"),
    "mae": re.compile(r"MAE\s*\(平均绝对误差\):\s*([0-9.]+)"),
    "rmse": re.compile(r"RMSE \(均方根误差\):\s*([0-9.]+)"),
    "accuracy_exact": re.compile(r"完全准确率:\s*([0-9.]+)%"),
    "accuracy_0.5": re.compile(r"误差<=0\.5 准确率:\s*([0-9.]+)%"),
    "accuracy_1.0": re.compile(r"误差<=1\.0 准确率:\s*([0-9.]+)%"),
    "input_tokens": re.compile(r"总输入Token:\s*([0-9]+)"),
    "output_tokens": re.compile(r"总输出Token:\s*([0-9]+)"),
    "total_cost": re.compile(r"总成本:\s*([0-9.]+)\s*元"),
    "total_revenue": re.compile(r"总收益:\s*([0-9.]+)\s*元"),
    "profit_rate": re.compile(r"收益率:\s*(-?[0-9.]+)"),
    "score_mae": re.compile(r"MAE得分\s*\(权重10%\):\s*([0-9.]+)"),
    "score_rmse": re.compile(r"RMSE得分\s*\(权重10%\):\s*([0-9.]+)"),
    "score_exact": re.compile(r"精准命中率\s*\(权重10%\):\s*([0-9.]+)"),
    "score_acc10": re.compile(r"准确率<=1\.0\s*\(权重30%\):\s*([0-9.]+)"),
    "score_token": re.compile(r"Token效率\s*\(权重10%\):\s*([0-9.]+)"),
    "score_profit": re.compile(r"收益率\s*\(权重50%\):\s*([0-9.]+)"),
    "raw_score": re.compile(r"原始得分:\s*([0-9.]+)"),
    "final_score": re.compile(r"综合得分\(截断100\):\s*([0-9.]+)"),
}


def mean_std(values):
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)


def build_split(python, script_dir, data_dir, out_dir, seed, test_users, holdout, cold_ratio):
    cmd = [
        python, str(script_dir / "build_eval_from_train.py"),
        "--data-dir", str(data_dir),
        "--output-dir", str(out_dir),
        "--seed", str(seed),
        "--test-users", str(test_users),
        "--holdout", str(holdout),
        "--cold-ratio", str(cold_ratio),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"build_eval_from_train.py 失败 (seed={seed}):\n{result.stdout}\n{result.stderr}")


def run_self_test(python, script_dir, data_dir, code, api_key, limit, mock,
                  line_sink=None):
    """运行 self_test.py。line_sink(line) 若提供，则每读到一行实时回调（流式进度）。"""
    cmd = [
        python, "-u", str(script_dir / "self_test.py"),
        "--data-dir", str(data_dir),
        "--code", str(code),
        "--type", "test",
    ]
    if api_key:
        cmd += ["--api-key", api_key]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    if mock:
        cmd += ["--mock"]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        if line_sink is not None:
            line_sink(line)
    proc.wait()
    out = "\n".join(lines)
    if proc.returncode != 0:
        raise RuntimeError(f"self_test.py 失败:\n{out}")

    parsed = {}
    for key, pattern in PATTERNS.items():
        m = pattern.search(out)
        if m:
            parsed[key] = float(m.group(1))
    return parsed, out


def process_seed(seed, python, script_dir, data_dir, args, line_sink=None):
    """单个 seed：建切分 + 评分。返回 (seed, metrics, full_output)。"""
    out_dir = data_dir / f"cv_seed{seed}"
    if line_sink is not None:
        line_sink("[build] 生成切分中...")
    build_split(python, script_dir, data_dir, out_dir, seed,
                args.test_users, args.holdout, args.cold_ratio)
    metrics, full_output = run_self_test(python, script_dir, out_dir, code_path(args, script_dir),
                                         args.api_key, args.limit, args.mock, line_sink=line_sink)
    metrics["seed"] = seed
    return seed, metrics, full_output


def code_path(args, script_dir):
    code = Path(args.code)
    if not code.is_absolute():
        code = (script_dir / code).resolve()
    return code


# 汇总展示的指标：分三组——核心误差、成本收益、最终得分分项
REPORT_GROUPS = [
    ("误差指标", [
        ("mae", "MAE", "{:.3f}"),
        ("rmse", "RMSE", "{:.3f}"),
        ("accuracy_exact", "完全准确率%", "{:.2f}"),
        ("accuracy_0.5", "误差<=0.5%", "{:.2f}"),
        ("accuracy_1.0", "误差<=1.0%", "{:.2f}"),
    ]),
    ("成本与收益", [
        ("input_tokens", "总输入Token", "{:.0f}"),
        ("output_tokens", "总输出Token", "{:.0f}"),
        ("total_cost", "总成本(元)", "{:.2f}"),
        ("total_revenue", "总收益(元)", "{:.2f}"),
        ("profit_rate", "收益率", "{:.3f}"),
    ]),
    ("最终得分分项", [
        ("score_mae", "MAE得分(10%)", "{:.2f}"),
        ("score_rmse", "RMSE得分(10%)", "{:.2f}"),
        ("score_exact", "精准命中(10%)", "{:.2f}"),
        ("score_acc10", "准确<=1.0(30%)", "{:.2f}"),
        ("score_token", "Token效率(10%)", "{:.2f}"),
        ("score_profit", "收益率得分(50%)", "{:.2f}"),
        ("final_score", "综合得分", "{:.2f}"),
    ]),
]


def main():
    parser = argparse.ArgumentParser(description="多套随机验证驱动：汇总各指标的均值±标准差")
    parser.add_argument("--data-dir", default="data", help="包含 all.json 的数据目录")
    parser.add_argument("--code", required=True, help="prompt 代码文件 (含 generate_prompt)")
    parser.add_argument("--api-key", default="", help="GLM API Key（真实评分用）")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="逗号分隔的 seed 列表")
    parser.add_argument("--test-users", type=int, default=50)
    parser.add_argument("--holdout", type=int, default=2)
    parser.add_argument("--cold-ratio", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0, help="每套只评前 N 个样本（控本/快速调试）")
    parser.add_argument("--jobs", type=int, default=1,
                        help="并行跑的 seed 数（API 评分时设为 seed 数可大幅缩短墙钟时间）")
    parser.add_argument("--mock", action="store_true", help="随机预测，验证管线（不计真实分）")
    parser.add_argument("--quiet", action="store_true", help="不实时转发每套 self_test 进度/报告，只看最终汇总")
    parser.add_argument("--keep", action="store_true", help="保留生成的 cv_seed* 切分目录")
    args = parser.parse_args()

    python = sys.executable or "python"
    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (script_dir / data_dir).resolve()
    args.data_dir = str(data_dir)
    code = code_path(args, script_dir)

    if not (data_dir / "all.json").exists():
        print(f"错误: 找不到 {data_dir / 'all.json'}（build_eval_from_train.py 需要它）")
        sys.exit(1)
    if not code.exists():
        print(f"错误: 找不到 prompt 代码 {code}")
        sys.exit(1)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    jobs = max(1, min(args.jobs, len(seeds)))

    print("=" * 60)
    print("多套随机验证")
    print("=" * 60)
    print(f"代码: {code.name}")
    print(f"数据: {data_dir}")
    print(f"seeds: {seeds}  test-users={args.test_users} holdout={args.holdout} cold-ratio={args.cold_ratio}")
    print(f"模式: {'mock(随机)' if args.mock else 'API'}  limit={args.limit or '全部'}  并行jobs={jobs}")
    if args.limit and not args.mock:
        print(f"【提醒】limit={args.limit} 时每套样本少、方差大，汇总数字仅供调试，正式评估请去掉 --limit 跑满。")
    print()

    created_dirs = [data_dir / f"cv_seed{s}" for s in seeds]
    results = {}
    print_lock = threading.Lock()

    def make_sink(seed):
        """实时转发 self_test 的每行输出，带 seed 前缀；并行时也不会交错错行。"""
        if args.quiet:
            return None

        def sink(line):
            if not line.strip():
                return
            with print_lock:
                print(f"[seed {seed}] {line}", flush=True)
        return sink

    def emit(seed, metrics, full_output):
        with print_lock:
            score = metrics.get("final_score")
            acc = metrics.get("accuracy_1.0")
            mae = metrics.get("mae")
            print(f">>> [seed={seed} 完成] 综合={score}  acc<=1.0={acc}%  MAE={mae}", flush=True)

    if jobs == 1:
        for seed in seeds:
            seed, metrics, full_output = process_seed(seed, python, script_dir, data_dir, args,
                                                      line_sink=make_sink(seed))
            results[seed] = metrics
            emit(seed, metrics, full_output)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(process_seed, s, python, script_dir, data_dir, args, make_sink(s)): s
                for s in seeds
            }
            for fut in as_completed(futures):
                seed, metrics, full_output = fut.result()
                results[seed] = metrics
                emit(seed, metrics, full_output)

    per_seed = [results[s] for s in seeds]  # 按 seeds 顺序，保证各套明细稳定

    # 汇总
    print("\n" + "=" * 60)
    print("汇总 (均值 ± 标准差)")
    print("=" * 60)
    if args.mock:
        print("【注意】mock 模式为随机预测，以下数字无参考意义，仅证明管线打通。")
    sample_counts = sorted({int(m["num_samples"]) for m in per_seed if "num_samples" in m})
    if sample_counts:
        print(f"每套样本数: {sample_counts}")
    print()

    for group_name, keys in REPORT_GROUPS:
        print(f"-- {group_name} --")
        print(f"{'指标':<18} {'均值':>12} {'标准差':>10}   各套(按seed顺序)")
        for key, label, fmt in keys:
            vals = [m[key] for m in per_seed if key in m]
            if not vals:
                continue
            mean_v, std_v = mean_std(vals)
            detail = ", ".join(fmt.format(m[key]) if key in m else "-" for m in per_seed)
            print(f"{label:<18} {fmt.format(mean_v):>12} {fmt.format(std_v):>10}   [{detail}]")
        print()

    print("解读：综合得分越高越好；其标准差越小说明越不依赖单套抽样、泛化越稳。")
    print("      收益率得分占总分50%、准确<=1.0占30%，是拉分主力。")

    # 清理
    if not args.keep:
        import shutil
        for d in created_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        print(f"\n已清理 {len(created_dirs)} 个临时切分目录（--keep 可保留）。")
    else:
        print(f"\n已保留切分目录：{[str(d) for d in created_dirs]}")


if __name__ == "__main__":
    main()
