"""
数据探索与统计 — 任务一第一步
读取 train.txt / test.txt，输出数据集的基本统计信息
同时将结果保存到 stats_output.txt
"""
import os
import sys
from collections import Counter
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "stats_output.txt")


class Tee:
    """同时输出到终端和文件"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def load_data(filename):
    """解析数据文件，返回 (users, items, ratings) 列表"""
    filepath = os.path.join(DATA_DIR, filename)
    entries = []          # [(user_id, item_id, rating), ...]
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        header = lines[i]
        i += 1
        user_id, count = header.split("|")
        count = int(count)
        for _ in range(count):
            parts = lines[i].split()
            i += 1
            if len(parts) == 2:          # train: item score
                entries.append((int(user_id), int(parts[0]), int(parts[1])))
            else:                         # test: item only
                entries.append((int(user_id), int(parts[0]), None))
    return entries


def stats():
    train = load_data("train.txt")
    test  = load_data("test.txt")

    # ---------- 基础计数 ----------
    train_users  = set(e[0] for e in train)
    train_items  = set(e[1] for e in train)
    test_users   = set(e[0] for e in test)
    test_items   = set(e[1] for e in test)
    all_users    = train_users | test_users
    all_items    = train_items | test_items

    # ---------- 评分分布 ----------
    scores = [e[2] for e in train]
    score_counter = Counter(scores)

    # ---------- 每个用户的评分数 ----------
    user_rating_counts = Counter(e[0] for e in train)

    # ---------- 每个物品的被评分数 ----------
    item_rating_counts = Counter(e[1] for e in train)

    # ---------- 输出 ----------
    print("=" * 55)
    print("数据集基本统计信息")
    print("=" * 55)
    print(f"训练集用户数:          {len(train_users):>8}")
    print(f"训练集物品数:          {len(train_items):>8}")
    print(f"训练集评分数:          {len(train):>8}")
    print(f"测试集 (u,i) 对数:     {len(test):>8}")
    print(f"测试集用户数:          {len(test_users):>8}")
    print(f"测试集物品数:          {len(test_items):>8}")
    print(f"全部用户数:            {len(all_users):>8}")
    print(f"全部物品数:            {len(all_items):>8}")

    print()
    print("-" * 55)
    print("稀疏度")
    print("-" * 55)
    possible = len(all_users) * len(all_items)
    rated    = len(train)
    sparsity = 1 - rated / possible
    print(f"可能的 (u,i) 对:       {possible:>10}")
    print(f"已知评分:              {rated:>10}")
    print(f"稀疏度:                {sparsity:>10.6f} ({sparsity*100:.4f}%)")

    print()
    print("-" * 55)
    print("评分分布")
    print("-" * 55)
    for s in sorted(score_counter):
        bar = "█" * (score_counter[s] // 500)
        print(f"  {s:>3}: {score_counter[s]:>6}  {bar}")

    print()
    print("-" * 55)
    print("评分统计量")
    print("-" * 55)
    arr = np.array(scores, dtype=np.float64)
    print(f"均值:   {arr.mean():.4f}")
    print(f"方差:   {arr.var():.4f}")
    print(f"标准差: {arr.std():.4f}")
    print(f"最小值: {arr.min():.0f}")
    print(f"最大值: {arr.max():.0f}")
    print(f"中位数: {np.median(arr):.0f}")

    print()
    print("-" * 55)
    print("用户评分数分布")
    print("-" * 55)
    u_counts = np.array(list(user_rating_counts.values()))
    print(f"人均评分数: {u_counts.mean():.2f}")
    print(f"最少:       {u_counts.min()}")
    print(f"最多:       {u_counts.max()}")
    print(f"中位数:     {np.median(u_counts):.0f}")
    bins = [0, 5, 10, 20, 50, 100, 500]
    for lo, hi in zip(bins, bins[1:]):
        cnt = np.sum((u_counts > lo) & (u_counts <= hi))
        print(f"  评分数 {lo:>3}-{hi:>3}: {cnt:>5} 人")

    print()
    print("-" * 55)
    print("物品被评分数分布")
    print("-" * 55)
    i_counts = np.array(list(item_rating_counts.values()))
    print(f"均被评分数: {i_counts.mean():.2f}")
    print(f"最少:       {i_counts.min()}")
    print(f"最多:       {i_counts.max()}")
    print(f"中位数:     {np.median(i_counts):.0f}")
    for lo, hi in zip(bins, bins[1:]):
        cnt = np.sum((i_counts > lo) & (i_counts <= hi))
        print(f"  被评 {lo:>3}-{hi:>3}: {cnt:>5} 个")

    print()
    print("-" * 55)
    print("交集分析")
    print("-" * 55)
    print(f"测试集用户在训练集中出现: {len(test_users & train_users):>5} / {len(test_users)}")
    print(f"测试集物品在训练集中出现: {len(test_items & train_items):>5} / {len(test_items)}")
    print(f"测试集中新用户 (冷启动):  {len(test_users - train_users):>5}")
    print(f"测试集中新物品 (冷启动):  {len(test_items - train_items):>5}")

    print("=" * 55)


if __name__ == "__main__":
    sys.stdout = Tee(OUTPUT_FILE)
    stats()
    print(f"\n[统计结果已保存到 {OUTPUT_FILE}]")
