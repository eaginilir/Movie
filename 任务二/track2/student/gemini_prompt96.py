def generate_prompt(ctx):
    """
    针对 Temp=0 且 无CoT (零样本直出) 优化的 Prompt 构建器。
    将计算前置到 Python，大模型仅负责语义标签的最终微调。
    """

    def text_value(value, default="未知", max_len=120):
        if value is None: return default
        value = str(value).strip()
        if not value: return default
        return value[:max_len]

    def to_float(value, default=0.0):
        try: return float(value)
        except: return default

    def clamp_rating(value):
        try: value = int(round(float(value)))
        except: value = 3
        if value < 1: return 1
        if value > 5: return 5
        return value

    def split_tags(tag_text, limit=8):
        tag_text = text_value(tag_text, "", 500)
        for ch in ["[", "]", "{", "}", "'", '"', "，", "/", "|", ";", "；", ":", "："]:
            tag_text = tag_text.replace(ch, ",")
        pieces = tag_text.split(",")
        tags = []
        for piece in pieces:
            piece = piece.strip()
            if not piece: continue
            numeric_piece = piece.replace(".", "", 1)
            if numeric_piece.isdigit(): continue
            if len(piece) > 18: piece = piece[:18]
            if piece not in tags: tags.append(piece)
            if len(tags) >= limit: break
        return tags

    def movie_info_for(item):
        movie_id = item.get("movie_id", "")
        return ctx.movies_info.get(movie_id, {}) if movie_id else {}

    def movie_name(item, info):
        return text_value(item.get("movie_name") or item.get("name") or info.get("name"), "未知", 32)

    def movie_tags(item, info, limit=4):
        tags = item.get("tags") or info.get("tags") or ""
        return "/".join(split_tags(tags, limit))

    def user_stats(history):
        if not history:
            return {"count": 0, "avg": 3.5, "baseline": 3}
        ratings = [clamp_rating(item.get("rating", 3)) for item in history]
        avg = sum(ratings) / len(ratings)
        return {"count": len(ratings), "avg": avg, "baseline": clamp_rating(avg)}

    def compact_history_lines(history, max_items=8):
        lines = []
        used = set()
        for item in history:
            key = item.get("movie_id", "") or item.get("movie_name", "") or item.get("name", "")
            if key in used: continue
            used.add(key)
            info = movie_info_for(item)
            name = movie_name(item, info)
            tags = movie_tags(item, info, 4)
            rating = clamp_rating(item.get("rating", 3))
            
            # 格式极简，方便模型瞬间捕捉特征
            lines.append(f"[{rating}星] {name} ({tags})")
            if len(lines) >= max_items: break
        return "\n".join(lines) if lines else "无"

    # === 数据准备 ===
    history = ctx.user_history or []
    target = ctx.target_movie or {}
    target_info = movie_info_for(target)

    target_name = movie_name(target, target_info)
    target_tags = movie_tags(target, target_info, 5)
    
    public_rating = to_float(target_info.get("rating"), 0.0)
    stats = user_stats(history)
    
    similar_movies = ctx.get_similar_movies(5)
    sim_avg = 0.0
    if similar_movies:
        sim_ratings = [clamp_rating(m.get("rating", 3)) for m in similar_movies]
        sim_avg = sum(sim_ratings) / len(sim_ratings)

    # === Python 前置计算核心锚点 (代替大模型的数学计算) ===
    suggested_anchor = 3
    if sim_avg > 0:
        suggested_anchor = clamp_rating(sim_avg)  # 强协同过滤信号
    elif public_rating > 0:
        base_star = public_rating / 2.0
        # 用户偏差修正：简单叠加
        bias = stats["avg"] - 3.5 if stats["count"] > 0 else 0
        suggested_anchor = clamp_rating(base_star + bias * 0.6)
    elif stats["count"] > 0:
        suggested_anchor = stats["baseline"]

    # === 历史提取 ===
    selected = []
    selected.extend(ctx.get_similar_movies(3))
    selected.extend(ctx.get_history_sample(2, "highest"))
    selected.extend(ctx.get_history_sample(2, "lowest"))
    selected.extend(ctx.get_history_sample(1, "recent"))
    history_text = compact_history_lines(selected, 8)

    # === Prompt 构建 ===
    system_prompt = "你是直觉型推荐算法。严格根据目标特征与历史记录的重合度微调分数。只输出 [Result:X]，X是1-5整数。"

    user_prompt = f"""【预设锚点】
根据大众口碑与用户均分，本片基础参考分为：{suggested_anchor} 星。

【用户历史参考】
{history_text}

【目标电影】
名称：{target_name}
特征：{target_tags}

【直觉判定规则】
1. 对比目标【特征】与【用户历史参考】：
   - 若目标特征与 4-5星 历史高度重叠，在参考分 {suggested_anchor} 基础上适度上调（最高5）。
   - 若目标特征与 1-2星 历史高度重叠，在参考分 {suggested_anchor} 基础上适度下调（最低1）。
   - 若无明显重叠或冲突，直接输出参考分 {suggested_anchor}。

请直接给出最终判定：
[Result:"""

    return system_prompt, user_prompt