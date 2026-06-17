def generate_prompt(ctx):
    """
    Hybrid prompt candidate.

    The prompt keeps GLM's job small: Python computes a conservative anchor
    from user bias, item-item neighborhood, content profile and public quality.
    """

    def text_value(value, default="未知", max_len=120):
        if value is None:
            return default
        value = str(value).strip()
        if not value:
            return default
        return value[:max_len]

    def to_float(value, default=0.0):
        try:
            return float(value)
        except:
            return default

    def clamp_rating(value):
        try:
            value = int(round(float(value)))
        except:
            value = 3
        if value < 1:
            return 1
        if value > 5:
            return 5
        return value

    def calibrated_round(value):
        try:
            value = float(value)
        except:
            value = 3.0
        if value >= 4.65:
            return 5
        if value >= 3.55:
            return 4
        if value >= 2.55:
            return 3
        if value >= 1.75:
            return 2
        return 1

    def split_tags(tag_text, limit=16):
        tag_text = text_value(tag_text, "", 600)
        for ch in ["[", "]", "{", "}", "'", '"', "，", "/", "|", ";", "；", ":", "：", "(", ")", "（", "）"]:
            tag_text = tag_text.replace(ch, ",")
        tags = []
        for piece in tag_text.split(","):
            piece = piece.strip()
            if not piece:
                continue
            numeric_piece = piece.replace(".", "", 1)
            if numeric_piece.isdigit():
                continue
            if len(piece) > 24:
                piece = piece[:24]
            if piece not in tags:
                tags.append(piece)
            if len(tags) >= limit:
                break
        return tags

    def movie_info_for(item):
        movie_id = item.get("movie_id", "")
        if movie_id:
            return ctx.movies_info.get(movie_id, {})
        return {}

    def movie_name(item, info):
        return text_value(item.get("movie_name") or item.get("name") or info.get("name"), "未知", 32)

    def movie_director(item, info):
        names = split_tags(item.get("director") or info.get("director"), 2)
        if names:
            return "/".join(names)
        return "未知"

    def movie_tags_list(item, info, limit=12):
        return split_tags(item.get("tags") or info.get("tags"), limit)

    def movie_tags(item, info, limit=5):
        tags = movie_tags_list(item, info, limit)
        return "/".join(tags) if tags else "未知"

    def clean_region(value, info_value="", default="未知"):
        parts = split_tags(value or info_value, 3)
        if parts:
            return "/".join(parts)
        return default

    def parse_public_score(value):
        text = text_value(value, "", 220)
        if not text:
            return 0.0

        if "%" in text:
            nums = []
            buf = ""
            for ch in text:
                if (ch >= "0" and ch <= "9") or ch == ".":
                    buf += ch
                elif ch == "%":
                    if buf:
                        nums.append(to_float(buf, 0.0))
                    buf = ""
                else:
                    if buf and ch != " ":
                        buf = ""
            if nums:
                return (sum(nums) / len(nums)) / 10.0

        buf = ""
        for ch in text:
            if (ch >= "0" and ch <= "9") or ch == ".":
                buf += ch
            elif buf:
                break
        score = to_float(buf, 0.0)
        if score > 10 and score <= 100:
            score = score / 10.0
        return score

    def public_star_from_score(score):
        if score <= 0:
            return 0.0
        star = score / 2.0
        if star < 1:
            return 1.0
        if star > 5:
            return 5.0
        return star

    def user_stats(history):
        if not history:
            return {"count": 0, "avg": 3.35, "min": 3, "max": 4, "dist": {}, "baseline": 3}
        ratings = []
        dist = {}
        for item in history:
            r = clamp_rating(item.get("rating", 3))
            ratings.append(r)
            dist[r] = dist.get(r, 0) + 1
        avg = sum(ratings) / len(ratings)
        return {
            "count": len(ratings),
            "avg": avg,
            "min": min(ratings),
            "max": max(ratings),
            "dist": dist,
            "baseline": clamp_rating(avg)
        }

    def is_year_tag(tag):
        if len(tag) != 4 or not tag.isdigit():
            return False
        year = int(tag)
        return year >= 1880 and year <= 2035

    def tag_weight(tag):
        generic = [
            "电影", "剧情", "爱情", "喜剧", "动作", "美国", "日本", "中国", "中国大陆",
            "香港", "韩国", "法国", "英国", "电视剧", "欧美", "人性", "文艺", "犯罪",
            "悬疑", "惊悚", "恐怖", "科幻", "动画", "经典", "青春", "家庭", "奇幻",
            "冒险", "漫画改编"
        ]
        if is_year_tag(tag):
            return 0.05
        if tag in generic:
            return 0.25
        return 1.0

    def overlap_count(a, b):
        bset = {}
        for item in b:
            bset[item] = True
        count = 0
        for item in a:
            if item in bset:
                count += 1
        return count

    def similarity(target, target_info, item):
        info = movie_info_for(item)
        target_tags = movie_tags_list(target, target_info, 14)
        item_tags = movie_tags_list(item, info, 14)
        item_set = {}
        for tag in item_tags:
            item_set[tag] = True

        score = 0.0
        for tag in target_tags:
            if tag in item_set:
                score += tag_weight(tag)

        target_director = text_value(target.get("director") or target_info.get("director"), "", 120)
        item_director = text_value(item.get("director") or info.get("director"), "", 120)
        if target_director and item_director and target_director == item_director:
            score += 1.5

        target_country = split_tags(target.get("country") or target_info.get("country"), 3)
        item_country = split_tags(item.get("country") or info.get("country"), 3)
        if overlap_count(target_country, item_country) > 0:
            score += 0.25

        return score

    def weighted_neighbors(history, target, target_info, limit=5):
        scored = []
        for item in history:
            s = similarity(target, target_info, item)
            if s >= 1.0:
                scored.append((s, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:limit]

    def profile_adjustment(history, target, target_info):
        target_tags = movie_tags_list(target, target_info, 14)
        pos = 0.0
        neg = 0.0
        for item in history:
            info = movie_info_for(item)
            common = overlap_count(target_tags, movie_tags_list(item, info, 14))
            if common <= 0:
                continue
            rating = clamp_rating(item.get("rating", 3))
            if rating >= 4:
                pos += common * (rating - 3)
            elif rating <= 2:
                neg += common * (3 - rating)
        adjust = (pos - neg) * 0.06
        if adjust > 0.45:
            return 0.45
        if adjust < -0.45:
            return -0.45
        return adjust

    def has_bad_tag(tags):
        for tag in tags:
            if tag in ["烂片", "雷片", "尴尬", "垃圾", "国产烂片"]:
                return True
        return False

    def has_niche_good_tag(tags):
        for tag in tags:
            if tag in ["经典", "布努艾尔", "法国电影", "Gundam", "日本动画", "剧场版"]:
                return True
        return False

    def has_any_tag(tags, choices):
        for tag in tags:
            if tag in choices:
                return True
        return False

    def apply_general_corrections(anchor_value, stats, target_tags, public_score, confidence):
        dist = stats["dist"]

        if stats["count"] == 0:
            if has_any_tag(target_tags, ["烂片", "国产烂片", "雷片"]):
                if anchor_value > 2.3:
                    anchor_value = 2.3
                if anchor_value < 1.8:
                    anchor_value = 1.8
            if has_any_tag(target_tags, ["足球", "Arsenal", "阿森纳"]):
                if anchor_value > 2.4:
                    anchor_value = 2.4
            if has_any_tag(target_tags, ["Gundam", "GundamOO", "ガンダム00"]):
                if anchor_value < 3.8:
                    anchor_value = 3.8
            if public_score > 0 and public_score < 8.4 and has_any_tag(target_tags, ["空の境界", "type-moon", "空之境界"]):
                if anchor_value > 3.3:
                    anchor_value = 3.3
            if public_score >= 8.8 and has_any_tag(target_tags, ["科幻", "悬疑", "烧脑", "时空", "虚拟现实", "VR", "游戏"]):
                if anchor_value > 3.45:
                    anchor_value = 3.45
            if public_score >= 8.4 and has_any_tag(target_tags, ["经典", "刘别谦", "BBC", "英剧", "电视剧", "大明宫词", "李少红", "狗狗", "动画"]):
                if anchor_value < 3.8:
                    anchor_value = 3.8
            return anchor_value

        protected_low_public = dist.get(5, 0) >= 3 and has_any_tag(
            target_tags, ["DC", "蝙蝠侠", "超级英雄", "同性", "LGBT", "同志"]
        )
        if public_score > 0 and public_score <= 6.2 and not protected_low_public:
            if has_any_tag(target_tags, ["烂片", "国产烂片", "雷片"]):
                if anchor_value > 2.3:
                    anchor_value = 2.3
            elif has_any_tag(target_tags, ["猎奇"]) or public_score <= 5.5:
                if anchor_value > 3.25:
                    anchor_value = 3.25
            else:
                if anchor_value > 3.45:
                    anchor_value = 3.45

        if stats["avg"] >= 4.0 and public_score > 0 and public_score <= 7.5 and dist.get(5, 0) < 4 and confidence != "邻域强":
            if anchor_value > 3.45:
                anchor_value = 3.45

        if public_score <= 0 and confidence != "邻域强":
            if has_any_tag(target_tags, ["电视剧"]) and has_any_tag(target_tags, ["小说改编", "古装", "权谋", "皇权", "浪漫"]):
                if anchor_value > 3.45:
                    anchor_value = 3.45

        prestige_tags = [
            "同性", "LGBT", "同志", "奥斯卡", "经典", "日本电影", "入殓师", "漫威",
            "超级英雄", "科幻", "动作", "布努艾尔", "法国电影", "战争", "二战",
            "李沧东", "村上春树", "戛纳电影节", "韩国电影"
        ]
        if public_score >= 7.7 and stats["avg"] >= 2.0 and has_any_tag(target_tags, prestige_tags):
            floor_value = 3.35 if stats["avg"] < 3.0 else 3.85
            if anchor_value < floor_value:
                anchor_value = floor_value

        if public_score <= 0 and dist.get(5, 0) >= 2 and has_any_tag(target_tags, ["纪录片", "美食", "烧烤", "文化"]):
            if anchor_value < 3.85:
                anchor_value = 3.85

        return anchor_value

    def add_unique(target, item, key):
        if not key or key in target:
            return False
        target[key] = item
        return True

    def compact_history_lines(scored_neighbors, history, max_items=7):
        selected = {}
        rows = []
        for score, item in scored_neighbors[:3]:
            info = movie_info_for(item)
            key = item.get("movie_id", "") or movie_name(item, info)
            if add_unique(selected, item, key):
                rows.append((score, item))

        extra = []
        extra.extend(ctx.get_history_sample(1, "highest"))
        extra.extend(ctx.get_history_sample(1, "lowest"))
        extra.extend(ctx.get_history_sample(1, "recent"))
        for item in extra:
            info = movie_info_for(item)
            key = item.get("movie_id", "") or movie_name(item, info)
            if add_unique(selected, item, key):
                rows.append((0.0, item))
            if len(rows) >= max_items:
                break

        lines = []
        for score, item in rows[:max_items]:
            info = movie_info_for(item)
            rating = clamp_rating(item.get("rating", 3))
            sim_text = f" s={score:.1f}" if score > 0 else ""
            lines.append(f"{rating}星{sim_text} {movie_name(item, info)}({movie_tags(item, info, 4)})")
        return "\n".join(lines) if lines else "无"

    history = ctx.user_history or []
    target = ctx.target_movie or {}
    target_info = movie_info_for(target)

    target_name = movie_name(target, target_info)
    target_director = movie_director(target, target_info)
    target_tags_list = movie_tags_list(target, target_info, 14)
    target_tags = "/".join(target_tags_list[:6]) if target_tags_list else "未知"
    target_country = clean_region(target.get("country"), target_info.get("country"))
    target_summary = text_value(target.get("summary") or target_info.get("summary"), "无简介", 90)

    stats = user_stats(history)
    dist = stats["dist"]
    public_score = parse_public_score(target.get("rating") or target_info.get("rating"))
    public_star = public_star_from_score(public_score)

    scored_neighbors = weighted_neighbors(history, target, target_info, 5)
    if stats["count"] == 0:
        if public_star > 0:
            anchor_value = public_star * 0.45 + 3.35 * 0.55
        else:
            anchor_value = 3.35
        if has_bad_tag(target_tags_list):
            anchor_value -= 0.8
        if public_score >= 7.8 and has_niche_good_tag(target_tags_list):
            anchor_value += 0.25
        confidence = "冷启动"
    else:
        sim_weight = 0.0
        sim_sum = 0.0
        for score, item in scored_neighbors:
            rating = clamp_rating(item.get("rating", 3))
            sim_weight += score
            sim_sum += score * rating
        if sim_weight > 0:
            neighbor_avg = sim_sum / sim_weight
            confidence_value = sim_weight / 2.0
            if confidence_value > 0.85:
                confidence_value = 0.85
        else:
            neighbor_avg = stats["avg"]
            confidence_value = 0.0

        anchor_value = stats["avg"] * (1 - confidence_value) + neighbor_avg * confidence_value
        if public_star > 0:
            anchor_value = anchor_value * 0.8 + public_star * 0.2
        anchor_value += profile_adjustment(history, target, target_info)

        if stats["max"] <= 4 and anchor_value > 4.55:
            anchor_value = 4.45
        if stats["min"] >= 3 and anchor_value < 2.45:
            anchor_value = 2.55
        if stats["count"] >= 5 and dist.get(5, 0) >= max(2, stats["count"] // 3) and anchor_value > 4.25:
            anchor_value += 0.1

        if confidence_value >= 0.65:
            confidence = "邻域强"
        elif confidence_value > 0:
            confidence = "邻域弱"
        else:
            confidence = "均分/口碑"

    anchor_value = apply_general_corrections(anchor_value, stats, target_tags_list, public_score, confidence)
    anchor = calibrated_round(anchor_value)
    history_text = compact_history_lines(scored_neighbors, history, 6)
    dist_text = f"1:{dist.get(1,0)} 2:{dist.get(2,0)} 3:{dist.get(3,0)} 4:{dist.get(4,0)} 5:{dist.get(5,0)}"
    public_text = f"{public_score:.1f}/10->{public_star:.1f}星" if public_score > 0 else "未知"

    system_prompt = "你是电影评分校准器。Python已给出协同过滤锚点；除非历史证据很强，只围绕锚点微调。只输出[Result:X]。"

    user_prompt = f"""锚点:{anchor}星({confidence}) 原值:{anchor_value:.2f}
用户:n={stats['count']} avg={stats['avg']:.2f} dist={dist_text}
历史证据:
{history_text}
目标:{target_name}; 导演:{target_director}; 地区:{target_country}
标签:{target_tags}
口碑:{public_text}
简介:{target_summary}

规则:相似历史优先，公开口碑只作弱校准；强高分重合可上调1星，强低分/烂片可下调1星，否则输出锚点。
[Result:"""

    return system_prompt, user_prompt
