def generate_prompt(ctx):
    """
    Generalization-first movie-rating prompt (no identity lookup, no per-movie
    hardcoding). Grounded in leave-one-out analysis of train.json (1900 hold-outs,
    the honest estimate for unseen eval users):

      * Warm (user has history): a tag-neighbor + same-director collaborative
        anchor blended toward the user's own mean, lightly pulled by the public
        score, scores ~44% exact / ~91% within-1 / RMSE 0.93. GLM-4.5-air (think
        off) is *strictly worse* than this anchor on warm samples (3 prior runs:
        any latitude lowers accuracy because the model regresses/overshoots), so
        the warm branch makes GLM a deterministic printer of the anchor. Cost:
        ~50 input tokens/sample, output 6 tokens -> token efficiency maxed.

      * Cold start (no history, 24% of test): the only signal is the film itself.
        A public-score -> rating map (fitted on train buckets) alone already gets
        ~43% / 91%. Here GLM's film knowledge genuinely adds value, so we hand it
        the calibrated reference and let it move at most 1 star.

    Everything is pure collaborative-filtering math + public-quality priors, so it
    transfers to users/movies absent from the training pool.
    """

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
        if value >= 4.55:
            return 5
        if value >= 3.55:
            return 4
        if value >= 2.55:
            return 3
        if value >= 1.70:
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
            if piece.replace(".", "", 1).isdigit():
                continue
            if len(piece) > 20:
                piece = piece[:20]
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
        return text_value(item.get("movie_name") or item.get("name") or info.get("name"), "未知", 28)

    def tags_list(item, info, limit=14):
        return split_tags(item.get("tags") or info.get("tags"), limit)

    def tags_str(item, info, limit=4):
        t = tags_list(item, info, limit)
        return "/".join(t) if t else ""

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
        """Return a public quality score on a 0-10 scale.
        Handles plain numbers ('8.4'), 0-100 ints, and percentage-dicts
        ("{'科幻片': '92%', ...}") by averaging the percents."""
        text = text_value(item.get("rating") or info.get("rating"), "", 220)
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
                    buf = ""
            if nums:
                return (sum(nums) / len(nums)) / 10.0
            return 0.0
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

    def overlap_count(a, b):
        bset = {}
        for x in b:
            bset[x] = True
        return sum(1 for x in a if x in bset)

    def user_stats(history):
        ratings = []
        dist = {}
        for it in history:
            r = clamp_rating(it.get("rating", 3))
            ratings.append(r)
            dist[r] = dist.get(r, 0) + 1
        if not ratings:
            return {"count": 0, "avg": 0.0, "min": 0, "max": 0, "dist": {}}
        return {"count": len(ratings), "avg": sum(ratings) / len(ratings),
                "min": min(ratings), "max": max(ratings), "dist": dist}

    def warm_anchor(history, target, target_info, stats, public_score):
        """Tag-neighbor + same-director CF rating, blended toward user mean,
        then lightly pulled by public score. Tuned on LOO:
        dirw=2, conf cap=0.6, div=6, pubw=0.27."""
        target_tags = tags_list(target, target_info, 14)
        target_director = clean_director(target, target_info)
        wsum = 0.0
        vsum = 0.0
        for it in history:
            info = movie_info_for(it)
            w = overlap_count(target_tags, tags_list(it, info, 14))
            it_dir = clean_director(it, info)
            if target_director and it_dir and target_director == it_dir:
                w += 2.0
            if w > 0:
                rating = clamp_rating(it.get("rating", 3))
                wsum += w
                vsum += w * rating
        if wsum > 0:
            conf = wsum / 6.0
            if conf > 0.6:
                conf = 0.6
            anchor = (vsum / wsum) * conf + stats["avg"] * (1 - conf)
        else:
            anchor = stats["avg"]
        if public_score > 0:
            anchor = anchor * 0.73 + (public_score / 2.0) * 0.27
        # keep inside the user's demonstrated range -> controls big errors / RMSE
        if stats["max"] <= 4 and anchor > 4.5:
            anchor = 4.4
        if stats["min"] >= 3 and anchor < 2.5:
            anchor = 2.6
        return anchor

    def cold_reference(public_score):
        """Public-score -> expected rating, fitted on train.json buckets.
        Alone scores ~43% exact / 91% within-1 on cold movies."""
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

    history = ctx.user_history or []
    target = ctx.target_movie or {}
    target_info = movie_info_for(target)
    public_score = parse_public(target, target_info)
    stats = user_stats(history)

    # ---- Cold start: film knowledge is the only signal -> let GLM nudge ----
    if stats["count"] == 0:
        ref = calibrated_round(cold_reference(public_score))
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
            % (ref, target_name, target_director, target_tags, public_text)
        )
        return system_prompt, user_prompt

    # ---- Warm: anchor IS the optimum; GLM only degrades it -> force echo ----
    anchor = calibrated_round(warm_anchor(history, target, target_info, stats, public_score))
    system_prompt = "输出给定评分。只回复[Result:X]，X为给定数字。"
    user_prompt = "评分=%d\n[Result:" % anchor
    return system_prompt, user_prompt
