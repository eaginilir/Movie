def generate_prompt(ctx):
    def clean(value, default="未知"):
        if value is None:
            return default
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        if text:
            return text
        return default

    def clip(value, limit, default="未知"):
        text = clean(value, default)
        if text == default:
            return text
        return text[:limit]

    def to_float(value):
        try:
            return float(str(value).strip())
        except:
            return None

    def to_rating(value):
        number = to_float(value)
        if number is None:
            return None
        rounded = int(number + 0.5)
        if rounded < 1:
            return 1
        if rounded > 5:
            return 5
        return rounded

    def bounded_rating(value):
        if value < 1:
            return 1
        if value > 5:
            return 5
        return int(value + 0.5)

    def safe_sample(n, strategy):
        try:
            return ctx.get_history_sample(n, strategy) or []
        except:
            history_items = ctx.user_history or []
            return history_items[-n:]

    def safe_similar(n):
        try:
            return ctx.get_similar_movies(n) or []
        except:
            return []

    def item_key(item):
        return clean(item.get("movie_id") or item.get("movie_name") or item.get("name"), "")

    def format_item(item, comment_limit):
        name = clip(item.get("movie_name") or item.get("name") or item.get("fullname"), 22)
        director = clip(item.get("director"), 14, "")
        tags = clip(item.get("tags") or item.get("tag") or item.get("genres"), 30, "")
        rating = clean(item.get("rating"), "?")
        comment = clip(item.get("comment"), comment_limit, "")

        pieces = [name]
        if director:
            pieces.append("导:" + director)
        if tags:
            pieces.append("类:" + tags)
        pieces.append("分:" + rating)
        if comment:
            pieces.append("评:" + comment)
        return "- " + "；".join(pieces)

    movie = ctx.target_movie or {}
    history = ctx.user_history or []

    ratings = []
    for item in history:
        rating = to_rating(item.get("rating"))
        if rating is not None:
            ratings.append(rating)

    try:
        stats = ctx.get_user_stats() or {}
    except:
        stats = {}

    avg = to_float(stats.get("avg"))
    if avg is None or avg <= 0:
        if ratings:
            avg = sum(ratings) / len(ratings)
        else:
            avg = 3.0

    distribution = stats.get("distribution") or {}
    if not distribution and ratings:
        distribution = {}
        for rating in ratings:
            distribution[rating] = distribution.get(rating, 0) + 1

    dist_text = ", ".join(
        str(score) + ":" + str(distribution.get(score, distribution.get(str(score), 0)))
        for score in [1, 2, 3, 4, 5]
    )

    public_rating = to_float(
        movie.get("rating")
        or movie.get("average_rating")
        or movie.get("avg_rating")
        or movie.get("score")
    )
    if public_rating is not None and public_rating > 5:
        public_rating = public_rating / 2.0

    if ratings:
        if public_rating is not None:
            base = avg * 0.9 + public_rating * 0.1
        else:
            base = avg
    else:
        if public_rating is not None:
            base = public_rating
        else:
            base = 3.0

    anchor = bounded_rating(base)

    chosen = []
    seen = set()
    groups = [
        safe_similar(2),
        safe_sample(1, "highest"),
        safe_sample(1, "lowest"),
        safe_sample(1, "recent"),
    ]
    for group in groups:
        for item in group:
            key = item_key(item)
            if key and key not in seen:
                chosen.append(item)
                seen.add(key)
            if len(chosen) >= 4:
                break
        if len(chosen) >= 4:
            break

    history_text = "\n".join(format_item(item, 26) for item in chosen)
    if not history_text:
        history_text = "无"

    similar = safe_similar(2)
    similar_text = "\n".join(format_item(item, 24) for item in similar)
    if not similar_text:
        similar_text = "无"

    target_name = movie.get("movie_name") or movie.get("name") or movie.get("fullname")
    target_tags = movie.get("tags") or movie.get("tag") or movie.get("genres")
    target_year = movie.get("year") or movie.get("pubdate")
    if public_rating is None:
        target_public = "未知"
    else:
        target_public = str(round(public_rating, 2))

    system_prompt = "你是电影评分预测器。只能输出一行 [Result:X]，X是1-5整数，不要解释。"

    user_prompt = f"""请预测用户对目标电影的1-5分整数评分。
稳定基准分={anchor}。最终答案必须使用该基准分，除非历史中有非常明确的相反证据。

用户：历史数={len(history)}；均分={round(avg, 2)}；分布(1-5)={dist_text}
相似参考：
{similar_text}
代表历史：
{history_text}
目标：{clip(target_name, 28)}；导演={clip(movie.get("director"), 18)}；类型={clip(target_tags, 42)}；年份={clip(target_year, 8, "未知")}；大众评分={target_public}；简介={clip(movie.get("summary") or movie.get("description"), 110, "无简介")}

只输出：
[Result:{anchor}]"""

    return system_prompt, user_prompt
