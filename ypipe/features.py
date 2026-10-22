"""Stage 3b: ETL / Feature engineering - từ lượt xem thô -> sự kiện, phiên, đặc trưng theo người dùng."""
import re

import numpy as np
import pandas as pd

from . import config

DAYPART_LABEL = {"sang": "Sáng (5-12h)", "chieu": "Chiều (12-18h)", "toi": "Tối (18-22h)", "dem": "Đêm (22-5h)"}
DOW = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]


def _daypart(h):
    return np.select([(h >= 5) & (h < 12), (h >= 12) & (h < 18), (h >= 18) & (h < 22)], ["sang", "chieu", "toi"], "dem")


def build_events(raw, meta):
    ev = raw.copy()
    ts = pd.to_datetime(ev["watched_at"]).dt.tz_localize("UTC").dt.tz_convert(config.TZ).dt.tz_localize(None)
    ev["ts"] = ts
    ev["date"] = ts.dt.normalize()
    ev["hour"] = ts.dt.hour
    ev["dow"] = ts.dt.dayofweek
    ev["is_weekend"] = ev["dow"] >= 5
    ev["daypart"] = _daypart(ev["hour"].values)
    ev["is_night"] = (ev["hour"] >= config.NIGHT_START) | (ev["hour"] < config.NIGHT_END)
    ev = ev.sort_values(["user_id", "ts"]).reset_index(drop=True)
    new_sess = ev.groupby("user_id")["ts"].diff() > pd.Timedelta(minutes=config.SESSION_GAP_MIN)
    ev["session_idx"] = new_sess.groupby(ev["user_id"]).cumsum().astype(int)
    ev["session_id"] = ev["user_id"] + "-" + ev["session_idx"].astype(str)
    ev["is_repeat"] = ev.duplicated(["user_id", "video_id"], keep="first")
    if meta is not None and len(meta):
        m = meta[meta["status"] == "ok"][["video_id", "category_name", "duration_sec", "like_count", "view_count"]]
        ev = ev.merge(m, on="video_id", how="left")
    else:
        for c in ("category_name", "duration_sec", "like_count", "view_count"):
            ev[c] = np.nan
    ev["has_meta"] = ev["duration_sec"].notna() | ev["category_name"].notna()
    ev["length_bucket"] = pd.cut(ev["duration_sec"], [0, 60, 300, 1200, 3600, np.inf],
                                 labels=["≤1p (Shorts)", "1-5p", "5-20p", "20-60p", ">60p"])
    return ev


def build_sessions(ev):
    s = ev.groupby("session_id").agg(user_id=("user_id", "first"), start=("ts", "min"), end=("ts", "max"),
                                     n_videos=("video_id", "size")).reset_index()
    s["approx_minutes"] = (s["end"] - s["start"]).dt.total_seconds() / 60
    return s


def session_clustering_features(ev, sessions):
    """Bổ sung đặc trưng riêng cho phân cụm phiên, không đổi user_features hiện có."""
    columns = ["session_id", "user_id", "n_videos", "approx_minutes", "start_hour",
               "start_hour_sin", "start_hour_cos", "shorts_ratio", "category_entropy"]
    if sessions.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for session_id, group in ev.groupby("session_id", sort=False):
        hour = int(group["ts"].min().hour)
        durations = group["duration_sec"].dropna() if "duration_sec" in group else pd.Series(dtype=float)
        categories = group["category_name"].dropna().value_counts() if "category_name" in group else pd.Series(dtype=int)
        if len(categories) > 1:
            p = categories.to_numpy(dtype=float) / categories.sum()
            category_entropy = float(-(p * np.log(p)).sum() / np.log(len(p)))
        elif len(categories) == 1:
            category_entropy = 0.0
        else:
            category_entropy = np.nan
        rows.append({"session_id": session_id, "start_hour": hour,
                     "start_hour_sin": float(np.sin(2 * np.pi * hour / 24)),
                     "start_hour_cos": float(np.cos(2 * np.pi * hour / 24)),
                     "shorts_ratio": float((durations <= config.SHORTS_MAX_SEC).mean()) if len(durations) else np.nan,
                     "category_entropy": category_entropy})
    out = sessions.merge(pd.DataFrame(rows), on="session_id", how="left")
    return out[columns]


def session_sensitivity(ev):
    """Tính lại các chỉ số phiên ở nhiều ngưỡng để kiểm tra độ nhạy.

    Các hàng sự kiện giống hệt nhau ở mọi lần tính; chỉ định nghĩa về khoảng im
    lặng thay đổi. Vì vậy, thay đổi ở đây phản ánh lựa chọn ngưỡng chứ không phải
    thay đổi trong dữ liệu nguồn.
    """
    columns = ["user_id", "gap_minutes", "n_sessions", "sessions_per_active_day",
               "avg_videos_per_session", "binge_session_ratio", "avg_session_minutes"]
    if ev.empty:
        return pd.DataFrame(columns=columns)
    base = ev[["user_id", "ts", "date", "video_id"]].sort_values(["user_id", "ts"]).copy()
    active = base.groupby("user_id")["date"].nunique().rename("active_days")
    rows = []
    for gap in config.SESSION_GAP_SENSITIVITY:
        new_session = base.groupby("user_id")["ts"].diff() > pd.Timedelta(minutes=gap)
        idx = new_session.groupby(base["user_id"]).cumsum().astype(int)
        x = base.assign(_session=base["user_id"] + "-" + idx.astype(str))
        sessions = (x.groupby(["user_id", "_session"])
                     .agg(n_videos=("video_id", "size"), start=("ts", "min"), end=("ts", "max"))
                     .reset_index())
        sessions["approx_minutes"] = (sessions["end"] - sessions["start"]).dt.total_seconds() / 60
        out = sessions.groupby("user_id").agg(
            n_sessions=("_session", "size"),
            avg_videos_per_session=("n_videos", "mean"),
            binge_session_ratio=("n_videos", lambda s: float((s >= 8).mean())),
            avg_session_minutes=("approx_minutes", "mean"),
        ).reset_index()
        out["gap_minutes"] = gap
        out["sessions_per_active_day"] = out["n_sessions"] / out["user_id"].map(active)
        rows.append(out[columns])
    return pd.concat(rows, ignore_index=True)


def _entropy(counts):
    p = np.asarray(counts, float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def _jsd(a, b):
    idx = a.index.union(b.index)
    p, q = a.reindex(idx, fill_value=0).values, b.reindex(idx, fill_value=0).values
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda x, y: float(np.sum(x[x > 0] * np.log2(x[x > 0] / y[x > 0])))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def user_features(ev, sess):
    rows = []
    for uid, g in ev.groupby("user_id"):
        s = sess[sess["user_id"] == uid]
        n, active = len(g), g["date"].nunique()
        span = (g["date"].max() - g["date"].min()).days + 1
        f = {"user_id": uid, "n_events": n, "first_date": g["date"].min(), "last_date": g["date"].max(),
             "span_days": span, "active_days": active, "active_day_ratio": active / span,
             "videos_per_active_day": n / active}
        hrs = g["hour"].value_counts(normalize=True).reindex(range(24), fill_value=0)
        f["peak_hour"] = int(hrs.idxmax())
        f["night_ratio"] = float(g["is_night"].mean())
        for k in DAYPART_LABEL:
            f[f"dp_{k}"] = float((g["daypart"] == k).mean())
        f["weekend_ratio"] = float(g["is_weekend"].mean())
        per_day = g.groupby(["date", "is_weekend"]).size().groupby("is_weekend").mean()
        weekday_mean = per_day.get(False, np.nan)
        f["weekend_boost"] = float(per_day.get(True, np.nan) / weekday_mean) if pd.notna(weekday_mean) and weekday_mean else np.nan
        f["n_sessions"] = len(s)
        f["sessions_per_active_day"] = len(s) / active
        f["avg_videos_per_session"] = float(s["n_videos"].mean())
        f["max_videos_per_session"] = int(s["n_videos"].max())
        f["binge_session_ratio"] = float((s["n_videos"] >= 8).mean())
        f["avg_session_minutes"] = float(s["approx_minutes"].mean())
        f["repeat_ratio"] = float(g["is_repeat"].mean())
        f["peak_dow"] = int(g["dow"].value_counts().idxmax())
        f["peak_month"] = str(g["ts"].dt.to_period("M").value_counts().idxmax())
        vc = g["channel_name"].dropna().value_counts()
        tot = vc.sum() or 1
        f["n_channels"] = len(vc)
        f["channel_top1_share"] = float(vc.iloc[:1].sum() / tot)
        f["channel_top5_share"] = float(vc.iloc[:5].sum() / tot)
        f["channel_top20_share"] = float(vc.iloc[:20].sum() / tot)
        f["channel_entropy"] = _entropy(vc.values) if len(vc) else np.nan
        cov = float(g["has_meta"].mean())
        f["meta_coverage"] = cov
        key = "channel_name"
        if cov >= config.MIN_META_COVERAGE:
            d = g["duration_sec"].dropna()
            f["shorts_ratio"] = float((d <= config.SHORTS_MAX_SEC).mean())
            f["long_ratio"] = float((d >= config.LONG_MIN_SEC).mean())
            f["avg_duration_min"] = float(d.mean() / 60)
            cc = g["category_name"].dropna().value_counts()
            f["category_entropy"] = _entropy(cc.values)
            f["top_category"] = cc.index[0] if len(cc) else None
            f["top_category_share"] = float(cc.iloc[0] / cc.sum()) if len(cc) else np.nan
            key = "category_name"
        half = g["ts"].median()
        a, b = g[g["ts"] <= half][key].dropna().value_counts(), g[g["ts"] > half][key].dropna().value_counts()
        f["topic_drift"] = _jsd(a, b) if a.sum() >= 100 and b.sum() >= 100 else np.nan
        f["drift_basis"] = "category" if key == "category_name" else "channel"
        rows.append(f)
    uf = pd.DataFrame(rows)
    for c in ("shorts_ratio", "long_ratio", "avg_duration_min", "category_entropy", "top_category_share"):
        if c not in uf:
            uf[c] = np.nan
    if "top_category" not in uf:
        uf["top_category"] = None
    return uf


def hourly_profile(ev):
    h = ev.groupby(["user_id", "hour"]).size().rename("n").reset_index()
    h["share"] = h["n"] / h.groupby("user_id")["n"].transform("sum")
    return h


def daily_profile(ev):
    """Tổng hợp lượt xem theo ngày để trả lời nhịp xem và phát hiện ngày bất thường."""
    return (ev.groupby(["user_id", "date"])
              .agg(n_views=("video_id", "size"), n_channels=("channel_name", "nunique"))
              .reset_index())


def monthly_profile(ev):
    """Tổng hợp theo tháng, giữ user để dashboard có thể drill-down."""
    out = ev.assign(month=ev["ts"].dt.to_period("M").astype(str))
    return (out.groupby(["user_id", "month"])
              .agg(n_views=("video_id", "size"), n_channels=("channel_name", "nunique"))
              .reset_index())


def channel_summary(ev):
    """Kênh thường xem nhất, kèm độ phủ người dùng và tỷ trọng lượt xem."""
    out = (ev.dropna(subset=["channel_name"])
             .groupby("channel_name")
             .agg(n_views=("video_id", "size"), n_users=("user_id", "nunique"),
                  n_videos=("video_id", "nunique"))
             .reset_index()
             .sort_values(["n_views", "n_users"], ascending=False))
    if len(out):
        out["share"] = out["n_views"] / out["n_views"].sum()
    return out


def category_summary(ev):
    """Thể loại và thời lượng video; chỉ tính các bản ghi có metadata."""
    out = (ev.dropna(subset=["category_name"])
             .groupby("category_name")
             .agg(n_views=("video_id", "size"), n_users=("user_id", "nunique"),
                  avg_duration_min=("duration_sec", lambda s: s.mean() / 60 if s.notna().any() else np.nan))
             .reset_index()
             .sort_values(["n_views", "n_users"], ascending=False))
    if len(out):
        out["share"] = out["n_views"] / out["n_views"].sum()
    return out


def _first_non_null(values):
    present = values.dropna()
    return present.iloc[0] if len(present) else None


def _title_features(value):
    title = str(value) if pd.notna(value) else ""
    letters = [char for char in title if char.isalpha()]
    vietnamese_chars = set("áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩị"
                           "óòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
                           "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊ"
                           "ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ")
    if any(char in vietnamese_chars for char in title):
        language = "vi"
    elif re.search(r"[A-Za-z]", title):
        language = "en"
    else:
        language = "und"
    emoji = any("\U0001F300" <= char <= "\U0001FAFF" or "\u2600" <= char <= "\u27BF"
                or "\U0001F1E6" <= char <= "\U0001F1FF" for char in title)
    return {
        "title_len_chars": len(title),
        "title_len_words": len(re.findall(r"\b\w+\b", title, flags=re.UNICODE)),
        "has_number": bool(re.search(r"\d", title)),
        "has_question": "?" in title or "？" in title,
        "has_exclaim": "!" in title or "！" in title,
        "upper_ratio": sum(char.isupper() for char in letters) / len(letters) if letters else np.nan,
        "has_emoji": emoji,
        "has_brackets": bool(re.search(r"[\[\](){}（）【】「」『』<>]", title)),
        "has_hashtag": "#" in title,
        "lang_guess": language,
    }


def build_video_facts(raw, meta, video_topics=None, channel_meta=None):
    """Tạo bảng một dòng mỗi video từng xuất hiện trong lịch sử xem.

    ``metadata_status`` phân biệt video API trả về, video không còn khả dụng và
    video chưa có metadata. ``published_at`` được đổi sang config.TZ; các chỉ số
    view/like là ảnh chụp tại fetched_at, không phải lúc người dùng xem.
    """
    events = (raw.groupby("video_id", dropna=False)
              .agg(n_watch_events=("video_id", "size"), n_users=("user_id", "nunique"),
                   watch_title=("title", _first_non_null),
                   watch_channel_name=("channel_name", _first_non_null),
                   watch_channel_id=("channel_id", _first_non_null))
              .reset_index())
    meta_cols = ["video_id", "status", "title", "channel_id", "category_name", "duration_sec",
                 "published_at", "view_count", "like_count", "comment_count", "fetched_at", "made_for_kids"]
    available_meta = meta[[c for c in meta_cols if c in meta.columns]].drop_duplicates("video_id")
    facts = events.merge(available_meta, on="video_id", how="left")
    facts["title"] = facts["title"].where(facts["title"].notna() & facts["title"].ne(""), facts["watch_title"])
    facts["channel_id"] = facts["channel_id"].combine_first(facts["watch_channel_id"])
    facts["channel_name"] = facts["watch_channel_name"]
    facts["metadata_status"] = facts["status"].fillna("missing")

    if channel_meta is not None and len(channel_meta):
        cm = channel_meta[[c for c in ["channel_id", "subscriber_count"] if c in channel_meta.columns]]
        facts = facts.merge(cm.drop_duplicates("channel_id"), on="channel_id", how="left")
    elif "subscriber_count" not in facts:
        facts["subscriber_count"] = pd.NA

    if video_topics is not None and len(video_topics):
        facts = facts.merge(video_topics[["video_id", "topic_id"]].drop_duplicates("video_id"),
                            on="video_id", how="left")
    elif "topic_id" not in facts:
        facts["topic_id"] = pd.NA

    published_utc = pd.to_datetime(facts["published_at"], utc=True, errors="coerce")
    facts["published_at"] = published_utc.dt.tz_convert(config.TZ).dt.tz_localize(None)
    facts["pub_hour"] = facts["published_at"].dt.hour.astype("Int64")
    facts["pub_dow"] = facts["published_at"].dt.dayofweek.astype("Int64")
    fetched_utc = pd.to_datetime(facts["fetched_at"], utc=True, errors="coerce")
    facts["age_days"] = (fetched_utc - published_utc).dt.total_seconds() / 86400
    facts["views_per_day"] = facts["view_count"] / facts["age_days"].clip(lower=1)

    title_features = pd.DataFrame(facts["title"].map(_title_features).tolist(), index=facts.index)
    facts = pd.concat([facts, title_features], axis=1)
    facts = facts.rename(columns={"status": "_status", "watch_title": "_watch_title",
                                  "watch_channel_name": "_watch_channel_name",
                                  "watch_channel_id": "_watch_channel_id"})
    columns = ["video_id", "title", "channel_id", "channel_name", "category_name", "duration_sec",
               "published_at", "pub_hour", "pub_dow", "view_count", "like_count", "comment_count",
               "fetched_at", "age_days", "views_per_day", "n_watch_events", "n_users", "topic_id",
               "made_for_kids", "subscriber_count", "metadata_status", "title_len_chars", "title_len_words",
               "has_number", "has_question", "has_exclaim", "upper_ratio", "has_emoji", "has_brackets",
               "has_hashtag", "lang_guess"]
    for column in columns:
        if column not in facts:
            facts[column] = pd.NA
    facts = facts[columns].sort_values("video_id").reset_index(drop=True)
    for column in ("view_count", "like_count", "comment_count", "n_watch_events", "n_users", "topic_id",
                   "subscriber_count", "title_len_chars", "title_len_words"):
        facts[column] = facts[column].astype("Int64")
    facts["made_for_kids"] = facts["made_for_kids"].astype("boolean")
    return facts


def topic_transitions(ev, video_topics):
    """Đếm các lần *đổi* chủ đề giữa hai video liền nhau trong cùng một phiên.

    Không suy luận nhân quả: đây chỉ là chuỗi mở video kề nhau.  Những video không
    có nhãn chủ đề sẽ làm đứt cặp, tránh việc nối hai video xa nhau qua dữ liệu thiếu.
    """
    columns = ["user_id", "source_topic_id", "target_topic_id", "n_transitions",
               "source_outgoing", "share_from_source"]
    if video_topics is None or video_topics.empty:
        return pd.DataFrame(columns=columns)
    x = (ev[["user_id", "session_id", "ts", "video_id"]]
           .merge(video_topics[["video_id", "topic_id"]], on="video_id", how="left")
           .sort_values(["user_id", "session_id", "ts"]))
    x["target_topic_id"] = x.groupby(["user_id", "session_id"])["topic_id"].shift(-1)
    x = x.dropna(subset=["topic_id", "target_topic_id"]).copy()
    x["source_topic_id"] = x.pop("topic_id").astype(int)
    x["target_topic_id"] = x["target_topic_id"].astype(int)
    # Chuỗi cùng một chủ đề không phải là một lần chuyển chủ đề.
    x = x[x["source_topic_id"] != x["target_topic_id"]]
    if x.empty:
        return pd.DataFrame(columns=columns)
    out = (x.groupby(["user_id", "source_topic_id", "target_topic_id"])
             .size().rename("n_transitions").reset_index())
    outgoing = out.groupby(["user_id", "source_topic_id"])["n_transitions"].transform("sum")
    out["source_outgoing"] = outgoing
    out["share_from_source"] = out["n_transitions"] / outgoing
    return out.sort_values(["user_id", "n_transitions"], ascending=[True, False]).reset_index(drop=True)
