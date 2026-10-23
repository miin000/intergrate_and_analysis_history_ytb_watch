"""Build per-user insight reports from the local DuckDB warehouse."""
from __future__ import annotations

import argparse
import calendar
import json
import re
from itertools import combinations
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import cluster, config


DAY_NAMES = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
DAY_PARTS = ["dp_sang", "dp_chieu", "dp_toi", "dp_dem"]
DAY_GAPS = [15, 30, 60]
SESSION_BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"),
                   (5, 7, "5-7"), (8, 11, "8-11"), (12, 19, "12-19"), (20, None, "20+")]
TITLE_FEATURES = [
    ("has_number", "Có con số"),
    ("has_brackets", "Có ngoặc [ ]"),
    ("has_emoji", "Có emoji"),
    ("has_question", "Có dấu hỏi"),
    ("has_exclaim", "Có dấu chấm than"),
    ("has_hashtag", "Có hashtag"),
    ("long_title", "Dài hơn 60 ký tự"),
    ("upper_title", "Viết HOA nhiều"),
]


def _frame(con, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def _matrix(events: pd.DataFrame, date_col: str = "watched_at") -> list[list[int]]:
    mat = [[0] * 24 for _ in DAY_NAMES]
    if events.empty:
        return mat
    ts = pd.to_datetime(events[date_col])
    grouped = pd.DataFrame({"dow": ts.dt.dayofweek, "hour": ts.dt.hour}).value_counts()
    for (dow, hour), n in grouped.items():
        mat[int(dow)][int(hour)] = int(n)
    return mat


def _channels(events: pd.DataFrame) -> pd.Series:
    names = events["channel_name"].fillna("").astype(str).str.strip()
    return names.where(names.ne(""), "Kênh không rõ")


def _title_effects(viewed_facts: pd.DataFrame) -> list[list[object]]:
    if viewed_facts.empty or "views_per_day" not in viewed_facts:
        return []
    frame = viewed_facts.dropna(subset=["category_name", "views_per_day"]).copy()
    if frame.empty:
        return []
    frame["long_title"] = frame["title_len_chars"].fillna(0) > 60
    frame["upper_title"] = frame["upper_ratio"].fillna(0) >= 0.5
    effects = []
    for col, label in TITLE_FEATURES:
        if col not in frame:
            continue
        ratios, weights, n_yes = [], [], 0
        for _, group in frame.groupby("category_name"):
            mask = group[col].fillna(False).astype(bool)
            yes = group.loc[mask, "views_per_day"]
            no = group.loc[~mask, "views_per_day"]
            if len(yes) < 5 or len(no) < 5:
                continue
            base = float(no.median())
            if base <= 0:
                continue
            ratios.append(float(yes.median()) / base)
            weights.append(min(len(yes), len(no)))
            n_yes += len(yes)
        if ratios:
            effects.append([label, round(float(np.average(ratios, weights=weights)), 2), int(n_yes)])
    effects.sort(key=lambda x: abs(x[1] - 1), reverse=True)
    return effects[:8]


def _category_combos(events: pd.DataFrame, facts: pd.DataFrame) -> list[list[object]]:
    if events.empty or facts.empty:
        return []
    ev = events[["watched_at", "video_id"]].copy()
    categories = facts.set_index("video_id")["category_name"].to_dict()
    views_day = facts.set_index("video_id")["views_per_day"].to_dict()
    ev["category"] = ev.video_id.map(categories)
    ev = ev.dropna(subset=["category"]).sort_values("watched_at").reset_index(drop=True)
    if ev.empty:
        return []
    gap = pd.to_timedelta(config.SESSION_GAP_MIN, unit="m")
    ev["session"] = ev.watched_at.diff().gt(gap).cumsum()
    baskets: list[set[str]] = []
    session_videos: list[set[str]] = []
    for _, group in ev.groupby("session", sort=False):
        cats = set(group.category.dropna().astype(str))
        if cats:
            baskets.append(cats)
            session_videos.append(set(group.video_id.astype(str)))
    if len(baskets) < 2:
        return []
    category_sessions: dict[str, int] = {}
    pair_sessions: dict[tuple[str, str], int] = {}
    pair_videos: dict[tuple[str, str], set[str]] = {}
    for cats, vids in zip(baskets, session_videos):
        for cat in cats:
            category_sessions[cat] = category_sessions.get(cat, 0) + 1
        for pair in combinations(sorted(cats), 2):
            pair_sessions[pair] = pair_sessions.get(pair, 0) + 1
            pair_videos.setdefault(pair, set()).update(vids)
    total = len(baskets)
    out = []
    for pair, count in pair_sessions.items():
        if count < 3:
            continue
        lift = count * total / (category_sessions[pair[0]] * category_sessions[pair[1]])
        rates = [views_day[v] for v in pair_videos[pair]
                 if v in views_day and pd.notna(views_day[v])]
        median_rate = float(np.median(rates)) if rates else None
        out.append([f"{pair[0]} + {pair[1]}", round(lift, 2),
                    round(median_rate, 2) if median_rate is not None else None,
                    count])
    out.sort(key=lambda row: (row[1], row[3]), reverse=True)
    return out[:8]


def _month_rows(events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[list, str]:
    months = events.groupby(events.watched_at.dt.to_period("M")).size().sort_index()
    rows = []
    for period, count in months.items():
        month_start = period.start_time.date()
        month_end = period.end_time.date()
        visible_start = max(start.date(), month_start)
        visible_end = min(end.date(), month_end)
        visible_days = max((visible_end - visible_start).days + 1, 1)
        rows.append([str(period), int(count), round(int(count) / visible_days, 1),
                     visible_days, visible_days < calendar.monthrange(period.year, period.month)[1]])
    rows = rows[-12:]
    partial = [r for r in rows if r[4]]
    if partial:
        labels = ", ".join(f"{r[0]} ({r[3]} ngày có dữ liệu)" for r in partial)
        note = f"Trung bình theo ngày trong phần tháng có dữ liệu; tháng chưa trọn: {labels}. Hiển thị tối đa 12 tháng gần nhất."
    else:
        note = "Trung bình theo ngày; biểu đồ hiển thị tối đa 12 tháng gần nhất."
    return rows, note


def build_user_payload(con, user_id: str, tables: dict[str, pd.DataFrame] | None = None) -> dict:
    if tables is None:
        tables = {
            "raw": _frame(con, "SELECT * FROM watch_raw"),
            "sessions": _frame(con, "SELECT * FROM sessions"),
            "sensitivity": _frame(con, "SELECT * FROM session_sensitivity"),
            "features": _frame(con, "SELECT * FROM user_features"),
            "clusters": _frame(con, "SELECT * FROM user_clusters"),
            "hourly": _frame(con, "SELECT * FROM user_hourly"),
            "facts": _frame(con, "SELECT * FROM video_facts"),
        }
    raw = tables["raw"]
    events = raw.loc[raw.user_id.eq(user_id)].copy().sort_values("watched_at")
    if events.empty:
        raise ValueError(f"Không có watch_raw cho {user_id}")
    events["watched_at"] = pd.to_datetime(events.watched_at)
    features = tables["features"].set_index("user_id").loc[user_id]
    start = pd.Timestamp(features.first_date)
    end = pd.Timestamp(features.last_date)
    channels = _channels(events)
    known_channels = channels.loc[channels.ne("Kênh không rõ")]
    channel_counts = known_channels.value_counts()
    all_channels = int(features.n_channels)
    total = int(len(events))
    top = [[str(name), int(n)] for name, n in channel_counts.head(10).items()]
    valid_channel_events = int(channel_counts.sum()) or 1
    top_share = lambda k: round(float(channel_counts.head(k).sum() / valid_channel_events * 100), 1)

    half = len(events) // 2
    early = events.iloc[:half].copy()
    late = events.iloc[half:].copy()
    early_channels = _channels(early)
    late_channels = _channels(late)
    early_channels = early_channels.loc[early_channels.ne("Kênh không rõ")]
    late_channels = late_channels.loc[late_channels.ne("Kênh không rõ")]
    early_counts, late_counts = early_channels.value_counts(), late_channels.value_counts()
    denominator = max(len(set(late_channels)), 1)
    new_channels = round(len(set(late_channels) - set(early_channels)) / denominator * 100, 1)

    distinct_titles = events.drop_duplicates("video_id")[["video_id", "title"]].copy()
    words: dict[str, set[str]] = {}
    stop = {"the", "and", "for", "with", "from", "this", "that", "của", "và", "là", "có", "cho", "những", "một"}
    for row in distinct_titles.itertuples(index=False):
        if not isinstance(row.title, str):
            continue
        for word in re.findall(r"[^\W_]+", row.title.casefold(), flags=re.UNICODE):
            if len(word) >= 3 and word not in stop:
                words.setdefault(word, set()).add(str(row.video_id))
    top_words = [[word, len(ids)] for word, ids in sorted(words.items(), key=lambda x: (-len(x[1]), x[0]))[:10]]

    user_sessions = tables["sessions"].loc[tables["sessions"].user_id.eq(user_id)].copy()
    bucket_counts = []
    for low, high, _ in SESSION_BUCKETS:
        mask = user_sessions.n_videos.ge(low)
        if high is not None:
            mask &= user_sessions.n_videos.le(high)
        bucket_counts.append(int(mask.sum()))
    gaps = {}
    sensitivity = tables["sensitivity"].loc[tables["sensitivity"].user_id.eq(user_id)]
    for gap in DAY_GAPS:
        match = sensitivity.loc[sensitivity.gap_minutes.eq(gap)]
        if match.empty:
            gaps[str(gap)] = {"sessions": 0, "avg": 0, "binge": 0, "mins": 0}
        else:
            row = match.iloc[0]
            gaps[str(gap)] = {
                "sessions": int(row.n_sessions),
                "avg": round(float(row.avg_videos_per_session), 1),
                "binge": round(float(row.binge_session_ratio) * 100, 1),
                "mins": round(float(row.avg_session_minutes), 1),
            }

    month_rows, month_note = _month_rows(events, start, end)
    facts = tables["facts"]
    ids = events.video_id.dropna().astype(str).unique()
    viewed_facts = facts.loc[facts.video_id.astype(str).isin(ids)].drop_duplicates("video_id").copy()
    pub_matrix = [[0] * 24 for _ in DAY_NAMES]
    pub = viewed_facts.dropna(subset=["pub_dow", "pub_hour"])
    if not pub.empty:
        for row in pub.groupby(["pub_dow", "pub_hour"]).size().items():
            (dow, hour), n = row
            if 0 <= int(dow) < 7 and 0 <= int(hour) < 24:
                pub_matrix[int(dow)][int(hour)] = int(n)

    assignments = tables["clusters"].set_index("user_id")
    cluster_info = None
    if user_id in assignments.index:
        assignment = assignments.loc[user_id]
        cluster_id = int(assignment.cluster)
        member_ids = assignments.loc[assignments.cluster.eq(cluster_id)].index
        member_events = raw.loc[raw.user_id.isin(member_ids)]
        cluster_by_hour = pd.to_datetime(member_events.watched_at).dt.hour.value_counts().reindex(range(24), fill_value=0).to_numpy(float)
        user_by_hour = events.watched_at.dt.hour.value_counts().reindex(range(24), fill_value=0).to_numpy(float)
        if cluster_by_hour.sum():
            cluster_by_hour /= cluster_by_hour.sum()
        if user_by_hour.sum():
            user_by_hour /= user_by_hour.sum()
        model = tables.get("cluster_result") or cluster.run(tables["features"].copy())
        cluster_info = {
            "name": str(assignment.cluster_name),
            "members": int(len(member_ids)),
            "k": int(assignments.cluster.nunique()),
            "sil": round(float(model["silhouette"]), 3) if model else None,
            "hour": [round(float(x), 5) for x in cluster_by_hour],
            "user_hour": [round(float(x), 5) for x in user_by_hour],
        }

    payload = {
        "id": user_id,
        "n": total,
        "vids": int(events.video_id.nunique()),
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "active": int(features.active_days),
        "sess": int(features.n_sessions),
        "rep": round(float(features.repeat_ratio) * 100, 1),
        "night": round(float(features.night_ratio) * 100, 1),
        "wk": round(float(features.weekend_ratio) * 100, 1),
        "dp": [round(float(features[col]) * 100, 1) for col in DAY_PARTS],
        "heat": _matrix(events),
        "mon": month_rows,
        "monNote": month_note,
        "sd": {"lab": [label for _, _, label in SESSION_BUCKETS], "v": bucket_counts},
        "gaps": gaps,
        "top": top,
        "conc": {"top1": top_share(1), "top5": top_share(5), "top20": top_share(20), "n": all_channels},
        "words": top_words,
        "drift": round(float(features.topic_drift), 4),
        "newch": new_channels,
        "ta": [str(x) for x in early_counts.head(5).index],
        "tb": [str(x) for x in late_counts.head(5).index],
        "peak": int(features.peak_hour),
        "perf": {
            "pub": pub_matrix,
            "title": _title_effects(viewed_facts),
            "combos": _category_combos(events, viewed_facts),
            "cluster": cluster_info,
        },
        "meta": round(float(features.meta_coverage) * 100, 1),
    }
    return payload


def _safe_json(payload: dict) -> str:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))


def render_user_report(template: str, payload: dict) -> str:
    marker = "const D=__INSIGHT_DATA__,"
    if marker not in template:
        raise ValueError("Template thiếu marker __INSIGHT_DATA__")
    return template.replace(marker, f"const D={_safe_json(payload)},", 1)


def export_user_reports(db_path: str | Path, template_path: str | Path,
                        output_dir: str | Path, user_ids: list[str] | None = None) -> list[Path]:
    template = Path(template_path).read_text(encoding="utf-8")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {
            "raw": _frame(con, "SELECT * FROM watch_raw"),
            "sessions": _frame(con, "SELECT * FROM sessions"),
            "sensitivity": _frame(con, "SELECT * FROM session_sensitivity"),
            "features": _frame(con, "SELECT * FROM user_features"),
            "clusters": _frame(con, "SELECT * FROM user_clusters"),
            "hourly": _frame(con, "SELECT * FROM user_hourly"),
            "facts": _frame(con, "SELECT * FROM video_facts"),
        }
    finally:
        con.close()
    tables["cluster_result"] = cluster.run(tables["features"].copy())
    available = sorted(tables["features"].user_id.astype(str).unique())
    selected = user_ids or available
    reports = []
    for uid in selected:
        if uid not in available:
            continue
        payload = build_user_payload(None, uid, tables)
        html = render_user_report(template, payload)
        path = out_dir / f"{uid}.html"
        path.write_text(html, encoding="utf-8")
        reports.append(path)
        if uid == "u01":
            insight_path = out_dir / "insight_u01.html"
            insight_path.write_text(html, encoding="utf-8")
            reports.append(insight_path)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Xuất báo cáo insight thực tế cho từng người dùng")
    parser.add_argument("--db", default="data/warehouse.duckdb", help="Đường dẫn DuckDB")
    parser.add_argument("--template", default="ypipe/templates/insight_user.html", help="HTML template")
    parser.add_argument("--out-dir", default="reports", help="Thư mục báo cáo đầu ra")
    args = parser.parse_args()
    outputs = export_user_reports(args.db, args.template, args.out_dir)
    print(f"Exported {len(outputs)} reports to {args.out_dir}")


if __name__ == "__main__":
    main()
