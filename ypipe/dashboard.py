"""Export an anonymized group dashboard payload from the DuckDB warehouse."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import cluster, config


WEEKDAYS = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
TITLE_FEATURES = [
    ("has_number", "Có con số"),
    ("has_brackets", "Có ngoặc [ ]"),
    ("has_emoji", "Có emoji"),
    ("has_question", "Có dấu hỏi"),
    ("has_exclaim", "Có dấu chấm than"),
    ("has_hashtag", "Có hashtag"),
]


def _records(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def _period(con: duckdb.DuckDBPyConnection) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    first, last = con.execute("SELECT min(watched_at), max(watched_at) FROM watch_raw").fetchone()
    if first is None or last is None:
        raise ValueError("watch_raw không có lượt xem để dựng dashboard")
    lo, hi = pd.Timestamp(first), pd.Timestamp(last)
    return f"{lo:%d/%m/%Y} – {hi:%d/%m/%Y}", lo, hi


def _matrix(con: duckdb.DuckDBPyConnection, sql: str) -> list[list[int]]:
    df = _records(con, sql)
    result = [[0] * 24 for _ in WEEKDAYS]
    for row in df.itertuples(index=False):
        result[int(row.dow)][int(row.hour)] = int(row.n)
    return result


def _title_effects(facts: pd.DataFrame) -> list[list[object]]:
    if facts.empty or "views_per_day" not in facts:
        return []
    effects = []
    valid = facts.dropna(subset=["category_name", "views_per_day"]).copy()
    for column, label in TITLE_FEATURES:
        if column not in valid:
            continue
        ratios, weights = [], []
        for _, group in valid.groupby("category_name"):
            yes = group.loc[group[column].fillna(False), "views_per_day"]
            no = group.loc[~group[column].fillna(False), "views_per_day"]
            if len(yes) < 20 or len(no) < 20:
                continue
            base = float(no.median())
            if base > 0:
                ratios.append(float(yes.median()) / base)
                weights.append(min(len(yes), len(no)))
        if ratios:
            effects.append([label, round(float(np.average(ratios, weights=weights)), 2)])
    return effects


def _category_combos(con: duckdb.DuckDBPyConnection) -> list[list[object]]:
    # Reconstruct the configured 30-minute session boundary and work with unique
    # category pairs per session. Video facts are one row per video.
    sql = """
        WITH ordered AS (
            SELECT w.user_id, w.watched_at, w.video_id,
                   coalesce(f.category_name, 'Không rõ') AS category_name,
                   f.views_per_day,
                   lag(w.watched_at) OVER (PARTITION BY w.user_id ORDER BY w.watched_at) AS prev_at
            FROM watch_raw w
            LEFT JOIN video_facts f USING (video_id)
        ), marked AS (
            SELECT *, sum(CASE WHEN prev_at IS NULL OR watched_at - prev_at > INTERVAL '{config.SESSION_GAP_MIN} minutes'
                               THEN 1 ELSE 0 END)
                       OVER (PARTITION BY user_id ORDER BY watched_at ROWS UNBOUNDED PRECEDING) AS sess
            FROM ordered
        ), baskets AS (
            SELECT user_id, sess, list(DISTINCT category_name ORDER BY category_name) AS cats,
                   median(views_per_day) AS views_day
            FROM marked GROUP BY user_id, sess
        ), pairs AS (
            SELECT a AS cat_a, b AS cat_b, count(*) AS n_sessions,
                   median(views_day) AS views_day
            FROM baskets, unnest(cats) aa(a), unnest(cats) bb(b)
            WHERE a < b GROUP BY a, b
        ), marginals AS (
            SELECT cat, count(*) AS n_sessions FROM baskets, unnest(cats) c(cat) GROUP BY cat
        ), total AS (SELECT count(*) AS n FROM baskets)
        SELECT p.cat_a, p.cat_b,
               (p.n_sessions * t.n)::DOUBLE / nullif(m1.n_sessions * m2.n_sessions, 0) AS lift,
               p.views_day, p.n_sessions
        FROM pairs p JOIN marginals m1 ON p.cat_a=m1.cat
        JOIN marginals m2 ON p.cat_b=m2.cat CROSS JOIN total t
        WHERE p.n_sessions >= 20 ORDER BY lift DESC, p.n_sessions DESC LIMIT 8
    """
    try:
        pairs = _records(con, sql)
    except duckdb.Error:
        return []
    return [
        [f"{r.cat_a} + {r.cat_b}", round(float(r.lift), 2),
         f"{float(r.views_day):,.1f}" if pd.notna(r.views_day) else "—"]
        for r in pairs.itertuples(index=False)
    ]


def build_group_payload(con: duckdb.DuckDBPyConnection) -> dict:
    """Create the JSON structure consumed by the companion group HTML."""
    period, first, last = _period(con)
    uf = _records(con, "SELECT * FROM user_features ORDER BY user_id")
    assignments = _records(con, "SELECT * FROM user_clusters ORDER BY user_id")
    hourly = _records(con, "SELECT * FROM user_hourly")
    if uf.empty or assignments.empty:
        raise ValueError("Cần user_features và user_clusters để dựng dashboard nhóm")

    # Recompute the same deterministic clustering score from the same feature
    # table. It is not persisted in user_clusters, so keep it tied to cluster.run.
    model = cluster.run(uf)
    sil = float(model["silhouette"]) if model else 0.0
    labels = sorted(assignments["cluster"].dropna().astype(int).unique())
    users = []
    anonymized = {uid: f"U{i:02d}" for i, uid in enumerate(uf.user_id, start=1)}
    by_user = uf.set_index("user_id")
    for row in assignments.itertuples(index=False):
        feat = by_user.loc[row.user_id]
        users.append({
            "id": anonymized[row.user_id],
            "n": int(feat.n_events),
            "night": round(float(feat.night_ratio) * 100),
            "spd": round(float(feat.sessions_per_active_day), 2),
            "cl": int(row.cluster),
            "x": float(row.pc1),
            "y": float(row.pc2),
        })

    facts = _records(con, "SELECT * FROM video_facts")
    watched_categories = _records(con, """
        SELECT coalesce(f.category_name, 'Không rõ') AS category_name, count(*) AS n
        FROM watch_raw w LEFT JOIN video_facts f USING(video_id)
        GROUP BY 1 ORDER BY n DESC
    """)
    cats = watched_categories.category_name.head(6).tolist()
    all_counts = dict(zip(watched_categories.category_name, watched_categories.n))
    total_events = sum(int(x) for x in all_counts.values()) or 1
    cat_shares = {c: int(all_counts.get(c, 0)) / total_events for c in cats}
    cluster_rows = []
    cluster_for_user = assignments.set_index("user_id")["cluster"].astype(int)
    for cid in labels:
        member_ids = [u for u in cluster_for_user.index if int(cluster_for_user[u]) == cid]
        query_ids = ",".join("'" + str(u).replace("'", "''") + "'" for u in member_ids)
        c_hour = hourly[hourly.user_id.isin(member_ids)].groupby("hour")["n"].sum().reindex(range(24), fill_value=0).to_numpy(float)
        c_hour = (c_hour / c_hour.sum()).tolist() if c_hour.sum() else [0.0] * 24
        c_counts = _records(con, f"""
            SELECT coalesce(f.category_name, 'Không rõ') AS category_name, count(*) AS n
            FROM watch_raw w LEFT JOIN video_facts f USING(video_id)
            WHERE w.user_id IN ({query_ids}) GROUP BY 1
        """)
        c_map = dict(zip(c_counts.category_name, c_counts.n))
        c_total = sum(int(x) for x in c_map.values()) or 1
        lift = [round((int(c_map.get(c, 0)) / c_total) / cat_shares[c], 2) if cat_shares[c] else 0.0 for c in cats]
        name = assignments.loc[assignments.cluster == cid, "cluster_name"].iloc[0]
        name = str(name).split(": ", 1)[-1]
        cluster_rows.append({"name": f"Cụm {cid + 1}", "traits": name, "hour": c_hour, "lift": lift})

    hm = _matrix(con, """
        SELECT ((extract(dow FROM watched_at)::INT + 6) % 7) AS dow,
               extract(hour FROM watched_at)::INT AS hour, count(*) AS n
        FROM watch_raw GROUP BY 1,2 ORDER BY 1,2
    """)
    pub = [[0] * 24 for _ in WEEKDAYS]
    if not facts.empty:
        pubs = facts.dropna(subset=["pub_dow", "pub_hour"]).groupby(["pub_dow", "pub_hour"]).size()
        for (dow, hour), n in pubs.items():
            # video_facts.pub_dow uses pandas dayofweek: Monday=0.
            pub[int(dow)][int(hour)] = int(n)

    monthly = _records(con, """
        SELECT strftime(w.watched_at, '%Y-%m') AS month,
               coalesce(f.category_name, 'Không rõ') AS category_name, count(*) AS n
        FROM watch_raw w LEFT JOIN video_facts f USING(video_id)
        GROUP BY 1,2 ORDER BY 1,2
    """)
    months = sorted(monthly.month.unique().tolist())[-12:]
    month_labels = [pd.Period(m).strftime("%m/%y") for m in months]
    trend = {}
    for cat in cats[:5]:
        vals = []
        for month in months:
            sub = monthly[monthly.month == month]
            all_n = int(sub.n.sum()) or 1
            cat_n = int(sub.loc[sub.category_name == cat, "n"].sum())
            vals.append(round(cat_n / all_n * 100, 2))
        trend[cat] = vals

    return {
        "period": period,
        "sil": round(sil, 4),
        "users": users,
        "cats": cats,
        "clusters": cluster_rows,
        "hm": hm,
        "pub": pub,
        "months": month_labels,
        "trend": trend,
        "title": _title_effects(facts),
        "combos": _category_combos(con),
    }


def export_group_json(db_path: str | Path, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        payload = build_group_payload(con)
    finally:
        con.close()
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def export_group_dashboard(db_path: str | Path, template_path: str | Path,
                           html_path: str | Path, json_path: str | Path) -> tuple[Path, Path]:
    """Write the real-data HTML dashboard and its loadable JSON snapshot."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        payload = build_group_payload(con)
    finally:
        con.close()
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(template_path).read_text(encoding="utf-8")
    marker = "const DATA=__GROUP_DATA__;"
    if marker not in template:
        template = template.replace("const $=id=>", marker + "\nconst $=id=>", 1)
    if marker not in template or "render(demo());" not in template:
        raise ValueError("Group dashboard template is missing its data/render markers")
    html = template.replace(marker, f"const DATA={data};", 1)
    html = html.replace("render(demo());", "render(DATA);", 1)
    demo_start = html.find("function demo(){")
    demo_end = html.find("function hmap(", demo_start)
    if demo_start >= 0 and demo_end > demo_start:
        html = html[:demo_start] + html[demo_end:]
    html, _ = re.subn(
        r'<span id="dmsg">.*?</span>',
        '<span id="dmsg"><b>Dữ liệu thật.</b> Số liệu từ DuckDB đã nhúng trong trang; có thể nạp snapshot group.json khác.</span>',
        html,
        count=1,
        flags=re.S,
    )
    html = html.replace(
        'render(DATA);',
        'render(DATA);$("dmsg").innerHTML="<b>D\\u1eef li\\u1ec7u th\\u1eadt</b> t\\u1eeb DuckDB \\u0111\\u01b0\\u1ee3c nh\\u00fang trong trang. C\\u00f3 th\\u1ec3 n\\u1ea1p group.json m\\u1edbi.";',
        1,
    )
    html_output, json_output = Path(html_path), Path(json_path)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(html, encoding="utf-8")
    json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return html_output, json_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Xuất JSON cho dashboard YouTube nhóm")
    parser.add_argument("--db", default="data/warehouse.duckdb", help="Đường dẫn DuckDB")
    parser.add_argument("--out", default="reports/group.json", help="Tệp JSON đầu ra")
    args = parser.parse_args()
    result = export_group_json(args.db, args.out)
    print(f"Đã xuất {result}")


if __name__ == "__main__":
    main()
