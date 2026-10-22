"""Stage 3a: Enrichment - lấy metadata video từ YouTube Data API v3 (videos.list, 50 id/lần, 1 quota unit/lần).
Có cache trong DuckDB: video nào đã lấy thì không gọi lại. Hết quota => dừng êm, lần chạy sau tiếp tục.
"""
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from . import config, store

API = "https://www.googleapis.com/youtube/v3/videos"
CHANNELS_API = "https://www.googleapis.com/youtube/v3/channels"
COMMENTS_API = "https://www.googleapis.com/youtube/v3/commentThreads"
CATS = {1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music", 15: "Pets & Animals", 17: "Sports",
        18: "Short Movies", 19: "Travel & Events", 20: "Gaming", 21: "Videoblogging", 22: "People & Blogs",
        23: "Comedy", 24: "Entertainment", 25: "News & Politics", 26: "Howto & Style", 27: "Education",
        28: "Science & Technology", 29: "Nonprofits & Activism", 30: "Movies", 43: "Shows", 44: "Trailers"}
DUR_RE = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$")


class QuotaExceeded(Exception):
    pass


def parse_duration(s):
    m = DUR_RE.match(s or "")
    if not m:
        return None
    d, h, mi, sec = (int(x or 0) for x in m.groups())
    total = d * 86400 + h * 3600 + mi * 60 + sec
    return float(total) if total > 0 else None   # P0D = livestream chưa kết thúc


def _fetch_batch(ids):
    """Gọi API thật. Trả về list items (dict theo định dạng YouTube)."""
    if config.MOCK_API:                                   # chế độ chạy thử không cần key
        with open(config.MOCK_API, encoding="utf-8") as f:
            cat = json.load(f)
        items = []
        for video_id in ids:
            if video_id not in cat:
                continue
            data = cat[video_id]
            data = data if isinstance(data, dict) else {"category_id": data[0], "duration_sec": data[1]}
            if "like_count" not in data:
                data["like_count"] = 50  # giữ nguyên dữ liệu mặc định của catalog mock cũ
            item = {"id": video_id,
                    "snippet": {"title": data.get("title", ""), "description": "",
                                "channelId": data.get("channel_id", "mock"),
                                "categoryId": str(data.get("category_id", 10)),
                                "publishedAt": data.get("published_at", "2025-01-01T00:00:00Z")},
                    "contentDetails": {"duration": f"PT{int(data.get('duration_sec', 60))}S"},
                    "statistics": {"viewCount": str(data.get("view_count", 1000)),
                                   "commentCount": str(data.get("comment_count", 5))}}
            if data.get("like_count") is not None:
                item["statistics"]["likeCount"] = str(data["like_count"])
            if "made_for_kids" in data:
                item["status"] = {"madeForKids": data["made_for_kids"]}
            items.append(item)
        return items
    q = urllib.parse.urlencode({"part": "snippet,contentDetails,statistics,status", "id": ",".join(ids),
                                "maxResults": 50, "key": config.API_KEY})
    try:
        with urllib.request.urlopen(f"{API}?{q}", timeout=30) as r:
            return json.load(r).get("items", [])
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        if e.code == 403 and "quota" in body.lower():
            raise QuotaExceeded()
        raise RuntimeError(f"YouTube API lỗi {e.code}: {body[:300]}")


def _fetch_channel_batch(ids):
    """Lấy thống kê kênh; subscriberCount có thể bị ẩn hoặc không được trả về."""
    if config.MOCK_API:
        with open(config.MOCK_API, encoding="utf-8") as f:
            catalog = json.load(f)
        channels = catalog.get("_channels", {})
        return [{"id": channel_id,
                 "statistics": {key: str(value) for key, value in channels.get(channel_id, {}).items()
                                if key in {"subscriberCount", "videoCount"} and value is not None}}
                for channel_id in ids]
    q = urllib.parse.urlencode({"part": "statistics", "id": ",".join(ids),
                                "maxResults": 50, "key": config.API_KEY})
    try:
        with urllib.request.urlopen(f"{CHANNELS_API}?{q}", timeout=30) as r:
            return json.load(r).get("items", [])
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        if e.code == 403 and "quota" in body.lower():
            raise QuotaExceeded()
        raise RuntimeError(f"YouTube channels API lỗi {e.code}: {body[:300]}")


def _to_row(it, now):
    sn, cd, st = it.get("snippet", {}), it.get("contentDetails", {}), it.get("statistics", {})
    cid = int(sn["categoryId"]) if sn.get("categoryId") else None
    num = lambda k: int(st[k]) if k in st else None     # likeCount có thể bị ẩn
    return {"video_id": it["id"], "status": "ok", "title": sn.get("title"),
            "description": (sn.get("description") or "")[:300], "channel_id": sn.get("channelId"),
            "category_id": cid, "category_name": CATS.get(cid, f"Other ({cid})") if cid else None,
            "published_at": pd.to_datetime(sn.get("publishedAt"), utc=True).tz_localize(None) if sn.get("publishedAt") else pd.NaT,
            "duration_sec": parse_duration(cd.get("duration")), "view_count": num("viewCount"),
            "like_count": num("likeCount"), "comment_count": num("commentCount"),
            "tags": "|".join(sn.get("tags", []))[:500], "default_language": sn.get("defaultLanguage"),
            "fetched_at": now, "made_for_kids": it.get("status", {}).get("madeForKids"),
            "status_fetched_at": now}


def _enrich_videos(con, max_calls=None):
    """Trả về (số video đã cập nhật, số lời gọi API)."""
    if not config.API_KEY and not config.MOCK_API:
        print("  (không có YOUTUBE_API_KEY -> bỏ qua enrichment; chỉ dùng time/title/channel)")
        return 0, 0
    missing = [r[0] for r in con.execute("""
        SELECT DISTINCT w.video_id FROM watch_raw w
        LEFT JOIN video_metadata m USING(video_id)
        WHERE m.video_id IS NULL OR (m.status='ok' AND m.status_fetched_at IS NULL)
        ORDER BY w.video_id
    """).fetchall()]
    if not missing:
        return 0, 0
    n_calls = (len(missing) + 49) // 50
    print(f"  {len(missing):,} video chưa có metadata (~{n_calls} lượt gọi API, {n_calls} quota units)")
    done, calls = 0, 0
    for k in range(0, len(missing), 50):
        if max_calls is not None and k // 50 >= max_calls:
            print(f"  đạt giới hạn --max-api-calls={max_calls}, sẽ tiếp tục lần chạy sau")
            break
        batch, now = missing[k:k + 50], datetime.now(timezone.utc).replace(tzinfo=None)
        try:
            items = _fetch_batch(batch)
        except QuotaExceeded:
            print("  hết quota API trong ngày -> dừng, lần chạy sau sẽ tiếp tục")
            break
        calls += 1
        rows = [_to_row(it, now) for it in items]
        got = {r["video_id"] for r in rows}
        rows += [{"video_id": v, "status": "unavailable", "fetched_at": now,
                  "status_fetched_at": now} for v in batch if v not in got]
        existing = {r[0] for r in con.execute(
            "SELECT video_id FROM video_metadata WHERE video_id IN (SELECT unnest(?))", [batch]).fetchall()}
        fresh_rows = [r for r in rows if r["video_id"] not in existing]
        for row in rows:
            if row["video_id"] in existing:
                con.execute("UPDATE video_metadata SET made_for_kids=?, status_fetched_at=? WHERE video_id=?",
                            [row.get("made_for_kids"), row["status_fetched_at"], row["video_id"]])
        df = pd.DataFrame(fresh_rows)
        if df.empty:
            done += len(batch)
            continue
        for c in ("category_id", "view_count", "like_count", "comment_count"):
            if c not in df:
                df[c] = pd.NA
            df[c] = df[c].astype("Int64")
        if "made_for_kids" not in df:
            df["made_for_kids"] = pd.NA
        df["made_for_kids"] = df["made_for_kids"].astype("boolean")
        for c in ("title", "description", "channel_id", "category_name", "tags", "default_language"):
            if c not in df:
                df[c] = None
        for c in ("published_at", "duration_sec"):
            if c not in df:
                df[c] = None
        if "status_fetched_at" not in df:
            df["status_fetched_at"] = now
        store.insert_df(con, "video_metadata", df)
        done += len(batch)
        if (k // 50) % 20 == 0:
            print(f"    {done:,}/{len(missing):,}")
    return done, calls


def _enrich_channels(con, max_calls=None):
    """Lưu cache channel_metadata trong DuckDB; trả về (số kênh, số lượt gọi)."""
    if not config.API_KEY and not config.MOCK_API:
        return 0, 0
    missing = [r[0] for r in con.execute("""
        SELECT DISTINCT channel_id FROM video_metadata
        WHERE status='ok' AND channel_id IS NOT NULL
          AND channel_id NOT IN (SELECT channel_id FROM channel_metadata)
        ORDER BY channel_id
    """).fetchall()]
    if not missing:
        return 0, 0
    print(f"  {len(missing):,} kênh chưa có metadata")
    done, calls = 0, 0
    for k in range(0, len(missing), 50):
        if max_calls is not None and calls >= max_calls:
            print(f"  đạt giới hạn --max-api-calls, còn {len(missing) - done:,} kênh sẽ tiếp tục lần chạy sau")
            break
        batch = missing[k:k + 50]
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        try:
            items = _fetch_channel_batch(batch)
        except QuotaExceeded:
            print("  hết quota API khi lấy metadata kênh -> dừng, lần chạy sau sẽ tiếp tục")
            break
        calls += 1
        by_id = {item["id"]: item.get("statistics", {}) for item in items}
        rows = []
        for channel_id in batch:
            stats = by_id.get(channel_id, {})
            rows.append({"channel_id": channel_id,
                         "subscriber_count": int(stats["subscriberCount"]) if stats.get("subscriberCount") is not None else None,
                         "video_count": int(stats["videoCount"]) if stats.get("videoCount") is not None else None,
                         "fetched_at": now})
        df = pd.DataFrame(rows)
        for column in ("subscriber_count", "video_count"):
            df[column] = df[column].astype("Int64")
        store.insert_df(con, "channel_metadata", df)
        done += len(batch)
    if done:
        store.export_parquet(con, "channel_metadata")
    return done, calls


def enrich_all(con, max_calls=None):
    """Enrich video và kênh theo thứ tự, chia sẻ chung giới hạn lời gọi API."""
    videos, used = _enrich_videos(con, max_calls)
    remaining = None if max_calls is None else max(0, max_calls - used)
    channels, _ = _enrich_channels(con, remaining)
    return videos + channels


def enrich(con, max_calls=None):
    """Giữ điểm gọi cũ; nay cũng lấy và cache metadata kênh."""
    return enrich_all(con, max_calls)


def enrich_comments(con, max_videos=None):
    """Lấy top-level comments tùy chọn; không chạy mặc định để kiểm soát quota và riêng tư."""
    if not config.COMMENTS_ENABLED or not config.API_KEY:
        return 0
    limit = max_videos or config.COMMENTS_MAX_VIDEOS
    videos = [r[0] for r in con.execute(
        "SELECT video_id FROM watch_raw GROUP BY video_id ORDER BY count(*) DESC LIMIT ?", [limit]).fetchall()]
    done = 0
    for video_id in videos:
        if con.execute("SELECT 1 FROM video_comments WHERE video_id=? LIMIT 1", [video_id]).fetchone():
            continue
        q = urllib.parse.urlencode({"part": "snippet", "videoId": video_id, "maxResults": config.COMMENTS_PER_VIDEO,
                                    "textFormat": "plainText", "key": config.API_KEY})
        try:
            with urllib.request.urlopen(f"{COMMENTS_API}?{q}", timeout=30) as r:
                items = json.load(r).get("items", [])
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            if e.code == 403 and "quota" in body.lower():
                print("  hết quota khi lấy comments -> dừng")
                break
            if e.code in (403, 404):
                continue
            raise RuntimeError(f"YouTube comments API lỗi {e.code}: {body[:300]}")
        rows = []
        for item in items:
            sn = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            rows.append({"comment_id": item.get("id"), "video_id": video_id,
                         "author": sn.get("authorDisplayName"), "text": sn.get("textDisplay", ""),
                         "published_at": pd.to_datetime(sn.get("publishedAt"), utc=True).tz_localize(None)
                         if sn.get("publishedAt") else pd.NaT,
                         "like_count": int(sn.get("likeCount", 0))})
        if rows:
            store.insert_df(con, "video_comments", pd.DataFrame(rows))
        done += 1
    return done
