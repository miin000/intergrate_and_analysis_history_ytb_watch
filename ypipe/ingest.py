"""Stage 2: Ingestion - phát hiện file mới trong inbox/, đọc, kiểm tra, nạp vào kho.

Hỗ trợ: .json (nhật ký xem / watch-history) hoặc .zip của Takeout.
user_id lấy từ: (1) tên thư mục con trong inbox/ , (2) tên file nếu không phải tên mặc định,
(3) nếu không có -> user_<hash6>.  Cùng user_id nạp thêm file mới => gộp & khử trùng lặp.
"""
import hashlib
import json
import re
import shutil
import unicodedata
import zipfile
from datetime import datetime, timezone

import pandas as pd

from . import config, store

VID_RE = re.compile(r"(?:[?&]v=|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})")
CH_RE = re.compile(r"/channel/([A-Za-z0-9_-]+)")
PREFIXES = ("Watched ", "Đã xem ", "Viewed ")
GENERIC = {"watch-history", "nhat-ky-xem", "lich-su-xem", "history", "watchhistory"}
MEMBER_RE = re.compile(r"(watch-history|nhật.ký.xem|lịch.sử.xem)\.json$", re.I)


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def find_inbox_files():
    config.INBOX.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(config.INBOX.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".json", ".zip", ".html") and not p.name.startswith("."):
            out.append(p)
    return out


def derive_user_id(path, content_hash):
    rel = path.relative_to(config.INBOX)
    if len(rel.parts) > 1:
        return _slug(rel.parts[0]) or f"user_{content_hash[:6]}"
    slug = _slug(path.stem)
    if slug and slug not in GENERIC and not slug.startswith("takeout"):
        return slug
    return f"user_{content_hash[:6]}"


def read_json_bytes(path):
    suf = path.suffix.lower()
    if suf == ".html":
        raise ValueError("File HTML chưa được hỗ trợ (giờ bị viết theo locale). "
                         "Hãy export lại Takeout với định dạng JSON.")
    if suf == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if MEMBER_RE.search(unicodedata.normalize("NFC", name)):
                    return z.read(name)
        raise ValueError("Không thấy file nhật ký xem (.json) trong zip.")
    return path.read_bytes()


def parse_watch_json(raw: bytes):
    data = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("JSON không đúng định dạng Takeout (cần là một danh sách).")
    rows, stats = [], {"total": len(data), "ads": 0, "non_video": 0, "bad_time": 0}
    for it in data:
        if any("Ads" in (d.get("name", "")) for d in (it.get("details") or [])):
            stats["ads"] += 1
            continue
        title = unicodedata.normalize("NFC", it.get("title") or "")
        url = it.get("titleUrl") or ""
        if not url:                        # video đã bị xoá: URL nằm trong title
            m = re.search(r"https?://\S+", title)
            url = m.group(0) if m else ""
        m = VID_RE.search(url)
        if not m or not it.get("time"):
            stats["non_video"] += 1        # bài đăng cộng đồng, thiếu URL...
            continue
        for p in PREFIXES:
            if title.startswith(p):
                title = title[len(p):]
                break
        if title.startswith("http"):
            title = ""                     # video đã xoá -> không có tiêu đề
        sub = (it.get("subtitles") or [{}])[0]
        cm = CH_RE.search(sub.get("url") or "")
        rows.append({
            "watched_at": it["time"], "video_id": m.group(1), "title": title,
            "channel_name": unicodedata.normalize("NFC", sub.get("name") or "") or None,
            "channel_id": cm.group(1) if cm else None,
            "product": (it.get("products") or [it.get("header")])[0],
        })
    df = pd.DataFrame(rows, columns=["watched_at", "video_id", "title", "channel_name", "channel_id", "product"])
    if df.empty:
        raise ValueError("Không có bản ghi video hợp lệ nào trong file.")
    t = pd.to_datetime(df["watched_at"], utc=True, format="ISO8601", errors="coerce")
    stats["bad_time"] = int(t.isna().sum())
    df["watched_at"] = t.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[t.notna()].drop_duplicates(["video_id", "watched_at"])
    stats["kept"] = len(df)
    return df, stats


def ingest_all(con):
    """Trả về (list user_id có dữ liệu mới, list lỗi)."""
    changed, errors = [], []
    for path in find_inbox_files():
        try:
            raw = read_json_bytes(path)
            h = hashlib.sha256(raw).hexdigest()
            if con.execute("SELECT 1 FROM ingested_files WHERE sha256=?", [h]).fetchone():
                print(f"  = {path.name}: đã nạp trước đó, bỏ qua")
                _archive(path, h, "dup")
                continue
            uid = derive_user_id(path, h)
            df, st = parse_watch_json(raw)
            df.insert(0, "user_id", uid)
            old = store.read(con, f"SELECT * FROM watch_raw WHERE user_id='{uid}'")
            merged = pd.concat([old, df], ignore_index=True).drop_duplicates(["user_id", "video_id", "watched_at"])
            n_new = len(merged) - len(old)
            con.execute("DELETE FROM watch_raw WHERE user_id=?", [uid])
            store.insert_df(con, "watch_raw", merged)
            con.execute("INSERT INTO ingested_files VALUES (?,?,?,?,?,?)",
                        [h, uid, path.name, len(df), n_new, datetime.now(timezone.utc).replace(tzinfo=None)])
            _archive(path, h, uid)
            changed.append(uid)
            print(f"  + {path.name} -> user '{uid}': {st['kept']:,} lượt xem hợp lệ "
                  f"({n_new:,} mới; bỏ {st['ads']} quảng cáo, {st['non_video']} không phải video)")
        except Exception as e:  # file hỏng không làm sập cả pipeline
            errors.append((path.name, str(e)))
            print(f"  ! {path.name}: LỖI - {e}")
    return sorted(set(changed)), errors


def _archive(path, h, uid):
    config.RAW.mkdir(parents=True, exist_ok=True)
    dest = config.RAW / f"{uid}__{h[:10]}{path.suffix.lower()}"
    if not dest.exists():
        shutil.copy2(path, dest)
    path.unlink()
