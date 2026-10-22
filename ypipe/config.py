"""Mọi tham số có thể chỉnh nằm ở đây."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("YTPIPE_HOME", Path(__file__).resolve().parent.parent))
INBOX = ROOT / "inbox"          # thả file lịch sử vào đây
DATA = ROOT / "data"
RAW = DATA / "raw"              # bản gốc được lưu lại sau khi nạp (Stage 1: Raw Data)
PARQUET = DATA / "parquet"      # bản xuất Parquet của các bảng kết quả
REPORTS = ROOT / "reports"
DB_PATH = DATA / "warehouse.duckdb"


def _load_env():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_env()
API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
MOCK_API = os.environ.get("YTPIPE_MOCK_API", "")   # đường dẫn catalog giả để chạy thử không cần key
COMMENTS_ENABLED = os.environ.get("YTPIPE_COMMENTS", "").lower() in {"1", "true", "yes"}
COMMENTS_MAX_VIDEOS = int(os.environ.get("YTPIPE_COMMENTS_MAX_VIDEOS", "100"))
COMMENTS_PER_VIDEO = int(os.environ.get("YTPIPE_COMMENTS_PER_VIDEO", "20"))

# ---- Thời gian ----
TZ = "Asia/Ho_Chi_Minh"        # Takeout lưu giờ UTC -> đổi sang giờ địa phương
SESSION_GAP_MIN = 30           # 2 video cách nhau > 30 phút => phiên mới
# Kiểm định độ nhạy: dashboard luôn so sánh các ngưỡng này với ngưỡng chính ở trên.
SESSION_GAP_SENSITIVITY = (15, 30, 60)
NIGHT_START, NIGHT_END = 22, 5  # "đêm" = 22h..4h59

# ---- Phân loại video theo thời lượng (cần API) ----
SHORTS_MAX_SEC = 60
LONG_MIN_SEC = 20 * 60
MIN_META_COVERAGE = 0.30       # chỉ dùng đặc trưng từ API nếu >= 30% video lấy được metadata

# ---- NLP chủ đề trên title ----
N_TOPICS = 12
MIN_TITLES_FOR_TOPICS = 300
TOPIC_AUDIT_PER_TOPIC = 10     # số title/lớp lấy mẫu ngẫu nhiên để gán nhãn thủ công
TOPIC_TOKENIZER = os.environ.get("YTPIPE_TOPIC_TOKENIZER", "none").lower()  # none, underthesea hoặc pyvi
TOPIC_EMBEDDING_MODEL = os.environ.get("YTPIPE_TOPIC_EMBEDDING_MODEL", "").strip()  # rỗng = TF-IDF
TOPIC_EMBEDDING_BATCH_SIZE = int(os.environ.get("YTPIPE_TOPIC_EMBEDDING_BATCH_SIZE", "128"))

# ---- Clustering người dùng ----
MIN_USERS_CLUSTER = 4
MAX_K = 6
TOPIC_WEIGHT = 0.5             # trọng số nhóm đặc trưng chủ đề so với đặc trưng hành vi
MIN_EVENTS_USER = 50           # user ít hơn số này bị bỏ qua khi phân tích

# ---- Phân cụm phiên xem (song song với phân cụm người dùng) ----
MIN_SESSIONS_CLUSTER = 20      # tối thiểu số phiên để chọn k và ước lượng cụm
MAX_SESSION_K = 6
SESSION_CLUSTER_SAMPLE = 5000  # số phiên tối đa khi ước lượng silhouette
