"""Stage 5a: NLP - gom nhóm chủ đề từ title (TF-IDF + KMeans). Chạy trên toàn bộ video của mọi user.
Lưu ý: khi thêm người mới, mô hình được huấn luyện lại nên số thứ tự chủ đề có thể đổi; dùng cột label để đọc."""
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config

STOP = {"và", "của", "là", "có", "cho", "với", "trong", "một", "những", "các", "để", "the", "and", "of", "to", "in",
        "for", "with", "on", "is", "a", "an", "official", "video", "full", "hd", "tập", "mv", "ft", "feat"}
POSITIVE = {"hay", "tuyệt", "thích", "yêu", "vui", "đỉnh", "excellent", "great", "good", "love", "like", "amazing"}
NEGATIVE = {"dở", "tệ", "ghét", "buồn", "kém", "lỗi", "thất vọng", "bad", "hate", "boring", "worst", "dislike"}


def fit_topics(raw):
    t = raw[raw["title"].fillna("") != ""].drop_duplicates("video_id")[["video_id", "title"]]
    if len(t) < config.MIN_TITLES_FOR_TOPICS:
        print(f"  (chỉ {len(t)} title < {config.MIN_TITLES_FOR_TOPICS} -> bỏ qua topic modeling)")
        return None, None
    texts = t["title"].fillna("").astype(str)
    tokenizer = config.TOPIC_TOKENIZER
    if tokenizer == "underthesea":
        try:
            from underthesea import word_tokenize
        except ImportError as exc:
            raise RuntimeError("YTPIPE_TOPIC_TOKENIZER=underthesea nhưng package underthesea chưa được cài") from exc
        texts = texts.map(lambda value: word_tokenize(value, format="text"))
    elif tokenizer == "pyvi":
        try:
            from pyvi import ViTokenizer
        except ImportError as exc:
            raise RuntimeError("YTPIPE_TOPIC_TOKENIZER=pyvi nhưng package pyvi chưa được cài") from exc
        texts = texts.map(ViTokenizer.tokenize)
    elif tokenizer != "none":
        raise ValueError("YTPIPE_TOPIC_TOKENIZER chỉ nhận none, underthesea hoặc pyvi")

    token_pattern = (r"(?u)\b[^\W\d_]{2,}(?:_[^\W\d_]{2,})*\b" if tokenizer != "none"
                     else r"(?u)\b[^\W\d_]{2,}\b")
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.4, max_features=30000,
                          token_pattern=token_pattern, stop_words=list(STOP))
    embedding_model = None
    if config.TOPIC_EMBEDDING_MODEL:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Đã đặt YTPIPE_TOPIC_EMBEDDING_MODEL nhưng sentence-transformers chưa được cài") from exc
        embedding_model = SentenceTransformer(config.TOPIC_EMBEDDING_MODEL)
        X = embedding_model.encode(texts.tolist(), batch_size=config.TOPIC_EMBEDDING_BATCH_SIZE,
                                   show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        # Dùng TF-IDF riêng để tạo từ khóa mô tả cho các cụm embedding.
        label_X = vec.fit_transform(texts)
    else:
        X = vec.fit_transform(texts)
        label_X = X
    if label_X.shape[1] < config.N_TOPICS:
        return None, None
    km = MiniBatchKMeans(n_clusters=config.N_TOPICS, random_state=42, n_init=5, batch_size=4096).fit(X)
    terms = np.array(vec.get_feature_names_out())
    labels = []
    for i in range(config.N_TOPICS):
        if embedding_model is None:
            weights = km.cluster_centers_[i]
        else:
            members = label_X[km.labels_ == i]
            weights = np.asarray(members.mean(axis=0)).ravel() if members.shape[0] else np.zeros(len(terms))
        top = terms[np.argsort(weights)[::-1][:4]]
        labels.append({"topic_id": i, "label": ", ".join(top), "n_videos": int((km.labels_ == i).sum())})
    vt = pd.DataFrame({"video_id": t["video_id"].values, "topic_id": km.labels_})
    return vt, pd.DataFrame(labels)


def topic_shares(ev, vt):
    m = ev.merge(vt, on="video_id", how="inner")
    sh = m.groupby(["user_id", "topic_id"]).size().unstack(fill_value=0)
    sh = sh.div(sh.sum(axis=1), axis=0)
    sh.columns = [f"topic_{c}" for c in sh.columns]
    return sh.reset_index()


def topic_validation_sample(raw, video_topics, topic_labels):
    """Tạo phiếu lấy mẫu cho kiểm định thủ công của topic model không giám sát.

    KMeans không có nhãn chuẩn nên chưa thể tính Precision/Recall/F1 một cách hợp
    lệ. Người nghiên cứu cần điền ``manual_topic_label`` theo một taxonomy đã chốt;
    sau đó mới ánh xạ cụm sang taxonomy đó và tính các chỉ số phân loại ở ngoài
    pipeline khám phá này.
    """
    columns = ["video_id", "title", "predicted_topic_id", "predicted_topic_label",
               "manual_topic_label", "reviewer", "review_notes"]
    if video_topics is None or video_topics.empty or topic_labels is None or topic_labels.empty:
        return pd.DataFrame(columns=columns)
    titles = raw[["video_id", "title"]].drop_duplicates("video_id")
    labels = topic_labels.rename(columns={"topic_id": "predicted_topic_id", "label": "predicted_topic_label"})
    candidates = (video_topics.rename(columns={"topic_id": "predicted_topic_id"})
                  .merge(titles, on="video_id", how="left")
                  .merge(labels[["predicted_topic_id", "predicted_topic_label"]], on="predicted_topic_id", how="left"))
    # Không dùng GroupBy.apply để giữ được cột khóa trên cả pandas 2.0 lẫn bản mới.
    picks = [g.sample(min(len(g), config.TOPIC_AUDIT_PER_TOPIC), random_state=42)
             for _, g in candidates.groupby("predicted_topic_id")]
    sample = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame(columns=candidates.columns)
    sample["manual_topic_label"] = ""
    sample["reviewer"] = ""
    sample["review_notes"] = ""
    return sample[columns].sort_values(["predicted_topic_id", "title"])


def comment_sentiment(comments):
    """Lexicon nhẹ, minh bạch và không cần thêm model tải ngoài."""
    if comments is None or comments.empty:
        return pd.DataFrame(columns=["comment_id", "video_id", "sentiment", "sentiment_score"])
    def score(text):
        words = set(str(text).lower().replace(",", " ").replace(".", " ").split())
        return len(words & POSITIVE) - len(words & NEGATIVE)
    out = comments[["comment_id", "video_id", "text"]].copy()
    out["sentiment_score"] = out["text"].map(score)
    out["sentiment"] = np.select([out["sentiment_score"] > 0, out["sentiment_score"] < 0],
                                 ["positive", "negative"], default="neutral")
    return out.drop(columns=["text"])


def user_comment_signals(ev, comment_sent):
    """Gắn tín hiệu bình luận của các video đã xem vào user, không coi là cảm xúc cá nhân."""
    cols = ["comment_positive_share", "comment_negative_share", "comment_video_coverage"]
    if comment_sent is None or comment_sent.empty:
        return pd.DataFrame(columns=["user_id", *cols])
    watched = ev[["user_id", "video_id"]].drop_duplicates()
    m = watched.merge(comment_sent[["video_id", "sentiment"]], on="video_id", how="inner")
    if m.empty:
        return pd.DataFrame(columns=["user_id", *cols])
    out = m.groupby("user_id").agg(
        comment_positive_share=("sentiment", lambda s: float((s == "positive").mean())),
        comment_negative_share=("sentiment", lambda s: float((s == "negative").mean())),
        comment_video_coverage=("video_id", "nunique"),
    ).reset_index()
    out["comment_video_coverage"] /= watched.groupby("user_id")["video_id"].nunique().reindex(
        out["user_id"]).values
    return out
