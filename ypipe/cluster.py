"""Stage 5b: Machine Learning - clustering người dùng (KMeans, chọn k theo silhouette) + PCA để vẽ."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config

BEHAV = ["videos_per_active_day", "active_day_ratio", "night_ratio", "weekend_boost", "sessions_per_active_day",
         "avg_videos_per_session", "repeat_ratio", "channel_top5_share", "channel_entropy", "topic_drift",
         "shorts_ratio", "long_ratio", "category_entropy"]
LABEL = {"videos_per_active_day": "Lượng xem/ngày", "active_day_ratio": "Tần suất ngày hoạt động",
         "night_ratio": "Xem ban đêm", "weekend_boost": "Tăng xem cuối tuần", "sessions_per_active_day": "Số phiên/ngày",
         "avg_videos_per_session": "Video/phiên", "repeat_ratio": "Xem lại", "channel_top5_share": "Trung thành kênh",
         "channel_entropy": "Đa dạng kênh", "topic_drift": "Thay đổi sở thích", "shorts_ratio": "Shorts",
         "long_ratio": "Video dài", "category_entropy": "Đa dạng thể loại"}


def run(uf):
    """Trả về dict(assign, profile, k, silhouette, features) hoặc None nếu chưa đủ người."""
    n = len(uf)
    if n < config.MIN_USERS_CLUSTER:
        return None
    beh = [c for c in BEHAV if c in uf and uf[c].notna().sum() >= max(3, 0.8 * n)]
    X = uf[beh].astype(float)
    X = X.fillna(X.median())
    beh = [c for c in beh if X[c].std() > 1e-9]
    Zb = StandardScaler().fit_transform(X[beh])
    tcols = [c for c in uf.columns if c.startswith("topic_") and c != "topic_drift"]
    Z = Zb
    if tcols:
        T = uf[tcols].fillna(0)
        T = T.loc[:, T.std() > 1e-9]
        if T.shape[1]:
            Z = np.hstack([Zb, StandardScaler().fit_transform(T) * config.TOPIC_WEIGHT])
    best = None
    for k in range(2, min(config.MAX_K, n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z)
        sil = silhouette_score(Z, km.labels_)
        if best is None or sil > best[0]:
            best = (sil, k, km.labels_)
    sil, k, lab = best
    pc = PCA(2, random_state=42).fit_transform(Z) if Z.shape[1] >= 2 else np.zeros((n, 2))
    assign = pd.DataFrame({"user_id": uf["user_id"].values, "cluster": lab, "pc1": pc[:, 0], "pc2": pc[:, 1]})
    zdf = pd.DataFrame(Zb, columns=beh)
    zdf["cluster"] = lab
    profile = zdf.groupby("cluster").mean()
    names = {}
    for c, row in profile.iterrows():
        up = [f"{LABEL[f]} ↑" for f in row.sort_values(ascending=False).index[:2] if row[f] > 0.3]
        dn = [f"{LABEL[f]} ↓" for f in row.sort_values().index[:1] if row[f] < -0.3]
        names[c] = " · ".join(up + dn) or "Hồ sơ trung bình"
    assign["cluster_name"] = assign["cluster"].map(lambda c: f"Cụm {c + 1}: {names[c]}")
    return {"assign": assign, "profile": profile, "k": k, "silhouette": float(sil), "features": beh}


SESSION_BEHAV = ["n_videos", "approx_minutes", "start_hour_sin", "start_hour_cos",
                 "shorts_ratio", "category_entropy"]


def run_sessions(session_features):
    """Phân cụm phiên độc lập với ``run``; trả None nếu chưa đủ phiên."""
    n = len(session_features)
    if n < config.MIN_SESSIONS_CLUSTER:
        return None
    features = [c for c in SESSION_BEHAV if c in session_features
                and session_features[c].notna().sum() >= max(3, 0.5 * n)]
    if not features:
        return None
    X = session_features[features].astype(float).copy()
    X = X.fillna(X.median())
    features = [c for c in features if X[c].std() > 1e-9]
    if not features:
        return None
    Z = StandardScaler().fit_transform(X[features])
    best = None
    max_k = min(config.MAX_SESSION_K, n - 1)
    for k in range(2, max_k + 1):
        model = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=42,
                                batch_size=min(4096, n)).fit(Z)
        if len(np.unique(model.labels_)) < 2 or len(np.unique(model.labels_)) >= n:
            continue
        sil = silhouette_score(Z, model.labels_,
                               sample_size=min(config.SESSION_CLUSTER_SAMPLE, n), random_state=42)
        if best is None or sil > best[0]:
            best = (float(sil), k, model.labels_)
    if best is None:
        return None
    sil, k, labels = best
    pc = PCA(2, random_state=42).fit_transform(Z) if Z.shape[1] >= 2 else np.zeros((n, 2))
    assign = session_features.copy()
    assign["session_cluster"] = labels
    assign["pc1"] = pc[:, 0]
    assign["pc2"] = pc[:, 1]
    return {"assign": assign, "k": k, "silhouette": sil, "features": features}


def session_cluster_shares(session_assignments):
    """Tỷ lệ mỗi người dùng có phiên thuộc từng loại phiên đã phân cụm."""
    if session_assignments is None or session_assignments.empty:
        return pd.DataFrame(columns=["user_id"])
    shares = (session_assignments.groupby(["user_id", "session_cluster"]).size()
              .unstack(fill_value=0))
    shares = shares.div(shares.sum(axis=1), axis=0)
    shares.columns = [f"session_type_{int(cluster_id)}" for cluster_id in shares.columns]
    return shares.reset_index()
