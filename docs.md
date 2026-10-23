# Brief cho AI agent: mở rộng pipeline `ytpipe` (phân tích lịch sử xem YouTube)

## 1. Bối cảnh

- Đồ án môn **Tích hợp và phân tích dữ liệu lớn**, theo 6 stage: Raw Data → Ingestion → ETL/ELT → Storage → Machine Learning → Visualization.
- Dữ liệu: file lịch sử xem YouTube từ **Google Takeout** (JSON), 10+ người, mỗi người khoảng 5–9 tháng, ≥ ~50.000 dòng; bổ sung metadata từ **YouTube Data API v3**.
- Hướng phân tích: **mô tả và khám phá** (không dự đoán), kèm phân cụm người dùng.
- Ràng buộc: công nghệ **miễn phí và linh hoạt** (hiện dùng DuckDB + Parquet), pipeline kiểu **"thả file mới vào `inbox/` là chạy"**, comment bị tắt mặc định (quota + riêng tư), `user_id` phải được ẩn danh khi đưa vào báo cáo.
- Ngôn ngữ: tài liệu, nhãn biểu đồ và nhận xét viết **tiếng Việt**; tên biến/cột dùng tiếng Anh như code hiện có.

## 2. Hiện trạng code (helper package `ypipe` và notebook pipeline)

| Module | Vai trò |
|---|---|
| `config.py` | Mọi tham số chỉnh được (TZ, ngưỡng phiên, N_TOPICS, MAX_K, …). Tham số mới phải thêm vào đây. |
| `ingest.py` | Đọc `.json`/`.zip` Takeout, lọc quảng cáo/không phải video, khử trùng lặp, nạp `watch_raw`. |
| `enrich.py` | Gọi `videos.list` (50 id/lần), cache trong `video_metadata`, dừng êm khi hết quota; có chế độ `YTPIPE_MOCK_API`. |
| `notebooks/00_pipeline.ipynb` | Định nghĩa DuckDB store, report HTML và điều phối pipeline; đây là entry point thay CLI. |
| `features.py` | `build_events`, `build_sessions`, `user_features`, các bảng profile theo giờ/ngày/tháng, `channel_summary`, `category_summary`, `topic_transitions`. |
| `nlp.py` | TF-IDF + MiniBatchKMeans trên title (`fit_topics`), phiếu kiểm định thủ công, sentiment comment bằng lexicon. |
| `cluster.py` | KMeans phân cụm **người dùng**, chọn k theo silhouette, PCA để vẽ. |

Bảng hiện có trong DuckDB: `watch_raw`, `ingested_files`, `video_metadata`, `video_comments`, `user_features`, `sessions`, `session_sensitivity`, `user_hourly`, `daily_profile`, `monthly_profile`, `channel_summary`, `category_summary`, `topic_transitions`, `user_clusters`, `topics`, `topic_validation_sample`, `comment_sentiment`.

**Các khoảng trống đã xác định (cần xử lý trước khi viết notebook):**

1. `features.build_events` chỉ merge `category_name`, `duration_sec`, `like_count`, `view_count`. Còn thiếu `published_at`, `comment_count`, `tags`, `default_language`, `channel_id`.
2. `video_id → topic_id` (biến `vt` trong pipeline notebook) được lưu thành bảng `video_topics`; notebook phân tích đọc `topics.label` để hiển thị nhãn.
3. Không có bảng mức **video** (mỗi `video_id` một dòng). Bảng sự kiện lặp video theo lượt xem nên làm lệch phân tích video.
4. `enrich.py` gọi `part=snippet,contentDetails,statistics`: chưa có `status` (cờ *made for kids*) và chưa có số subscriber của kênh (`channels.list`).
5. Tokenize TF-IDF theo âm tiết (`token_pattern`), chưa tách từ tiếng Việt.

## 3. Việc cần làm ở pipeline (Stage 3–4), theo thứ tự

> Yêu cầu chung: **không phá hành vi hiện có**; mọi thay đổi schema phải tương thích với DuckDB đã có dữ liệu (dùng `ALTER TABLE … ADD COLUMN` hoặc cơ chế migrate; không bắt chạy lại từ đầu). Chế độ mock API vẫn phải chạy được.

**P1. Mở rộng enrichment**
- Thêm `part=status` để lấy `madeForKids` vào `video_metadata`.
- Thêm bảng `channel_metadata` (`channel_id`, `subscriber_count`, `video_count`, `fetched_at`) qua `channels.list` (50 id/lần), có cache và dừng êm khi hết quota như `enrich`. `subscriber_count` có thể bị ẩn → để NULL, không đặt 0.
- Chỉ gọi API cho dữ liệu còn thiếu (backfill các video cũ chưa có cột mới), tôn trọng `--max-api-calls`.

**P2. Lưu gán topic cho video**
- Trong `pipeline.analyze`, lưu `vt` thành bảng `video_topics` (`video_id`, `topic_id`) bằng `store.replace_table`. Ghi chú rằng `topic_id` có thể đổi khi thêm người mới (mô hình huấn luyện lại); notebook phải đọc cột `label` ở `topics`.

**P3. Bảng mức video `video_facts`** (một dòng mỗi `video_id`)
- Cột gợi ý: `video_id`, `title`, `channel_id`, `channel_name`, `category_name`, `duration_sec`, `published_at` (đã đổi sang `config.TZ`), `pub_hour`, `pub_dow`, `view_count`, `like_count`, `comment_count`, `fetched_at`, `age_days` (= `fetched_at` − `published_at`), `views_per_day`, `n_watch_events`, `n_users`, `topic_id`, `made_for_kids`, `subscriber_count`.
- `views_per_day` dùng `max(age_days, 1)` để tránh chia cho 0.
- Thêm đặc trưng tiêu đề: `title_len_chars`, `title_len_words`, `has_number`, `has_question`, `has_exclaim`, `upper_ratio`, `has_emoji`, `has_brackets`, `has_hashtag`, `lang_guess`.
- Dùng cùng quy ước NULL: `like_count` ẩn → NULL.

**P4. Cải tiến NLP (tùy chọn, bật/tắt bằng `config`)**
- Tùy chọn tách từ tiếng Việt (`underthesea` hoặc `pyvi`) và tùy chọn embedding đa ngôn ngữ (sentence-transformers) thay cho TF-IDF. Mặc định giữ hành vi hiện tại; notebook sẽ so sánh các cách.
- Cho phép chọn `N_TOPICS` qua tham số; không ghi cứng 12 trong notebook.

**P5. Phân cụm theo phiên (Stage 5)**
- Thêm hàm phân cụm **session** (đặc trưng: số video, thời lượng phiên, giờ bắt đầu, tỷ lệ Shorts, đa dạng thể loại…) và bảng `session_clusters`. Biểu diễn mỗi người dùng bằng **tỷ lệ các loại phiên**.
- Giữ nguyên `cluster.run` (phân cụm người dùng) song song để so sánh.

## 4. Notebook (`notebooks/*.ipynb`)

**Quy ước chung**
- `notebooks/00_pipeline.ipynb` là entry point để nạp Takeout, gọi enrichment qua `enrich.py`, ghi DuckDB/Parquet và tạo báo cáo. Store, report và điều phối pipeline được định nghĩa trong notebook này; các helper tái sử dụng như ingest/enrich/features/NLP/clustering vẫn ở `ypipe/`.
- Các notebook phân tích `00_data_overview`–`06_summary_dashboard` chỉ đọc DuckDB/Parquet, không gọi API, và đóng connection khi hoàn tất. Không chạy pipeline đồng thời khi notebook phân tích còn mở database.
- Notebook phân tích: cell đầu đọc config, mở DuckDB read-only, đặt seed 42; notebook pipeline đọc config/seed trước, mở DuckDB trong hàm run và luôn đóng bằng `finally`.
- Mỗi notebook phân tích có đủ 5 mục: **Câu hỏi → Phương pháp → Kết quả → Insight → Hạn chế**; notebook pipeline mô tả cách chạy, đầu ra và hạn chế dữ liệu.
- Ngưỡng (số view tối thiểu, số video tối thiểu mỗi combo, số cụm…) khai báo thành biến ở đầu notebook, giải thích lý do chọn.
- Biểu đồ: nhãn tiếng Việt, có tiêu đề và đơn vị; không dùng `user_id` thật trong hình (dùng U1, U2, …).
- Không kết luận quan hệ nhân quả từ tương quan.

**00_data_overview** (Stage 1–4)
- Số dòng, số user, khoảng thời gian, độ phủ metadata (`status='ok'` so với `unavailable`), tỷ lệ video bị xóa/thiếu, phân bố theo thể loại. Báo cáo rủi ro lệch mẫu do video không còn tồn tại.

**01_title_nlp** (Stage 5)
- Làm sạch title, vector hóa, KMeans topic; chọn số topic bằng nhiều giá trị (silhouette hoặc tiêu chí khác).
- Đối chiếu topic với `category_name` (API) coi như **nhãn yếu**: bảng chéo topic × thể loại, purity, NMI, ARI. Nêu rõ đây không phải nhãn chuẩn.
- Phân tích các cụm title không khớp thể loại; so sánh TF-IDF với biến thể tách từ/embedding nếu có.

**02_video_performance** (Stage 5)
- Giờ đăng: heatmap thứ × giờ đăng (theo TZ Việt Nam), tách theo thể loại; độ trễ giữa `published_at` và giờ xem của người dùng, theo thể loại.
- Lượt xem: chuẩn hóa bằng `views_per_day` và xếp hạng phân vị **trong cùng thể loại**; dùng log vì phân bố lệch mạnh.
- Tiêu đề nhiều view vs ít view: chỉ xét video vượt ngưỡng view tối thiểu; so sánh đặc trưng tiêu đề (Mann-Whitney hoặc hồi quy log(views/ngày) có kiểm soát thể loại và tuổi video); từ/cụm từ đặc trưng cho nhóm cao/thấp bằng log-odds ratio.

**03_category_combos** (Stage 5)
- Combo trong phiên: Apriori/FP-Growth trên "giỏ" thể loại của mỗi phiên; xếp hạng theo support, confidence, lift.
- Combo thuộc tính (thể loại × độ dài × khung giờ đăng): xếp theo trung vị `views_per_day`, chỉ giữ combo có ≥ N video.

**04_user_segments** (Stage 5)
- KMeans người dùng: so sánh silhouette, Davies–Bouldin, elbow; đánh giá **độ ổn định** (nhiều seed, bootstrap, ARI giữa các lần).
- Phân cụm theo phiên (P5) và mô tả người dùng bằng tỷ lệ các loại phiên.
- "Nội dung cho khách hàng nào": heatmap cụm × (thể loại/topic/kênh/độ dài) bằng lift; mô tả hồ sơ từng cụm. Nói rõ "khách hàng" = cụm người dùng, vì API không có nhân khẩu học người xem.

**05_trends** (Stage 5)
- Chuỗi theo tháng/tuần: tỷ trọng thể loại và topic, tỷ lệ Shorts, độ dài trung bình, tỷ lệ xem đêm, số phiên/ngày; chuẩn hóa theo số người dùng đang hoạt động.
- Kiểm tra xu hướng (hồi quy hoặc Mann-Kendall), tách chu kỳ tuần (STL), thử phát hiện điểm đổi. Nêu giới hạn: chỉ 5–9 tháng nên không kết luận được mùa vụ theo năm.

**06_summary_dashboard** (Stage 6)
- Tổng hợp biểu đồ chính và kết luận; liên hệ ngược với 6 stage và phần "dữ liệu lớn" (tích hợp Takeout + API, DuckDB/Parquet, khối lượng ~500k+ sự kiện, hướng mở rộng khi tăng số người dùng).

## 5. Lưu ý về dữ liệu (phải nêu trong notebook)

- `view_count` là **ảnh chụp tại `fetched_at`**, không phải lượt xem lúc người dùng xem.
- Phân tích video chỉ trên **video mà người dùng đã xem** (và còn tồn tại), không đại diện cho toàn YouTube.
- Video đã xóa không có metadata → mẫu bị lệch; báo cáo tỷ lệ này.
- Khi phân tích mức video, dùng bảng `video_facts` (mỗi video một dòng); nếu cần trọng số theo lượt xem thì ghi rõ.
- Với 10+ người dùng, kết quả phân cụm người dùng dễ thiếu ổn định; luôn kèm chỉ số ổn định.

## 6. Tiêu chí hoàn thành

- Chạy pipeline với `inbox/` rỗng hoặc dữ liệu cũ vẫn không lỗi; chế độ mock API vẫn chạy.
- Các bảng mới (`video_topics`, `video_facts`, `channel_metadata`, `session_clusters`) được tạo và xuất Parquet.
- Mỗi notebook chạy tuần tự từ đầu đến cuối (Restart & Run All) không lỗi trên dữ liệu hiện có.
- Mọi ngưỡng và tham số nằm ở `config.py` hoặc ở đầu notebook.
- Có `README` ngắn mô tả thứ tự chạy: notebook pipeline → notebook 00 → 06.

## 7. Điều agent **không** được làm

- Không tự ý đổi cách tính các đặc trưng hiện có trong `user_features` hoặc `cluster.BEHAV` (chỉ thêm mới).
- Không bật lấy comment mặc định, không in `user_id` thật hoặc tên người dùng vào báo cáo.
- Notebook pipeline chỉ gọi API qua `enrich.py`; notebook phân tích không gọi API.
- Không dùng kết quả "nhãn yếu" (thể loại API) như thể là độ chính xác phân loại thật.
- Khi gặp điểm chưa rõ (ví dụ ngưỡng view tối thiểu), nêu giả định rõ ràng thay vì im lặng chọn.

## 8. Thứ tự đề xuất

P1 → P2 → P3 → notebook pipeline → notebook 00 → 02 → 01 → 03 → P5 → 04 → 05 → 06; P4 làm sau cùng nếu còn thời gian.

## 9. Dashboard HTML và dữ liệu xuất

- `reports/insight_u01.html` là bản insight cá nhân cho `u01`, được sinh từ warehouse. Các thẻ video dùng `video_facts`; khi metadata hoặc số lượng video chưa đủ, trang hiện trạng thái thiếu dữ liệu thay vì số minh họa.
- `ypipe/templates/insight_user.html` là mẫu dùng chung; `ypipe/insight.py` dựng báo cáo cho từng người từ `watch_raw`, `sessions`, `session_sensitivity`, `user_features`, `user_clusters` và `video_facts`. Notebook 06 xuất `reports/uXX.html` cho mọi người và `reports/insight_u01.html` làm bản u01 tương thích với liên kết hiện có.
- Có thể tạo riêng báo cáo cá nhân bằng `python -m ypipe.insight --db data/warehouse.duckdb --template ypipe/templates/insight_user.html --out-dir reports`.
- `ypipe/templates/group_dashboard.html` là mẫu nhóm; notebook 06 dựng `reports/group_dashboard.html` và nhúng dữ liệu thật vào trang. Nút **Nạp group.json** vẫn cho phép thay payload bằng snapshot khác. Có thể xuất JSON riêng bằng `python -m ypipe.dashboard --db data/warehouse.duckdb --out reports/group.json`.
- Hàm xuất nằm trong `ypipe/dashboard.py`, chỉ mở DuckDB ở chế độ read-only và chỉ đưa mã ẩn danh `U01`, `U02`, … vào JSON. Không xuất `user_id`, video ID hay tên người xem.
- Cấu trúc JSON: `period` (chuỗi giai đoạn), `sil` (silhouette), `users` (`id`, `n`, `night`, `spd`, `cl`, `x`, `y`), `cats`, `clusters` (`name`, `traits`, `hour` 24 giá trị, `lift` cùng thứ tự `cats`), `hm`/`pub` (ma trận thứ × giờ 7×24), `months`, `trend` (tỷ trọng lượt mở theo thể loại/tháng), `title` (đặc điểm tiêu đề và tỷ số views/ngày trong cùng thể loại), và `combos` (cặp thể loại trong phiên, lift, views/ngày trung vị).
- `hm`, nhịp giờ theo cụm, thể loại và xu hướng tháng được tổng hợp từ `watch_raw`; giờ đăng, đặc điểm tiêu đề và combo cần `video_facts`. Metadata không có sẽ không được suy đoán: giờ đăng hiển thị 0, kết quả tiêu đề/combo không đủ dữ liệu sẽ để danh sách rỗng, thể loại thiếu được gom thành `Không rõ`.
- Xu hướng thể loại trong JSON lấy 12 tháng gần nhất để biểu đồ đọc được; `period` vẫn ghi toàn bộ khoảng thời gian hiện có trong kho.
- Để dựng lại dashboard từ warehouse, chạy notebook `06_summary_dashboard`; cell cuối xuất các báo cáo cá nhân, `reports/group_dashboard.html` và `reports/group.json`. Các tệp HTML/JSON là snapshot dữ liệu tại thời điểm chạy.
- Cả hai dashboard là báo cáo tĩnh. Không mở YouTube API khi dựng dashboard; enrichment chỉ chạy trong notebook pipeline.
