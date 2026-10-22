"""Stage 4: Storage - DuckDB (1 file, miễn phí, không cần server). Có thể thay bằng PostgreSQL ở đây."""
import duckdb
import pandas as pd
from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS watch_raw(
  user_id VARCHAR, watched_at TIMESTAMP, video_id VARCHAR, title VARCHAR,
  channel_name VARCHAR, channel_id VARCHAR, product VARCHAR);
CREATE TABLE IF NOT EXISTS ingested_files(
  sha256 VARCHAR PRIMARY KEY, user_id VARCHAR, filename VARCHAR,
  rows_kept INTEGER, rows_new INTEGER, ingested_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS video_metadata(
  video_id VARCHAR PRIMARY KEY, status VARCHAR, title VARCHAR, description VARCHAR,
  channel_id VARCHAR, category_id INTEGER, category_name VARCHAR, published_at TIMESTAMP,
  duration_sec DOUBLE, view_count BIGINT, like_count BIGINT, comment_count BIGINT,
  tags VARCHAR, default_language VARCHAR, fetched_at TIMESTAMP,
  made_for_kids BOOLEAN, status_fetched_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS channel_metadata(
  channel_id VARCHAR PRIMARY KEY, subscriber_count BIGINT, video_count BIGINT,
  fetched_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS video_comments(
  comment_id VARCHAR PRIMARY KEY, video_id VARCHAR, author VARCHAR,
  text VARCHAR, published_at TIMESTAMP, like_count BIGINT);
"""


def connect():
    config.DATA.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.DB_PATH))
    con.execute(SCHEMA)
    # DuckDB files created by earlier versions already have video_metadata.
    # Add new fields in place so existing warehouses remain usable.
    columns = {row[0] for row in con.execute("DESCRIBE video_metadata").fetchall()}
    if "made_for_kids" not in columns:
        con.execute("ALTER TABLE video_metadata ADD COLUMN made_for_kids BOOLEAN")
    if "status_fetched_at" not in columns:
        con.execute("ALTER TABLE video_metadata ADD COLUMN status_fetched_at TIMESTAMP")
    return con


def insert_df(con, table, df):
    cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    df = df[cols]
    con.register("_df", df)
    con.execute(f"INSERT INTO {table} SELECT * FROM _df")
    con.unregister("_df")


def replace_table(con, table, df):
    con.register("_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _df")
    con.unregister("_df")
    config.PARQUET.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PARQUET / f"{table}.parquet", index=False)


def export_parquet(con, table):
    """Xuất snapshot Parquet của bảng đã được cập nhật theo từng lô."""
    df = con.execute(f"SELECT * FROM {table}").df()
    config.PARQUET.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PARQUET / f"{table}.parquet", index=False)


def read(con, sql):
    return con.execute(sql).df()
