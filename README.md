# YouTube watch history notebooks

## Set up and run

Use the Jupyter kernel you will run for these notebooks, then install dependencies there:

```python
%pip install -r requirements.txt
```

Restart the kernel after installation. The pipeline notebook applies a default cap of 100 API batch calls per run; raise it only when you intend to spend more quota.

1. Put Google Takeout `.json` or `.zip` exports in `inbox/`.
2. Open `notebooks/00_pipeline.ipynb` and run all cells. Review `ENRICH_API`, `MAX_API_CALLS`, and `FORCE` before running. The API key is read from `.env`; for offline runs set `ENRICH_API = False` or configure `YTPIPE_MOCK_API`.
3. The notebook persists the warehouse at `data/warehouse.duckdb`, exports table snapshots to `data/parquet/`, and writes HTML reports under `reports/`.
4. Run the analysis notebooks in order: `00_data_overview`, `02_video_performance`, `01_title_nlp`, `03_category_combos`, `04_user_segments`, `05_trends`, `06_summary_dashboard`.

The pipeline notebook contains the DuckDB store, report generation, and pipeline orchestration. Reusable ingestion, YouTube enrichment, feature, NLP, and clustering helpers remain in `ypipe/`. It closes its DuckDB connection even if a step fails, reloads current helper code when run, and defaults to 100 API batch calls per run. Analysis notebooks use short-lived read-only connections and release each DuckDB file handle after every query. A kernel from an older notebook version can still retain an existing connection; restart that kernel or run `con.close()` once to release it.

`python -m ypipe` is retired. Run the pipeline from `notebooks/00_pipeline.ipynb`.
