# GitHub packaging notes (task 305)

Source: `OneDay_Replication_20261107.zip`, SHA-256 `ac9eefe8ece24590946c0d52c506ab5965406350e5e628227cc325e5472e331d`.

The 2026-11-06 authority is `OneDay_LEEKangRyu2026.11.06_tables_figures.docx`. The manuscript DOCX is omitted because the requested GitHub distribution permits manuscript PDFs only. The exact previously extracted table JSON, text and figure panels remain unchanged. `code/extract.py` now checks their pinned hashes when the DOCX is absent. This is the only calculation-entrypoint packaging adaptation; the scientific calculations, analysis data, saved outputs, and all existing verification CSVs are unchanged.

All 202 stated-definition numeric differences and seven significance-star differences are retained in `verify/`. The historical generating-version replay remains intact. This task did not rerun estimation or edit a manuscript/result. The clean/full runs and resource measurements in README are task-270 records.

The source package documents public Deribit historical API and Binance archive provenance. No paid Deribit/Tardis raw data or credentials are included. Offline wheels are retained as executable dependencies. No file exceeds 100 MB, so no Release attachment is necessary. The 170 MB source ZIP itself is not committed.

The current root MANIFEST and CHECKSUMS describe this GitHub package; the unchanged original ones are under `verify/original_270_delivery/`. `FILE_MANIFEST.csv` pins all tracked payloads except itself. See `EXCLUDED_FILES.md` and `SECURITY_SCAN.json`.
