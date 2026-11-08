# One-day bitcoin and ether options replication

Authority: `OneDay_LEEKangRyu2026.11.06_tables_figures.docx`, Tables 1–10 and Figures 1–3. The manuscript has not been edited. No paid data-provider inputs were identified.

Unzip on a laptop with Python 3.12 and Bash, then run one command:

```bash
bash run_all.sh --offline
```

The command creates a virtual environment, installs pinned dependencies, recalculates every table and figure, and compares the displayed manuscript cells. Default **quick mode** uses archived daily forecast and VaR input paths, and recomputes losses, tests, regressions, MCS membership, bootstrap intervals and plots. **Full mode** also refits all 120 rolling-model specifications and rebuilds VaR residual quantiles:

```bash
bash run_all.sh --full --offline
```

Folders: `code/` final calculation programs; `data/` compressed analysis-ready observations; `raw_rebuild/` optional raw extraction programs and provenance; `output/` regenerated tables and figures; `verify/` manuscript comparisons; `logs/` execution records; `wheels/` offline dependencies. See `DATA_DICTIONARY.csv` for field definitions.

|Table / figure|Package script|Data|
|---|---|---|
|Table 1|`tables.py: tables_premia`|`premium_*`|
|Tables 2–3|`estimate.py`, `tables.py: tables_forecasts`|`features_*`, `forecast_inputs/`, `forecast_cache/`|
|Table 4|Same|`*matrix*` forecast inputs and paths|
|Table 5|`tables.py: tables_premia`|`premium_*`|
|Tables 6–7|`tables.py: decomposition`|`decomposition_*`, `numerical_grid_diagnostics_*`|
|Table 8|`tables.py: tables_trading`|`straddle_daily_*`, `strategy_availability_*`|
|Table 9|`tables.py: tables_trading`, `tables_premia`|`atm_trade_daily_by_side`, `trade_quality_*`, `premium_*`, `straddle_daily_*`|
|Table 10|`tables.py: tables_risk`, `estimate.py`|`risk_inputs_*`, forecast inputs and paths|
|Figures 1–3|`figures.py`|`premium_*`|

Scripts live under `code/`, data under `data/`. `verify/TABLE_FIGURE_LINEAGE.csv` gives the detailed **table/figure ↔ original generating version ↔ original result-file** map. ROUND10 concerns the separate variance-clock paper and is not an input to this manuscript.

Measured clean run: quick **61 s including a new offline virtual environment; 41 s rerun**, full **627 s (10 min 27 s) for 120 fresh refits; final definition reconciliation rerun 38 s**. Four worker processes are used, with one numerical-library thread per process. Quick mode is well within 30 minutes. Bundled wheels support **macOS Apple Silicon / Python 3.12**. On other platforms, Python 3.12 plus Bash and `bash run_all.sh` can install the same pinned packages from the internet; offline operation on those platforms has not been tested.

Read `output/TABLE_1.md`–`TABLE_10.md` and their CSVs, `output/FIGURE_COMPARISON.html`, and `verify/*_side_by_side.png`. Underlying figure CSVs are in `output/`. Numerical equality means rounding at each manuscript cell’s displayed precision. Numeric denominators and MCS text checks are reported separately.

The main outputs enforce the stated 23-hour one-day target and January 2020 sample start. This changes 202 numeric tokens and 7 significance-star displays, concentrated in Tables 1, 5 and 9. The historical generating-version replay is preserved in `output/source_version/` and `verify/source_version/`; all 1,460 original numeric tokens and 52 MCS cells match there. The original one-day descriptive/term inputs actually used interpolated 24-hour maturities, and Table 9A included pre-2020 trades. Exact replacement values are in `verify/MANUSCRIPT_CORRECTIONS.csv`, with reasons in `verify/MISMATCH_REPORT.md`. The stated-definition one-day mark measure uses the exchange-underlying forward; historical R5 mark inputs used the index as a forward proxy. Longer fixed-day maturities and the explicitly 24-hour decomposition remain as specified in their table notes. Full mode starts from daily variance and P&L observations; optional minute/trade reconstruction is separate.

`CHECKSUMS.sha256` records hashes. `MANIFEST.csv` records file sizes, purposes and corresponding tables/figures. Upstream exchange terms continue to apply.

## GitHub distribution

See `PACKAGING_NOTES.md` for the omitted manuscript DOCX, hash-verified extracted fixtures, and unchanged task-270 verification scope. Run `bash run_all.sh --offline` as before.
