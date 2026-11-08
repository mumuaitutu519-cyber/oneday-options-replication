# Optional raw reconstruction sources

The one-command replication uses only `../data/`; no original project, API key, paid subscription, exchange connection, or raw trade archive is needed to regenerate any manuscript table or figure.

The retained source code under `original_code/` is the transitive import closure of the actual R2/R4/R5/R9 extraction and measurement programs. It is reference material for reconstructing analysis-ready inputs from large raw archives, and is not invoked by the professor’s command. These network/download and minute/trade reconstruction stages were not rerun as part of the clean offline test. Raw extraction can be much larger and slower than the packaged analysis.

Deribit trades were obtained from the public historical JSON-RPC v2 endpoint:

- `https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time`
- Instrument metadata: `https://history.deribit.com/api/v2/public/get_instruments` with `currency`, `kind=option`, and `expired=true`.
- DVOL: `https://www.deribit.com/api/v2/public/get_volatility_index_data`.
- Perpetual chart bars, delivery prices and funding: the public methods in `original_code/r2_hedge_data.py`, `r3_options.py`, and `r5_measurement.py`.

`original_code/r2_download.py` is the exact historical-trade downloader used for the source archive. It requests `count=10000`, ascending timestamps, inclusive overlapping page boundaries and unique `trade_id` deduplication. It stops on stalled pagination instead of silently skipping trades. `API_EXTRACTION_PERIOD.json` records the successful extraction window, 2026-10-07 UTC. `RAW_SOURCE_MANIFEST.csv` retains all 6,356 complete daily files’ trade counts and SHA-256 values (BTC 2016-11-29 onward; ETH 2019-03-21 onward; both through 2026-10-06). No Tardis or other paid historical-trade source was used by this R2 archive.

Required trade fields include `trade_id`, `timestamp` (UTC milliseconds), `instrument_name`, `price` (coin units), `index_price`, `underlying_price` where available, `mark_price`, `direction`, `amount`, and block/combo identifiers. Instrument metadata supplies strike, expiry, option type and creation timestamp. Actual USD trade price is `price * index_price`; mark price is converted in the same way. Implied variance is solved with the retained Black inversion code. Inspect `r4_options.py` and `r5_options.py`: their forward-price conventions differ, and the supplied `data/` preserves the convention actually used in each manuscript result.

`recover_trade_aggregates.py` is the portable recovery program actually run during task 270. Given a local raw project with the archived daily files and instrument metadata, it independently regenerates `data/atm_trade_daily_by_side.parquet` with four processes:

```bash
ONEDAY_RAW_PROJECT=/path/to/raw_project ../.venv/bin/python recover_trade_aggregates.py
```

It retains daily side-specific valid-mark counts, sums, means, and IV medians, including the pre-2020 observations used in Table 9A. The other missing R4 intermediate, `term_panel`, is fully preserved as `trade_ATM` rows inside the supplied R5 premium panel.

Binance one-minute klines come from the public archive `https://data.binance.vision/` and its official archive layout `https://github.com/binance/binance-public-data`. Example: `https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2020-01.zip`. `download_binance.py` downloads monthly BTCUSDT/ETHUSDT perpetual and spot archives in four processes, streams files, and checks each published `.CHECKSUM`:

```bash
python download_binance.py --destination /path/to/large_temporary_storage --start 2019-12 --end 2026-10
```

The kline fields are open time, OHLC, base volume, close time, quote volume, number of trades, taker-buy base/quote volume, and ignore. Timestamps are normalized to UTC; spot archive timestamps may be in microseconds in later files. The original `01_binance.py` performs normalization, duplicate and missing-interval checks. `r5_measurement.py` documents boundary conventions, five-minute sampling, subsampling, Parzen-kernel estimation and delivery-price variants. `r9_prepare.py` documents bipower variation, truncated variation and BNS jump classification. `r5_strategy.py` documents entry legs, hourly Black deltas, perpetual hedge cashflows, funding, turnover and cost grids.

Reconstruction dependency order: public archives and instrument metadata → normalized minute prices and daily option trades → R3 grid/clock/event and delivery inputs → R4 expiry buckets and fixed-maturity panels → R5 alternative option/realized measures, straddle daily cashflows and forecasts → R9 realized decomposition and calendar flags. The packaged `code/tables.py::decomposition` is the recovered generator for the one-off grid-corrected R9 result; the original one-off command itself was not retained in the project. Its complete numerical result and exact original HAC helper are retained, and all manuscript cells are independently regenerated from daily observations.
