# Mismatch report

No manuscript file was edited. No statistical estimate was changed to match a manuscript value.

|Table|Block / row / column / token|Manuscript|Recalculated at manuscript precision|Source result and keys|
|---|---|---:|---:|---|

All mapped manuscript numbers agree after rounding. There is no numerical correction list.

## Version and sample facts

ROUND10 is a separate variance-clock project and is not a source for these one-day-option tables. Tables 1–5 and 8–10 use ROUND4/5; Tables 6–7 use the grid-corrected ROUND9 decomposition. The original R4 term_panel and trade_mark intermediate files are no longer present at their recorded paths; the term panel was recovered from the preserved R5 trade_ATM rows, and trade/mark daily aggregates were recomputed from the original daily API trade archive. This package includes the recovered observations and the recovery code.

Daily one-day descriptive/premium rows in Tables 1 and 5 use fixed 09:00-to-09:00 interpolated maturities in the original result pipeline; forecast inputs for Table 2 use the actual 23-hour next expiry. This definition difference is disclosed in NOTE_VERIFICATION.csv. No statistic has been recomputed under a different window and presented as the original number.

## Table-note and sample corrections

These are definition/sample discrepancies; table statistics were not changed.

|Location|Manuscript scope|Scope actually used for the reproduced numbers|
|---|---|---|
|1 / Figure 1: IV / RV measurement window BTC|23 hours|24–24 hours; The archived R4 term panel uses fixed-maturity interpolation to 09:00 next day. Recovered rows are preserved in R5 trade_ATM.|
|9: Panel A sample begins BTC|2020-01 (global manuscript sample)|2017-11-16; Includes 89 days and 576 trades before 2020. Table values reproduce the wider source sample.|
|5: One-day rows A–C and Panel D maturity BTC|23 hours (equation (3) reference)|24 hours; The historical one-day term row and variance-unit premia use a fixed 24-hour interpolated panel, different from the one-day actual-expiry definition.|
|9: Panel B maturity BTC|23 hours (one-day definition; Panel B equation (3) reference)|24 hours in Panel B and annual premia in Panel C; Side-specific premium is calculated from the same interpolated fixed-day panel as Table 5.|
|1 / Figure 1: IV / RV measurement window ETH|23 hours|24–24 hours; The archived R4 term panel uses fixed-maturity interpolation to 09:00 next day. Recovered rows are preserved in R5 trade_ATM.|
|9: Panel A sample begins ETH|2020-01 (global manuscript sample)|2019-04-11; Includes 22 days and 47 trades before 2020. Table values reproduce the wider source sample.|
|5: One-day rows A–C and Panel D maturity ETH|23 hours (equation (3) reference)|24 hours; The historical one-day term row and variance-unit premia use a fixed 24-hour interpolated panel, different from the one-day actual-expiry definition.|
|9: Panel B maturity ETH|23 hours (one-day definition; Panel B equation (3) reference)|24 hours in Panel B and annual premia in Panel C; Side-specific premium is calculated from the same interpolated fixed-day panel as Table 5.|
