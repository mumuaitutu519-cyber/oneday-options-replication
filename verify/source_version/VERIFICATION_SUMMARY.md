# Verification summary

Mode: quick; protocol: historical_generating_version. The 2026-11-06 DOCX is the sole displayed-value authority. All 10 tables are rebuilt from analysis-ready observations. Numeric equality uses decimal rounding at each manuscript cell’s displayed precision. Confidence-set text is also checked separately; it is excluded from the numeric denominators in the table below. Structural numbers in headings, variable definitions, and row labels are inventoried separately.

|Table|Matched|Checked|Rate|
|---|---:|---:|---:|
|1|48|48|100.00%|
|2|122|122|100.00%|
|3|324|324|100.00%|
|4|96|96|100.00%|
|5|300|300|100.00%|
|6|168|168|100.00%|
|7|98|98|100.00%|
|8|98|98|100.00%|
|9|106|106|100.00%|
|10|100|100|100.00%|

Unmapped estimated numbers: 0.

Quick mode uses archived dated forecast and VaR quantile paths, recalculates losses, tests, regressions, confidence sets, premia, bootstrap intervals, and all figures. Full mode refits every rolling model and reconstructs VaR inputs, and uses those new forecasts for the same checks. Both modes use the supplied daily variance and straddle P&L inputs; raw extraction is a separate reference workflow.

Table notes and sample definitions are checked separately in NOTE_VERIFICATION.csv. Figure data are saved in output/figure*_data.csv; manuscript images and regenerated images appear side by side in output/FIGURE_COMPARISON.html.
