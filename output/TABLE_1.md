# Table 1 Panel A. Variable definitions

Mode: quick. Statistics are recomputed from the supplied daily observations.

| C0 | C1 | C2 |
|---|---|---|
|  |  | Definition |
| IV |  | One-day implied volatility. Square root of the annualized one-day implied variance in equation (1), times 100. The implied variance uses at-the-money trades between 08:00 and 09:00 UTC in the option series that expires at 08:00 UTC the next day. |
| RV |  | Realized volatility. Square root of the annualized realized variance in equation (2), times 100. The realized variance is the sum of squared five-minute log returns of the perpetual futures price from 09:00 UTC to 08:00 UTC the next day. |
| VP |  | Variance premium in volatility points. IV minus RV. |

# Table 1 Panel B. Descriptive statistics

Mode: quick. Statistics are recomputed from the supplied daily observations.

| C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 |
|---|---|---|---|---|---|---|---|---|---|
|  |  | Obs | Mean | SD | Min | P25 | P50 | P75 | Max |
| Bitcoin |
| IV |  | 2423 | 52.62 | 23.13 | 11.09 | 37.15 | 48.35 | 63.98 | 247.9 |
| RV |  | 2423 | 54.26 | 36.06 | 5.760 | 33.88 | 46.83 | 64.13 | 694.6 |
| VP |  | 2423 | -1.646 | 25.54 | -611.9 | -6.589 | 2.251 | 9.211 | 66.90 |
| Ether |
| IV |  | 2373 | 65.48 | 29.09 | 12.36 | 46.92 | 60.98 | 78.15 | 296.2 |
| RV |  | 2373 | 70.51 | 47.08 | 6.557 | 45.03 | 60.87 | 82.91 | 1065.1 |
| VP |  | 2373 | -5.032 | 34.71 | -902.6 | -11.14 | 0.091 | 9.309 | 152.6 |