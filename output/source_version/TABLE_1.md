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
| IV |  | 2422 | 52.69 | 23.00 | 12.22 | 37.20 | 48.33 | 63.89 | 247.0 |
| RV |  | 2422 | 54.09 | 35.85 | 5.697 | 33.72 | 46.36 | 64.17 | 689.7 |
| VP |  | 2422 | -1.396 | 25.23 | -607.2 | -6.182 | 2.655 | 9.461 | 69.29 |
| Ether |
| IV |  | 2323 | 65.11 | 28.67 | 13.53 | 46.95 | 60.53 | 77.55 | 294.8 |
| RV |  | 2323 | 69.94 | 46.69 | 6.492 | 44.67 | 60.21 | 82.18 | 1044.2 |
| VP |  | 2323 | -4.826 | 34.27 | -882.5 | -10.95 | 0.470 | 9.424 | 146.9 |