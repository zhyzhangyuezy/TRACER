# Fixed-budget alert stability audit

Scores are converted into a fixed review budget without tuning a threshold on the test split. Episodes are contiguous selected windows within the same incident after timestamp sorting. Stable incident recall requires at least two consecutive selected windows containing a positive forecast-horizon window.

Primary table uses a 5% review budget and a 2-window confirmation rule.

| Setting | Method | AUPRC | IncRec | StableIncRec | StableLead | Episodes/100 | FalseEpisodes/100 |
|---|---|---:|---:|---:|---:|---:|---:|
| AIT-ADS chronology | Adaptive | $0.532\pm0.021$ | $1.000\pm0.000$ | $1.000\pm0.000$ | $25.4$ | $0.948\pm0.083$ | $0.353\pm0.076$ |
| AIT-ADS chronology | DLinear | $0.512\pm0.014$ | $1.000\pm0.000$ | $1.000\pm0.000$ | $23.4$ | $1.104\pm0.075$ | $0.449\pm0.067$ |
| AIT-ADS chronology | Prefix-Only | $0.530\pm0.022$ | $1.000\pm0.000$ | $1.000\pm0.000$ | $25.6$ | $0.982\pm0.103$ | $0.394\pm0.092$ |
| AIT-ADS chronology | Transformer | $0.536\pm0.020$ | $1.000\pm0.000$ | $1.000\pm0.000$ | $25.1$ | $0.977\pm0.086$ | $0.381\pm0.069$ |
| AIT-ADS scenario-held-out | Adaptive | $0.450\pm0.013$ | $1.000\pm0.000$ | $1.000\pm0.000$ | $24.2$ | $1.103\pm0.164$ | $0.396\pm0.089$ |
| AIT-ADS scenario-held-out | DLinear | $0.450\pm0.013$ | $1.000\pm0.000$ | $1.000\pm0.000$ | $24.2$ | $1.103\pm0.164$ | $0.396\pm0.089$ |
| AIT-ADS scenario-held-out | Prefix-Only | $0.416\pm0.019$ | $1.000\pm0.000$ | $0.988\pm0.054$ | $22.3$ | $1.464\pm0.361$ | $0.559\pm0.140$ |
| AIT-ADS scenario-held-out | TCN | $0.435\pm0.016$ | $1.000\pm0.000$ | $1.000\pm0.000$ | $23.2$ | $1.307\pm0.198$ | $0.465\pm0.070$ |
| ATLASv2 held-out-family | Adaptive | $0.763\pm0.041$ | $1.000\pm0.000$ | $1.000\pm0.000$ | $20.0$ | $2.590\pm0.575$ | $1.386\pm0.575$ |
| ATLASv2 held-out-family | DLinear | $0.723\pm0.037$ | $1.000\pm0.000$ | $1.000\pm0.000$ | $17.5$ | $2.952\pm0.599$ | $1.747\pm0.599$ |
| ATLASv2 held-out-family | LSTM | $0.741\pm0.038$ | $1.000\pm0.000$ | $1.000\pm0.000$ | $18.0$ | $2.892\pm0.590$ | $1.687\pm0.590$ |
| ATLASv2 held-out-family | Prefix-Only | $0.669\pm0.121$ | $1.000\pm0.000$ | $0.950\pm0.218$ | $18.2$ | $2.651\pm1.240$ | $1.386\pm1.221$ |
