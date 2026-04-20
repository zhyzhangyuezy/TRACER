# Cross-dataset transfer and few-shot adaptation audit

All datasets are projected to the shared 15 current-prefix feature channels and z-scored using the constructed train split only. Few-shot support is sampled from the target-domain train/dev splits; target test windows are never sampled as support.

| Setting | Shot positives | Method | AUPRC | AUROC | Brier | AF@5 | Route regime |
|---|---:|---|---:|---:|---:|---:|---|
| AIT-ADS -> ATLASv2 | 5 | DLinear + replay10 | 0.074 +/- 0.018 | 0.462 +/- 0.024 | 0.142 +/- 0.009 | -- | -- |
| AIT-ADS -> ATLASv2 | 5 | Prefix-Only + replay10 | 0.070 +/- 0.012 | 0.540 +/- 0.134 | 0.114 +/- 0.050 | 65.4 +/- 7.3 | -- |
| AIT-ADS -> ATLASv2 | 5 | TRACER adaptive + replay10 | 0.071 +/- 0.012 | 0.637 +/- 0.065 | 0.184 +/- 0.054 | 58.0 +/- 6.6 | sparse_diverse |
| AIT-ADS -> ATLASv2 | 20 | DLinear + replay10 | 0.069 +/- 0.010 | 0.527 +/- 0.030 | 0.130 +/- 0.040 | -- | -- |
| AIT-ADS -> ATLASv2 | 20 | Prefix-Only + replay10 | 0.065 +/- 0.018 | 0.522 +/- 0.137 | 0.122 +/- 0.017 | 59.1 +/- 17.1 | -- |
| AIT-ADS -> ATLASv2 | 20 | TRACER adaptive + replay10 | 0.100 +/- 0.009 | 0.780 +/- 0.017 | 0.121 +/- 0.036 | 63.0 +/- 9.1 | sparse_diverse |
| ATLASv2 -> AIT-ADS | 5 | DLinear + replay10 | 0.332 +/- 0.083 | 0.771 +/- 0.071 | 0.091 +/- 0.055 | -- | -- |
| ATLASv2 -> AIT-ADS | 5 | Prefix-Only + replay10 | 0.334 +/- 0.052 | 0.809 +/- 0.004 | 0.054 +/- 0.003 | 70.1 +/- 3.2 | -- |
| ATLASv2 -> AIT-ADS | 5 | TRACER adaptive + replay10 | 0.317 +/- 0.043 | 0.787 +/- 0.032 | 0.052 +/- 0.008 | 75.6 +/- 7.9 | sparse_diverse |
| ATLASv2 -> AIT-ADS | 20 | DLinear + replay10 | 0.419 +/- 0.061 | 0.802 +/- 0.056 | 0.052 +/- 0.011 | -- | -- |
| ATLASv2 -> AIT-ADS | 20 | Prefix-Only + replay10 | 0.294 +/- 0.064 | 0.774 +/- 0.041 | 0.068 +/- 0.014 | 78.1 +/- 1.9 | -- |
| ATLASv2 -> AIT-ADS | 20 | TRACER adaptive + replay10 | 0.270 +/- 0.061 | 0.799 +/- 0.031 | 0.060 +/- 0.015 | 83.7 +/- 4.6 | sparse_diverse |
