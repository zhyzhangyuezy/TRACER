# Cross-dataset transfer variants audit

Standard uses source train/dev plus optional target support once. Target-replay10 repeats the sampled target train/dev support windows ten times during training/model selection; target test windows are never used as support.

| Setting | Variant | Shot | DLinear | Prefix | TRACER | Best | TRACER route |
|---|---|---:|---:|---:|---:|---|---|
| AIT-ADS -> ATLASv2 | standard | 0 | 0.045 +/- 0.005 | 0.052 +/- 0.002 | 0.066 +/- 0.006 | TRACER (0.066) | dense_low_diversity |
| AIT-ADS -> ATLASv2 | standard | 5 | 0.050 +/- 0.008 | 0.058 +/- 0.009 | 0.066 +/- 0.006 | TRACER (0.066) | sparse_diverse |
| AIT-ADS -> ATLASv2 | target-replay10 | 5 | 0.074 +/- 0.018 | 0.070 +/- 0.012 | 0.071 +/- 0.012 | DLinear (0.074) | sparse_diverse |
| AIT-ADS -> ATLASv2 | standard | 20 | 0.054 +/- 0.011 | 0.053 +/- 0.005 | 0.085 +/- 0.019 | TRACER (0.085) | sparse_diverse |
| AIT-ADS -> ATLASv2 | target-replay10 | 20 | 0.069 +/- 0.010 | 0.065 +/- 0.018 | 0.100 +/- 0.009 | TRACER (0.100) | sparse_diverse |
| ATLASv2 -> AIT-ADS | standard | 0 | 0.051 +/- 0.007 | 0.086 +/- 0.007 | 0.058 +/- 0.004 | Prefix-Only (0.086) | sparse_diverse |
| ATLASv2 -> AIT-ADS | standard | 5 | 0.214 +/- 0.031 | 0.365 +/- 0.103 | 0.379 +/- 0.047 | TRACER (0.379) | sparse_diverse |
| ATLASv2 -> AIT-ADS | target-replay10 | 5 | 0.332 +/- 0.083 | 0.334 +/- 0.052 | 0.317 +/- 0.043 | Prefix-Only (0.334) | sparse_diverse |
| ATLASv2 -> AIT-ADS | standard | 20 | 0.255 +/- 0.110 | 0.366 +/- 0.043 | 0.311 +/- 0.051 | Prefix-Only (0.366) | sparse_diverse |
| ATLASv2 -> AIT-ADS | target-replay10 | 20 | 0.419 +/- 0.061 | 0.294 +/- 0.064 | 0.270 +/- 0.061 | DLinear (0.419) | sparse_diverse |
