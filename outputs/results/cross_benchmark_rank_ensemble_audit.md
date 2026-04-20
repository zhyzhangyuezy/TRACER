# Cross-benchmark rank-ensemble stress audit

The rank ensemble is a calibration-free average of fixed-family prediction ranks. It is evaluated only from prediction exports and is not used to tune the released deterministic adaptive policy.

| Setting | Adaptive | Rank fixed | Delta | Rank+adaptive | Gain vs rank | Uniform | Single-family oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| ATLASv2 held-out-family | 0.763 +/- 0.041 | 0.847 +/- 0.079 | +0.084 | 0.859 +/- 0.067 | +0.012 | 0.738 | 0.764 |
| AIT-ADS chronology | 0.532 +/- 0.021 | 0.645 +/- 0.014 | +0.112 | 0.649 +/- 0.013 | +0.005 | 0.553 | 0.546 |
| AIT-ADS scenario-held-out | 0.450 +/- 0.013 | 0.578 +/- 0.010 | +0.129 | 0.582 +/- 0.009 | +0.004 | 0.450 | 0.454 |
| CAM-LDS chronology | 0.996 +/- 0.002 | 0.997 +/- 0.002 | +0.001 | 0.998 +/- 0.001 | +0.001 | 0.996 | 0.997 |
| CAM-LDS event-held-out | 0.945 +/- 0.022 | 0.951 +/- 0.008 | +0.006 | 0.954 +/- 0.007 | +0.002 | 0.927 | 0.971 |
