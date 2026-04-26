# Policy-v2 tabular admission audit

Choose the tabular recipe with the largest development AUPRC within each dataset, then compare its held-out target split to the released policy. Ex-post best rows are reported only as test-only oracle diagnostics.

| Setting | Released policy | Dev-best tabular | Dev AUPRC | Target AUPRC | Ex-post best | Ex-post target | Read |
|---|---:|---|---:|---:|---|---:|---|
| ATLASv2 chronology | 0.677 | PrefixStats-RandomForest | 0.300 | 0.546 | FlatPrefix-Logistic | 0.866 | keep policy; oracle gain not dev-supported |
| ATLASv2 held-out-family | 0.763 | PrefixStats-RandomForest | 0.300 | 0.238 | FlatPrefix-Logistic | 0.663 | keep policy |
| AIT-ADS chronology | 0.532 | PrefixStats-Logistic | 0.278 | 0.426 | FlatPrefix-HistGB | 0.544 | keep policy; oracle gain not dev-supported |
| AIT-ADS held-out-scenario | 0.450 | PrefixStats-Logistic | 0.278 | 0.360 | FlatPrefix-HistGB | 0.409 | keep policy |
| ATLAS-Raw chronology | 0.461 | PrefixStats-Logistic | 0.003 | 0.087 | FlatPrefix-HistGB | 0.169 | keep policy |
| ATLAS-Raw held-out-family | 0.403 | PrefixStats-Logistic | 0.003 | 0.002 | PrefixStats-Logistic | 0.002 | keep policy |
