# Splunk Attack Data public probe audit

Independent public raw-log probe from selected Splunk Attack Data files. The probe uses deterministic MITRE-stage/keyword labels and incident-disjoint event-bucket windows; it is a stress test rather than a replacement for analyst-labeled triage data.

| Method | Seeds | Test AUPRC | Event AUPRC | AUROC | Brier | AF@5 |
|---|---:|---:|---:|---:|---:|---:|
| TRACER adaptive | 7,13,21 | 0.688 +/- 0.008 | 0.688 +/- 0.008 | 0.902 +/- 0.000 | 0.123 +/- 0.002 | 83.3 +/- 0.2 |
| TRACER core | 7,13,21 | 0.690 +/- 0.008 | 0.690 +/- 0.008 | 0.899 +/- 0.004 | 0.114 +/- 0.002 | 83.5 +/- 0.4 |
| DLinear | 7,13,21 | 0.657 +/- 0.012 | 0.657 +/- 0.012 | 0.890 +/- 0.005 | 0.118 +/- 0.002 | 83.3 +/- 0.2 |
| Prefix-only | 7,13,21 | 0.687 +/- 0.004 | 0.687 +/- 0.004 | 0.899 +/- 0.001 | 0.123 +/- 0.005 | 83.6 +/- 0.3 |
| Pure kNN | 7 | 0.550 +/- 0.000 | 0.550 +/- 0.000 | 0.763 +/- 0.000 | 0.152 +/- 0.000 | 84.4 +/- 0.0 |
