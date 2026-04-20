# Retrieval Case Diagnosis

## Campaign Advantage Cases

| Query | Label | Campaign | Prefix | Margin | Campaign Top1 | Prefix Top1 | Campaign Dist | Prefix Dist | Campaign TTE Err | Prefix TTE Err |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| atlasv2/h1-m6 @ 2022-07-19T23:55:00+00:00 | 1 | 0.0608 | 0.0432 | 0.0177 | atlasv2/h1-benign-seg002 @ 2022-07-15T20:40:00+00:00 | atlasv2/h1-benign-seg002 @ 2022-07-15T20:40:00+00:00 | 3.760 | 3.760 | 40.0 | 40.0 |
| atlasv2/h1-m6 @ 2022-07-19T23:30:00+00:00 | 1 | 0.0339 | 0.0260 | 0.0079 | atlasv2/h1-benign-seg001 @ 2022-07-15T15:25:00+00:00 | atlasv2/h1-benign-seg001 @ 2022-07-15T15:25:00+00:00 | 2.803 | 2.803 | 15.0 | 15.0 |
| atlasv2/h1-m6 @ 2022-07-19T23:35:00+00:00 | 1 | 0.0339 | 0.0260 | 0.0079 | atlasv2/h1-benign-seg001 @ 2022-07-15T15:25:00+00:00 | atlasv2/h1-benign-seg001 @ 2022-07-15T15:25:00+00:00 | 2.927 | 2.927 | 20.0 | 20.0 |

## Campaign Failure Cases

| Query | Label | Campaign | Prefix | Margin | Campaign Top1 | Prefix Top1 | Campaign Dist | Prefix Dist | Campaign TTE Err | Prefix TTE Err |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| atlasv2/h1-m6 @ 2022-07-19T23:50:00+00:00 | 1 | 0.0880 | 0.4455 | -0.3575 | atlasv2/h1-benign-seg002 @ 2022-07-15T20:35:00+00:00 | atlasv2/h1-benign-seg002 @ 2022-07-15T20:35:00+00:00 | 3.726 | 3.726 | 35.0 | 35.0 |
| atlasv2/h1-m5 @ 2022-07-19T23:20:00+00:00 | 1 | 0.0822 | 0.4036 | -0.3214 | atlasv2/h1-s2 @ 2022-07-19T13:45:00+00:00 | atlasv2/h1-s2 @ 2022-07-19T13:10:00+00:00 | 5.545 | 7.257 | 20.0 | 55.0 |

## Campaign False Positive Cases

| Query | Label | Campaign | Prefix | Margin | Campaign Top1 | Prefix Top1 | Campaign Dist | Prefix Dist | Campaign TTE Err | Prefix TTE Err |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| atlasv2/h1-m5 @ 2022-07-19T23:40:00+00:00 | 0 | 0.0365 | 0.0004 | 0.0361 | atlasv2/h1-benign-seg002 @ 2022-07-15T20:05:00+00:00 | atlasv2/h2-m2 @ 2022-07-19T19:15:00+00:00 | 0.875 | 0.931 | 0.0 | 0.0 |
| atlasv2/h2-m6 @ 2022-07-19T23:50:00+00:00 | 0 | 0.0343 | 0.0003 | 0.0340 | atlasv2/h2-benign-seg002 @ 2022-07-15T20:35:00+00:00 | atlasv2/h2-benign-seg002 @ 2022-07-15T20:35:00+00:00 | 1.055 | 1.055 | 0.0 | 0.0 |
| atlasv2/h2-m5 @ 2022-07-19T23:05:00+00:00 | 0 | 0.0327 | 0.0004 | 0.0323 | atlasv2/h2-benign-seg002 @ 2022-07-15T20:35:00+00:00 | atlasv2/h2-benign-seg002 @ 2022-07-15T20:35:00+00:00 | 0.688 | 0.688 | 0.0 | 0.0 |
