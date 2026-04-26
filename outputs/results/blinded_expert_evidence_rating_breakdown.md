# Blinded Expert Evidence-Rating Breakdown

- Completed rater rows: 240
- Unique paired cases: 80
- Majority preference by case: {'TRACER route': 25, 'Tie': 38, 'Prefix-Only': 17}
- All three raters exactly agreed on 62/80 cases.

## Overall Row-Level Results
| Metric | TRACER | Prefix-Only | Delta | 95% CI |
|---|---:|---:|---:|---:|
| Relevance | 4.246 | 4.225 | +0.021 | [-0.108, +0.150] |
| Supportiveness | 4.342 | 4.317 | +0.025 | [-0.075, +0.129] |
| Actionability | 3.821 | 3.696 | +0.125 | [+0.029, +0.217] |
| Explanation quality | 4.250 | 4.142 | +0.108 | [+0.033, +0.183] |
| Misleading safety | 4.296 | 4.225 | +0.071 | [-0.054, +0.192] |

## Dataset-Level Row-Level Delta
| Dataset | Relevance | Supportiveness | Actionability | Explanation | Misleading safety | Preference |
|---|---:|---:|---:|---:|---:|---|
| AIT-ADS chronology | +0.350 | +0.275 | +0.375 | +0.275 | +0.333 | {'TRACER route': 63, 'Tie': 41, 'Prefix-Only': 16} |
| ATLASv2 held-out-family | -0.308 | -0.225 | -0.125 | -0.058 | -0.192 | {'TRACER route': 17, 'Tie': 65, 'Prefix-Only': 38} |

## Case-Averaged Overall Delta
| Metric | Delta | 95% CI |
|---|---:|---:|
| Relevance | +0.021 | [-0.196, +0.242] |
| Supportiveness | +0.025 | [-0.138, +0.200] |
| Actionability | +0.125 | [-0.017, +0.279] |
| Explanation quality | +0.108 | [-0.004, +0.225] |
| Misleading safety | +0.071 | [-0.133, +0.275] |
