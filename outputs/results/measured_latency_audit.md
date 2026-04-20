# Measured latency audit

The microbenchmark isolates retrieval lookup cost by using random normalized 128-dimensional float32 embeddings with the same bank sizes as the released retrieval-active routes. It measures CPU wall-clock lookup time on this workstation and reports an LSH-style approximate candidate lookup as a lightweight ANN stress test, not as a production index recommendation.

| Setting | Q | Bank | Exact ms/query | LSH ms/query | LSH Recall@5 | Candidates | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| ATLASv2 held-out-family | 83 | 438 | 0.0039 | 0.0245 | 26.5% | 101.2 | 21.7% |
| AIT-ADS chronology | 4579 | 9119 | 0.0473 | 0.0080 | 1.7% | 39.2 | 0.0% |
| ATLAS-Raw chronology | 1880 | 46156 | 0.2420 | 0.0132 | 0.6% | 52.2 | 0.0% |
