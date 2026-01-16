
# Performance Benchmark Results

## Embedding Generation Latency

| Metric | Face (512-dim) | Voice (192-dim) |
|--------|----------------|-----------------|
| Mean | 24.23 ms | 2.47 ms |
| P50 | 22.95 ms | 2.48 ms |
| P95 | 30.34 ms | 2.56 ms |
| P99 | 48.36 ms | 2.62 ms |

## Database Query Latency

| Gallery Size | NumPy (ms) | pgvector (ms) | Speedup |
|--------------|------------|---------------|---------|
| 10 | 0.00 | 3.96 | 0.0x |
| 50 | 0.02 | 1.61 | 0.0x |
| 100 | 0.02 | 1.60 | 0.0x |
| 500 | 0.01 | 1.68 | 0.0x |
| 1000 | 0.01 | 1.71 | 0.0x |
