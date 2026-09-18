# BiRefNet General FP16 ONNX — CPU Production Benchmark Report
**Timestamp**: 2026-09-18 08:52:15 UTC

## Environment Metadata
- **OS**: Windows 11 (10.0.26100)
- **CPU**: AMD Ryzen 5 7535HS with Radeon Graphics (6 physical cores, 12 logical threads)
- **System RAM**: 7498.79 MB
- **ONNX Runtime**: 1.24.4 (CPUExecutionProvider)
- **Model**: BiRefNet General (FP16 ONNX) (1024x1024 FP16 input)
- **Test Image**: 1408 × 768 JPEG (485004 bytes)

## Result Table (9 Real API Requests)
| Threads | Run | Inference (s) | Total (s) | Peak RSS (MB) | CPU Peak (%) | Status |
|---------|-----|---------------|-----------|---------------|--------------|--------|
| 4 | 1 | 63.536 | 63.911 | 3148.5 | 1606.8 | SUCCESS |
| 4 | 2 | 75.060 | 75.436 | 3069.05 | 1358.4 | SUCCESS |
| 4 | 3 | 66.158 | 66.533 | 3525.92 | 1338.2 | SUCCESS |
| 6 | 1 | 61.003 | 61.326 | 3161.79 | 1371.0 | SUCCESS |
| 6 | 2 | 57.497 | 57.872 | 3346.17 | 1077.5 | SUCCESS |
| 6 | 3 | 59.525 | 59.903 | 3340.63 | 1205.3 | SUCCESS |
| 8 | 1 | 72.646 | 73.142 | 3483.07 | 1501.3 | SUCCESS |
| 8 | 2 | 76.073 | 76.550 | 2773.66 | 1517.6 | SUCCESS |
| 8 | 3 | 68.871 | 69.383 | 3179.3 | 1513.6 | SUCCESS |

## Averages Table
| Threads | Avg Inference (s) | Median Inference (s) | Avg Total (s) | Median Total (s) | Avg Peak RSS (MB) | Max Peak RSS (MB) | Avg CPU Peak (%) |
|---------|-------------------|----------------------|---------------|------------------|-------------------|-------------------|------------------|
| 4 | 68.251 | 66.158 | 68.627 | 66.533 | 3247.82 | 3525.92 | 1434.47 |
| 6 | 59.342 | 59.525 | 59.700 | 59.903 | 3282.86 | 3346.17 | 1217.93 |
| 8 | 72.530 | 72.646 | 73.025 | 73.142 | 3145.34 | 3483.07 | 1510.83 |

## Performance Comparison (Relative to 4 Threads)
| Comparison | Inference Improv. (%) | Total Processing Improv. (%) | Peak RSS Change (%) | CPU Peak Difference (%) | Speedup Factor |
|------------|-----------------------|------------------------------|----------------------|-------------------------|----------------|
| 6T_VS_4T | +13.05% | +13.01% | +1.08% | -216.54% | 1.15x |
| 8T_VS_4T | -6.27% | -6.41% | -3.16% | +76.36% | 0.94x |

## Cold Model Load Times
- **4 Threads**: 15645.31 ms (15.65 s)
- **6 Threads**: 13448.13 ms (13.45 s)
- **8 Threads**: 13101.22 ms (13.10 s)

## Quality Validation
- **Output Resolution**: 1408 × 768 (100% native resolution preserved)
- **Format**: PNG RGBA (4 channels)
- **Data Integrity**: 0 NaN values, 0 Inf values
- **Parity**: Exact 100% pixel match across 4-thread, 6-thread, and 8-thread output alpha masks.

## Factual Deployment Analysis
1. **Fastest Configuration**: 6 Threads
2. **Average Processing Times**: 4T = 68.627s | 6T = 59.700s | 8T = 73.025s
3. **True Peak RAM Requirement**: 3525.92 MB RSS peak observed during real API inference.
4. **4 → 6 Threads Improvement**: Inference improved by +13.05%, Total by +13.01%.
5. **6 → 8 Threads Improvement**: Inference improved by -6.27%, Total by -6.41%.
6. **Workload Bottleneck**: BiRefNet ONNX inference accounts for **~99.4%** of total API processing time.
7. **CPU Suitability**: Pure CPU execution is functional and accurate, but response time (~59s–73s) is far too slow for synchronous web user experience.
8. **Minimum RAM Target**: Allocate at least **1.5 GB to 2.0 GB RAM** per instance to comfortably support process RSS (3525.92 MB) + system overhead.
9. **Render Host Suitability**: **Render Free tier (0.5 CPU, 512MB RAM) is INSUFFICIENT**. Requests will hit 30s HTTP gateway timeouts. A GPU instance (NVIDIA CUDA) or higher CPU tier is strongly recommended for production.