# PatchCore: separable Gaussian smoothing

## Motivation

Profiling identified anomaly-map generation as a candidate for optimization.
Inspection of the installed implementation showed a full 33 x 33 Gaussian
convolution with same-size output and reflect padding.

Profiler scope percentages were treated as diagnostic indications rather
than a reliable additive decomposition of total GPU work.
The performance conclusion below comes from an unprofiled paired benchmark.

## Implementation

The candidate replaces the full 2D convolution with horizontal and vertical
1D convolutions.

The filters are derived from the existing checkpoint kernel. Initialization
checks that their outer product reconstructs the original kernel within
tolerance. Reflect padding is preserved.

The reference checkpoint, feature extractor, memory bank, and classification
threshold are unchanged.

## Numerical validation

Compared both implementations on all 83 bottle test images:

- Compared map pixels: 5439488.
- Maximum absolute map difference: approximately 0.0001907349.
- Mean absolute map difference: approximately 0.0000125786.
- Maximum image-score difference: 0.
- Changed classification decisions: 0.
- Map comparison tolerances: rtol=1e-5, atol=1e-4.

Maps are numerically close, not bit-identical.
Pixel AUROC and AP were not recomputed for the candidate in this experiment.

## Benchmark protocol

- NVIDIA GeForce RTX 5080.
- Float32 tensors, no autocast.
- 30 warmup calls per variant and batch size.
- 100 measured calls per variant and batch size.
- Same input batch within each pair.
- Alternating variant order.
- Sequential calls with CUDA synchronization.
- No profiler active during timing.

Scope: prepared CPU tensor to CPU scores, decisions, and anomaly maps.
File reading, resizing, batch assembly, model loading, and service overhead
are excluded.

## Results

| Batch | Variant | Mean ms | p50 ms | p95 ms | Images/s |
|---|---|---:|---:|---:|---:|
| 1 | Original 2D | 6.126 | 4.313 | 9.091 | 163.23 |
| 1 | Separable 1D | 5.292 | 3.327 | 8.361 | 188.97 |
| 8 | Original 2D | 18.777 | 18.926 | 19.648 | 426.06 |
| 8 | Separable 1D | 11.528 | 11.734 | 12.221 | 693.97 |

Mean latency speedup:
- Batch 1: 1.158x, approximately 13.6% lower latency.
- Batch 8: 1.629x, approximately 38.6% lower latency.

Full-precision measurements and runtime settings are in benchmark.json.
Individual timings are in latencies.csv.

## Conclusion and scope

Separable smoothing reduced measured full-inference latency while preserving
image scores and decisions exactly on this test set, with small map differences.

These are results from one short paired run on one hardware configuration.
They do not establish sustained service throughput or a latency guarantee.

## Runtime integration

PatchcorePredictor supports explicit filter selection:

    predictor = PatchcorePredictor(
        run_dir,
        device="cuda",
        blur_backend="separable_1d",
    )

Available options:

- original_2d: reference implementation and constructor default.
- separable_1d: optimized implementation.

The predictor check and benchmark scripts accept --blur-backend.
Their reports record the selected backend and filter class.

Both configuration paths were checked on all 83 test images:

- Exact preprocessing match.
- Maximum image-score difference: 0.
- Changed decisions: 0.
- Valid anomaly-map shapes and finite values.

Pixel-wise map agreement was checked separately in the numerical
validation experiment described above.
