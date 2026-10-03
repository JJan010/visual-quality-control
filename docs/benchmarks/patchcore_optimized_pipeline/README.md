# PatchCore: TensorRT FP32 and separable Gaussian blur

## Objective

Measure the combined effect of TensorRT feature extraction and separable
Gaussian smoothing on the complete PatchCore inference pipeline.

## Configuration

- GPU: NVIDIA GeForce RTX 5080.
- PyTorch reference: IEEE FP32.
- TensorRT: FP32 with TF32 disabled.
- TensorRT batch profile: min=1, opt=1, max=8.
- Input resolution: 256 x 256.
- Warmup: 30 calls per configuration and batch size.
- Measurements: 120 calls per configuration and batch size.
- All six configuration orderings were equally represented and shuffled.
- The same input batch was used for every configuration within each round.

## Timing scope

Timing starts with a prepared RGB float32 batch on CPU and includes:

- Transfer to GPU and normalization.
- Feature extraction.
- PatchCore scoring and anomaly-map generation.
- Transfer of scores, decisions and anomaly maps back to CPU.
- CUDA synchronization needed to measure completed work.

File loading, image decoding, resizing and CPU batch construction are excluded.
Model loading and engine building are also excluded.

## Results

All latency values are milliseconds per batch.

| Batch | Configuration | Mean | p50 | p95 | Images/s | Speedup vs PyTorch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | PyTorch + original blur | 7.075 | 6.843 | 8.214 | 141.35 | 1.000x |
| 1 | TensorRT + original blur | 3.954 | 3.772 | 4.867 | 252.92 | 1.789x |
| 1 | TensorRT + separable blur | 2.971 | 2.756 | 3.858 | 336.57 | 2.381x |
| 8 | PyTorch + original blur | 25.722 | 25.745 | 26.817 | 311.01 | 1.000x |
| 8 | TensorRT + original blur | 20.589 | 20.673 | 21.457 | 388.55 | 1.249x |
| 8 | TensorRT + separable blur | 12.610 | 12.790 | 13.340 | 634.39 | 2.040x |

Within TensorRT, replacing the original blur reduced mean latency by a
factor of 1.331x for batch 1 and 1.633x for batch 8.

## Numerical validation

The combined pipeline was checked on all 83 MVTec AD bottle test images,
separately for batch sizes 1 and 8.

- No decisions changed against the PyTorch reference.
- No decisions changed against TensorRT with the original blur.
- No decisions changed against the saved baseline.
- Image scores were exactly unchanged when replacing only the TensorRT
  pipeline's blur.
- Scores and maps passed the recorded numerical tolerances.

Maximum absolute differences against the PyTorch reference:

| Batch | Image score | Anomaly map |
| --- | ---: | ---: |
| 1 | 0.0001525879 | 0.0004825592 |
| 8 | 0.0001354218 | 0.0004262924 |

Pixel AUROC and pixel average precision were not recomputed in this check.
Numerical agreement does not imply bitwise-identical maps.

## Evidence

- benchmark.json: configuration, runtime metadata and aggregated timings.
- latencies.csv: individual measured latencies.
- optimized_parity.json: numerical validation of the combined pipeline.

## Limitations

Results describe one short benchmark on one hardware configuration.
Images/s is calculated from batch size and mean batch latency.
It is not a sustained service-throughput or latency guarantee.

The bottle test set has been reused for engineering regression checks.
These checks are not a new evaluation on unseen production data.

FP16 was not evaluated in this experiment.
