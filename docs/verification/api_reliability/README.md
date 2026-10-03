# API reliability verification

## Runtime

- NVIDIA GeForce RTX 5080 under WSL.
- TensorRT FP32 feature extractor with TF32 disabled.
- Separable Gaussian blur.
- One Uvicorn process and one dedicated inference worker thread.

## Worker tests

Command:

    python -m unittest discover -s tests -p "test_api_worker.py" -v

Result: 8 tests passed.

The tests cover:

- Loading, inference and unloading on the same non-main thread.
- Rejection of overlapping inference requests with HTTP exception 429.
- Availability of the health handler while the worker is busy.
- Retention of the busy state after cancellation of the awaiting task.
- Recovery after a synthetic inference exception.
- Mapping an inference exception to HTTP exception 500.
- Clearing the busy state when executor submission fails.
- Startup failure and executor cleanup.

These tests use controlled substitute inference operations. Handler tests
call application functions directly rather than making HTTP requests.

## Real-server restart test

Command:

    python scripts/check_api_restart.py

Result: both server cycles passed.

Each cycle starts the actual API, waits for readiness, submits a defective
and a normal image with localization enabled, and requests graceful shutdown.

The test checks model identity, classification and localization thresholds,
localization calibration identity, expected image decisions, and shutdown.

Predictions agree across restarts within the configured score tolerance
(relative tolerance 1e-4, absolute tolerance 1e-4).

Displayed scores in both cycles:

| Image | Score | Decision |
| --- | ---: | --- |
| broken_large/011.png | 54.92676163 | ANOMALY |
| good/017.png | 31.41547203 | NORMAL |

The machine-readable evidence is stored in restart_report.json.

## Scope and limitations

This verification covers controlled worker behavior and two sequential
real-server lifecycles with two sample images per cycle.

It does not establish sustained service throughput, behavior under prolonged
load, network-disconnect handling, recovery from CUDA faults, or graceful
shutdown during an active inference.

The health endpoint reports application readiness and worker occupancy;
it is not a fresh GPU diagnostic.
