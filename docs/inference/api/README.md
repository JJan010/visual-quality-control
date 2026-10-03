# Local inference API

The API reuses the existing runtime configuration and PatchcorePredictor.
The current configuration uses TensorRT FP32 and separable Gaussian blur.

## Dependencies

The API's direct dependencies are listed in requirements-api.txt.
The complete environment snapshot is recorded in requirements-frozen.txt.

For an existing, working project ML environment:

    python -m pip install \
      --constraint requirements-frozen.txt \
      --requirement requirements-api.txt

Local model, calibration and TensorRT artifacts are required.
They are not included in Git.

## Start the server

Run from the repository root with the project virtual environment active:

    VQC_CONFIG="$PWD/configs/runtime/patchcore_tensorrt.json" \
    python -m uvicorn visual_quality.api.app:app \
      --host 127.0.0.1 \
      --port 8000 \
      --workers 1

Wait for "Application startup complete".

Interactive documentation:

    http://localhost:8000/docs

Stop the server with Ctrl+C.

Use one server process. Additional processes would load additional
model instances and allocate additional GPU resources.

## Model lifecycle and execution

FastAPI lifespan initializes the model before accepting requests.

A dedicated executor with one worker thread owns model loading,
inference and cleanup. The same model instance is reused across requests.

Only one prediction is admitted at a time. Requests reaching the
prediction worker while it is busy receive HTTP 429.

The busy flag is released when the executor task actually completes.
Cancellation of the waiting request does not release the flag early.

## Endpoints

### GET /health

Returns startup readiness, current busy state and runtime selection.

Readiness means that model initialization completed. This endpoint does
not execute a new GPU health check.

### POST /predict

Accepts a multipart form with one file field named "file".

Supported inputs:

- PNG or JPEG.
- A single frame.
- Maximum file size: 10 MiB.
- Maximum decoded dimensions: 16 million pixels.

The file-size check runs after framework multipart parsing. It is not
a transport-level limit on the complete HTTP request.

The response contains:

- Request ID.
- Raw anomaly score and calibrated image threshold.
- Boolean anomaly decision.
- Original image dimensions.
- Backend and blur selection.
- Checkpoint SHA-256.

The decision rule is:

    is_anomaly = score > threshold

Scores are not probabilities.

The default response contains classification results and timing measurements.
Use include_visualization=true to include the resized input and heatmap PNGs.
Prediction history is not persisted. See ../gui.md for the inspection UI.

## HTTP status codes

| Code | Meaning |
| --- | --- |
| 200 | Successful request |
| 400 | Empty file, unreadable image or unsupported multi-frame input |
| 413 | File size or image dimensions exceed the application limit |
| 415 | Decoded image format is unsupported |
| 422 | Request does not match the endpoint schema |
| 429 | Prediction worker is busy |
| 500 | Unexpected inference failure; details are in server logs |

## Verification performed

Two manually executed API-versus-CLI comparisons produced identical
image scores:

| Image | API score | Absolute difference vs CLI | Decision |
| --- | ---: | ---: | --- |
| test/broken_large/011.png | 54.926761627197266 | 0.0 | Anomaly |
| test/good/017.png | 31.41547203063965 | 0.0 | Normal |

These checks also compared the threshold and checkpoint identity.

Run the input-error checks while the server is running:

    python scripts/check_api_errors.py

Verified cases:

- Empty file: HTTP 400.
- Text disguised as a PNG: HTTP 400.
- Valid BMP image: HTTP 415.
- File larger than 10 MiB: HTTP 413.
- Valid normal image after those errors: HTTP 200, normal decision.
- Final health response: ready, not busy.

See error_handling_report.json for the recorded results.

These checks do not verify concurrent-request rejection, cancellation
behavior, the decoded-pixel limit or sustained service performance.

## Current scope

This is a local development API bound to loopback.
Authentication, public deployment and production load testing are
outside this implementation stage.
