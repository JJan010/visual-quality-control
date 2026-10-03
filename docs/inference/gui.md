# Inspection workspace

The English-language UI is served at `/` by the existing FastAPI process.
`/docs` remains available for API exploration. No Node server, external font,
CDN or frontend dependency is required at runtime.

## Start

From the repository root, with the virtual environment active:

    VQC_CONFIG="$PWD/configs/runtime/patchcore_tensorrt.json" \
    python -m uvicorn visual_quality.api.app:app \
      --host 127.0.0.1 --port 8000 --workers 1

Open http://localhost:8000/ and wait for **Engine ready**.

1. Choose or drop one PNG/JPEG image (up to 10 MiB).
2. Select **Run inspection**.
3. Read the image decision, raw score and fixed threshold.
4. Inspect the overlay; adjust opacity as needed.
5. Optionally download the JSON report, overlay PNG or calibrated region mask.

Use the bottle model with appropriate bottle images. The UI is not a generic
quality detector for arbitrary industrial objects. A normal decision means
the model score did not exceed its threshold, not a guarantee of defect absence.

No inspection history is persisted by the UI. Selecting a new image clears
the previous result. Failed requests also clear results to avoid stale decisions.

## Files and learning guide

- `api/presentation.py`: maps raw scores to display colors and encodes PNGs.
- `api/web.py`: serves HTML, CSS and JavaScript from the application package.
- `api/static/index.html`: page structure and accessible controls.
- `api/static/styles.css`: responsive page layout, typography and states.
- `api/static/app.js`: file selection, HTTP requests, canvas overlay and downloads.
- `api/app.py`: existing worker, extended with optional visualization and timing.

The data flow is: file selection → multipart POST → existing worker → model →
JSON response → decision cards and canvas. The browser only displays the
returned decision; it does not calculate a replacement classification threshold.

## API extension

`POST /predict?include_visualization=true` adds a `visualization` object.
Without the query option, `visualization` is null and no PNGs are encoded.
The existing score, threshold and decision fields retain their meaning.
Responses also include `timings`. Optional calibrated contours are described
in [Predicted regions](predicted_regions.md). Set `VQC_LOCALIZATION_CALIBRATION`
to enable them; an absent calibration is explicitly reported in the UI.

Visualization contains two base64-encoded PNGs: the exact resized model input
(converted back to 8-bit RGB for display) and the heatmap. Both share model-input
coordinates. The original panel shows the uploaded file; the overlay uses the
resized input. The browser alpha-blends the heatmap onto that input.

## Color scale

For each image, the raw map is scaled using `(value - min) / (max - min)` and
the Inferno colormap. A constant map is mapped uniformly to the low end.
This display-only operation does not change the model score or decision.

Colors are relative within an image and are NOT comparable between images.
A normal image can still have bright regions. Colors are neither defect
probabilities nor a binary segmentation. Raw min/max are included in the report.
The image threshold is not applied to map pixels. The overlay PNG is a display
artifact, not the raw anomaly map; use the CLI's `.npy` output for raw values.

## Timing

- `preprocessing_ms`: image decoding and existing image preparation.
- `inference_ms`: normalization, transfers to GPU, complete model execution,
  transfer of scores/labels and (when requested) the anomaly map to CPU.
  CUDA is synchronized at the measurement boundaries.
- `visualization_ms`: display normalization and PNG/base64 encoding; zero when
  visualization is not requested.
- Browser round trip: from starting fetch through reading/parsing the response.
  Includes upload, HTTP overhead and server processing; excludes selecting the
  file and rendering/decoding the returned PNGs.

These are individual request measurements, not a replacement for warmed-up
paired benchmarks. First calls can include warmup costs. PNG transfer and
serialization are not included in inference time.

## Validation and packaging

Run existing CLI/API parity and input-error checks after updating the server.
Verify both a normal and anomalous image in the UI. Check that file replacement,
HTTP errors and server disconnection do not leave a stale successful result.

The current project runs from an editable installation. Before distributing a
wheel, include `api/static/*` in setuptools package data and verify the built
wheel contains these assets. No wheel packaging change is made by this update.

This local interface does not add authentication or change the one-process,
one-GPU-worker execution policy.
