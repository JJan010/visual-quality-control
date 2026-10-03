"use strict";

const $ = (id) => document.getElementById(id);
const state = {file: null, url: null, ready: false, running: false, result: null, inputImage: null, heatmapImage: null, contourImage: null, elapsed: null};
const resultCard = document.querySelector(".result-card");

function message(text = "") { $("message").textContent = text; $("message").hidden = !text; }
function updateControls() {
  $("run").disabled = !state.file || !state.ready || state.running;
  $("file-input").disabled = state.running;
  $("clear-file").disabled = state.running;
  $("run-label").textContent = state.running ? "Inspecting…" : "Run inspection";
  $("opacity").disabled = !state.inputImage || state.running;
  $("download-report").disabled = !state.result || state.running;
  $("download-overlay").disabled = !state.inputImage || state.running;
  $("show-region").disabled = !state.contourImage || state.running;
  $("download-mask").disabled = !state.result?.visualization?.localization || state.running;
}
function resetResult() {
  state.result = null; state.inputImage = null; state.heatmapImage = null; state.contourImage = null; state.elapsed = null;
  resultCard.classList.remove("normal", "anomaly");
  $("result-badge").className = "badge";
  $("result-badge").textContent = state.file ? "Ready to inspect" : "Awaiting image";
  $("decision").textContent = state.file ? "Image selected." : "Ready when you are.";
  $("decision-detail").textContent = state.file ? "Run the inspection to evaluate this image." : "Select an image to start your first inspection.";
  $("decision-icon").textContent = "◎";
  for (const id of ["score", "threshold", "inference-time", "score-max"]) $(id).textContent = "—";
  $("score-fill").style.width = "0"; $("threshold-marker").hidden = true;
  $("threshold-label").textContent = "Score above threshold → anomaly";
  $("request-time").textContent = "No measurement yet";
  $("overlay").hidden = true; $("overlay-empty").hidden = false;
  $("map-caption").textContent = "Awaiting inspection";
  $("region-summary").textContent = "Run an inspection to see predicted regions.";
  $("region-summary").className = "region-summary";
  $("technical-details").replaceChildren();
  updateControls();
}
function selectFile(file) {
  if (state.running) return;
  // Zawsze usuwamy stary wynik, również przy wyborze błędnego pliku.
  if (state.url) URL.revokeObjectURL(state.url);
  state.url = null; state.file = null;
  $("original-image").hidden = true; $("original-image").removeAttribute("src");
  $("original-empty").hidden = false; $("file-info").hidden = true;
  $("original-caption").textContent = "Image preview";
  message(); resetResult();
  if (!file) return;
  if (!/\.(png|jpe?g)$/i.test(file.name)) return message("Choose a PNG or JPG image.");
  if (!file.size) return message("The selected file is empty.");
  if (file.size > 10 * 1024 * 1024) return message("The image must be no larger than 10 MiB.");
  state.file = file; state.url = URL.createObjectURL(file);
  $("file-name").textContent = file.name;
  $("file-size").textContent = `${(file.size / 1024).toFixed(1)} KiB`;
  $("file-info").hidden = false;
  $("original-image").src = state.url; $("original-image").hidden = false;
  $("original-empty").hidden = true;
  $("original-caption").textContent = "Uploaded image";
  resetResult();
}
$("original-image").addEventListener("error", () => {
  if (state.running) return;
  selectFile(null); message("This file cannot be displayed as an image.");
});
$("file-input").addEventListener("change", (event) => { selectFile(event.target.files[0]); event.target.value = ""; });
$("clear-file").addEventListener("click", () => selectFile(null));
const dropZone = $("drop-zone");
for (const name of ["dragenter", "dragover"]) dropZone.addEventListener(name, (event) => { event.preventDefault(); if (!state.running) dropZone.classList.add("dragover"); });
for (const name of ["dragleave", "drop"]) dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove("dragover"); });
dropZone.addEventListener("drop", (event) => {
  if (state.running) return;
  if (event.dataTransfer.files.length !== 1) { selectFile(null); message("Please select one image at a time."); return; }
  selectFile(event.dataTransfer.files[0]);
});
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

async function refreshHealth() {
  try {
    const response = await fetch("/health", {cache: "no-store", signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error("Unavailable");
    const health = await response.json();
    state.ready = health.status === "ready";
    $("connection").className = `connection ${state.ready ? "online" : "offline"}`;
    $("connection").textContent = state.ready ? (health.busy ? "Engine processing" : "Engine ready") : "Engine unavailable";
    $("backend").textContent = ({tensorrt: "TensorRT", onnx: "ONNX Runtime", pytorch: "PyTorch"})[health.backend] || health.backend;
  } catch {
    state.ready = false;
    $("connection").className = "connection offline";
    $("connection").textContent = "Server offline";
  }
  updateControls();
}

function loadPng(base64) {
  return new Promise((resolve, reject) => {
    const image = new Image(); image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not display the returned visualization."));
    image.src = `data:image/png;base64,${base64}`;
  });
}
function drawOverlay() {
  if (!state.inputImage || !state.heatmapImage) return;
  const canvas = $("overlay"), context = canvas.getContext("2d");
  canvas.width = state.inputImage.naturalWidth; canvas.height = state.inputImage.naturalHeight;
  context.globalAlpha = 1; context.drawImage(state.inputImage, 0, 0);
  context.globalAlpha = Number($("opacity").value) / 100;
  context.drawImage(state.heatmapImage, 0, 0); context.globalAlpha = 1;
  if ($("show-region").checked && state.contourImage) context.drawImage(state.contourImage, 0, 0);
  $("opacity-value").textContent = `${$("opacity").value}%`;
}
$("opacity").addEventListener("input", drawOverlay);
$("show-region").addEventListener("change", drawOverlay);
function addDetail(label, value) {
  const row = document.createElement("div"), term = document.createElement("dt"), description = document.createElement("dd");
  term.textContent = label; description.textContent = String(value);
  row.append(term, description); $("technical-details").append(row);
}
async function displayResult(result, elapsed) {
  const visual = result.visualization;
  if (!visual) throw new Error("The server did not return a visualization. Restart the updated server.");
  const [input, heatmap] = await Promise.all([loadPng(visual.input_png_base64), loadPng(visual.heatmap_png_base64)]);
  const localization = visual.localization;
  const contour = localization ? await loadPng(localization.contour_png_base64) : null;
  state.inputImage = input; state.heatmapImage = heatmap; state.contourImage = contour;
  state.result = result; state.elapsed = elapsed;
  const decision = result.is_anomaly ? "anomaly" : "normal";
  resultCard.classList.add(decision);
  $("result-badge").className = `badge ${decision}`;
  $("result-badge").textContent = "Inspection complete";
  $("decision").textContent = result.is_anomaly ? "Anomaly detected" : "No anomaly detected";
  $("decision-detail").textContent = result.is_anomaly ? "The image score exceeds the calibrated threshold." : "The image score does not exceed the calibrated threshold.";
  $("decision-icon").textContent = result.is_anomaly ? "!" : "✓";
  $("score").textContent = result.score.toFixed(4);
  $("threshold").textContent = result.threshold.toFixed(4);
  $("inference-time").textContent = `${result.timings.inference_ms.toFixed(1)} ms`;
  const limit = Math.max(result.score, result.threshold, 1) * 1.2;
  $("score-fill").style.width = `${Math.max(0, result.score / limit * 100)}%`;
  $("threshold-marker").style.left = `${result.threshold / limit * 100}%`;
  $("threshold-marker").hidden = false;
  $("score-max").textContent = limit.toFixed(1);
  $("threshold-label").textContent = `Threshold ${result.threshold.toFixed(4)} · marker`;
  $("score-fill").parentElement.setAttribute("aria-label", `Score ${result.score.toFixed(4)}, threshold ${result.threshold.toFixed(4)}`);
  $("request-time").textContent = `Browser round trip: ${elapsed.toFixed(0)} ms`;
  $("original-caption").textContent = `${result.original_width} × ${result.original_height} · uploaded image`;
  $("map-caption").textContent = `${visual.width} × ${visual.height} · model input`;
  $("input-resolution").textContent = `${visual.width} × ${visual.height}`;
  $("overlay").hidden = false; $("overlay-empty").hidden = true;
  drawOverlay();
  $("region-summary").className = `region-summary ${localization?.mask_pixels > 0 ? "has-region" : ""}`;
  $("region-summary").textContent = !localization
    ? "Region calibration is not loaded. Only the heatmap is available."
    : localization.mask_pixels > 0
      ? `Predicted region: ${localization.mask_pixels.toLocaleString()} pixels (${(localization.mask_fraction * 100).toFixed(2)}% of the model input). Localization threshold: ${localization.threshold.toFixed(4)}.${!result.is_anomaly ? " The image-level decision remains normal." : ""}`
      : `No region exceeds the localization threshold (${localization.threshold.toFixed(4)}).${result.is_anomaly ? " The image-level decision remains anomaly." : ""}`;
  $("technical-details").replaceChildren();
  const details = {
    "Request ID": result.request_id,
    "Backend / blur": `${result.backend} / ${result.blur_backend}`,
    "Checkpoint SHA-256": result.checkpoint_sha256,
    "Preprocessing": `${result.timings.preprocessing_ms.toFixed(2)} ms`,
    "Inference + output transfers": `${result.timings.inference_ms.toFixed(2)} ms`,
    "Visualization encoding": `${result.timings.visualization_ms.toFixed(2)} ms`,
    "Browser round trip (upload + server + response parsing)": `${elapsed.toFixed(2)} ms`,
    "Raw map range": `${visual.map_min.toFixed(4)} – ${visual.map_max.toFixed(4)}`,
    "Color scale": "Per-image min–max; not comparable between images",
    "Decision rule": result.decision_rule
  };
  for (const [label, value] of Object.entries(details)) addDetail(label, value);
  if (localization) {
    addDetail("Localization threshold", localization.threshold);
    addDetail("Localization calibration SHA-256", localization.calibration_sha256);
    addDetail("Region rule", localization.decision_rule);
    addDetail("Region area", `${localization.mask_pixels} / ${localization.total_pixels} model-input pixels`);
  }
}

$("run").addEventListener("click", async () => {
  if (!state.file || !state.ready || state.running) return;
  resetResult(); message(); state.running = true; updateControls();
  $("result-badge").textContent = "Processing";
  $("decision").textContent = "Inspecting your image…";
  $("decision-detail").textContent = "Evaluating the image and preparing its anomaly map.";
  try {
    const body = new FormData(); body.append("file", state.file);
    const started = performance.now();
    const response = await fetch("/predict?include_visualization=true", {method: "POST", body, signal: AbortSignal.timeout(120000)});
    const result = await response.json();
    const elapsed = performance.now() - started;
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : `Inspection failed (HTTP ${response.status}).`);
    await displayResult(result, elapsed);
  } catch (error) {
    resetResult();
    $("result-badge").textContent = "Inspection failed";
    $("decision").textContent = "Inspection unavailable";
    $("decision-detail").textContent = "No valid result is available for this request.";
    message(error.name === "TimeoutError" ? "The request timed out. The server may still be processing; wait before retrying." : error.message);
  } finally { state.running = false; updateControls(); refreshHealth(); }
});

function download(blob, filename) {
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("download-report").addEventListener("click", () => {
  if (!state.result) return;
  const {visualization, ...prediction} = state.result;
  const {input_png_base64, heatmap_png_base64, localization, ...mapDetails} = visualization;
  if (localization) {
    const {mask_png_base64, contour_png_base64, ...regionMetadata} = localization;
    mapDetails.localization = {...regionMetadata, contour_visible: $("show-region").checked};
  } else { mapDetails.localization = null; }
  const report = {...prediction, input_filename: state.file.name, browser_round_trip_ms: state.elapsed, visualization: {...mapDetails, overlay_opacity: Number($("opacity").value) / 100}};
  download(new Blob([JSON.stringify(report, null, 2) + "\n"], {type: "application/json"}), `inspection_${prediction.request_id}.json`);
});
$("download-overlay").addEventListener("click", () => {
  if (!state.result) return;
  const requestId = state.result.request_id;
  $("overlay").toBlob((blob) => { if (blob) download(blob, `overlay_${requestId}.png`); }, "image/png");
});
$("download-mask").addEventListener("click", () => {
  const localization = state.result?.visualization?.localization;
  if (!localization) return;
  const bytes = Uint8Array.from(atob(localization.mask_png_base64), (character) => character.charCodeAt(0));
  download(new Blob([bytes], {type: "image/png"}), `region_mask_${state.result.request_id}.png`);
});
refreshHealth();
setInterval(refreshHealth, 10000);
