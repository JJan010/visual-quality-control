"""Start, query, gracefully stop and restart the real local GPU API.

Runs two server processes sequentially on port 18080 by default.
Only subprocesses created by this script are stopped.
"""

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4


DEFAULT_CALIBRATION = (
    "artifacts/runs/patchcore_20261003_135714_557111/"
    "localization_calibration/20261003_182936_190431/threshold.json"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def assert_port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"Port {port} is occupied; no existing process was stopped.") from error


def get_health(base_url):
    with urlopen(base_url + "/health", timeout=3) as response:
        return json.load(response)


def wait_ready(process, base_url, log_path):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited during startup. See {log_path}")
        try:
            health = get_health(base_url)
            if health.get("status") == "ready":
                return health
        except (URLError, OSError, ValueError):
            pass
        time.sleep(0.2)
    raise TimeoutError(f"Server did not become ready within 180 seconds. See {log_path}")


def upload(base_url, path):
    boundary = uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = Request(
        base_url + "/predict?include_visualization=true",
        data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def stop_owned_process(process):
    if process.poll() is not None:
        return {"graceful_signal_sent": False, "forced": False, "returncode": process.returncode}
    process.send_signal(signal.SIGINT)
    forced = False
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        forced = True
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    return {"graceful_signal_sent": True, "forced": forced, "returncode": process.returncode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/runtime/patchcore_tensorrt.json"))
    parser.add_argument("--calibration-file", type=Path, default=Path(DEFAULT_CALIBRATION))
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    root = Path.cwd()
    require((root / "src/visual_quality/api/app.py").is_file(), "Run from the repository root.")
    require(os.name == "posix", "Run this script inside WSL/Linux.")
    require(1024 <= args.port <= 65535, "Choose a port in the range 1024–65535.")
    calibration_bytes = args.calibration_file.read_bytes()
    calibration = json.loads(calibration_bytes)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    calibration_sha = hashlib.sha256(calibration_bytes).hexdigest()
    cases = [("broken_large/011.png", True), ("good/017.png", False)]
    for relative, _ in cases:
        require((root / "data/mvtec_ad/bottle/test" / relative).is_file(), f"Missing image: {relative}")
    assert_port_free(args.port)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = root / "artifacts/api_checks" / f"restart_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "scope": "two sequential real-server lifecycles; not a load or disconnect test",
        "port": args.port,
        "calibration_sha256": calibration_sha,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "app_sha256": hashlib.sha256((root / "src/visual_quality/api/app.py").read_bytes()).hexdigest(),
        "cycles": [],
    }
    env = dict(os.environ)
    env["VQC_CONFIG"] = str(args.config.resolve())
    env["VQC_LOCALIZATION_CALIBRATION"] = str(args.calibration_file.resolve())
    env["PYTHONUNBUFFERED"] = "1"
    base_url = f"http://127.0.0.1:{args.port}"

    try:
        for cycle_number in (1, 2):
            assert_port_free(args.port)
            cycle = {"cycle": cycle_number, "predictions": []}
            report["cycles"].append(cycle)
            log_path = output / f"server_{cycle_number}.log"
            print(f"Cycle {cycle_number}/2: starting real server...", flush=True)
            with log_path.open("wb") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "visual_quality.api.app:app",
                     "--host", "127.0.0.1", "--port", str(args.port), "--workers", "1"],
                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                )
                cycle["pid"] = process.pid
                try:
                    health = wait_ready(process, base_url, log_path)
                    require(health["backend"] == config["backend"], "Wrong active backend.")
                    require(health["blur_backend"] == config["blur_backend"], "Wrong blur backend.")
                    require(health["busy"] is False, "Server started in a busy state.")
                    cycle["startup_health"] = health
                    print("  Startup: ready", flush=True)
                    for relative, expected in cases:
                        prediction = upload(base_url, root / "data/mvtec_ad/bottle/test" / relative)
                        require(prediction["is_anomaly"] is expected, f"Unexpected decision: {relative}")
                        require(math.isfinite(prediction["score"]), "Non-finite image score.")
                        require(prediction["threshold"] == calibration["runtime"]["threshold"], "Wrong image threshold.")
                        require(prediction["checkpoint_sha256"] == calibration["runtime"]["checkpoint_sha256"], "Wrong checkpoint.")
                        localization = prediction["visualization"]["localization"]
                        require(localization is not None, "Localization was not enabled.")
                        require(localization["calibration_sha256"] == calibration_sha, "Wrong localization calibration.")
                        require(localization["threshold"] == calibration["threshold"], "Wrong localization threshold.")
                        cycle["predictions"].append({
                            "image": relative, "score": prediction["score"],
                            "is_anomaly": prediction["is_anomaly"],
                            "region_pixels": localization["mask_pixels"],
                            "request_id": prediction["request_id"],
                        })
                        print(f"  {relative}: decision OK; score={prediction['score']:.8f}", flush=True)
                    require(get_health(base_url)["busy"] is False, "Worker stayed busy after prediction.")
                finally:
                    cycle["shutdown"] = stop_owned_process(process)

            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            cycle["shutdown"]["completion_logged"] = "Application shutdown complete." in log_text
            require(cycle["shutdown"]["graceful_signal_sent"], "Server exited before the shutdown request.")
            require(not cycle["shutdown"]["forced"], "Server required forced termination.")
            require(cycle["shutdown"]["completion_logged"], f"No shutdown completion in {log_path}")
            require(cycle["shutdown"]["returncode"] in (0, -signal.SIGINT), "Unexpected server exit code.")
            print("  Graceful shutdown: OK", flush=True)

        differences = []
        for first, second in zip(report["cycles"][0]["predictions"], report["cycles"][1]["predictions"]):
            require(first["image"] == second["image"], "Image mismatch between cycles.")
            require(first["is_anomaly"] == second["is_anomaly"], "Decision changed after restart.")
            require(math.isclose(first["score"], second["score"], rel_tol=1e-4, abs_tol=1e-4), "Score changed beyond tolerance after restart.")
            differences.append({"image": first["image"],
                                "score_difference": abs(first["score"] - second["score"]),
                                "region_pixel_count_difference": second["region_pixels"] - first["region_pixels"]})
        report["restart_comparisons"] = differences
        report["status"] = "passed"
        print("Prediction agreement after restart: OK")
        print("API RESTART: OK")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("Report:", output / "report.json", flush=True)


if __name__ == "__main__":
    main()
