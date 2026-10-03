"""Verify an installed wheel and GUI routes using existing venv dependencies."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

PROBE = r'''
import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import sys

target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
api = importlib.import_module("visual_quality.api.app")

async def request(path):
    messages = []
    delivered = False
    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.Future()
    async def send(message):
        messages.append(message)
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1", "method": "GET", "scheme": "http",
        "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "headers": [], "server": ("localhost", 80),
        "client": ("127.0.0.1", 12345),
    }
    await asyncio.wait_for(api.app(scope, receive, send), timeout=15)
    starts = [m for m in messages if m["type"] == "http.response.start"]
    if len(starts) != 1 or starts[0]["status"] != 200:
        raise RuntimeError(f"Unexpected response: {path}: {starts}")
    body = b"".join(m.get("body", b"") for m in messages
                    if m["type"] == "http.response.body")
    return {"path": path, "status": 200, "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest()}

async def main():
    results = []
    for path in ("/", "/assets/index.html", "/assets/styles.css", "/assets/app.js"):
        results.append(await request(path))
    for name, module in list(sys.modules.items()):
        if name == "visual_quality" or name.startswith("visual_quality."):
            locations = list(getattr(module, "__path__", []))
            if getattr(module, "__file__", None):
                locations.append(module.__file__)
            for location in locations:
                if not Path(location).resolve().is_relative_to(target):
                    raise RuntimeError(f"Source checkout leaked into import: {name}: {location}")
    print(json.dumps({"import_origin": str(Path(api.__file__).relative_to(target)),
                      "routes": results}))

asyncio.run(main())
'''


def main():
    root = Path.cwd()
    wheels = list((root / "artifacts/packaging").glob("visual_quality_control-*.whl"))
    if not wheels:
        raise SystemExit("STOP: build the wheel first, then run from the repository root.")
    wheel = max(wheels, key=lambda p: p.stat().st_mtime).resolve()
    expected = {}
    with ZipFile(wheel) as archive:
        for route, name in (("/", "index.html"), ("/assets/index.html", "index.html"),
                            ("/assets/styles.css", "styles.css"), ("/assets/app.js", "app.js")):
            content = archive.read(f"visual_quality/api/static/{name}")
            expected[route] = hashlib.sha256(content).hexdigest()
    with TemporaryDirectory(prefix="vqc-wheel-check-") as temporary:
        work = Path(temporary)
        target = work / "installed"
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
            "--no-compile", "--target", str(target), str(wheel),
        ], check=True, cwd=work)
        result = subprocess.run(
            [sys.executable, "-I", "-c", PROBE, str(target)],
            check=True, cwd=work, capture_output=True, text=True, timeout=120,
        )
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        report = json.loads(result.stdout)
    for route in report["routes"]:
        if route["sha256"] != expected[route["path"]]:
            raise RuntimeError(f"Response content mismatch: {route['path']}")
        print(f"{route['path']}: HTTP {route['status']}; content matches wheel")
    report.update({
        "status": "passed", "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "scope": "installed package imports and in-process ASGI GUI requests",
        "dependencies": "reused from the active environment",
        "lifespan_started": False, "gpu_inference_checked": False,
    })
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = root / "artifacts/packaging" / f"installed_wheel_{stamp}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Report:", output)
    print("INSTALLED WHEEL GUI: OK")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout)
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise
