# Installed package verification

Verified on the author's WSL/Python 3.10 environment on 2026-10-03.
The pre-reproducibility-update wheel was built successfully; HTML, CSS and JS
matched their source bytes. The frozen dependency snapshot matched the active
environment (excluding the project itself and default pip freeze exclusions).

`check_installed_wheel.py` installed that wheel into a temporary target without
installing dependencies, then imported it outside the checkout in a separate
Python process. It checked that loaded project modules came from the temporary
installation and exercised `/`, `/assets/index.html`, `/assets/styles.css` and
`/assets/app.js` directly via ASGI. All four responses were 200 and matched the
wheel contents. Evidence: `installed_wheel_report.json`.

This was not a fresh-environment installation, a browser rendering check or a
GPU inference check. The lifespan was not started. The report applies to the
wheel hash recorded in it, not automatically to later builds.

Build and verify a new wheel from the repository root with:

    python -m pip wheel . --no-deps --wheel-dir artifacts/packaging
    python scripts/check_installed_wheel.py
