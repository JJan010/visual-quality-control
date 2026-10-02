# Visual Quality Control

Visual anomaly detection for manufacturing quality control.

## Status

Initial project setup. Models and application are not implemented yet.

## Goal

Build a reproducible system that detects and localizes visual anomalies
in product images, records inspection results, and supports operator review.

## Planned scope

- Explore the MVTec AD dataset, starting with the bottle category.
- Implement a convolutional autoencoder baseline in PyTorch.
- Compare the baseline with PatchCore.
- Evaluate detection quality, localization, and inference latency.
- Build a FastAPI service and a Streamlit interface.
- Add result storage, experiment tracking, and deployment tooling.

## Data

Dataset: MVTec AD
https://www.mvtec.com/research-teaching/datasets/mvtec-ad

Dataset license: CC BY-NC-SA 4.0.
Dataset files are not included in this repository.

## Project layout

- src/visual_quality/ - Python package
- configs/ - experiment configurations
- tests/ - automated tests
- docs/ - documentation and reports
- data/ - local datasets, excluded from Git
- artifacts/ - generated outputs, excluded from Git
