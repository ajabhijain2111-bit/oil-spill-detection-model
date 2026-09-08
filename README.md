# U-Net Oil Spill Detection Model

## Overview

This module uses a trained U-Net model to detect oil spills from
2-band GeoTIFF satellite imagery.

## Input

The model expects:

- GeoTIFF image
- 2 image bands
- Satellite image with georeferencing when latitude/longitude are required

Example:

00000.tif

## Model

Model architecture:

U-Net

Input channels:

2

Output:

Binary oil-spill segmentation mask.

## Basic Usage

```python
from inference import detect_oil

result = detect_oil("00000.tif")

# Oil Spill Detection Model

A deep-learning-based oil spill detection and analysis system using a U-Net segmentation model and geospatial processing.

## Overview

This repository contains the ML component of the oil spill detection system.

The model takes a georeferenced satellite `.tif` image with 2 input bands and produces:

- Oil spill segmentation mask
- Oil pixel count
- Oil percentage
- Spill centroid
- Latitude and longitude
- Spill area in km²
- Georeferencing information
- Spill geometry
- Principal-axis orientation
- Candidate spill-origin locations
- JSON report

The model is designed so that another application/backend can call the inference function and consume the returned Python dictionary/JSON.

---

## Project Structure

```text
oil-spill-detection-model/
│
├── best_oil_spill_unet.pth
├── model.py
├── inference.py
├── spill_geometry.py
├── origin_estimator.py
├── team_test.py
├── main.py
├── app.py
│
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── .gitattributes
└── README.md