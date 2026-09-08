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

print(result)