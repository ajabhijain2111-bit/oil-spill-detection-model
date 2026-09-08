import os
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform
from scipy import ndimage
from pyproj import Geod

from spill_geometry import get_spill_geometry
from origin_estimator import estimate_origin
# ============================================================
# MODEL
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(2, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(
            1024, 512, 2, stride=2
        )
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(
            512, 256, 2, stride=2
        )
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)
        d4 = torch.cat(
            [d4, e4],
            dim=1
        )
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat(
            [d3, e3],
            dim=1
        )
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat(
            [d2, e2],
            dim=1
        )
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat(
            [d1, e1],
            dim=1
        )
        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "best_oil_spill_unet.pth"

PATCH_SIZE = 512

MIN_COMPONENT_SIZE = 500

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = UNet().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.eval()

    return model


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(model, image):

    bands, height, width = image.shape

    prediction = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for y in range(
        0,
        height,
        PATCH_SIZE
    ):

        for x in range(
            0,
            width,
            PATCH_SIZE
        ):

            patch = image[
                :,
                y:min(
                    y + PATCH_SIZE,
                    height
                ),
                x:min(
                    x + PATCH_SIZE,
                    width
                )
            ]

            ph = patch.shape[1]
            pw = patch.shape[2]

            padded = np.zeros(
                (
                    bands,
                    PATCH_SIZE,
                    PATCH_SIZE
                ),
                dtype=np.float32
            )

            padded[
                :,
                :ph,
                :pw
            ] = patch

            tensor = torch.from_numpy(
                padded
            ).unsqueeze(0).float().to(device)

            with torch.no_grad():

                output = model(tensor)

                output = torch.sigmoid(
                    output
                )

                pred = (
                    output > 0.5
                ).cpu().numpy()[0, 0]

                pred = pred.astype(
                    np.uint8
                )

            prediction[
                y:y + ph,
                x:x + pw
            ] = pred[
                :ph,
                :pw
            ]

    return prediction


# ============================================================
# REMOVE SMALL COMPONENTS
# ============================================================

def clean_prediction(prediction):

    structure = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    labeled, num_components = ndimage.label(
        prediction,
        structure=structure
    )

    sizes = np.bincount(
        labeled.ravel()
    )

    cleaned = np.zeros_like(
        prediction
    )

    for component_id in range(
        1,
        len(sizes)
    ):

        if (
            sizes[component_id]
            >= MIN_COMPONENT_SIZE
        ):

            cleaned[
                labeled == component_id
            ] = 1

    return (
        cleaned,
        int(num_components)
    )


# ============================================================
# PIXEL -> LATITUDE / LONGITUDE
# ============================================================

def pixel_to_latlon(
    row,
    col,
    transform_obj,
    src_crs
):

    x, y = xy(
        transform_obj,
        row,
        col,
        offset="center"
    )

    if src_crs is None:
        return None, None

    lon, lat = transform(
        src_crs,
        "EPSG:4326",
        [x],
        [y]
    )

    return (
        float(lat[0]),
        float(lon[0])
    )


# ============================================================
# CALCULATE PIXEL AREA
# ============================================================

def calculate_pixel_area_km2(
    row,
    col,
    transform_obj,
    src_crs
):

    if src_crs is None:
        return None

    geod = Geod(
        ellps="WGS84"
    )

    # Get pixel corners in source CRS
    corners = [
        xy(
            transform_obj,
            row,
            col,
            offset="ul"
        ),
        xy(
            transform_obj,
            row,
            col,
            offset="ur"
        ),
        xy(
            transform_obj,
            row,
            col,
            offset="lr"
        ),
        xy(
            transform_obj,
            row,
            col,
            offset="ll"
        )
    ]

    xs = [
        point[0]
        for point in corners
    ]

    ys = [
        point[1]
        for point in corners
    ]

    # Transform corners to WGS84
    lons, lats = transform(
        src_crs,
        "EPSG:4326",
        xs,
        ys
    )

    area_m2, _ = geod.polygon_area_perimeter(
        lons,
        lats
    )

    area_m2 = abs(
        float(area_m2)
    )

    return area_m2 / 1_000_000.0


# ============================================================
# MAIN OIL DETECTION FUNCTION
# ============================================================

def detect_oil(
    input_path,
    output_path="oil_spill_prediction.tif",
    json_path="oil_spill_report.json"
):

    if not os.path.exists(input_path):

        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    print("Loading model...")

    model = load_model()

    print("Model loaded successfully")

    # ========================================================
    # READ IMAGE
    # ========================================================

    with rasterio.open(
        input_path
    ) as src:

        image = src.read()

        profile = src.profile.copy()

        transform_obj = src.transform

        src_crs = src.crs

        height = src.height

        width = src.width

        bounds = src.bounds

    print(
        f"Input shape: {image.shape}"
    )

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if image.ndim != 3:

        raise ValueError(
            "Input image must have shape "
            "(bands, height, width)"
        )

    if image.shape[0] != 2:

        raise ValueError(
            f"U-Net expects exactly 2 bands. "
            f"Received {image.shape[0]} bands."
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = predict_image(
        model,
        image
    )

    raw_oil_pixels = int(
        np.sum(prediction)
    )

    total_pixels = int(
        height * width
    )

    raw_oil_percentage = float(
        raw_oil_pixels
        / total_pixels
        * 100.0
    )

    # ========================================================
    # CLEAN PREDICTION
    # ========================================================

    cleaned_prediction, num_components = (
        clean_prediction(
            prediction
        )
    )

    cleaned_oil_pixels = int(
        np.sum(cleaned_prediction)
    )

    cleaned_oil_percentage = float(
        cleaned_oil_pixels
        / total_pixels
        * 100.0
    )

    oil_detected = (
        cleaned_oil_pixels > 0
    )

    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    pixel_x = None
    pixel_y = None

    latitude = None
    longitude = None

    x_min = None
    x_max = None
    y_min = None
    y_max = None

    latitude_min = None
    latitude_max = None

    longitude_min = None
    longitude_max = None

    area_km2 = None

    # ========================================================
    # OIL ANALYSIS
    # ========================================================

    if oil_detected:

        rows, cols = np.where(
            cleaned_prediction == 1
        )

        # ----------------------------------------------------
        # CENTROID
        # ----------------------------------------------------

        pixel_y = float(
            np.mean(rows)
        )

        pixel_x = float(
            np.mean(cols)
        )

        # ----------------------------------------------------
        # PIXEL BOUNDING BOX
        # ----------------------------------------------------

        x_min = int(
            np.min(cols)
        )

        x_max = int(
            np.max(cols)
        )

        y_min = int(
            np.min(rows)
        )

        y_max = int(
            np.max(rows)
        )

        # ----------------------------------------------------
        # CENTROID LAT/LON
        # ----------------------------------------------------

        latitude, longitude = pixel_to_latlon(
            pixel_y,
            pixel_x,
            transform_obj,
            src_crs
        )

        # ----------------------------------------------------
        # GEOGRAPHIC BOUNDING BOX
        # ----------------------------------------------------

        lat_a, lon_a = pixel_to_latlon(
            y_min,
            x_min,
            transform_obj,
            src_crs
        )

        lat_b, lon_b = pixel_to_latlon(
            y_max,
            x_max,
            transform_obj,
            src_crs
        )

        if lat_a is not None:

            latitude_min = float(
                min(lat_a, lat_b)
            )

            latitude_max = float(
                max(lat_a, lat_b)
            )

            longitude_min = float(
                min(lon_a, lon_b)
            )

            longitude_max = float(
                max(lon_a, lon_b)
            )

        # ----------------------------------------------------
        # AREA
        # ----------------------------------------------------

        # Calculate actual area of a representative
        # pixel using the GeoTIFF transform.

        representative_row = int(
            np.clip(
                round(pixel_y),
                0,
                height - 1
            )
        )

        representative_col = int(
            np.clip(
                round(pixel_x),
                0,
                width - 1
            )
        )

        pixel_area_km2 = calculate_pixel_area_km2(
            representative_row,
            representative_col,
            transform_obj,
            src_crs
        )

        if pixel_area_km2 is not None:

            area_km2 = float(
                cleaned_oil_pixels
                * pixel_area_km2
            )

    # ========================================================
    # SAVE PREDICTION TIFF
    # ========================================================

    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress="lzw"
    )

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            cleaned_prediction,
            1
        )
    # ========================================================
    # SPILL ORIGIN ESTIMATION
    # ========================================================

    origin = estimate_origin(
        cleaned_prediction,
        transform_obj,
        src_crs
    )
    # ========================================================
    # RESULT
    # ========================================================
    geometry = get_spill_geometry(cleaned_prediction)
    result = {

        "model":
            "U-Net Oil Spill Detector",

        "input_file":
            os.path.basename(input_path),

        "device":
            str(device),

        "image": {

            "bands":
                int(image.shape[0]),

            "height":
                int(height),

            "width":
                int(width)
        },

        "prediction": {

            "oil_detected":
                bool(oil_detected),

            "raw_oil_pixels":
                int(raw_oil_pixels),

            "raw_oil_percentage":
                float(raw_oil_percentage),

            "cleaned_oil_pixels":
                int(cleaned_oil_pixels),

            "cleaned_oil_percentage":
                float(cleaned_oil_percentage)
        },

        "centroid": {

            "pixel_x":
                pixel_x,

            "pixel_y":
                pixel_y,

            "latitude":
                latitude,

            "longitude":
                longitude
        },

        "pixel_bounding_box": {

            "x_min":
                x_min,

            "x_max":
                x_max,

            "y_min":
                y_min,

            "y_max":
                y_max
        },

        "georeferencing": {

            "available":
                bool(src_crs is not None),

            "crs":
                str(src_crs)
                if src_crs is not None
                else None,

            "latitude_min":
                latitude_min,

            "latitude_max":
                latitude_max,

            "longitude_min":
                longitude_min,

            "longitude_max":
                longitude_max
        },

        "area": {

            "estimated_area_km2":
                area_km2
        },

        "components": {

            "total":
                int(num_components),

            "minimum_area":
                int(MIN_COMPONENT_SIZE)
        },
        "geometry": geometry,
                
        "origin": origin,

        "output": {

            "prediction_mask":
                output_path,

            "json_report":
                json_path
        }
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
            allow_nan=False
        )

    # ========================================================
    # PRINT RESULT
    # ========================================================

    print("\nPrediction completed")

    print(
        "Oil detected:",
        oil_detected
    )

    print(
        "Oil pixels:",
        cleaned_oil_pixels
    )

    print(
        "Oil percentage:",
        cleaned_oil_percentage
    )

    if latitude is not None:

        print(
            "Latitude:",
            latitude
        )

        print(
            "Longitude:",
            longitude
        )

    if area_km2 is not None:

        print(
            "Area:",
            area_km2,
            "km²"
        )

    print(
        "Saved:",
        output_path
    )

    print(
        "Saved:",
        json_path
    )

    return result


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "uv run python inference.py <input.tif>"
        )

        sys.exit(1)

    input_file = sys.argv[1]

    detect_oil(
        input_file
    )