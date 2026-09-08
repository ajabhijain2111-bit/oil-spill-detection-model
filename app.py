import streamlit as st
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import json
import cv2

from rasterio.transform import xy
from pyproj import Transformer, Geod

from model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "best_oil_spill_unet.pth"
PATCH_SIZE = 512

# Remove very small predicted regions
MIN_COMPONENT_AREA = 500

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Oil Spill Detection System",
    page_icon="🛢️",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🛢️ Oil Spill Detection System")

st.write(
    "AI-based oil spill detection and segmentation "
    "using Sentinel-1 SAR imagery and U-Net."
)

st.caption(
    f"Device: {DEVICE}"
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = UNet().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# CALCULATE GEOGRAPHIC INFORMATION
# ============================================================

def calculate_geo_information(
    transform,
    crs,
    centroid_x,
    centroid_y,
    height,
    width,
    oil_pixels
):

    result = {
        "georeferenced": False,
        "crs": None,
        "latitude": None,
        "longitude": None,
        "bbox_latitude_min": None,
        "bbox_latitude_max": None,
        "bbox_longitude_min": None,
        "bbox_longitude_max": None,
        "pixel_size_x": None,
        "pixel_size_y": None,
        "estimated_area_km2": None
    }

    # --------------------------------------------------------
    # Check georeferencing
    # --------------------------------------------------------

    if crs is None:
        return result

    if transform is None:
        return result

    if transform == rasterio.Affine.identity():
        return result

    result["georeferenced"] = True
    result["crs"] = str(crs)

    # --------------------------------------------------------
    # Pixel size
    # --------------------------------------------------------

    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)

    result["pixel_size_x"] = float(pixel_width)
    result["pixel_size_y"] = float(pixel_height)

    # --------------------------------------------------------
    # Convert centroid to geographic coordinates
    # --------------------------------------------------------

    try:

        if crs.is_geographic:

            longitude, latitude = xy(
                transform,
                centroid_y,
                centroid_x,
                offset="center"
            )

        else:

            transformer = Transformer.from_crs(
                crs,
                "EPSG:4326",
                always_xy=True
            )

            map_x, map_y = xy(
                transform,
                centroid_y,
                centroid_x,
                offset="center"
            )

            longitude, latitude = transformer.transform(
                map_x,
                map_y
            )

        result["latitude"] = float(latitude)
        result["longitude"] = float(longitude)

    except Exception:
        return result

    # --------------------------------------------------------
    # Geographic bounding box
    # --------------------------------------------------------

    try:

        corners = [
            xy(transform, 0, 0, offset="center"),
            xy(transform, 0, width - 1, offset="center"),
            xy(transform, height - 1, 0, offset="center"),
            xy(transform, height - 1, width - 1, offset="center")
        ]

        if crs.is_geographic:

            geo_corners = [
                (point[0], point[1])
                for point in corners
            ]

        else:

            transformer = Transformer.from_crs(
                crs,
                "EPSG:4326",
                always_xy=True
            )

            geo_corners = [
                transformer.transform(
                    point[0],
                    point[1]
                )
                for point in corners
            ]

        longitudes = [
            point[0]
            for point in geo_corners
        ]

        latitudes = [
            point[1]
            for point in geo_corners
        ]

        result["bbox_longitude_min"] = float(
            min(longitudes)
        )

        result["bbox_longitude_max"] = float(
            max(longitudes)
        )

        result["bbox_latitude_min"] = float(
            min(latitudes)
        )

        result["bbox_latitude_max"] = float(
            max(latitudes)
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # Estimate spill area
    # --------------------------------------------------------

    try:

        if crs.is_projected:

            # Exact pixel area for a north-up projected raster
            pixel_area_m2 = (
                abs(transform.a * transform.e)
            )

            area_m2 = (
                oil_pixels * pixel_area_m2
            )

            result["estimated_area_km2"] = float(
                area_m2 / 1_000_000
            )

        else:

            # Geographic CRS:
            # estimate pixel ground dimensions at centroid
            geod = Geod(ellps="WGS84")

            lat = result["latitude"]
            lon = result["longitude"]

            lon1, lat1 = lon, lat
            lon2, lat2 = lon + pixel_width, lat
            lon3, lat3 = lon, lat + pixel_height

            _, _, horizontal_distance = geod.inv(
                lon1,
                lat1,
                lon2,
                lat2
            )

            _, _, vertical_distance = geod.inv(
                lon1,
                lat1,
                lon3,
                lat3
            )

            pixel_area_m2 = (
                abs(horizontal_distance)
                * abs(vertical_distance)
            )

            area_m2 = (
                oil_pixels * pixel_area_m2
            )

            result["estimated_area_km2"] = float(
                area_m2 / 1_000_000
            )

    except Exception:
        pass

    return result


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(image_path, progress_bar):

    model = load_model()

    # --------------------------------------------------------
    # Read GeoTIFF
    # --------------------------------------------------------

    with rasterio.open(image_path) as src:

        image = src.read().astype(np.float32)

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

        input_bounds = src.bounds

    # --------------------------------------------------------
    # Validate number of bands
    # --------------------------------------------------------

    if image.shape[0] != 2:

        raise ValueError(
            f"Expected exactly 2 SAR bands, "
            f"but found {image.shape[0]} bands."
        )

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    height = image.shape[1]
    width = image.shape[2]

    if (
        height % PATCH_SIZE != 0
        or width % PATCH_SIZE != 0
    ):

        raise ValueError(
            f"Image dimensions must be divisible by "
            f"{PATCH_SIZE}. "
            f"Received {height} × {width}."
        )

    # --------------------------------------------------------
    # Number of patches
    # --------------------------------------------------------

    rows = height // PATCH_SIZE
    cols = width // PATCH_SIZE

    total_patches = rows * cols

    # --------------------------------------------------------
    # Full prediction
    # --------------------------------------------------------

    raw_prediction = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    completed = 0

    # --------------------------------------------------------
    # Patch inference
    # --------------------------------------------------------

    with torch.no_grad():

        for r in range(rows):

            for c in range(cols):

                y1 = r * PATCH_SIZE
                y2 = (r + 1) * PATCH_SIZE

                x1 = c * PATCH_SIZE
                x2 = (c + 1) * PATCH_SIZE

                patch = image[
                    :,
                    y1:y2,
                    x1:x2
                ]

                tensor = torch.from_numpy(
                    patch
                ).unsqueeze(0).to(DEVICE)

                output = model(tensor)

                probability = torch.sigmoid(
                    output
                )

                prediction = (
                    probability > 0.5
                ).float()

                raw_prediction[
                    y1:y2,
                    x1:x2
                ] = (
                    prediction[
                        0,
                        0
                    ]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                completed += 1

                progress_bar.progress(
                    completed / total_patches,
                    text=(
                        f"Processing patch "
                        f"{completed}/{total_patches}"
                    )
                )

    # ========================================================
    # RAW STATISTICS
    # ========================================================

    raw_oil_pixels = int(
        np.sum(raw_prediction == 1)
    )

    total_pixels = raw_prediction.size

    raw_oil_percentage = (
        raw_oil_pixels / total_pixels
    ) * 100

    # ========================================================
    # REMOVE SMALL COMPONENTS
    # ========================================================

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            raw_prediction,
            connectivity=8
        )
    )

    clean_prediction = np.zeros_like(
        raw_prediction
    )

    component_information = []

    for label in range(1, num_labels):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        x = int(
            stats[
                label,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                label,
                cv2.CC_STAT_TOP
            ]
        )

        w = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH
            ]
        )

        h = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT
            ]
        )

        component_information.append({
            "component": label,
            "pixels": area,
            "percentage": (
                area / total_pixels
            ) * 100,
            "x": x,
            "y": y,
            "width": w,
            "height": h
        })

        if area >= MIN_COMPONENT_AREA:

            clean_prediction[
                labels == label
            ] = 1

    # Use cleaned prediction from this point
    full_prediction = clean_prediction

    # ========================================================
    # CLEAN STATISTICS
    # ========================================================

    oil_pixels = int(
        np.sum(full_prediction == 1)
    )

    oil_percentage = (
        oil_pixels / total_pixels
    ) * 100

    # ========================================================
    # SORT COMPONENTS
    # ========================================================

    component_information.sort(
        key=lambda x: x["pixels"],
        reverse=True
    )

    # Re-number for reporting
    largest_components = []

    for i, component in enumerate(
        component_information[:10],
        start=1
    ):

        largest_components.append({
            "rank": i,
            "original_component":
                component["component"],
            "pixels":
                component["pixels"],
            "percentage":
                component["percentage"],
            "x":
                component["x"],
            "y":
                component["y"],
            "width":
                component["width"],
            "height":
                component["height"]
        })

    # ========================================================
    # CENTROID
    # ========================================================

    ys, xs = np.where(
        full_prediction == 1
    )

    if len(xs) > 0:

        centroid_x = float(
            xs.mean()
        )

        centroid_y = float(
            ys.mean()
        )

    else:

        centroid_x = None
        centroid_y = None

    # ========================================================
    # PIXEL BOUNDING BOX
    # ========================================================

    if len(xs) > 0:

        bbox_x_min = int(xs.min())
        bbox_x_max = int(xs.max())

        bbox_y_min = int(ys.min())
        bbox_y_max = int(ys.max())

    else:

        bbox_x_min = None
        bbox_x_max = None
        bbox_y_min = None
        bbox_y_max = None

    # ========================================================
    # GEOGRAPHIC INFORMATION
    # ========================================================

    if centroid_x is not None:

        geo_information = (
            calculate_geo_information(
                transform,
                crs,
                centroid_x,
                centroid_y,
                height,
                width,
                oil_pixels
            )
        )

    else:

        geo_information = {
            "georeferenced": False,
            "crs": None,
            "latitude": None,
            "longitude": None,
            "bbox_latitude_min": None,
            "bbox_latitude_max": None,
            "bbox_longitude_min": None,
            "bbox_longitude_max": None,
            "pixel_size_x": None,
            "pixel_size_y": None,
            "estimated_area_km2": None
        }

    # ========================================================
    # FIND CONTOURS
    # ========================================================

    contours, _ = cv2.findContours(
        full_prediction,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # ========================================================
    # OUTPUT GEOTIFF
    # ========================================================

    profile.update(
        count=1,
        dtype="uint8",
        nodata=0
    )

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".tif"
    )

    output_path = output_file.name

    output_file.close()

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            full_prediction,
            1
        )

    return {
        "image": image,
        "prediction": full_prediction,
        "raw_prediction": raw_prediction,
        "oil_pixels": oil_pixels,
        "oil_percentage": oil_percentage,
        "raw_oil_pixels": raw_oil_pixels,
        "raw_oil_percentage": raw_oil_percentage,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "bbox_x_min": bbox_x_min,
        "bbox_x_max": bbox_x_max,
        "bbox_y_min": bbox_y_min,
        "bbox_y_max": bbox_y_max,
        "contours": contours,
        "components": component_information,
        "largest_components": largest_components,
        "geo": geo_information,
        "crs": str(crs) if crs else None,
        "transform": str(transform),
        "bounds": {
            "left": float(input_bounds.left),
            "bottom": float(input_bounds.bottom),
            "right": float(input_bounds.right),
            "top": float(input_bounds.top)
        },
        "height": height,
        "width": width,
        "output_path": output_path
    }


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload Sentinel-1 GeoTIFF",
    type=["tif", "tiff"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button(
        "🔍 Detect Oil Spill",
        type="primary"
    ):

        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".tif"
        )

        temp_input.write(
            uploaded_file.getvalue()
        )

        temp_input.close()

        progress_bar = st.progress(
            0,
            text="Starting prediction..."
        )

        try:

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            with st.spinner(
                "Running U-Net prediction..."
            ):

                result = predict(
                    temp_input.name,
                    progress_bar
                )

            progress_bar.empty()

            # ------------------------------------------------
            # Extract results
            # ------------------------------------------------

            image = result["image"]

            prediction = result["prediction"]

            oil_pixels = result["oil_pixels"]

            oil_percentage = result[
                "oil_percentage"
            ]

            raw_oil_pixels = result[
                "raw_oil_pixels"
            ]

            raw_oil_percentage = result[
                "raw_oil_percentage"
            ]

            centroid_x = result[
                "centroid_x"
            ]

            centroid_y = result[
                "centroid_y"
            ]

            contours = result[
                "contours"
            ]

            geo = result[
                "geo"
            ]

            st.success(
                "✅ Oil spill detection completed!"
            )

            # =================================================
            # DETECTION STATUS
            # =================================================

            if oil_pixels > 0:

                st.success(
                    "🟢 Oil spill detected"
                )

            else:

                st.info(
                    "🔵 No oil spill detected"
                )

            # =================================================
            # MAIN RESULTS
            # =================================================

            st.subheader(
                "📊 Detection Results"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Detected Oil Pixels",
                    f"{oil_pixels:,}"
                )

            with col2:

                st.metric(
                    "Oil Coverage",
                    f"{oil_percentage:.2f}%"
                )

            with col3:

                if centroid_x is not None:

                    st.metric(
                        "Spill Centroid",
                        (
                            f"({centroid_x:.1f}, "
                            f"{centroid_y:.1f})"
                        )
                    )

                else:

                    st.metric(
                        "Spill Centroid",
                        "Not detected"
                    )

            # =================================================
            # INPUT INFORMATION
            # =================================================

            st.divider()

            st.subheader(
                "🛰️ Input Information"
            )

            info1, info2, info3, info4 = (
                st.columns(4)
            )

            with info1:

                st.write(
                    f"**Bands:** {image.shape[0]}"
                )

            with info2:

                st.write(
                    f"**Height:** {image.shape[1]}"
                )

            with info3:

                st.write(
                    f"**Width:** {image.shape[2]}"
                )

            with info4:

                st.write(
                    f"**CRS:** "
                    f"{geo['crs'] or 'Not available'}"
                )

            # =================================================
            # RAW VS CLEANED
            # =================================================

            with st.expander(
                "🔬 Raw vs Cleaned Prediction"
            ):

                a, b = st.columns(2)

                with a:

                    st.write(
                        f"Raw oil pixels: "
                        f"**{raw_oil_pixels:,}**"
                    )

                    st.write(
                        f"Raw oil percentage: "
                        f"**{raw_oil_percentage:.2f}%**"
                    )

                with b:

                    st.write(
                        f"Cleaned oil pixels: "
                        f"**{oil_pixels:,}**"
                    )

                    st.write(
                        f"Cleaned oil percentage: "
                        f"**{oil_percentage:.2f}%**"
                    )

            # =================================================
            # GEOLOCATION
            # =================================================

            st.divider()

            st.subheader(
                "🌍 Geographic Information"
            )

            if geo["georeferenced"]:

                g1, g2, g3 = st.columns(3)

                with g1:

                    st.metric(
                        "Latitude",
                        f"{geo['latitude']:.6f}"
                    )

                with g2:

                    st.metric(
                        "Longitude",
                        f"{geo['longitude']:.6f}"
                    )

                with g3:

                    if (
                        geo["estimated_area_km2"]
                        is not None
                    ):

                        st.metric(
                            "Estimated Spill Area",
                            (
                                f"{geo['estimated_area_km2']:.4f} "
                                f"km²"
                            )
                        )

                    else:

                        st.metric(
                            "Estimated Spill Area",
                            "Unavailable"
                        )

                st.success(
                    "✅ Input GeoTIFF is properly georeferenced."
                )

                st.write(
                    "**Spill centroid:** "
                    f"{geo['latitude']:.6f}, "
                    f"{geo['longitude']:.6f}"
                )

                st.write(
                    "**Coordinate system:** "
                    f"{geo['crs']}"
                )

                # --------------------------------------------
                # Geographic bounding box
                # --------------------------------------------

                st.write(
                    "**Geographic bounding box:**"
                )

                st.code(
                    (
                        f"Latitude: "
                        f"{geo['bbox_latitude_min']:.6f} "
                        f"to "
                        f"{geo['bbox_latitude_max']:.6f}\n"
                        f"Longitude: "
                        f"{geo['bbox_longitude_min']:.6f} "
                        f"to "
                        f"{geo['bbox_longitude_max']:.6f}"
                    )
                )

                # --------------------------------------------
                # Map
                # --------------------------------------------

                st.subheader(
                    "🗺️ Spill Location"
                )

                map_data = {
                    "latitude": [
                        geo["latitude"]
                    ],
                    "longitude": [
                        geo["longitude"]
                    ]
                }

                st.map(
                    map_data,
                    latitude="latitude",
                    longitude="longitude",
                    zoom=8
                )

            else:

                st.warning(
                    "⚠️ This GeoTIFF does not contain "
                    "a valid geotransform. "
                    "Latitude/longitude and real-world "
                    "spill area cannot be calculated."
                )

                st.info(
                    "The centroid currently shown is "
                    "in pixel coordinates only."
                )

                st.write(
                    "**Pixel centroid:** "
                    f"X = {centroid_x}, "
                    f"Y = {centroid_y}"
                )

            # =================================================
            # PIXEL BOUNDING BOX
            # =================================================

            st.divider()

            st.subheader(
                "📦 Spill Bounding Box"
            )

            if centroid_x is not None:

                b1, b2 = st.columns(2)

                with b1:

                    st.write(
                        f"**X:** "
                        f"{result['bbox_x_min']} "
                        f"to "
                        f"{result['bbox_x_max']}"
                    )

                with b2:

                    st.write(
                        f"**Y:** "
                        f"{result['bbox_y_min']} "
                        f"to "
                        f"{result['bbox_y_max']}"
                    )

            # =================================================
            # VISUALIZATION
            # =================================================

            st.divider()

            st.subheader(
                "🖼️ Prediction Visualization"
            )

            col1, col2 = st.columns(2)

            # -------------------------------------------------
            # SAR
            # -------------------------------------------------

            with col1:

                fig1, ax1 = plt.subplots()

                ax1.imshow(
                    image[0],
                    cmap="gray"
                )

                ax1.set_title(
                    "Sentinel-1 SAR Image"
                )

                ax1.axis("off")

                st.pyplot(
                    fig1,
                    use_container_width=True
                )

                plt.close(fig1)

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------

            with col2:

                fig2, ax2 = plt.subplots()

                ax2.imshow(
                    prediction,
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )

                ax2.set_title(
                    "Predicted Oil Spill Mask"
                )

                ax2.axis("off")

                st.pyplot(
                    fig2,
                    use_container_width=True
                )

                plt.close(fig2)

            # =================================================
            # OVERLAY
            # =================================================

            st.subheader(
                "🛢️ Oil Spill Overlay"
            )

            fig3, ax3 = plt.subplots()

            ax3.imshow(
                image[0],
                cmap="gray"
            )

            overlay = np.ma.masked_where(
                prediction == 0,
                prediction
            )

            ax3.imshow(
                overlay,
                alpha=0.45,
                cmap="Reds",
                vmin=0,
                vmax=1
            )

            ax3.set_title(
                "Detected Oil Spill Overlay"
            )

            ax3.axis("off")

            st.pyplot(
                fig3,
                use_container_width=True
            )

            plt.close(fig3)

            # =================================================
            # BOUNDARY
            # =================================================

            st.subheader(
                "🔴 Detected Oil Spill Boundary"
            )

            fig4, ax4 = plt.subplots()

            ax4.imshow(
                image[0],
                cmap="gray"
            )

            overlay = np.ma.masked_where(
                prediction == 0,
                prediction
            )

            ax4.imshow(
                overlay,
                alpha=0.25,
                cmap="Reds",
                vmin=0,
                vmax=1
            )

            for contour in contours:

                contour = contour.squeeze()

                if (
                    contour.ndim == 2
                    and len(contour) > 2
                ):

                    ax4.plot(
                        contour[:, 0],
                        contour[:, 1],
                        linewidth=2,
                        color="red"
                    )

            ax4.set_title(
                "Detected Oil Spill Boundary"
            )

            ax4.axis("off")

            st.pyplot(
                fig4,
                use_container_width=True
            )

            plt.close(fig4)

            # =================================================
            # COMPONENT ANALYSIS
            # =================================================

            st.divider()

            st.subheader(
                "🧩 Spill Component Analysis"
            )

            total_components = len(
                result["components"]
            )

            st.write(
                f"**Total connected components: "
                f"{total_components}**"
            )

            st.write(
                f"Minimum component area used for "
                f"cleaning: **{MIN_COMPONENT_AREA} pixels**"
            )

            if result["largest_components"]:

                st.write(
                    "**Largest predicted regions:**"
                )

                for component in result[
                    "largest_components"
                ]:

                    st.write(
                        f"Component {component['rank']} — "
                        f"{component['pixels']:,} pixels "
                        f"({component['percentage']:.4f}%)"
                    )

            # =================================================
            # JSON REPORT
            # =================================================

            report = {

                "model":
                    "U-Net Oil Spill Detector",

                "input_file":
                    uploaded_file.name,

                "device":
                    str(DEVICE),

                "image": {
                    "bands":
                        int(image.shape[0]),

                    "height":
                        int(image.shape[1]),

                    "width":
                        int(image.shape[2])
                },

                "prediction": {

                    "oil_detected":
                        bool(oil_pixels > 0),

                    "raw_oil_pixels":
                        int(raw_oil_pixels),

                    "raw_oil_percentage":
                        float(raw_oil_percentage),

                    "cleaned_oil_pixels":
                        int(oil_pixels),

                    "cleaned_oil_percentage":
                        float(oil_percentage)
                },

                "centroid": {

                    "pixel_x":
                        centroid_x,

                    "pixel_y":
                        centroid_y,

                    "latitude":
                        geo["latitude"],

                    "longitude":
                        geo["longitude"]
                },

                "pixel_bounding_box": {

                    "x_min":
                        result["bbox_x_min"],

                    "x_max":
                        result["bbox_x_max"],

                    "y_min":
                        result["bbox_y_min"],

                    "y_max":
                        result["bbox_y_max"]
                },

                "georeferencing": {

                    "available":
                        geo["georeferenced"],

                    "crs":
                        geo["crs"],

                    "latitude_min":
                        geo["bbox_latitude_min"],

                    "latitude_max":
                        geo["bbox_latitude_max"],

                    "longitude_min":
                        geo["bbox_longitude_min"],

                    "longitude_max":
                        geo["bbox_longitude_max"]
                },

                "area": {

                    "estimated_area_km2":
                        geo["estimated_area_km2"]
                },

                "components": {

                    "total":
                        len(result["components"]),

                    "minimum_area":
                        MIN_COMPONENT_AREA,

                    "largest":
                        result[
                            "largest_components"
                        ]
                }

            }

            report_json = json.dumps(
                report,
                indent=4
            )

            # =================================================
            # DOWNLOADS
            # =================================================

            st.divider()

            st.subheader(
                "⬇️ Download Results"
            )

            download1, download2 = (
                st.columns(2)
            )

            # -------------------------------------------------
            # TIFF
            # -------------------------------------------------

            with download1:

                with open(
                    result["output_path"],
                    "rb"
                ) as f:

                    st.download_button(
                        label=(
                            "⬇️ Download "
                            "Prediction TIFF"
                        ),
                        data=f.read(),
                        file_name=(
                            "oil_spill_prediction.tif"
                        ),
                        mime="image/tiff"
                    )

            # -------------------------------------------------
            # JSON
            # -------------------------------------------------

            with download2:

                st.download_button(
                    label=(
                        "📄 Download "
                        "Complete JSON Report"
                    ),
                    data=report_json,
                    file_name=(
                        "oil_spill_report.json"
                    ),
                    mime="application/json"
                )

            # =================================================
            # REPORT PREVIEW
            # =================================================

            with st.expander(
                "📄 View Complete JSON Report"
            ):

                st.code(
                    report_json,
                    language="json"
                )

        except Exception as e:

            progress_bar.empty()

            st.error(
                f"❌ Prediction failed: {e}"
            )

        finally:

            if os.path.exists(
                temp_input.name
            ):

                os.remove(
                    temp_input.name
                )