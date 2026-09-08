import numpy as np
import cv2
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform


def get_largest_contour(mask):
    mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None

    return max(
        contours,
        key=cv2.contourArea
    )


def calculate_principal_axis(contour):
    points = contour.reshape(-1, 2).astype(np.float64)

    center = np.mean(points, axis=0)

    centered = points - center

    covariance = np.cov(
        centered,
        rowvar=False
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    largest_index = np.argmax(eigenvalues)

    direction = eigenvectors[:, largest_index]

    direction = direction / np.linalg.norm(direction)

    return center, direction


def get_axis_endpoints(contour):
    center, direction = calculate_principal_axis(
        contour
    )

    points = contour.reshape(-1, 2).astype(
        np.float64
    )

    projections = np.dot(
        points - center,
        direction
    )

    min_projection = np.min(
        projections
    )

    max_projection = np.max(
        projections
    )

    endpoint_1 = (
        center
        + direction * min_projection
    )

    endpoint_2 = (
        center
        + direction * max_projection
    )

    return (
        center,
        endpoint_1,
        endpoint_2
    )


def pixel_to_latlon(
    row,
    col,
    transform_obj,
    crs
):

    x, y = xy(
        transform_obj,
        row,
        col,
        offset="center"
    )

    if crs is None:
        return None, None

    longitude, latitude = transform(
        crs,
        "EPSG:4326",
        [x],
        [y]
    )

    return (
        float(latitude[0]),
        float(longitude[0])
    )


def calculate_distance_pixels(
    point1,
    point2
):

    return float(
        np.linalg.norm(
            np.array(point1)
            - np.array(point2)
        )
    )


def estimate_origin(
    mask,
    transform_obj,
    crs
):

    height, width = mask.shape

    contour = get_largest_contour(mask)

    if contour is None:

        return {
            "available": False,
            "method": "principal_axis",
            "message": "No spill region detected",
            "candidates": []
        }

    center, endpoint_1, endpoint_2 = (
        get_axis_endpoints(contour)
    )

    # --------------------------------------------------------
    # Clamp candidate points to image boundaries
    # --------------------------------------------------------

    endpoint_1_x = float(
        np.clip(
            endpoint_1[0],
            0,
            width - 1
        )
    )

    endpoint_1_y = float(
        np.clip(
            endpoint_1[1],
            0,
            height - 1
        )
    )

    endpoint_2_x = float(
        np.clip(
            endpoint_2[0],
            0,
            width - 1
        )
    )

    endpoint_2_y = float(
        np.clip(
            endpoint_2[1],
            0,
            height - 1
        )
    )

    # --------------------------------------------------------
    # Clamp center as well
    # --------------------------------------------------------

    center_x = float(
        np.clip(
            center[0],
            0,
            width - 1
        )
    )

    center_y = float(
        np.clip(
            center[1],
            0,
            height - 1
        )
    )

    # --------------------------------------------------------
    # Convert to latitude / longitude
    # --------------------------------------------------------

    center_lat, center_lon = pixel_to_latlon(
        center_y,
        center_x,
        transform_obj,
        crs
    )

    lat_1, lon_1 = pixel_to_latlon(
        endpoint_1_y,
        endpoint_1_x,
        transform_obj,
        crs
    )

    lat_2, lon_2 = pixel_to_latlon(
        endpoint_2_y,
        endpoint_2_x,
        transform_obj,
        crs
    )

    # --------------------------------------------------------
    # Principal axis
    # --------------------------------------------------------

    direction_vector = (
        np.array([
            endpoint_2_x,
            endpoint_2_y
        ])
        -
        np.array([
            endpoint_1_x,
            endpoint_1_y
        ])
    )

    axis_length = float(
        np.linalg.norm(
            direction_vector
        )
    )

    orientation = np.degrees(
        np.arctan2(
            direction_vector[1],
            direction_vector[0]
        )
    )

    if orientation < 0:
        orientation += 180.0

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {

        "available": True,

        "method":
            "Principal Axis Candidate Origin Estimation",

        "warning":
            "True spill origin cannot be determined reliably "
            "from a single segmentation mask alone. The two "
            "endpoints are candidate origin locations. Wind, "
            "ocean currents, temporal imagery, or vessel "
            "trajectory data are required to determine the "
            "most likely source.",

        "spill_center": {

            "pixel_x":
                center_x,

            "pixel_y":
                center_y,

            "latitude":
                center_lat,

            "longitude":
                center_lon
        },

        "principal_axis": {

            "orientation_degrees":
                float(orientation),

            "length_pixels":
                float(axis_length)
        },

        "candidates": [

            {
                "candidate_id":
                    1,

                "pixel_x":
                    endpoint_1_x,

                "pixel_y":
                    endpoint_1_y,

                "latitude":
                    lat_1,

                "longitude":
                    lon_1
            },

            {
                "candidate_id":
                    2,

                "pixel_x":
                    endpoint_2_x,

                "pixel_y":
                    endpoint_2_y,

                "latitude":
                    lat_2,

                "longitude":
                    lon_2
            }
        ]
    }

def estimate_origin_from_file(
    prediction_file
):

    with rasterio.open(
        prediction_file
    ) as src:

        mask = src.read(1)

        transform_obj = src.transform

        crs = src.crs

    return estimate_origin(
        mask,
        transform_obj,
        crs
    )


if __name__ == "__main__":

    prediction_file = (
        "oil_spill_prediction.tif"
    )

    result = estimate_origin_from_file(
        prediction_file
    )

    print("\n========== SPILL ORIGIN ESTIMATION ==========")

    print(
        "Available:",
        result["available"]
    )

    print(
        "Method:",
        result["method"]
    )

    if result["available"]:

        center = result["spill_center"]

        print("\nSpill Center")

        print(
            "Latitude:",
            center["latitude"]
        )

        print(
            "Longitude:",
            center["longitude"]
        )

        axis = result["principal_axis"]

        print("\nPrincipal Axis")

        print(
            "Orientation:",
            axis["orientation_degrees"],
            "degrees"
        )

        print(
            "Length:",
            axis["length_pixels"],
            "pixels"
        )

        print("\nCandidate Origin 1")

        candidate_1 = result["candidates"][0]

        print(
            "Latitude:",
            candidate_1["latitude"]
        )

        print(
            "Longitude:",
            candidate_1["longitude"]
        )

        print("\nCandidate Origin 2")

        candidate_2 = result["candidates"][1]

        print(
            "Latitude:",
            candidate_2["latitude"]
        )

        print(
            "Longitude:",
            candidate_2["longitude"]
        )

        print(
            "\nWARNING:",
            result["warning"]
        )

    print(
        "=============================================="
    )