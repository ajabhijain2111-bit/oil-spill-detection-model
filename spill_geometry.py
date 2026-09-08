import numpy as np
import cv2


def extract_contours(mask):
    mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours

def calculate_spill_shape(mask):
    contours = extract_contours(mask)

    if not contours:
        return {
            "area_pixels": 0,
            "perimeter_pixels": 0,
            "orientation_degrees": None,
            "width_pixels": 0,
            "height_pixels": 0
        }

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    area_pixels = float(
        cv2.contourArea(largest_contour)
    )

    perimeter_pixels = float(
        cv2.arcLength(
            largest_contour,
            True
        )
    )

    x, y, width, height = cv2.boundingRect(
        largest_contour
    )

    orientation = None

    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(
            largest_contour
        )

        orientation = float(
            ellipse[2]
        )

    return {
        "area_pixels": area_pixels,
        "perimeter_pixels": perimeter_pixels,
        "orientation_degrees": orientation,
        "width_pixels": int(width),
        "height_pixels": int(height)
    }

def get_spill_geometry(mask):
    shape = calculate_spill_shape(mask)

    return {
        "shape": shape
    }