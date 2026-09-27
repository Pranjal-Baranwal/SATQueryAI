"""
Visual Grounding Utilities
===========================

Parses the JSON point-grounding output produced by the SIA
grounding prompt and converts it into pixel-space points and
masks usable by geospatial.visualization.SpatialVisualizer.

Points (not boxes) are used as the VLM's output format because
they are a much lower-precision target for a general VLM to get
right (2 numbers instead of 4), and a single point that lands
anywhere inside the true object is enough for SAM to recover
the full, correct boundary.
"""

import json
import re

import numpy as np


CONFIDENCE_LEVELS = {
    "high",
    "medium",
    "low",
}


def _extract_json_object(text):
    """
    Extract the first top-level JSON object found in a string.

    Model output is expected to be pure JSON, but this also
    tolerates stray text or markdown code fences around it.
    """

    if not text:
        return None

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    start = text.find("{")

    if start == -1:
        return None

    depth = 0

    for index in range(start, len(text)):

        if text[index] == "{":
            depth += 1

        elif text[index] == "}":
            depth -= 1

            if depth == 0:

                candidate = text[start:index + 1]

                try:
                    return json.loads(candidate)

                except json.JSONDecodeError:
                    return None

    return None


def _extract_points_fallback(text):
    """
    Fallback extraction when the model output is not valid JSON.

    Looks for any [x, y]-shaped array in the text. If none are
    found, falls back further to [x1, y1, x2, y2]-shaped boxes
    (from an older prompt/response format) and uses their center.
    """

    if not text:
        return []

    point_pattern = re.compile(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )

    points = []

    for match in point_pattern.finditer(text):

        x, y = [float(value) for value in match.groups()]

        points.append(
            {
                "label": "region",
                "point": [x, y],
                "location_hint": "",
                "confidence": "medium",
            }
        )

    if points:
        return points

    box_pattern = re.compile(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )

    for match in box_pattern.finditer(text):

        x1, y1, x2, y2 = [float(value) for value in match.groups()]

        points.append(
            {
                "label": "region",
                "point": [
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                ],
                "location_hint": "",
                "confidence": "medium",
            }
        )

    return points


def _point_from_item(item):
    """
    Extract a normalized [x, y] point from one parsed object,
    accepting either the current "point" format or the older
    "box" format (using its center) for resilience.
    """

    point = item.get("point")

    if (
        isinstance(point, (list, tuple))
        and len(point) == 2
    ):

        try:
            return [float(point[0]), float(point[1])]

        except (TypeError, ValueError):
            return None

    box = item.get("box")

    if (
        isinstance(box, (list, tuple))
        and len(box) == 4
    ):

        try:
            x1, y1, x2, y2 = [float(value) for value in box]

        except (TypeError, ValueError):
            return None

        return [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        ]

    return None


def parse_grounding_response(
    text,
    image_width,
    image_height
):
    """
    Parse the raw model response into pixel-space objects.

    Returns:
        List of dicts:

            {
                "label": str,
                "confidence": str,
                "location_hint": str,
                "point_normalized": [x, y],
                "point": [x, y],  # pixel coordinates
            }
    """

    parsed = _extract_json_object(text)

    raw_objects = None

    if isinstance(parsed, dict):

        raw_objects = parsed.get("objects")

    if not isinstance(raw_objects, list):

        raw_objects = _extract_points_fallback(text)

    objects = []

    for item in raw_objects:

        if not isinstance(item, dict):
            continue

        point = _point_from_item(item)

        if point is None:
            continue

        x, y = point

        x = max(0.0, min(1000.0, x))
        y = max(0.0, min(1000.0, y))

        pixel_point = [
            x / 1000.0 * image_width,
            y / 1000.0 * image_height,
        ]

        confidence = str(
            item.get("confidence", "medium")
        ).lower().strip()

        if confidence not in CONFIDENCE_LEVELS:
            confidence = "medium"

        label = str(
            item.get("label", "region")
        ).strip() or "region"

        location_hint = str(
            item.get("location_hint", "")
        ).strip()

        objects.append(
            {
                "label": label,
                "confidence": confidence,
                "location_hint": location_hint,
                "point_normalized": [x, y],
                "point": pixel_point,
            }
        )

    return objects


def points_to_mask(objects, image_width, image_height, radius=20):
    """
    Rasterize a list of pixel-space points into a boolean mask
    of small filled circles.

    This is only used as a fallback when SAM is unavailable; the
    real mask normally comes from SAM's point-prompted
    segmentation instead.
    """

    mask = np.zeros(
        (image_height, image_width),
        dtype=bool,
    )

    y_grid, x_grid = np.ogrid[:image_height, :image_width]

    for item in objects:

        x, y = item["point"]

        distance_squared = (
            (x_grid - x) ** 2
            +
            (y_grid - y) ** 2
        )

        mask |= distance_squared <= radius ** 2

    return mask


if __name__ == "__main__":

    sample_response = """
    {"objects": [{"label": "water", "point": [650, 550], "location_hint": "rectangular basin, right of center", "confidence": "high"}]}
    """

    result = parse_grounding_response(
        sample_response,
        image_width=1024,
        image_height=768,
    )

    print(result)

    mask = points_to_mask(
        result,
        image_width=1024,
        image_height=768,
    )

    print(
        "Masked pixels:",
        int(mask.sum())
    )
