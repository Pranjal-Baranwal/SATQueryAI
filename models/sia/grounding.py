"""
Visual Grounding Utilities
===========================

Parses the JSON bounding-box output produced by the SIA
grounding prompt and converts it into pixel-space boxes and
masks usable by geospatial.visualization.SpatialVisualizer.
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


def _extract_boxes_fallback(text):
    """
    Fallback extraction when the model output is not valid JSON.

    Looks for any [x1, y1, x2, y2]-shaped array in the text.
    """

    if not text:
        return []

    pattern = re.compile(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )

    boxes = []

    for match in pattern.finditer(text):

        values = [float(value) for value in match.groups()]

        boxes.append(
            {
                "label": "region",
                "box": values,
                "confidence": "medium",
            }
        )

    return boxes


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
                "box_normalized": [x1, y1, x2, y2],
                "box": [x1, y1, x2, y2],  # pixel coordinates
            }
    """

    parsed = _extract_json_object(text)

    raw_objects = None

    if isinstance(parsed, dict):

        raw_objects = parsed.get("objects")

    if not isinstance(raw_objects, list):

        raw_objects = _extract_boxes_fallback(text)

    objects = []

    for item in raw_objects:

        if not isinstance(item, dict):
            continue

        box = item.get("box")

        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
        ):
            continue

        try:
            x1, y1, x2, y2 = [float(value) for value in box]

        except (TypeError, ValueError):
            continue

        if x1 > x2:
            x1, x2 = x2, x1

        if y1 > y2:
            y1, y2 = y2, y1

        x1 = max(0.0, min(1000.0, x1))
        x2 = max(0.0, min(1000.0, x2))
        y1 = max(0.0, min(1000.0, y1))
        y2 = max(0.0, min(1000.0, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        pixel_box = [
            x1 / 1000.0 * image_width,
            y1 / 1000.0 * image_height,
            x2 / 1000.0 * image_width,
            y2 / 1000.0 * image_height,
        ]

        confidence = str(
            item.get("confidence", "medium")
        ).lower().strip()

        if confidence not in CONFIDENCE_LEVELS:
            confidence = "medium"

        label = str(
            item.get("label", "region")
        ).strip() or "region"

        objects.append(
            {
                "label": label,
                "confidence": confidence,
                "box_normalized": [x1, y1, x2, y2],
                "box": pixel_box,
            }
        )

    return objects


def boxes_to_mask(objects, image_width, image_height):
    """
    Rasterize a list of pixel-space boxes into a boolean mask.

    Returns:
        NumPy boolean array with shape (height, width).
    """

    mask = np.zeros(
        (image_height, image_width),
        dtype=bool,
    )

    for item in objects:

        x1, y1, x2, y2 = item["box"]

        row_start = max(0, int(round(y1)))
        row_end = min(image_height, int(round(y2)))

        col_start = max(0, int(round(x1)))
        col_end = min(image_width, int(round(x2)))

        if row_end <= row_start or col_end <= col_start:
            continue

        mask[row_start:row_end, col_start:col_end] = True

    return mask


if __name__ == "__main__":

    sample_response = """
    {"objects": [{"label": "water", "box": [120, 300, 480, 620], "confidence": "high"}]}
    """

    result = parse_grounding_response(
        sample_response,
        image_width=1024,
        image_height=768,
    )

    print(result)

    mask = boxes_to_mask(
        result,
        image_width=1024,
        image_height=768,
    )

    print(
        "Masked pixels:",
        int(mask.sum())
    )
