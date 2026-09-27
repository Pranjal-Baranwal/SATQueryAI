"""
Deterministic Pixel-Level Class Masks
======================================

For a handful of common land-cover classes (water, vegetation,
urban/built-up, roads), asking a general VQA-tuned VLM to
localize the class is unreliable: it has to place a point or
box purely from learned association, with no guarantee it was
ever trained on spatial grounding at all. For these classes the
correct pixel-level answer is directly computable from color
and texture, with no model call and no chance of "hallucinating"
the wrong region.

This module is tried FIRST for a recognized class keyword.
Only queries that do not match a known class fall back to the
VLM + SAM grounding path in models/sia/inference.py.

These are RGB-only heuristics (no NIR/SWIR band available from
a plain PNG/JPG upload), so they are deliberately conservative
rather than physically exact indices like NDVI/NDWI.
"""

import numpy as np
from scipy import ndimage
from skimage import color as skcolor
from skimage import morphology, measure


WATER_KEYWORDS = [
    "water",
    "lake",
    "pond",
    "river",
    "reservoir",
    "basin",
    "flooded",
    "flood",
    "sea",
    "ocean",
    "canal",
    "stream",
]

VEGETATION_KEYWORDS = [
    "vegetation",
    "tree",
    "trees",
    "forest",
    "green cover",
    "grass",
    "park",
    "garden",
    "crop",
    "farmland",
    "foliage",
]

URBAN_KEYWORDS = [
    "building",
    "buildings",
    "urban",
    "rooftop",
    "rooftops",
    "roof",
    "structure",
    "structures",
    "settlement",
    "built-up",
    "built up",
]

ROAD_KEYWORDS = [
    "road",
    "roads",
    "street",
    "streets",
    "highway",
    "pathway",
]


def match_known_class(query):
    """
    Return the recognized class name ("water", "vegetation",
    "urban", "road") if the query clearly asks for one of them,
    else None.
    """

    if not query:
        return None

    text = query.lower()

    if any(keyword in text for keyword in WATER_KEYWORDS):
        return "water"

    if any(keyword in text for keyword in VEGETATION_KEYWORDS):
        return "vegetation"

    if any(keyword in text for keyword in URBAN_KEYWORDS):
        return "urban"

    if any(keyword in text for keyword in ROAD_KEYWORDS):
        return "road"

    return None


def _local_std(gray, size=9):
    """
    Fast local standard deviation via a uniform (box) filter,
    used as a texture measure: water/roads are smooth (low
    local std), rooftops/rubble are textured (high local std).
    """

    mean = ndimage.uniform_filter(
        gray,
        size=size
    )

    mean_of_square = ndimage.uniform_filter(
        gray ** 2,
        size=size
    )

    variance = np.clip(
        mean_of_square - mean ** 2,
        0,
        None
    )

    return np.sqrt(variance)


def _clean_mask(
    mask,
    min_area_fraction=0.001,
    min_solidity=None
):
    """
    Remove speckle noise and keep only regions large enough to
    plausibly be a real instance of the class, rather than a
    single mis-colored pixel or a small shadow fragment.

    min_solidity (region.area / region.convex_hull area) is an
    optional shape-regularity filter: a compact, roughly convex
    outline (typical of a reservoir/tank/pond) scores close to
    1.0, while a jagged, organic outline (typical of a tree
    canopy's shadow, which can otherwise match water's color
    and smoothness in plain RGB) scores much lower. Only used
    where that distinction is meaningful for the class.
    """

    if not mask.any():
        return mask

    mask = morphology.opening(
        mask,
        morphology.disk(2)
    )

    mask = morphology.closing(
        mask,
        morphology.disk(3)
    )

    labeled = measure.label(mask)

    if labeled.max() == 0:
        return mask

    min_area = mask.size * min_area_fraction

    cleaned = np.zeros_like(mask)

    for region in measure.regionprops(labeled):

        if region.area < min_area:
            continue

        if (
            min_solidity is not None
            and region.solidity < min_solidity
        ):
            continue

        cleaned[labeled == region.label] = True

    return cleaned


def detect_water(image):
    """
    Water in optical imagery is dark, smooth (low local
    texture), and grayish/blue-black rather than green (which
    would indicate shaded tree canopy) or reddish (which would
    indicate a shadow on a warm-toned rooftop or bare soil).
    """

    rgb = np.asarray(
        image.convert("RGB")
    ).astype(np.float32)

    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]

    gray = rgb.mean(axis=2)

    brightness = gray / 255.0

    texture = _local_std(gray, size=9) / 255.0

    dark = brightness < 0.35

    smooth = texture < 0.035

    not_reddish = red <= (blue + 15)

    not_greenish = green <= (
        np.maximum(red, blue) + 10
    )

    mask = dark & smooth & not_reddish & not_greenish

    return _clean_mask(
        mask,
        min_area_fraction=0.0015,
        min_solidity=0.75
    )


def detect_vegetation(image):
    """
    Excess Green Index (2G - R - B): a standard RGB-only proxy
    for vegetation when no NIR band is available.
    """

    rgb = np.asarray(
        image.convert("RGB")
    ).astype(np.float32)

    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]

    excess_green = (2 * green) - red - blue

    mask = excess_green > 20

    return _clean_mask(
        mask,
        min_area_fraction=0.0004
    )


def detect_urban(image):
    """
    Built-up areas: moderately bright, textured (roof edges,
    shadows between structures), unlike smooth water/roads or
    green vegetation.
    """

    rgb = np.asarray(
        image.convert("RGB")
    ).astype(np.float32)

    gray = rgb.mean(axis=2)

    brightness = gray / 255.0

    texture = _local_std(gray, size=9)

    mask = (
        (brightness > 0.35)
        & (brightness < 0.88)
        & (texture > 8)
    )

    return _clean_mask(
        mask,
        min_area_fraction=0.001
    )


def detect_road(image):
    """
    Roads: mid-gray, low saturation, smooth compared to
    surrounding built-up texture.
    """

    rgb = np.asarray(
        image.convert("RGB")
    ).astype(np.float32) / 255.0

    hsv = skcolor.rgb2hsv(rgb)

    saturation = hsv[..., 1]
    value = hsv[..., 2]

    mask = (
        (saturation < 0.15)
        & (value > 0.30)
        & (value < 0.70)
    )

    return _clean_mask(
        mask,
        min_area_fraction=0.0008
    )


DETECTORS = {
    "water": detect_water,
    "vegetation": detect_vegetation,
    "urban": detect_urban,
    "road": detect_road,
}


def detect_class_mask(image, class_name):
    """
    Compute a pixel-level mask for a recognized class name.

    Returns:
        Boolean mask, shape (height, width), or None if
        class_name is not recognized.
    """

    detector = DETECTORS.get(class_name)

    if detector is None:
        return None

    return detector(image)


if __name__ == "__main__":

    from PIL import Image

    print(
        "\n========== SEMANTIC MASK TEST =========="
    )

    image_path = "data/input/T1.png"

    image = Image.open(
        image_path
    ).convert("RGB")

    for class_name in DETECTORS:

        mask = detect_class_mask(
            image,
            class_name
        )

        print(
            f"{class_name}: "
            f"{int(mask.sum())} pixels "
            f"({100.0 * mask.mean():.2f}% of image)"
        )
