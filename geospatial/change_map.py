"""
TriNetra - Geospatial Change Analysis
--------------------------------------

This module performs geospatial change analysis between two
already-preprocessed / spatially aligned rasters or index arrays.

IMPORTANT:
    Geospatial alignment/reprojection is handled by:
        preprocessing.geospatial_alignment.GeoSpatialAligner

This module does NOT perform reprojection.

Its responsibility is to convert two comparable observations:

        T1
        +
        T2
        ↓
    Difference
        ↓
    Change Mask
        ↓
    Spatial Statistics
        ↓
    Area / Coverage
        ↓
    Geographic Bounds
        ↓
    Structured Evidence

Primary use:
    BTP = Bi-Temporal Processing
    CDVQA = Change Detection / Change-aware Visual Question Answering

Secondary use:
    - SIA when a change map is explicitly requested
    - Agentic Router evidence
    - Optical + SAR workflows when comparable derived features are supplied

Typical examples:

    NDVI(T1) vs NDVI(T2)
    NDBI(T1) vs NDBI(T2)
    NDWI(T1) vs NDWI(T2)

The module is deliberately model-independent:
    it produces spatial evidence;
    the VLM / specialist model interprets that evidence.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import rasterio


# =====================================================================
# Constants
# =====================================================================

SUPPORTED_MODES = {
    "absolute",
    "signed",
}


# =====================================================================
# Main Change Analyzer
# =====================================================================

class ChangeMapAnalyzer:
    """
    Performs change analysis for two temporally separated observations.

    Parameters
    ----------
    reference_path : str
        GeoTIFF corresponding to T1.

    comparison_path : str
        GeoTIFF corresponding to T2.

    Both rasters should already be aligned to the same spatial grid.
    """

    def __init__(
        self,
        reference_path: str,
        comparison_path: str
    ):

        self.reference_path = Path(
            reference_path
        )

        self.comparison_path = Path(
            comparison_path
        )

        self._validate_file(
            self.reference_path
        )

        self._validate_file(
            self.comparison_path
        )

        self.reference_info = None
        self.comparison_info = None

        self._load_metadata()

    # =================================================================
    # Validation
    # =================================================================

    @staticmethod
    def _validate_file(
        file_path: Path
    ) -> None:
        """
        Validate a GeoTIFF input.
        """

        if not file_path.exists():

            raise FileNotFoundError(
                f"GeoTIFF file not found: {file_path}"
            )

        if file_path.suffix.lower() not in {
            ".tif",
            ".tiff"
        }:

            raise ValueError(
                f"File must be a GeoTIFF (.tif/.tiff): "
                f"{file_path}"
            )

    # =================================================================
    # Metadata
    # =================================================================

    @staticmethod
    def _get_raster_info(
        file_path: Path
    ) -> Dict:

        with rasterio.open(
            file_path
        ) as src:

            return {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": src.crs,
                "transform": src.transform,
                "resolution": src.res,
                "bounds": src.bounds,
                "nodata": src.nodata,
                "dtype": src.dtypes
            }

    def _load_metadata(
        self
    ) -> None:

        self.reference_info = (
            self._get_raster_info(
                self.reference_path
            )
        )

        self.comparison_info = (
            self._get_raster_info(
                self.comparison_path
            )
        )

    # =================================================================
    # Alignment validation
    # =================================================================

    def check_alignment(
        self
    ) -> Dict:

        ref = self.reference_info
        comp = self.comparison_info

        same_crs = (
            ref["crs"] == comp["crs"]
        )

        same_resolution = np.allclose(
            ref["resolution"],
            comp["resolution"]
        )

        same_dimensions = (
            ref["width"] == comp["width"]
            and
            ref["height"] == comp["height"]
        )

        same_transform = (
            ref["transform"]
            ==
            comp["transform"]
        )

        same_bounds = np.allclose(
            [
                ref["bounds"].left,
                ref["bounds"].bottom,
                ref["bounds"].right,
                ref["bounds"].top,
            ],
            [
                comp["bounds"].left,
                comp["bounds"].bottom,
                comp["bounds"].right,
                comp["bounds"].top,
            ]
        )

        aligned = all(
            [
                same_crs,
                same_resolution,
                same_dimensions,
                same_transform,
                same_bounds
            ]
        )

        return {
            "same_crs": same_crs,
            "same_resolution": same_resolution,
            "same_dimensions": same_dimensions,
            "same_transform": same_transform,
            "same_bounds": same_bounds,
            "aligned": aligned
        }

    # =================================================================
    # Read raster
    # =================================================================

    @staticmethod
    def _read_band(
        file_path: Path,
        band_number: int
    ) -> np.ndarray:

        with rasterio.open(
            file_path
        ) as src:

            if (
                band_number < 1
                or
                band_number > src.count
            ):

                raise ValueError(
                    f"Band {band_number} does not exist "
                    f"in {file_path}. "
                    f"Available bands: 1-{src.count}"
                )

            data = src.read(
                band_number
            ).astype(
                np.float32
            )

            if src.nodata is not None:

                data[
                    data == src.nodata
                ] = np.nan

            return data

    # =================================================================
    # Prepare arrays
    # =================================================================

    @staticmethod
    def _prepare_array(
        array: np.ndarray
    ) -> np.ndarray:

        array = np.asarray(
            array,
            dtype=np.float32
        )

        if array.ndim != 2:

            raise ValueError(
                "Input array must be 2D "
                "(height, width)."
            )

        array = array.copy()

        array[
            ~np.isfinite(array)
        ] = np.nan

        return array

    # =================================================================
    # Validate two arrays
    # =================================================================

    @staticmethod
    def _validate_array_pair(
        reference: np.ndarray,
        comparison: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        reference = (
            ChangeMapAnalyzer._prepare_array(
                reference
            )
        )

        comparison = (
            ChangeMapAnalyzer._prepare_array(
                comparison
            )
        )

        if reference.shape != comparison.shape:

            raise ValueError(
                "Reference and comparison arrays "
                "must have identical dimensions. "
                f"T1: {reference.shape}, "
                f"T2: {comparison.shape}"
            )

        return reference, comparison

    # =================================================================
    # Difference
    # =================================================================

    @staticmethod
    def difference(
        reference: np.ndarray,
        comparison: np.ndarray,
        mode: str = "signed"
    ) -> np.ndarray:
        """
        Calculate change between T1 and T2.

        signed:
            T2 - T1

        absolute:
            |T2 - T1|
        """

        mode = mode.lower().strip()

        if mode not in SUPPORTED_MODES:

            raise ValueError(
                f"Unsupported difference mode '{mode}'. "
                f"Use one of: {sorted(SUPPORTED_MODES)}"
            )

        reference, comparison = (
            ChangeMapAnalyzer._validate_array_pair(
                reference,
                comparison
            )
        )

        result = np.full(
            reference.shape,
            np.nan,
            dtype=np.float32
        )

        valid = (
            np.isfinite(reference)
            &
            np.isfinite(comparison)
        )

        if mode == "signed":

            result[valid] = (
                comparison[valid]
                -
                reference[valid]
            )

        else:

            result[valid] = np.abs(
                comparison[valid]
                -
                reference[valid]
            )

        return result

    # =================================================================
    # Change mask
    # =================================================================

    @staticmethod
    def change_mask(
        difference: np.ndarray,
        threshold: float,
        absolute: bool = True
    ) -> np.ndarray:
        """
        Create a binary mask of significant change.

        Example:

            abs(T2 - T1) > 0.10

        """

        difference = np.asarray(
            difference,
            dtype=np.float32
        )

        if difference.ndim != 2:

            raise ValueError(
                "Difference array must be 2D."
            )

        if threshold < 0:

            raise ValueError(
                "Threshold cannot be negative."
            )

        valid = np.isfinite(
            difference
        )

        if absolute:

            mask = (
                np.abs(difference)
                >
                threshold
            )

        else:

            mask = (
                difference
                >
                threshold
            )

        return mask & valid

    # =================================================================
    # Increase mask
    # =================================================================

    @staticmethod
    def increase_mask(
        difference: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Identify areas where the value increased
        by more than the threshold.

        Example:

            NDVI(T2) - NDVI(T1) > 0.10
        """

        difference = np.asarray(
            difference,
            dtype=np.float32
        )

        if threshold < 0:

            raise ValueError(
                "Threshold cannot be negative."
            )

        valid = np.isfinite(
            difference
        )

        return (
            (difference > threshold)
            &
            valid
        )

    # =================================================================
    # Decrease mask
    # =================================================================

    @staticmethod
    def decrease_mask(
        difference: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Identify areas where the value decreased
        by more than the threshold.

        Example:

            NDVI(T2) - NDVI(T1) < -0.10
        """

        difference = np.asarray(
            difference,
            dtype=np.float32
        )

        if threshold < 0:

            raise ValueError(
                "Threshold cannot be negative."
            )

        valid = np.isfinite(
            difference
        )

        return (
            (difference < -threshold)
            &
            valid
        )

    # =================================================================
    # Valid pixel count
    # =================================================================

    @staticmethod
    def valid_pixel_count(
        array: np.ndarray
    ) -> int:

        array = np.asarray(
            array
        )

        return int(
            np.count_nonzero(
                np.isfinite(array)
            )
        )

    # =================================================================
    # Changed pixel count
    # =================================================================

    @staticmethod
    def changed_pixel_count(
        mask: np.ndarray
    ) -> int:

        mask = np.asarray(
            mask,
            dtype=bool
        )

        return int(
            np.count_nonzero(mask)
        )

    # =================================================================
    # Change percentage
    # =================================================================

    @staticmethod
    def change_percentage(
        mask: np.ndarray,
        reference: np.ndarray,
        comparison: np.ndarray
    ) -> float:
        """
        Percentage of jointly valid pixels that changed.
        """

        mask = np.asarray(
            mask,
            dtype=bool
        )

        reference, comparison = (
            ChangeMapAnalyzer._validate_array_pair(
                reference,
                comparison
            )
        )

        jointly_valid = (
            np.isfinite(reference)
            &
            np.isfinite(comparison)
        )

        total_valid = int(
            np.count_nonzero(
                jointly_valid
            )
        )

        if total_valid == 0:

            return 0.0

        changed = int(
            np.count_nonzero(
                mask
                &
                jointly_valid
            )
        )

        return (
            changed /
            total_valid
        ) * 100.0

    # =================================================================
    # Pixel area
    # =================================================================

    def pixel_area(
        self
    ) -> Optional[float]:
        """
        Return pixel area in square metres.

        Only directly valid for a projected CRS
        whose units are metric.

        Geographic CRS returns None.
        """

        crs = self.reference_info["crs"]

        if crs is None:
            return None

        if crs.is_geographic:
            return None

        x_resolution = abs(
            float(
                self.reference_info["resolution"][0]
            )
        )

        y_resolution = abs(
            float(
                self.reference_info["resolution"][1]
            )
        )

        return (
            x_resolution
            *
            y_resolution
        )

    # =================================================================
    # Changed area
    # =================================================================

    def changed_area_m2(
        self,
        mask: np.ndarray
    ) -> Optional[float]:
        """
        Calculate changed area in square metres.
        """

        area = self.pixel_area()

        if area is None:
            return None

        return (
            self.changed_pixel_count(mask)
            *
            area
        )

    # =================================================================
    # Changed area km²
    # =================================================================

    def changed_area_km2(
        self,
        mask: np.ndarray
    ) -> Optional[float]:
        """
        Calculate changed area in square kilometres.
        """

        area_m2 = self.changed_area_m2(
            mask
        )

        if area_m2 is None:
            return None

        return (
            area_m2 /
            1_000_000.0
        )

    # =================================================================
    # Spatial bounds
    # =================================================================

    def mask_bounds(
        self,
        mask: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Return geographic bounds of the changed region.
        """

        mask = np.asarray(
            mask,
            dtype=bool
        )

        rows, cols = np.where(
            mask
        )

        if rows.size == 0:

            return None

        min_row = int(
            rows.min()
        )

        max_row = int(
            rows.max()
        )

        min_col = int(
            cols.min()
        )

        max_col = int(
            cols.max()
        )

        top_left = rasterio.transform.xy(
            self.reference_info[
                "transform"
            ],
            min_row,
            min_col,
            offset="ul"
        )

        bottom_right = rasterio.transform.xy(
            self.reference_info[
                "transform"
            ],
            max_row,
            max_col,
            offset="lr"
        )

        return {
            "left": float(
                top_left[0]
            ),
            "bottom": float(
                bottom_right[1]
            ),
            "right": float(
                bottom_right[0]
            ),
            "top": float(
                top_left[1]
            )
        }

    # =================================================================
    # Change statistics
    # =================================================================

    @staticmethod
    def difference_statistics(
        difference: np.ndarray
    ) -> Dict:

        difference = np.asarray(
            difference,
            dtype=np.float32
        )

        valid = difference[
            np.isfinite(difference)
        ]

        if valid.size == 0:

            return {
                "valid_pixels": 0,
                "mean": None,
                "median": None,
                "minimum": None,
                "maximum": None,
                "std": None
            }

        return {
            "valid_pixels": int(
                valid.size
            ),
            "mean": float(
                np.mean(valid)
            ),
            "median": float(
                np.median(valid)
            ),
            "minimum": float(
                np.min(valid)
            ),
            "maximum": float(
                np.max(valid)
            ),
            "std": float(
                np.std(valid)
            )
        }

    # =================================================================
    # Complete change analysis
    # =================================================================

    def analyze(
        self,
        reference: np.ndarray,
        comparison: np.ndarray,
        threshold: float,
        difference_mode: str = "signed"
    ) -> Dict:
        """
        Run a complete change analysis.

        Parameters
        ----------
        reference:
            T1 array.

        comparison:
            T2 array.

        threshold:
            Minimum meaningful change.

        difference_mode:
            signed or absolute.
        """

        reference, comparison = (
            self._validate_array_pair(
                reference,
                comparison
            )
        )

        # Check raster dimensions.
        expected_shape = (
            self.reference_info["height"],
            self.reference_info["width"]
        )

        if reference.shape != expected_shape:

            raise ValueError(
                "Array dimensions do not match "
                "the GeoTIFF dimensions. "
                f"Expected: {expected_shape}, "
                f"Received: {reference.shape}"
            )

        alignment = (
            self.check_alignment()
        )

        if not alignment["aligned"]:

            raise ValueError(
                "T1 and T2 GeoTIFFs are not "
                "spatially aligned. "
                "Use preprocessing.geospatial_alignment "
                "before running change analysis."
            )

        difference = self.difference(
            reference=reference,
            comparison=comparison,
            mode=difference_mode
        )

        mask = self.change_mask(
            difference=difference,
            threshold=threshold,
            absolute=(
                difference_mode == "absolute"
                or
                difference_mode == "signed"
            )
        )

        change_pixels = (
            self.changed_pixel_count(mask)
        )

        change_percent = (
            self.change_percentage(
                mask=mask,
                reference=reference,
                comparison=comparison
            )
        )

        result = {
            "type": "bi_temporal_change_analysis",

            "difference_mode": difference_mode,

            "threshold": threshold,

            "alignment": alignment,

            "difference_statistics": (
                self.difference_statistics(
                    difference
                )
            ),

            "changed_pixels": change_pixels,

            "change_percentage": round(
                change_percent,
                3
            ),

            "changed_region_bounds": (
                self.mask_bounds(mask)
            ),

            "crs": (
                str(
                    self.reference_info["crs"]
                )
                if self.reference_info["crs"]
                else None
            ),

            "resolution": (
                [
                    float(
                        self.reference_info[
                            "resolution"
                        ][0]
                    ),
                    float(
                        self.reference_info[
                            "resolution"
                        ][1]
                    )
                ]
            )
        }

        area = (
            self.changed_area_km2(
                mask
            )
        )

        if area is not None:

            result["changed_area_km2"] = round(
                area,
                4
            )

        return result

    # =================================================================
    # Directional change analysis
    # =================================================================

    def analyze_directional_change(
        self,
        reference: np.ndarray,
        comparison: np.ndarray,
        threshold: float
    ) -> Dict:
        """
        Separate increases and decreases.

        Particularly useful for NDVI / NDWI / NDBI:

            NDVI increase   → possible vegetation gain
            NDVI decrease   → possible vegetation loss

        IMPORTANT:
            The semantic interpretation belongs to the calling
            module. This class only measures numerical change.
        """

        reference, comparison = (
            self._validate_array_pair(
                reference,
                comparison
            )
        )

        alignment = (
            self.check_alignment()
        )

        if not alignment["aligned"]:

            raise ValueError(
                "T1 and T2 GeoTIFFs are not aligned."
            )

        difference = (
            self.difference(
                reference,
                comparison,
                mode="signed"
            )
        )

        increase = (
            self.increase_mask(
                difference,
                threshold
            )
        )

        decrease = (
            self.decrease_mask(
                difference,
                threshold
            )
        )

        return {
            "increase_pixels": int(
                np.count_nonzero(
                    increase
                )
            ),

            "decrease_pixels": int(
                np.count_nonzero(
                    decrease
                )
            ),

            "increase_percentage": round(
                self.change_percentage(
                    increase,
                    reference,
                    comparison
                ),
                3
            ),

            "decrease_percentage": round(
                self.change_percentage(
                    decrease,
                    reference,
                    comparison
                ),
                3
            ),

            "increase_bounds": (
                self.mask_bounds(
                    increase
                )
            ),

            "decrease_bounds": (
                self.mask_bounds(
                    decrease
                )
            )
        }

    # =================================================================
    # TriNetra evidence
    # =================================================================

    def generate_evidence(
        self,
        reference: np.ndarray,
        comparison: np.ndarray,
        variable_name: str,
        threshold: float
    ) -> Dict:
        """
        Produce structured evidence for TriNetra's BTP/CDVQA
        specialist and agentic router.
        """

        reference, comparison = (
            self._validate_array_pair(
                reference,
                comparison
            )
        )

        difference = self.difference(
            reference,
            comparison,
            mode="signed"
        )

        mask = self.change_mask(
            difference=difference,
            threshold=threshold,
            absolute=True
        )

        analysis = self.analyze(
            reference=reference,
            comparison=comparison,
            threshold=threshold,
            difference_mode="signed"
        )

        directional = (
            self.analyze_directional_change(
                reference=reference,
                comparison=comparison,
                threshold=threshold
            )
        )

        evidence = {

            "type":
                "bi_temporal_geospatial_evidence",

            "capability":
                "change_analysis",

            "variable":
                variable_name.upper(),

            "evidence_available":
                True,

            "time_points":
                ["T1", "T2"],

            "threshold":
                threshold,

            "metrics_available": [
                "difference_statistics",
                "change_percentage",
                "changed_pixels",
                "changed_area_km2",
                "changed_region_bounds",
                "increase_percentage",
                "decrease_percentage"
            ],

            "change_percentage":
                analysis[
                    "change_percentage"
                ],

            "changed_pixels":
                analysis[
                    "changed_pixels"
                ],

            "changed_region_bounds":
                analysis[
                    "changed_region_bounds"
                ],

            "increase_percentage":
                directional[
                    "increase_percentage"
                ],

            "decrease_percentage":
                directional[
                    "decrease_percentage"
                ],

            "increase_bounds":
                directional[
                    "increase_bounds"
                ],

            "decrease_bounds":
                directional[
                    "decrease_bounds"
                ],

            "crs":
                analysis["crs"],

            "resolution":
                analysis["resolution"]
        }

        if "changed_area_km2" in analysis:

            evidence[
                "changed_area_km2"
            ] = analysis[
                "changed_area_km2"
            ]

        return evidence


# =====================================================================
# Convenience functions
# =====================================================================

def calculate_difference(
    reference: np.ndarray,
    comparison: np.ndarray,
    mode: str = "signed"
) -> np.ndarray:
    """
    Calculate T2 - T1 or |T2 - T1|.
    """

    return ChangeMapAnalyzer.difference(
        reference=reference,
        comparison=comparison,
        mode=mode
    )


def create_change_mask(
    difference: np.ndarray,
    threshold: float,
    absolute: bool = True
) -> np.ndarray:
    """
    Create a binary change mask.
    """

    return ChangeMapAnalyzer.change_mask(
        difference=difference,
        threshold=threshold,
        absolute=absolute
    )


# =====================================================================
# Example
# =====================================================================

if __name__ == "__main__":

    print(
        "TriNetra geospatial/change_map.py "
        "loaded successfully."
    )

    print(
        "\nThis module expects:"
    )

    print(
        "T1 GeoTIFF + T2 GeoTIFF"
    )

    print(
        "and comparable 2D arrays "
        "such as NDVI(T1) and NDVI(T2)."
    )

    print(
        "\nTypical flow:"
    )

    print(
        "Preprocessing alignment"
    )

    print(
        "        ↓"
    )

    print(
        "T1 / T2 index arrays"
    )

    print(
        "        ↓"
    )

    print(
        "Difference"
    )

    print(
        "        ↓"
    )

    print(
        "Thresholded change mask"
    )

    print(
        "        ↓"
    )

    print(
        "Area + Coverage + Bounds"
    )

    print(
        "        ↓"
    )

    print(
        "TriNetra BTP / CDVQA Evidence"
    )