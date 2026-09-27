from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import rasterio


SUPPORTED_INDICES = {
    "NDVI",
    "NDWI",
    "NDBI",
}


class GeospatialIndexAnalyzer:

    def __init__(self, raster_path: str):

        self.raster_path = Path(raster_path)

        if not self.raster_path.exists():
            raise FileNotFoundError(
                f"GeoTIFF file not found: {self.raster_path}"
            )

        if self.raster_path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(
                "GeospatialIndexAnalyzer requires a GeoTIFF (.tif/.tiff)."
            )

        self.crs = None
        self.transform = None
        self.width = None
        self.height = None
        self.resolution = None
        self.bounds = None

        self._load_spatial_metadata()

    def _load_spatial_metadata(self) -> None:
        """
        Read only the geospatial information.

        Index calculation remains inside preprocessing.
        """

        with rasterio.open(self.raster_path) as src:

            self.crs = src.crs
            self.transform = src.transform

            self.width = src.width
            self.height = src.height

            self.resolution = src.res

            self.bounds = src.bounds


    @staticmethod
    def _prepare_index(index: np.ndarray) -> np.ndarray:
        """
        Validate and clean an already-computed index.

        NDVI / NDWI / NDBI normally have meaningful values
        approximately within [-1, 1].

        NaN / infinity values are treated as invalid pixels.
        """

        index = np.asarray(index, dtype=np.float32)

        if index.ndim != 2:
            raise ValueError(
                "Index must be a 2D NumPy array "
                "with shape (height, width)."
            )

        cleaned = index.copy()

        cleaned[~np.isfinite(cleaned)] = np.nan

        cleaned[
            (cleaned < -1.0) |
            (cleaned > 1.0)
        ] = np.nan

        return cleaned

    @staticmethod
    def _validate_index_name(index_name: str) -> str:
        """
        Validate and normalize an index name.
        """

        name = str(index_name).upper().strip()

        if name not in SUPPORTED_INDICES:
            raise ValueError(
                f"Unsupported index '{index_name}'. "
                f"Supported indices: {sorted(SUPPORTED_INDICES)}"
            )

        return name


    def statistics(
        self,
        index: np.ndarray
    ) -> Dict[str, Optional[float]]:
        """
        Calculate basic statistics for an index.

        Returns:
            valid_pixels
            mean
            median
            minimum
            maximum
            standard deviation
        """

        index = self._prepare_index(index)

        valid = index[np.isfinite(index)]

        if valid.size == 0:

            return {
                "valid_pixels": 0,
                "mean": None,
                "median": None,
                "minimum": None,
                "maximum": None,
                "std": None,
            }

        return {
            "valid_pixels": int(valid.size),
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "minimum": float(np.min(valid)),
            "maximum": float(np.max(valid)),
            "std": float(np.std(valid)),
        }


    def valid_pixel_count(
        self,
        index: np.ndarray
    ) -> int:
        """
        Count valid index pixels.
        """

        index = self._prepare_index(index)

        return int(
            np.count_nonzero(
                np.isfinite(index)
            )
        )


    def pixel_area(self) -> Optional[float]:
        """
        Return approximate pixel area in square metres.

        This is directly valid when the GeoTIFF uses a projected
        coordinate reference system whose units are metres.

        For geographic CRS (degrees), we return None rather than
        incorrectly multiplying degrees together.
        """

        if self.crs is None:
            return None

        if self.resolution is None:
            return None

        if self.crs.is_geographic:
            return None

        x_res = abs(float(self.resolution[0]))
        y_res = abs(float(self.resolution[1]))

        return x_res * y_res


    def area_from_mask(
        self,
        mask: np.ndarray
    ) -> Optional[float]:
        """
        Calculate area covered by a boolean mask.

        Returns:
            area in square metres
            or None for geographic CRS.
        """

        mask = np.asarray(mask, dtype=bool)

        if mask.ndim != 2:
            raise ValueError(
                "Mask must be a 2D NumPy array."
            )

        pixel_area = self.pixel_area()

        if pixel_area is None:
            return None

        selected_pixels = int(
            np.count_nonzero(mask)
        )

        return selected_pixels * pixel_area


    def area_km2(
        self,
        mask: np.ndarray
    ) -> Optional[float]:
        """
        Calculate masked area in square kilometres.
        """

        area_m2 = self.area_from_mask(mask)

        if area_m2 is None:
            return None

        return area_m2 / 1_000_000.0


    def coverage_percentage(
        self,
        mask: np.ndarray,
        index: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate what percentage of valid pixels are covered
        by the supplied mask.

        Example:
            vegetation coverage =
            vegetated pixels / valid pixels × 100
        """

        mask = np.asarray(mask, dtype=bool)

        if mask.ndim != 2:
            raise ValueError(
                "Mask must be a 2D NumPy array."
            )

        if index is not None:

            index = self._prepare_index(index)

            valid_pixels = np.isfinite(index)

        else:

            valid_pixels = np.ones(
                mask.shape,
                dtype=bool
            )

        total_valid = int(
            np.count_nonzero(valid_pixels)
        )

        if total_valid == 0:
            return 0.0

        selected_valid = int(
            np.count_nonzero(
                mask & valid_pixels
            )
        )

        return (
            selected_valid /
            total_valid
        ) * 100.0


    @staticmethod
    def threshold_mask(
        index: np.ndarray,
        threshold: Union[float, tuple],
        condition: str = "greater"
    ) -> np.ndarray:
        """
        Create a boolean mask from an index.

        Supported conditions:
            greater
            greater_equal
            less
            less_equal
            between

        Example:

            NDVI > 0.30

        becomes:

            threshold_mask(
                ndvi,
                0.30,
                "greater"
            )
        """

        index = np.asarray(
            index,
            dtype=np.float32
        )

        valid = np.isfinite(index)

        condition = condition.lower().strip()

        if condition == "greater":

            mask = index > threshold

        elif condition == "greater_equal":

            mask = index >= threshold

        elif condition == "less":

            mask = index < threshold

        elif condition == "less_equal":

            mask = index <= threshold

        elif condition == "between":

            if (
                not isinstance(
                    threshold,
                    (tuple, list)
                )
                or len(threshold) != 2
            ):
                raise ValueError(
                    "For condition='between', "
                    "threshold must be (lower, upper)."
                )

            lower, upper = threshold

            if lower > upper:
                raise ValueError(
                    "Lower threshold cannot exceed upper threshold."
                )

            mask = (
                (index >= lower) &
                (index <= upper)
            )

        else:

            raise ValueError(
                "Unsupported condition. Use "
                "'greater', 'greater_equal', 'less', "
                "'less_equal' or 'between'."
            )

        return mask & valid


    def mask_bounds(
        self,
        mask: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Convert the outer bounds of a selected pixel region
        from raster coordinates to map coordinates.

        Returns:
            left
            bottom
            right
            top
        """

        mask = np.asarray(
            mask,
            dtype=bool
        )

        if mask.ndim != 2:
            raise ValueError(
                "Mask must be a 2D NumPy array."
            )

        rows, cols = np.where(mask)

        if rows.size == 0:
            return None

        min_row = int(rows.min())
        max_row = int(rows.max())

        min_col = int(cols.min())
        max_col = int(cols.max())

        top_left = rasterio.transform.xy(
            self.transform,
            min_row,
            min_col,
            offset="ul"
        )

        bottom_right = rasterio.transform.xy(
            self.transform,
            max_row,
            max_col,
            offset="lr"
        )

        return {
            "left": float(top_left[0]),
            "bottom": float(bottom_right[1]),
            "right": float(bottom_right[0]),
            "top": float(top_left[1]),
        }


    def summarize(
        self,
        index: np.ndarray,
        index_name: str,
        threshold: Optional[Union[float, tuple]] = None,
        condition: str = "greater"
    ) -> Dict:
        """
        Generate a complete geospatial summary.

        This is the primary function other TriNetra modules
        should use.
        """

        index_name = self._validate_index_name(
            index_name
        )

        index = self._prepare_index(index)

        if (
            index.shape[0] != self.height
            or
            index.shape[1] != self.width
        ):
            raise ValueError(
                "Index dimensions do not match the GeoTIFF dimensions. "
                f"Index: {index.shape}, "
                f"GeoTIFF: {(self.height, self.width)}"
            )

        stats = self.statistics(index)

        result = {
            "index": index_name,
            "statistics": stats,
            "crs": str(self.crs) if self.crs else None,
            "raster_bounds": {
                "left": float(self.bounds.left),
                "bottom": float(self.bounds.bottom),
                "right": float(self.bounds.right),
                "top": float(self.bounds.top),
            } if self.bounds else None,
            "resolution": (
                [float(self.resolution[0]),
                 float(self.resolution[1])]
                if self.resolution
                else None
            ),
        }

        if threshold is None:
            return result

        mask = self.threshold_mask(
            index=index,
            threshold=threshold,
            condition=condition
        )

        selected_pixels = int(
            np.count_nonzero(mask)
        )

        coverage = self.coverage_percentage(
            mask=mask,
            index=index
        )

        result.update({
            "threshold": threshold,
            "condition": condition,
            "selected_pixels": selected_pixels,
            "coverage_percentage": round(
                coverage,
                3
            ),
            "selected_region_bounds": self.mask_bounds(
                mask
            ),
        })

        area = self.area_km2(mask)

        if area is not None:
            result["area_km2"] = round(
                area,
                4
            )

        return result


    def generate_evidence(
        self,
        index: np.ndarray,
        index_name: str,
        threshold: float,
        condition: str = "greater"
    ) -> Dict:
        """
        Convert an index analysis result into a structured
        evidence object suitable for TriNetra.

        This format can later be passed to:
            - VQA specialist
            - Bi-temporal specialist
            - Cross-modal specialist
            - Agentic router
            - Response generator
        """

        summary = self.summarize(
            index=index,
            index_name=index_name,
            threshold=threshold,
            condition=condition
        )

        evidence = {
            "type": "geospatial_index_analysis",

            "capability": "spatial_index_analysis",

            "evidence_available": True,

            "index": summary["index"],

            "metrics_available": [
                "mean",
                "median",
                "minimum",
                "maximum",
                "standard_deviation",
                "coverage_percentage",
                "selected_pixels",
                "spatial_bounds",
            ],

            "mean_value": summary[
                "statistics"
            ]["mean"],

            "median_value": summary[
                "statistics"
            ]["median"],

            "minimum_value": summary[
                "statistics"
            ]["minimum"],

            "maximum_value": summary[
                "statistics"
            ]["maximum"],

            "coverage_percentage": summary[
                "coverage_percentage"
            ],

            "selected_pixels": summary[
                "selected_pixels"
            ],

            "spatial_bounds": summary[
                "selected_region_bounds"
            ],

            "crs": summary["crs"],

            "resolution": summary[
                "resolution"
            ],
        }

        if "area_km2" in summary:

            evidence["area_km2"] = (
                summary["area_km2"]
            )

        return evidence



def analyze_index(
    raster_path: str,
    index: np.ndarray,
    index_name: str,
    threshold: Optional[Union[float, tuple]] = None,
    condition: str = "greater"
) -> Dict:
    """
    Convenience wrapper.

    Example:

        result = analyze_index(
            "image.tif",
            ndvi,
            "NDVI",
            threshold=0.30,
            condition="greater"
        )
    """

    analyzer = GeospatialIndexAnalyzer(
        raster_path
    )

    return analyzer.summarize(
        index=index,
        index_name=index_name,
        threshold=threshold,
        condition=condition
    )



def generate_index_evidence(
    raster_path: str,
    index: np.ndarray,
    index_name: str,
    threshold: float,
    condition: str = "greater"
) -> Dict:
    """
    Generate structured evidence directly.

    This is useful when another module wants to do:

        evidence = generate_index_evidence(...)
    """

    analyzer = GeospatialIndexAnalyzer(
        raster_path
    )

    return analyzer.generate_evidence(
        index=index,
        index_name=index_name,
        threshold=threshold,
        condition=condition
    )


if __name__ == "__main__":

    print(
        "TriNetra geospatial/indices.py loaded successfully."
    )

    print(
        "Supported indices:",
        ", ".join(sorted(SUPPORTED_INDICES))
    )