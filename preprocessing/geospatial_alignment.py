import rasterio
import numpy as np

from pathlib import Path
from rasterio.warp import reproject
from rasterio.enums import Resampling


class GeoSpatialAligner:
    """
    Aligns one GeoTIFF to the spatial grid of another GeoTIFF.

    The reference GeoTIFF determines:
        - CRS
        - Resolution
        - Width
        - Height
        - Transform
        - Geographic extent
    """

    def __init__(self, reference_path, source_path):
        self.reference_path = Path(reference_path)
        self.source_path = Path(source_path)

        self._validate_file(self.reference_path)
        self._validate_file(self.source_path)

        self.reference_info = None
        self.source_info = None

    @staticmethod
    def _validate_file(file_path):
        """Validate that the input is a GeoTIFF."""

        if not file_path.exists():
            raise FileNotFoundError(
                f"GeoTIFF file not found: {file_path}"
            )

        if file_path.suffix.lower() not in [".tif", ".tiff"]:
            raise ValueError(
                f"File must be a GeoTIFF (.tif or .tiff): {file_path}"
            )

    @staticmethod
    def _get_raster_info(file_path):
        """Extract spatial information from a GeoTIFF."""

        with rasterio.open(file_path) as src:
            return {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": src.crs,
                "transform": src.transform,
                "resolution": src.res,
                "bounds": src.bounds,
                "dtype": src.dtypes,
                "nodata": src.nodata
            }

    def inspect(self):
        """
        Inspect both rasters and return their spatial information.
        """

        self.reference_info = self._get_raster_info(
            self.reference_path
        )

        self.source_info = self._get_raster_info(
            self.source_path
        )

        return {
            "reference": self.reference_info,
            "source": self.source_info
        }

    def check_alignment(self):
        """
        Check whether the source raster is already aligned
        with the reference raster.
        """

        if self.reference_info is None or self.source_info is None:
            self.inspect()

        reference = self.reference_info
        source = self.source_info

        checks = {
            "same_crs": reference["crs"] == source["crs"],
            "same_resolution": np.allclose(
                reference["resolution"],
                source["resolution"]
            ),
            "same_dimensions": (
                reference["width"] == source["width"]
                and
                reference["height"] == source["height"]
            ),
            "same_transform": (
                reference["transform"] == source["transform"]
            ),
            "same_bounds": (
                np.allclose(
                    [
                        reference["bounds"].left,
                        reference["bounds"].bottom,
                        reference["bounds"].right,
                        reference["bounds"].top
                    ],
                    [
                        source["bounds"].left,
                        source["bounds"].bottom,
                        source["bounds"].right,
                        source["bounds"].top
                    ]
                )
            )
        }

        checks["aligned"] = all(checks.values())

        return checks

    def align(
        self,
        output_path,
        resampling_method="bilinear"
    ):
        """
        Reproject and align the source raster to the
        exact spatial grid of the reference raster.

        Parameters
        ----------
        output_path : str
            Path where the aligned GeoTIFF will be saved.

        resampling_method : str
            Resampling method.

            Supported:
                nearest
                bilinear
                cubic
                cubic_spline
                lanczos
                average
                mode

        Returns
        -------
        str
            Path to the aligned GeoTIFF.
        """

        if self.reference_info is None or self.source_info is None:
            self.inspect()

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        reference = self.reference_info

        # Convert string to Rasterio Resampling enum
        try:
            resampling = getattr(
                Resampling,
                resampling_method
            )
        except AttributeError:
            raise ValueError(
                f"Unsupported resampling method: "
                f"{resampling_method}"
            )

        with rasterio.open(self.reference_path) as reference_src:

            reference_crs = reference_src.crs
            reference_transform = reference_src.transform
            reference_width = reference_src.width
            reference_height = reference_src.height

        with rasterio.open(self.source_path) as source_src:

            # Output profile follows the source raster,
            # but spatial properties follow the reference.
            profile = source_src.profile.copy()

            profile.update(
                {
                    "crs": reference_crs,
                    "transform": reference_transform,
                    "width": reference_width,
                    "height": reference_height,
                    "driver": "GTiff",
                    "compress": "deflate"
                }
            )

            # Use float32 so NaN can represent NoData safely.
            profile.update(
                {
                    "dtype": "float32",
                    "nodata": np.nan
                }
            )

            with rasterio.open(output_path, "w", **profile) as destination:

                for band_index in range(1, source_src.count + 1):

                    destination_array = np.full(
                        (
                            reference_height,
                            reference_width
                        ),
                        np.nan,
                        dtype=np.float32
                    )

                    reproject(
                        source=rasterio.band(
                            source_src,
                            band_index
                        ),
                        destination=destination_array,

                        src_transform=source_src.transform,
                        src_crs=source_src.crs,

                        dst_transform=reference_transform,
                        dst_crs=reference_crs,

                        src_nodata=source_src.nodata,
                        dst_nodata=np.nan,

                        resampling=resampling
                    )

                    destination.write(
                        destination_array,
                        band_index
                    )

        return str(output_path)

    def print_report(self):
        """Print a readable spatial alignment report."""

        if self.reference_info is None or self.source_info is None:
            self.inspect()

        checks = self.check_alignment()

        print("\n========== GeoSpatial Alignment ==========")

        print("\nREFERENCE RASTER")
        print(f"File        : {self.reference_path}")
        print(
            f"Dimensions  : "
            f"{self.reference_info['width']} x "
            f"{self.reference_info['height']}"
        )
        print(f"Bands       : {self.reference_info['count']}")
        print(f"CRS         : {self.reference_info['crs']}")
        print(
            f"Resolution  : "
            f"{self.reference_info['resolution']}"
        )
        print(
            f"Bounds      : "
            f"{self.reference_info['bounds']}"
        )

        print("\nSOURCE RASTER")
        print(f"File        : {self.source_path}")
        print(
            f"Dimensions  : "
            f"{self.source_info['width']} x "
            f"{self.source_info['height']}"
        )
        print(f"Bands       : {self.source_info['count']}")
        print(f"CRS         : {self.source_info['crs']}")
        print(
            f"Resolution  : "
            f"{self.source_info['resolution']}"
        )
        print(
            f"Bounds      : "
            f"{self.source_info['bounds']}"
        )

        print("\nALIGNMENT CHECK")

        print(
            f"Same CRS        : "
            f"{checks['same_crs']}"
        )

        print(
            f"Same Resolution : "
            f"{checks['same_resolution']}"
        )

        print(
            f"Same Dimensions : "
            f"{checks['same_dimensions']}"
        )

        print(
            f"Same Transform  : "
            f"{checks['same_transform']}"
        )

        print(
            f"Same Bounds     : "
            f"{checks['same_bounds']}"
        )

        print(
            f"\nAlready Aligned : "
            f"{checks['aligned']}"
        )

        print("===========================================\n")


def align_geotiffs(
    reference_path,
    source_path,
    output_path,
    resampling_method="bilinear"
):
    """
    Convenience function for aligning a source GeoTIFF
    to a reference GeoTIFF.
    """

    aligner = GeoSpatialAligner(
        reference_path=reference_path,
        source_path=source_path
    )

    return aligner.align(
        output_path=output_path,
        resampling_method=resampling_method
    )


if __name__ == "__main__":

    reference_file = "data/input/Sample.tif"
    source_file = "data/input/Sample.tif"

    output_file = "data/output/Aligned_sample.tif"

    try:

        aligner = GeoSpatialAligner(
            reference_path=reference_file,
            source_path=source_file
        )

        # Show alignment information
        aligner.print_report()

        # Align source to reference
        result = aligner.align(
            output_path=output_file,
            resampling_method="bilinear"
        )

        print(f"Aligned GeoTIFF saved to: {result}")

    except Exception as e:

        print(
            f"Error during geospatial alignment: {e}"
        )