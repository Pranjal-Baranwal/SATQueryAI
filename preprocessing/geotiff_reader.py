import rasterio
import numpy as np
from pathlib import Path


class GeoTIFFReader:
    """
    Reader and basic handler for GeoTIFF satellite imagery.

    Provides:
    - Raster band data
    - Metadata
    - CRS
    - Resolution
    - Bounds
    - Transform
    - NoData handling
    """

    def __init__(self, file_path):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"GeoTIFF file not found: {self.file_path}"
            )

        if self.file_path.suffix.lower() not in [".tif", ".tiff"]:
            raise ValueError(
                "Input file must be a GeoTIFF (.tif or .tiff)"
            )

        self.dataset = None
        self.data = None
        self.metadata = None

    def read(self):
        """Read the complete GeoTIFF into a NumPy array."""

        with rasterio.open(self.file_path) as src:

            self.dataset = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes,
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "resolution": src.res,
                "nodata": src.nodata,
                "driver": src.driver,
                "name": src.name
            }

            self.data = src.read()

            if src.nodata is not None:
                self.data = self.data.astype(np.float32)
                self.data[self.data == src.nodata] = np.nan

            self.metadata = self.dataset

        return self.data

    def get_metadata(self):
        """
        Return GeoTIFF metadata.

        If the file has not been read yet, it reads the file first.
        """

        if self.metadata is None:
            self.read()

        return self.metadata

    def get_band(self, band_number):
        """
        Return a specific raster band.

        Band numbering starts from 1, following the GeoTIFF convention.
        """

        if self.data is None:
            self.read()

        if band_number < 1 or band_number > self.metadata["count"]:
            raise ValueError(
                f"Band {band_number} does not exist. "
                f"Available bands: 1-{self.metadata['count']}"
            )

        return self.data[band_number - 1]

    def get_all_bands(self):
        """Return all raster bands."""

        if self.data is None:
            self.read()

        return self.data

    def get_rgb(self, red=1, green=2, blue=3):
        """
        Return RGB bands as a (height, width, 3) NumPy array.

        Default assumes:
            Band 1 = Red
            Band 2 = Green
            Band 3 = Blue

        For satellite datasets with different band numbering,
        specify the appropriate band numbers.
        """

        if self.data is None:
            self.read()

        required_bands = [red, green, blue]

        for band in required_bands:
            if band < 1 or band > self.metadata["count"]:
                raise ValueError(
                    f"Band {band} does not exist. "
                    f"Available bands: 1-{self.metadata['count']}"
                )

        rgb = np.stack(
            [
                self.get_band(red),
                self.get_band(green),
                self.get_band(blue)
            ],
            axis=-1
        )

        return rgb

    def get_shape(self):
        """Return raster dimensions as (bands, height, width)."""

        if self.data is None:
            self.read()

        return self.data.shape

    def close(self):
        """Release stored raster data."""

        self.dataset = None
        self.data = None
        self.metadata = None


def read_geotiff(file_path):
    """
    Convenience function for reading a GeoTIFF.

    Returns:
        data: NumPy array with shape (bands, height, width)
        metadata: Dictionary containing geospatial metadata
    """

    reader = GeoTIFFReader(file_path)

    data = reader.read()
    metadata = reader.get_metadata()

    return data, metadata


if __name__ == "__main__":

    file_path = "data/input/Sample.tif"

    try:

        reader = GeoTIFFReader(file_path)

        data = reader.read()
        metadata = reader.get_metadata()

        print("\n========== GeoTIFF Information ==========")

        print(f"File       : {metadata['name']}")
        print(f"Dimensions : {metadata['width']} x {metadata['height']}")
        print(f"Bands     : {metadata['count']}")
        print(f"Data Type : {metadata['dtype']}")
        print(f"CRS       : {metadata['crs']}")
        print(f"Resolution: {metadata['resolution']}")
        print(f"Bounds    : {metadata['bounds']}")
        print(f"NoData    : {metadata['nodata']}")
        print(f"Shape     : {data.shape}")

        print("=========================================\n")

    except Exception as e:
        print(f"Error reading GeoTIFF: {e}")