import rasterio
import numpy as np

from pathlib import Path
from PIL import Image


class OpticalProcessor:
    """
    Processing utilities for optical satellite imagery.

    Supports:
        - Band extraction
        - Band normalization
        - RGB generation
        - False-color generation
        - NDVI
        - NDWI
        - NDBI
        - Saving processed images
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

        self.data = None
        self.profile = None
        self.band_count = None

        self._load()

    def _load(self):
        """Load the GeoTIFF into memory."""

        with rasterio.open(self.file_path) as src:

            self.data = src.read().astype(np.float32)

            self.profile = src.profile.copy()

            self.band_count = src.count

            self.nodata = src.nodata

            self.crs = src.crs

            self.transform = src.transform


    def get_band(self, band_number):
        """
        Return a single band.

        Band numbering follows GeoTIFF convention:
        Band 1, Band 2, Band 3, ...
        """

        if band_number < 1 or band_number > self.band_count:
            raise ValueError(
                f"Band {band_number} does not exist. "
                f"Available bands: 1-{self.band_count}"
            )

        return self.data[band_number - 1]

    def get_band_count(self):
        """Return the number of bands."""

        return self.band_count

    @staticmethod
    def normalize_band(
        band,
        lower_percentile=2,
        upper_percentile=98
    ):
        """
        Normalize a band to the range 0-255.

        Percentile-based normalization reduces the influence
        of extreme pixels.
        """

        band = band.astype(np.float32)

        valid_pixels = band[np.isfinite(band)]

        if valid_pixels.size == 0:
            return np.zeros(
                band.shape,
                dtype=np.uint8
            )

        low = np.percentile(
            valid_pixels,
            lower_percentile
        )

        high = np.percentile(
            valid_pixels,
            upper_percentile
        )

        if high <= low:
            return np.zeros(
                band.shape,
                dtype=np.uint8
            )

        normalized = (
            (band - low) /
            (high - low)
        )

        normalized = np.clip(
            normalized,
            0,
            1
        )

        normalized = (
            normalized * 255
        )

        normalized[~np.isfinite(normalized)] = 0

        return normalized.astype(np.uint8)


    def create_rgb(
        self,
        red_band,
        green_band,
        blue_band
    ):
        """
        Create an RGB image from three bands.

        Returns:
            NumPy array with shape:
            (height, width, 3)
        """

        red = self.get_band(red_band)
        green = self.get_band(green_band)
        blue = self.get_band(blue_band)

        red = self.normalize_band(red)
        green = self.normalize_band(green)
        blue = self.normalize_band(blue)

        rgb = np.stack(
            [
                red,
                green,
                blue
            ],
            axis=-1
        )

        return rgb


    def create_false_color(
        self,
        nir_band,
        red_band,
        green_band
    ):
        """
        Create a false-color composite.

        Standard vegetation false-color:
            Red channel   = NIR
            Green channel = Red
            Blue channel  = Green
        """

        nir = self.get_band(nir_band)
        red = self.get_band(red_band)
        green = self.get_band(green_band)

        nir = self.normalize_band(nir)
        red = self.normalize_band(red)
        green = self.normalize_band(green)

        false_color = np.stack(
            [
                nir,
                red,
                green
            ],
            axis=-1
        )

        return false_color

    def calculate_ndvi(
        self,
        nir_band,
        red_band
    ):
        """
        Calculate NDVI.

        NDVI = (NIR - Red) / (NIR + Red)

        Typical range:
            -1 to +1
        """

        nir = self.get_band(nir_band)
        red = self.get_band(red_band)

        denominator = nir + red

        ndvi = np.full(
            nir.shape,
            np.nan,
            dtype=np.float32
        )

        valid = (
            np.isfinite(nir) &
            np.isfinite(red) &
            (denominator != 0)
        )

        ndvi[valid] = (
            (nir[valid] - red[valid]) /
            denominator[valid]
        )

        return np.clip(
            ndvi,
            -1,
            1
        )


    def calculate_ndwi(
        self,
        green_band,
        nir_band
    ):
        """
        Calculate NDWI.

        NDWI = (Green - NIR) / (Green + NIR)
        """

        green = self.get_band(green_band)
        nir = self.get_band(nir_band)

        denominator = green + nir

        ndwi = np.full(
            green.shape,
            np.nan,
            dtype=np.float32
        )

        valid = (
            np.isfinite(green) &
            np.isfinite(nir) &
            (denominator != 0)
        )

        ndwi[valid] = (
            (green[valid] - nir[valid]) /
            denominator[valid]
        )

        return np.clip(
            ndwi,
            -1,
            1
        )


    def calculate_ndbi(
        self,
        swir_band,
        nir_band
    ):
        """
        Calculate NDBI.

        NDBI = (SWIR - NIR) / (SWIR + NIR)
        """

        swir = self.get_band(swir_band)
        nir = self.get_band(nir_band)

        denominator = swir + nir

        ndbi = np.full(
            swir.shape,
            np.nan,
            dtype=np.float32
        )

        valid = (
            np.isfinite(swir) &
            np.isfinite(nir) &
            (denominator != 0)
        )

        ndbi[valid] = (
            (swir[valid] - nir[valid]) /
            denominator[valid]
        )

        return np.clip(
            ndbi,
            -1,
            1
        )


    @staticmethod
    def normalize_index(index):
        """
        Convert an index such as NDVI/NDWI/NDBI
        from approximately [-1, 1] to [0, 255].
        """

        result = (
            (index + 1) /
            2
        )

        result = np.clip(
            result,
            0,
            1
        )

        result[~np.isfinite(result)] = 0

        return (
            result * 255
        ).astype(np.uint8)

    @staticmethod
    def save_image(
        image,
        output_path
    ):
        """
        Save an RGB or grayscale NumPy array as PNG/JPEG.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        image = np.asarray(image)

        if image.dtype != np.uint8:
            image = np.clip(
                image,
                0,
                255
            ).astype(np.uint8)

        Image.fromarray(image).save(
            output_path
        )

        return str(output_path)

    def save_index_geotiff(
        self,
        index,
        output_path
    ):
        """
        Save a calculated index such as NDVI as a GeoTIFF.

        The original CRS and spatial transform are preserved.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        profile = self.profile.copy()

        profile.update(
            {
                "driver": "GTiff",
                "count": 1,
                "dtype": "float32",
                "nodata": np.nan,
                "compress": "deflate"
            }
        )

        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as dst:

            dst.write(
                index.astype(np.float32),
                1
            )

        return str(output_path)


    def print_info(self):
        """Print basic information about the optical raster."""

        print("\n========== Optical Image Information ==========")

        print(f"File       : {self.file_path}")
        print(f"Bands      : {self.band_count}")
        print(
            f"Dimensions : "
            f"{self.data.shape[2]} x "
            f"{self.data.shape[1]}"
        )
        print(f"CRS        : {self.crs}")
        print(f"NoData     : {self.nodata}")

        print("\nAvailable Bands:")

        for i in range(1, self.band_count + 1):

            band = self.get_band(i)

            valid = band[np.isfinite(band)]

            if valid.size > 0:

                print(
                    f"Band {i}: "
                    f"min={valid.min():.2f}, "
                    f"max={valid.max():.2f}, "
                    f"mean={valid.mean():.2f}"
                )

            else:

                print(
                    f"Band {i}: no valid pixels"
                )

        print("===============================================\n")



def calculate_ndvi(
    file_path,
    nir_band,
    red_band
):
    """
    Convenience function for NDVI calculation.
    """

    processor = OpticalProcessor(
        file_path
    )

    return processor.calculate_ndvi(
        nir_band=nir_band,
        red_band=red_band
    )


def create_rgb(
    file_path,
    red_band,
    green_band,
    blue_band
):
    """
    Convenience function for RGB creation.
    """

    processor = OpticalProcessor(
        file_path
    )

    return processor.create_rgb(
        red_band=red_band,
        green_band=green_band,
        blue_band=blue_band
    )



if __name__ == "__main__":

    file_path = "data/input/Sample.tif"

    try:

        processor = OpticalProcessor(
            file_path
        )

        processor.print_info()

        print(
            "\nOptical processing module "
            "loaded successfully."
        )

        print(
            f"Number of bands: "
            f"{processor.get_band_count()}"
        )

    except Exception as e:

        print(
            f"Error processing optical image: {e}"
        )