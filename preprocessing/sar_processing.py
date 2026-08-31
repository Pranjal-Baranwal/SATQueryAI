import rasterio
import numpy as np

from pathlib import Path
from PIL import Image


class SARProcessor:

    def __init__(self, file_path):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"SAR GeoTIFF file not found: {self.file_path}"
            )

        if self.file_path.suffix.lower() not in [".tif", ".tiff"]:
            raise ValueError(
                "Input file must be a GeoTIFF (.tif or .tiff)"
            )

        self.data = None
        self.profile = None
        self.band_count = None
        self.nodata = None
        self.crs = None
        self.transform = None

        self._load()


    def _load(self):
        """Load the SAR GeoTIFF into memory."""

        with rasterio.open(self.file_path) as src:

            self.data = src.read().astype(np.float32)

            self.profile = src.profile.copy()

            self.band_count = src.count

            self.nodata = src.nodata

            self.crs = src.crs

            self.transform = src.transform


    def get_band(self, band_number):
        """
        Return a specific SAR band.

        Band numbering starts from 1.
        """

        if band_number < 1 or band_number > self.band_count:
            raise ValueError(
                f"Band {band_number} does not exist. "
                f"Available bands: 1-{self.band_count}"
            )

        return self.data[band_number - 1]

    def get_band_count(self):
        """Return number of SAR bands."""

        return self.band_count


    @staticmethod
    def linear_to_db(
        power,
        min_power=1e-10
    ):

        power = np.asarray(
            power,
            dtype=np.float32
        )

        power = np.maximum(
            power,
            min_power
        )

        db = 10.0 * np.log10(power)

        return db


    @staticmethod
    def db_to_linear(db):

        db = np.asarray(
            db,
            dtype=np.float32
        )

        return np.power(
            10.0,
            db / 10.0
        )


    @staticmethod
    def normalize(
        image,
        lower_percentile=2,
        upper_percentile=98
    ):
        """
        Normalize SAR data to 0-255 for visualization.

        Percentile clipping reduces the effect of extreme
        SAR speckle values.
        """

        image = image.astype(
            np.float32
        )

        valid = image[
            np.isfinite(image)
        ]

        if valid.size == 0:
            return np.zeros(
                image.shape,
                dtype=np.uint8
            )

        low = np.percentile(
            valid,
            lower_percentile
        )

        high = np.percentile(
            valid,
            upper_percentile
        )

        if high <= low:
            return np.zeros(
                image.shape,
                dtype=np.uint8
            )

        normalized = (
            (image - low) /
            (high - low)
        )

        normalized = np.clip(
            normalized,
            0,
            1
        )

        normalized[
            ~np.isfinite(normalized)
        ] = 0

        return (
            normalized * 255
        ).astype(np.uint8)


    def get_vv(
        self,
        band_number=1,
        input_scale="linear"
    ):

        vv = self.get_band(
            band_number
        )

        if input_scale.lower() == "linear":

            return self.linear_to_db(vv)

        elif input_scale.lower() == "db":

            return vv

        else:

            raise ValueError(
                "input_scale must be "
                "'linear' or 'db'"
            )

    def get_vh(
        self,
        band_number=2,
        input_scale="linear"
    ):
        """
        Get VH polarization.

        Parameters
        ----------
        band_number : int
            Band containing VH.

        input_scale : str
            "linear" or "db"
        """

        vh = self.get_band(
            band_number
        )

        if input_scale.lower() == "linear":

            return self.linear_to_db(vh)

        elif input_scale.lower() == "db":

            return vh

        else:

            raise ValueError(
                "input_scale must be "
                "'linear' or 'db'"
            )


    def calculate_vv_vh_ratio(
        self,
        vv_band=1,
        vh_band=2
    ):
        """
        Calculate VV/VH ratio in linear space.

        Formula:

            VV/VH
        """

        vv = self.get_band(
            vv_band
        )

        vh = self.get_band(
            vh_band
        )

        ratio = np.full(
            vv.shape,
            np.nan,
            dtype=np.float32
        )

        valid = (
            np.isfinite(vv) &
            np.isfinite(vh) &
            (vh != 0)
        )

        ratio[valid] = (
            vv[valid] /
            vh[valid]
        )

        return ratio


    def calculate_vv_vh_difference(
        self,
        vv_band=1,
        vh_band=2,
        input_scale="db"
    ):
        """
        Calculate VV-VH difference.

        If input_scale is "db":

            Difference = VV_dB - VH_dB

        If input_scale is "linear":

            Difference = VV - VH
        """

        vv = self.get_band(
            vv_band
        )

        vh = self.get_band(
            vh_band
        )

        if input_scale.lower() == "db":

            vv = self.linear_to_db(vv)
            vh = self.linear_to_db(vh)

        difference = (
            vv - vh
        )

        difference[
            ~np.isfinite(difference)
        ] = np.nan

        return difference


    def create_sar_image(
        self,
        band_number=1,
        input_scale="linear"
    ):

        if input_scale.lower() == "linear":

            band = self.get_band(
                band_number
            )

            band = self.linear_to_db(
                band
            )

        elif input_scale.lower() == "db":

            band = self.get_band(
                band_number
            )

        else:

            raise ValueError(
                "input_scale must be "
                "'linear' or 'db'"
            )

        return self.normalize(
            band
        )

    def create_pseudo_rgb(
        self,
        vv_band=1,
        vh_band=2
    ):
        """
        Create a pseudo-RGB representation from SAR.

        Channels:

            R = VV
            G = VH
            B = VV - VH

        This does NOT represent true RGB.

        It is a visualization representation for
        multimodal/VLM processing.
        """

        if self.band_count < 2:

            raise ValueError(
                "Pseudo-RGB requires at least "
                "two SAR bands (VV and VH)."
            )

        vv = self.get_vv(
            vv_band,
            input_scale="linear"
        )

        vh = self.get_vh(
            vh_band,
            input_scale="linear"
        )

        difference = (
            vv - vh
        )

        vv_norm = self.normalize(
            vv
        )

        vh_norm = self.normalize(
            vh
        )

        diff_norm = self.normalize(
            difference
        )

        pseudo_rgb = np.stack(
            [
                vv_norm,
                vh_norm,
                diff_norm
            ],
            axis=-1
        )

        return pseudo_rgb


    @staticmethod
    def save_image(
        image,
        output_path
    ):
        """
        Save a SAR visualization as PNG/JPEG.
        """

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        image = np.asarray(
            image
        )

        if image.dtype != np.uint8:

            image = np.clip(
                image,
                0,
                255
            ).astype(
                np.uint8
            )

        Image.fromarray(
            image
        ).save(
            output_path
        )

        return str(output_path)

    def save_geotiff(
        self,
        image,
        output_path
    ):
        """
        Save a processed SAR product as GeoTIFF.

        The original CRS, transform, width and height
        are preserved.
        """

        output_path = Path(
            output_path
        )

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
                image.astype(
                    np.float32
                ),
                1
            )

        return str(output_path)


    def print_info(self):
        """Print SAR raster information and statistics."""

        print(
            "\n========== SAR Image Information =========="
        )

        print(
            f"File       : {self.file_path}"
        )

        print(
            f"Bands      : {self.band_count}"
        )

        print(
            f"Dimensions : "
            f"{self.data.shape[2]} x "
            f"{self.data.shape[1]}"
        )

        print(
            f"CRS        : {self.crs}"
        )

        print(
            f"NoData     : {self.nodata}"
        )

        print("\nBand Statistics:")

        for i in range(
            1,
            self.band_count + 1
        ):

            band = self.get_band(i)

            valid = band[
                np.isfinite(band)
            ]

            if valid.size > 0:

                print(
                    f"Band {i}: "
                    f"min={valid.min():.4f}, "
                    f"max={valid.max():.4f}, "
                    f"mean={valid.mean():.4f}"
                )

            else:

                print(
                    f"Band {i}: "
                    f"no valid pixels"
                )

        print(
            "============================================\n"
        )


def linear_to_db(power):
    """
    Convenience function for linear → dB conversion.
    """

    return SARProcessor.linear_to_db(
        power
    )


def db_to_linear(db):
    """
    Convenience function for dB → linear conversion.
    """

    return SARProcessor.db_to_linear(
        db
    )


if __name__ == "__main__":

    file_path = "data/input/Sample.tif"

    try:

        processor = SARProcessor(
            file_path
        )

        processor.print_info()

        print(
            "SAR processing module "
            "loaded successfully."
        )

        print(
            f"Number of bands: "
            f"{processor.get_band_count()}"
        )


        sar_image = processor.create_sar_image(
            band_number=1,
            input_scale="linear"
        )

        output_path = processor.save_image(
            sar_image,
            "data/output/SAR_band1.png"
        )

        print(
            f"SAR visualization saved to: "
            f"{output_path}"
        )

    except Exception as e:

        print(
            f"Error processing SAR image: {e}"
        )