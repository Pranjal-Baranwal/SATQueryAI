
from pathlib import Path

import numpy as np
import rasterio

from PIL import Image



DEFAULT_OUTPUT_DIR = "data/output"

DEFAULT_TARGET_SIZE = 1024



class CrossModalFusion:
    """
    Optical + SAR fusion engine.

    Supports:

        Optical:
            PNG
            JPG
            JPEG
            TIFF
            GeoTIFF

        SAR:
            TIFF
            GeoTIFF
            PNG
            JPG
            JPEG
    """

    def __init__(
        self,
        output_dir=DEFAULT_OUTPUT_DIR,
        target_size=DEFAULT_TARGET_SIZE
    ):

        self.output_dir = Path(
            output_dir
        )

        self.target_size = target_size

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


    @staticmethod
    def validate_file(
        file_path
    ):

        file_path = Path(
            file_path
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        supported_formats = [
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff"
        ]

        if file_path.suffix.lower() not in supported_formats:

            raise ValueError(
                f"Unsupported file format: "
                f"{file_path.suffix}"
            )

        return file_path


    @staticmethod
    def load_standard_image(
        file_path
    ):
        """
        Load PNG/JPG/JPEG image.

        Returns:
            NumPy array with shape:
            (bands, height, width)
        """

        image = Image.open(
            file_path
        )

        image = image.convert(
            "RGB"
        )

        array = np.asarray(
            image
        )

        array = array.astype(
            np.float32
        )

        # HWC → CHW

        array = np.transpose(
            array,
            (2, 0, 1)
        )

        return array

    @staticmethod
    def load_geotiff(
        file_path
    ):
        """
        Load GeoTIFF using Rasterio.

        Returns:
            NumPy array:
            (bands, height, width)
        """

        with rasterio.open(
            file_path
        ) as src:

            data = src.read()

        return data.astype(
            np.float32
        )

    def load_image(
        self,
        file_path
    ):
        """
        Automatically select the correct loader based
        on file extension.
        """

        file_path = self.validate_file(
            file_path
        )

        extension = (
            file_path.suffix.lower()
        )

        if extension in [
            ".png",
            ".jpg",
            ".jpeg"
        ]:

            print(
                f"Loading standard image: "
                f"{file_path}"
            )

            data = self.load_standard_image(
                file_path
            )

        elif extension in [
            ".tif",
            ".tiff"
        ]:

            print(
                f"Loading GeoTIFF: "
                f"{file_path}"
            )

            data = self.load_geotiff(
                file_path
            )

        else:

            raise ValueError(
                f"Unsupported image type: "
                f"{extension}"
            )

        return data


    @staticmethod
    def normalize(
        data
    ):
        """
        Normalize every band to [0, 1].

        Uses percentile clipping to reduce the effect
        of extreme pixel values.
        """

        data = data.astype(
            np.float32
        )

        data = np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        normalized = np.zeros_like(
            data,
            dtype=np.float32
        )

        for band_index in range(
            data.shape[0]
        ):

            band = data[
                band_index
            ]

            low = np.percentile(
                band,
                2
            )

            high = np.percentile(
                band,
                98
            )

            if high > low:

                normalized[
                    band_index
                ] = np.clip(
                    (
                        band - low
                    )
                    /
                    (
                        high - low
                    ),
                    0.0,
                    1.0
                )

            else:

                normalized[
                    band_index
                ] = 0.0

        return normalized

    @staticmethod
    def resize_array(
        data,
        target_height,
        target_width
    ):
        """
        Resize each raster band.
        """

        resized = []

        for band in data:


            band_min = band.min()
            band_max = band.max()

            if band_max > band_min:

                band_uint8 = (
                    (
                        band - band_min
                    )
                    /
                    (
                        band_max - band_min
                    )
                    *
                    255
                ).astype(
                    np.uint8
                )

            else:

                band_uint8 = np.zeros_like(
                    band,
                    dtype=np.uint8
                )

            image = Image.fromarray(
                band_uint8
            )

            image = image.resize(
                (
                    target_width,
                    target_height
                ),
                Image.Resampling.BILINEAR
            )

            resized_band = (
                np.asarray(
                    image
                ).astype(
                    np.float32
                )
                /
                255.0
            )

            resized.append(
                resized_band
            )

        return np.stack(
            resized,
            axis=0
        )


    def resize_to_target(
        self,
        data
    ):
        """
        Resize while preserving aspect ratio.

        The longest dimension is reduced to
        target_size.
        """

        _, height, width = (
            data.shape
        )

        scale = min(
            1.0,
            self.target_size
            /
            max(
                height,
                width
            )
        )

        target_height = max(
            1,
            int(
                height * scale
            )
        )

        target_width = max(
            1,
            int(
                width * scale
            )
        )

        if (
            target_height == height
            and
            target_width == width
        ):

            return data

        return self.resize_array(
            data,
            target_height,
            target_width
        )


    @staticmethod
    def align_dimensions(
        optical,
        sar
    ):
        """
        Ensure Optical and SAR have identical
        spatial dimensions.

        The smaller dimensions are used as the
        common target.
        """

        optical_height = (
            optical.shape[1]
        )

        optical_width = (
            optical.shape[2]
        )

        sar_height = (
            sar.shape[1]
        )

        sar_width = (
            sar.shape[2]
        )

        target_height = min(
            optical_height,
            sar_height
        )

        target_width = min(
            optical_width,
            sar_width
        )

        if (
            optical_height != target_height
            or
            optical_width != target_width
        ):

            optical = (
                CrossModalFusion.resize_array(
                    optical,
                    target_height,
                    target_width
                )
            )

        if (
            sar_height != target_height
            or
            sar_width != target_width
        ):

            sar = (
                CrossModalFusion.resize_array(
                    sar,
                    target_height,
                    target_width
                )
            )

        return optical, sar


    @staticmethod
    def create_optical_features(
        optical
    ):
        """
        Prepare normalized Optical representation.
        """

        return optical


    @staticmethod
    def create_sar_features(
        sar
    ):
        """
        Prepare normalized SAR representation.
        """

        return sar


    @staticmethod
    def fuse_features(
        optical,
        sar
    ):
        """
        Concatenate Optical and SAR features
        along the channel dimension.

        Example:

            Optical = 3 channels
            SAR     = 1 channel

            Output  = 4 channels
        """

        if optical.shape[1:] != sar.shape[1:]:

            raise ValueError(
                "Optical and SAR spatial dimensions "
                "do not match."
            )

        fused = np.concatenate(
            [
                optical,
                sar
            ],
            axis=0
        )

        return fused

    @staticmethod
    def calculate_statistics(
        optical,
        sar,
        fused
    ):
        """
        Calculate cross-modal statistics.
        """

        optical_mean = float(
            np.mean(
                optical
            )
        )

        optical_std = float(
            np.std(
                optical
            )
        )

        sar_mean = float(
            np.mean(
                sar
            )
        )

        sar_std = float(
            np.std(
                sar
            )
        )

        fused_mean = float(
            np.mean(
                fused
            )
        )

        fused_std = float(
            np.std(
                fused
            )
        )

        return {
            "optical_bands": int(
                optical.shape[0]
            ),
            "sar_bands": int(
                sar.shape[0]
            ),
            "fused_channels": int(
                fused.shape[0]
            ),
            "height": int(
                fused.shape[1]
            ),
            "width": int(
                fused.shape[2]
            ),
            "optical_mean": optical_mean,
            "optical_std": optical_std,
            "sar_mean": sar_mean,
            "sar_std": sar_std,
            "fused_mean": fused_mean,
            "fused_std": fused_std
        }


    def save_features(
        self,
        fused
    ):
        """
        Save fused representation as NumPy array.
        """

        output_path = (
            self.output_dir
            /
            "cross_modal_features.npy"
        )

        np.save(
            output_path,
            fused
        )

        return output_path

    def process(
        self,
        optical_path,
        sar_path
    ):
        """
        Complete Optical + SAR fusion pipeline.
        """

        print(
            "\n========== CROSS-MODAL FUSION =========="
        )

        print(
            f"Optical: {optical_path}"
        )

        print(
            f"SAR:     {sar_path}"
        )

        optical = self.load_image(
            optical_path
        )

        print(
            f"Optical shape: "
            f"{optical.shape}"
        )


        sar = self.load_image(
            sar_path
        )

        print(
            f"SAR shape: "
            f"{sar.shape}"
        )


        print(
            "\nNormalizing modalities..."
        )

        optical = self.normalize(
            optical
        )

        sar = self.normalize(
            sar
        )

        print(
            "Resizing modalities..."
        )

        optical = self.resize_to_target(
            optical
        )

        sar = self.resize_to_target(
            sar
        )

        print(
            f"Optical resized: "
            f"{optical.shape}"
        )

        print(
            f"SAR resized: "
            f"{sar.shape}"
        )

        print(
            "\nAligning modalities..."
        )

        optical, sar = (
            self.align_dimensions(
                optical,
                sar
            )
        )

        print(
            f"Aligned dimensions: "
            f"{optical.shape[2]} x "
            f"{optical.shape[1]}"
        )


        optical_features = (
            self.create_optical_features(
                optical
            )
        )

        sar_features = (
            self.create_sar_features(
                sar
            )
        )


        print(
            "\nFusing Optical + SAR features..."
        )

        fused = self.fuse_features(
            optical_features,
            sar_features
        )

        print(
            f"Fused shape: "
            f"{fused.shape}"
        )


        statistics = (
            self.calculate_statistics(
                optical_features,
                sar_features,
                fused
            )
        )

        fused_path = (
            self.save_features(
                fused
            )
        )

        print(
            f"\nFused features saved to: "
            f"{fused_path}"
        )

        print(
            "\n========== FUSION STATISTICS =========="
        )

        for key, value in statistics.items():

            print(
                f"{key}: {value}"
            )

        print(
            "======================================="
        )

        return {
            "optical": optical_features,
            "sar": sar_features,
            "fused": fused,
            "statistics": statistics,
            "fused_path": str(
                fused_path
            )
        }



fusion_engine = (
    CrossModalFusion()
)



def fuse_optical_sar(
    optical_path,
    sar_path
):

    return fusion_engine.process(
        optical_path=optical_path,
        sar_path=sar_path
    )


if __name__ == "__main__":

    optical_path = (
        "data/input/T1.png"
    )

    sar_path = (
        "data/input/Sample.tif"
    )

    try:

        result = fuse_optical_sar(
            optical_path=optical_path,
            sar_path=sar_path
        )

        print(
            "\nCross-modal fusion completed successfully."
        )

    except Exception as e:

        print(
            f"\nCross-modal fusion error: {e}"
        )