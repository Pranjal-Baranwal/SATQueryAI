import rasterio
from pathlib import Path


class GeoTIFFMetadata:
    """
    Extracts and manages metadata from a GeoTIFF file.
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

        self.metadata = None

    def extract(self):
        """
        Extract important metadata from the GeoTIFF.
        """

        with rasterio.open(self.file_path) as src:

            self.metadata = {
                "file_name": self.file_path.name,
                "file_path": str(self.file_path),

                # Raster information
                "width": src.width,
                "height": src.height,
                "band_count": src.count,
                "data_types": list(src.dtypes),

                # Geospatial information
                "crs": str(src.crs) if src.crs else None,
                "crs_epsg": src.crs.to_epsg() if src.crs else None,
                "transform": src.transform,

                # Spatial extent
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top
                },

                # Pixel resolution
                "resolution": {
                    "x": src.res[0],
                    "y": src.res[1]
                },

                # Raster properties
                "nodata": src.nodata,
                "driver": src.driver,

                # Additional information
                "units": src.units,
                "descriptions": src.descriptions
            }

        return self.metadata

    def get(self, key, default=None):
        """
        Get a specific metadata value.
        """

        if self.metadata is None:
            self.extract()

        return self.metadata.get(key, default)

    def print_metadata(self):
        """
        Print metadata in a readable format.
        """

        if self.metadata is None:
            self.extract()

        print("\n========== GeoTIFF Metadata ==========")

        print(f"File Name       : {self.metadata['file_name']}")
        print(f"File Path       : {self.metadata['file_path']}")

        print("\n--- Raster Information ---")
        print(f"Width           : {self.metadata['width']}")
        print(f"Height          : {self.metadata['height']}")
        print(f"Bands           : {self.metadata['band_count']}")
        print(f"Data Types      : {self.metadata['data_types']}")

        print("\n--- Geospatial Information ---")
        print(f"CRS             : {self.metadata['crs']}")
        print(f"EPSG            : {self.metadata['crs_epsg']}")

        print("\n--- Resolution ---")
        print(f"X Resolution    : {self.metadata['resolution']['x']}")
        print(f"Y Resolution    : {self.metadata['resolution']['y']}")

        print("\n--- Bounds ---")
        print(f"Left            : {self.metadata['bounds']['left']}")
        print(f"Bottom          : {self.metadata['bounds']['bottom']}")
        print(f"Right           : {self.metadata['bounds']['right']}")
        print(f"Top             : {self.metadata['bounds']['top']}")

        print("\n--- Raster Properties ---")
        print(f"NoData          : {self.metadata['nodata']}")
        print(f"Driver          : {self.metadata['driver']}")
        print(f"Units           : {self.metadata['units']}")
        print(f"Descriptions    : {self.metadata['descriptions']}")

        print("======================================\n")


def get_geotiff_metadata(file_path):
    """
    Convenience function to extract GeoTIFF metadata.

    Returns:
        dict: GeoTIFF metadata
    """

    metadata_reader = GeoTIFFMetadata(file_path)

    return metadata_reader.extract()


if __name__ == "__main__":

    # Change this to your actual GeoTIFF file
    file_path = "data/input/Sample.tif"

    try:

        metadata_reader = GeoTIFFMetadata(file_path)

        metadata_reader.extract()
        metadata_reader.print_metadata()

    except Exception as e:
        print(f"Error extracting metadata: {e}")