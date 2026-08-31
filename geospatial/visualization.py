from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from PIL import Image, ImageDraw

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]

class SpatialVisualizer:
    """
    Visualize spatial outputs over the original image.

    Supported spatial outputs:
        1. Bounding boxes
        2. Segmentation masks
        3. Change masks
        4. Pixel coordinates
        5. Geographic coordinates
    """

    def __init__(self, image_path: str):

        self.image_path = Path(image_path)

        if not self.image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {self.image_path}"
            )

        self.width = 0
        self.height = 0

        self.crs = None
        self.transform = None
        self.bounds = None

        self.image = self._load_image()


    def _load_image(self) -> Image.Image:
        """
        Load JPEG/PNG directly or create an RGB display image
        from a GeoTIFF.
        """

        suffix = self.image_path.suffix.lower()

        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:

            image = Image.open(
                self.image_path
            ).convert("RGB")

            self.width, self.height = image.size

            return image

        if suffix in {".tif", ".tiff"}:

            return self._load_geotiff()

        raise ValueError(
            "Unsupported image format. "
            "Use GeoTIFF, JPEG or PNG."
        )


    def _load_geotiff(self) -> Image.Image:
        """
        Read a GeoTIFF and create a displayable RGB image.

        For the prototype:
            - 3+ bands -> first three bands
            - 2 bands  -> band1, band2, band1
            - 1 band   -> grayscale

        The geospatial metadata is retained.
        """

        with rasterio.open(
            self.image_path
        ) as src:

            self.crs = src.crs
            self.transform = src.transform
            self.bounds = src.bounds

            self.width = src.width
            self.height = src.height

            if src.count >= 3:

                data = src.read(
                    [1, 2, 3],
                    out_shape=(
                        3,
                        min(src.height, 1500),
                        min(src.width, 1500)
                    ),
                    resampling=rasterio.enums.Resampling.bilinear
                )

            elif src.count == 2:

                band1 = src.read(1)
                band2 = src.read(2)

                data = np.stack(
                    [
                        band1,
                        band2,
                        band1
                    ]
                )

            else:

                band = src.read(1)

                data = np.stack(
                    [
                        band,
                        band,
                        band
                    ]
                )

        channels = []

        for channel in data:

            channel = channel.astype(
                np.float32
            )

            valid = np.isfinite(
                channel
            )

            if not valid.any():

                scaled = np.zeros(
                    channel.shape,
                    dtype=np.uint8
                )

            else:

                values = channel[
                    valid
                ]

                low, high = np.percentile(
                    values,
                    [2, 98]
                )

                if high <= low:

                    scaled = np.zeros(
                        channel.shape,
                        dtype=np.uint8
                    )

                else:

                    scaled = (
                        np.clip(
                            (
                                channel - low
                            ) /
                            (
                                high - low
                            ),
                            0,
                            1
                        )
                        * 255
                    ).astype(
                        np.uint8
                    )

            channels.append(
                scaled
            )

        rgb = np.stack(
            channels[:3],
            axis=-1
        )

        return Image.fromarray(
            rgb,
            mode="RGB"
        )


    def save(
        self,
        output_path: str
    ) -> str:
        """
        Save the generated evidence image.
        """

        output = Path(
            output_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.image.save(
            output
        )

        return str(output)

    def draw_bounding_box(
        self,
        box: Box,
        label: Optional[str] = None,
        width: int = 4
    ) -> None:
        """
        Draw one bounding box.

        box format:
            (x1, y1, x2, y2)

        Coordinates are pixel coordinates.
        """

        draw = ImageDraw.Draw(
            self.image
        )

        x1, y1, x2, y2 = map(
            int,
            box
        )

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline="red",
            width=width
        )

        if label:

            text_y = max(
                0,
                y1 - 22
            )

            draw.text(
                (
                    x1,
                    text_y
                ),
                label,
                fill="red"
            )


    def draw_bounding_boxes(
        self,
        boxes: Sequence[
            Union[
                Box,
                Dict
            ]
        ],
        width: int = 4
    ) -> None:
        """
        Draw multiple bounding boxes.

        Supported formats:

            (x1, y1, x2, y2)

        or:

            {
                "box": (x1, y1, x2, y2),
                "label": "Building"
            }
        """

        for item in boxes:

            if isinstance(
                item,
                dict
            ):

                if "box" not in item:

                    raise ValueError(
                        "Bounding box dictionary "
                        "must contain 'box'."
                    )

                box = item["box"]
                label = item.get(
                    "label"
                )

            else:

                box = item
                label = None

            self.draw_bounding_box(
                box=box,
                label=label,
                width=width
            )


    @staticmethod
    def _prepare_mask(
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Convert a mask to boolean format.
        """

        mask = np.asarray(
            mask
        )

        if mask.ndim != 2:

            raise ValueError(
                "Mask must be a 2D array."
            )

        if mask.dtype == bool:

            return mask

        return mask > 0

    def _resize_mask(
        self,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Resize a spatial mask to the display image size.

        Nearest-neighbour interpolation is used so mask boundaries
        are not smoothed.
        """

        mask = self._prepare_mask(
            mask
        )

        mask_image = Image.fromarray(
            (
                mask.astype(
                    np.uint8
                )
                * 255
            )
        )

        mask_image = mask_image.resize(
            (
                self.width,
                self.height
            ),
            Image.Resampling.NEAREST
        )

        return (
            np.asarray(
                mask_image
            ) > 0
        )


    def draw_mask(
        self,
        mask: np.ndarray,
        opacity: int = 80
    ) -> None:
        """
        Overlay a spatial mask on the image.

        The mask itself must already have been produced by
        the model/geospatial layer.
        """

        if not 0 <= opacity <= 255:

            raise ValueError(
                "Opacity must be between 0 and 255."
            )

        mask = self._resize_mask(
            mask
        )

        overlay = Image.new(
            "RGBA",
            self.image.size,
            (220, 40, 40, 0)
        )

        alpha = np.zeros(
            (
                self.height,
                self.width
            ),
            dtype=np.uint8
        )

        alpha[mask] = np.uint8(
            opacity
        )

        overlay.putalpha(
            Image.fromarray(
                alpha
            )
        )

        self.image = Image.alpha_composite(
            self.image.convert("RGBA"),
            overlay
        ).convert("RGB")


    def draw_mask_boundary(
        self,
        mask: np.ndarray,
        width: int = 3
    ) -> None:
        """
        Draw only the boundary of a mask.

        Useful when you want to preserve the underlying satellite
        image while still clearly showing the detected region.
        """

        if width <= 0:

            raise ValueError(
                "Boundary width must be positive."
            )

        mask = self._resize_mask(
            mask
        )

        # Compare each pixel with neighbouring pixels.
        up = np.roll(
            mask,
            -1,
            axis=0
        )

        down = np.roll(
            mask,
            1,
            axis=0
        )

        left = np.roll(
            mask,
            -1,
            axis=1
        )

        right = np.roll(
            mask,
            1,
            axis=1
        )

        edge = (
            mask
            &
            (
                (~up)
                |
                (~down)
                |
                (~left)
                |
                (~right)
            )
        )

        rows, cols = np.where(
            edge
        )

        draw = ImageDraw.Draw(
            self.image
        )

        for row, col in zip(
            rows,
            cols
        ):

            draw.rectangle(
                [
                    int(col),
                    int(row),
                    int(col) + width - 1,
                    int(row) + width - 1
                ],
                fill="red"
            )

    def draw_point(
        self,
        point: Point,
        radius: int = 8,
        label: Optional[str] = None
    ) -> None:
        """
        Draw a point using image pixel coordinates.
        """

        if radius <= 0:

            raise ValueError(
                "Point radius must be positive."
            )

        draw = ImageDraw.Draw(
            self.image
        )

        x, y = map(
            int,
            point
        )

        draw.ellipse(
            [
                x - radius,
                y - radius,
                x + radius,
                y + radius
            ],
            fill="red",
            outline="white",
            width=2
        )

        if label:

            draw.text(
                (
                    x + radius + 4,
                    y - radius
                ),
                label,
                fill="red"
            )

    def geographic_to_pixel(
        self,
        x: float,
        y: float
    ) -> Point:
        """
        Convert coordinates in the GeoTIFF CRS into
        image pixel coordinates.

        Only meaningful for GeoTIFF inputs.
        """

        if self.transform is None:

            raise ValueError(
                "Geographic coordinate conversion requires "
                "a GeoTIFF input."
            )

        row, col = rasterio.transform.rowcol(
            self.transform,
            x,
            y
        )

        return (
            float(col),
            float(row)
        )

    def draw_geographic_point(
        self,
        x: float,
        y: float,
        radius: int = 8,
        label: Optional[str] = None
    ) -> None:
        """
        Draw a point provided in the GeoTIFF's coordinate system.
        """

        pixel = self.geographic_to_pixel(
            x,
            y
        )

        self.draw_point(
            pixel,
            radius=radius,
            label=label
        )


    def render_evidence(
        self,
        bounding_boxes: Optional[
            Sequence[
                Union[
                    Box,
                    Dict
                ]
            ]
        ] = None,
        mask: Optional[np.ndarray] = None,
        points: Optional[
            Sequence[
                Union[
                    Point,
                    Dict
                ]
            ]
        ] = None,
        mask_opacity: int = 80,
        show_mask_boundary: bool = True
    ) -> Image.Image:
        """
        Render every supplied spatial output.

        Nothing is inferred here.
        """

        if bounding_boxes:

            self.draw_bounding_boxes(
                bounding_boxes
            )

        if mask is not None:

            self.draw_mask(
                mask,
                opacity=mask_opacity
            )

            if show_mask_boundary:

                self.draw_mask_boundary(
                    mask
                )

        if points:

            for item in points:

                if isinstance(
                    item,
                    dict
                ):

                    if "point" not in item:

                        raise ValueError(
                            "Point dictionary "
                            "must contain 'point'."
                        )

                    point = item["point"]

                    label = item.get(
                        "label"
                    )

                else:

                    point = item
                    label = None

                self.draw_point(
                    point=point,
                    label=label
                )

        return self.image


    def spatial_metadata(
        self
    ) -> Dict:
        """
        Return metadata associated with the source image.
        """

        metadata = {
            "image": str(
                self.image_path
            ),
            "width": self.width,
            "height": self.height,
            "crs": (
                str(self.crs)
                if self.crs
                else None
            )
        }

        if self.bounds:

            metadata["bounds"] = {
                "left": float(
                    self.bounds.left
                ),
                "bottom": float(
                    self.bounds.bottom
                ),
                "right": float(
                    self.bounds.right
                ),
                "top": float(
                    self.bounds.top
                )
            }

        else:

            metadata["bounds"] = None

        return metadata


def visualize_bounding_boxes(
    image_path: str,
    boxes: Sequence[
        Union[
            Box,
            Dict
        ]
    ],
    output_path: str
) -> str:
    """
    Create an evidence image with model-generated bounding boxes.
    """

    visualizer = SpatialVisualizer(
        image_path
    )

    visualizer.draw_bounding_boxes(
        boxes
    )

    return visualizer.save(
        output_path
    )


def visualize_mask(
    image_path: str,
    mask: np.ndarray,
    output_path: str,
    opacity: int = 80
) -> str:
    """
    Create an evidence image from a segmentation mask.
    """

    visualizer = SpatialVisualizer(
        image_path
    )

    visualizer.draw_mask(
        mask,
        opacity=opacity
    )

    visualizer.draw_mask_boundary(
        mask
    )

    return visualizer.save(
        output_path
    )


def visualize_change(
    image_path: str,
    change_mask: np.ndarray,
    output_path: str,
    opacity: int = 85
) -> str:
    """
    Create a change-evidence image.

    change_mask MUST already have been generated by
    change_map.py.

    This function does not perform change detection.
    """

    visualizer = SpatialVisualizer(
        image_path
    )

    visualizer.draw_mask(
        change_mask,
        opacity=opacity
    )

    visualizer.draw_mask_boundary(
        change_mask
    )

    return visualizer.save(
        output_path
    )


def visualize_spatial_output(
    image_path: str,
    output_path: str,
    boxes: Optional[
        Sequence[
            Union[
                Box,
                Dict
            ]
        ]
    ] = None,
    mask: Optional[np.ndarray] = None,
    points: Optional[
        Sequence[
            Union[
                Point,
                Dict
            ]
        ]
    ] = None
) -> str:
    """
    General-purpose TriNetra visualization helper.

    Use this when the upstream model/router gives a combination
    of bounding boxes, masks and points.
    """

    visualizer = SpatialVisualizer(
        image_path
    )

    visualizer.render_evidence(
        bounding_boxes=boxes,
        mask=mask,
        points=points
    )

    return visualizer.save(
        output_path
    )



if __name__ == "__main__":

    print(
        "TriNetra visualization.py loaded successfully."
    )

    print(
        "\nSupported spatial outputs:"
    )

    print(
        "  • Bounding boxes"
    )

    print(
        "  • Segmentation masks"
    )

    print(
        "  • Change masks"
    )

    print(
        "  • Pixel coordinates"
    )

    print(
        "  • Geographic coordinates"
    )

    print(
        "\nSupported images:"
    )

    print(
        "  • GeoTIFF"
    )

    print(
        "  • JPEG"
    )

    print(
        "  • PNG"
    )