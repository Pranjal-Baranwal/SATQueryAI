"""
Pixel-Level Grounding via Segment Anything (SAM)
=================================================

The VLM (SIA) only gives a rough bounding box for "where is
the requested object". A box can never follow an irregular
boundary (tree canopies, water edges, road curves), so it will
always over- or under-cover the true region.

This module takes those boxes as *prompts* into SAM, which
snaps them to the actual pixel-level boundary of the object,
producing a real segmentation mask instead of a filled
rectangle.

Runs locally on CPU using the smallest SAM checkpoint
(ViT-B, ~375MB). The checkpoint is downloaded once on first
use and cached under models/sam/checkpoints/.
"""

from pathlib import Path

import numpy as np
import requests


MODEL_TYPE = "vit_b"

CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/"
    "sam_vit_b_01ec64.pth"
)

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

CHECKPOINT_PATH = CHECKPOINT_DIR / "sam_vit_b_01ec64.pth"

DEVICE = "cpu"


def download_checkpoint(
    url=CHECKPOINT_URL,
    destination=CHECKPOINT_PATH
):
    """
    Download the SAM checkpoint if it is not already cached.

    This is a one-time ~375MB download from Meta's public
    Segment Anything release.
    """

    destination = Path(destination)

    if destination.exists() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nDownloading SAM checkpoint ({MODEL_TYPE}) "
        f"from:\n{url}"
    )

    print(
        "This is a one-time ~375MB download and will be "
        f"cached at:\n{destination}\n"
    )

    response = requests.get(
        url,
        stream=True,
        timeout=120
    )

    response.raise_for_status()

    total_bytes = int(
        response.headers.get(
            "content-length",
            0
        )
    )

    downloaded = 0

    temp_path = destination.with_suffix(".part")

    with open(temp_path, "wb") as file:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if not chunk:
                continue

            file.write(chunk)

            downloaded += len(chunk)

            if total_bytes:

                percent = (
                    downloaded
                    /
                    total_bytes
                    * 100
                )

                print(
                    f"\rDownloading SAM checkpoint: "
                    f"{percent:5.1f}%",
                    end="",
                    flush=True
                )

    print()

    temp_path.replace(destination)

    print(
        "SAM checkpoint downloaded successfully."
    )

    return destination


class SAMSegmenter:
    """
    Wraps Meta's Segment Anything Model for box-prompted
    pixel-level segmentation.
    """

    def __init__(
        self,
        model_type=MODEL_TYPE,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE
    ):

        self.model_type = model_type
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

        self.sam = None
        self.predictor = None

        self.loaded = False

        self._current_image_id = None

    def load_model(self):

        if self.loaded:
            return

        download_checkpoint(
            destination=self.checkpoint_path
        )

        from segment_anything import (
            sam_model_registry,
            SamPredictor,
        )

        print(
            f"\nLoading SAM ({self.model_type}) on "
            f"{self.device}..."
        )

        self.sam = sam_model_registry[
            self.model_type
        ](
            checkpoint=str(
                self.checkpoint_path
            )
        )

        self.sam.to(
            device=self.device
        )

        self.predictor = SamPredictor(
            self.sam
        )

        self.loaded = True

        print(
            "SAM loaded successfully."
        )

    def _set_image_if_needed(
        self,
        image
    ):
        """
        Only re-encode the image if it changed. Encoding the
        image is the expensive part of SAM; reuse it across
        every box prompt for the same image.
        """

        self.load_model()

        image_array = np.asarray(
            image.convert("RGB")
        )

        image_id = (
            image_array.shape,
            image_array.tobytes()[:1024],
        )

        if image_id != self._current_image_id:

            self.predictor.set_image(
                image_array
            )

            self._current_image_id = image_id

    def segment_from_box(
        self,
        image,
        box
    ):
        """
        Segment the object inside a single [x1, y1, x2, y2]
        pixel-space box.

        Returns:
            Boolean mask, shape (height, width).
        """

        self._set_image_if_needed(
            image
        )

        box_array = np.array(
            box,
            dtype=np.float32
        )

        masks, scores, _ = self.predictor.predict(
            box=box_array,
            multimask_output=False
        )

        return masks[0].astype(bool)

    def segment_from_boxes(
        self,
        image,
        boxes
    ):
        """
        Segment every box and combine the results into one
        mask via logical OR.

        Returns:
            Boolean mask, shape (height, width).
        """

        if not boxes:

            width, height = image.size

            return np.zeros(
                (height, width),
                dtype=bool
            )

        combined = None

        for box in boxes:

            mask = self.segment_from_box(
                image,
                box
            )

            if combined is None:

                combined = mask

            else:

                combined = combined | mask

        return combined

    def segment_from_point(
        self,
        image,
        point
    ):
        """
        Segment the object at a single [x, y] pixel-space
        point.

        SAM returns 3 candidate masks per point prompt at
        different granularities (whole object, part, subpart).
        The one with the highest predicted IoU score is used,
        since a single foreground point is ambiguous about
        which granularity is intended.

        Returns:
            Boolean mask, shape (height, width).
        """

        self._set_image_if_needed(
            image
        )

        point_array = np.array(
            [point],
            dtype=np.float32
        )

        label_array = np.array(
            [1]
        )

        masks, scores, _ = self.predictor.predict(
            point_coords=point_array,
            point_labels=label_array,
            multimask_output=True
        )

        best_index = int(
            np.argmax(scores)
        )

        return masks[best_index].astype(bool)

    def segment_from_points(
        self,
        image,
        points
    ):
        """
        Segment every point and combine the results into one
        mask via logical OR.

        Returns:
            Boolean mask, shape (height, width).
        """

        if not points:

            width, height = image.size

            return np.zeros(
                (height, width),
                dtype=bool
            )

        combined = None

        for point in points:

            mask = self.segment_from_point(
                image,
                point
            )

            if combined is None:

                combined = mask

            else:

                combined = combined | mask

        return combined


sam_segmenter = SAMSegmenter()


def segment_objects(image, boxes):
    """
    Convenience wrapper: box-prompted SAM segmentation for a
    list of [x1, y1, x2, y2] pixel-space boxes on one image.
    """

    return sam_segmenter.segment_from_boxes(
        image=image,
        boxes=boxes
    )


def segment_objects_from_points(image, points):
    """
    Convenience wrapper: point-prompted SAM segmentation for a
    list of [x, y] pixel-space points on one image.
    """

    return sam_segmenter.segment_from_points(
        image=image,
        points=points
    )


if __name__ == "__main__":

    from PIL import Image

    print(
        "\n========== SAM SEGMENTATION TEST =========="
    )

    image_path = "data/input/T1.png"

    image = Image.open(
        image_path
    ).convert("RGB")

    width, height = image.size

    test_box = [
        width * 0.1,
        height * 0.1,
        width * 0.4,
        height * 0.4,
    ]

    mask = segment_objects(
        image,
        [test_box]
    )

    print(
        "Mask shape:",
        mask.shape
    )

    print(
        "Masked pixels:",
        int(mask.sum())
    )
