import os
import base64
import io
import requests
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv

from models.sia.prompts import (
    SYSTEM_PROMPT,
    build_general_prompt,
    build_scene_description_prompt,
    build_object_prompt,
    build_land_cover_prompt,
    build_vegetation_prompt,
    build_water_prompt,
    build_urban_prompt,
    build_change_aware_prompt,
    build_grounding_prompt,
)

from models.sia.grounding import (
    parse_grounding_response,
    points_to_mask,
)

from geospatial.visualization import (
    SpatialVisualizer,
)


load_dotenv()


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_ID = "Qingyun/RSCoVLM-7B-2512"

MODAL_SIA_ENDPOINT = (
    "https://trinetrasih--satqueryai-base-models-basevlm-sia.modal.run"
)

DEVICE = "Modal A10G"


# ============================================================
# SIA INFERENCE ENGINE
# ============================================================

class SIAInference:
    """
    Single Image Analysis inference engine.

    Image preprocessing and prompt construction happen locally.

    The actual RSCoVLM-7B inference runs remotely on:
        Modal A10G

    IMPORTANT:
        This uses ONLY the base model.
        No fine-tuned LoRA adapter is used.
    """

    def __init__(
        self,
        model_id=MODEL_ID,
        device=DEVICE
    ):

        self.model_id = model_id
        self.device = device

        # No local model is loaded.
        self.model = None
        self.processor = None

        self.loaded = False


    # ========================================================
    # MODEL
    # ========================================================

    def load_model(self):
        """
        The model runs remotely on Modal A10G.

        Nothing is loaded onto the local machine.
        No LoRA adapter is used.
        """

        if self.loaded:
            return

        print("\nSIA model: Modal A10G")
        print("Model: Qingyun/RSCoVLM-7B-2512")
        print("Adapter: None")

        self.loaded = True


    # ========================================================
    # IMAGE VALIDATION
    # ========================================================

    @staticmethod
    def validate_image(image_path):

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        supported_formats = [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff"
        ]

        if image_path.suffix.lower() not in supported_formats:

            raise ValueError(
                "Unsupported image format. "
                f"Supported formats: {supported_formats}"
            )

        return image_path


    # ========================================================
    # IMAGE LOADING
    # ========================================================

    @staticmethod
    def load_image(image_path):

        image_path = SIAInference.validate_image(
            image_path
        )

        image = Image.open(
            image_path
        )

        image = image.convert(
            "RGB"
        )

        return image


    # ========================================================
    # PROMPT BUILDING
    # ========================================================

    @staticmethod
    def build_prompt(
        query,
        analysis_type="general"
    ):

        analysis_type = (
            analysis_type
            .lower()
            .strip()
        )

        if analysis_type == "general":

            return build_general_prompt(
                query
            )

        elif analysis_type == "scene":

            return build_scene_description_prompt()

        elif analysis_type == "object":

            return build_object_prompt(
                query
            )

        elif analysis_type == "land_cover":

            return build_land_cover_prompt(
                query
            )

        elif analysis_type == "vegetation":

            return build_vegetation_prompt(
                query
            )

        elif analysis_type == "water":

            return build_water_prompt(
                query
            )

        elif analysis_type == "urban":

            return build_urban_prompt(
                query
            )

        elif analysis_type == "change_aware":

            return build_change_aware_prompt(
                query
            )

        else:

            raise ValueError(
                f"Unsupported analysis type: "
                f"{analysis_type}"
            )


    # ========================================================
    # MESSAGE CREATION
    # ========================================================

    @staticmethod
    def create_messages(prompt):

        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image"
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]


    # ========================================================
    # SIA ANALYSIS
    # ========================================================

    def analyze(
        self,
        image_path,
        query,
        analysis_type="general",
        max_new_tokens=256,
        temperature=0.2
    ):
        """
        Perform Single Image Analysis.

        Local:
            - image validation
            - image loading
            - RGB conversion
            - prompt construction

        Modal:
            - RSCoVLM-7B base model
            - GPU inference on A10G
        """

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )


        # ----------------------------------------------------
        # Make sure remote model is considered ready
        # ----------------------------------------------------

        self.load_model()


        # ----------------------------------------------------
        # Load image locally
        # ----------------------------------------------------

        image = self.load_image(
            image_path
        )


        # ----------------------------------------------------
        # Build SIA prompt locally
        # ----------------------------------------------------

        prompt = self.build_prompt(
            query=query,
            analysis_type=analysis_type
        )


        # ----------------------------------------------------
        # Encode image as PNG -> Base64
        # ----------------------------------------------------

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")


        # ----------------------------------------------------
        # Prepare Modal request
        # ----------------------------------------------------

        payload = {
            "image": image_base64,
            "query": query,
            "prompt": prompt,
            "system_prompt": SYSTEM_PROMPT,
            "max_new_tokens": max_new_tokens,
        }


        print(
            "\nSending SIA request to Modal A10G..."
        )


        # ----------------------------------------------------
        # Call Modal endpoint
        # ----------------------------------------------------

        try:

            response = requests.post(
                MODAL_SIA_ENDPOINT,
                json=payload,
                timeout=1800
            )

            response.raise_for_status()

        except requests.RequestException as e:

            raise RuntimeError(
                f"Modal SIA request failed: {e}"
            ) from e


        # ----------------------------------------------------
        # Read Modal response
        # ----------------------------------------------------

        try:

            result = response.json()

        except ValueError as e:

            raise RuntimeError(
                "Modal SIA returned an invalid JSON response."
            ) from e


        if not result.get("success", True):

            raise RuntimeError(
                result.get(
                    "error",
                    "Unknown error from Modal SIA."
                )
            )


        answer = result.get(
            "answer",
            ""
        )

        answer = answer.strip()


        # ----------------------------------------------------
        # Return standard SIA result
        # ----------------------------------------------------

        return {
            "model": self.model_id,
            "analysis_type": analysis_type,
            "query": query,
            "answer": answer,
            "device": "Modal A10G",
            "gpu_memory_gb": None
        }


    # ========================================================
    # SIMPLE ASK INTERFACE
    # ========================================================

    def ask(
        self,
        image_path,
        query
    ):

        result = self.analyze(
            image_path=image_path,
            query=query,
            analysis_type="general"
        )

        return result["answer"]


    # ========================================================
    # VISUAL GROUNDING
    # ========================================================

    def ground(
        self,
        image_path,
        query,
        max_new_tokens=1024,
        output_path=None
    ):
        """
        Localize the region(s) requested in `query` and return
        an annotated image highlighting them.

        Remote:
            RSCoVLM-7B on Modal is asked (via the same /sia
            endpoint used by analyze()) to return normalized
            bounding boxes as JSON.

        Local:
            The JSON is parsed, rasterized into a mask, and
            rendered with geospatial.visualization.SpatialVisualizer
            so the same highlight styling used for change masks
            elsewhere in the app is reused here.
        """

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        image = self.load_image(
            image_path
        )

        image_width, image_height = image.size

        heuristic_result = self._ground_with_heuristic(
            image=image,
            image_path=image_path,
            query=query,
            output_path=output_path
        )

        if heuristic_result is not None:

            return heuristic_result

        self.load_model()

        prompt = build_grounding_prompt(
            query
        )

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        payload = {
            "image": image_base64,
            "query": query,
            "prompt": prompt,
            "system_prompt": SYSTEM_PROMPT,
            "max_new_tokens": max_new_tokens,
        }

        print(
            "\nSending grounding request to Modal A10G..."
        )

        try:

            response = requests.post(
                MODAL_SIA_ENDPOINT,
                json=payload,
                timeout=1800
            )

            response.raise_for_status()

        except requests.RequestException as e:

            raise RuntimeError(
                f"Modal SIA request failed: {e}"
            ) from e

        try:

            result = response.json()

        except ValueError as e:

            raise RuntimeError(
                "Modal SIA returned an invalid JSON response."
            ) from e

        if not result.get("success", True):

            raise RuntimeError(
                result.get(
                    "error",
                    "Unknown error from Modal SIA."
                )
            )

        raw_response = result.get(
            "answer",
            ""
        ).strip()

        objects = parse_grounding_response(
            raw_response,
            image_width=image_width,
            image_height=image_height
        )

        if objects:

            mask = self._refine_mask_with_sam(
                image=image,
                objects=objects,
                image_width=image_width,
                image_height=image_height
            )

            visualizer = SpatialVisualizer(
                image_path
            )

            visualizer.draw_mask(mask)
            visualizer.draw_mask_boundary(mask)

            if output_path is None:

                stem = Path(image_path).stem

                output_path = (
                    Path("data/output")
                    /
                    f"grounding_{stem}.png"
                )

            output_path = visualizer.save(
                output_path
            )

            descriptions = []

            for item in objects:

                description = (
                    f"{item['label']} "
                    f"({item['confidence']} confidence)"
                )

                if item.get("location_hint"):

                    description += (
                        f" — {item['location_hint']}"
                    )

                descriptions.append(
                    description
                )

            answer = (
                f"Found {len(objects)} region(s) matching "
                f"'{query.strip()}': "
                + "; ".join(descriptions)
            )

        else:

            output_path = None

            answer = (
                f"No region matching '{query.strip()}' "
                "could be confidently localized in the image."
            )

        return {
            "model": self.model_id,
            "query": query,
            "answer": answer,
            "raw_response": raw_response,
            "objects": objects,
            "image_width": image_width,
            "image_height": image_height,
            "device": "Modal A10G",
            "image_path": (
                str(output_path)
                if output_path
                else None
            ),
        }


    @staticmethod
    def _ground_with_heuristic(
        image,
        image_path,
        query,
        output_path=None
    ):
        """
        Try deterministic pixel-level detection for a handful
        of common, visually well-defined classes (water,
        vegetation, urban, road) before ever calling the VLM.

        Asking a general VQA-tuned VLM to localize these is
        unreliable: there is no guarantee it was ever trained
        on spatial grounding at all, and a wrong answer here
        looks identical to a right one until you inspect the
        image. For these classes the correct region is directly
        computable from color and texture, with zero chance of
        the model confidently pointing at the wrong place, or
        missing an obvious instance entirely.

        Returns:
            A full grounding result dict if a known class was
            matched and produced a non-empty mask, else None
            (the caller should fall back to VLM + SAM
            grounding).
        """

        from geospatial.semantic_masks import (
            match_known_class,
            detect_class_mask,
        )

        class_name = match_known_class(
            query
        )

        if class_name is None:
            return None

        print(
            f"\nQuery matched known class '{class_name}'; "
            "using deterministic pixel detection instead of "
            "VLM grounding."
        )

        mask = detect_class_mask(
            image,
            class_name
        )

        if mask is None or not mask.any():

            print(
                f"No '{class_name}' regions were detected "
                "by the heuristic; falling back to VLM "
                "grounding."
            )

            return None

        visualizer = SpatialVisualizer(
            image_path
        )

        visualizer.draw_mask(mask)
        visualizer.draw_mask_boundary(mask)

        if output_path is None:

            stem = Path(image_path).stem

            output_path = (
                Path("data/output")
                /
                f"grounding_{stem}.png"
            )

        output_path = visualizer.save(
            output_path
        )

        coverage_percent = round(
            100.0 * float(mask.mean()),
            2
        )

        answer = (
            f"Highlighted all detected '{class_name}' "
            f"regions ({coverage_percent}% of the image) "
            "using direct pixel color/texture analysis."
        )

        image_width, image_height = image.size

        return {
            "model": "heuristic:" + class_name,
            "query": query,
            "answer": answer,
            "raw_response": None,
            "objects": [
                {
                    "label": class_name,
                    "confidence": "high",
                    "location_hint": (
                        "detected directly from pixel "
                        "color/texture, not model-localized"
                    ),
                }
            ],
            "image_width": image_width,
            "image_height": image_height,
            "device": "local",
            "image_path": str(
                output_path
            ),
        }


    @staticmethod
    def _refine_mask_with_sam(
        image,
        objects,
        image_width,
        image_height
    ):
        """
        Turn the VLM's rough points into a true pixel-level
        mask using Segment Anything.

        Each point is used as a prompt: SAM grows it into the
        full boundary of whatever object sits at that location.
        Falls back to small filled circles around each point if
        SAM is unavailable (not installed, checkpoint download
        failed, etc.) so grounding degrades instead of breaking
        outright.
        """

        points = [
            item["point"]
            for item in objects
        ]

        try:

            from models.sam.segment import (
                segment_objects_from_points,
            )

            print(
                "\nRefining points into pixel masks "
                "with SAM..."
            )

            return segment_objects_from_points(
                image,
                points
            )

        except Exception as exc:

            print(
                "\nSAM refinement unavailable, falling "
                f"back to point-radius mask: {exc}"
            )

            return points_to_mask(
                objects,
                image_width=image_width,
                image_height=image_height
            )


    # ========================================================
    # STATUS
    # ========================================================

    def status(self):

        return {
            "model_id": self.model_id,
            "device": self.device,
            "cuda_available": False,
            "loaded": self.loaded,
            "adapter": None,
            "remote": True
        }


    # ========================================================
    # UNLOAD
    # ========================================================

    def unload_model(self):

        # Nothing is loaded locally.

        self.model = None
        self.processor = None
        self.loaded = False

        print(
            "SIA remote model connection reset."
        )


# ============================================================
# GLOBAL SIA ENGINE
# ============================================================

sia_engine = SIAInference()


# ============================================================
# PUBLIC API
# ============================================================

def analyze_image(
    image_path,
    query,
    analysis_type="general"
):

    return sia_engine.analyze(
        image_path=image_path,
        query=query,
        analysis_type=analysis_type
    )


def run_sia(
    image_path,
    query,
    analysis_type="general"
):
    """
    Execute SIA through the routing pipeline.

    Used by routing.graph / app.py.
    """

    return sia_engine.analyze(
        image_path=image_path,
        query=query,
        analysis_type=analysis_type
    )


def ground_image(
    image_path,
    query,
    output_path=None
):
    """
    Convenience wrapper around SIAInference.ground().
    """

    return sia_engine.ground(
        image_path=image_path,
        query=query,
        output_path=output_path
    )


def run_grounding(
    image_path,
    query
):
    """
    Execute visual grounding through the routing pipeline.

    Used by routing.graph / app.py.
    """

    return sia_engine.ground(
        image_path=image_path,
        query=query
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    image_path = "data/input/T1.png"

    query = (
        "Describe the major features visible "
        "in this satellite image."
    )


    print(
        "\n========== SIA TEST =========="
    )


    print(
        f"Model : {MODEL_ID}"
    )


    print(
        "Adapter: None"
    )


    print(
        f"Device: {DEVICE}"
    )


    try:

        result = analyze_image(
            image_path=image_path,
            query=query,
            analysis_type="general"
        )


        print(
            "\nModel Answer:"
        )

        print(
            result["answer"]
        )


        print(
            "\nModel:"
        )

        print(
            result["model"]
        )


        print(
            "\nDevice:"
        )

        print(
            result["device"]
        )


        print(
            "\nAdapter:"
        )

        print(
            "None"
        )


        print(
            "\n==============================\n"
        )


    except Exception as e:

        print(
            f"\nSIA inference error: {e}"
        )