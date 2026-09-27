import os
import time
import base64
import io
import requests
from pathlib import Path

import numpy as np
import torch

from PIL import Image
from dotenv import load_dotenv

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from peft import PeftModel

from models.cross_modal.fusion import (
    CrossModalFusion,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_ID = os.getenv(
    "SIA_MODEL_ID",
    "Qingyun/RSCoVLM-7B-2512"
)


# Modal endpoint hosting the fine-tuned Optical + SAR VLM on A10G.
MODAL_ENDPOINT = os.getenv(
    "SATQUERY_MODAL_ENDPOINT",
    "https://trinetrasih--satqueryai-cross-modal-crossmodalvlm-predict.modal.run"
)


# Project root:
#
# SATQueryAI/
# ├── final_adapter/
# ├── models/
# │   └── cross_modal/
# │       └── inference.py
#
# parents[0] = cross_modal
# parents[1] = models
# parents[2] = SATQueryAI
#
PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


# Your downloaded fine-tuned LoRA adapter
ADAPTER_PATH = Path(
    os.getenv(
        "SATQUERY_FINETUNED_ADAPTER",
        str(
            PROJECT_ROOT /
            "final_adapter"
        )
    )
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# GENERATION CONFIGURATION
# ============================================================

MAX_NEW_TOKENS = 96

MAX_IMAGE_SIZE = 1024


# ============================================================
# 4-BIT QUANTIZATION
#
# Same configuration used during your fine-tuning.
# ============================================================

QUANTIZATION_CONFIG = None

if DEVICE == "cuda":

    QUANTIZATION_CONFIG = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    )


# ============================================================
# CROSS-MODAL INFERENCE
# ============================================================

class CrossModalInference:

    def __init__(
        self,
        model_id=MODEL_ID,
        device=DEVICE
    ):

        self.model_id = model_id

        self.device = device

        self.model = None

        self.processor = None

        self.fusion_engine = (
            CrossModalFusion()
        )

        self.loaded = False


    # ========================================================
    # LOAD BASE MODEL + FINE-TUNED ADAPTER
    # ========================================================

    def load_model(self):

        if self.loaded:
            return

        print(
            "\n"
            "=================================================="
        )

        print(
            "PREPARING REMOTE FINE-TUNED CROSS-MODAL VLM"
        )

        print(
            "=================================================="
        )

        print(
            f"Base model : {self.model_id}"
        )

        print(
            "Execution  : Modal A10G"
        )

        print(
            f"Endpoint   : {MODAL_ENDPOINT}"
        )

        # The VLM itself is NOT loaded on the local machine.
        # Modal loads the base model + fine-tuned LoRA adapter on A10G.
        self.model = None
        self.processor = None
        self.loaded = True

        print(
            "\nRemote fine-tuned model ready."
        )


    # ========================================================
    # ARRAY → RGB IMAGE
    # ========================================================

    @staticmethod
    def array_to_image(
        data
    ):

        """
        Convert normalized CHW array into RGB PIL image.
        """

        data = (
            np.asarray(data)
            .astype(np.float32)
        )


        if data.shape[0] == 1:

            band = data[0]

            rgb = np.stack(
                [
                    band,
                    band,
                    band
                ],
                axis=-1
            )


        elif data.shape[0] == 2:

            rgb = np.stack(
                [
                    data[0],
                    data[1],
                    data[1]
                ],
                axis=-1
            )


        else:

            rgb = np.transpose(
                data[:3],
                (1, 2, 0)
            )


        rgb = np.clip(
            rgb * 255.0,
            0,
            255
        ).astype(
            np.uint8
        )


        image = Image.fromarray(
            rgb,
            mode="RGB"
        )


        image.thumbnail(
            (
                MAX_IMAGE_SIZE,
                MAX_IMAGE_SIZE
            ),
            Image.Resampling.LANCZOS
        )


        return image


    # ========================================================
    # FUSED VISUALIZATION
    #
    # Still retained for SATQueryAI's existing UI/output.
    # The fused image is NOT sent to the fine-tuned VLM.
    # ========================================================

    @staticmethod
    def create_fused_visualization(
        optical,
        sar
    ):

        """
        Create a visual Optical + SAR representation.
        """

        optical = (
            np.asarray(optical)
            .astype(np.float32)
        )

        sar = (
            np.asarray(sar)
            .astype(np.float32)
        )


        if optical.shape[0] >= 3:

            optical_rgb = np.transpose(
                optical[:3],
                (1, 2, 0)
            )


        else:

            optical_rgb = np.repeat(
                optical[0:1],
                3,
                axis=0
            )

            optical_rgb = np.transpose(
                optical_rgb,
                (1, 2, 0)
            )


        sar_band = sar[0]


        sar_rgb = np.stack(
            [
                sar_band,
                sar_band,
                sar_band
            ],
            axis=-1
        )


        height = min(
            optical_rgb.shape[0],
            sar_rgb.shape[0]
        )


        width = min(
            optical_rgb.shape[1],
            sar_rgb.shape[1]
        )


        optical_rgb = (
            optical_rgb[
                :height,
                :width
            ]
        )


        sar_rgb = (
            sar_rgb[
                :height,
                :width
            ]
        )


        fused = (
            0.5 * optical_rgb
            +
            0.5 * sar_rgb
        )


        fused = np.clip(
            fused * 255.0,
            0,
            255
        ).astype(
            np.uint8
        )


        image = Image.fromarray(
            fused,
            mode="RGB"
        )


        image.thumbnail(
            (
                MAX_IMAGE_SIZE,
                MAX_IMAGE_SIZE
            ),
            Image.Resampling.LANCZOS
        )


        return image


    # ========================================================
    # PROMPT
    #
    # IMPORTANT:
    # This follows the task used during fine-tuning:
    #
    # SAR + Optical → land-cover analysis
    #
    # ========================================================

    @staticmethod
    def create_prompt(
        query
    ):

        return f"""
Analyze the Sentinel-1 SAR and Sentinel-2
optical imagery together.

Identify the land-cover classes present
in this scene.

Use both modalities together when interpreting
the scene.

USER QUESTION:
{query}
"""


    # ========================================================
    # PREPARE INPUTS
    # ========================================================

    def prepare_inputs(
        self,
        optical_path,
        sar_path
    ):

        print(
            "\n========== CROSS-MODAL FUSION =========="
        )


        fusion_result = (
            self.fusion_engine.process(
                optical_path=optical_path,
                sar_path=sar_path
            )
        )


        optical = (
            fusion_result["optical"]
        )


        sar = (
            fusion_result["sar"]
        )


        # ----------------------------------------------------
        # Optical image
        # ----------------------------------------------------

        optical_image = (
            self.array_to_image(
                optical
            )
        )


        # ----------------------------------------------------
        # SAR image
        # ----------------------------------------------------

        sar_image = (
            self.array_to_image(
                sar
            )
        )


        # ----------------------------------------------------
        # Fused visualization
        #
        # This remains available for the project.
        # ----------------------------------------------------

        fused_image = (
            self.create_fused_visualization(
                optical,
                sar
            )
        )


        fused_path = (
            Path("data/output")
            /
            "cross_modal_vlm_fusion.png"
        )


        fused_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        fused_image.save(
            fused_path
        )


        print(
            f"Fused VLM image saved to: "
            f"{fused_path}"
        )


        return (
            optical_image,
            sar_image,
            fused_image,
            fusion_result
        )


    # ========================================================
    # ANALYZE
    # ========================================================

    @staticmethod
    def image_to_base64(image):

        """Convert a PIL image to a base64-encoded PNG string."""

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")


    def analyze(
        self,
        optical_path,
        sar_path,
        query,
        max_new_tokens=MAX_NEW_TOKENS
    ):

        if (
            not query
            or
            not query.strip()
        ):

            raise ValueError(
                "Query cannot be empty."
            )


        total_start = time.time()


        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model_start = time.time()


        self.load_model()


        model_time = (
            time.time()
            -
            model_start
        )


        # ----------------------------------------------------
        # Prepare images
        # ----------------------------------------------------

        fusion_start = time.time()


        (
            optical_image,
            sar_image,
            fused_image,
            fusion_result
        ) = self.prepare_inputs(
            optical_path,
            sar_path
        )


        fusion_time = (
            time.time()
            -
            fusion_start
        )


        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = (
            self.create_prompt(
                query
            )
        )


        # ====================================================
        # IMPORTANT MULTIMODAL FORMAT
        #
        # YOUR FINE-TUNING USED:
        #
        #   IMAGE 1 → SAR
        #   IMAGE 2 → OPTICAL
        #
        # Therefore we send exactly TWO images.
        #
        # ====================================================

        messages = [

            {
                "role": "user",

                "content": [

                    {
                        "type": "image"
                    },

                    {
                        "type": "image"
                    },

                    {
                        "type": "text",
                        "text": prompt
                    },

                ]

            }

        ]


        # ----------------------------------------------------
        # Send Optical + SAR to Modal
        # ----------------------------------------------------

        processing_start = time.time()

        payload = {
            "sar_image": self.image_to_base64(sar_image),
            "optical_image": self.image_to_base64(optical_image),
            "query": query,
        }

        processing_time = (
            time.time()
            -
            processing_start
        )

        print(
            "\nSending Optical + SAR to "
            "fine-tuned VLM on Modal A10G..."
        )

        generation_start = time.time()

        response = requests.post(
            MODAL_ENDPOINT,
            json=payload,
            timeout=1800,
        )

        response.raise_for_status()

        remote_result = response.json()

        if not remote_result.get("success", False):
            raise RuntimeError(
                remote_result.get(
                    "error",
                    "Modal Cross-Modal inference failed."
                )
            )

        answer = str(
            remote_result.get("answer", "")
        ).strip()

        generation_time = (
            time.time()
            -
            generation_start
        )


        # ----------------------------------------------------
        # GPU memory
        # ----------------------------------------------------

        gpu_memory = None


        if self.device == "cuda":

            gpu_memory = round(
                torch.cuda.memory_allocated()
                /
                (1024 ** 3),
                2
            )


        # ----------------------------------------------------
        # Total time
        # ----------------------------------------------------

        total_time = (
            time.time()
            -
            total_start
        )


        # ----------------------------------------------------
        # Timing output
        # ----------------------------------------------------

        print(
            "\n========== CROSS-MODAL TIMING =========="
        )


        print(
            f"Model loading : "
            f"{model_time:.2f} sec"
        )


        print(
            f"Fusion        : "
            f"{fusion_time:.2f} sec"
        )


        print(
            f"Processing    : "
            f"{processing_time:.2f} sec"
        )


        print(
            f"Generation    : "
            f"{generation_time:.2f} sec"
        )


        print(
            f"Total         : "
            f"{total_time:.2f} sec"
        )


        print(
            "========================================="
        )


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {

            "model": (
                f"{self.model_id} + "
                "SATQueryAI fine-tuned LoRA adapter"
            ),

            "query": query,

            "answer": answer,

            "device": "Modal A10G",

            "gpu_memory_gb": gpu_memory,

            "fusion_statistics": (
                fusion_result["statistics"]
            ),

            "generation_time_seconds": round(
                generation_time,
                2
            ),

            "total_time_seconds": round(
                total_time,
                2
            )

        }


    # ========================================================
    # COMPARE
    # ========================================================

    def compare(
        self,
        optical_path,
        sar_path,
        query=(
            "What information can be identified "
            "by combining the optical and SAR imagery?"
        )
    ):

        result = (
            self.analyze(
                optical_path=optical_path,
                sar_path=sar_path,
                query=query
            )
        )


        return result["answer"]


    # ========================================================
    # STATUS
    # ========================================================

    def status(self):

        return {

            "model_id": self.model_id,

            "adapter_path": str(
                ADAPTER_PATH
            ),

            "device": "Modal A10G",

            "cuda_available": (
                torch.cuda.is_available()
            ),

            "loaded": self.loaded

        }


    # ========================================================
    # UNLOAD MODEL
    # ========================================================

    def unload_model(self):

        if self.model is not None:

            del self.model


        if self.processor is not None:

            del self.processor


        self.model = None

        self.processor = None

        self.loaded = False


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


        print(
            "Fine-tuned Cross-Modal model unloaded."
        )


# ============================================================
# GLOBAL ENGINE
# ============================================================

cross_modal_engine = (
    CrossModalInference()
)


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def analyze_cross_modal(
    optical_path,
    sar_path,
    query
):

    return (
        cross_modal_engine.analyze(
            optical_path=optical_path,
            sar_path=sar_path,
            query=query
        )
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    optical_path = (
        "data/input/T1.png"
    )


    sar_path = (
        "data/input/Sample.tif"
    )


    query = (
        "What information can be identified "
        "by combining the optical and SAR imagery?"
    )


    print(
        "\n"
        "========== FINE-TUNED CROSS-MODAL TEST =========="
    )


    print(
        f"Base model : {MODEL_ID}"
    )


    print(
        f"Adapter    : {ADAPTER_PATH}"
    )


    print(
        f"Device     : {DEVICE}"
    )


    try:

        result = (
            analyze_cross_modal(
                optical_path=optical_path,
                sar_path=sar_path,
                query=query
            )
        )


        print(
            "\nCross-Modal Answer:"
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


        if (
            result["gpu_memory_gb"]
            is not None
        ):

            print(
                f"\nGPU memory allocated: "
                f"{result['gpu_memory_gb']} GB"
            )


        print(
            "\n=============================="
        )


    except Exception as e:

        print(
            f"\nCross-Modal inference error: {e}"
        )