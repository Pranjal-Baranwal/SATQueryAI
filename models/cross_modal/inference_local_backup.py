import os
import time
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
            "LOADING FINE-TUNED CROSS-MODAL VLM"
        )

        print(
            "=================================================="
        )

        print(
            f"Base model : {self.model_id}"
        )

        print(
            f"Adapter    : {ADAPTER_PATH}"
        )

        print(
            f"Device     : {self.device}"
        )


        # ----------------------------------------------------
        # Verify adapter
        # ----------------------------------------------------

        if not ADAPTER_PATH.exists():

            raise FileNotFoundError(
                "\n"
                "Fine-tuned adapter was not found.\n\n"
                f"Expected location:\n"
                f"{ADAPTER_PATH}\n\n"
                "Make sure the final_adapter folder exists "
                "inside the SATQueryAI project."
            )


        required_files = [

            ADAPTER_PATH /
            "adapter_config.json",

            ADAPTER_PATH /
            "adapter_model.safetensors",

        ]


        for file_path in required_files:

            if not file_path.exists():

                raise FileNotFoundError(
                    "\n"
                    "Required adapter file is missing:\n"
                    f"{file_path}"
                )


        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        print(
            "\nLoading processor..."
        )


        self.processor = (
            AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
        )


        # ----------------------------------------------------
        # GPU
        # ----------------------------------------------------

        if self.device == "cuda":

            print(
                "\nUsing 4-bit NF4 quantization."
            )

            print(
                f"GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

            print(
                f"VRAM: "
                f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
            )


            # ------------------------------------------------
            # Load base RSCoVLM
            # ------------------------------------------------

            print(
                "\nLoading base RSCoVLM model..."
            )


            self.model = (
                Qwen2_5_VLForConditionalGeneration
                .from_pretrained(
                    self.model_id,
                    quantization_config=(
                        QUANTIZATION_CONFIG
                    ),
                    device_map="auto",
                    trust_remote_code=True,
                )
            )


        # ----------------------------------------------------
        # CPU
        # ----------------------------------------------------

        else:

            print(
                "\nWARNING: CUDA unavailable."
            )

            print(
                "Loading model on CPU."
            )


            self.model = (
                Qwen2_5_VLForConditionalGeneration
                .from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                )
            )


        # ----------------------------------------------------
        # Load YOUR LoRA adapter
        # ----------------------------------------------------

        print(
            "\nLoading SATQueryAI fine-tuned LoRA adapter..."
        )


        self.model = (
            PeftModel.from_pretrained(
                self.model,
                str(ADAPTER_PATH),
                is_trainable=False,
            )
        )


        # ----------------------------------------------------
        # Evaluation mode
        # ----------------------------------------------------

        self.model.eval()


        self.loaded = True


        print(
            "\n"
            "=================================================="
        )

        print(
            "FINE-TUNED MODEL LOADED SUCCESSFULLY"
        )

        print(
            "=================================================="
        )

        print(
            f"Base model : {self.model_id}"
        )

        print(
            f"LoRA       : {ADAPTER_PATH}"
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
        # Processor
        # ----------------------------------------------------

        processing_start = time.time()


        text = (
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )


        # IMPORTANT:
        #
        # Image order MUST match message order:
        #
        #   message image 1 = SAR
        #   message image 2 = Optical
        #

        inputs = (
            self.processor(
                text=[text],
                images=[
                    sar_image,
                    optical_image
                ],
                padding=True,
                return_tensors="pt"
            )
        )


        # ----------------------------------------------------
        # Move tensors to GPU
        # ----------------------------------------------------

        if self.device == "cuda":

            inputs = {

                key: (
                    value.to("cuda")
                    if hasattr(value, "to")
                    else value
                )

                for key, value
                in inputs.items()

            }


        processing_time = (
            time.time()
            -
            processing_start
        )


        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        print(
            "\nGenerating response using "
            "fine-tuned VLM..."
        )


        generation_start = time.time()


        if self.device == "cuda":

            torch.cuda.synchronize()


        with torch.inference_mode():

            generated_ids = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            )


        if self.device == "cuda":

            torch.cuda.synchronize()


        generation_time = (
            time.time()
            -
            generation_start
        )


        # ----------------------------------------------------
        # Remove input tokens
        # ----------------------------------------------------

        input_token_length = (
            inputs[
                "input_ids"
            ].shape[1]
        )


        generated_ids = (
            generated_ids[
                :,
                input_token_length:
            ]
        )


        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        answer = (
            self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
        )


        answer = answer.strip()


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

            "device": self.device,

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

            "device": self.device,

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